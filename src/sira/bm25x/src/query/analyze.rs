// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Query analysis: AST → Query rewrite layer.
//!
//! Each AST leaf node undergoes the correct text processing:
//!
//! | Node type | Analysis pipeline                                       |
//! |-----------|---------------------------------------------------------|
//! | Term      | Full: lowercase → normalize → stem → resolve vocab ID   |
//! | Phrase    | Full per token, then positional match                    |
//! | Fuzzy     | Full analysis → Levenshtein scan on stemmed vocab        |
//! | Wildcard  | Union of raw-lowered prefix + stemmed prefix scan        |
//! | Regex     | Raw pattern matched against stemmed term dictionary      |
//! | IN set    | Full analysis per element                                |
//! | Exists    | All docs (single-field index)                            |

use std::collections::HashSet;

use regex::Regex;
use tantivy_query_grammar::{Delimiter, Occur, UserInputAst, UserInputLeaf};

use super::Query;
use crate::index::BM25;

/// Transform a parsed AST into a resolved [`Query`].
///
/// Plain bag-of-words queries like `"machine learning"` (unquoted) get parsed
/// by `tantivy_query_grammar` as a `Clause` of two `Should` literals. The
/// per-leaf path would resolve them as two independent `Term` queries and
/// drop any cross-token n-gram contribution (the bigram "machine learning"
/// would never enter the score). We detect this case at the top level and
/// re-route through [`analyze_terms`] on the joined text — keeping
/// `search_expr` consistent with [`BM25::search`] for plain queries while
/// preserving the per-leaf semantic for queries that mix operators
/// (`AND`/`NOT`/wildcards/phrases/etc.).
pub(super) fn analyze(ast: &UserInputAst, bm25: &BM25) -> Query {
    if let Some(joined) = collect_plain_should_text(ast) {
        return analyze_terms(&joined, bm25);
    }
    match ast {
        UserInputAst::Leaf(leaf) => analyze_leaf(leaf, bm25),
        UserInputAst::Boost(inner, boost) => {
            let q = analyze(inner, bm25);
            Query::Boost {
                inner: Box::new(q),
                factor: boost.into_inner() as f32,
            }
        }
        UserInputAst::Clause(clauses) => {
            let analyzed: Vec<(Occur, Query)> = clauses
                .iter()
                .map(|(occur, ast)| {
                    let occur = occur.unwrap_or(Occur::Should);
                    (occur, analyze(ast, bm25))
                })
                .collect();
            Query::Boolean(analyzed)
        }
    }
}

/// Recognize a plain unquoted bag-of-words query and reconstruct its text by
/// joining each leaf's literal with a single space. Returns `None` if the
/// query has any operator, modifier, phrase, wildcard, fuzzy, regex, etc. —
/// anything that would make per-leaf analysis the right thing to do.
fn collect_plain_should_text(ast: &UserInputAst) -> Option<String> {
    let clauses = match ast {
        UserInputAst::Clause(c) => c,
        _ => return None,
    };
    if clauses.len() < 2 {
        // Single-leaf queries already go through analyze_terms by their leaf path.
        return None;
    }
    let mut parts: Vec<&str> = Vec::with_capacity(clauses.len());
    for (occur, child) in clauses {
        // Only accept implicit (None) or explicit Should — Must/MustNot/+/-
        // change semantics and must keep their per-leaf path.
        if !matches!(occur, None | Some(Occur::Should)) {
            return None;
        }
        let leaf_box = match child {
            UserInputAst::Leaf(l) => l,
            _ => return None,
        };
        let lit = match leaf_box.as_ref() {
            UserInputLeaf::Literal(l) => l,
            _ => return None,
        };
        // Reject anything that needs special analysis: phrases (delimiter set),
        // wildcards (prefix or trailing `*`), fuzzy syntax (`~N`), or in-leaf
        // slop. Per-leaf analyze_leaf is the correct path for those.
        if lit.delimiter != Delimiter::None || lit.prefix || lit.slop != 0 {
            return None;
        }
        if lit.phrase.ends_with('*') || parse_fuzzy_syntax(&lit.phrase).is_some() {
            return None;
        }
        parts.push(&lit.phrase);
    }
    Some(parts.join(" "))
}

/// Dispatch a leaf node to the appropriate analysis pipeline.
fn analyze_leaf(leaf: &UserInputLeaf, bm25: &BM25) -> Query {
    match leaf {
        UserInputLeaf::Literal(lit) => {
            if lit.prefix || (lit.delimiter == Delimiter::None && lit.phrase.ends_with('*')) {
                analyze_wildcard(&lit.phrase, bm25)
            } else if lit.delimiter != Delimiter::None {
                // Quoted: phrase query (slop = phrase proximity)
                analyze_phrase(&lit.phrase, lit.slop, bm25)
            } else if let Some((term, max_edits)) = parse_fuzzy_syntax(&lit.phrase) {
                // Grammar parses `foz~1` as phrase="foz~1" — detect fuzzy ourselves.
                analyze_fuzzy(term, max_edits, bm25)
            } else {
                analyze_terms(&lit.phrase, bm25)
            }
        }
        UserInputLeaf::All => Query::All,
        UserInputLeaf::Set { elements, .. } => {
            // IN set: full analysis per element, union all term IDs (and n-gram slots
            // when `max_n >= 2`). Both sides are deduped across elements so the
            // resulting `UniAndNgram`/`Terms` IR has no duplicates.
            let mut uni_seen: HashSet<u32> = HashSet::new();
            let mut uni_ids: Vec<u32> = Vec::new();
            let mut ng_seen: HashSet<u32> = HashSet::new();
            let mut ngram_slots: Vec<u32> = Vec::new();
            for e in elements {
                let (uni, ng) = bm25.query_targets(e);
                for tid in uni {
                    if uni_seen.insert(tid) { uni_ids.push(tid); }
                }
                for slot in ng {
                    if ng_seen.insert(slot) { ngram_slots.push(slot); }
                }
            }
            if uni_ids.is_empty() && ngram_slots.is_empty() {
                Query::Empty
            } else if ngram_slots.is_empty() {
                match uni_ids.len() {
                    1 => Query::Term(uni_ids[0]),
                    _ => Query::Terms(uni_ids),
                }
            } else {
                Query::UniAndNgram { uni_ids, ngram_slots }
            }
        }
        UserInputLeaf::Exists { .. } => {
            // Single-field index: every document "exists".
            Query::All
        }
        UserInputLeaf::Regex { pattern, .. } => analyze_regex(pattern, bm25),
        UserInputLeaf::Range { .. } => Query::Empty,
    }
}

// ── Analysis pipelines ──────────────────────────────────────────────────────

/// Unquoted terms: full analysis pipeline → bag-of-words OR.
///
/// When the index has `max_n >= 2` and any of the analyzed n-grams hash to a
/// slot with `df > 0`, the result is a `UniAndNgram` variant carrying both
/// tiers' targets. Otherwise we fall back to the simpler `Term`/`Terms`
/// variants for a smaller IR and lower scorer overhead.
fn analyze_terms(text: &str, bm25: &BM25) -> Query {
    let (uni_ids, ngram_slots) = bm25.query_targets(text);
    if uni_ids.is_empty() && ngram_slots.is_empty() {
        return Query::Empty;
    }
    if ngram_slots.is_empty() {
        return match uni_ids.len() {
            1 => Query::Term(uni_ids[0]),
            _ => Query::Terms(uni_ids),
        };
    }
    Query::UniAndNgram { uni_ids, ngram_slots }
}

/// Quoted phrase: full analysis per token, then positional scoring.
fn analyze_phrase(text: &str, slop: u32, bm25: &BM25) -> Query {
    let term_ids = resolve_tokens(text, bm25);
    match term_ids.len() {
        0 => Query::Empty,
        1 => Query::Term(term_ids[0]),
        _ => Query::Phrase { terms: term_ids, slop },
    }
}

/// Wildcard prefix: union of raw-lowered and stemmed prefix scans.
///
/// This handles the stemming mismatch: the vocab contains stemmed terms, but
/// the user may type an inflected prefix (e.g. `running*`). We try both:
/// - `running*` → raw lowered "running" prefix scan (no vocab hits, but correct for partial words)
/// - `running*` → stemmed "run" prefix scan (matches "run" in vocab)
///
/// The union ensures we never miss results, regardless of whether the prefix
/// is a complete word or a partial fragment.
fn analyze_wildcard(phrase: &str, bm25: &BM25) -> Query {
    let raw_prefix = if phrase.ends_with('*') {
        &phrase[..phrase.len() - 1]
    } else {
        phrase
    };

    let lowered = raw_prefix.to_lowercase();
    let stemmed = bm25.stem_token(raw_prefix);

    // Scan vocab for terms matching either prefix, deduplicate via HashSet.
    let mut matched_ids: HashSet<u32> = HashSet::new();
    for (term, &id) in bm25.get_vocab() {
        if term.starts_with(&lowered) || (stemmed != lowered && term.starts_with(&stemmed)) {
            matched_ids.insert(id);
        }
    }

    let term_ids: Vec<u32> = matched_ids.into_iter().collect();
    if term_ids.is_empty() {
        Query::Empty
    } else {
        Query::Terms(term_ids)
    }
}

/// Regex: raw pattern matched against the stemmed term dictionary.
///
/// The regex is NOT analyzed — it matches directly against vocab keys,
/// which are already stemmed. This matches Lucene/Elasticsearch behavior.
fn analyze_regex(pattern: &str, bm25: &BM25) -> Query {
    let re = match Regex::new(&format!("^{}$", pattern)) {
        Ok(re) => re,
        Err(_) => return Query::Empty,
    };
    let term_ids: Vec<u32> = bm25
        .get_vocab()
        .iter()
        .filter(|(term, _)| re.is_match(term))
        .map(|(_, &id)| id)
        .collect();
    if term_ids.is_empty() {
        Query::Empty
    } else {
        Query::Terms(term_ids)
    }
}

/// Fuzzy query: full analysis → Levenshtein distance scan on stemmed vocab.
///
/// Matches Lucene's `FuzzyQuery` behavior:
/// 1. Analyze the input term (lowercase + normalize + stem)
/// 2. Scan the vocab for all terms within `max_edits` Levenshtein distance
/// 3. Return as OR of matching terms
///
/// `max_edits` is clamped to 2 (Lucene's maximum) to avoid runaway expansion.
fn analyze_fuzzy(text: &str, max_edits: u32, bm25: &BM25) -> Query {
    let tokens = bm25.tokenize(text);
    if tokens.is_empty() {
        return Query::Empty;
    }

    // Clamp to 2 edits max (Lucene convention — higher values produce too many false positives)
    let max_edits = max_edits.min(2) as usize;

    // For each analyzed token, find vocab terms within edit distance
    let mut matched_ids: HashSet<u32> = HashSet::new();
    for token in &tokens {
        for (vocab_term, &id) in bm25.get_vocab() {
            if levenshtein(token, vocab_term) <= max_edits {
                matched_ids.insert(id);
            }
        }
    }

    let term_ids: Vec<u32> = matched_ids.into_iter().collect();
    if term_ids.is_empty() {
        Query::Empty
    } else {
        Query::Terms(term_ids)
    }
}

/// Levenshtein edit distance (insertions, deletions, substitutions).
///
/// Uses the standard single-row DP optimization: O(min(m,n)) space.
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (m, n) = (a.len(), b.len());

    if m == 0 { return n; }
    if n == 0 { return m; }

    // Single-row DP: prev[j] = edit distance for a[..i] vs b[..j]
    let mut prev: Vec<usize> = (0..=n).collect();

    for i in 1..=m {
        let mut curr = vec![0; n + 1];
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1)           // deletion
                .min(curr[j - 1] + 1)          // insertion
                .min(prev[j - 1] + cost);      // substitution
        }
        prev = curr;
    }

    prev[n]
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Tokenize text through the full pipeline (lowercase + normalize + stem)
/// and resolve each token to its vocab ID, skipping unknowns.
fn resolve_tokens(text: &str, bm25: &BM25) -> Vec<u32> {
    bm25.tokenize(text)
        .iter()
        .filter_map(|token| bm25.get_term_id(token))
        .collect()
}

/// Detect `term~N` fuzzy syntax in a phrase string.
///
/// The tantivy grammar doesn't separate `~N` from unquoted terms —
/// `foz~1` is parsed as `phrase="foz~1"`. We detect this pattern ourselves.
/// Returns `(term, max_edits)` if the pattern matches.
fn parse_fuzzy_syntax(phrase: &str) -> Option<(&str, u32)> {
    // Find the last `~` followed by 1-2 digits at the end of the string.
    let tilde_pos = phrase.rfind('~')?;
    if tilde_pos == 0 {
        return None; // "~2" is not a valid fuzzy query (no term)
    }
    let suffix = &phrase[tilde_pos + 1..];
    let max_edits: u32 = suffix.parse().ok()?;
    let term = &phrase[..tilde_pos];
    // Sanity: term must be non-empty and not contain spaces (single term only)
    if term.is_empty() || term.contains(char::is_whitespace) {
        return None;
    }
    Some((term, max_edits))
}
