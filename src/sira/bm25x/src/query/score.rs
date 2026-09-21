// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Query scoring: [`Query`] → scored results.
//!
//! This module knows nothing about text analysis — it receives fully-resolved
//! term IDs and query structure, and computes BM25 scores against the index.

use std::collections::{HashMap, HashSet};

use tantivy_query_grammar::Occur;

use super::Query;
use crate::index::{SearchResult, BM25};
use crate::scoring::{self, ScoringParams};

/// Execute a resolved query against the index, returning top-k results.
pub(super) fn execute(query: &Query, bm25: &BM25, k: usize) -> Vec<SearchResult> {
    let sub = score_query(query, bm25);
    sub_to_topk(sub, bm25, k)
}

// ── Internal result type ────────────────────────────────────────────────────

/// Intermediate result: doc_id → score for docs matching a subquery.
struct SubResult {
    scores: HashMap<u32, f32>,
}

impl SubResult {
    fn empty() -> Self {
        Self {
            scores: HashMap::new(),
        }
    }

    fn doc_ids(&self) -> HashSet<u32> {
        self.scores.keys().copied().collect()
    }
}

// ── Query dispatch ──────────────────────────────────────────────────────────

/// Recursively score a query node.
fn score_query(query: &Query, bm25: &BM25) -> SubResult {
    match query {
        Query::Term(term_id) => score_term_ids(&[*term_id], bm25, false),
        Query::Terms(term_ids) => score_term_ids(term_ids, bm25, false),
        Query::UniAndNgram { uni_ids, ngram_slots } => {
            score_uni_and_ngram(uni_ids, ngram_slots, bm25)
        }
        Query::Phrase { terms, slop } => phrase_match(terms, *slop, bm25),
        Query::Boolean(clauses) => score_boolean(clauses, bm25),
        Query::Boost { inner, factor } => {
            let mut sub = score_query(inner, bm25);
            for score in sub.scores.values_mut() {
                *score *= factor;
            }
            sub
        }
        Query::All => {
            let mut scores = HashMap::new();
            for doc_id in 0..bm25.num_docs {
                scores.insert(doc_id, 1.0);
            }
            SubResult { scores }
        }
        Query::Empty => SubResult::empty(),
    }
}

/// Score the combined unigram-tier + n-gram-tier query (same loop shape as
/// `BM25::search`, but emits a `SubResult` so it composes inside boolean
/// expressions). Mirrors `BM25::search`'s scoring discipline: per-doc unigram +
/// n-gram contributions sum into a single accumulator.
///
/// The unigram half is gated on [`BM25::score_unigram`] — when the user
/// selected an n-gram set without `1` (e.g. bigram-only BM25), unigram
/// postings are skipped entirely so the score reflects only the chosen tiers.
fn score_uni_and_ngram(uni_ids: &[u32], ngram_slots: &[u32], bm25: &BM25) -> SubResult {
    let params = ScoringParams {
        k1: bm25.k1,
        b: bm25.b,
        delta: bm25.delta,
        avgdl: bm25.total_tokens as f32 / bm25.num_docs as f32,
    };

    let mut scores: HashMap<u32, f32> = HashMap::new();

    if bm25.score_unigram() {
        for &tid in uni_ids {
            let df = bm25.doc_freqs.get(tid as usize).copied().unwrap_or(0);
            if df == 0 {
                continue;
            }
            let idf_val = scoring::idf(bm25.method, bm25.num_docs, df);
            bm25.for_each_posting(tid, |doc_id, tf| {
                let dl = bm25.get_doc_length(doc_id);
                let s = scoring::score(bm25.method, tf, dl, &params, idf_val);
                *scores.entry(doc_id).or_insert(0.0) += s;
            });
        }
    }

    if let Some(side) = bm25.ngram_side() {
        for &slot in ngram_slots {
            let df = side.df(slot);
            if df == 0 {
                continue;
            }
            let idf_val = scoring::idf(bm25.method, bm25.num_docs, df);
            side.for_each_posting(slot, |doc_id, tf| {
                let dl = bm25.get_doc_length(doc_id);
                let s = scoring::score(bm25.method, tf, dl, &params, idf_val);
                *scores.entry(doc_id).or_insert(0.0) += s;
            });
        }
    }

    SubResult { scores }
}

// ── Term scoring ────────────────────────────────────────────────────────────

/// Score documents for a set of term IDs using BM25.
///
/// If `require_all` is true, only docs matching every term are returned (AND).
/// Otherwise, scores are additive across terms (OR).
///
/// Returns empty when the BM25 index has `score_unigram == false` (e.g.
/// `ngrams=[2]`): pure-bigram BM25 has no notion of "score for a single
/// term", and propagating an unscored unigram contribution would silently
/// re-introduce the unigram tier into the BM25 sum.
fn score_term_ids(term_ids: &[u32], bm25: &BM25, require_all: bool) -> SubResult {
    if !bm25.score_unigram() {
        return SubResult::empty();
    }
    let params = ScoringParams {
        k1: bm25.k1,
        b: bm25.b,
        delta: bm25.delta,
        avgdl: bm25.total_tokens as f32 / bm25.num_docs as f32,
    };

    let mut scores: HashMap<u32, f32> = HashMap::new();
    let mut term_hits: HashMap<u32, u32> = HashMap::new();
    let mut seen_terms: HashSet<u32> = HashSet::new();
    let mut num_unique_terms = 0u32;

    for &term_id in term_ids {
        if !seen_terms.insert(term_id) {
            continue;
        }
        num_unique_terms += 1;

        let df = bm25.doc_freqs.get(term_id as usize).copied().unwrap_or(0);
        if df == 0 {
            if require_all {
                return SubResult::empty();
            }
            continue;
        }

        let idf_val = scoring::idf(bm25.method, bm25.num_docs, df);

        bm25.for_each_posting(term_id, |doc_id, tf| {
            let dl = bm25.get_doc_length(doc_id);
            let s = scoring::score(bm25.method, tf, dl, &params, idf_val);
            *scores.entry(doc_id).or_insert(0.0) += s;
            if require_all {
                *term_hits.entry(doc_id).or_insert(0) += 1;
            }
        });
    }

    if require_all && num_unique_terms > 0 {
        scores.retain(|doc_id, _| {
            term_hits.get(doc_id).copied().unwrap_or(0) == num_unique_terms
        });
    }

    SubResult { scores }
}

// ── Phrase scoring ──────────────────────────────────────────────────────────

/// Match a phrase query using the positional index.
///
/// Finds documents where all terms appear in the correct order with allowed slop.
/// Falls back to AND semantics if no positions are available (v1 index).
///
/// Returns empty when `score_unigram == false`: phrase matching reads from
/// the unigram positional index and contributes unigram-style BM25 scores;
/// propagating those when the user excluded n=1 from the score set would
/// silently re-introduce the unigram tier.
fn phrase_match(term_ids: &[u32], slop: u32, bm25: &BM25) -> SubResult {
    if !bm25.score_unigram() {
        return SubResult::empty();
    }
    if term_ids.is_empty() {
        return SubResult::empty();
    }
    if term_ids.len() == 1 {
        return score_term_ids(term_ids, bm25, false);
    }

    let positions = bm25.get_positions();

    // Check if positions are available for all terms
    let has_positions = term_ids.iter().all(|&tid| {
        let tid = tid as usize;
        tid < positions.len() && !positions[tid].is_empty()
    });

    if !has_positions {
        // Fall back to AND semantics (v1 index without positions)
        return score_term_ids(term_ids, bm25, true);
    }

    // Find candidate docs: intersection of all terms' posting lists
    let postings = bm25.get_postings();
    let first_docs: HashSet<u32> = postings[term_ids[0] as usize]
        .iter()
        .map(|&(doc_id, _)| doc_id)
        .collect();
    let mut candidates = first_docs;
    for &tid in &term_ids[1..] {
        let term_docs: HashSet<u32> = postings[tid as usize]
            .iter()
            .map(|&(doc_id, _)| doc_id)
            .collect();
        candidates.retain(|id| term_docs.contains(id));
    }

    let params = ScoringParams {
        k1: bm25.k1,
        b: bm25.b,
        delta: bm25.delta,
        avgdl: bm25.total_tokens as f32 / bm25.num_docs as f32,
    };

    let mut scores: HashMap<u32, f32> = HashMap::new();

    for &doc_id in &candidates {
        let mut term_positions: Vec<&[u32]> = Vec::with_capacity(term_ids.len());
        let mut all_found = true;

        for &tid in term_ids {
            let tid = tid as usize;
            let plist = &postings[tid];
            match plist.binary_search_by_key(&doc_id, |&(did, _)| did) {
                Ok(idx) => {
                    if idx < positions[tid].len() {
                        term_positions.push(&positions[tid][idx]);
                    } else {
                        all_found = false;
                        break;
                    }
                }
                Err(_) => {
                    all_found = false;
                    break;
                }
            }
        }

        if !all_found {
            continue;
        }

        if check_phrase_positions(&term_positions, slop) {
            let mut total = 0.0f32;
            for &tid in term_ids {
                let df = bm25.doc_freqs.get(tid as usize).copied().unwrap_or(0);
                if df == 0 {
                    continue;
                }
                let idf_val = scoring::idf(bm25.method, bm25.num_docs, df);
                let plist = &postings[tid as usize];
                if let Ok(idx) = plist.binary_search_by_key(&doc_id, |&(did, _)| did) {
                    let tf = plist[idx].1;
                    let dl = bm25.get_doc_length(doc_id);
                    total += scoring::score(bm25.method, tf, dl, &params, idf_val);
                }
            }
            scores.insert(doc_id, total);
        }
    }

    SubResult { scores }
}

/// Check if positions form a valid phrase sequence with allowed slop.
fn check_phrase_positions(term_positions: &[&[u32]], slop: u32) -> bool {
    if term_positions.is_empty() {
        return false;
    }
    if term_positions.len() == 1 {
        return !term_positions[0].is_empty();
    }

    for &start_pos in term_positions[0] {
        let mut expected = start_pos;
        let mut matched = true;

        for positions in term_positions.iter().skip(1) {
            expected += 1;
            let found = positions.iter().any(|&p| p >= expected && p <= expected + slop);
            if !found {
                matched = false;
                break;
            }
            if let Some(&actual) = positions.iter().find(|&&p| p >= expected && p <= expected + slop)
            {
                expected = actual;
            }
        }

        if matched {
            return true;
        }
    }
    false
}

// ── Boolean scoring ─────────────────────────────────────────────────────────

/// Evaluate a boolean clause (AND/OR/NOT/+/-).
///
/// Follows Lucene-style boolean semantics:
/// - Must clauses: docs must match ALL. Should clauses add bonus score.
/// - No Must: docs must match at least one Should.
/// - MustNot: always excludes matching docs.
fn score_boolean(clauses: &[(Occur, Query)], bm25: &BM25) -> SubResult {
    if clauses.is_empty() {
        return SubResult::empty();
    }

    // Single clause with MustNot → match all docs EXCEPT these
    if clauses.len() == 1 {
        let (occur, query) = &clauses[0];
        let sub = score_query(query, bm25);
        return match occur {
            Occur::MustNot => {
                let excluded = sub.doc_ids();
                let mut scores = HashMap::new();
                for doc_id in 0..bm25.num_docs {
                    if !excluded.contains(&doc_id) {
                        scores.insert(doc_id, 1.0);
                    }
                }
                SubResult { scores }
            }
            _ => sub,
        };
    }

    // Evaluate all sub-queries
    let evaluated: Vec<(Occur, SubResult)> = clauses
        .iter()
        .map(|(occur, query)| (*occur, score_query(query, bm25)))
        .collect();

    // Partition into Must, Should, MustNot
    let mut must_results: Vec<&SubResult> = Vec::new();
    let mut should_results: Vec<&SubResult> = Vec::new();
    let mut must_not_ids: HashSet<u32> = HashSet::new();

    for (occur, sub) in &evaluated {
        match occur {
            Occur::Must => must_results.push(sub),
            Occur::MustNot => {
                must_not_ids.extend(sub.doc_ids());
            }
            Occur::Should => should_results.push(sub),
        }
    }

    let mut result_scores: HashMap<u32, f32> = HashMap::new();

    if !must_results.is_empty() {
        // Must mode: intersect all Must results
        let mut candidates: HashSet<u32> = must_results[0].doc_ids();
        for must in &must_results[1..] {
            let other = must.doc_ids();
            candidates.retain(|id| other.contains(id));
        }

        for doc_id in &candidates {
            let mut total = 0.0f32;
            for must in &must_results {
                if let Some(&s) = must.scores.get(doc_id) {
                    total += s;
                }
            }
            for should in &should_results {
                if let Some(&s) = should.scores.get(doc_id) {
                    total += s;
                }
            }
            result_scores.insert(*doc_id, total);
        }
    } else {
        // No Must: pure Should/OR mode — union all Should results
        for should in &should_results {
            for (&doc_id, &score) in &should.scores {
                *result_scores.entry(doc_id).or_insert(0.0) += score;
            }
        }
    }

    // Exclude MustNot docs
    for id in &must_not_ids {
        result_scores.remove(id);
    }

    SubResult {
        scores: result_scores,
    }
}

// ── Top-k extraction ────────────────────────────────────────────────────────

/// Convert a SubResult to sorted top-k SearchResults.
fn sub_to_topk(sub: SubResult, bm25: &BM25, k: usize) -> Vec<SearchResult> {
    crate::index::with_score_buf(bm25.num_docs as usize, |scores_flat| {
        let mut touched = Vec::with_capacity(sub.scores.len());
        for (&doc_id, &score) in &sub.scores {
            if score > 0.0 && doc_id < bm25.num_docs {
                scores_flat[doc_id as usize] = score;
                touched.push(doc_id);
            }
        }
        let result = BM25::topk_from_scores(scores_flat, &touched, k);
        for &doc_id in &touched {
            scores_flat[doc_id as usize] = 0.0;
        }
        result
    })
}
