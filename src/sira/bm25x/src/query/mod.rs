// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Query engine: parse → analyze → score.
//!
//! Separates text analysis from scoring, following the Lucene architecture:
//!
//! 1. **Parse**: `tantivy_query_grammar` produces an AST from the query string.
//! 2. **Analyze** ([`analyze`]): rewrites the AST into a [`Query`] IR, applying
//!    the correct text processing per node type (stemming for terms/phrases,
//!    stem+raw union for wildcards, raw pattern for regex).
//! 3. **Score** ([`score`]): evaluates the `Query` against the inverted index
//!    using BM25 scoring. Knows nothing about text analysis.

mod analyze;
mod score;

use tantivy_query_grammar::{Occur, UserInputAst};

use crate::index::{SearchResult, BM25};

/// A fully-analyzed query ready for scoring.
///
/// All text processing (stemming, normalization) is complete.
/// All terms are resolved to vocab IDs or patterns are expanded.
pub(crate) enum Query {
    /// Match documents containing this term (single vocab ID).
    Term(u32),

    /// Match documents containing any of these terms (OR semantics, additive BM25).
    Terms(Vec<u32>),

    /// Combined unigram term_ids and n-gram side slots — emitted by `analyze_terms`
    /// (and the `Set` case) when `max_n >= 2` and the analyzed text yields any
    /// hashed n-gram slot with `df > 0`. Both vecs are deduped. Scoring is
    /// additive across both tiers, mirroring `BM25::search`.
    UniAndNgram { uni_ids: Vec<u32>, ngram_slots: Vec<u32> },

    /// Match documents where terms appear in order with allowed slop.
    Phrase { terms: Vec<u32>, slop: u32 },

    /// Boolean combination of sub-queries (AND/OR/NOT/+/-).
    Boolean(Vec<(Occur, Query)>),

    /// Multiply scores of the inner query by a factor.
    Boost { inner: Box<Query>, factor: f32 },

    /// Match all documents (score 1.0).
    All,

    /// Match no documents.
    Empty,
}

/// Evaluate a query expression against the index, returning top-k results.
///
/// This is the main entry point. Flow: AST → analyze → score.
pub(crate) fn evaluate(ast: &UserInputAst, bm25: &BM25, k: usize) -> Vec<SearchResult> {
    if bm25.num_docs == 0 {
        return Vec::new();
    }
    let query = analyze::analyze(ast, bm25);
    score::execute(&query, bm25, k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Method, TokenizerMode};
    use std::collections::HashSet;

    fn test_index() -> BM25 {
        let mut index =
            BM25::with_tokenizer(Method::Lucene, 1.5, 0.75, 0.5, TokenizerMode::UnicodeStem, true);
        index
            .add(&[
                "the quick brown fox jumps over the lazy dog",
                "a lazy dog sleeps in the sun all day",
                "the fast cat chases a red ball quickly",
                "a brown bear fishes in the river for salmon",
                "the fox and the cat are friends in the forest",
                "a quick red car drives over the bridge",
            ])
            .unwrap();
        index
    }

    fn search(index: &BM25, query: &str, k: usize) -> Vec<usize> {
        let (ast, _) = tantivy_query_grammar::parse_query_lenient(query);
        evaluate(&ast, index, k)
            .iter()
            .map(|r| r.index)
            .collect()
    }

    fn search_set(index: &BM25, query: &str) -> HashSet<usize> {
        search(index, query, 10).into_iter().collect()
    }

    // ── Basic term queries ─────────────────────────────────────────────

    #[test]
    fn test_basic_term() {
        let idx = test_index();
        let ids = search_set(&idx, "fox");
        assert!(ids.contains(&0));
        assert!(ids.contains(&4));
        assert!(!ids.contains(&1));
    }

    #[test]
    fn test_empty_query() {
        let idx = test_index();
        let (ast, _) = tantivy_query_grammar::parse_query_lenient("");
        let results = evaluate(&ast, &idx, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_unknown_term() {
        let idx = test_index();
        assert!(search_set(&idx, "xylophone").is_empty());
    }

    #[test]
    fn test_matches_bag_of_words() {
        let idx = test_index();
        let expr_results = search(&idx, "fox", 10);
        let bow_results: Vec<usize> = idx.search("fox", 10).iter().map(|r| r.index).collect();
        assert_eq!(expr_results, bow_results);
    }

    // ── Boolean queries ────────────────────────────────────────────────

    #[test]
    fn test_boolean_and() {
        let idx = test_index();
        let ids = search_set(&idx, "fox AND dog");
        assert!(ids.contains(&0));
        assert!(!ids.contains(&4));
        assert!(!ids.contains(&1));
    }

    #[test]
    fn test_boolean_or() {
        let idx = test_index();
        let ids = search_set(&idx, "fox OR cat");
        assert!(ids.contains(&0));
        assert!(ids.contains(&2));
        assert!(ids.contains(&4));
        assert!(!ids.contains(&1));
    }

    #[test]
    fn test_must_mustnot() {
        let idx = test_index();
        let ids = search_set(&idx, "+fox -dog");
        assert!(ids.contains(&4));
        assert!(!ids.contains(&0));
    }

    #[test]
    fn test_not_operator() {
        let idx = test_index();
        let ids = search_set(&idx, "brown NOT bear");
        assert!(ids.contains(&0));
        assert!(!ids.contains(&3));
    }

    #[test]
    fn test_parenthesized_grouping() {
        let idx = test_index();
        let ids = search_set(&idx, "(fox OR cat) AND brown");
        assert!(ids.contains(&0));
        assert!(!ids.contains(&3));
        assert!(!ids.contains(&4));
    }

    #[test]
    fn test_nested_grouping() {
        let idx = test_index();
        let ids = search_set(&idx, "fox AND (dog OR cat)");
        assert!(ids.contains(&0));
        assert!(ids.contains(&4));
        assert!(!ids.contains(&1));
    }

    // ── Boost ──────────────────────────────────────────────────────────

    #[test]
    fn test_boost() {
        let idx = test_index();
        let results = search(&idx, "fox^10 dog", 10);
        assert!(!results.is_empty());
        let ids: HashSet<usize> = results.iter().copied().collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&4));
    }

    // ── Phrase queries ─────────────────────────────────────────────────

    #[test]
    fn test_phrase_exact_match() {
        let idx = test_index();
        let ids = search_set(&idx, "\"quick brown\"");
        assert!(ids.contains(&0));
        assert!(!ids.contains(&1));
        let ids2 = search_set(&idx, "\"brown quick\"");
        assert!(!ids2.contains(&0));
    }

    #[test]
    fn test_phrase_adjacency_required() {
        let idx = test_index();
        let ids = search_set(&idx, "\"fox dog\"");
        assert!(!ids.contains(&0));
    }

    #[test]
    fn test_phrase_three_words() {
        let idx = test_index();
        let ids = search_set(&idx, "\"quick brown fox\"");
        assert!(ids.contains(&0));
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn test_phrase_no_match() {
        let idx = test_index();
        assert!(search_set(&idx, "\"quick cat\"").is_empty());
    }

    #[test]
    fn test_phrase_slop() {
        let idx = test_index();
        assert!(!search_set(&idx, "\"quick fox\"").contains(&0));
        assert!(search_set(&idx, "\"quick fox\"~1").contains(&0));
    }

    #[test]
    fn test_phrase_and_boolean() {
        let idx = test_index();
        let ids = search_set(&idx, "\"quick brown\" AND cat");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_phrase_survives_save_load() {
        let idx = test_index();
        let dir = tempfile::tempdir().unwrap();
        idx.save(dir.path()).unwrap();
        let loaded = BM25::load(dir.path(), false).unwrap();

        assert!(search_set(&loaded, "\"quick brown\"").contains(&0));
        assert!(
            !search_set(&loaded, "\"brown quick\"").contains(&0),
            "phrase order not enforced after load"
        );
    }

    // ── Wildcard queries ───────────────────────────────────────────────

    #[test]
    fn test_wildcard_prefix() {
        let idx = test_index();
        let ids = search_set(&idx, "fox*");
        assert!(ids.contains(&0));
        assert!(ids.contains(&4));
        assert!(!ids.contains(&1));
    }

    #[test]
    fn test_wildcard_prefix_broad() {
        let idx = test_index();
        let ids = search_set(&idx, "brow*");
        assert!(ids.contains(&0));
        assert!(ids.contains(&3));
    }

    /// Wildcard with an inflected prefix: `running*` should stem to `run*`
    /// and match documents containing "run", "runs", "running", etc.
    #[test]
    fn test_wildcard_stemmed_prefix() {
        let idx = test_index();
        // "jumps" is stemmed to "jump" in the vocab.
        // "jumps*" with stemming → prefix "jump" → matches doc 0.
        let ids = search_set(&idx, "jumps*");
        assert!(ids.contains(&0), "stemmed wildcard should match 'jumps' via 'jump'");
    }

    /// `cats*` should stem to `cat*` and match "cat" / "chases" stems.
    #[test]
    fn test_wildcard_cats() {
        let idx = test_index();
        let ids = search_set(&idx, "cats*");
        // "cats" stems to "cat", which matches doc 2 ("cat") and doc 4 ("cat")
        assert!(ids.contains(&2), "cats* should match 'cat' in doc 2");
        assert!(ids.contains(&4), "cats* should match 'cat' in doc 4");
    }

    // ── Regex queries ──────────────────────────────────────────────────

    #[test]
    fn test_regex() {
        let idx = test_index();
        let ids = search_set(&idx, "/fox.*/");
        assert!(ids.contains(&0));
        assert!(ids.contains(&4));
        assert!(!ids.contains(&1));
    }

    #[test]
    fn test_regex_suffix() {
        let idx = test_index();
        let ids = search_set(&idx, "/.*on/");
        assert!(ids.contains(&3)); // salmon
    }

    // ── IN set queries ─────────────────────────────────────────────────

    #[test]
    fn test_in_set() {
        let idx = test_index();
        let ids = search_set(&idx, "body: IN [fox bear]");
        assert!(ids.contains(&0));
        assert!(ids.contains(&3));
        assert!(ids.contains(&4));
        assert!(!ids.contains(&1));
    }

    #[test]
    fn test_exists() {
        let idx = test_index();
        let ids = search_set(&idx, "body:*");
        assert_eq!(ids.len(), 6);
    }

    #[test]
    fn test_field_name_ignored() {
        let idx = test_index();
        assert_eq!(search_set(&idx, "title:fox"), search_set(&idx, "fox"));
    }

    // ── Fuzzy queries ──────────────────────────────────────────────────

    /// `foz~1` should match "fox" (1 edit: z→x).
    #[test]
    fn test_fuzzy_single_edit() {
        let idx = test_index();
        let ids = search_set(&idx, "foz~1");
        assert!(ids.contains(&0), "foz~1 should match 'fox' in doc 0");
        assert!(ids.contains(&4), "foz~1 should match 'fox' in doc 4");
    }

    /// `dg~1` should match "dog" (1 edit: insertion of 'o').
    #[test]
    fn test_fuzzy_insertion() {
        let idx = test_index();
        let ids = search_set(&idx, "dg~1");
        assert!(ids.contains(&0), "dg~1 should match 'dog' in doc 0");
        assert!(ids.contains(&1), "dg~1 should match 'dog' in doc 1");
    }

    /// `foxes~1` is stemmed to "fox" first, then fuzzy matched — exact hit.
    #[test]
    fn test_fuzzy_with_stemming() {
        let idx = test_index();
        let ids = search_set(&idx, "foxes~1");
        // "foxes" stems to "fox", edit distance 0 from vocab "fox" → match
        assert!(ids.contains(&0));
        assert!(ids.contains(&4));
    }

    /// `xyz~1` has no vocab neighbors within 1 edit.
    #[test]
    fn test_fuzzy_no_match() {
        let idx = test_index();
        assert!(search_set(&idx, "xyz~1").is_empty());
    }

    /// `fax~2` should match "fox" (2 edits: a→o, but only 1 edit actually).
    /// Also tests that max_edits=2 works.
    #[test]
    fn test_fuzzy_two_edits() {
        let idx = test_index();
        let ids = search_set(&idx, "fax~2");
        assert!(ids.contains(&0), "fax~2 should match 'fox' (1 edit a→o)");
    }

    /// `fax~1` should also match "fox" (1 substitution: a→o).
    #[test]
    fn test_fuzzy_fax_one_edit() {
        let idx = test_index();
        let ids = search_set(&idx, "fax~1");
        assert!(ids.contains(&0), "fax~1 should match 'fox' (a→o)");
    }

    /// Fuzzy combined with boolean: `foz~1 AND cat`
    #[test]
    fn test_fuzzy_with_boolean() {
        let idx = test_index();
        let ids = search_set(&idx, "foz~1 AND cat");
        // doc 4 has fox + cat
        assert!(ids.contains(&4));
        // doc 0 has fox but no cat
        assert!(!ids.contains(&0));
    }

    // ── Multi-gram: UniAndNgram variant ───────────────────────────────────

    #[test]
    fn search_expr_with_max_n_2_matches_bigram() {
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 2, 1 << 16,
        );
        idx.add(&["machine learning is the future", "the kitchen sink"]).unwrap();
        let r = idx.search_expr("machine learning", 5);
        assert!(!r.is_empty());
        assert_eq!(r[0].index, 0);
    }

    #[test]
    fn search_expr_phrase_unaffected_by_max_n() {
        // Phrase queries continue to use unigram positions only; n-gram tier untouched.
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 4, 1 << 16,
        );
        idx.add(&["alpha beta gamma", "alpha gamma beta"]).unwrap();
        let r = idx.search_expr("\"alpha beta\"", 5);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].index, 0);
    }

    #[test]
    fn wildcard_does_not_leak_into_ngram_tier() {
        // Wildcard matches unigram vocab only (vocab is unigram-only by construction).
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 2, 1 << 16,
        );
        idx.add(&["machine learning"]).unwrap();
        let r = idx.search_expr("machin*", 10);
        assert_eq!(r.len(), 1, "exactly doc 0 from unigram match; no double-count");
    }
}
