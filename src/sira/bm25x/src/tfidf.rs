// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Decoupled TF-IDF keyword extraction module.
//!
//! Computes per-document TF-IDF scores and extracts top-k keywords.
//! Fully independent from the BM25 index — uses its own vocabulary and IDF.
//!
//! Two modes:
//! - **Vocabulary mode** (`use_hashing=false`): builds a full vocabulary.
//!   Best for small-to-medium corpora or unigram-only workloads.
//! - **Hashing mode** (`use_hashing=true`): hashes n-grams to a fixed-size
//!   feature space. Memory-bounded for large corpora with high-order n-grams.
//!
//! Reuses the project's [`Tokenizer`] for consistent text processing,
//! with n-gram generation on top.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

use crate::ngram::{iter_ngram_slots, iter_ngram_slots_with_str, ngram_tf_map};
use crate::tokenizer::{Tokenizer, TokenizerMode, ENGLISH_STOPWORDS};

/// Create a progress bar for TF-IDF phases.
fn tfidf_progress(len: u64, msg: &str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::with_template("{msg} [{bar:40.cyan/blue}] {pos}/{len} ({elapsed})")
            .unwrap()
            .progress_chars("=>-"),
    );
    pb.set_message(msg.to_string());
    pb
}

/// Configuration for TF-IDF keyword extraction.
pub struct TfidfConfig {
    /// Number of top keywords to extract per document.
    pub top_k: usize,
    /// N-gram range `(min_n, max_n)`. `(1, 1)` = unigrams, `(1, 3)` = uni+bi+trigrams.
    pub ngram_range: (usize, usize),
    /// Use sublinear TF: `1 + log(tf)` instead of raw `tf`.
    pub sublinear_tf: bool,
    /// Minimum document frequency to keep a term in the vocabulary.
    pub min_df: u32,
    /// Use hashing mode for memory-bounded n-gram processing.
    pub use_hashing: bool,
    /// Hash space size (must be a power of 2). Only used when `use_hashing=true`.
    pub n_features: usize,
    /// Remove n-grams that are word sub-phrases of a higher-scored kept keyword.
    pub dedup: bool,
}

impl Default for TfidfConfig {
    fn default() -> Self {
        TfidfConfig {
            top_k: 20,
            ngram_range: (1, 1),
            sublinear_tf: true,
            min_df: 1,
            use_hashing: false,
            n_features: 1 << 23, // 8M
            dedup: false,
        }
    }
}

/// TF-IDF keyword extractor.
///
/// Fits a vocabulary and IDF values from a corpus, then extracts top-k
/// keywords per document.
pub struct Tfidf {
    config: TfidfConfig,
    tokenizer: Tokenizer,
    /// Processed stopwords for post-filtering unigram stopwords from results.
    /// Populated when `use_stopwords=false` (stopwords kept during tokenization
    /// to preserve n-grams like "king of the world", but unigrams filtered after).
    post_filter_stopwords: Option<HashSet<String>>,
    // Fitted state (populated after fit_transform)
    vocab: HashMap<String, u32>,
    terms: Vec<String>,
    idf: Vec<f32>,
    num_docs: u32,
}

impl Tfidf {
    /// Create a new TF-IDF extractor.
    pub fn new(config: TfidfConfig, mode: TokenizerMode, use_stopwords: bool) -> Self {
        if config.use_hashing {
            assert!(
                config.n_features.is_power_of_two(),
                "n_features must be a power of 2, got {}",
                config.n_features
            );
        }
        let tokenizer = Tokenizer::with_mode(mode, use_stopwords);
        // When stopwords are NOT removed during tokenization, build a processed
        // stopword set for post-filtering unigram stopwords from results.
        let post_filter_stopwords = if !use_stopwords {
            Some(
                ENGLISH_STOPWORDS
                    .iter()
                    .map(|w| tokenizer.stem_single(w))
                    .collect(),
            )
        } else {
            None
        };
        Tfidf {
            config,
            tokenizer,
            post_filter_stopwords,
            vocab: HashMap::new(),
            terms: Vec::new(),
            idf: Vec::new(),
            num_docs: 0,
        }
    }

    /// Access the configuration.
    pub fn config(&self) -> &TfidfConfig {
        &self.config
    }

    // ----------------------------------------------------------------
    // Public API
    // ----------------------------------------------------------------

    /// Fit IDF from corpus. Lightweight: does NOT store per-doc TF maps.
    ///
    /// - **Hashing**: O(n_features) memory (atomic DF array).
    /// - **Vocab**: O(vocab_size) memory (DF map → vocab + IDF).
    pub fn fit(&mut self, texts: &[&str]) {
        if texts.is_empty() {
            return;
        }
        if self.config.use_hashing {
            self.fit_hashing(texts);
        } else {
            self.fit_vocab(texts);
        }
    }

    /// Transform texts using pre-fitted IDF. Per-doc scoring — each rayon
    /// thread holds only one doc's TF map at a time (transient, ~2KB).
    pub fn transform(&self, texts: &[&str]) -> (Vec<Vec<String>>, Vec<Vec<f32>>) {
        if texts.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let pb = tfidf_progress(texts.len() as u64, "Scoring");
        let result = self.transform_with_pb(texts, &pb);
        pb.finish();
        result
    }

    /// Transform using an external progress bar (for chunked callers).
    fn transform_with_pb(
        &self,
        texts: &[&str],
        pb: &ProgressBar,
    ) -> (Vec<Vec<String>>, Vec<Vec<f32>>) {
        if texts.is_empty() {
            return (Vec::new(), Vec::new());
        }
        if self.config.use_hashing {
            self.transform_hashing(texts, pb)
        } else {
            self.transform_vocab(texts, pb)
        }
    }

    /// Apply dedup and/or stopword post-filtering to raw transform results.
    fn post_process(
        &self,
        words: Vec<Vec<String>>,
        scores: Vec<Vec<f32>>,
    ) -> (Vec<Vec<String>>, Vec<Vec<f32>>) {
        let n = words.len();
        let (words, scores) = if self.config.dedup {
            let pb = tfidf_progress(n as u64, "Dedup n-grams");
            let result = dedup_results(words, scores, &pb);
            pb.finish();
            result
        } else {
            (words, scores)
        };
        if let Some(ref sw) = self.post_filter_stopwords {
            let pb = tfidf_progress(n as u64, "Stopword filter");
            let result = filter_stopword_ngrams(words, scores, sw, &pb);
            pb.finish();
            result
        } else {
            (words, scores)
        }
    }

    /// Post-process using an external progress bar (for chunked callers).
    fn post_process_with_pb(
        &self,
        words: Vec<Vec<String>>,
        scores: Vec<Vec<f32>>,
        pb: &ProgressBar,
    ) -> (Vec<Vec<String>>, Vec<Vec<f32>>) {
        let (words, scores) = if self.config.dedup {
            dedup_results(words, scores, pb)
        } else {
            (words, scores)
        };
        if let Some(ref sw) = self.post_filter_stopwords {
            filter_stopword_ngrams(words, scores, sw, pb)
        } else {
            (words, scores)
        }
    }

    /// Fit vocabulary + IDF and extract top-k keywords.
    ///
    /// Returns `(top_words, top_scores)` where each is `Vec<Vec<_>>` with
    /// outer length = `n_docs`, inner length = `top_k`.
    /// Rows sorted descending by score, padded with `""`/`0.0`.
    pub fn fit_transform(&mut self, texts: &[&str]) -> (Vec<Vec<String>>, Vec<Vec<f32>>) {
        if texts.is_empty() {
            return (Vec::new(), Vec::new());
        }
        self.fit(texts);
        let (words, scores) = self.transform(texts);
        self.post_process(words, scores)
    }

    /// Fit + transform + serialize to ndjson in memory-bounded chunks.
    ///
    /// Each line: `{"_id":"<id>","keywords":["w1","w2"],"scores":[1.0,2.0]}`.
    /// Only one chunk's results live in memory at a time.
    /// Shows two progress bars: one for fit, one for the entire transform pipeline.
    pub fn fit_transform_ndjson(
        &mut self,
        texts: &[&str],
        ids: &[&str],
        batch_size: usize,
    ) -> Vec<u8> {
        if texts.is_empty() {
            return Vec::new();
        }
        self.fit(texts);

        let n_docs = texts.len();
        let chunk_size = if batch_size == 0 { n_docs } else { batch_size };

        // Single progress bar for the entire transform pipeline.
        let pb = tfidf_progress(n_docs as u64, "Transforming");
        let mut out = Vec::with_capacity(n_docs * 200);

        for (chunk_texts, chunk_ids) in texts.chunks(chunk_size).zip(ids.chunks(chunk_size)) {
            let (words, scores) = self.transform_with_pb(chunk_texts, &pb);
            let (words, scores) = self.post_process_with_pb(words, scores, &pb);
            serialize_ndjson_chunk(chunk_ids, &words, &scores, &mut out, &pb);
        }
        pb.finish();
        out
    }

    /// Number of terms in the fitted vocabulary (or n_features in hashing mode).
    pub fn vocab_size(&self) -> usize {
        if self.config.use_hashing {
            self.idf.iter().filter(|&&v| v > 0.0).count()
        } else {
            self.vocab.len()
        }
    }

    /// Number of documents seen during fit.
    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    // ----------------------------------------------------------------
    // Fit internals
    // ----------------------------------------------------------------

    /// Hashing fit: parallel tokenize → per-doc HashSet for DF → atomic DF array → IDF.
    /// Memory: O(n_features). Per-doc HashSet is transient (~2KB, dropped each iteration).
    fn fit_hashing(&mut self, texts: &[&str]) {
        let n_docs = texts.len();
        let n_features = self.config.n_features;
        let mask = (n_features - 1) as u32;
        let (min_n, max_n) = self.config.ngram_range;
        let tokenizer = &self.tokenizer;

        // Parallel tokenize → DF accumulation (no TF maps stored).
        let pb = tfidf_progress(n_docs as u64, "Fitting (DF)");
        let df: Vec<AtomicU32> = (0..n_features).map(|_| AtomicU32::new(0)).collect();
        texts.par_iter().for_each(|text| {
            let tokens = tokenizer.tokenize_owned(text);
            let mut seen = HashSet::new();
            for (slot, _start) in iter_ngram_slots(&tokens, min_n, max_n, mask) {
                if seen.insert(slot) {
                    df[slot as usize].fetch_add(1, Ordering::Relaxed);
                }
            }
            pb.inc(1);
        });
        pb.finish();

        // Compute IDF.
        let min_df = self.config.min_df;
        let n = n_docs as f32;
        let mut idf = vec![0.0f32; n_features];
        for (i, d) in df.iter().enumerate() {
            let d = d.load(Ordering::Relaxed);
            if d >= min_df {
                idf[i] = ((1.0 + n) / (1.0 + d as f32)).ln() + 1.0;
            }
        }
        self.idf = idf;
        self.num_docs = n_docs as u32;
    }

    /// Vocab fit: parallel fold/reduce to build DF map without storing all TF maps.
    /// Each rayon thread tokenizes one doc, builds a TF map, merges unique terms
    /// into a per-thread DF accumulator, then drops the TF map.
    /// Memory: O(vocab_size) for the DF map + per-thread accumulators.
    fn fit_vocab(&mut self, texts: &[&str]) {
        let n_docs = texts.len();
        let (min_n, max_n) = self.config.ngram_range;
        let tokenizer = &self.tokenizer;

        let pb = tfidf_progress(n_docs as u64, "Fitting (DF)");
        let df_map: HashMap<String, u32> = texts
            .par_iter()
            .fold(
                HashMap::new,
                |mut df: HashMap<String, u32>, text| {
                    let tokens = tokenizer.tokenize_owned(text);
                    let tf_map = ngram_tf_map(&tokens, min_n, max_n);
                    // Count each unique term once per doc.
                    for term in tf_map.into_keys() {
                        *df.entry(term).or_insert(0) += 1;
                    }
                    pb.inc(1);
                    df
                },
            )
            .reduce(HashMap::new, |mut a, b| {
                for (term, count) in b {
                    *a.entry(term).or_insert(0) += count;
                }
                a
            });
        pb.finish();

        // Filter by min_df, build vocab, compute IDF.
        let min_df = self.config.min_df;
        let n = n_docs as f32;
        let mut terms = Vec::with_capacity(df_map.len());
        let mut vocab = HashMap::with_capacity(df_map.len());
        let mut idf_values = Vec::with_capacity(df_map.len());

        for (term, df) in df_map {
            if df >= min_df {
                let term_id = terms.len() as u32;
                vocab.insert(term.clone(), term_id);
                terms.push(term);
                idf_values.push(((1.0 + n) / (1.0 + df as f32)).ln() + 1.0);
            }
        }

        self.vocab = vocab;
        self.terms = terms;
        self.idf = idf_values;
        self.num_docs = n_docs as u32;
    }

    // ----------------------------------------------------------------
    // Transform internals
    // ----------------------------------------------------------------

    /// Hashing transform: per-doc tokenize → TF → score → inline word reconstruction.
    /// No global hash→word map — each doc's top-k hashes are resolved from its own tokens.
    fn transform_hashing(&self, texts: &[&str], pb: &ProgressBar) -> (Vec<Vec<String>>, Vec<Vec<f32>>) {
        let n_features = self.config.n_features;
        let mask = (n_features - 1) as u32;
        let (min_n, max_n) = self.config.ngram_range;
        let top_k = self.config.top_k;
        let sublinear_tf = self.config.sublinear_tf;
        let tokenizer = &self.tokenizer;
        let idf_ref = &self.idf;

        let results: Vec<(Vec<String>, Vec<f32>)> = texts
            .par_iter()
            .map(|text| {
                let tokens = tokenizer.tokenize_owned(text);

                // Build TF map (transient, ~2KB per doc).
                let mut tf_map: HashMap<u32, u32> = HashMap::new();
                for (slot, _start) in iter_ngram_slots(&tokens, min_n, max_n, mask) {
                    *tf_map.entry(slot).or_insert(0) += 1;
                }

                // Score top-k.
                let (top_hashes, top_scores) =
                    extract_top_k_hashed(&tf_map, idf_ref, top_k, sublinear_tf);

                // Inline word reconstruction: find the n-gram that produced each
                // top-k hash by scanning this doc's own tokens.
                let needed: HashSet<u32> = top_hashes
                    .iter()
                    .copied()
                    .filter(|&h| h != u32::MAX)
                    .collect();

                if needed.is_empty() {
                    pb.inc(1);
                    return (vec![String::new(); top_k], top_scores);
                }

                let mut hash_to_word: HashMap<u32, String> = HashMap::with_capacity(needed.len());
                for (slot, ngram, _start) in
                    iter_ngram_slots_with_str(&tokens, min_n, max_n, mask)
                {
                    if needed.contains(&slot) && !hash_to_word.contains_key(&slot) {
                        hash_to_word.insert(slot, ngram);
                        if hash_to_word.len() == needed.len() {
                            break;
                        }
                    }
                }

                let words: Vec<String> = top_hashes
                    .iter()
                    .map(|&h| {
                        if h == u32::MAX {
                            String::new()
                        } else {
                            hash_to_word.get(&h).cloned().unwrap_or_default()
                        }
                    })
                    .collect();

                pb.inc(1);
                (words, top_scores)
            })
            .collect();

        unzip_results(results)
    }

    /// Vocab transform: per-doc tokenize → TF → score against pre-fitted vocab/IDF.
    fn transform_vocab(&self, texts: &[&str], pb: &ProgressBar) -> (Vec<Vec<String>>, Vec<Vec<f32>>) {
        let (min_n, max_n) = self.config.ngram_range;
        let top_k = self.config.top_k;
        let sublinear_tf = self.config.sublinear_tf;
        let tokenizer = &self.tokenizer;
        let vocab_ref = &self.vocab;
        let idf_ref = &self.idf;
        let terms_ref = &self.terms;

        let results: Vec<(Vec<String>, Vec<f32>)> = texts
            .par_iter()
            .map(|text| {
                let tokens = tokenizer.tokenize_owned(text);
                let tf_map = ngram_tf_map(&tokens, min_n, max_n);
                let result = extract_top_k_vocab(
                    &tf_map, vocab_ref, idf_ref, terms_ref, top_k, sublinear_tf,
                );
                pb.inc(1);
                result
            })
            .collect();

        unzip_results(results)
    }
}

// ---- Shared helpers ----
//
// N-gram TF map building and per-window hashing live in `crate::ngram` —
// see that module for the single source of truth used by `Tfidf`,
// `NGramIndex`, and the hashed-n-gram tier of `BM25`.

/// Partial-sort comparator for descending f32.
#[inline]
fn cmp_desc(a: &(u32, f32), b: &(u32, f32)) -> std::cmp::Ordering {
    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
}

/// Top-k extraction for vocabulary mode.
fn extract_top_k_vocab(
    tf_map: &HashMap<String, u32>,
    vocab: &HashMap<String, u32>,
    idf: &[f32],
    terms: &[String],
    top_k: usize,
    sublinear_tf: bool,
) -> (Vec<String>, Vec<f32>) {
    let mut scored: Vec<(u32, f32)> = Vec::with_capacity(tf_map.len());
    for (term, &tf) in tf_map {
        if let Some(&term_id) = vocab.get(term) {
            let tf_weight = if sublinear_tf {
                1.0 + (tf as f32).ln()
            } else {
                tf as f32
            };
            scored.push((term_id, tf_weight * idf[term_id as usize]));
        }
    }

    let (top_ids, top_scores) = select_top_k(&mut scored, top_k);

    let words: Vec<String> = top_ids
        .iter()
        .map(|&id| {
            if id == u32::MAX {
                String::new()
            } else {
                terms[id as usize].clone()
            }
        })
        .collect();
    (words, top_scores)
}

/// Top-k extraction for hashing mode (returns hash indices, not words).
fn extract_top_k_hashed(
    tf_map: &HashMap<u32, u32>,
    idf: &[f32],
    top_k: usize,
    sublinear_tf: bool,
) -> (Vec<u32>, Vec<f32>) {
    let mut scored: Vec<(u32, f32)> = Vec::with_capacity(tf_map.len());
    for (&h, &tf) in tf_map {
        let idf_val = idf[h as usize];
        if idf_val > 0.0 {
            let tf_weight = if sublinear_tf {
                1.0 + (tf as f32).ln()
            } else {
                tf as f32
            };
            scored.push((h, tf_weight * idf_val));
        }
    }
    select_top_k(&mut scored, top_k)
}

/// Generic partial-sort top-k: select k largest by score, return (ids, scores) padded to k.
fn select_top_k(scored: &mut Vec<(u32, f32)>, top_k: usize) -> (Vec<u32>, Vec<f32>) {
    let actual_k = top_k.min(scored.len());
    if actual_k == 0 {
        return (vec![u32::MAX; top_k], vec![0.0f32; top_k]);
    }

    if actual_k < scored.len() {
        scored.select_nth_unstable_by(actual_k - 1, cmp_desc);
        scored.truncate(actual_k);
    }
    scored.sort_unstable_by(cmp_desc);

    let mut ids = Vec::with_capacity(top_k);
    let mut scores = Vec::with_capacity(top_k);
    for &(id, score) in scored.iter() {
        ids.push(id);
        scores.push(score);
    }
    ids.resize(top_k, u32::MAX);
    scores.resize(top_k, 0.0);
    (ids, scores)
}

/// Unzip a Vec of (words, scores) pairs into two separate Vecs.
fn unzip_results(results: Vec<(Vec<String>, Vec<f32>)>) -> (Vec<Vec<String>>, Vec<Vec<f32>>) {
    let n = results.len();
    let mut all_words = Vec::with_capacity(n);
    let mut all_scores = Vec::with_capacity(n);
    for (words, scores) in results {
        all_words.push(words);
        all_scores.push(scores);
    }
    (all_words, all_scores)
}

/// Serialize TF-IDF results to ndjson bytes in parallel.
///
/// Each line: `{"_id":"<id>","keywords":["w1","w2"],"scores":[1.0,2.0]}`
/// Returns the complete ndjson as a single `Vec<u8>`.
pub fn results_to_ndjson(
    ids: &[&str],
    words: &[Vec<String>],
    scores: &[Vec<f32>],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(ids.len() * 200);
    let pb = tfidf_progress(ids.len() as u64, "Serializing");
    serialize_ndjson_chunk(ids, words, scores, &mut out, &pb);
    pb.finish();
    out
}

/// Serialize a chunk of TF-IDF results to ndjson, appending to `out`.
fn serialize_ndjson_chunk(
    ids: &[&str],
    words: &[Vec<String>],
    scores: &[Vec<f32>],
    out: &mut Vec<u8>,
    pb: &ProgressBar,
) {
    let n = ids.len();

    // Serialize each row in parallel, then join with newlines.
    let lines: Vec<Vec<u8>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut buf = Vec::with_capacity(256);
            buf.extend_from_slice(b"{\"_id\":\"");
            json_escape_into(ids[i], &mut buf);
            buf.extend_from_slice(b"\",\"keywords\":[");
            for (j, w) in words[i].iter().enumerate() {
                if j > 0 {
                    buf.push(b',');
                }
                buf.push(b'"');
                json_escape_into(w, &mut buf);
                buf.push(b'"');
            }
            buf.extend_from_slice(b"],\"scores\":[");
            for (j, &s) in scores[i].iter().enumerate() {
                if j > 0 {
                    buf.push(b',');
                }
                let mut fbuf = ryu::Buffer::new();
                buf.extend_from_slice(fbuf.format(s).as_bytes());
            }
            buf.extend_from_slice(b"]}");
            pb.inc(1);
            buf
        })
        .collect();

    for line in &lines {
        out.extend_from_slice(line);
        out.push(b'\n');
    }
}

/// Escape quotes and backslashes for JSON string values.
#[inline]
fn json_escape_into(s: &str, buf: &mut Vec<u8>) {
    for b in s.bytes() {
        match b {
            b'"' => buf.extend_from_slice(b"\\\""),
            b'\\' => buf.extend_from_slice(b"\\\\"),
            b'\n' => buf.extend_from_slice(b"\\n"),
            b'\r' => buf.extend_from_slice(b"\\r"),
            b'\t' => buf.extend_from_slice(b"\\t"),
            0x00..=0x1f => {
                buf.extend_from_slice(b"\\u00");
                let hi = b >> 4;
                let lo = b & 0x0f;
                buf.push(if hi < 10 { b'0' + hi } else { b'a' + hi - 10 });
                buf.push(if lo < 10 { b'0' + lo } else { b'a' + lo - 10 });
            }
            _ => buf.push(b),
        }
    }
}

/// Remove n-grams that are word sub-phrases of a higher-scored kept keyword.
///
/// Assumes input is sorted by score descending. Builds a set of word sub-phrases
/// from each kept keyword; subsequent keywords found in that set are dropped.
/// Also deduplicates exact repeats. Returns variable-length (no padding).
fn dedup_ngrams(words: Vec<String>, scores: Vec<f32>) -> (Vec<String>, Vec<f32>) {
    let mut kept_w = Vec::new();
    let mut kept_s = Vec::new();
    let mut subphrases: HashSet<String> = HashSet::new();

    for (w, s) in words.into_iter().zip(scores) {
        if w.is_empty() {
            continue;
        }
        if subphrases.contains(&w) {
            continue;
        }
        kept_w.push(w.clone());
        kept_s.push(s);
        // Register this keyword and all contiguous word sub-phrases.
        subphrases.insert(w.clone());
        let tokens: Vec<&str> = w.split(' ').collect();
        let n = tokens.len();
        for len in 1..n {
            for start in 0..=n - len {
                let sub: String = tokens[start..start + len].join(" ");
                subphrases.insert(sub);
            }
        }
    }

    (kept_w, kept_s)
}

/// Apply dedup_ngrams to each row in parallel via Rayon.
fn dedup_results(
    all_words: Vec<Vec<String>>,
    all_scores: Vec<Vec<f32>>,
    pb: &ProgressBar,
) -> (Vec<Vec<String>>, Vec<Vec<f32>>) {
    let results: Vec<(Vec<String>, Vec<f32>)> = all_words
        .into_par_iter()
        .zip(all_scores.into_par_iter())
        .map(|(w, s)| {
            let result = dedup_ngrams(w, s);
            pb.inc(1);
            result
        })
        .collect();
    unzip_results(results)
}

/// Remove noisy stopword n-grams from results (parallel).
///
/// Filters out:
/// - Unigrams that are stopwords ("the", "of")
/// - N-grams that start or end with a stopword ("macrophages the", "of replication")
/// - N-grams where all tokens are pure digits ("14", "0 71")
///
/// Keeps n-grams with stopwords only in the middle ("king of the world").
fn filter_stopword_ngrams(
    all_words: Vec<Vec<String>>,
    all_scores: Vec<Vec<f32>>,
    stopwords: &HashSet<String>,
    pb: &ProgressBar,
) -> (Vec<Vec<String>>, Vec<Vec<f32>>) {
    let results: Vec<(Vec<String>, Vec<f32>)> = all_words
        .into_par_iter()
        .zip(all_scores.into_par_iter())
        .map(|(words, scores)| {
            let (w, s): (Vec<String>, Vec<f32>) = words
                .into_iter()
                .zip(scores)
                .filter(|(w, _)| {
                    if w.is_empty() {
                        return true;
                    }
                    // Reject if all tokens are pure digits ("14", "0 71").
                    if w.split(' ').all(|t| t.bytes().all(|b| b.is_ascii_digit())) {
                        return false;
                    }
                    if !w.contains(' ') {
                        // Unigram: keep only if not a stopword.
                        return !stopwords.contains(w.as_str());
                    }
                    // N-gram: reject if first or last token is a stopword.
                    let first = w.split(' ').next().unwrap();
                    let last = w.rsplit(' ').next().unwrap();
                    !stopwords.contains(first) && !stopwords.contains(last)
                })
                .unzip();
            pb.inc(1);
            (w, s)
        })
        .collect();
    unzip_results(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(top_k: usize, ngram_range: (usize, usize), sublinear_tf: bool, min_df: u32) -> TfidfConfig {
        TfidfConfig {
            top_k,
            ngram_range,
            sublinear_tf,
            min_df,
            use_hashing: false,
            n_features: 1 << 23,
            dedup: false,
        }
    }

    fn make_hashing_config(top_k: usize, ngram_range: (usize, usize), sublinear_tf: bool, min_df: u32, n_features: usize) -> TfidfConfig {
        TfidfConfig {
            top_k,
            ngram_range,
            sublinear_tf,
            min_df,
            use_hashing: true,
            n_features,
            dedup: false,
        }
    }

    #[test]
    fn test_basic_unigrams() {
        let mut tfidf = Tfidf::new(
            make_config(3, (1, 1), false, 1),
            TokenizerMode::Plain,
            true,
        );

        let texts = [
            "the cat sat on the mat",
            "the dog chased the cat",
            "the bird flew over the mat",
        ];
        let (words, scores) = tfidf.fit_transform(&texts);

        assert_eq!(words.len(), 3);
        assert_eq!(words[0].len(), 3);
        assert_scores_valid(&scores);
    }

    #[test]
    fn test_ngrams_produce_bigrams() {
        let mut tfidf = Tfidf::new(
            make_config(10, (1, 2), true, 1),
            TokenizerMode::Plain,
            false,
        );
        let texts = ["hello world foo", "hello bar baz"];
        let (words, _) = tfidf.fit_transform(&texts);

        assert!(
            words.iter().flat_map(|w| w.iter()).any(|w| w.contains(' ')),
            "ngram_range (1,2) should produce bigrams"
        );
    }

    #[test]
    fn test_min_df_filtering() {
        let mut tfidf = Tfidf::new(
            make_config(5, (1, 1), false, 2),
            TokenizerMode::Plain,
            false,
        );
        let texts = [
            "common word here",
            "common word there",
            "unique rare term",
        ];
        let (words, _) = tfidf.fit_transform(&texts);

        let doc2_non_empty: Vec<&str> = words[2]
            .iter()
            .map(|s| s.as_str())
            .filter(|s| !s.is_empty())
            .collect();
        assert!(doc2_non_empty.is_empty(), "doc 2 should have no keywords, got {:?}", doc2_non_empty);
    }

    #[test]
    fn test_empty_input() {
        let mut tfidf = Tfidf::new(TfidfConfig::default(), TokenizerMode::Plain, true);
        let (words, scores) = tfidf.fit_transform(&[]);
        assert!(words.is_empty());
        assert!(scores.is_empty());
    }

    #[test]
    fn test_stemming_consolidates_terms() {
        let mut tfidf = Tfidf::new(
            make_config(5, (1, 1), false, 1),
            TokenizerMode::UnicodeStem,
            true,
        );
        let texts = ["running runs runner"];
        let (words, scores) = tfidf.fit_transform(&texts);

        assert_eq!(words.len(), 1);
        assert!(scores[0][0] > 0.0);
    }

    #[test]
    fn test_sublinear_tf() {
        let texts = ["word word word word word unique"];

        let mut linear = Tfidf::new(make_config(2, (1, 1), false, 1), TokenizerMode::Plain, false);
        let (_, scores_linear) = linear.fit_transform(&texts);

        let mut sublinear = Tfidf::new(make_config(2, (1, 1), true, 1), TokenizerMode::Plain, false);
        let (_, scores_sub) = sublinear.fit_transform(&texts);

        assert!(scores_linear[0][0] > scores_sub[0][0]);
    }

    #[test]
    fn test_idf_weights_rare_terms_higher() {
        let mut tfidf = Tfidf::new(
            make_config(5, (1, 1), false, 1),
            TokenizerMode::Plain,
            false,
        );
        let texts = ["rare common", "frequent common", "another common"];
        let (words, scores) = tfidf.fit_transform(&texts);

        assert_eq!(words[0][0], "rare");
        assert!(scores[0][0] > scores[0][1]);
    }

    #[test]
    fn test_padding() {
        let mut tfidf = Tfidf::new(
            make_config(10, (1, 1), false, 1),
            TokenizerMode::Plain,
            true,
        );
        let texts = ["the cat"];
        let (words, scores) = tfidf.fit_transform(&texts);

        assert_eq!(words[0].len(), 10);
        assert!(!words[0][0].is_empty());
        assert!(words[0][1].is_empty());
        assert_eq!(scores[0][1], 0.0);
    }

    // ---- Hashing mode tests ----

    #[test]
    fn test_hashing_basic() {
        let mut tfidf = Tfidf::new(
            make_hashing_config(3, (1, 1), false, 1, 1 << 16),
            TokenizerMode::Plain,
            true,
        );
        let texts = [
            "the cat sat on the mat",
            "the dog chased the cat",
            "the bird flew over the mat",
        ];
        let (words, scores) = tfidf.fit_transform(&texts);

        assert_eq!(words.len(), 3);
        assert_eq!(words[0].len(), 3);
        assert_scores_valid(&scores);

        // Words should be non-empty (reconstructed successfully)
        for doc_words in &words {
            assert!(!doc_words[0].is_empty(), "top word should be reconstructed");
        }
    }

    #[test]
    fn test_hashing_ngrams() {
        let mut tfidf = Tfidf::new(
            make_hashing_config(10, (1, 3), true, 1, 1 << 16),
            TokenizerMode::Plain,
            false,
        );
        let texts = ["hello world foo bar", "hello baz qux"];
        let (words, _) = tfidf.fit_transform(&texts);

        let has_bigram = words.iter().flat_map(|w| w.iter()).any(|w| w.contains(' '));
        assert!(has_bigram, "hashing mode with ngram_range (1,3) should produce bigrams/trigrams");
    }

    #[test]
    fn test_hashing_matches_vocab_ranking() {
        // Both modes should rank the same unique term highest
        let texts = ["rare common", "frequent common", "another common"];

        let mut vocab = Tfidf::new(make_config(3, (1, 1), false, 1), TokenizerMode::Plain, false);
        let (vocab_words, _) = vocab.fit_transform(&texts);

        let mut hashing = Tfidf::new(
            make_hashing_config(3, (1, 1), false, 1, 1 << 16),
            TokenizerMode::Plain,
            false,
        );
        let (hash_words, _) = hashing.fit_transform(&texts);

        // Doc 0: "rare" should be #1 in both modes
        assert_eq!(vocab_words[0][0], "rare");
        assert_eq!(hash_words[0][0], "rare");
    }

    #[test]
    fn test_hashing_min_df() {
        let mut tfidf = Tfidf::new(
            make_hashing_config(5, (1, 1), false, 2, 1 << 16),
            TokenizerMode::Plain,
            false,
        );
        let texts = [
            "common word here",
            "common word there",
            "unique rare term",
        ];
        let (words, _) = tfidf.fit_transform(&texts);

        let doc2_non_empty: Vec<&str> = words[2]
            .iter()
            .map(|s| s.as_str())
            .filter(|s| !s.is_empty())
            .collect();
        assert!(doc2_non_empty.is_empty(), "hashing: doc 2 should have no keywords, got {:?}", doc2_non_empty);
    }

    /// Assert all score rows are non-negative and sorted descending.
    fn assert_scores_valid(scores: &[Vec<f32>]) {
        for doc_scores in scores {
            for s in doc_scores {
                assert!(*s >= 0.0);
            }
            for i in 1..doc_scores.len() {
                assert!(doc_scores[i - 1] >= doc_scores[i], "scores should be descending");
            }
        }
    }

    // ---- Dedup tests ----

    #[test]
    fn test_dedup_removes_subphrases() {
        let words = vec![
            "cerebral white matter".to_string(),
            "cerebral white".to_string(),
            "white matter".to_string(),
            "brain".to_string(),
        ];
        let scores = vec![23.1, 20.7, 20.0, 18.5];
        let (kept_w, kept_s) = dedup_ngrams(words, scores);
        assert_eq!(kept_w, vec!["cerebral white matter", "brain"]);
        assert_eq!(kept_s, vec![23.1, 18.5]);
    }

    #[test]
    fn test_dedup_unigrams_only_noop() {
        let words = vec!["fox".to_string(), "dog".to_string(), "cat".to_string()];
        let scores = vec![10.0, 8.0, 6.0];
        let (kept_w, _) = dedup_ngrams(words, scores);
        assert_eq!(kept_w, vec!["fox", "dog", "cat"]);
    }

    #[test]
    fn test_dedup_exact_duplicates() {
        let words = vec!["fox".to_string(), "fox".to_string(), "dog".to_string()];
        let scores = vec![10.0, 10.0, 8.0];
        let (kept_w, _) = dedup_ngrams(words, scores);
        assert_eq!(kept_w, vec!["fox", "dog"]);
    }

    #[test]
    fn test_dedup_skips_padding() {
        let words = vec!["fox".to_string(), "".to_string(), "".to_string()];
        let scores = vec![10.0, 0.0, 0.0];
        let (kept_w, kept_s) = dedup_ngrams(words, scores);
        assert_eq!(kept_w, vec!["fox"]);
        assert_eq!(kept_s, vec![10.0]);
    }

    #[test]
    fn test_dedup_different_word_order_kept() {
        let words = vec!["white matter".to_string(), "matter white".to_string()];
        let scores = vec![10.0, 8.0];
        let (kept_w, _) = dedup_ngrams(words, scores);
        assert_eq!(kept_w, vec!["white matter", "matter white"]);
    }

    #[test]
    fn test_dedup_integrated_vocab() {
        let mut tfidf = Tfidf::new(
            TfidfConfig { dedup: true, ..make_config(10, (1, 3), true, 1) },
            TokenizerMode::Plain,
            false,
        );
        let texts = ["the quick brown fox jumps over the lazy dog"];
        let (words, scores) = tfidf.fit_transform(&texts);

        // With dedup, no keyword should be a sub-phrase of an earlier
        // (higher-scored) kept keyword.
        for (doc_words, doc_scores) in words.iter().zip(scores.iter()) {
            for (i, w) in doc_words.iter().enumerate() {
                for j in 0..i {
                    let other = &doc_words[j];
                    // w should NOT be a contiguous sub-phrase of other (which scored higher)
                    let w_tokens: Vec<&str> = w.split(' ').collect();
                    let other_tokens: Vec<&str> = other.split(' ').collect();
                    if other_tokens.len() > w_tokens.len() {
                        let is_subphrase = other_tokens.windows(w_tokens.len())
                            .any(|win| win == w_tokens.as_slice());
                        assert!(!is_subphrase,
                            "{:?} (score={}) is a sub-phrase of {:?} (score={})",
                            w, doc_scores[i], other, doc_scores[j]);
                    }
                }
            }
        }
    }
}
