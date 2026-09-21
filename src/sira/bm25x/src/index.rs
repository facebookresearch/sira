// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::scoring::{self, Method, ScoringParams};
use crate::storage::MmapData;
use crate::tokenizer::{Tokenizer, TokenizerMode};
use indicatif::{ProgressBar, ProgressStyle};

mod ngram_side;
pub(crate) use ngram_side::NgramSide;

/// Capped rayon pool (max 12 threads) — reused across all calls.
pub(crate) fn capped_pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::thread::available_parallelism()
            .map(|p| p.get().min(12))
            .unwrap_or(4);
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("failed to build rayon pool")
    })
}

/// Uncapped rayon pool (all available cores) — reused across all calls.
fn uncapped_pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(12);
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("failed to build rayon pool")
    })
}

thread_local! {
    static SCORE_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Run `f` with a thread-local score buffer of at least `n` elements.
/// The buffer is guaranteed to be zeroed on entry (callers must zero touched
/// entries before returning from `f`).
pub(crate) fn with_score_buf<R>(n: usize, f: impl FnOnce(&mut Vec<f32>) -> R) -> R {
    SCORE_BUF.with_borrow_mut(|buf| {
        if buf.len() < n {
            buf.resize(n, 0.0);
        }
        f(buf)
    })
}

/// A scored document result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The document index (0-based, contiguous).
    pub index: usize,
    /// The BM25 score.
    pub score: f32,
}

/// Wrapper for BinaryHeap min-heap (we want to keep top-k highest scores).
struct MinScored(f32, u32);

impl PartialEq for MinScored {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for MinScored {}
impl PartialOrd for MinScored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MinScored {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
    }
}

/// The core BM25 index.
///
/// Document indices are contiguous 0..n. Deleting a document compacts the index:
/// all documents after the deleted one shift down by one.
///
/// ## Multi-gram (`max_n >= 2`)
///
/// When `max_n >= 2`, the index also maintains a hashed n-gram tier
/// ([`NgramSide`]). Unigrams continue to live in the exact `vocab`; n-grams
/// (n in `2..=max_n`) hash to one of `n_features` slots via MurmurHash3.
/// Both tiers contribute additively to BM25 scoring.
///
/// Two invariants:
/// - `doc_lengths` and `total_tokens` count **unigrams only** so BM25's
///   `b·dl/avgdl` length normalization stays meaningful.
/// - N-gram positions encode the **start** unigram index (e.g. bigram
///   "machin learn" at unigram positions [3, 4] stores 3). Phrase queries
///   continue to use unigram positions only.
pub struct BM25 {
    // Scoring parameters
    pub k1: f32,
    pub b: f32,
    pub delta: f32,
    pub method: Method,

    // Inverted index: term_id -> Vec<(doc_id, tf)> sorted by doc_id
    postings: Vec<Vec<(u32, u32)>>,

    // Term positions: positions[term_id][posting_idx] = sorted positions in doc.
    // Parallel to `postings` — positions[t][i] corresponds to postings[t][i].
    positions: Vec<Vec<Vec<u32>>>,

    // Cached document frequency per term
    pub doc_freqs: Vec<u32>,

    // Document metadata
    doc_lengths: Vec<u32>,
    pub(crate) total_tokens: u64,
    pub(crate) num_docs: u32,

    // Vocabulary: token string -> term_id
    vocab: FxHashMap<String, u32>,

    // Tokenizer
    tokenizer: Tokenizer,

    // Mmap backing (if loaded from disk)
    mmap_data: Option<MmapData>,

    // Auto-save path (if set, mutations persist to disk automatically)
    index_path: Option<PathBuf>,

    // If true, CUDA is required — operations fail instead of falling back to CPU.
    cuda_required: bool,

    // Multi-gram support: n=1 lives in the unigram tier above; n>=2 values
    // live in `ngram_side` (None when no n>=2 selected). `ngram_set` is the
    // canonical record of which n's contribute to BM25 scoring — sorted,
    // deduped, all in 1..=8. `max_n` and `n_features` are derived caches kept
    // in sync for storage v2 compatibility and the GPU upload layout.
    //
    // Two independent dimensions:
    // - WHICH n's get *indexed* (always n=1 in unigram tier; only `ns(side)`
    //   in the n-gram tier).
    // - WHICH n's contribute to BM25 *score*: `score_unigram = ngram_set.contains(&1)`
    //   gates the unigram score loop; the n-gram side only stores selected n's
    //   so its loop is intrinsically filtered.
    //
    // The unigram tier is ALWAYS populated even when `1 ∉ ngram_set` — many
    // query-time facilities depend on it (term DF lookups, prefix scans,
    // search_expr term resolution, position-based phrase queries). The score
    // gate is the right place to express "score with bigram-only BM25".
    pub(crate) ngram_set: Vec<u8>,
    pub(crate) score_unigram: bool,
    pub(crate) max_n: u8,
    pub(crate) n_features: u32,
    pub(crate) ngram_side: Option<NgramSide>,
}

impl BM25 {
    /// Require CUDA: if set, operations will fail instead of falling back to CPU.
    /// Call after construction: `BM25::default().require_cuda()`
    pub fn require_cuda(mut self) -> Self {
        self.cuda_required = true;
        self
    }

    /// Returns true if this index requires CUDA.
    pub fn is_cuda_required(&self) -> bool {
        self.cuda_required
    }

    /// Toggle the CUDA-required flag in place. Used by Python bindings that
    /// load an index via [`BM25::load`] (which doesn't take a `cuda` flag) and
    /// then need to mark it as GPU-required without rebuilding the struct.
    #[inline]
    pub fn set_cuda_required(&mut self, required: bool) {
        self.cuda_required = required;
    }

    /// Set the auto-save path so subsequent mutations persist to disk. Used by
    /// the Python `BM25(index=...)` constructor when it loads an existing index
    /// via [`BM25::load`] and wants to enable auto-persistence afterwards.
    #[inline]
    pub fn set_index_path<P: AsRef<Path>>(&mut self, path: P) {
        self.index_path = Some(path.as_ref().to_path_buf());
    }

    /// Maximum n-gram order (1 = unigram-only). Derived from [`Self::ngram_set`].
    #[inline]
    pub fn max_n(&self) -> u8 { self.max_n }

    /// Sorted, deduped set of n values that contribute to BM25 scoring.
    /// Each value is in `1..=8`. E.g. `[1, 3]` = unigrams + trigrams only.
    #[inline]
    pub fn ngram_set(&self) -> &[u8] { &self.ngram_set }

    /// Whether the unigram tier contributes to the BM25 score.
    /// (The unigram tier is always BUILT for query infra, even when
    /// `false` — see `BM25` struct docs for rationale.)
    #[inline]
    pub fn score_unigram(&self) -> bool { self.score_unigram }

    /// Hashed n-gram slot count (0 when max_n == 1).
    #[inline]
    pub fn n_features(&self) -> u32 { self.n_features }

    /// The tokenizer mode this index was built with. Used by `save` to persist
    /// the tokenizer config in storage v2 (root-cause fix for the legacy
    /// `BM25::load` hardcode that silently broke non-`UnicodeStem` indices).
    #[inline]
    pub fn tokenizer_mode(&self) -> TokenizerMode { self.tokenizer.mode() }

    /// Whether stopwords are filtered. Persisted alongside `tokenizer_mode()`.
    #[inline]
    pub fn use_stopwords(&self) -> bool { self.tokenizer.use_stopwords() }

    /// Read-only access to the hashed n-gram tier (None when max_n == 1).
    /// `pub(crate)` because `NgramSide` is itself `pub(crate)`; Python callers
    /// instead use [`BM25::ngram_df`] which returns the slot DF without leaking
    /// the `NgramSide` type through PyO3.
    #[inline]
    pub(crate) fn ngram_side(&self) -> Option<&NgramSide> { self.ngram_side.as_ref() }

    /// Document frequency for an n-gram in the hashed n-gram tier (slot DF
    /// — over-estimates due to hash collisions). Returns 0 when `max_n == 1`
    /// or the slot is empty. Convenience accessor for Python callers; equivalent
    /// to `self.ngram_side().map_or(0, |s| s.df(s.slot_for(ngram)))`.
    pub fn ngram_df(&self, ngram: &str) -> u32 {
        self.ngram_side
            .as_ref()
            .map_or(0, |s| s.df(s.slot_for(ngram)))
    }

    /// Raw slot index for an n-gram string in the hashed n-gram tier.
    /// Returns 0 (valid but usually empty) when `ngram_side` is None.
    pub fn ngram_slot(&self, ngram: &str) -> u32 {
        self.ngram_side
            .as_ref()
            .map_or(0, |s| s.slot_for(ngram))
    }

    /// Create a new empty index.
    pub fn new(method: Method, k1: f32, b: f32, delta: f32, use_stopwords: bool) -> Self {
        Self::with_options(
            method,
            k1,
            b,
            delta,
            TokenizerMode::UnicodeStem,
            use_stopwords,
            false,
        )
    }

    /// Create a new empty index with a specific tokenizer mode.
    pub fn with_tokenizer(
        method: Method,
        k1: f32,
        b: f32,
        delta: f32,
        tokenizer_mode: TokenizerMode,
        use_stopwords: bool,
    ) -> Self {
        Self::with_options(method, k1, b, delta, tokenizer_mode, use_stopwords, false)
    }

    /// Create a new empty index with all options including CUDA.
    ///
    /// If `cuda` is true, operations will return errors instead of falling back to CPU.
    pub fn with_options(
        method: Method,
        k1: f32,
        b: f32,
        delta: f32,
        tokenizer_mode: TokenizerMode,
        use_stopwords: bool,
        cuda: bool,
    ) -> Self {
        Self::with_options_full(
            method,
            k1,
            b,
            delta,
            tokenizer_mode,
            use_stopwords,
            cuda,
            /* max_n */ 1,
            /* n_features */ 0,
        )
    }

    /// Create a new empty index with all options including CUDA, max_n, and n_features.
    ///
    /// `max_n = 1` allocates no n-gram tier (unigram-only, identical to legacy
    /// behavior). `max_n >= 2` requires `n_features` to be a power of 2 (the
    /// hashed-slot space; default 1<<23 = 8M slots upfront ~200MB).
    ///
    /// Builds a *contiguous* `1..=max_n` n-gram set. For arbitrary subsets
    /// (e.g. unigrams + 4-grams only), use [`Self::with_options_ngrams`].
    ///
    /// If `cuda` is true, operations will return errors instead of falling back to CPU.
    #[allow(clippy::too_many_arguments)]
    pub fn with_options_full(
        method: Method,
        k1: f32,
        b: f32,
        delta: f32,
        tokenizer_mode: TokenizerMode,
        use_stopwords: bool,
        cuda: bool,
        max_n: u8,
        n_features: u32,
    ) -> Self {
        assert!(max_n >= 1, "max_n must be >= 1");
        let ngrams: Vec<u8> = (1..=max_n).collect();
        Self::with_options_ngrams(
            method, k1, b, delta, tokenizer_mode, use_stopwords, cuda,
            ngrams, n_features,
        )
    }

    /// Create a new empty index with an arbitrary subset of n-gram orders.
    ///
    /// `ngrams` is the set of n values that contribute to BM25 scoring (each
    /// in `1..=8`). The unigram tier is always built regardless — query
    /// facilities (term DF, search_expr, prefix scan) depend on it. The score
    /// path filters by `ngrams`:
    /// - `1 ∈ ngrams` ⇒ unigram contribution included.
    /// - any `n >= 2 ∈ ngrams` ⇒ n-gram side allocated, only the listed n's
    ///   are emitted into it.
    ///
    /// Examples:
    /// - `ngrams = vec![1]` — pure unigram BM25 (legacy `max_n = 1` behavior).
    /// - `ngrams = vec![1, 2, 3, 4]` — unigrams through 4-grams (default).
    /// - `ngrams = vec![1, 4]` — unigrams + 4-grams, skip bigrams and trigrams.
    /// - `ngrams = vec![2]` — bigram-only score (unigram tier built but not scored).
    ///
    /// `n_features` must be a power of 2 if any `n >= 2` is in the set.
    #[allow(clippy::too_many_arguments)]
    pub fn with_options_ngrams(
        method: Method,
        k1: f32,
        b: f32,
        delta: f32,
        tokenizer_mode: TokenizerMode,
        use_stopwords: bool,
        cuda: bool,
        ngrams: Vec<u8>,
        n_features: u32,
    ) -> Self {
        let mut ngram_set = ngrams;
        ngram_set.sort_unstable();
        ngram_set.dedup();
        assert!(!ngram_set.is_empty(), "ngrams must be non-empty");
        assert!(
            ngram_set.iter().all(|&n| (1..=8).contains(&n)),
            "ngram values must each be in 1..=8 (got {:?})",
            ngram_set
        );

        let score_unigram = ngram_set.contains(&1);
        let side_ns: Vec<u8> = ngram_set.iter().copied().filter(|&n| n >= 2).collect();
        let max_n = *ngram_set.last().unwrap();
        let ngram_side = if !side_ns.is_empty() {
            Some(NgramSide::with_ns(side_ns, n_features))
        } else {
            None
        };
        let stored_n_features = if ngram_side.is_some() { n_features } else { 0 };

        BM25 {
            k1,
            b,
            delta,
            method,
            postings: Vec::new(),
            positions: Vec::new(),
            doc_freqs: Vec::new(),
            doc_lengths: Vec::new(),
            total_tokens: 0,
            num_docs: 0,
            vocab: FxHashMap::default(),
            tokenizer: Tokenizer::with_mode(tokenizer_mode, use_stopwords),
            mmap_data: None,
            index_path: None,
            cuda_required: cuda,
            ngram_set,
            score_unigram,
            max_n,
            n_features: stored_n_features,
            ngram_side,
        }
    }

    /// Open a persistent index at the given path.
    ///
    /// - If the directory already contains a saved index, it is loaded (mmap).
    /// - Every mutation (`add`, `delete`, `update`) auto-saves to disk.
    /// - If the directory doesn't exist yet, a new empty index is created.
    /// - If `cuda` is true, operations will return errors instead of falling back to CPU.
    pub fn open<P: AsRef<Path>>(
        path: P,
        method: Method,
        k1: f32,
        b: f32,
        delta: f32,
        tokenizer_mode: TokenizerMode,
        use_stopwords: bool,
    ) -> io::Result<Self> {
        Self::open_with_cuda(
            path,
            method,
            k1,
            b,
            delta,
            tokenizer_mode,
            use_stopwords,
            false,
        )
    }

    /// Open a persistent index with explicit CUDA control.
    #[allow(clippy::too_many_arguments)]
    pub fn open_with_cuda<P: AsRef<Path>>(
        path: P,
        method: Method,
        k1: f32,
        b: f32,
        delta: f32,
        tokenizer_mode: TokenizerMode,
        use_stopwords: bool,
        cuda: bool,
    ) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if path.join("header.bin").exists() {
            let mut index = Self::load(&path, true)?;
            index.index_path = Some(path);
            index.cuda_required = cuda;
            Ok(index)
        } else {
            let mut index =
                Self::with_options(method, k1, b, delta, tokenizer_mode, use_stopwords, cuda);
            index.index_path = Some(path);
            Ok(index)
        }
    }

    /// Auto-save to disk if an index path is configured.
    fn auto_save(&self) -> io::Result<()> {
        if let Some(ref path) = self.index_path {
            self.save(path)?;
        }
        Ok(())
    }

    /// Disable automatic saving after mutations (add, delete, update, enrich).
    ///
    /// After calling this, mutations only affect the in-memory index.
    /// Use `save()` for explicit persistence.
    pub fn disable_auto_save(&mut self) {
        self.index_path = None;
    }

    /// Add documents to the index. Returns the assigned document indices.
    ///
    /// Tokenization runs in parallel (rayon), then posting lists are merged
    /// sequentially. Progress bars are shown for both phases.
    pub fn add(&mut self, documents: &[&str]) -> io::Result<Vec<usize>> {
        if self.mmap_data.is_some() {
            self.materialize_mmap();
        }

        self.add_cpu(documents)
    }

    /// CPU-only add implementation (original algorithm).
    fn add_cpu(&mut self, documents: &[&str]) -> io::Result<Vec<usize>> {
        let t_total = std::time::Instant::now();

        // Phase 1: tokenize + compute TF maps with positions in parallel
        let t0 = std::time::Instant::now();
        let pool = capped_pool();
        let pb = ProgressBar::new(documents.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40} {pos}/{len} tokenize ({per_sec})")
                .unwrap(),
        );
        let max_n = self.max_n;  // capture by value for the closure
        let tokenized: Vec<(u32, HashMap<String, (u32, Vec<u32>)>, Vec<String>)> = pool.install(|| {
            documents
                .par_iter()
                .map(|doc| {
                    let tokens = self.tokenizer.tokenize_owned(doc);
                    let doc_len = tokens.len() as u32;

                    if max_n == 1 {
                        // Legacy fast path: move tokens into the TF map (no clone tax).
                        // `Vec::new()` placeholder for the n-gram side is never read because
                        // the merge loop guards on `self.ngram_side.as_mut()`, which is None
                        // when max_n == 1. See the merge-phase comment below.
                        let mut tf_pos_map: HashMap<String, (u32, Vec<u32>)> = HashMap::new();
                        for (pos, token) in tokens.into_iter().enumerate() {
                            let entry = tf_pos_map.entry(token).or_insert((0, Vec::new()));
                            entry.0 += 1;
                            entry.1.push(pos as u32);
                        }
                        pb.inc(1);
                        (doc_len, tf_pos_map, Vec::new())
                    } else {
                        // Multigram path requires `tokens` to survive the merge phase
                        // (NgramSide::add_doc re-walks them). Cost: one heap alloc per
                        // token (stems are NOT interned). The max_n == 1 branch above
                        // takes the legacy move-iter path to avoid this tax.
                        let mut tf_pos_map: HashMap<String, (u32, Vec<u32>)> = HashMap::new();
                        for (pos, token) in tokens.iter().enumerate() {
                            let entry = tf_pos_map.entry(token.clone()).or_insert((0, Vec::new()));
                            entry.0 += 1;
                            entry.1.push(pos as u32);
                        }
                        pb.inc(1);
                        (doc_len, tf_pos_map, tokens)
                    }
                })
                .collect()
        });
        pb.finish_and_clear();
        let t_tokenize = t0.elapsed();

        // Phase 2: merge into index sequentially (cheap — just HashMap lookups + Vec pushes)
        let t0 = std::time::Instant::now();
        let base_id = self.num_docs;
        let mut ids = Vec::with_capacity(documents.len());
        let pb = ProgressBar::new(tokenized.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40} {pos}/{len} merge ({per_sec})")
                .unwrap(),
        );

        for (i, (doc_len, tf_pos_map, tokens)) in tokenized.into_iter().enumerate() {
            let doc_id = base_id + i as u32;

            for (token, (tf, pos_list)) in tf_pos_map {
                let term_id = self.get_or_create_term(&token);
                self.postings[term_id as usize].push((doc_id, tf));
                self.positions[term_id as usize].push(pos_list);
                self.doc_freqs[term_id as usize] += 1;
            }

            // N-gram tier merge (only when allocated, i.e. max_n >= 2).
            // doc_id is monotonically increasing across the loop — `add_doc`'s
            // sorted-append contract holds.
            // Crash safety: a panic in `add_doc` leaves the unigram tier ahead by one
            // doc — same partial-update window as the rest of `add_cpu`. Full transactional
            // `add` is out of scope; callers that need atomicity must wrap in their own retry.
            if let Some(side) = self.ngram_side.as_mut() {
                side.add_doc(doc_id, &tokens);
            }

            self.doc_lengths.push(doc_len);
            self.total_tokens += doc_len as u64;
            ids.push(doc_id as usize);
            pb.inc(1);
        }
        pb.finish_and_clear();
        let t_merge = t0.elapsed();

        self.num_docs = base_id + documents.len() as u32;

        let t_total_elapsed = t_total.elapsed();
        if std::env::var("BM25X_PROFILE").is_ok() {
            eprintln!(
                "[bm25x-cpu] {:.3}s total | tokenize={:.3}s merge={:.3}s | {:.0} d/s",
                t_total_elapsed.as_secs_f64(),
                t_tokenize.as_secs_f64(),
                t_merge.as_secs_f64(),
                documents.len() as f64 / t_total_elapsed.as_secs_f64(),
            );
        }

        self.auto_save()?;
        Ok(ids)
    }

    /// Search the index and return top-k results sorted by descending score.
    ///
    /// When `max_n >= 2`, scores combine the unigram tier and the hashed
    /// n-gram side: both tiers contribute additively to each doc's BM25 score.
    /// IDF for the n-gram tier uses slot DF (over-estimate of the true n-gram
    /// DF; D7 in the multi-gram plan).
    pub fn search(&self, query: &str, k: usize) -> Vec<SearchResult> {
        if self.num_docs == 0 {
            return Vec::new();
        }

        let (uni_term_ids, ngram_slots) = self.query_targets(query);
        let params = ScoringParams {
            k1: self.k1,
            b: self.b,
            delta: self.delta,
            avgdl: self.total_tokens as f32 / self.num_docs as f32,
        };

        with_score_buf(self.num_docs as usize, |scores| {
            let mut touched = Vec::new();

            if self.score_unigram {
                for tid in uni_term_ids {
                    let df = self.doc_freqs.get(tid as usize).copied().unwrap_or(0);
                    if df == 0 {
                        continue;
                    }
                    let idf_val = scoring::idf(self.method, self.num_docs, df);
                    self.for_each_posting(tid, |doc_id, tf| {
                        let dl = self.get_doc_length(doc_id);
                        let s = scoring::score(self.method, tf, dl, &params, idf_val);
                        let i = doc_id as usize;
                        if scores[i] == 0.0 {
                            touched.push(doc_id);
                        }
                        scores[i] += s;
                    });
                }
            }

            if let Some(side) = self.ngram_side.as_ref() {
                for slot in ngram_slots {
                    let df = side.df(slot);
                    if df == 0 {
                        continue;
                    }
                    let idf_val = scoring::idf(self.method, self.num_docs, df);
                    side.for_each_posting(slot, |doc_id, tf| {
                        let dl = self.get_doc_length(doc_id);
                        let s = scoring::score(self.method, tf, dl, &params, idf_val);
                        let i = doc_id as usize;
                        if scores[i] == 0.0 {
                            touched.push(doc_id);
                        }
                        scores[i] += s;
                    });
                }
            }

            let result = Self::topk_from_scores(scores, &touched, k);
            for &doc_id in &touched {
                scores[doc_id as usize] = 0.0;
            }
            result
        })
    }

    /// Search with expression syntax (AND, OR, NOT, +/-, boost, grouping).
    ///
    /// Parses the query string using tantivy's query grammar and evaluates
    /// boolean logic against the inverted index.
    pub fn search_expr(&self, query: &str, k: usize) -> Vec<SearchResult> {
        let text = query.trim();
        if text.is_empty() {
            return Vec::new();
        }
        let (ast, _errors) = tantivy_query_grammar::parse_query_lenient(text);
        crate::query::evaluate(&ast, self, k)
    }

    /// Batch expression search with rayon parallelism.
    pub fn search_expr_batch(&self, queries: &[&str], k: usize) -> Vec<Vec<SearchResult>> {
        if self.num_docs == 0 {
            return queries.iter().map(|_| Vec::new()).collect();
        }

        let pool = capped_pool();
        pool.install(|| {
            queries
                .par_iter()
                .map(|q| self.search_expr(q, k))
                .collect()
        })
    }

    /// Batch search: run multiple queries in parallel using rayon.
    /// Returns one result list per query. Much faster than sequential search()
    /// because it amortizes score array allocation and enables CPU parallelism.
    pub fn search_batch(&self, queries: &[&str], k: usize) -> Vec<Vec<SearchResult>> {
        if self.num_docs == 0 {
            return queries.iter().map(|_| Vec::new()).collect();
        }

        let pool = capped_pool();
        pool.install(|| {
            queries
                .par_iter()
                .map(|query| self.search(query, k))
                .collect()
        })
    }

    /// Batch filtered search: run multiple queries with per-query subsets in parallel.
    pub fn search_filtered_batch(
        &self,
        queries: &[&str],
        k: usize,
        subsets: &[&[usize]],
    ) -> Vec<Vec<SearchResult>> {
        if self.num_docs == 0 {
            return queries.iter().map(|_| Vec::new()).collect();
        }

        let pool = capped_pool();
        pool.install(|| {
            queries
                .par_iter()
                .zip(subsets.par_iter())
                .map(|(query, subset)| self.search_filtered(query, k, subset))
                .collect()
        })
    }

    /// Batch GPU search: process multiple queries across multiple GPUs.
    /// If a MultiGpuSearchIndex is provided, queries are distributed across GPUs.
    /// Otherwise falls back to single-GPU sequential processing.
    #[cfg(feature = "cuda")]
    pub fn search_gpu_batch(
        &self,
        gpu_index: &mut crate::cuda::GpuSearchIndex,
        queries: &[&str],
        k: usize,
    ) -> Vec<Vec<SearchResult>> {
        let ctx = match crate::cuda::get_global_context() {
            Some(c) => c,
            None => return self.search_batch(queries, k),
        };

        let params = ScoringParams {
            k1: self.k1,
            b: self.b,
            delta: self.delta,
            avgdl: self.total_tokens as f32 / self.num_docs as f32,
        };

        let all_query_terms = self.tokenize_queries(queries);

        let mut results = Vec::with_capacity(queries.len());
        for query_terms in &all_query_terms {
            match gpu_index.search(&ctx, query_terms, params.k1, params.b, params.avgdl, k) {
                Ok(r) => results.push(
                    r.into_iter()
                        .map(|(doc_id, score)| SearchResult {
                            index: doc_id as usize,
                            score,
                        })
                        .collect(),
                ),
                Err(e) => {
                    eprintln!("[bm25x] GPU batch search failed: {}", e);
                    results.push(Vec::new());
                }
            }
        }
        results
    }

    /// Batch GPU search across multiple GPUs. Queries distributed evenly.
    #[cfg(feature = "cuda")]
    pub fn search_multi_gpu_batch(
        &self,
        multi_gpu: &mut crate::multi_gpu::MultiGpuSearchIndex,
        queries: &[&str],
        k: usize,
    ) -> Vec<Vec<SearchResult>> {
        let params = ScoringParams {
            k1: self.k1,
            b: self.b,
            delta: self.delta,
            avgdl: self.total_tokens as f32 / self.num_docs as f32,
        };

        let n_queries = queries.len();
        let n_gpus = multi_gpu.num_gpus();
        if n_queries == 0 {
            return Vec::new();
        }

        let chunk_size = n_queries.div_ceil(n_gpus);
        let mut all_results: Vec<Vec<SearchResult>> = vec![Vec::new(); n_queries];

        // Each GPU thread tokenizes + scores its own queries inline.
        // Tokenization of query N+1 overlaps with GPU scoring of query N
        // within the same thread — no separate pre-tokenization phase.
        multi_gpu.search_batch_with_tokenizer(
            queries,
            chunk_size,
            |query| self.tokenize_single(query),
            params.k1,
            params.b,
            params.avgdl,
            k,
            &mut all_results,
        );

        all_results
    }

    /// Tokenize a query and resolve it into both tier targets:
    /// - unigram tokens → exact vocab term_ids (deduped, in emission order)
    /// - 2..=max_n grams → n-gram side slots that have postings (deduped, in emission order)
    ///
    /// When `max_n == 1`, the n-gram slot vec is empty.
    /// When the n-gram side has DF=0 for a hashed slot (i.e. no document n-gram
    /// hashes there), the slot is omitted from the result. This avoids touching
    /// empty slots in the scoring loop.
    pub(crate) fn query_targets(&self, query: &str) -> (Vec<u32>, Vec<u32>) {
        let tokens = self.tokenizer.tokenize_owned(query);

        // Unigram pass.
        let mut uni: Vec<u32> = Vec::with_capacity(tokens.len());
        let mut uni_seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for t in &tokens {
            if let Some(&tid) = self.vocab.get(t.as_str()) {
                if uni_seen.insert(tid) { uni.push(tid); }
            }
        }

        // N-gram pass (only when side is allocated; only the n's in side.ns()).
        let ng = match self.ngram_side.as_ref() {
            Some(side) => {
                let max_n = side.max_n() as usize;
                let mut out: Vec<u32> = Vec::with_capacity(tokens.len() * max_n.saturating_sub(1));
                let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
                for (slot, _start) in crate::ngram::iter_ngram_slots_in(&tokens, side.ns(), side.mask()) {
                    if side.df(slot) > 0 && seen.insert(slot) {
                        out.push(slot);
                    }
                }
                out
            }
            None => Vec::new(),
        };

        (uni, ng)
    }

    /// Like `query_targets` but also looks up each `extra_ngrams` key as an
    /// exact slot, bypassing the sliding window. Used by query expansion so
    /// that expansion n-grams contribute only their exact slot and not their
    /// sub-n-grams.
    ///
    /// Each element of `extra_ngrams` is a space-separated sequence of
    /// already-stemmed tokens (e.g. `"diagnost accuraci"`). The function
    /// tokenises it and looks up the single slot for the full sequence.
    /// Elements whose length does not match any n in `side.ns()`, or whose
    /// slot has df == 0, are silently skipped.
    pub(crate) fn query_targets_with_extras(
        &self,
        query: &str,
        extra_ngrams: &[String],
    ) -> (Vec<u32>, Vec<u32>) {
        let (uni, mut ng) = self.query_targets(query);
        if let Some(side) = self.ngram_side.as_ref() {
            let mut seen: std::collections::HashSet<u32> =
                ng.iter().copied().collect();
            for key in extra_ngrams {
                let tokens = self.tokenizer.tokenize_owned(key.as_str());
                let n = tokens.len() as u8;
                if n == 0 || !side.ns().contains(&n) {
                    continue;
                }
                // Compute the slot for this exact sequence — only one window of length n.
                for (slot, _) in
                    crate::ngram::iter_ngram_slots_in(&tokens, &[n], side.mask())
                {
                    if side.df(slot) > 0 && seen.insert(slot) {
                        ng.push(slot);
                    }
                    break;
                }
            }
        }
        (uni, ng)
    }

    /// Like `search` but uses `query_targets_with_extras` so that each
    /// element of `extra_ngrams` is matched as an exact slot (no sub-n-gram
    /// expansion).
    pub fn search_with_extras(
        &self,
        query: &str,
        extra_ngrams: &[String],
        k: usize,
    ) -> Vec<SearchResult> {
        if self.num_docs == 0 {
            return Vec::new();
        }

        let (uni_term_ids, ngram_slots) = self.query_targets_with_extras(query, extra_ngrams);
        let params = ScoringParams {
            k1: self.k1,
            b: self.b,
            delta: self.delta,
            avgdl: self.total_tokens as f32 / self.num_docs as f32,
        };

        with_score_buf(self.num_docs as usize, |scores| {
            let mut touched: Vec<u32> = Vec::new();

            if self.score_unigram {
                for tid in uni_term_ids {
                    let df = self.doc_freqs.get(tid as usize).copied().unwrap_or(0);
                    if df == 0 {
                        continue;
                    }
                    let idf_val = scoring::idf(self.method, self.num_docs, df);
                    self.for_each_posting(tid, |doc_id, tf| {
                        let dl = self.get_doc_length(doc_id);
                        let s = scoring::score(self.method, tf, dl, &params, idf_val);
                        let i = doc_id as usize;
                        if scores[i] == 0.0 {
                            touched.push(doc_id);
                        }
                        scores[i] += s;
                    });
                }
            }

            if let Some(side) = self.ngram_side.as_ref() {
                for slot in ngram_slots {
                    let df = side.df(slot);
                    if df == 0 {
                        continue;
                    }
                    let idf_val = scoring::idf(self.method, self.num_docs, df);
                    side.for_each_posting(slot, |doc_id, tf| {
                        let dl = self.get_doc_length(doc_id);
                        let s = scoring::score(self.method, tf, dl, &params, idf_val);
                        let i = doc_id as usize;
                        if scores[i] == 0.0 {
                            touched.push(doc_id);
                        }
                        scores[i] += s;
                    });
                }
            }

            let result = Self::topk_from_scores(scores, &touched, k);
            for &doc_id in &touched {
                scores[doc_id as usize] = 0.0;
            }
            result
        })
    }

    /// Batch version of `search_with_extras`. `extra_ngrams[i]` is the list
    /// of pre-stemmed keys for `queries[i]`.
    pub fn search_batch_with_extras(
        &self,
        queries: &[&str],
        extra_ngrams: &[Vec<String>],
        k: usize,
    ) -> Vec<Vec<SearchResult>> {
        assert_eq!(queries.len(), extra_ngrams.len());
        if self.num_docs == 0 {
            return queries.iter().map(|_| Vec::new()).collect();
        }

        let pool = capped_pool();
        pool.install(|| {
            queries
                .par_iter()
                .zip(extra_ngrams.par_iter())
                .map(|(q, extras)| self.search_with_extras(q, extras, k))
                .collect()
        })
    }

    /// Per-expansion-key BM25 score contribution for specific documents.
    ///
    /// For each (slot, doc_index) pair, binary-searches the posting list
    /// (sorted by doc_id) and computes the Lucene BM25 score contribution.
    ///
    /// Returns non-zero entries only as
    /// ``(slot_idx, doc_pos, idf, tf_raw, score)`` where:
    /// - ``slot_idx`` indexes into the input ``slots`` slice
    /// - ``doc_pos`` indexes into the input ``doc_indices`` slice
    pub fn ngram_score_breakdown_for_docs(
        &self,
        doc_indices: &[u32],
        slots: &[u32],
    ) -> Vec<(u32, u32, f32, u32, f32)> {
        let side = match self.ngram_side.as_ref() {
            Some(s) => s,
            None => return Vec::new(),
        };
        let params = ScoringParams {
            k1: self.k1,
            b: self.b,
            delta: self.delta,
            avgdl: self.total_tokens as f32 / self.num_docs as f32,
        };
        let mut out: Vec<(u32, u32, f32, u32, f32)> = Vec::new();
        for (slot_idx, &slot) in slots.iter().enumerate() {
            let df = side.df(slot);
            if df == 0 {
                continue;
            }
            let idf = scoring::idf(self.method, self.num_docs, df);
            let postings = &side.postings[slot as usize];
            for (doc_pos, &doc_idx) in doc_indices.iter().enumerate() {
                if let Ok(p) = postings.binary_search_by_key(&doc_idx, |&(d, _)| d) {
                    let tf = postings[p].1;
                    let dl = self.get_doc_length(doc_idx);
                    let score = scoring::score(self.method, tf, dl, &params, idf);
                    out.push((slot_idx as u32, doc_pos as u32, idf, tf, score));
                }
            }
        }
        out
    }

    /// Tokenize a single query and resolve term IDs.
    ///
    /// Output term IDs occupy two ranges in the GPU's flat posting layout:
    /// `[0, vocab.len())` are unigram vocab IDs and
    /// `[vocab.len(), vocab.len() + n_features)` are n-gram side slots offset
    /// by `vocab.len()`. The GPU upload (`combined_postings`) lays the two
    /// tiers contiguously in the same flat array, so kernels need zero changes.
    #[cfg(feature = "cuda")]
    fn tokenize_single(&self, query: &str) -> Vec<(u32, f32)> {
        let (uni_ids, ngram_slots) = self.query_targets(query);
        let mut out = Vec::with_capacity(uni_ids.len() + ngram_slots.len());
        if self.score_unigram {
            for tid in uni_ids {
                let df = self.doc_freqs.get(tid as usize).copied().unwrap_or(0);
                if df > 0 {
                    out.push((tid, scoring::idf(self.method, self.num_docs, df)));
                }
            }
        }
        if let Some(side) = self.ngram_side.as_ref() {
            let base = self.vocab.len() as u32;
            for slot in ngram_slots {
                let df = side.df(slot);
                if df > 0 {
                    out.push((base + slot, scoring::idf(self.method, self.num_docs, df)));
                }
            }
        }
        out
    }

    /// Helper: tokenize + resolve term IDs for a batch of queries (parallel on CPU).
    /// Same offset-encoding scheme as `tokenize_single` (see its docstring).
    #[cfg(feature = "cuda")]
    fn tokenize_queries(&self, queries: &[&str]) -> Vec<Vec<(u32, f32)>> {
        uncapped_pool().install(|| {
            queries
                .par_iter()
                .map(|query| self.tokenize_single(query))
                .collect()
        })
    }

    /// Search restricted to a subset of document IDs (pre-filtering).
    /// Only documents whose index is in `subset` will be scored.
    /// IDF is computed from global corpus stats so scores stay comparable.
    ///
    /// Uses a doc-centric approach: iterates subset IDs and looks up each doc's
    /// TF via binary search on posting lists. Cost is O(|subset| * |query_terms| * log(posting_len)).
    pub fn search_filtered(&self, query: &str, k: usize, subset: &[usize]) -> Vec<SearchResult> {
        if self.num_docs == 0 || subset.is_empty() {
            return Vec::new();
        }

        let (uni_term_ids, ngram_slots) = self.query_targets(query);
        let params = ScoringParams {
            k1: self.k1,
            b: self.b,
            delta: self.delta,
            avgdl: self.total_tokens as f32 / self.num_docs as f32,
        };

        // Build per-unigram (term_id, idf) only when the unigram tier is in
        // the score set. When `score_unigram == false`, leave it empty so the
        // per-doc inner loop short-circuits without touching `get_tf`.
        let mut uni_terms: Vec<(u32, f32)> = if self.score_unigram {
            Vec::with_capacity(uni_term_ids.len())
        } else {
            Vec::new()
        };
        if self.score_unigram {
            for tid in uni_term_ids {
                let df = self.doc_freqs.get(tid as usize).copied().unwrap_or(0);
                if df == 0 {
                    continue;
                }
                uni_terms.push((tid, scoring::idf(self.method, self.num_docs, df)));
            }
        }
        let ngram_terms: Vec<(u32, f32)> = match self.ngram_side.as_ref() {
            Some(side) => ngram_slots
                .iter()
                .filter_map(|&slot| {
                    let df = side.df(slot);
                    if df == 0 {
                        None
                    } else {
                        Some((slot, scoring::idf(self.method, self.num_docs, df)))
                    }
                })
                .collect(),
            None => Vec::new(),
        };
        if uni_terms.is_empty() && ngram_terms.is_empty() {
            return Vec::new();
        }

        let mut heap = BinaryHeap::with_capacity(k + 1);
        for &doc_idx in subset {
            let doc_id = doc_idx as u32;
            if doc_id >= self.num_docs {
                continue;
            }
            let dl = self.get_doc_length(doc_id);
            let mut total_score = 0.0f32;
            for &(term_id, idf_val) in &uni_terms {
                if let Some(tf) = self.get_tf(term_id, doc_id) {
                    total_score += scoring::score(self.method, tf, dl, &params, idf_val);
                }
            }
            if let Some(side) = self.ngram_side.as_ref() {
                for &(slot, idf_val) in &ngram_terms {
                    if let Some(tf) = side.posting_tf(slot, doc_id) {
                        total_score += scoring::score(self.method, tf, dl, &params, idf_val);
                    }
                }
            }
            if total_score > 0.0 {
                heap.push(MinScored(total_score, doc_id));
                if heap.len() > k {
                    heap.pop();
                }
            }
        }

        let mut results: Vec<SearchResult> = heap
            .into_iter()
            .map(|MinScored(score, doc_id)| SearchResult {
                index: doc_id as usize,
                score,
            })
            .collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        results
    }

    pub(crate) fn topk_from_scores(scores: &[f32], touched: &[u32], k: usize) -> Vec<SearchResult> {
        let mut heap = BinaryHeap::with_capacity(k + 1);
        for &doc_id in touched {
            let s = scores[doc_id as usize];
            if s > 0.0 {
                heap.push(MinScored(s, doc_id));
                if heap.len() > k {
                    heap.pop();
                }
            }
        }
        let mut results: Vec<SearchResult> = heap
            .into_iter()
            .map(|MinScored(score, doc_id)| SearchResult {
                index: doc_id as usize,
                score,
            })
            .collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        results
    }

    fn merge_unigram_postings(&mut self, id: u32, tf_pos_map: HashMap<String, (u32, Vec<u32>)>) {
        for (token, (extra_tf, extra_positions)) in tf_pos_map {
            let term_id = self.get_or_create_term(&token);
            let plist = &mut self.postings[term_id as usize];
            match plist.binary_search_by_key(&id, |&(did, _)| did) {
                Ok(idx) => {
                    plist[idx].1 += extra_tf;
                    if term_id < self.positions.len() as u32
                        && idx < self.positions[term_id as usize].len()
                    {
                        self.positions[term_id as usize][idx].extend(extra_positions);
                    }
                }
                Err(insert_pos) => {
                    plist.insert(insert_pos, (id, extra_tf));
                    if term_id < self.positions.len() as u32 {
                        self.positions[term_id as usize].insert(insert_pos, extra_positions);
                    }
                    self.doc_freqs[term_id as usize] += 1;
                }
            }
        }
    }

    fn subtract_unigram_postings(&mut self, id: u32, tf_map: HashMap<String, u32>) {
        for (token, extra_tf) in tf_map {
            if let Some(term_id) = self.vocab.get(&token) {
                let term_id = *term_id;
                let plist = &mut self.postings[term_id as usize];
                if let Ok(idx) = plist.binary_search_by_key(&id, |&(did, _)| did) {
                    if plist[idx].1 <= extra_tf {
                        plist.remove(idx);
                        if term_id < self.positions.len() as u32
                            && idx < self.positions[term_id as usize].len()
                        {
                            self.positions[term_id as usize].remove(idx);
                        }
                        self.doc_freqs[term_id as usize] =
                            self.doc_freqs[term_id as usize].saturating_sub(1);
                    } else {
                        plist[idx].1 -= extra_tf;
                    }
                }
            }
        }
    }

    /// Delete one or more documents by their indices.
    /// All documents after a deleted index shift down to fill the gap.
    /// For example: deleting doc 1 from [0,1,2] makes old doc 2 become new doc 1.
    pub fn delete(&mut self, doc_ids: &[usize]) -> io::Result<()> {
        if doc_ids.is_empty() {
            return Ok(());
        }
        if self.mmap_data.is_some() {
            self.materialize_mmap();
        }

        // Sort and deduplicate, filter out-of-range
        let mut to_delete: Vec<u32> = doc_ids
            .iter()
            .map(|&id| id as u32)
            .filter(|&id| id < self.num_docs)
            .collect();
        to_delete.sort_unstable();
        to_delete.dedup();

        if to_delete.is_empty() {
            return Ok(());
        }

        // Subtract deleted doc lengths from total_tokens
        for &id in &to_delete {
            self.total_tokens -= self.doc_lengths[id as usize] as u64;
        }

        // Build old_id -> new_id mapping.
        // Deleted docs map to u32::MAX (sentinel), others shift down.
        let old_count = self.num_docs as usize;
        let mut id_map: Vec<u32> = Vec::with_capacity(old_count);
        let mut del_idx = 0;
        let mut shift = 0u32;
        for old_id in 0..old_count as u32 {
            if del_idx < to_delete.len() && to_delete[del_idx] == old_id {
                id_map.push(u32::MAX); // sentinel: deleted
                shift += 1;
                del_idx += 1;
            } else {
                id_map.push(old_id - shift);
            }
        }

        let new_count = self.num_docs - to_delete.len() as u32;

        // Compact doc_lengths
        let mut new_doc_lengths = Vec::with_capacity(new_count as usize);
        for (old_id, &dl) in self.doc_lengths.iter().enumerate() {
            if id_map[old_id] != u32::MAX {
                new_doc_lengths.push(dl);
            }
        }
        self.doc_lengths = new_doc_lengths;

        // Remap posting lists and positions: remove deleted entries, remap doc_ids
        for (term_id, plist) in self.postings.iter_mut().enumerate() {
            let old_len = plist.len();
            // Build retain mask before mutating
            let keep: Vec<bool> = plist.iter().map(|&(did, _)| id_map[did as usize] != u32::MAX).collect();
            // Compact positions in lockstep
            if term_id < self.positions.len() {
                let pos_list = &mut self.positions[term_id];
                let mut new_pos = Vec::with_capacity(pos_list.len());
                for (i, &retain) in keep.iter().enumerate() {
                    if retain {
                        if i < pos_list.len() {
                            new_pos.push(std::mem::take(&mut pos_list[i]));
                        }
                    }
                }
                *pos_list = new_pos;
            }
            plist.retain(|&(did, _)| id_map[did as usize] != u32::MAX);
            let removed = old_len - plist.len();
            if removed > 0 {
                self.doc_freqs[term_id] -= removed as u32;
            }
            for entry in plist.iter_mut() {
                entry.0 = id_map[entry.0 as usize];
            }
        }

        // N-gram tier compaction (only when allocated). Reuses the same
        // `id_map` built for the unigram tier so the two tiers stay in lockstep
        // (sentinel `u32::MAX` for deleted, shifted compacted IDs otherwise).
        // Crash safety: a panic in `compact` leaves the unigram tier already
        // compacted while the n-gram tier still holds the pre-delete IDs —
        // same partial-update window as the rest of `delete`. Full transactional
        // `delete` is out of scope; callers that need atomicity must wrap in
        // their own retry.
        if let Some(side) = self.ngram_side.as_mut() {
            side.compact(&id_map);
        }

        self.num_docs = new_count;
        self.auto_save()
    }

    /// Update a document's text. The document keeps its index.
    ///
    /// When `max_n >= 2`, the n-gram tier is also rebuilt, which is
    /// O(n_features) per call (~50ms at default n_features=8M).
    pub fn update(&mut self, doc_id: usize, new_text: &str) -> io::Result<()> {
        if self.mmap_data.is_some() {
            self.materialize_mmap();
        }

        let id = doc_id as u32;
        assert!(
            id < self.num_docs,
            "doc_id {} out of range (num_docs={})",
            doc_id,
            self.num_docs
        );

        // Remove old postings and positions for this doc
        let old_dl = self.doc_lengths[id as usize];
        self.total_tokens -= old_dl as u64;
        for (term_id, postings) in self.postings.iter_mut().enumerate() {
            if let Some(idx) = postings.iter().position(|&(did, _)| did == id) {
                postings.remove(idx);
                if term_id < self.positions.len() && idx < self.positions[term_id].len() {
                    self.positions[term_id].remove(idx);
                }
                self.doc_freqs[term_id] -= 1;
            }
        }

        // Re-tokenize and re-index with positions.
        // ONE tokenization is shared between the unigram tier and the n-gram side
        // (when allocated). Mirrors the `add_cpu` fast-path discipline:
        // when `max_n == 1`, move tokens straight into the TF map (no clone tax);
        // when `max_n >= 2`, clone into the TF map so `tokens` survives for
        // `NgramSide::replace_doc` below.
        let tokens = self.tokenizer.tokenize_owned(new_text);
        let doc_len = tokens.len() as u32;

        // Build TF/positions map (consumes or borrows `tokens` per branch). The
        // multigram branch keeps `tokens` alive; we hold it in `kept_tokens` for
        // the n-gram tier call after the unigram merge. Borrow-checker note: we
        // can't gate the `&tokens` reference inside `if let Some(side) = ...`
        // because the `max_n == 1` branch moves `tokens` — so the post-merge
        // `as_mut()` would still fail to compile. Two-branch struct keeps both
        // paths clean.
        let (tf_pos_map, kept_tokens): (HashMap<String, (u32, Vec<u32>)>, Option<Vec<String>>) =
            if self.max_n == 1 {
                let mut m: HashMap<String, (u32, Vec<u32>)> = HashMap::new();
                for (pos, token) in tokens.into_iter().enumerate() {
                    let entry = m.entry(token).or_insert((0, Vec::new()));
                    entry.0 += 1;
                    entry.1.push(pos as u32);
                }
                (m, None)
            } else {
                let mut m: HashMap<String, (u32, Vec<u32>)> = HashMap::new();
                for (pos, token) in tokens.iter().enumerate() {
                    let entry = m.entry(token.clone()).or_insert((0, Vec::new()));
                    entry.0 += 1;
                    entry.1.push(pos as u32);
                }
                (m, Some(tokens))
            };

        for (token, (tf, pos_list)) in tf_pos_map {
            let term_id = self.get_or_create_term(&token);
            let plist = &mut self.postings[term_id as usize];
            let insert_pos = plist.partition_point(|&(did, _)| did < id);
            plist.insert(insert_pos, (id, tf));
            if term_id < self.positions.len() as u32 {
                self.positions[term_id as usize].insert(insert_pos, pos_list);
            }
            self.doc_freqs[term_id as usize] += 1;
        }

        // N-gram tier re-index (only when allocated, i.e. max_n >= 2). Both
        // sides of the pair are populated together by the multigram branch
        // above, so `kept_tokens.is_some()` iff `self.ngram_side.is_some()`.
        // Crash safety: a panic in `replace_doc` leaves the unigram tier
        // already re-indexed while the n-gram tier still holds the previous
        // doc's postings — same partial-update window as the rest of `update`.
        // Full transactional `update` is out of scope; callers that need
        // atomicity must wrap in their own retry.
        if let (Some(side), Some(toks)) = (self.ngram_side.as_mut(), kept_tokens.as_ref()) {
            side.replace_doc(id, toks);
        }

        self.doc_lengths[id as usize] = doc_len;
        self.total_tokens += doc_len as u64;
        self.auto_save()
    }

    /// Enrich a document by adding extra tokens to its index.
    ///
    /// Unlike `update()`, this does NOT remove existing postings — it merges
    /// new tokens into the document's existing index representation:
    /// - Existing terms: TF is accumulated, new positions are appended.
    /// - New terms: a fresh posting entry is inserted.
    /// - `doc_lengths[doc_id]` and `total_tokens` grow by the extra token count.
    ///
    /// Use this for document augmentation (keywords, entities, synonyms).
    ///
    /// When `max_n >= 2`, n-gram contributions from `extra_text` are also
    /// added; cross-boundary n-grams between the doc's existing tokens and
    /// the new tokens are NOT created (extra_text is tokenized in isolation).
    pub fn enrich(&mut self, doc_id: usize, extra_text: &str) -> io::Result<()> {
        if self.mmap_data.is_some() {
            self.materialize_mmap();
        }

        let id = doc_id as u32;
        assert!(
            id < self.num_docs,
            "doc_id {} out of range (num_docs={})",
            doc_id,
            self.num_docs
        );

        // Tokenize extra text ONCE; share with the n-gram side when allocated.
        let tokens = self.tokenizer.tokenize_owned(extra_text);
        let extra_len = tokens.len() as u32;
        if extra_len == 0 {
            return Ok(());
        }

        // Current doc length is the position offset for new tokens (used by both
        // the unigram tier here and the n-gram side below).
        let pos_offset = self.doc_lengths[id as usize];

        // Build TF + position map for the extra tokens.
        // Same fast-path discipline as `update`: `max_n == 1` consumes `tokens`
        // (no clone tax); `max_n >= 2` clones into the map and stashes `tokens`
        // for the n-gram side via `kept_tokens`. See `update` for the borrow-
        // checker rationale (can't gate the `&tokens` reference inside an
        // `if let Some(side)` after a conditional move).
        let (tf_pos_map, kept_tokens): (HashMap<String, (u32, Vec<u32>)>, Option<Vec<String>>) =
            if self.max_n == 1 {
                let mut m: HashMap<String, (u32, Vec<u32>)> = HashMap::new();
                for (i, token) in tokens.into_iter().enumerate() {
                    let entry = m.entry(token).or_insert((0, Vec::new()));
                    entry.0 += 1;
                    entry.1.push(pos_offset + i as u32);
                }
                (m, None)
            } else {
                let mut m: HashMap<String, (u32, Vec<u32>)> = HashMap::new();
                for (i, token) in tokens.iter().enumerate() {
                    let entry = m.entry(token.clone()).or_insert((0, Vec::new()));
                    entry.0 += 1;
                    entry.1.push(pos_offset + i as u32);
                }
                (m, Some(tokens))
            };

        self.merge_unigram_postings(id, tf_pos_map);

        // N-gram tier merge (only when allocated). `pos_offset` is the doc's
        // pre-enrich length — same offset the unigram tier just used to
        // position the new tokens, so `extra_text`'s n-grams land at matching
        // positions on the n-gram side. Cross-boundary n-grams (last existing
        // token + first extra token) are intentionally NOT created — extra_text
        // is tokenized in isolation, mirroring the unigram enrich semantics.
        // Crash safety: a panic in `enrich_doc` leaves the unigram tier ahead
        // by the new tokens while the n-gram tier still holds the pre-enrich
        // postings — same partial-update window as the rest of `enrich`. Full
        // transactional `enrich` is out of scope; callers that need atomicity
        // must wrap in their own retry.
        if let (Some(side), Some(toks)) = (self.ngram_side.as_mut(), kept_tokens.as_ref()) {
            side.enrich_doc(id, toks, pos_offset);
        }

        // Update doc length and total tokens
        self.doc_lengths[id as usize] += extra_len;
        self.total_tokens += extra_len as u64;
        self.auto_save()
    }

    /// Reverse a previous `enrich()` call by subtracting the same tokens.
    ///
    /// Must be called with the exact same `extra_text` that was passed to
    /// `enrich()`.  Decrements TF for each token; if a term's TF reaches
    /// zero for this document, the posting entry is removed and `doc_freqs`
    /// is decremented.  `doc_lengths` and `total_tokens` are reduced.
    ///
    /// **Positions are not modified** — BM25 scoring uses only TF, doc
    /// length, and DF, so stale positions have no effect on scores.
    /// Phrase / proximity queries may return slightly stale results for
    /// unenriched terms; reload the index if exact phrase matching is needed.
    ///
    /// When `max_n >= 2`, the matching n-gram contributions are also
    /// subtracted; positions on both tiers are left stale (BM25 scoring uses
    /// TF/DF/dl only).
    pub fn unenrich(&mut self, doc_id: usize, extra_text: &str) -> io::Result<()> {
        if self.mmap_data.is_some() {
            self.materialize_mmap();
        }

        let id = doc_id as u32;
        assert!(
            id < self.num_docs,
            "doc_id {} out of range (num_docs={})",
            doc_id,
            self.num_docs
        );

        let tokens = self.tokenizer.tokenize_owned(extra_text);
        let extra_len = tokens.len() as u32;
        if extra_len == 0 {
            return Ok(());
        }

        // Build TF map for the tokens to remove. Same fast-path discipline as
        // `enrich`: `max_n == 1` consumes `tokens`; `max_n >= 2` clones into the
        // map and stashes `tokens` for the n-gram side via `kept_tokens`.
        let (tf_map, kept_tokens): (HashMap<String, u32>, Option<Vec<String>>) = if self.max_n == 1
        {
            let mut m: HashMap<String, u32> = HashMap::new();
            for token in tokens.into_iter() {
                *m.entry(token).or_insert(0) += 1;
            }
            (m, None)
        } else {
            let mut m: HashMap<String, u32> = HashMap::new();
            for token in tokens.iter() {
                *m.entry(token.clone()).or_insert(0) += 1;
            }
            (m, Some(tokens))
        };

        self.subtract_unigram_postings(id, tf_map);

        // N-gram tier subtraction (only when allocated). Like the unigram path,
        // positions are NOT mutated on the n-gram side — `unenrich_doc`
        // documents this caveat. BM25 scoring uses TF/DF/dl only, so stale
        // positions don't affect scores; phrase queries on unenriched n-grams
        // may return stale matches.
        // Crash safety: a panic in `unenrich_doc` leaves the unigram tier with
        // the subtraction already applied while the n-gram tier still reflects
        // the pre-unenrich state — same partial-update window as the rest of
        // `unenrich`. Full transactional `unenrich` is out of scope; callers
        // that need atomicity must wrap in their own retry.
        if let (Some(side), Some(toks)) = (self.ngram_side.as_mut(), kept_tokens.as_ref()) {
            side.unenrich_doc(id, toks);
        }

        self.doc_lengths[id as usize] = self.doc_lengths[id as usize].saturating_sub(extra_len);
        self.total_tokens = self.total_tokens.saturating_sub(extra_len as u64);
        self.auto_save()
    }

    /// Enrich a doc treating `extra_text` as a SINGLE n-gram (no sub-windows).
    ///
    /// Same unigram-tier semantics as [`Self::enrich`] (each token gets its
    /// own posting / df / position update, doc_lengths/total_tokens grow by
    /// `tokens.len()`). Differs only on the n-gram tier: instead of writing
    /// every contiguous (n=2..=max_n) sub-window of the token sequence, this
    /// writes EXACTLY ONE slot — the hash of the full token sequence as a
    /// single n-gram of length `tokens.len()`.
    ///
    /// Use case: LLM-proposed enrichment phrases where the caller wants the
    /// proposed phrase to be the only n-gram-tier signal. Sub-windows would
    /// inflate df on common bigrams ("of the") that the LLM never intended
    /// to emphasise; this path keeps the LLM's intent intact. See the
    /// "How `add()` propagates" section in
    /// `sira.search.bm25.EnrichmentAdapter` for the rationale.
    ///
    /// Constraints (silent no-op outcomes):
    /// - `tokens.len() < 2` → only the unigram tier is touched (no n-gram
    ///   tier write; matches the "n=1 lives in unigram tier" invariant).
    /// - `tokens.len() > max_n` (or not in `ngram_side.ns`) → still no
    ///   n-gram-tier write; phrase contributes only its unigram terms.
    pub fn enrich_exact(&mut self, doc_id: usize, extra_text: &str) -> io::Result<()> {
        if self.mmap_data.is_some() {
            self.materialize_mmap();
        }

        let id = doc_id as u32;
        assert!(
            id < self.num_docs,
            "doc_id {} out of range (num_docs={})",
            doc_id,
            self.num_docs
        );

        let tokens = self.tokenizer.tokenize_owned(extra_text);
        let extra_len = tokens.len() as u32;
        if extra_len == 0 {
            return Ok(());
        }

        let pos_offset = self.doc_lengths[id as usize];

        // Same fast-path discipline as `enrich`: `max_n == 1` consumes
        // `tokens`; `max_n >= 2` clones into the map and stashes `tokens`
        // for the n-gram side via `kept_tokens`.
        let (tf_pos_map, kept_tokens): (HashMap<String, (u32, Vec<u32>)>, Option<Vec<String>>) =
            if self.max_n == 1 {
                let mut m: HashMap<String, (u32, Vec<u32>)> = HashMap::new();
                for (i, token) in tokens.into_iter().enumerate() {
                    let entry = m.entry(token).or_insert((0, Vec::new()));
                    entry.0 += 1;
                    entry.1.push(pos_offset + i as u32);
                }
                (m, None)
            } else {
                let mut m: HashMap<String, (u32, Vec<u32>)> = HashMap::new();
                for (i, token) in tokens.iter().enumerate() {
                    let entry = m.entry(token.clone()).or_insert((0, Vec::new()));
                    entry.0 += 1;
                    entry.1.push(pos_offset + i as u32);
                }
                (m, Some(tokens))
            };

        self.merge_unigram_postings(id, tf_pos_map);

        // N-gram tier — single slot for the WHOLE phrase (vs `enrich`'s
        // contiguous sub-window enumeration). Length-out-of-ns is silently
        // ignored inside `enrich_doc_exact`.
        if let (Some(side), Some(toks)) = (self.ngram_side.as_mut(), kept_tokens.as_ref()) {
            side.enrich_doc_exact(id, toks, pos_offset);
        }

        self.doc_lengths[id as usize] += extra_len;
        self.total_tokens += extra_len as u64;
        self.auto_save()
    }

    /// Reverse [`Self::enrich_exact`]. Same unigram-tier semantics as
    /// [`Self::unenrich`]; on the n-gram tier, decrements TF on the SINGLE
    /// slot hashed from the full token sequence (vs `unenrich`'s per-window
    /// subtraction). MUST be called with the exact same `extra_text` that
    /// was passed to `enrich_exact`.
    pub fn unenrich_exact(&mut self, doc_id: usize, extra_text: &str) -> io::Result<()> {
        if self.mmap_data.is_some() {
            self.materialize_mmap();
        }

        let id = doc_id as u32;
        assert!(
            id < self.num_docs,
            "doc_id {} out of range (num_docs={})",
            doc_id,
            self.num_docs
        );

        let tokens = self.tokenizer.tokenize_owned(extra_text);
        let extra_len = tokens.len() as u32;
        if extra_len == 0 {
            return Ok(());
        }

        let (tf_map, kept_tokens): (HashMap<String, u32>, Option<Vec<String>>) = if self.max_n == 1
        {
            let mut m: HashMap<String, u32> = HashMap::new();
            for token in tokens.into_iter() {
                *m.entry(token).or_insert(0) += 1;
            }
            (m, None)
        } else {
            let mut m: HashMap<String, u32> = HashMap::new();
            for token in tokens.iter() {
                *m.entry(token.clone()).or_insert(0) += 1;
            }
            (m, Some(tokens))
        };

        self.subtract_unigram_postings(id, tf_map);

        if let (Some(side), Some(toks)) = (self.ngram_side.as_mut(), kept_tokens.as_ref()) {
            side.unenrich_doc_exact(id, toks);
        }

        self.doc_lengths[id as usize] = self.doc_lengths[id as usize].saturating_sub(extra_len);
        self.total_tokens = self.total_tokens.saturating_sub(extra_len as u64);
        self.auto_save()
    }

    /// Get the number of documents.
    pub fn len(&self) -> usize {
        self.num_docs as usize
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.num_docs == 0
    }

    /// Score a query against a list of documents.
    /// Returns one score per document using the same tokenizer and BM25 parameters.
    /// The documents are treated as the corpus for IDF computation.
    ///
    /// When `max_n >= 2`, n-gram contributions are included additively. The
    /// hashed n-gram slot DF is computed from the PROVIDED documents (NOT
    /// `self.doc_freqs`) — this preserves `score()`'s corpus-relative semantics.
    pub fn score(&self, query: &str, documents: &[&str]) -> Vec<f32> {
        let n = documents.len();
        if n == 0 {
            return Vec::new();
        }

        // Tokenize documents in parallel; build per-doc unigram TF map AND
        // (when n-gram side is allocated) per-doc slot TF map. The n-gram
        // side's `ns()` controls which orders are emitted (subset, not
        // necessarily contiguous).
        let pool = capped_pool();
        let side_ns: Option<Vec<u8>> = self.ngram_side.as_ref().map(|s| s.ns().to_vec());
        let mask = self.ngram_side.as_ref().map(|s| s.mask());
        let tokenized: Vec<(u32, HashMap<String, u32>, HashMap<u32, u32>)> = pool.install(|| {
            documents
                .par_iter()
                .map(|doc| {
                    let tokens = self.tokenizer.tokenize_owned(doc);
                    let dl = tokens.len() as u32;
                    let mut uni_tf: HashMap<String, u32> = HashMap::new();
                    for t in &tokens {
                        *uni_tf.entry(t.clone()).or_insert(0) += 1;
                    }
                    let slot_tf: HashMap<u32, u32> = match (side_ns.as_ref(), mask) {
                        (Some(ns), Some(m)) => {
                            let mut slot_map: HashMap<u32, u32> = HashMap::new();
                            for (slot, _pos) in
                                crate::ngram::iter_ngram_slots_in(&tokens, ns, m)
                            {
                                *slot_map.entry(slot).or_insert(0) += 1;
                            }
                            slot_map
                        }
                        _ => HashMap::new(),
                    };
                    (dl, uni_tf, slot_tf)
                })
                .collect()
        });

        let mut doc_lens: Vec<u32> = Vec::with_capacity(n);
        let mut total_tokens = 0u64;
        for (dl, _, _) in &tokenized {
            total_tokens += *dl as u64;
            doc_lens.push(*dl);
        }
        let avgdl = total_tokens as f32 / n as f32;

        // Compute DF per term + per slot (across the provided documents).
        let mut uni_df: HashMap<&str, u32> = HashMap::new();
        let mut slot_df: HashMap<u32, u32> = HashMap::new();
        for (_, uni_tf, slot_tf) in &tokenized {
            for term in uni_tf.keys() {
                *uni_df.entry(term.as_str()).or_insert(0) += 1;
            }
            for &slot in slot_tf.keys() {
                *slot_df.entry(slot).or_insert(0) += 1;
            }
        }

        // Tokenize query, dedup unigrams + n-gram slots in emission order.
        let q_tokens = self.tokenizer.tokenize_owned(query);
        let mut seen_uni: HashSet<&str> = HashSet::new();
        let mut q_uni: Vec<&str> = Vec::with_capacity(q_tokens.len());
        for t in &q_tokens {
            if seen_uni.insert(t.as_str()) {
                q_uni.push(t.as_str());
            }
        }
        let q_slots: Vec<u32> = match (side_ns.as_ref(), mask) {
            (Some(ns), Some(m)) => {
                let mut seen: HashSet<u32> = HashSet::new();
                let mut out: Vec<u32> = Vec::new();
                for (slot, _) in crate::ngram::iter_ngram_slots_in(&q_tokens, ns, m) {
                    if seen.insert(slot) {
                        out.push(slot);
                    }
                }
                out
            }
            _ => Vec::new(),
        };

        let params = ScoringParams {
            k1: self.k1,
            b: self.b,
            delta: self.delta,
            avgdl,
        };

        // Score each document. Unigram contribution gated on `score_unigram`
        // so e.g. `ngrams=[2]` produces a pure-bigram score even when the
        // unigram tier is built (it is — for query infra parity with the
        // search() path).
        let score_uni = self.score_unigram;
        let n_u32 = n as u32;
        let mut scores = Vec::with_capacity(n);
        for (i, (_, uni_tf, slot_tf)) in tokenized.iter().enumerate() {
            let dl = doc_lens[i];
            let mut total = 0.0f32;
            if score_uni {
                for &qt in &q_uni {
                    if let Some(&tf) = uni_tf.get(qt) {
                        let df = *uni_df.get(qt).unwrap_or(&0);
                        let idf_val = scoring::idf(self.method, n_u32, df);
                        total += scoring::score(self.method, tf, dl, &params, idf_val);
                    }
                }
            }
            for &qs in &q_slots {
                if let Some(&tf) = slot_tf.get(&qs) {
                    let df = *slot_df.get(&qs).unwrap_or(&0);
                    let idf_val = scoring::idf(self.method, n_u32, df);
                    total += scoring::score(self.method, tf, dl, &params, idf_val);
                }
            }
            scores.push(total);
        }
        scores
    }

    /// Score multiple queries against their respective document lists.
    /// `queries[i]` is scored against `documents[i]`.
    pub fn score_batch(&self, queries: &[&str], documents: &[&[&str]]) -> Vec<Vec<f32>> {
        let pool = capped_pool();
        pool.install(|| {
            queries
                .par_iter()
                .zip(documents.par_iter())
                .map(|(q, docs)| self.score(q, docs))
                .collect()
        })
    }

    // --- Internal helpers ---

    /// Tokenize a text string using the index's tokenizer.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        self.tokenizer.tokenize_owned(text)
    }

    /// Stem a single token using the index's tokenizer (lowercase + normalize + stem).
    ///
    /// Unlike `tokenize()`, this does NOT split on non-alphanumeric boundaries
    /// or filter stopwords. Used by the query rewrite layer for wildcard prefixes.
    pub fn stem_token(&self, token: &str) -> String {
        self.tokenizer.stem_single(token)
    }

    /// Look up a token in the vocabulary, returning its term_id.
    pub fn get_term_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn get_or_create_term(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.vocab.get(token) {
            id
        } else {
            let id = self.postings.len() as u32;
            self.vocab.insert(token.to_string(), id);
            self.postings.push(Vec::new());
            self.positions.push(Vec::new());
            self.doc_freqs.push(0);
            id
        }
    }

    /// Get document length for a doc_id.
    pub(crate) fn get_doc_length(&self, doc_id: u32) -> u32 {
        if let Some(ref mmap) = self.mmap_data {
            mmap.get_doc_length(doc_id)
        } else {
            *self.doc_lengths.get(doc_id as usize).unwrap_or(&0)
        }
    }

    /// Iterate over postings for a term.
    pub(crate) fn for_each_posting<F: FnMut(u32, u32)>(&self, term_id: u32, mut f: F) {
        if let Some(ref mmap) = self.mmap_data {
            mmap.for_each_posting(term_id, &mut f);
        } else if let Some(postings) = self.postings.get(term_id as usize) {
            for &(doc_id, tf) in postings {
                f(doc_id, tf);
            }
        }
    }

    /// Look up term frequency for a specific (term_id, doc_id) via binary search.
    #[inline]
    fn get_tf(&self, term_id: u32, doc_id: u32) -> Option<u32> {
        if let Some(ref mmap) = self.mmap_data {
            mmap.get_tf(term_id, doc_id)
        } else if let Some(postings) = self.postings.get(term_id as usize) {
            postings
                .binary_search_by_key(&doc_id, |&(did, _)| did)
                .ok()
                .map(|idx| postings[idx].1)
        } else {
            None
        }
    }

    /// Convert mmap-backed data to in-memory vectors (needed before mutation).
    fn materialize_mmap(&mut self) {
        if let Some(mmap) = self.mmap_data.take() {
            let num_terms = self.vocab.len();
            self.postings = Vec::with_capacity(num_terms);
            for term_id in 0..num_terms as u32 {
                let mut entries = Vec::new();
                mmap.for_each_posting(term_id, &mut |doc_id, tf| {
                    entries.push((doc_id, tf));
                });
                self.postings.push(entries);
            }
            // Positions are not stored in v1 mmap format — initialize empty.
            self.positions = vec![Vec::new(); num_terms];
            self.doc_lengths = mmap.all_doc_lengths();
        }
    }

    // --- Accessors for storage module ---

    pub(crate) fn get_postings(&self) -> &Vec<Vec<(u32, u32)>> {
        &self.postings
    }

    pub(crate) fn get_positions(&self) -> &Vec<Vec<Vec<u32>>> {
        &self.positions
    }

    pub(crate) fn get_doc_lengths_slice(&self) -> &[u32] {
        &self.doc_lengths
    }

    pub(crate) fn get_mmap_data(&self) -> Option<&MmapData> {
        self.mmap_data.as_ref()
    }

    pub fn get_vocab(&self) -> &FxHashMap<String, u32> {
        &self.vocab
    }

    /// Find terms that co-occur with `term` in the same documents.
    ///
    /// Returns up to `limit` terms sorted by co-occurrence count (descending),
    /// excluding the query term itself. Each entry is (term_string, cooccurrence_count, df).
    pub fn cooccurring_terms(&self, term: &str, limit: usize) -> Vec<(String, u32, u32)> {
        let tokens = self.tokenize(term);
        let token = match tokens.first() {
            Some(t) => t,
            None => return Vec::new(),
        };
        let term_id = match self.get_term_id(token) {
            Some(id) => id,
            None => return Vec::new(),
        };

        // Collect doc_ids where the query term appears
        let mut target_docs = HashSet::new();
        self.for_each_posting(term_id, |doc_id, _tf| {
            target_docs.insert(doc_id);
        });

        if target_docs.is_empty() {
            return Vec::new();
        }

        // Count co-occurrences in parallel across all other terms.
        // Each term's posting list is independent so this is trivially safe.
        let num_terms = self.doc_freqs.len();
        let other_ids: Vec<u32> = (0..num_terms as u32)
            .filter(|&id| id != term_id)
            .collect();

        let pool = capped_pool();
        let mut counts: Vec<(u32, u32)> = pool.install(|| {
            other_ids
                .par_iter()
                .filter_map(|&other_id| {
                    let mut count = 0u32;
                    self.for_each_posting(other_id, |doc_id, _tf| {
                        if target_docs.contains(&doc_id) {
                            count += 1;
                        }
                    });
                    if count > 0 {
                        Some((other_id, count))
                    } else {
                        None
                    }
                })
                .collect()
        });

        // Sort by co-occurrence count descending, keep top `limit`
        counts.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        counts.truncate(limit);

        // Resolve term_id -> term_string only for the final results.
        // Building a full reverse-vocab HashMap would be wasteful; instead look up
        // each winning term_id directly by scanning the vocab (limit is small).
        let result_ids: HashSet<u32> = counts.iter().map(|&(tid, _)| tid).collect();
        let id_to_term: HashMap<u32, &str> = self.vocab.iter()
            .filter(|(_, &id)| result_ids.contains(&id))
            .map(|(term, &id)| (id, term.as_str()))
            .collect();

        counts.iter()
            .filter_map(|&(tid, count)| {
                id_to_term.get(&tid).map(|&term_str| {
                    (term_str.to_string(), count, self.doc_freqs[tid as usize])
                })
            })
            .collect()
    }

    /// Resolve a raw term string to its term_id after tokenization + stemming.
    ///
    /// Tokenizes the input (which may split multi-word strings) and looks up
    /// the first resulting token in the vocabulary.
    pub fn resolve_term_id(&self, term: &str) -> Option<u32> {
        let tokens = self.tokenize(term);
        let token = tokens.first()?;
        self.get_term_id(token)
    }

    /// Get the IDF score for a term, using the index's scoring method.
    ///
    /// Tokenizes and stems the input, then computes IDF using the same formula
    /// as search (varies by method: Lucene, Robertson, Atire, BM25L, BM25Plus).
    /// Returns None if the term is not in the vocabulary.
    pub fn get_term_idf(&self, term: &str) -> Option<f32> {
        let term_id = self.resolve_term_id(term)?;
        let df = self.doc_freqs[term_id as usize];
        Some(scoring::idf(self.method, self.num_docs, df))
    }

    /// Filter expansion phrases for query enrichment.
    ///
    /// For each phrase, tokenize+stem, then generate all sliding-window
    /// n-grams (1..=max_n). A phrase is **kept** if at least one n-gram
    /// has 0 < DF <= max_df in the BM25 index:
    ///   - n=1: exact unigram DF from `doc_freqs`
    ///   - n>=2: hashed n-gram DF from `ngram_side`
    ///
    /// This mirrors the doc enrichment filter in `NGramIndex::filter_candidates`:
    /// a phrase adds value iff any sub-n-gram is informative (present but
    /// not too common).
    ///
    /// Returns `(kept_phrases, rejected)` where `kept_phrases` are original
    /// phrase strings (pass to `search_with_expansion` after joining stems).
    /// `rejected` is a list of `(phrase, reason)` pairs for tracing.
    pub fn filter_query_expansion(
        &self,
        _query: &str,
        phrases: &[String],
        max_df: u32,
    ) -> (Vec<String>, Vec<(String, String)>) {
        let max_n = self.max_n() as usize;
        let mut kept: Vec<String> = Vec::new();
        let mut rejected: Vec<(String, String)> = Vec::new();

        for phrase in phrases {
            let stems = self.tokenizer.tokenize_owned(phrase);
            if stems.is_empty() {
                rejected.push((phrase.clone(), "no_stems".to_string()));
                continue;
            }

            let upper = max_n.min(stems.len());
            let mut has_rare = false;

            'outer: for n in 1..=upper {
                for window in stems.windows(n) {
                    let df = if n == 1 {
                        self.get_term_id(&window[0])
                            .map(|tid| self.doc_freqs[tid as usize])
                            .unwrap_or(0)
                    } else {
                        let key = window.join(" ");
                        self.ngram_df(&key)
                    };
                    if df > 0 && (max_df == 0 || df <= max_df) {
                        has_rare = true;
                        break 'outer;
                    }
                }
            }

            if has_rare {
                kept.push(phrase.clone());
            } else {
                rejected.push((phrase.clone(), "too_common_or_unseen".to_string()));
            }
        }
        (kept, rejected)
    }

    /// Batch enrich: apply multiple phrases to multiple documents in one call.
    ///
    /// `items` is a list of `(doc_id, phrases)` pairs. Each phrase is
    /// enriched via sliding-window n-gram insertion (same as `self.enrich()`).
    ///
    /// Uses four-phase parallelism:
    /// 1. Tokenization runs in parallel via rayon (the expensive stemming work).
    /// 2. Position offsets are computed serially (they depend on cumulative doc lengths).
    /// 3. TF+position map construction runs in parallel.
    /// 4. Posting-list merges run in parallel across term_ids: updates are
    ///    reorganized by term, then each term's postings/positions/df are
    ///    merged independently. N-gram side and doc_lengths update serially.
    pub fn enrich_batch(
        &mut self,
        items: &[(usize, Vec<String>)],
    ) -> io::Result<()> {
        if items.is_empty() {
            return Ok(());
        }

        if self.mmap_data.is_some() {
            self.materialize_mmap();
        }

        let t_batch = std::time::Instant::now();

        // --- Phase 1: parallel tokenization ---
        // Flatten items into (doc_id, phrase) pairs, preserving order within
        // each doc so that position offsets remain deterministic.
        let flat: Vec<(usize, &str)> = items
            .iter()
            .flat_map(|(doc_id, phrases)| {
                phrases.iter().map(move |p| (*doc_id, p.as_str()))
            })
            .collect();

        let pool = capped_pool();
        let tokenizer = &self.tokenizer;
        let tokenized: Vec<(usize, Vec<String>)> = pool.install(|| {
            flat.par_iter()
                .map(|&(doc_id, phrase)| {
                    let tokens = tokenizer.tokenize_owned(phrase);
                    (doc_id, tokens)
                })
                .collect()
        });
        let t_phase1 = t_batch.elapsed();

        // --- Phase 2: compute per-item position offsets (serial) ---
        // Each phrase's pos_offset = doc_lengths[doc_id] + sum of token counts
        // from all earlier phrases for the same doc_id in this batch.
        let mut extra_per_doc: HashMap<usize, u32> = HashMap::new();
        // (index_into_tokenized, doc_id_u32, pos_offset, extra_len)
        let mut work: Vec<(usize, u32, u32, u32)> = Vec::with_capacity(tokenized.len());

        for (idx, (doc_id, tokens)) in tokenized.iter().enumerate() {
            let id = *doc_id as u32;
            assert!(
                id < self.num_docs,
                "doc_id {} out of range (num_docs={})",
                doc_id,
                self.num_docs
            );

            let extra_len = tokens.len() as u32;
            if extra_len == 0 {
                continue;
            }

            let cumulative_extra = extra_per_doc.entry(*doc_id).or_insert(0);
            let pos_offset = self.doc_lengths[*doc_id] + *cumulative_extra;
            *cumulative_extra += extra_len;

            work.push((idx, id, pos_offset, extra_len));
        }
        let t_phase2 = t_batch.elapsed();

        // --- Phase 3+4a: resolve tokens → term_ids, counting-sort by term_id ---
        let max_n = self.max_n;
        let total_tokens_est: usize = work.iter().map(|&(ti, _, _, _)| tokenized[ti].1.len()).sum();

        // Pass 1: resolve term_ids, collect flat (term_id, doc_id, pos), count per term
        let mut raw: Vec<(u32, u32, u32)> = Vec::with_capacity(total_tokens_est);

        for &(tok_idx, id, pos_offset, extra_len) in &work {
            let tokens = &tokenized[tok_idx].1;
            for (i, token) in tokens.iter().enumerate() {
                let term_id = self.get_or_create_term(token);
                raw.push((term_id, id, pos_offset + i as u32));
            }
            if max_n > 1 {
                if let Some(side) = self.ngram_side.as_mut() {
                    side.enrich_doc(id, tokens, self.doc_lengths[id as usize]);
                }
            }
            self.doc_lengths[id as usize] += extra_len;
            self.total_tokens += extra_len as u64;
        }

        // Counting sort by term_id: O(n) instead of O(n log n)
        let n_terms = self.postings.len();
        let mut counts = vec![0u32; n_terms];
        for &(tid, _, _) in &raw {
            counts[tid as usize] += 1;
        }
        let mut offsets = vec![0u32; n_terms + 1];
        for i in 0..n_terms {
            offsets[i + 1] = offsets[i] + counts[i];
        }
        // Place into sorted array; within each term bucket, doc_id order is
        // preserved from the work iteration (stable by insertion order).
        let mut sorted = vec![(0u32, 0u32); raw.len()];
        let mut write_pos = offsets.clone();
        for (tid, doc_id, pos) in raw {
            let idx = write_pos[tid as usize] as usize;
            sorted[idx] = (doc_id, pos);
            write_pos[tid as usize] += 1;
        }

        let t_phase4a = t_batch.elapsed();

        // --- Phase 4b: merge each term's bucket into its posting list ---
        for tid in 0..n_terms {
            let start = offsets[tid] as usize;
            let end = offsets[tid + 1] as usize;
            if start == end {
                continue;
            }
            let group = &sorted[start..end];
            // group entries are (doc_id, pos), in work-iteration order.
            // Need to sort by doc_id for the merge. Within each term bucket,
            // doc_ids from different work items may be interleaved.
            // Use a local sort only on this small slice.

            let old_postings = std::mem::take(&mut self.postings[tid]);
            let mut old_positions = if tid < self.positions.len() {
                std::mem::take(&mut self.positions[tid])
            } else {
                Vec::new()
            };
            let has_positions = !old_positions.is_empty();

            // Build sorted (doc_id, tf, positions) from the group
            // Since group may not be sorted by doc_id, sort a temp index
            let mut group_sorted: Vec<(u32, u32)> = group.to_vec();
            group_sorted.sort_unstable_by_key(|&(doc_id, _)| doc_id);

            let mut new_postings: Vec<(u32, u32)> = Vec::with_capacity(old_postings.len() + group_sorted.len());
            let mut new_positions: Vec<Vec<u32>> = if has_positions {
                Vec::with_capacity(old_postings.len() + group_sorted.len())
            } else {
                Vec::new()
            };

            let mut oi = 0;
            let mut gi = 0;

            while oi < old_postings.len() && gi < group_sorted.len() {
                let old_doc = old_postings[oi].0;
                let upd_doc = group_sorted[gi].0;
                match old_doc.cmp(&upd_doc) {
                    std::cmp::Ordering::Less => {
                        new_postings.push(old_postings[oi]);
                        if has_positions {
                            new_positions.push(
                                if oi < old_positions.len() { std::mem::take(&mut old_positions[oi]) } else { Vec::new() }
                            );
                        }
                        oi += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        let mut tf = 0u32;
                        let mut pos = Vec::new();
                        while gi < group_sorted.len() && group_sorted[gi].0 == upd_doc {
                            tf += 1;
                            if has_positions { pos.push(group_sorted[gi].1); }
                            gi += 1;
                        }
                        new_postings.push((upd_doc, tf));
                        if has_positions { new_positions.push(pos); }
                        self.doc_freqs[tid] += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        let mut tf = old_postings[oi].1;
                        let mut pos = if has_positions {
                            if oi < old_positions.len() { std::mem::take(&mut old_positions[oi]) } else { Vec::new() }
                        } else { Vec::new() };
                        while gi < group_sorted.len() && group_sorted[gi].0 == old_doc {
                            tf += 1;
                            if has_positions { pos.push(group_sorted[gi].1); }
                            gi += 1;
                        }
                        new_postings.push((old_doc, tf));
                        if has_positions { new_positions.push(pos); }
                        oi += 1;
                    }
                }
            }
            while oi < old_postings.len() {
                new_postings.push(old_postings[oi]);
                if has_positions {
                    new_positions.push(
                        if oi < old_positions.len() { std::mem::take(&mut old_positions[oi]) } else { Vec::new() }
                    );
                }
                oi += 1;
            }
            while gi < group_sorted.len() {
                let upd_doc = group_sorted[gi].0;
                let mut tf = 0u32;
                let mut pos = Vec::new();
                while gi < group_sorted.len() && group_sorted[gi].0 == upd_doc {
                    tf += 1;
                    if has_positions { pos.push(group_sorted[gi].1); }
                    gi += 1;
                }
                new_postings.push((upd_doc, tf));
                if has_positions { new_positions.push(pos); }
                self.doc_freqs[tid] += 1;
            }

            self.postings[tid] = new_postings;
            if has_positions && tid < self.positions.len() {
                self.positions[tid] = new_positions;
            }
        }

        let t_phase4 = t_batch.elapsed();
        eprintln!(
            "[enrich_batch] {} items  P1(tokenize)={:.0}ms  P2(offsets)={:.0}ms  \
             P4a(resolve+sort)={:.0}ms  P4b(merge)={:.0}ms  total={:.0}ms",
            flat.len(),
            t_phase1.as_secs_f64() * 1e3,
            (t_phase2 - t_phase1).as_secs_f64() * 1e3,
            (t_phase4a - t_phase2).as_secs_f64() * 1e3,
            (t_phase4 - t_phase4a).as_secs_f64() * 1e3,
            t_phase4.as_secs_f64() * 1e3,
        );

        self.auto_save()
    }

    /// Batch version of [`Self::enrich_exact`].
    ///
    /// `items` is a list of `(doc_id, phrases)` pairs. Each phrase is
    /// passed to `self.enrich_exact()` (single n-gram slot insertion).
    pub fn enrich_exact_batch(
        &mut self,
        items: &[(usize, Vec<String>)],
    ) -> io::Result<()> {
        for (doc_id, phrases) in items {
            for phrase in phrases {
                self.enrich_exact(*doc_id, phrase)?;
            }
        }
        Ok(())
    }

    /// Two-pass search with weighted expansion scoring.
    ///
    /// For each (query, expansion) pair:
    /// 1. Score the original query → base scores
    /// 2. Score the expansion terms → expansion scores
    /// 3. Merge: final_score = base + weight * expansion
    /// 4. Return top-k by merged score
    ///
    /// `expansion_terms[i]` is a space-separated string of stemmed tokens.
    /// Empty strings mean no expansion for that query.
    pub fn search_with_expansion(
        &self,
        queries: &[&str],
        expansion_terms: &[&str],
        k: usize,
        weight: f32,
    ) -> Vec<Vec<SearchResult>> {
        assert_eq!(queries.len(), expansion_terms.len());
        if self.num_docs == 0 {
            return queries.iter().map(|_| Vec::new()).collect();
        }

        let pool = capped_pool();
        let n_docs = self.num_docs as usize;

        pool.install(|| {
            queries.par_iter().zip(expansion_terms.par_iter()).map(|(&query, &exp)| {
                // Pass 1: original query
                let (uni_base, ng_base) = self.query_targets(query);
                let params = ScoringParams {
                    k1: self.k1, b: self.b, delta: self.delta,
                    avgdl: self.total_tokens as f32 / self.num_docs as f32,
                };

                with_score_buf(n_docs, |scores| {
                    let mut touched: Vec<u32> = Vec::new();

                    if self.score_unigram {
                        for tid in &uni_base {
                            let df = self.doc_freqs.get(*tid as usize).copied().unwrap_or(0);
                            if df == 0 { continue; }
                            let idf_val = scoring::idf(self.method, self.num_docs, df);
                            self.for_each_posting(*tid, |doc_id, tf| {
                                let dl = self.get_doc_length(doc_id);
                                let s = scoring::score(self.method, tf, dl, &params, idf_val);
                                let i = doc_id as usize;
                                if scores[i] == 0.0 { touched.push(doc_id); }
                                scores[i] += s;
                            });
                        }
                    }
                    if let Some(side) = self.ngram_side.as_ref() {
                        for slot in &ng_base {
                            let df = side.df(*slot);
                            if df == 0 { continue; }
                            let idf_val = scoring::idf(self.method, self.num_docs, df);
                            side.for_each_posting(*slot, |doc_id, tf| {
                                let dl = self.get_doc_length(doc_id);
                                let s = scoring::score(self.method, tf, dl, &params, idf_val);
                                let i = doc_id as usize;
                                if scores[i] == 0.0 { touched.push(doc_id); }
                                scores[i] += s;
                            });
                        }
                    }

                    if !exp.is_empty() && weight > 0.0 {
                        let (uni_exp, ng_exp) = self.query_targets(exp);
                        if self.score_unigram {
                            for tid in &uni_exp {
                                let df = self.doc_freqs.get(*tid as usize).copied().unwrap_or(0);
                                if df == 0 { continue; }
                                let idf_val = scoring::idf(self.method, self.num_docs, df);
                                self.for_each_posting(*tid, |doc_id, tf| {
                                    let dl = self.get_doc_length(doc_id);
                                    let s = scoring::score(self.method, tf, dl, &params, idf_val);
                                    let i = doc_id as usize;
                                    if scores[i] == 0.0 { touched.push(doc_id); }
                                    scores[i] += weight * s;
                                });
                            }
                        }
                        if let Some(side) = self.ngram_side.as_ref() {
                            for slot in &ng_exp {
                                let df = side.df(*slot);
                                if df == 0 { continue; }
                                let idf_val = scoring::idf(self.method, self.num_docs, df);
                                side.for_each_posting(*slot, |doc_id, tf| {
                                    let dl = self.get_doc_length(doc_id);
                                    let s = scoring::score(self.method, tf, dl, &params, idf_val);
                                    let i = doc_id as usize;
                                    if scores[i] == 0.0 { touched.push(doc_id); }
                                    scores[i] += weight * s;
                                });
                            }
                        }
                    }

                    let result = Self::topk_from_scores(scores, &touched, k);
                    for &doc_id in &touched {
                        scores[doc_id as usize] = 0.0;
                    }
                    result
                })
            }).collect()
        })
    }

    pub(crate) fn get_total_tokens(&self) -> u64 {
        self.total_tokens
    }

    pub fn get_num_docs(&self) -> u32 {
        self.num_docs
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn set_internals(
        &mut self,
        vocab: FxHashMap<String, u32>,
        doc_lengths: Vec<u32>,
        postings: Vec<Vec<(u32, u32)>>,
        positions: Vec<Vec<Vec<u32>>>,
        total_tokens: u64,
        num_docs: u32,
    ) {
        self.doc_freqs = postings.iter().map(|p| p.len() as u32).collect();
        self.positions = positions;
        self.vocab = vocab;
        self.doc_lengths = doc_lengths;
        self.postings = postings;
        self.total_tokens = total_tokens;
        self.num_docs = num_docs;
    }

    pub(crate) fn set_mmap_internals(
        &mut self,
        vocab: FxHashMap<String, u32>,
        total_tokens: u64,
        num_docs: u32,
        mmap_data: MmapData,
    ) {
        let num_terms = vocab.len() as u32;
        self.doc_freqs = (0..num_terms).map(|t| mmap_data.posting_count(t)).collect();
        self.positions = vec![Vec::new(); num_terms as usize];
        self.vocab = vocab;
        self.total_tokens = total_tokens;
        self.num_docs = num_docs;
        self.mmap_data = Some(mmap_data);
    }

    /// Build the combined flat term-id → posting layout for GPU upload.
    ///
    /// Term IDs `[0, vocab.len())` are unigram postings (verbatim from
    /// `self.postings`, or materialized from `mmap` for mmap-backed indices);
    /// `[vocab.len(), vocab.len() + n_features)` are n-gram slot postings from
    /// `ngram_side` (always in-memory — mmap doesn't store the n-gram tier as
    /// of storage v1). When `ngram_side` is `None`, returns just the unigram
    /// postings (zero overhead beyond the unigram materialization).
    ///
    /// Kept centralized so both `to_gpu_search_index` and
    /// `to_multi_gpu_search_index` agree byte-for-byte on the upload layout —
    /// must stay in sync with the offset-encoding scheme used by
    /// `tokenize_single` / `tokenize_queries` (n-gram slot `s` → GPU term ID
    /// `vocab.len() + s`).
    #[cfg(feature = "cuda")]
    pub(crate) fn combined_postings_for_gpu(&self) -> Vec<Vec<(u32, u32)>> {
        let mut combined: Vec<Vec<(u32, u32)>> = if let Some(ref mmap) = self.mmap_data {
            (0..self.vocab.len() as u32)
                .map(|t| {
                    let mut entries = Vec::new();
                    mmap.for_each_posting(t, &mut |doc_id, tf| entries.push((doc_id, tf)));
                    entries
                })
                .collect()
        } else {
            self.postings.clone()
        };
        if let Some(side) = self.ngram_side.as_ref() {
            combined.reserve(side.postings.len());
            combined.extend(side.postings.iter().cloned());
        }
        combined
    }

    /// Create a multi-GPU search index. Replicates the index across all available GPUs.
    ///
    /// When `max_n >= 2`, the upload uses the combined uni+ngram flat layout
    /// (see [`Self::combined_postings_for_gpu`]): unigram postings occupy term
    /// IDs `[0, vocab.len())` and n-gram side slots occupy
    /// `[vocab.len(), vocab.len() + n_features)`. Kernels are
    /// term-identity-agnostic; no GPU code change is required.
    #[cfg(feature = "cuda")]
    pub fn to_multi_gpu_search_index(
        &self,
    ) -> Result<crate::multi_gpu::MultiGpuSearchIndex, String> {
        let combined = self.combined_postings_for_gpu();
        let doc_lengths = if let Some(ref mmap) = self.mmap_data {
            mmap.all_doc_lengths()
        } else {
            self.doc_lengths.clone()
        };
        crate::multi_gpu::MultiGpuSearchIndex::from_index(&combined, &doc_lengths, self.num_docs)
    }

    /// Create a GPU search index for fast search. Call once after indexing.
    ///
    /// When `max_n >= 2`, the upload uses the combined uni+ngram flat layout
    /// (see [`Self::combined_postings_for_gpu`]): unigram postings occupy term
    /// IDs `[0, vocab.len())` and n-gram side slots occupy
    /// `[vocab.len(), vocab.len() + n_features)`. Kernels are
    /// term-identity-agnostic; no GPU code change is required.
    #[cfg(feature = "cuda")]
    pub fn to_gpu_search_index(&self) -> Result<crate::cuda::GpuSearchIndex, String> {
        let ctx =
            crate::cuda::get_global_context().ok_or_else(|| "CUDA not available".to_string())?;

        let combined = self.combined_postings_for_gpu();
        let doc_lengths = if let Some(ref mmap) = self.mmap_data {
            mmap.all_doc_lengths()
        } else {
            self.doc_lengths.clone()
        };

        crate::cuda::GpuSearchIndex::from_index(&ctx, &combined, &doc_lengths, self.num_docs)
    }

    /// Search using GPU-resident index. Much faster than CPU search for large indices.
    #[cfg(feature = "cuda")]
    pub fn search_gpu(
        &self,
        gpu_index: &mut crate::cuda::GpuSearchIndex,
        query: &str,
        k: usize,
    ) -> Vec<SearchResult> {
        if self.num_docs == 0 {
            return Vec::new();
        }

        let ctx = match crate::cuda::get_global_context() {
            Some(c) => c,
            None => return self.search(query, k),
        };

        let params = ScoringParams {
            k1: self.k1,
            b: self.b,
            delta: self.delta,
            avgdl: self.total_tokens as f32 / self.num_docs as f32,
        };

        // Build (term_id, idf) pairs covering BOTH tiers via the offset-encoded
        // term-ID space (unigrams in `[0, vocab.len())`, n-gram slots offset by
        // `vocab.len()`). Pairs with `combined_postings` on the upload side.
        let query_terms = self.tokenize_single(query);

        match gpu_index.search(&ctx, &query_terms, params.k1, params.b, params.avgdl, k) {
            Ok(results) => results
                .into_iter()
                .map(|(doc_id, score)| SearchResult {
                    index: doc_id as usize,
                    score,
                })
                .collect(),
            Err(e) => {
                eprintln!("[bm25x] GPU search failed: {}, falling back to CPU", e);
                self.search(query, k)
            }
        }
    }
}

impl Default for BM25 {
    fn default() -> Self {
        Self::new(Method::Lucene, 1.5, 0.75, 0.5, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_search() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        let ids = index
            .add(&[
                "the quick brown fox jumps over the lazy dog",
                "a fast brown car drives over the bridge",
                "the fox is quick and clever",
            ])
            .unwrap();
        assert_eq!(ids, vec![0, 1, 2]);
        assert_eq!(index.len(), 3);

        let results = index.search("quick fox", 10);
        assert!(!results.is_empty());
        assert!(results[0].index == 0 || results[0].index == 2);
    }

    #[test]
    fn test_delete_compacts() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&["hello world", "foo bar", "hello foo"]).unwrap();
        assert_eq!(index.len(), 3);

        // Delete doc 0 ("hello world")
        index.delete(&[0]).unwrap();
        assert_eq!(index.len(), 2);

        // Old doc 1 ("foo bar") is now doc 0
        // Old doc 2 ("hello foo") is now doc 1
        let results = index.search("foo", 10);
        assert_eq!(results.len(), 2);
        let ids: Vec<usize> = results.iter().map(|r| r.index).collect();
        assert!(ids.contains(&0)); // was "foo bar"
        assert!(ids.contains(&1)); // was "hello foo"
    }

    #[test]
    fn test_delete_middle() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&["alpha", "beta", "gamma", "delta"]).unwrap();

        // Delete doc 1 ("beta"): [alpha, gamma, delta]
        index.delete(&[1]).unwrap();
        assert_eq!(index.len(), 3);

        let results = index.search("gamma", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].index, 1); // gamma shifted from 2 to 1

        let results = index.search("delta", 10);
        assert_eq!(results[0].index, 2); // delta shifted from 3 to 2
    }

    #[test]
    fn test_delete_multiple() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&["a", "b", "c", "d", "e"]).unwrap();

        // Delete docs 1 and 3: [a, c, e]
        index.delete(&[1, 3]).unwrap();
        assert_eq!(index.len(), 3);

        let results = index.search("c", 10);
        assert_eq!(results[0].index, 1); // c shifted from 2 to 1

        let results = index.search("e", 10);
        assert_eq!(results[0].index, 2); // e shifted from 4 to 2
    }

    #[test]
    fn test_update() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&["hello world", "foo bar"]).unwrap();

        index.update(0, "goodbye universe").unwrap();

        let results = index.search("hello", 10);
        assert!(results.is_empty() || results.iter().all(|r| r.index != 0));

        let results = index.search("goodbye", 10);
        assert!(!results.is_empty());
        assert_eq!(results[0].index, 0);
    }

    #[test]
    fn test_empty_search() {
        let index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        let results = index.search("anything", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_streaming_add() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        let ids1 = index.add(&["first document"]).unwrap();
        let ids2 = index.add(&["second document"]).unwrap();
        assert_eq!(ids1, vec![0]);
        assert_eq!(ids2, vec![1]);
        assert_eq!(index.len(), 2);

        let results = index.search("first", 10);
        assert_eq!(results[0].index, 0);
    }

    #[test]
    fn test_all_methods() {
        for method in [
            Method::Lucene,
            Method::Robertson,
            Method::Atire,
            Method::BM25L,
            Method::BM25Plus,
        ] {
            let mut index = BM25::new(method, 1.5, 0.75, 0.5, false);
            index
                .add(&[
                    "the cat sat on the mat",
                    "the dog played in the park",
                    "birds fly over the river",
                    "fish swim in the ocean",
                ])
                .unwrap();
            let results = index.search("cat mat", 10);
            assert!(!results.is_empty(), "{:?} returned no results", method);
            assert_eq!(results[0].index, 0, "{:?} ranked wrong doc first", method);
        }
    }

    #[test]
    fn test_search_filtered_basic() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index
            .add(&[
                "the quick brown fox",  // 0
                "the lazy brown dog",   // 1
                "the quick red car",    // 2
                "the slow brown truck", // 3
            ])
            .unwrap();

        let results = index.search("brown", 10);
        assert_eq!(results.len(), 3);

        let results = index.search_filtered("brown", 10, &[1, 3]);
        assert_eq!(results.len(), 2);
        let ids: Vec<usize> = results.iter().map(|r| r.index).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
        assert!(!ids.contains(&0));
    }

    #[test]
    fn test_search_filtered_respects_k() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index
            .add(&[
                "apple banana cherry",   // 0
                "apple date elderberry", // 1
                "apple fig grape",       // 2
                "apple hazelnut ice",    // 3
                "apple jackfruit kiwi",  // 4
            ])
            .unwrap();

        let results = index.search_filtered("apple", 2, &[0, 1, 2, 3]);
        assert_eq!(results.len(), 2);
        for r in &results {
            assert!(r.index <= 3);
        }
    }

    #[test]
    fn test_search_filtered_empty_filter() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&["hello world", "foo bar"]).unwrap();

        let results = index.search_filtered("hello", 10, &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_filtered_no_overlap() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index
            .add(&[
                "the quick brown fox", // 0
                "the lazy dog",        // 1
            ])
            .unwrap();

        let results = index.search_filtered("fox", 10, &[1]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_filtered_scores_match_unfiltered() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index
            .add(&[
                "rust is fast and safe",    // 0
                "python is slow but easy",  // 1
                "rust and python together", // 2
            ])
            .unwrap();

        let unfiltered = index.search("rust", 10);
        let filtered = index.search_filtered("rust", 10, &[0]);

        let score_unfiltered = unfiltered.iter().find(|r| r.index == 0).unwrap().score;
        let score_filtered = filtered.iter().find(|r| r.index == 0).unwrap().score;
        assert!(
            (score_unfiltered - score_filtered).abs() < 1e-6,
            "scores differ: {} vs {}",
            score_unfiltered,
            score_filtered
        );
    }

    #[test]
    fn test_delete_then_add() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&["alpha", "beta", "gamma"]).unwrap();
        index.delete(&[1]).unwrap(); // remove "beta", now [alpha, gamma]
        assert_eq!(index.len(), 2);

        let ids = index.add(&["delta"]).unwrap();
        assert_eq!(ids, vec![2]); // appended at end
        assert_eq!(index.len(), 3);

        let results = index.search("delta", 10);
        assert_eq!(results[0].index, 2);
    }

    #[test]
    fn test_score_basic() {
        let index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        let scores = index.score("fox", &["the quick brown fox", "lazy dog", "fox fox fox"]);
        assert_eq!(scores.len(), 3);
        assert!(scores[0] > 0.0); // "fox" appears
        assert_eq!(scores[1], 0.0); // no match
        assert!(scores[2] > scores[0]); // more "fox" occurrences
    }

    #[test]
    fn test_score_empty() {
        let index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        assert!(index.score("hello", &[]).is_empty());
    }

    #[test]
    fn test_score_batch() {
        let index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        let docs1: &[&str] = &["the cat", "the dog"];
        let docs2: &[&str] = &["rust lang", "python lang", "go lang"];
        let results = index.score_batch(&["cat", "rust"], &[docs1, docs2]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 2);
        assert_eq!(results[1].len(), 3);
        assert!(results[0][0] > 0.0); // "cat" matches "the cat"
        assert!(results[1][0] > 0.0); // "rust" matches "rust lang"
    }

    /// Helper: assert score() and index+search produce identical scores.
    fn assert_scores_match(
        method: Method,
        k1: f32,
        b: f32,
        delta: f32,
        use_stopwords: bool,
        query: &str,
        docs: &[&str],
    ) {
        let scorer = BM25::new(method, k1, b, delta, use_stopwords);
        let direct = scorer.score(query, docs);

        let mut index = BM25::new(method, k1, b, delta, use_stopwords);
        index.add(docs).unwrap();

        // Check every document — including zero-scoring ones
        for (i, &ds) in direct.iter().enumerate() {
            let indexed_score = index
                .search(query, docs.len())
                .iter()
                .find(|r| r.index == i)
                .map(|r| r.score)
                .unwrap_or(0.0);
            assert!(
                (ds - indexed_score).abs() < 1e-6,
                "{:?} doc {}: score()={} != search()={} (query={:?})",
                method,
                i,
                ds,
                indexed_score,
                query
            );
        }
    }

    #[test]
    fn test_score_matches_search_lucene() {
        assert_scores_match(
            Method::Lucene,
            1.5,
            0.75,
            0.5,
            false,
            "beta gamma",
            &[
                "alpha beta gamma",
                "beta gamma delta",
                "gamma delta epsilon",
            ],
        );
    }

    #[test]
    fn test_score_matches_search_all_methods() {
        let docs = &[
            "the quick brown fox jumps over the lazy dog",
            "a brown dog chased the fox",
            "quick red car on the highway",
            "lazy sleeping cat on the mat",
        ];
        for method in [
            Method::Lucene,
            Method::Robertson,
            Method::Atire,
            Method::BM25L,
            Method::BM25Plus,
        ] {
            assert_scores_match(method, 1.5, 0.75, 0.5, false, "brown fox", docs);
            assert_scores_match(method, 1.5, 0.75, 0.5, false, "lazy", docs);
            assert_scores_match(method, 1.5, 0.75, 0.5, false, "quick brown fox", docs);
            assert_scores_match(method, 1.5, 0.75, 0.5, false, "nonexistent", docs);
        }
    }

    #[test]
    fn test_score_matches_search_with_stopwords() {
        let docs = &[
            "the quick brown fox",
            "a lazy brown dog",
            "the fox is quick and clever",
        ];
        // With stopwords enabled, "the" and "is" are removed
        assert_scores_match(Method::Lucene, 1.5, 0.75, 0.5, true, "the quick fox", docs);
        assert_scores_match(Method::Lucene, 1.5, 0.75, 0.5, true, "brown", docs);
    }

    #[test]
    fn test_score_matches_search_custom_params() {
        let docs = &["rust is fast", "python is easy", "rust and python together"];
        // Different k1, b values
        assert_scores_match(Method::Lucene, 2.0, 0.5, 0.5, false, "rust", docs);
        assert_scores_match(Method::Atire, 1.2, 0.9, 0.5, false, "rust python", docs);
        assert_scores_match(Method::BM25Plus, 1.5, 0.75, 1.0, false, "rust python", docs);
    }

    #[test]
    fn test_score_matches_search_single_doc() {
        assert_scores_match(
            Method::Lucene,
            1.5,
            0.75,
            0.5,
            false,
            "hello",
            &["hello world"],
        );
    }

    #[test]
    fn test_score_matches_search_no_match() {
        let scorer = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        let scores = scorer.score("xyz", &["alpha beta", "gamma delta"]);
        assert!(scores.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_score_matches_search_duplicate_query_terms() {
        // "fox fox" should produce the same score as "fox"
        let docs = &["the quick brown fox", "lazy dog"];
        let scorer = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        let scores_single = scorer.score("fox", docs);
        let scores_dup = scorer.score("fox fox", docs);
        for i in 0..docs.len() {
            assert!(
                (scores_single[i] - scores_dup[i]).abs() < 1e-6,
                "duplicate query terms changed score for doc {}: {} vs {}",
                i,
                scores_single[i],
                scores_dup[i]
            );
        }
    }

    #[test]
    fn test_score_batch_matches_individual() {
        let scorer = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        let docs1: &[&str] = &["the cat sat", "the dog ran"];
        let docs2: &[&str] = &["rust lang", "python lang", "go lang"];

        let batch = scorer.score_batch(&["cat", "rust"], &[docs1, docs2]);
        let individual1 = scorer.score("cat", docs1);
        let individual2 = scorer.score("rust", docs2);

        for i in 0..docs1.len() {
            assert!(
                (batch[0][i] - individual1[i]).abs() < 1e-6,
                "batch[0][{}]={} != individual={}",
                i,
                batch[0][i],
                individual1[i]
            );
        }
        for i in 0..docs2.len() {
            assert!(
                (batch[1][i] - individual2[i]).abs() < 1e-6,
                "batch[1][{}]={} != individual={}",
                i,
                batch[1][i],
                individual2[i]
            );
        }
    }

    #[test]
    fn test_default_config() {
        // Regression guard on the documented defaults:
        // Lucene, k1=1.5, b=0.75, delta=0.5, UnicodeStem, stopwords=true
        let index = BM25::default();
        assert_eq!(index.method, Method::Lucene);
        assert_eq!(index.k1, 1.5);
        assert_eq!(index.b, 0.75);
        assert_eq!(index.delta, 0.5);
    }

    #[test]
    fn test_default_applies_stemming() {
        // Default must stem: "running" → "run"
        let index = BM25::default();
        let scores = index.score("run", &["running quickly", "sleeping deeply"]);
        assert!(scores[0] > 0.0, "stemming should match 'run' to 'running'");
        assert_eq!(scores[1], 0.0);
    }

    #[test]
    fn test_default_applies_unicode_folding() {
        // Default must fold diacritics: "cafe" matches "café"
        let index = BM25::default();
        let scores = index.score("cafe", &["café latte", "green tea"]);
        assert!(
            scores[0] > 0.0,
            "unicode folding should match 'cafe' to 'café'"
        );
        assert_eq!(scores[1], 0.0);
    }

    #[test]
    fn test_default_applies_stopwords() {
        // Default must remove stopwords: "the" alone should not match
        let mut index = BM25::default();
        index.add(&["the quick fox", "lazy dog"]).unwrap();
        // "the" is a stopword, so querying "the" should return no results
        let results = index.search("the", 10);
        assert!(results.is_empty(), "stopwords should be filtered");
    }

    #[test]
    fn test_default_scores_match_explicit_best_config() {
        // BM25::default() must produce identical scores to the explicit best config
        let docs = &[
            "the café résumé running cancellation",
            "quick brown fox jumping",
            "lazy sleeping dog",
        ];
        let query = "running fox café";

        let default_index = BM25::default();
        let explicit_index = BM25::with_tokenizer(
            Method::Lucene,
            1.5,
            0.75,
            0.5,
            TokenizerMode::UnicodeStem,
            true,
        );

        let default_scores = default_index.score(query, docs);
        let explicit_scores = explicit_index.score(query, docs);

        for i in 0..docs.len() {
            assert!(
                (default_scores[i] - explicit_scores[i]).abs() < 1e-6,
                "doc {}: default={} != explicit={}",
                i,
                default_scores[i],
                explicit_scores[i]
            );
        }
    }

    #[test]
    fn test_enrich_adds_new_terms() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&["hello world", "foo bar"]).unwrap();

        // "hello world" can't find "extra"
        let results = index.search("extra", 10);
        assert!(results.is_empty() || results.iter().all(|r| r.index != 0));

        // Enrich doc 0 with new terms
        index.enrich(0, "extra keywords").unwrap();

        // Now "extra" finds doc 0
        let results = index.search("extra", 10);
        assert!(!results.is_empty());
        assert_eq!(results[0].index, 0);

        // Original terms still findable
        let results = index.search("hello", 10);
        assert!(!results.is_empty());
        assert_eq!(results[0].index, 0);
    }

    #[test]
    fn test_enrich_accumulates_tf() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&["cat dog", "bird fish"]).unwrap();

        let score_before = index.search("cat", 10)[0].score;

        // Enrich with more "cat" mentions — TF should increase, boosting score
        index.enrich(0, "cat cat cat").unwrap();

        let score_after = index.search("cat", 10)[0].score;
        assert!(
            score_after > score_before,
            "score should increase after enrichment: {} vs {}",
            score_after,
            score_before
        );
    }

    #[test]
    fn test_enrich_preserves_doc_count() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&["hello world", "foo bar"]).unwrap();
        assert_eq!(index.len(), 2);

        index.enrich(0, "extra stuff").unwrap();
        assert_eq!(index.len(), 2); // no new docs created
    }

    #[test]
    fn test_enrich_empty_text() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&["hello world"]).unwrap();
        let dl_before = index.doc_lengths[0];

        index.enrich(0, "").unwrap(); // no-op
        assert_eq!(index.doc_lengths[0], dl_before);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_enrich_out_of_range() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&["hello"]).unwrap();
        index.enrich(5, "boom").unwrap(); // panics
    }

    #[test]
    fn with_options_full_max_n_2_allocates_ngram_side() {
        let idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, /*max_n*/ 2, /*n_features*/ 1 << 16,
        );
        assert_eq!(idx.max_n(), 2);
        assert_eq!(idx.n_features(), 1 << 16);
        assert!(idx.ngram_side().is_some());
    }

    #[test]
    fn with_options_full_max_n_1_does_not_allocate_ngram_side() {
        let idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 1, 0,
        );
        assert_eq!(idx.max_n(), 1);
        assert_eq!(idx.n_features(), 0);
        assert!(idx.ngram_side().is_none());
    }

    #[test]
    fn add_with_max_n_2_indexes_bigrams_via_hash_tier() {
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 2, 1 << 16,
        );
        idx.add(&["machine learning is fun", "deep learning rocks"]).unwrap();

        // Unigram tier unchanged.
        assert!(idx.get_term_id("learn").is_some());
        // N-gram tier populated.
        let side = idx.ngram_side().expect("max_n=2 must allocate side");
        let slot_ml = side.slot_for("machin learn");
        assert!(side.df(slot_ml) >= 1, "bigram 'machin learn' must be indexed");
        let slot_dl = side.slot_for("deep learn");
        assert!(side.df(slot_dl) >= 1, "bigram 'deep learn' must be indexed");
    }

    #[test]
    fn add_doc_lengths_count_unigrams_only_with_hash_tier() {
        let mut a = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 1, 0,
        );
        let mut b = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 4, 1 << 16,
        );
        a.add(&["the quick brown fox jumps"]).unwrap();
        b.add(&["the quick brown fox jumps"]).unwrap();
        // doc_lengths must NOT inflate when bigrams/trigrams/4-grams are added —
        // BM25's b·dl/avgdl normalization would break.
        assert_eq!(a.get_doc_length(0), b.get_doc_length(0));
        assert_eq!(a.total_tokens, b.total_tokens);
    }

    /// Vocab IDs differ per instance (HashMap iteration order); this test verifies
    /// functional equivalence instead — same term set, same per-term DF, same postings.
    ///
    /// Why not byte-equal vocab maps: `add_cpu` builds a per-doc `HashMap<String, _>`
    /// of token -> (tf, positions), then iterates it to assign term IDs. HashMap
    /// iteration order depends on the per-process RandomState seed AND the insertion
    /// pattern, so two BM25 instances with the same input can legally produce
    /// different `term -> id` mappings while remaining functionally identical.
    /// This predates the n-gram tier and is not a regression from it.
    #[test]
    fn add_with_max_n_1_functionally_equivalent_to_legacy() {
        use std::collections::HashSet;

        let mut legacy = BM25::with_options(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false,
        );
        let mut explicit = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 1, 0,
        );
        let docs = ["alpha beta gamma", "beta delta", "alpha alpha gamma"];
        legacy.add(&docs).unwrap();
        explicit.add(&docs).unwrap();

        // Vocab-id-invariant comparison: term IDs differ per instance because
        // `add_cpu` iterates a per-doc HashMap with non-deterministic order.
        // Compare functional equivalence (same term set, same per-term DF/postings).

        // 1. Same term set.
        let legacy_terms: HashSet<&String> = legacy.get_vocab().keys().collect();
        let explicit_terms: HashSet<&String> = explicit.get_vocab().keys().collect();
        assert_eq!(legacy_terms, explicit_terms, "vocab term sets differ");

        // 2. Scalar metadata is id-invariant.
        assert_eq!(legacy.num_docs, explicit.num_docs);
        assert_eq!(legacy.total_tokens, explicit.total_tokens);
        assert_eq!(legacy.get_doc_lengths_slice(), explicit.get_doc_lengths_slice());

        // 3. Per-term DF + posting list contents (resolve term IDs through both vocabs).
        for term in legacy.get_vocab().keys() {
            let lid = legacy.get_term_id(term).unwrap();
            let eid = explicit.get_term_id(term).unwrap();
            assert_eq!(
                legacy.doc_freqs[lid as usize],
                explicit.doc_freqs[eid as usize],
                "DF mismatch for term {:?}", term,
            );
            let mut a: Vec<(u32, u32)> = Vec::new();
            legacy.for_each_posting(lid, |d, t| a.push((d, t)));
            let mut b: Vec<(u32, u32)> = Vec::new();
            explicit.for_each_posting(eid, |d, t| b.push((d, t)));
            a.sort();
            b.sort();
            assert_eq!(a, b, "posting list mismatch for term {:?}", term);
        }

        // 4. ngram_side must be None for both.
        assert!(legacy.ngram_side().is_none());
        assert!(explicit.ngram_side().is_none());
    }

    #[test]
    #[should_panic(expected = "n_features must be a power of 2")]
    fn with_options_full_rejects_zero_n_features_when_max_n_ge_2() {
        let _ = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, /*max_n*/ 4, /*n_features*/ 0,
        );
    }

    // ----- update() routes through NgramSide::replace_doc -----

    #[test]
    fn update_replaces_ngram_side_postings() {
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, /*max_n*/ 2, /*n_features*/ 1 << 16,
        );
        idx.add(&["alpha beta gamma"]).unwrap();
        let side = idx.ngram_side().unwrap();
        let slot_ab = side.slot_for("alpha beta");
        assert!(side.df(slot_ab) >= 1, "old bigram present before update");

        idx.update(0, "delta epsilon").unwrap();
        let side = idx.ngram_side().unwrap();
        assert_eq!(side.df(slot_ab), 0, "old bigram dropped after update");
        let slot_de = side.slot_for("delta epsilon");
        assert!(side.df(slot_de) >= 1, "new bigram present after update");
    }

    #[test]
    fn update_max_n_1_unaffected() {
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 1, 0,
        );
        idx.add(&["alpha beta"]).unwrap();
        idx.update(0, "delta epsilon").unwrap();
        // No ngram side allocated; just confirm update still succeeds + vocab updates.
        assert!(idx.ngram_side().is_none());
        assert!(idx.get_term_id("delta").is_some());
    }

    // ----- enrich/unenrich route through NgramSide -----

    #[test]
    fn enrich_adds_to_ngram_side_at_offset_positions() {
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 2, 1 << 16,
        );
        idx.add(&["alpha beta"]).unwrap();
        let dl_before = idx.get_doc_length(0);
        idx.enrich(0, "gamma delta").unwrap();
        // doc_length grows ONLY by extra unigram count (n-grams don't inflate it).
        assert_eq!(idx.get_doc_length(0), dl_before + 2);

        let side = idx.ngram_side().unwrap();
        let slot_gd = side.slot_for("gamma delta");
        assert!(side.df(slot_gd) >= 1, "enriched bigram indexed");

        // Cross-boundary "beta gamma" intentionally NOT created (matches unigram
        // enrich semantics — extra_text is tokenized in isolation).
        let slot_bg = side.slot_for("beta gamma");
        assert_eq!(side.df(slot_bg), 0);
    }

    #[test]
    fn unenrich_reverses_enrich_on_ngram_side() {
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 2, 1 << 16,
        );
        idx.add(&["alpha beta"]).unwrap();
        idx.enrich(0, "gamma delta").unwrap();
        idx.unenrich(0, "gamma delta").unwrap();
        let side = idx.ngram_side().unwrap();
        assert_eq!(side.df(side.slot_for("gamma delta")), 0);
        assert_eq!(side.df(side.slot_for("alpha beta")), 1, "original bigram untouched");
    }

    #[test]
    fn enrich_exact_writes_only_full_phrase_slot_no_subwindows() {
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 4, 1 << 16,
        );
        idx.add(&["alpha beta"]).unwrap();
        let dl_before = idx.get_doc_length(0);

        // 4-token phrase. Only the n=4 slot should fire on the n-gram side.
        idx.enrich_exact(0, "gamma delta epsilon zeta").unwrap();

        // Unigram tier: each of the 4 stems is now indexed for doc 0.
        for tok in ["gamma", "delta", "epsilon", "zeta"] {
            let term_id = idx.get_term_id(tok)
                .unwrap_or_else(|| panic!("unigram {tok} should be in vocab"));
            let mut found = false;
            idx.for_each_posting(term_id, |did, _tf| if did == 0 { found = true });
            assert!(found, "unigram {tok} must have a posting for doc 0");
        }

        // doc_length grows by 4 (number of stemmed tokens added).
        assert_eq!(idx.get_doc_length(0), dl_before + 4);

        let side = idx.ngram_side().unwrap();
        // Full 4-gram slot got the write.
        let slot_4g = side.slot_for("gamma delta epsilon zeta");
        assert!(side.df(slot_4g) >= 1, "full 4-gram slot must be written");

        // Sub-windows must NOT be written (skipping any that happen to
        // hash-collide with the 4-gram slot).
        for sub in ["gamma delta", "delta epsilon", "epsilon zeta",
                    "gamma delta epsilon", "delta epsilon zeta"] {
            let s = side.slot_for(sub);
            if s == slot_4g {
                continue;
            }
            assert_eq!(side.df(s), 0, "sub-window {sub:?} must not be written by enrich_exact");
        }
    }

    #[test]
    fn unenrich_exact_reverses_enrich_exact() {
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 4, 1 << 16,
        );
        idx.add(&["alpha beta"]).unwrap();
        let dl_before = idx.get_doc_length(0);

        idx.enrich_exact(0, "gamma delta epsilon").unwrap();
        idx.unenrich_exact(0, "gamma delta epsilon").unwrap();

        // doc_length restored.
        assert_eq!(idx.get_doc_length(0), dl_before);

        let side = idx.ngram_side().unwrap();
        let slot = side.slot_for("gamma delta epsilon");
        assert_eq!(side.df(slot), 0, "n-gram slot DF restored to 0");

        // Original "alpha beta" untouched.
        assert_eq!(side.df(side.slot_for("alpha beta")), 1);
    }

    #[test]
    fn enrich_exact_unigram_tier_only_when_phrase_too_long_for_max_n() {
        // 5-token phrase, max_n=4 → n-gram tier no-op (defence in depth).
        // But the unigram tier still indexes each stem. Use non-stopword
        // tokens so the English stopwords filter (use_stopwords=true)
        // doesn't drop them.
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 4, 1 << 16,
        );
        idx.add(&["seedword"]).unwrap();
        idx.enrich_exact(0, "alphax betax gammax deltax epsilonx").unwrap();

        let side = idx.ngram_side().unwrap();
        // Total n-gram-tier df contribution from this enrich must be 0
        // (slot for the 5-gram is out of `ns=[2,3,4]`).
        let slot_5g = side.slot_for("alphax betax gammax deltax epsilonx");
        assert_eq!(side.df(slot_5g), 0, "5-gram slot must be a no-op when max_n=4");

        // Unigram tier still got the writes.
        for tok in ["alphax", "betax", "gammax", "deltax", "epsilonx"] {
            assert!(idx.get_term_id(tok).is_some(), "unigram {tok} must be in vocab");
        }
    }

    #[test]
    fn enrich_max_n_1_unaffected() {
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 1, 0,
        );
        idx.add(&["alpha beta"]).unwrap();
        idx.enrich(0, "gamma").unwrap();
        assert!(idx.ngram_side().is_none());
        assert!(idx.get_term_id("gamma").is_some());
    }

    // ----- delete() compacts the n-gram side -----

    #[test]
    fn delete_compacts_ngram_side() {
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 2, 1 << 16,
        );
        idx.add(&[
            "alpha beta gamma",     // doc 0
            "beta gamma delta",     // doc 1
            "gamma delta epsilon",  // doc 2
        ]).unwrap();
        let side = idx.ngram_side().unwrap();
        let slot_bg = side.slot_for("beta gamma");
        assert_eq!(side.df(slot_bg), 2, "bigram in docs 0 + 1");

        idx.delete(&[1]).unwrap();
        let side = idx.ngram_side().unwrap();
        assert_eq!(side.df(slot_bg), 1, "doc 1 deleted, only doc 0 remains");

        // Surviving posting belongs to original doc 0 (still ID 0 after compaction).
        let mut found = false;
        side.for_each_posting(slot_bg, |did, _| if did == 0 { found = true; });
        assert!(found, "surviving doc id correctly preserved");
    }

    // ----- query_targets resolves uni term_ids + n-gram slots -----

    #[test]
    fn query_targets_emits_uni_term_ids_and_ngram_slots() {
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, /*max_n*/ 2, /*n_features*/ 1 << 16,
        );
        idx.add(&["machine learning is fun"]).unwrap();
        let (uni_ids, ngram_slots) = idx.query_targets("machine learning");
        assert_eq!(uni_ids.len(), 2, "two unigram tokens (machin, learn)");
        assert_eq!(ngram_slots.len(), 1, "one bigram slot");
        let side = idx.ngram_side().unwrap();
        assert!(side.df(ngram_slots[0]) >= 1, "bigram slot has postings");
    }

    #[test]
    fn query_targets_no_ngram_for_max_n_1() {
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 1, 0,
        );
        idx.add(&["alpha beta"]).unwrap();
        let (uni, ng) = idx.query_targets("alpha beta");
        assert_eq!(uni.len(), 2);
        assert!(ng.is_empty(), "max_n=1 yields no n-gram slots");
    }

    #[test]
    fn query_targets_skips_unseen_uni_and_ngrams() {
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 2, 1 << 16,
        );
        idx.add(&["alpha beta"]).unwrap();
        let (uni, ng) = idx.query_targets("alpha gamma");
        assert_eq!(uni.len(), 1, "only 'alpha' in vocab; 'gamma' skipped");
        // 'alpha gamma' bigram hashes to a slot — the slot may or may not have
        // postings (collision with another bigram) but at small n_features=1<<16
        // and a tiny corpus the chance is negligible. Just check it's NOT in ng
        // because side.df(slot) == 0.
        assert!(ng.is_empty(), "no bigrams from this query are in the corpus");
    }

    // ----- score() incorporates n-gram contributions -----

    #[test]
    fn score_with_max_n_2_ranks_bigram_doc_higher() {
        let idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, /*max_n*/ 2, /*n_features*/ 1 << 16,
        );
        let docs = ["machine learning models", "machine vision systems"];
        let scores = idx.score("machine learning", &docs);
        assert_eq!(scores.len(), 2);
        assert!(scores[0] > scores[1], "doc with bigram match scores higher");
    }

    // ----- search/search_filtered combine uni + n-gram tier scores -----

    #[test]
    fn search_with_max_n_2_prefers_bigram_match() {
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, /*max_n*/ 2, /*n_features*/ 1 << 16,
        );
        idx.add(&[
            "machine learning is the future",        // doc 0: contains bigram
            "machine breaks learning is hard",       // doc 1: unigrams scattered
        ]).unwrap();
        let res = idx.search("machine learning", 2);
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].index, 0, "doc 0 (bigram match) ranks first");
        assert!(res[0].score > res[1].score);
    }

    #[test]
    fn search_max_n_1_byte_identical_baseline() {
        let mut a = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 1, 0,
        );
        let mut b = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 1, 0,
        );
        a.add(&["the quick brown fox", "lazy dog"]).unwrap();
        b.add(&["the quick brown fox", "lazy dog"]).unwrap();
        let ra = a.search("quick fox", 5);
        let rb = b.search("quick fox", 5);
        assert_eq!(ra.len(), rb.len());
        for (x, y) in ra.iter().zip(rb.iter()) {
            assert_eq!(x.index, y.index);
            assert!((x.score - y.score).abs() < 1e-5);
        }
        assert_eq!(ra.len(), 1, "only doc 0 matches");
        assert_eq!(ra[0].index, 0);
    }

    #[test]
    fn search_filtered_with_max_n_2_scores_ngram_in_subset() {
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 2, 1 << 16,
        );
        idx.add(&[
            "machine learning is fun",   // doc 0
            "machine vision",            // doc 1
            "deep learning rocks",       // doc 2
        ]).unwrap();
        let r = idx.search_filtered("machine learning", 3, &[0, 1, 2]);
        // doc 0 has the bigram + both unigrams; should rank first.
        assert_eq!(r[0].index, 0);
    }

    #[test]
    fn query_targets_dedups_repeated_uni_and_ngrams() {
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 2, 1 << 16,
        );
        idx.add(&["alpha beta alpha"]).unwrap();
        // Query repeats the bigram — should appear in slots only once.
        let (uni, ng) = idx.query_targets("alpha beta alpha beta");
        // unique unigrams: alpha, beta → 2
        assert_eq!(uni.len(), 2);
        // unique bigrams in query: "alpha beta" (positions 0, 2), "beta alpha" (position 1).
        // Both are in the corpus (corpus has "alpha beta alpha" → bigrams "alpha beta", "beta alpha").
        // Should dedupe to at most 2 unique slots, with no duplicates.
        let slot_set: std::collections::HashSet<u32> = ng.iter().copied().collect();
        assert_eq!(ng.len(), slot_set.len(), "no duplicate slots");
    }

    // ----- BM25::ngram_df thin wrapper -----

    #[test]
    fn ngram_df_returns_zero_for_unigram_only() {
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 1, 0,
        );
        idx.add(&["alpha beta"]).unwrap();
        // No n-gram tier exists when max_n == 1, so any query returns 0.
        assert_eq!(idx.ngram_df("alpha beta"), 0);
    }

    #[test]
    fn ngram_df_returns_slot_df_for_known_bigram() {
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 2, 1 << 16,
        );
        idx.add(&["machine learning is fun"]).unwrap();
        // The bigram "machin learn" (post-stem) is in the corpus.
        assert!(idx.ngram_df("machin learn") >= 1);
        // An unseen bigram with low collision probability returns 0.
        assert_eq!(idx.ngram_df("nothing here"), 0);
    }

    /// CUDA parity smoke test: combined uni+n-gram flat term-id space must
    /// produce identical top-k as the CPU path. Skipped when no GPU is present
    /// so it never blocks CPU-only hosts.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_search_matches_cpu_for_max_n_2() {
        use crate::cuda;
        if !cuda::is_cuda_available() { return; }
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, /*max_n*/ 2, /*n_features*/ 1 << 16,
        );
        idx.add(&[
            "machine learning is fun",
            "deep learning rocks",
            "machine vision matters",
        ]).unwrap();
        let cpu = idx.search("machine learning", 3);
        let mut gpu_idx = idx.to_gpu_search_index().unwrap();
        let gpu = idx.search_gpu(&mut gpu_idx, "machine learning", 3);
        assert_eq!(cpu.len(), gpu.len());
        for (c, g) in cpu.iter().zip(gpu.iter()) {
            assert_eq!(c.index, g.index);
            assert!((c.score - g.score).abs() < 1e-4);
        }
    }

    // ── Arbitrary-subset n-gram tests (with_options_ngrams) ────────────────

    /// `ngrams=[1, 4]` skips bigrams + trigrams in the n-gram tier — the
    /// side stores 4-grams only, even though tokens generate bi/tri/4-grams.
    #[test]
    fn with_options_ngrams_skips_unselected_orders_in_tier() {
        let mut idx = BM25::with_options_ngrams(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Plain, false, false,
            vec![1u8, 4u8], 1 << 14,
        );
        idx.add(&["alpha beta gamma delta epsilon"]).unwrap();
        assert_eq!(idx.ngram_set(), &[1u8, 4u8]);
        assert_eq!(idx.max_n(), 4);
        assert!(idx.score_unigram(), "n=1 in set ⇒ unigram contributes");
        let side = idx.ngram_side().expect("4 ∈ set ⇒ side allocated");
        assert_eq!(side.ns(), &[4u8]);
        assert!(side.df(side.slot_for("alpha beta gamma delta")) >= 1, "4-gram present");
        assert_eq!(side.df(side.slot_for("alpha beta")), 0, "bigram skipped");
        assert_eq!(side.df(side.slot_for("alpha beta gamma")), 0, "trigram skipped");
    }

    /// `ngrams=[1]` is functionally equivalent to legacy `max_n=1`: no n-gram
    /// side allocated, unigram score is the BM25 score.
    #[test]
    fn with_options_ngrams_unigram_only_matches_legacy_max_n_1() {
        let mut new = BM25::with_options_ngrams(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false,
            vec![1u8], 0,
        );
        let mut legacy = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 1, 0,
        );
        let docs = ["the quick brown fox", "lazy dog", "quick fox jumps"];
        new.add(&docs).unwrap();
        legacy.add(&docs).unwrap();
        assert!(new.ngram_side().is_none(), "no side for ngrams=[1]");
        let r_new = new.search("quick fox", 5);
        let r_legacy = legacy.search("quick fox", 5);
        assert_eq!(r_new.len(), r_legacy.len());
        for (a, b) in r_new.iter().zip(r_legacy.iter()) {
            assert_eq!(a.index, b.index);
            assert!((a.score - b.score).abs() < 1e-5);
        }
    }

    /// `ngrams=[2]` builds the unigram tier (for query infra) but EXCLUDES
    /// it from BM25 scoring — only bigram contributions accumulate. A
    /// single-token query has no bigrams ⇒ zero score ⇒ no results.
    #[test]
    fn with_options_ngrams_bigram_only_score_excludes_unigrams() {
        let mut idx = BM25::with_options_ngrams(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false,
            vec![2u8], 1 << 14,
        );
        idx.add(&[
            "machine learning is the future",     // doc 0: bigram match
            "machine breaks learning down",       // doc 1: only unigrams match
        ]).unwrap();
        assert!(!idx.score_unigram(), "n=1 ∉ {{2}} ⇒ unigram tier not scored");
        // Unigram tier IS built — vocab/term_df work for query infra.
        assert!(idx.get_term_id("machin").is_some());

        // Two-token query: only the bigram contributes. Doc 0 must beat doc 1
        // (doc 1 has no continuous "machin learn" bigram).
        let r = idx.search("machine learning", 2);
        assert!(!r.is_empty(), "bigram present ⇒ score > 0");
        assert_eq!(r[0].index, 0, "doc 0 wins on the bigram");

        // Single-token query: produces no bigram ⇒ no score ⇒ no results.
        let r1 = idx.search("machine", 5);
        assert!(r1.is_empty(), "single-token query has no bigram contribution");
    }

    /// `ngrams=[2, 4]` skips both unigrams (score gate) AND trigrams
    /// (n-gram-tier emit). Trigram-only queries return empty; bigram or
    /// 4-gram queries score.
    #[test]
    fn with_options_ngrams_bi_and_4_gram_only() {
        let mut idx = BM25::with_options_ngrams(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Plain, false, false,
            vec![2u8, 4u8], 1 << 14,
        );
        idx.add(&[
            "alpha beta gamma delta epsilon",
            "x y z w q",
        ]).unwrap();
        assert!(!idx.score_unigram());
        let side = idx.ngram_side().unwrap();
        assert_eq!(side.ns(), &[2u8, 4u8]);
        // Trigram queries against this index have no slot in the side ⇒ score 0.
        // Bigram query — ranks doc 0 (which contains "alpha beta").
        let r = idx.search("alpha beta", 2);
        assert!(!r.is_empty());
        assert_eq!(r[0].index, 0);
    }

    #[test]
    fn test_enrich_batch_matches_serial() {
        // Build two identical indexes, enrich one serially and one via batch,
        // then verify they produce identical search results and doc lengths.
        let docs = &[
            "the quick brown fox",
            "a lazy dog sleeps",
            "hello world program",
        ];
        let items: Vec<(usize, Vec<String>)> = vec![
            (0, vec!["extra keywords alpha beta".into(), "gamma delta".into()]),
            (1, vec!["dog running fast".into()]),
            (2, vec!["rust programming language".into(), "systems software".into()]),
        ];

        // Serial enrichment
        let mut serial = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        serial.add(docs).unwrap();
        for (doc_id, phrases) in &items {
            for phrase in phrases {
                serial.enrich(*doc_id, phrase).unwrap();
            }
        }

        // Batch enrichment
        let mut batch = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        batch.add(docs).unwrap();
        batch.enrich_batch(&items).unwrap();

        // Verify identical doc lengths
        for i in 0..docs.len() {
            assert_eq!(
                serial.doc_lengths[i], batch.doc_lengths[i],
                "doc_lengths mismatch for doc {i}"
            );
        }
        assert_eq!(serial.total_tokens, batch.total_tokens);

        // Verify identical search results for several queries
        for query in &["alpha beta", "dog running", "rust programming", "quick fox", "gamma"] {
            let r_serial = serial.search(query, 10);
            let r_batch = batch.search(query, 10);
            assert_eq!(
                r_serial.len(), r_batch.len(),
                "result count mismatch for query '{query}'"
            );
            for (rs, rb) in r_serial.iter().zip(r_batch.iter()) {
                assert_eq!(rs.index, rb.index, "index mismatch for query '{query}'");
                assert!(
                    (rs.score - rb.score).abs() < 1e-6,
                    "score mismatch for query '{query}': {} vs {}",
                    rs.score, rb.score,
                );
            }
        }

        // Verify posting-list contents match (compare by term name since
        // term_id assignment order may differ for enrichment-only terms).
        assert_eq!(serial.vocab.len(), batch.vocab.len(), "vocab size mismatch");
        for (term, &s_tid) in &serial.vocab {
            let b_tid = *batch.vocab.get(term).unwrap_or_else(|| {
                panic!("term '{}' missing from batch vocab", term)
            });
            assert_eq!(
                serial.postings[s_tid as usize],
                batch.postings[b_tid as usize],
                "postings mismatch for term '{}'", term,
            );
        }
    }

    #[test]
    fn test_enrich_batch_with_ngrams_matches_serial() {
        // Same test but with max_n > 1 to exercise the ngram side path.
        let docs = &["alpha beta gamma delta", "x y z w"];
        let items: Vec<(usize, Vec<String>)> = vec![
            (0, vec!["epsilon zeta".into()]),
            (1, vec!["a b c d".into(), "e f".into()]),
        ];

        let mut serial = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, true);
        serial.add(docs).unwrap();
        for (doc_id, phrases) in &items {
            for phrase in phrases {
                serial.enrich(*doc_id, phrase).unwrap();
            }
        }

        let mut batch = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, true);
        batch.add(docs).unwrap();
        batch.enrich_batch(&items).unwrap();

        for i in 0..docs.len() {
            assert_eq!(serial.doc_lengths[i], batch.doc_lengths[i],
                "doc_lengths mismatch for doc {i}");
        }
        assert_eq!(serial.total_tokens, batch.total_tokens);

        // Search queries should produce identical results
        for query in &["alpha beta", "epsilon zeta", "a b", "x y z"] {
            let r_serial = serial.search(query, 10);
            let r_batch = batch.search(query, 10);
            assert_eq!(r_serial.len(), r_batch.len(),
                "result count mismatch for query '{query}'");
            for (rs, rb) in r_serial.iter().zip(r_batch.iter()) {
                assert_eq!(rs.index, rb.index, "index mismatch for query '{query}'");
                assert!((rs.score - rb.score).abs() < 1e-6,
                    "score mismatch for query '{query}'");
            }
        }
    }

    #[test]
    #[ignore]  // Run explicitly: cargo test bench_enrich_batch_100k -- --ignored --nocapture
    fn bench_enrich_batch_100k() {
        // Micro-benchmark: 100k enrichments comparing serial vs parallel batch.
        use std::time::Instant;

        let n_docs = 10_000usize;
        let phrases_per_doc = 10usize;

        // Build a base index with n_docs short documents
        let docs: Vec<String> = (0..n_docs)
            .map(|i| format!("document number {} with some baseline text", i))
            .collect();
        let doc_refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();

        // Prepare enrichment items: 10 phrases per doc = 100k total
        let items: Vec<(usize, Vec<String>)> = (0..n_docs)
            .map(|i| {
                let phrases: Vec<String> = (0..phrases_per_doc)
                    .map(|j| {
                        format!(
                            "enrichment phrase {} for document {} with extra keywords \
                             alpha beta gamma delta epsilon zeta eta theta iota kappa \
                             lambda mu nu xi omicron pi rho sigma tau upsilon phi chi \
                             psi omega variant {} text",
                            j, i, j * i
                        )
                    })
                    .collect();
                (i, phrases)
            })
            .collect();

        // --- Serial baseline ---
        let mut serial_idx = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        serial_idx.add(&doc_refs).unwrap();
        let t0 = Instant::now();
        for (doc_id, phrases) in &items {
            for phrase in phrases {
                serial_idx.enrich(*doc_id, phrase).unwrap();
            }
        }
        let serial_ms = t0.elapsed().as_millis();

        // --- Parallel batch ---
        let mut batch_idx = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        batch_idx.add(&doc_refs).unwrap();
        let t0 = Instant::now();
        batch_idx.enrich_batch(&items).unwrap();
        let batch_ms = t0.elapsed().as_millis();

        eprintln!(
            "[bench_enrich_batch_100k] serial={serial_ms}ms  batch={batch_ms}ms  \
             speedup={:.2}x  ({n_docs} docs x {phrases_per_doc} phrases = {} total)",
            serial_ms as f64 / batch_ms.max(1) as f64,
            n_docs * phrases_per_doc,
        );

        // Sanity check: results must match
        assert_eq!(serial_idx.total_tokens, batch_idx.total_tokens);

        // Verify posting-list contents match (compare by term name since
        // term_id assignment order may differ for enrichment-only terms).
        assert_eq!(serial_idx.vocab.len(), batch_idx.vocab.len(),
            "vocab size mismatch");
        for (term, &s_tid) in &serial_idx.vocab {
            let b_tid = *batch_idx.vocab.get(term).unwrap_or_else(|| {
                panic!("term '{}' (serial tid={}) missing from batch vocab", term, s_tid)
            });
            assert_eq!(
                serial_idx.postings[s_tid as usize],
                batch_idx.postings[b_tid as usize],
                "postings mismatch for term '{}'", term,
            );
        }
    }
}
