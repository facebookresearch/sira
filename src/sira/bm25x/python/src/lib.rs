// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use bm25x_core::{Method, SearchResult, TokenizerMode};

/// Wrapper for a raw (*const u8, usize) string pointer pair.
/// These are extracted under the GIL, then used after GIL release.
/// Safety: the Python list must stay alive (keeping the strings alive)
/// for the entire duration these are used.
struct RawStr {
    ptr: *const u8,
    len: usize,
}

// Raw pointers are not Send by default, but this is safe because:
// 1. The Python strings are immutable and reference-counted
// 2. The list object keeps all strings alive
// 3. We only read from these pointers, never write
unsafe impl Send for RawStr {}
unsafe impl Sync for RawStr {}

/// Extract raw UTF-8 pointers from a Python list of strings using
/// direct C-API calls. ~60-200ns per item vs ~2500ns for safe PyO3.
///
/// Safety: all items must be `str`. The returned pointers are valid
/// as long as the Python list (and its string elements) are alive.
fn extract_raw_ptrs(list: &Bound<'_, PyList>) -> PyResult<Vec<RawStr>> {
    let len = list.len();
    let mut result = Vec::with_capacity(len);
    unsafe {
        let list_ptr = list.as_ptr();
        for i in 0..len as pyo3::ffi::Py_ssize_t {
            let item = pyo3::ffi::PyList_GET_ITEM(list_ptr, i);
            let mut size: pyo3::ffi::Py_ssize_t = 0;
            let data = pyo3::ffi::PyUnicode_AsUTF8AndSize(item, &mut size);
            if data.is_null() {
                // Clear Python error state and return our own error
                pyo3::ffi::PyErr_Clear();
                return Err(PyValueError::new_err(format!(
                    "element {} is not a string",
                    i
                )));
            }
            result.push(RawStr {
                ptr: data as *const u8,
                len: size as usize,
            });
        }
    }
    Ok(result)
}

/// Reconstruct &str slices from raw pointers.
/// Safety: pointers must be valid (Python strings still alive).
#[inline]
fn raw_to_strs(ptrs: &[RawStr]) -> Vec<&str> {
    ptrs.iter()
        .map(|r| unsafe {
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(r.ptr, r.len))
        })
        .collect()
}

fn pack_batch_results_numpy(
    py: Python<'_>,
    batch_results: &[Vec<SearchResult>],
) -> PyResult<PyObject> {
    use numpy::PyArray1;

    let n_queries = batch_results.len();
    let total_hits: usize = batch_results.iter().map(|r| r.len()).sum();

    let mut indices = Vec::with_capacity(total_hits);
    let mut scores = Vec::with_capacity(total_hits);
    let mut offsets = Vec::with_capacity(n_queries + 1);
    offsets.push(0i64);

    for results in batch_results {
        for r in results {
            indices.push(r.index as i32);
            scores.push(r.score);
        }
        offsets.push(indices.len() as i64);
    }

    let py_indices = PyArray1::from_vec(py, indices);
    let py_scores = PyArray1::from_vec(py, scores);
    let py_offsets = PyArray1::from_vec(py, offsets);

    Ok((py_indices, py_scores, py_offsets)
        .into_pyobject(py)?
        .into_any()
        .unbind())
}

fn parse_method(method: &str) -> PyResult<Method> {
    match method.to_lowercase().as_str() {
        "lucene" => Ok(Method::Lucene),
        "robertson" => Ok(Method::Robertson),
        "atire" => Ok(Method::Atire),
        "bm25l" => Ok(Method::BM25L),
        "bm25+" | "bm25plus" => Ok(Method::BM25Plus),
        _ => Err(PyValueError::new_err(format!("Unknown method: {}", method))),
    }
}

fn parse_tokenizer(tokenizer: &str) -> PyResult<TokenizerMode> {
    match tokenizer.to_lowercase().as_str() {
        "plain" => Ok(TokenizerMode::Plain),
        "unicode" => Ok(TokenizerMode::Unicode),
        "stem" => Ok(TokenizerMode::Stem),
        "unicode_stem" | "unicodestem" => Ok(TokenizerMode::UnicodeStem),
        "char_45" => Ok(TokenizerMode::CharNgram { min_n: 4, max_n: 5 }),
        "char_78" => Ok(TokenizerMode::CharNgram { min_n: 7, max_n: 8 }),
        _ => Err(PyValueError::new_err(format!(
            "Unknown tokenizer: {tokenizer}. \
             Choose from: plain, unicode, stem, unicode_stem, char_45, char_78"
        ))),
    }
}

fn io_err(e: std::io::Error) -> PyErr {
    PyValueError::new_err(format!("{}", e))
}

#[pyclass(name = "Tfidf")]
struct PyTfidf {
    inner: bm25x_core::Tfidf,
    /// True when output rows may be variable-length (dedup or stopword post-filter).
    variable_length: bool,
}

#[pymethods]
impl PyTfidf {
    #[new]
    #[pyo3(signature = (top_k=20, ngram_range=(1, 1), sublinear_tf=true, min_df=1, tokenizer="unicode_stem", use_stopwords=true, use_hashing=false, n_features=8388608, dedup=false))]
    fn new(
        top_k: usize,
        ngram_range: (usize, usize),
        sublinear_tf: bool,
        min_df: u32,
        tokenizer: &str,
        use_stopwords: bool,
        use_hashing: bool,
        n_features: usize,
        dedup: bool,
    ) -> PyResult<Self> {
        let mode = parse_tokenizer(tokenizer)?;
        if use_hashing && !n_features.is_power_of_two() {
            return Err(PyValueError::new_err(format!(
                "n_features must be a power of 2, got {}",
                n_features
            )));
        }
        let config = bm25x_core::TfidfConfig {
            top_k,
            ngram_range,
            sublinear_tf,
            min_df,
            use_hashing,
            n_features,
            dedup,
        };
        Ok(PyTfidf {
            variable_length: dedup || !use_stopwords,
            inner: bm25x_core::Tfidf::new(config, mode, use_stopwords),
        })
    }

    /// Fit vocabulary + IDF and extract top-k keywords per document.
    ///
    /// Returns `(top_words, top_scores)`:
    /// - `top_words`: list of lists of strings
    /// - `top_scores`: numpy float32 array `(n_docs, top_k)` when `dedup=false`,
    ///   or `list[list[float]]` when `dedup=true` (variable-length rows).
    fn fit_transform(
        &mut self,
        py: Python<'_>,
        texts: &Bound<'_, PyList>,
    ) -> PyResult<PyObject> {
        use numpy::IntoPyArray;

        let ptrs = extract_raw_ptrs(texts)?;
        let inner = &mut self.inner;
        let variable_length = self.variable_length;

        let (words, scores_vecs) = py.allow_threads(|| {
            let refs = raw_to_strs(&ptrs);
            inner.fit_transform(&refs)
        });

        let py_words: Vec<Vec<String>> = words;

        if variable_length {
            // Variable-length rows — return as list[list[float]]
            let py_scores: Vec<Vec<f32>> = scores_vecs;
            Ok((py_words, py_scores)
                .into_pyobject(py)?
                .into_any()
                .unbind())
        } else {
            // Fixed-shape — return as numpy 2D array
            let n_docs = py_words.len();
            let top_k = if n_docs > 0 { py_words[0].len() } else { 0 };
            let scores_flat: Vec<f32> = scores_vecs.into_iter().flatten().collect();
            let scores_ndarray =
                numpy::ndarray::Array2::from_shape_vec((n_docs, top_k), scores_flat)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let py_scores = scores_ndarray.into_pyarray(py);
            Ok((py_words, py_scores)
                .into_pyobject(py)?
                .into_any()
                .unbind())
        }
    }

    /// Fit + extract keywords, then serialize directly to ndjson bytes.
    ///
    /// Returns `bytes` containing one JSON line per document:
    /// `{"_id":"<id>","keywords":["w1","w2"],"scores":[1.0,2.0]}\n`
    ///
    /// Processing is chunked by `batch_size` to bound peak memory.
    #[pyo3(signature = (texts, ids, batch_size=500_000))]
    fn fit_transform_ndjson(
        &mut self,
        py: Python<'_>,
        texts: &Bound<'_, PyList>,
        ids: &Bound<'_, PyList>,
        batch_size: usize,
    ) -> PyResult<PyObject> {
        let text_ptrs = extract_raw_ptrs(texts)?;
        let id_ptrs = extract_raw_ptrs(ids)?;
        let inner = &mut self.inner;

        let ndjson_bytes = py.allow_threads(|| {
            let text_refs = raw_to_strs(&text_ptrs);
            let id_refs = raw_to_strs(&id_ptrs);
            inner.fit_transform_ndjson(&text_refs, &id_refs, batch_size)
        });

        Ok(pyo3::types::PyBytes::new(py, &ndjson_bytes)
            .into_any()
            .unbind())
    }

    /// Number of terms in the fitted vocabulary.
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Number of documents seen during fit.
    fn num_docs(&self) -> u32 {
        self.inner.num_docs()
    }
}

#[pyclass(name = "BM25")]
struct PyBM25 {
    inner: bm25x_core::BM25,
    /// If true, CUDA is required — errors are raised instead of silent fallback.
    cuda_required: bool,
    #[cfg(feature = "cuda")]
    gpu_search_index: Option<bm25x_core::cuda::GpuSearchIndex>,
    #[cfg(feature = "cuda")]
    multi_gpu_index: Option<bm25x_core::multi_gpu::MultiGpuSearchIndex>,
}

#[pymethods]
impl PyBM25 {
    /// Create a new index.
    ///
    /// If `index` is provided, the index is persisted to that directory.
    /// `tokenizer` can be: "plain", "unicode", "stem", "unicode_stem" (default),
    ///   "char_45" (char 4–5 grams), "char_78" (char 7–8 grams).
    ///   For char_* modes, use `ngrams=[1]` and `use_stopwords=false`.
    /// `cuda`: if True, require CUDA — raises an error if GPU is unavailable.
    ///         if False (default), auto-detect GPU and fall back to CPU silently.
    /// `max_n`: maximum n-gram order (1 = unigram-only, legacy behavior).
    ///          Values >= 2 enable the hashed n-gram tier. Implies a
    ///          *contiguous* `1..=max_n` set; for arbitrary subsets pass
    ///          `ngrams=` instead.
    /// `ngrams`: explicit list of n values that contribute to BM25 scoring
    ///           (each in 1..=8). Overrides `max_n` when provided. Examples:
    ///           `[1]` (pure unigram, == legacy max_n=1), `[1, 2, 3, 4]` (==
    ///           legacy max_n=4), `[1, 4]` (unigrams + 4-grams, skip
    ///           bigrams + trigrams), `[2]` (bigram-only score). The unigram
    ///           tier is always BUILT (query infrastructure depends on it),
    ///           but its score contribution is gated on `1 ∈ ngrams`.
    /// `n_features`: hashed-slot count for the n-gram tier (must be a power
    ///               of 2 when any n>=2 is selected). Ignored when ngrams
    ///               selects only n=1.
    #[new]
    #[pyo3(signature = (
        index=None,
        method="lucene",
        k1=1.5,
        b=0.75,
        delta=0.5,
        tokenizer="unicode_stem",
        use_stopwords=true,
        cuda=false,
        max_n=4,
        ngrams=None,
        n_features=8_388_608,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        index: Option<&str>,
        method: &str,
        k1: f32,
        b: f32,
        delta: f32,
        tokenizer: &str,
        use_stopwords: bool,
        cuda: bool,
        max_n: u8,
        ngrams: Option<Vec<u8>>,
        n_features: u32,
    ) -> PyResult<Self> {
        // Resolve the n-gram subset: explicit `ngrams=` wins; otherwise
        // derive a contiguous `1..=max_n` from `max_n` (legacy semantic).
        let ngram_set: Vec<u8> = match ngrams {
            Some(ns) => {
                let mut s = ns;
                s.sort_unstable();
                s.dedup();
                if s.is_empty() {
                    return Err(PyValueError::new_err(
                        "ngrams must be non-empty (must include at least one of 1..=8)",
                    ));
                }
                if !s.iter().all(|&n| (1..=8).contains(&n)) {
                    return Err(PyValueError::new_err(
                        "ngrams values must each be in 1..=8 \
                         (n>8 is impractical: vocab/disk explosion)",
                    ));
                }
                s
            }
            None => {
                if max_n == 0 {
                    return Err(PyValueError::new_err("max_n must be >= 1"));
                }
                if max_n > 8 {
                    return Err(PyValueError::new_err(
                        "max_n > 8 is impractical (vocab/disk explosion)",
                    ));
                }
                (1..=max_n).collect()
            }
        };

        // n_features is required iff any selected n is >= 2.
        let any_ngram = ngram_set.iter().any(|&n| n >= 2);
        if any_ngram && !n_features.is_power_of_two() {
            return Err(PyValueError::new_err(
                "n_features must be a power of 2 when any n>=2 is selected",
            ));
        }
        let stored_n_features = if any_ngram { n_features } else { 0 };

        // If cuda=True, verify GPU is available immediately
        if cuda && !bm25x_core::is_gpu_available() {
            return Err(PyValueError::new_err(
                "cuda=True but no CUDA GPU is available. \
                 Check that CUDA drivers are installed and a GPU is visible \
                 (CUDA_VISIBLE_DEVICES).",
            ));
        }

        let m = parse_method(method)?;
        let tok = parse_tokenizer(tokenizer)?;
        let inner = match index {
            Some(path) => {
                // Keep legacy "open or create" semantics: load existing on-disk
                // index (its persisted ngram_set wins over the constructor args)
                // OR create a new one with the requested config. Branch on
                // whether the directory already has a header.
                let p = std::path::Path::new(path);
                if p.join("header.bin").exists() {
                    // Existing index — header v3 dictates ngram_set/n_features.
                    let mut idx = bm25x_core::BM25::load(path, false).map_err(io_err)?;
                    idx.set_index_path(path);
                    idx.set_cuda_required(cuda);
                    idx
                } else {
                    // New index at this path — construct with requested options
                    // and let auto-save persist on the first mutation.
                    std::fs::create_dir_all(p).map_err(io_err)?;
                    let mut idx = bm25x_core::BM25::with_options_ngrams(
                        m, k1, b, delta, tok, use_stopwords, cuda,
                        ngram_set, stored_n_features,
                    );
                    idx.set_index_path(path);
                    idx
                }
            }
            None => bm25x_core::BM25::with_options_ngrams(
                m, k1, b, delta, tok, use_stopwords, cuda,
                ngram_set, stored_n_features,
            ),
        };
        Ok(PyBM25 {
            inner,
            cuda_required: cuda,
            #[cfg(feature = "cuda")]
            gpu_search_index: None,
            #[cfg(feature = "cuda")]
            multi_gpu_index: None,
        })
    }

    /// Upload the index to GPU for fast search. Call once after adding documents.
    /// Single queries use one GPU. Batch queries auto-dispatch across all GPUs.
    #[cfg(feature = "cuda")]
    fn upload_to_gpu(&mut self) -> PyResult<()> {
        // Create multi-GPU index first (replicates across all GPUs).
        match self.inner.to_multi_gpu_search_index() {
            Ok(mgpu) => {
                self.multi_gpu_index = Some(mgpu);
            }
            Err(e) => {
                eprintln!("[bm25x] Multi-GPU init failed (falling back to single GPU): {}", e);
                // Fall back to single-GPU only if multi-GPU fails.
                self.gpu_search_index = Some(
                    self.inner
                        .to_gpu_search_index()
                        .map_err(PyValueError::new_err)?,
                );
            }
        }
        Ok(())
    }

    /// Add documents to the index. Returns list of assigned indices.
    ///
    /// Uses unsafe FFI for ~12x faster string extraction, then releases
    /// the GIL for the entire Rust processing phase.
    fn add(&mut self, py: Python<'_>, documents: &Bound<'_, PyList>) -> PyResult<Vec<usize>> {
        // Phase 1: Extract raw UTF-8 pointers via C-API (~2s for 8.8M docs)
        let ptrs = extract_raw_ptrs(documents)?;

        // Phase 2: Release GIL, reconstruct &str, run Rust indexing (~11s, GIL-free)
        let inner = &mut self.inner;
        py.allow_threads(|| {
            let refs = raw_to_strs(&ptrs);
            inner.add(&refs).map_err(io_err)
        })
    }

    /// Add documents from a newline-delimited bytes blob.
    ///
    /// This is the fastest path: zero-copy buffer extraction (~0.1s for 2.6GB),
    /// then GIL-free Rust processing. Use from Python:
    ///
    /// ```python
    /// index.add_bytes(b"\n".join(s.encode() for s in corpus))
    /// # or if corpus is already List[str]:
    /// index.add_bytes("\n".join(corpus).encode("utf-8"))
    /// ```
    fn add_bytes(&mut self, py: Python<'_>, data: &[u8]) -> PyResult<Vec<usize>> {
        let t0 = std::time::Instant::now();
        let inner = &mut self.inner;
        let result = py.allow_threads(|| {
            let t_split = std::time::Instant::now();
            let docs: Vec<&str> = data
                .split(|&b| b == b'\n')
                .map(|chunk| {
                    std::str::from_utf8(chunk).map_err(|e| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
                    })
                })
                .collect::<Result<Vec<&str>, _>>()?;
            let split_time = t_split.elapsed();
            let t_add = std::time::Instant::now();
            let r = inner.add(&docs);
            let add_time = t_add.elapsed();
            if std::env::var("BM25X_PROFILE").is_ok() {
                eprintln!(
                    "[add_bytes] split={:.3}s add={:.3}s docs={}",
                    split_time.as_secs_f64(),
                    add_time.as_secs_f64(),
                    docs.len()
                );
            }
            r
        });
        if std::env::var("BM25X_PROFILE").is_ok() {
            eprintln!("[add_bytes] total={:.3}s", t0.elapsed().as_secs_f64());
        }
        result.map_err(io_err)
    }

    /// Search the index. Accepts a single query string or a list of queries.
    ///
    /// - Single query: `search("fox", k=10)` → `[(doc_id, score), ...]`
    /// - Batch queries: `search(["fox", "dog"], k=10)` → `[[(doc_id, score), ...], ...]`
    /// - With subset: `search("fox", k=10, subset=[0, 2])` for pre-filtered search
    /// - Batch with subsets: `search(["fox", "dog"], k=10, subset=[[0,2], [1,3]])`
    ///
    /// Batch mode is faster: CPU uses rayon parallelism, GPU amortizes kernel overhead.
    #[pyo3(signature = (query, k, subset=None))]
    fn search(
        &mut self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        k: usize,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        // Auto-upload to GPU only when cuda=True
        #[cfg(feature = "cuda")]
        if self.cuda_required && self.gpu_search_index.is_none() && self.multi_gpu_index.is_none() && !self.inner.is_empty() {
            self.upload_to_gpu()?;
        }

        // Check if query is a list (batch mode) or a string (single mode)
        if let Ok(query_str) = query.extract::<&str>() {
            // Single query mode
            let results = match subset {
                Some(s) => {
                    let ids: Vec<usize> = s.extract()?;
                    self.inner.search_filtered(query_str, k, &ids)
                }
                None => {
                    #[cfg(feature = "cuda")]
                    {
                        if let Some(ref mut gpu_idx) = self.gpu_search_index {
                            return Ok(self
                                .inner
                                .search_gpu(gpu_idx, query_str, k)
                                .into_iter()
                                .map(|r| (r.index, r.score))
                                .collect::<Vec<(usize, f32)>>()
                                .into_pyobject(py)?
                                .into_any()
                                .unbind());
                        }
                    }
                    self.inner.search(query_str, k)
                }
            };
            Ok(results
                .into_iter()
                .map(|r| (r.index, r.score))
                .collect::<Vec<(usize, f32)>>()
                .into_pyobject(py)?
                .into_any()
                .unbind())
        } else {
            // Batch mode: list of queries
            let query_list: Vec<String> = query.extract()?;
            let query_refs: Vec<&str> = query_list.iter().map(|s| s.as_str()).collect();

            let batch_results = match subset {
                Some(s) => {
                    let subset_lists: Vec<Vec<usize>> = s.extract()?;
                    let subset_refs: Vec<&[usize]> =
                        subset_lists.iter().map(|v| v.as_slice()).collect();
                    self.inner
                        .search_filtered_batch(&query_refs, k, &subset_refs)
                }
                None => {
                    #[cfg(feature = "cuda")]
                    {
                        // Multi-GPU batch: distribute queries across all GPUs
                        if let Some(ref mut mgpu) = self.multi_gpu_index {
                            return Ok(self
                                .inner
                                .search_multi_gpu_batch(mgpu, &query_refs, k)
                                .into_iter()
                                .map(|results| {
                                    results
                                        .into_iter()
                                        .map(|r| (r.index, r.score))
                                        .collect::<Vec<(usize, f32)>>()
                                })
                                .collect::<Vec<Vec<(usize, f32)>>>()
                                .into_pyobject(py)?
                                .into_any()
                                .unbind());
                        }
                        // Fallback: single-GPU batch
                        if let Some(ref mut gpu_idx) = self.gpu_search_index {
                            return Ok(self
                                .inner
                                .search_gpu_batch(gpu_idx, &query_refs, k)
                                .into_iter()
                                .map(|results| {
                                    results
                                        .into_iter()
                                        .map(|r| (r.index, r.score))
                                        .collect::<Vec<(usize, f32)>>()
                                })
                                .collect::<Vec<Vec<(usize, f32)>>>()
                                .into_pyobject(py)?
                                .into_any()
                                .unbind());
                        }
                    }
                    self.inner.search_batch(&query_refs, k)
                }
            };

            Ok(batch_results
                .into_iter()
                .map(|results| {
                    results
                        .into_iter()
                        .map(|r| (r.index, r.score))
                        .collect::<Vec<(usize, f32)>>()
                })
                .collect::<Vec<Vec<(usize, f32)>>>()
                .into_pyobject(py)?
                .into_any()
                .unbind())
        }
    }

    /// Search with expression syntax (AND, OR, NOT, +/-, boost, grouping).
    ///
    /// - Single query: `search_expr("fox AND cat", k=10)` → `[(doc_id, score), ...]`
    /// - Batch queries: `search_expr(["fox AND cat", "+dog -cat"], k=10)` → `[[(doc_id, score), ...], ...]`
    ///
    /// Batch mode uses rayon parallelism for maximum throughput.
    #[pyo3(signature = (query, k))]
    fn search_expr(
        &mut self,
        py: Python<'_>,
        query: &Bound<'_, PyAny>,
        k: usize,
    ) -> PyResult<PyObject> {
        // Auto-upload to GPU only when cuda=True
        #[cfg(feature = "cuda")]
        if self.cuda_required && self.gpu_search_index.is_none() && self.multi_gpu_index.is_none() && !self.inner.is_empty() {
            self.upload_to_gpu()?;
        }

        if let Ok(query_str) = query.extract::<&str>() {
            // Single query — try GPU, fall back to CPU expression search
            #[cfg(feature = "cuda")]
            {
                if let Some(ref mut gpu_idx) = self.gpu_search_index {
                    return Ok(self
                        .inner
                        .search_gpu(gpu_idx, query_str, k)
                        .into_iter()
                        .map(|r| (r.index, r.score))
                        .collect::<Vec<(usize, f32)>>()
                        .into_pyobject(py)?
                        .into_any()
                        .unbind());
                }
            }
            let results = self.inner.search_expr(query_str, k);
            Ok(results
                .into_iter()
                .map(|r| (r.index, r.score))
                .collect::<Vec<(usize, f32)>>()
                .into_pyobject(py)?
                .into_any()
                .unbind())
        } else {
            // Batch mode — try multi-GPU, then single-GPU, fall back to CPU
            let query_list: Vec<String> = query.extract()?;
            let query_refs: Vec<&str> = query_list.iter().map(|s| s.as_str()).collect();

            #[cfg(feature = "cuda")]
            {
                if let Some(ref mut mgpu) = self.multi_gpu_index {
                    return Ok(self
                        .inner
                        .search_multi_gpu_batch(mgpu, &query_refs, k)
                        .into_iter()
                        .map(|results| {
                            results
                                .into_iter()
                                .map(|r| (r.index, r.score))
                                .collect::<Vec<(usize, f32)>>()
                        })
                        .collect::<Vec<Vec<(usize, f32)>>>()
                        .into_pyobject(py)?
                        .into_any()
                        .unbind());
                }
                if let Some(ref mut gpu_idx) = self.gpu_search_index {
                    return Ok(self
                        .inner
                        .search_gpu_batch(gpu_idx, &query_refs, k)
                        .into_iter()
                        .map(|results| {
                            results
                                .into_iter()
                                .map(|r| (r.index, r.score))
                                .collect::<Vec<(usize, f32)>>()
                        })
                        .collect::<Vec<Vec<(usize, f32)>>>()
                        .into_pyobject(py)?
                        .into_any()
                        .unbind());
                }
            }

            let batch_results = self.inner.search_expr_batch(&query_refs, k);
            Ok(batch_results
                .into_iter()
                .map(|results| {
                    results
                        .into_iter()
                        .map(|r| (r.index, r.score))
                        .collect::<Vec<(usize, f32)>>()
                })
                .collect::<Vec<Vec<(usize, f32)>>>()
                .into_pyobject(py)?
                .into_any()
                .unbind())
        }
    }

    /// Delete documents by their indices.
    fn delete(&mut self, doc_ids: Vec<usize>) -> PyResult<()> {
        self.inner.delete(&doc_ids).map_err(io_err)
    }

    /// Batch search returning flat numpy arrays instead of nested Python lists.
    ///
    /// Returns `(indices, scores, offsets)`:
    /// - `indices`: int32 array of doc indices, length = total hits
    /// - `scores`: float32 array of scores, length = total hits
    /// - `offsets`: int64 array of per-query start offsets, length = n_queries + 1
    ///
    /// Query i's results are `indices[offsets[i]:offsets[i+1]]`.
    #[pyo3(signature = (queries, k))]
    fn search_expr_numpy(
        &mut self,
        py: Python<'_>,
        queries: Vec<String>,
        k: usize,
    ) -> PyResult<PyObject> {
        // Auto-upload to GPU only when cuda=True
        #[cfg(feature = "cuda")]
        if self.cuda_required && self.gpu_search_index.is_none() && self.multi_gpu_index.is_none() && !self.inner.is_empty() {
            self.upload_to_gpu()?;
        }

        let query_refs: Vec<&str> = queries.iter().map(|s| s.as_str()).collect();

        let batch_results = {
            #[cfg(feature = "cuda")]
            {
                if let Some(ref mut mgpu) = self.multi_gpu_index {
                    self.inner.search_multi_gpu_batch(mgpu, &query_refs, k)
                } else if let Some(ref mut gpu_idx) = self.gpu_search_index {
                    self.inner.search_gpu_batch(gpu_idx, &query_refs, k)
                } else {
                    self.inner.search_expr_batch(&query_refs, k)
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                self.inner.search_expr_batch(&query_refs, k)
            }
        };

        pack_batch_results_numpy(py, &batch_results)
    }

    /// Batch BM25 search with per-query extra exact n-gram keys.
    ///
    /// `extra_ngrams`: list of lists of pre-stemmed n-gram key strings,
    /// one inner list per query. Each string is a space-joined stemmed key
    /// (e.g. ``"diagnost accuraci"``). These keys are looked up as exact
    /// slots, bypassing the sliding-window tokeniser, so no sub-n-grams of
    /// the expansion phrases are generated.
    ///
    /// Returns the same ``(indices, scores, offsets)`` tuple as
    /// ``search_expr_numpy``.
    #[pyo3(signature = (queries, extra_ngrams, k))]
    fn search_expr_numpy_with_extras(
        &mut self,
        py: Python<'_>,
        queries: &Bound<'_, PyList>,
        extra_ngrams: &Bound<'_, PyList>,
        k: usize,
    ) -> PyResult<PyObject> {
        let q_ptrs = extract_raw_ptrs(queries)?;
        let extras: Vec<Vec<String>> = extra_ngrams
            .iter()
            .map(|inner| {
                let inner_list = inner.downcast::<PyList>().map_err(|_| {
                    PyValueError::new_err("extra_ngrams must be list[list[str]]")
                })?;
                inner_list
                    .iter()
                    .map(|s| s.extract::<String>())
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<_>>()?;

        if extras.len() != q_ptrs.len() {
            return Err(PyValueError::new_err(
                "queries and extra_ngrams must have the same length",
            ));
        }

        let batch_results = py.allow_threads(|| {
            let qs: Vec<&str> = raw_to_strs(&q_ptrs);
            self.inner.search_batch_with_extras(&qs, &extras, k)
        });

        pack_batch_results_numpy(py, &batch_results)
    }

    /// Per-expansion-key BM25 score contribution for specific documents.
    ///
    /// Args:
    ///   doc_indices: internal doc indices (u32) to evaluate.
    ///   ngram_keys:  list of pre-stemmed n-gram key strings (space-separated stems).
    ///
    /// Returns list of (slot_idx, doc_pos, idf, tf_raw, score) tuples for
    /// non-zero entries. ``slot_idx`` indexes into ``ngram_keys``; ``doc_pos``
    /// indexes into ``doc_indices``.
    #[pyo3(signature = (doc_indices, ngram_keys))]
    fn ngram_score_breakdown_for_docs(
        &self,
        py: Python<'_>,
        doc_indices: Vec<u32>,
        ngram_keys: Vec<String>,
    ) -> PyResult<Vec<(u32, u32, f32, u32, f32)>> {
        let slots: Vec<u32> = ngram_keys
            .iter()
            .map(|key| self.inner.ngram_slot(key))
            .collect();
        let result = py.allow_threads(|| {
            self.inner.ngram_score_breakdown_for_docs(&doc_indices, &slots)
        });
        Ok(result)
    }

    /// Update a document's text at the given index.
    fn update(&mut self, doc_id: usize, new_text: &str) -> PyResult<()> {
        self.inner.update(doc_id, new_text).map_err(io_err)
    }

    /// Enrich a document by adding extra tokens without removing existing ones.
    fn enrich(&mut self, doc_id: usize, extra_text: &str) -> PyResult<()> {
        self.inner.enrich(doc_id, extra_text).map_err(io_err)
    }

    /// Reverse a previous enrich() call by subtracting the same tokens.
    fn unenrich(&mut self, doc_id: usize, extra_text: &str) -> PyResult<()> {
        self.inner.unenrich(doc_id, extra_text).map_err(io_err)
    }

    /// Enrich a document treating ``extra_text`` as a SINGLE n-gram (no
    /// sliding sub-windows). Unigram tier behavior matches ``enrich``;
    /// the n-gram tier writes exactly one slot — the hash of the full
    /// joined token sequence. See Rust ``BM25::enrich_exact`` for the
    /// LLM-enrichment use case rationale.
    fn enrich_exact(&mut self, doc_id: usize, extra_text: &str) -> PyResult<()> {
        self.inner.enrich_exact(doc_id, extra_text).map_err(io_err)
    }

    /// Reverse a previous enrich_exact() call by subtracting the same
    /// tokens (single n-gram slot, not sub-windows).
    fn unenrich_exact(&mut self, doc_id: usize, extra_text: &str) -> PyResult<()> {
        self.inner.unenrich_exact(doc_id, extra_text).map_err(io_err)
    }

    /// Disable automatic saving after mutations.
    fn disable_auto_save(&mut self) {
        self.inner.disable_auto_save();
    }

    /// Save the index to a directory (explicit save, useful for in-memory indices).
    fn save(&self, index: &str) -> PyResult<()> {
        self.inner.save(index).map_err(io_err)
    }

    /// Load an index from a directory.
    #[staticmethod]
    #[pyo3(signature = (index, mmap=false, cuda=false))]
    fn load(index: &str, mmap: bool, cuda: bool) -> PyResult<Self> {
        if cuda && !bm25x_core::is_gpu_available() {
            return Err(PyValueError::new_err(
                "cuda=True but no CUDA GPU is available.",
            ));
        }
        let inner = bm25x_core::BM25::load(index, mmap).map_err(io_err)?;
        Ok(PyBM25 {
            inner,
            cuda_required: cuda,
            #[cfg(feature = "cuda")]
            gpu_search_index: None,
            #[cfg(feature = "cuda")]
            multi_gpu_index: None,
        })
    }

    /// Score a query against a list of documents.
    fn score(&self, query: &Bound<'_, PyAny>, documents: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let py = query.py();
        if let Ok(q) = query.extract::<String>() {
            let docs: Vec<String> = documents.extract()?;
            let refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
            let scores = self.inner.score(&q, &refs);
            Ok(scores.into_pyobject(py)?.into_any().unbind())
        } else {
            let queries: Vec<String> = query.extract()?;
            let doc_lists: Vec<Vec<String>> = documents.extract()?;
            let q_refs: Vec<&str> = queries.iter().map(|s| s.as_str()).collect();
            let d_refs: Vec<Vec<&str>> = doc_lists
                .iter()
                .map(|dl| dl.iter().map(|s| s.as_str()).collect())
                .collect();
            let d_slices: Vec<&[&str]> = d_refs.iter().map(|v| v.as_slice()).collect();
            let results = self.inner.score_batch(&q_refs, &d_slices);
            Ok(results.into_pyobject(py)?.into_any().unbind())
        }
    }

    /// Number of active documents.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Number of unique terms in the vocabulary.
    fn vocab_size(&self) -> usize {
        self.inner.get_vocab().len()
    }

    /// Maximum n-gram order this index was built with (1 = unigram-only).
    /// Derived from [`Self::ngrams`] (`= max(ngrams)`).
    fn max_n(&self) -> u8 {
        self.inner.max_n()
    }

    /// The n-gram subset that contributes to BM25 scoring (sorted, deduped,
    /// each in 1..=8). E.g. `[1]` = unigram-only, `[1, 2, 3, 4]` = default
    /// uni..4-gram, `[1, 4]` = unigrams + 4-grams skipping bi/tri.
    ///
    /// Returns `Vec<u32>` (not `Vec<u8>`) to force Python's list — PyO3 maps
    /// `Vec<u8>` to `bytes` by default, which is the wrong type here.
    fn ngrams(&self) -> Vec<u32> {
        self.inner.ngram_set().iter().map(|&n| n as u32).collect()
    }

    /// Whether the unigram tier contributes to BM25 scoring. The unigram tier
    /// is always BUILT (term DF, search_expr, prefix scan depend on it), but
    /// its score contribution is gated on `1 ∈ ngrams`. False when e.g.
    /// `ngrams=[2]` (bigram-only score).
    fn score_unigram(&self) -> bool {
        self.inner.score_unigram()
    }

    /// Hashed n-gram slot count (0 when no n>=2 is in `ngrams`).
    fn n_features(&self) -> u32 {
        self.inner.n_features()
    }

    /// Document frequency for an n-gram in the hashed n-gram tier.
    /// Over-estimates the true DF due to hash collisions; returns 0 when
    /// `max_n == 1` or the slot is empty.
    fn ngram_df(&self, ngram: &str) -> u32 {
        self.inner.ngram_df(ngram)
    }

    /// Tokenize and stem a text string using the index's tokenizer.
    ///
    /// Useful for understanding how queries/documents are processed internally.
    fn tokenize(&self, text: &str) -> Vec<String> {
        self.inner.tokenize(text)
    }

    /// Batch enrich: apply enrichment phrases to multiple documents in one call.
    ///
    /// `items` is a list of `(doc_index, phrases)` tuples.
    fn enrich_batch(&mut self, py: Python<'_>, items: Vec<(usize, Vec<String>)>) -> PyResult<()> {
        py.allow_threads(|| self.inner.enrich_batch(&items).map_err(io_err))
    }

    /// Columnar batch enrich: zero-copy from polars/Arrow.
    ///
    /// `indices`      — numpy u32 array of doc indices (one per doc).
    /// `doc_offsets`   — numpy u64 array, len = n_docs+1: phrases for doc i
    ///                   are the flat strings at positions doc_offsets[i]..doc_offsets[i+1].
    /// `str_data`      — raw UTF-8 bytes of all phrases concatenated (Arrow data buffer).
    /// `str_offsets`   — numpy i64 array, len = n_phrases+1: phrase j is
    ///                   str_data[str_offsets[j]..str_offsets[j+1]] (Arrow offsets buffer).
    #[pyo3(signature = (indices, doc_offsets, str_data, str_offsets))]
    fn enrich_batch_columnar(
        &mut self,
        py: Python<'_>,
        indices: numpy::PyReadonlyArray1<u32>,
        doc_offsets: numpy::PyReadonlyArray1<u64>,
        str_data: &[u8],
        str_offsets: numpy::PyReadonlyArray1<i64>,
    ) -> PyResult<()> {
        let idx = indices.as_slice()?;
        let doc_off = doc_offsets.as_slice()?;
        let s_off = str_offsets.as_slice()?;
        if doc_off.len() != idx.len() + 1 {
            return Err(PyValueError::new_err(format!(
                "doc_offsets length {} != indices length {} + 1",
                doc_off.len(), idx.len()
            )));
        }
        let items: Vec<(usize, Vec<String>)> = idx
            .iter()
            .enumerate()
            .map(|(i, &doc_idx)| {
                let pstart = doc_off[i] as usize;
                let pend = doc_off[i + 1] as usize;
                let phrases: Vec<String> = (pstart..pend)
                    .map(|j| {
                        let beg = s_off[j] as usize;
                        let end = s_off[j + 1] as usize;
                        // Safety: Arrow guarantees valid UTF-8
                        unsafe { String::from_utf8_unchecked(str_data[beg..end].to_vec()) }
                    })
                    .collect();
                (doc_idx as usize, phrases)
            })
            .collect();
        py.allow_threads(|| self.inner.enrich_batch(&items).map_err(io_err))
    }

    /// Batch enrich_exact: apply single-slot enrichment phrases to multiple documents.
    ///
    /// `items` is a list of `(doc_index, phrases)` tuples.
    fn enrich_exact_batch(&mut self, py: Python<'_>, items: Vec<(usize, Vec<String>)>) -> PyResult<()> {
        py.allow_threads(|| self.inner.enrich_exact_batch(&items).map_err(io_err))
    }

    /// Two-pass search with weighted expansion scoring.
    ///
    /// For each (query, expansion) pair, scores the original query and
    /// the expansion terms separately, then merges:
    /// ``final_score = base_score + weight * expansion_score``.
    /// Returns top-k results per query.
    #[pyo3(signature = (queries, expansion_terms, k, weight))]
    fn search_with_expansion(
        &self,
        py: Python<'_>,
        queries: Vec<String>,
        expansion_terms: Vec<String>,
        k: usize,
        weight: f32,
    ) -> PyResult<Vec<Vec<(usize, f32)>>> {
        let q_refs: Vec<&str> = queries.iter().map(|s| s.as_str()).collect();
        let e_refs: Vec<&str> = expansion_terms.iter().map(|s| s.as_str()).collect();
        let results = py.allow_threads(|| {
            self.inner.search_with_expansion(&q_refs, &e_refs, k, weight)
        });
        Ok(results
            .into_iter()
            .map(|rs| rs.into_iter().map(|r| (r.index, r.score)).collect())
            .collect())
    }

    /// Filter expansion phrases for query enrichment.
    ///
    /// Keeps a phrase if any sliding-window n-gram (1..=max_n) has
    /// 0 < DF <= max_df in the BM25 index. Returns ``(kept_phrases,
    /// rejected)`` — kept are original strings, tokenize before joining
    /// as expansion terms.
    #[pyo3(signature = (query, phrases, max_df))]
    fn filter_query_expansion(
        &self,
        query: &str,
        phrases: Vec<String>,
        max_df: u32,
    ) -> (Vec<String>, Vec<(String, String)>) {
        self.inner.filter_query_expansion(query, &phrases, max_df)
    }

    /// Get document frequency for a term (after tokenization + stemming).
    ///
    /// Returns None if the term is not in the vocabulary.
    fn get_term_df(&self, term: &str) -> Option<u32> {
        let term_id = self.inner.resolve_term_id(term)?;
        Some(self.inner.doc_freqs[term_id as usize])
    }

    /// Get IDF score for a term, using the index's scoring method.
    ///
    /// The formula varies by method (Lucene, Robertson, Atire, BM25L, BM25Plus).
    /// Returns None if the term is not in the vocabulary.
    fn get_term_idf(&self, term: &str) -> Option<f32> {
        self.inner.get_term_idf(term)
    }

    /// Find vocabulary terms matching a prefix, with their document frequencies.
    ///
    /// Returns up to `limit` matches sorted by document frequency (descending).
    #[pyo3(signature = (prefix, limit=20))]
    fn get_vocab_matches(&self, prefix: &str, limit: usize) -> Vec<(String, u32)> {
        let stemmed_prefix = if prefix.is_empty() {
            String::new()
        } else {
            // Stem the prefix to match against stemmed vocab
            self.inner.stem_token(prefix)
        };
        let vocab = self.inner.get_vocab();
        let mut matches: Vec<(String, u32)> = vocab
            .iter()
            .filter(|(term, _)| term.starts_with(&stemmed_prefix))
            .map(|(term, term_id)| (term.clone(), self.inner.doc_freqs[*term_id as usize]))
            .collect();
        matches.sort_by(|a, b| b.1.cmp(&a.1));
        matches.truncate(limit);
        matches
    }

    /// Find terms that co-occur with `term` in the same documents.
    ///
    /// Returns up to `limit` results as `[(term, cooccurrence_count, df), ...]`,
    /// sorted by co-occurrence count descending.
    #[pyo3(signature = (term, limit=20))]
    fn cooccurring_terms(&self, term: &str, limit: usize) -> Vec<(String, u32, u32)> {
        self.inner.cooccurring_terms(term, limit)
    }
}

/// Returns True if bm25x was compiled with CUDA support and a GPU is available.
#[pyfunction]
fn is_gpu_available() -> bool {
    bm25x_core::is_gpu_available()
}

// ---- NGramIndex ----

#[pyclass(name = "NGramIndex")]
struct PyNGramIndex {
    inner: bm25x_core::ngram_index::NGramIndex,
}

#[pymethods]
impl PyNGramIndex {
    #[new]
    #[pyo3(signature = (
        max_n=4,
        n_features=bm25x_core::ngram_index::DEFAULT_N_FEATURES,
        tokenizer="unicode_stem",
        use_stopwords=true,
    ))]
    fn new(
        max_n: usize,
        n_features: usize,
        tokenizer: &str,
        use_stopwords: bool,
    ) -> PyResult<Self> {
        if max_n == 0 {
            return Err(PyValueError::new_err("max_n must be >= 1"));
        }
        if !n_features.is_power_of_two() {
            return Err(PyValueError::new_err(format!(
                "n_features must be a power of 2 (got {n_features})"
            )));
        }
        let mode = parse_tokenizer(tokenizer)?;
        Ok(Self {
            inner: bm25x_core::ngram_index::NGramIndex::new(
                max_n,
                n_features,
                mode,
                use_stopwords,
            ),
        })
    }

    fn add(&mut self, py: Python<'_>, docs: Vec<String>) {
        py.allow_threads(|| self.inner.add(&docs));
    }

    fn df(&self, ngram: &str) -> u32 {
        self.inner.df(ngram)
    }

    fn contains(&self, ngram: &str) -> bool {
        self.inner.contains(ngram)
    }

    /// For a single phrase: tokenize (same pipeline as build), generate
    /// every n-gram with `n` in `[n_min, n_max]`, return the rarest seen
    /// n-gram as `(df, ngram_string, n)`. None when the phrase has too
    /// few tokens or every generated n-gram has DF=0.
    #[pyo3(signature = (text, n_min=1, n_max=4))]
    fn min_df_ngram(
        &self,
        text: &str,
        n_min: usize,
        n_max: usize,
    ) -> Option<(u32, String, usize)> {
        self.inner.min_df_ngram(text, n_min, n_max)
    }

    /// Batched version of `min_df_ngram`. Releases the GIL and
    /// parallelizes across phrases via rayon. Use this from Python
    /// loops over many phrases — replaces millions of `df()` calls
    /// with one Python ↔ Rust round-trip.
    #[pyo3(signature = (texts, n_min=1, n_max=4))]
    fn min_df_ngram_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        n_min: usize,
        n_max: usize,
    ) -> Vec<Option<(u32, String, usize)>> {
        py.allow_threads(|| self.inner.min_df_ngram_batch(&texts, n_min, n_max))
    }

    fn num_docs(&self) -> u32 {
        self.inner.num_docs()
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn n_features(&self) -> usize {
        self.inner.n_features()
    }

    /// Return the hash slot a given n-gram string maps to.  Useful for
    /// correctness oracles that need to reason about collisions in
    /// Python without re-implementing MurmurHash3.
    fn slot(&self, ngram: &str) -> u32 {
        self.inner.slot_of(ngram)
    }

    #[pyo3(signature = (query_ngrams, top_k=20, df_max=u32::MAX))]
    fn cooccur(
        &self,
        py: Python<'_>,
        query_ngrams: Vec<String>,
        top_k: usize,
        df_max: u32,
    ) -> Vec<(String, u32)> {
        py.allow_threads(|| self.inner.cooccur(&query_ngrams, top_k, df_max))
    }

    /// Filter LLM-proposed enrichment phrases. Releases the GIL.
    ///
    /// Filters: ``no_stems`` (all stopwords), ``already_in_doc``,
    /// ``not_in_vocab``, ``too_common`` (all sub-n-grams have DF > max_df).
    /// The ``too_common`` check uses sliding-window DF (1..=max_n),
    /// matching the write path of ``BM25::enrich()``.
    ///
    /// Returns ``(kept_phrases, stats, verdicts)``.
    /// ``kept_phrases`` are original candidate strings — pass directly
    /// to ``BM25::enrich_batch()``.
    #[pyo3(signature = (
        candidates, doc_text, prior_enrichments, max_df,
        max_n=4, require_in_vocab=true, collect_verdicts=false,
    ))]
    fn filter_candidates(
        &self,
        py: Python<'_>,
        candidates: Vec<String>,
        doc_text: String,
        prior_enrichments: Vec<String>,
        max_df: u32,
        max_n: usize,
        require_in_vocab: bool,
        collect_verdicts: bool,
    ) -> (
        Vec<String>,
        Vec<(String, u32)>,
        Vec<(String, String, Option<u32>, bool, String)>,
    ) {
        py.allow_threads(|| {
            let (kept, stats_static, verdicts_static) = self.inner.filter_candidates(
                &candidates,
                &doc_text,
                &prior_enrichments,
                max_df,
                max_n,
                require_in_vocab,
                collect_verdicts,
            );
            // ``&'static str`` reasons → Python ``str`` requires String;
            // copying once at the boundary keeps the hot Rust loop
            // alloc-free for the (false, false) case while still giving
            // PyO3 owned data it can convert without lifetime gymnastics.
            let stats: Vec<(String, u32)> = stats_static
                .into_iter()
                .map(|(r, c)| (r.to_string(), c))
                .collect();
            let verdicts: Vec<(String, String, Option<u32>, bool, String)> = verdicts_static
                .into_iter()
                .map(|(raw, key, df, kept, reason)| (raw, key, df, kept, reason.to_string()))
                .collect();
            (kept, stats, verdicts)
        })
    }

    /// One-shot per-doc context prep — see Rust docstring on
    /// ``NGramIndex::prepare_doc``. Combines ``extract_ngrams`` (Python
    /// triple-loop that was holding the GIL), ``cooccur`` (already Rust),
    /// and ``ban_tokens`` surface-form dedup (Python loop) into a single
    /// GIL-released call. Returns ``(doc_ngrams, cooccur_hints, ban_tokens)``.
    #[pyo3(signature = (doc_text, max_n=4, max_df=u32::MAX))]
    fn prepare_doc(
        &self,
        py: Python<'_>,
        doc_text: String,
        max_n: usize,
        max_df: u32,
    ) -> (Vec<String>, Vec<(String, u32)>, Vec<String>) {
        py.allow_threads(|| self.inner.prepare_doc(&doc_text, max_n, max_df))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.inner
            .save(path)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        bm25x_core::ngram_index::NGramIndex::load(path)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// Python wrapper around `bm25x_core::tokenizer::Tokenizer`. Exposed primarily for
/// brute-force correctness oracles in tests, but useful anywhere the caller
/// needs the same lowercase/normalize/stopword/stem pipeline that BM25 uses.
#[pyclass(name = "Tokenizer")]
struct PyTokenizer {
    inner: bm25x_core::tokenizer::Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    #[new]
    #[pyo3(signature = (mode="unicode_stem", use_stopwords=true))]
    fn new(mode: &str, use_stopwords: bool) -> PyResult<Self> {
        let mode = parse_tokenizer(mode)?;
        Ok(Self {
            inner: bm25x_core::tokenizer::Tokenizer::with_mode(mode, use_stopwords),
        })
    }

    /// Tokenize one document. Pure function — safe to call from many threads.
    fn tokenize(&self, text: &str) -> Vec<String> {
        self.inner.tokenize_owned(text)
    }
}

#[pymodule]
fn bm25x(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBM25>()?;
    m.add_class::<PyTfidf>()?;
    m.add_class::<PyNGramIndex>()?;
    m.add_class::<PyTokenizer>()?;
    m.add_function(wrap_pyfunction!(is_gpu_available, m)?)?;
    Ok(())
}
