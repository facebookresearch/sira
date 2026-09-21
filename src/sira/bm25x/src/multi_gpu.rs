// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Multi-GPU search: replicate the index across N GPUs, dispatch batch queries
//! round-robin across devices for near-linear throughput scaling.
//!
//! Each GPU gets its own CudaIndexer + GpuSearchIndex on a dedicated thread.
//! Queries are split into N chunks and processed in parallel using scoped threads.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext as CudarcContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use crate::cuda::CUDA_KERNELS;

/// Per-GPU context: CUDA device + compiled kernels + search index data.
struct GpuDevice {
    stream: Arc<CudaStream>,
    zero_f32_fn: CudaFunction,
    bm25_score_fused_fn: CudaFunction,
    topk_per_block_fn: CudaFunction,
    // Index data resident on this GPU
    flat_doc_ids: CudaSlice<u32>,
    flat_tfs: CudaSlice<u32>,
    doc_lengths: CudaSlice<u32>,
    scores: CudaSlice<f32>,
    topk_block_doc_ids: CudaSlice<u32>,
    topk_block_scores: CudaSlice<f32>,
    topk_num_blocks: u32,
    // Index metadata (shared across GPUs, read-only)
    offsets: Vec<u64>,
    counts: Vec<u32>,
    num_docs: u32,
}

/// Multi-GPU search index. Holds replicated index data across multiple GPUs.
pub struct MultiGpuSearchIndex {
    devices: Vec<GpuDevice>,
}

// Safety: GpuDevice contains CudaSlice and CudaFunction which wrap raw
// pointers but represent owned GPU resources. Send is required so that
// scoped threads can take &mut GpuDevice via iter_mut(). Each device is
// exclusively accessed by one thread at a time (no shared &GpuDevice).
unsafe impl Send for GpuDevice {}

impl MultiGpuSearchIndex {
    /// Detect available GPUs and upload the index to each one.
    ///
    /// `postings[term_id]` is term-identity-agnostic — uni- and n-gram entries
    /// share the same flat layout. Multi-gram support requires no GPU code change;
    /// CPU-side `tokenize_single`/`tokenize_queries` produce the offset-encoded IDs
    /// (n-gram slots offset by `vocab.len()`).
    pub fn from_index(
        postings: &[Vec<(u32, u32)>],
        doc_lengths: &[u32],
        num_docs: u32,
    ) -> Result<Self, String> {
        // Flatten posting lists (shared computation, done once on CPU)
        let total_entries: usize = postings.iter().map(|p| p.len()).sum();
        let mut flat_doc_ids_cpu = Vec::with_capacity(total_entries);
        let mut flat_tfs_cpu = Vec::with_capacity(total_entries);
        let mut offsets = Vec::with_capacity(postings.len());
        let mut counts = Vec::with_capacity(postings.len());
        let mut offset = 0u64;

        for plist in postings {
            offsets.push(offset);
            counts.push(plist.len() as u32);
            for &(doc_id, tf) in plist {
                flat_doc_ids_cpu.push(doc_id);
                flat_tfs_cpu.push(tf);
            }
            offset += plist.len() as u64;
        }

        // Detect GPUs
        let num_gpus = Self::detect_gpu_count();
        if num_gpus == 0 {
            return Err("No CUDA GPUs available".to_string());
        }
        eprintln!(
            "[bm25x] Multi-GPU: uploading index to {} GPUs ({:.1} GB each)",
            num_gpus,
            (total_entries * 8 + num_docs as usize * 4) as f64 / 1e9
        );

        // Initialize each GPU on a separate thread (CUDA contexts are per-thread)
        let mut devices = Vec::with_capacity(num_gpus);
        // We must init each device on its own thread because cudarc binds context to thread
        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for device_id in 0..num_gpus {
                let flat_doc_ids_ref = &flat_doc_ids_cpu;
                let flat_tfs_ref = &flat_tfs_cpu;
                let doc_lengths_ref = doc_lengths;
                let offsets_ref = &offsets;
                let counts_ref = &counts;

                handles.push(s.spawn(move || {
                    Self::init_device(
                        device_id,
                        flat_doc_ids_ref,
                        flat_tfs_ref,
                        doc_lengths_ref,
                        offsets_ref,
                        counts_ref,
                        num_docs,
                    )
                }));
            }
            for handle in handles {
                match handle.join() {
                    Ok(Ok(dev)) => devices.push(dev),
                    Ok(Err(e)) => eprintln!("[bm25x] GPU init failed: {}", e),
                    Err(_) => eprintln!("[bm25x] GPU init thread panicked"),
                }
            }
        });

        if devices.is_empty() {
            return Err("Failed to initialize any GPU".to_string());
        }
        eprintln!("[bm25x] Multi-GPU: {} GPUs ready for search", devices.len());

        Ok(MultiGpuSearchIndex { devices })
    }

    fn detect_gpu_count() -> usize {
        // Try to create contexts on successive device IDs until one fails
        for i in 0..16 {
            if CudarcContext::new(i).is_err() {
                return i;
            }
        }
        16
    }

    fn init_device(
        device_id: usize,
        flat_doc_ids: &[u32],
        flat_tfs: &[u32],
        doc_lengths: &[u32],
        offsets: &[u64],
        counts: &[u32],
        num_docs: u32,
    ) -> Result<GpuDevice, String> {
        let device =
            CudarcContext::new(device_id).map_err(|e| format!("GPU {}: {:?}", device_id, e))?;
        let stream = device.default_stream();

        let ptx = compile_ptx(CUDA_KERNELS).map_err(|e| format!("NVRTC: {:?}", e))?;
        let module = device
            .load_module(ptx)
            .map_err(|e| format!("module: {:?}", e))?;

        let zero_f32_fn = module
            .load_function("zero_f32")
            .map_err(|e| format!("{:?}", e))?;
        let bm25_score_fused_fn = module
            .load_function("bm25_score_fused_v2")
            .map_err(|e| format!("{:?}", e))?;
        let topk_per_block_fn = module
            .load_function("topk_per_block")
            .map_err(|e| format!("{:?}", e))?;

        // Upload index data
        let d_flat_doc_ids = stream
            .clone_htod(flat_doc_ids)
            .map_err(|e| format!("{:?}", e))?;
        let d_flat_tfs = stream
            .clone_htod(flat_tfs)
            .map_err(|e| format!("{:?}", e))?;
        let d_doc_lengths = stream
            .clone_htod(doc_lengths)
            .map_err(|e| format!("{:?}", e))?;
        let d_scores: CudaSlice<f32> = stream
            .alloc_zeros(num_docs as usize)
            .map_err(|e| format!("{:?}", e))?;

        let topk_num_blocks = 256u32;
        let topk_buf_size = topk_num_blocks as usize * 1024;
        let d_topk_doc_ids: CudaSlice<u32> = stream
            .alloc_zeros(topk_buf_size)
            .map_err(|e| format!("{:?}", e))?;
        let d_topk_scores: CudaSlice<f32> = stream
            .alloc_zeros(topk_buf_size)
            .map_err(|e| format!("{:?}", e))?;

        Ok(GpuDevice {
            stream,
            zero_f32_fn,
            bm25_score_fused_fn,
            topk_per_block_fn,
            flat_doc_ids: d_flat_doc_ids,
            flat_tfs: d_flat_tfs,
            doc_lengths: d_doc_lengths,
            scores: d_scores,
            topk_block_doc_ids: d_topk_doc_ids,
            topk_block_scores: d_topk_scores,
            topk_num_blocks,
            offsets: offsets.to_vec(),
            counts: counts.to_vec(),
            num_docs,
        })
    }

    /// Number of GPUs in use.
    pub fn num_gpus(&self) -> usize {
        self.devices.len()
    }

    /// Batch search across multiple GPUs. Queries are split evenly across devices.
    pub fn search_batch(
        &mut self,
        all_query_terms: &[Vec<(u32, f32)>],
        k1: f32,
        b: f32,
        avgdl: f32,
        k: usize,
    ) -> Vec<Vec<(u32, f32)>> {
        let n_queries = all_query_terms.len();
        let n_gpus = self.devices.len();

        if n_queries == 0 {
            return Vec::new();
        }

        // Split queries across GPUs
        let chunk_size = n_queries.div_ceil(n_gpus);
        let mut all_results: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_queries];

        std::thread::scope(|s| {
            let mut handles = Vec::new();

            for (gpu_idx, device) in self.devices.iter_mut().enumerate() {
                let start = gpu_idx * chunk_size;
                let end = (start + chunk_size).min(n_queries);
                if start >= n_queries {
                    break;
                }
                let query_chunk = &all_query_terms[start..end];

                handles.push((
                    start,
                    s.spawn(move || {
                        let mut results = Vec::with_capacity(query_chunk.len());
                        for query_terms in query_chunk {
                            match Self::search_single(device, query_terms, k1, b, avgdl, k) {
                                Ok(r) => results.push(r),
                                Err(e) => {
                                    eprintln!("[bm25x] GPU {} search failed: {}", gpu_idx, e);
                                    results.push(Vec::new());
                                }
                            }
                        }
                        results
                    }),
                ));
            }

            for (start, handle) in handles {
                if let Ok(chunk_results) = handle.join() {
                    for (i, r) in chunk_results.into_iter().enumerate() {
                        all_results[start + i] = r;
                    }
                }
            }
        });

        all_results
    }

    /// Batch search with inline tokenization. Each GPU thread tokenizes its own
    /// queries and scores them immediately, so CPU tokenization of query N+1
    /// overlaps with GPU scoring of query N.
    pub fn search_batch_with_tokenizer<F>(
        &mut self,
        queries: &[&str],
        chunk_size: usize,
        tokenize: F,
        k1: f32,
        b: f32,
        avgdl: f32,
        k: usize,
        all_results: &mut Vec<Vec<crate::index::SearchResult>>,
    ) where
        F: Fn(&str) -> Vec<(u32, f32)> + Sync,
    {
        let n_queries = queries.len();
        if n_queries == 0 {
            return;
        }

        std::thread::scope(|s| {
            let mut handles = Vec::new();
            let tokenize_ref = &tokenize;

            for (gpu_idx, device) in self.devices.iter_mut().enumerate() {
                let start = gpu_idx * chunk_size;
                let end = (start + chunk_size).min(n_queries);
                if start >= n_queries {
                    break;
                }
                let query_chunk = &queries[start..end];

                handles.push((
                    start,
                    s.spawn(move || {
                        let mut results = Vec::with_capacity(query_chunk.len());
                        for query in query_chunk {
                            let terms = tokenize_ref(query);
                            match Self::search_single(device, &terms, k1, b, avgdl, k) {
                                Ok(r) => results.push(
                                    r.into_iter()
                                        .map(|(doc_id, score)| crate::index::SearchResult {
                                            index: doc_id as usize,
                                            score,
                                        })
                                        .collect(),
                                ),
                                Err(e) => {
                                    eprintln!("[bm25x] GPU {} search failed: {}", gpu_idx, e);
                                    results.push(Vec::new());
                                }
                            }
                        }
                        results
                    }),
                ));
            }

            for (start, handle) in handles {
                if let Ok(chunk_results) = handle.join() {
                    for (i, r) in chunk_results.into_iter().enumerate() {
                        all_results[start + i] = r;
                    }
                }
            }
        });
    }

    /// Process one query on one GPU device. Same logic as GpuSearchIndex::search.
    fn search_single(
        device: &mut GpuDevice,
        query_term_ids: &[(u32, f32)],
        k1: f32,
        b: f32,
        avgdl: f32,
        k: usize,
    ) -> Result<Vec<(u32, f32)>, String> {
        let block = 256u32;
        let grid_docs = device.num_docs.div_ceil(block);
        let n_docs = device.num_docs as i32;

        // 1. Zero scores
        unsafe {
            device
                .stream
                .launch_builder(&device.zero_f32_fn)
                .arg(&mut device.scores)
                .arg(&n_docs)
                .launch(LaunchConfig {
                    block_dim: (block, 1, 1),
                    grid_dim: (grid_docs, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| format!("{:?}", e))?;
        }

        // 2. Build fused params + launch scoring
        let mut flat_offsets: Vec<i64> = Vec::new();
        let mut virtual_starts: Vec<i64> = Vec::new();
        let mut idfs: Vec<f32> = Vec::new();
        let mut total_entries: i64 = 0;

        for &(term_id, idf_val) in query_term_ids {
            let tid = term_id as usize;
            if tid >= device.counts.len() || device.counts[tid] == 0 {
                continue;
            }
            flat_offsets.push(device.offsets[tid] as i64);
            virtual_starts.push(total_entries);
            idfs.push(idf_val);
            total_entries += device.counts[tid] as i64;
        }

        if total_entries == 0 {
            return Ok(Vec::new());
        }

        let num_terms = flat_offsets.len() as i32;
        let total = total_entries as i32;
        let grid = (total as u32).div_ceil(block);

        let d_flat_offsets = device
            .stream
            .clone_htod(&flat_offsets)
            .map_err(|e| format!("{:?}", e))?;
        let d_virtual_starts = device
            .stream
            .clone_htod(&virtual_starts)
            .map_err(|e| format!("{:?}", e))?;
        let d_idfs = device
            .stream
            .clone_htod(&idfs)
            .map_err(|e| format!("{:?}", e))?;

        unsafe {
            device
                .stream
                .launch_builder(&device.bm25_score_fused_fn)
                .arg(&device.flat_doc_ids)
                .arg(&device.flat_tfs)
                .arg(&device.doc_lengths)
                .arg(&mut device.scores)
                .arg(&d_flat_offsets)
                .arg(&d_virtual_starts)
                .arg(&d_idfs)
                .arg(&k1)
                .arg(&b)
                .arg(&avgdl)
                .arg(&num_terms)
                .arg(&total)
                .launch(LaunchConfig {
                    block_dim: (block, 1, 1),
                    grid_dim: (grid, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| format!("{:?}", e))?;
        }

        // 3. GPU top-k
        let k_clamped = k.min(1024) as i32;
        let topk_blocks = device.topk_num_blocks;

        unsafe {
            device
                .stream
                .launch_builder(&device.topk_per_block_fn)
                .arg(&device.scores)
                .arg(&mut device.topk_block_doc_ids)
                .arg(&mut device.topk_block_scores)
                .arg(&n_docs)
                .arg(&k_clamped)
                .launch(LaunchConfig {
                    block_dim: (256, 1, 1),
                    grid_dim: (topk_blocks, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| format!("{:?}", e))?;
        }

        // 4. Download block top-k results
        let download_count = topk_blocks as usize * k_clamped as usize;
        let block_doc_ids: Vec<u32> = device
            .stream
            .clone_dtoh(&device.topk_block_doc_ids.slice(..download_count))
            .map_err(|e| format!("{:?}", e))?;
        let block_scores: Vec<f32> = device
            .stream
            .clone_dtoh(&device.topk_block_scores.slice(..download_count))
            .map_err(|e| format!("{:?}", e))?;

        // 5. CPU merge
        let mut results: Vec<(u32, f32)> = block_doc_ids
            .into_iter()
            .zip(block_scores)
            .filter(|&(_, s)| s > 0.0)
            .collect();

        if results.len() > k {
            results.select_nth_unstable_by(k, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            results.truncate(k);
        }
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }
}
