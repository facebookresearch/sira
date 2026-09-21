// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! CUDA-accelerated BM25 search.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{
    CudaContext as CudarcContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

// ---------------------------------------------------------------------------
// CUDA kernel source
// ---------------------------------------------------------------------------

pub const CUDA_KERNELS: &str = r#"
extern "C" __global__ void compute_histogram(
    const unsigned int* __restrict__ term_ids,
    unsigned int* __restrict__ histogram,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&histogram[term_ids[idx]], 1);
    }
}

extern "C" __global__ void scatter_postings(
    const unsigned int* __restrict__ term_ids,
    const unsigned int* __restrict__ doc_ids,
    const unsigned int* __restrict__ tfs,
    const unsigned long long* __restrict__ offsets,
    unsigned int* __restrict__ counters,
    unsigned int* __restrict__ out_doc_ids,
    unsigned int* __restrict__ out_tfs,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned int term = term_ids[idx];
        unsigned int pos = (unsigned int)offsets[term] + atomicAdd(&counters[term], 1);
        out_doc_ids[pos] = doc_ids[idx];
        out_tfs[pos] = tfs[idx];
    }
}

extern "C" __global__ void zero_u32(unsigned int* buf, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) buf[idx] = 0;
}

extern "C" __global__ void zero_f32(float* buf, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) buf[idx] = 0.0f;
}

// BM25 Lucene scoring kernel: each thread scores one posting entry
// scores[doc_id] += idf * (tf / (k1 * (1 - b + b * dl/avgdl) + tf))
extern "C" __global__ void bm25_score_lucene(
    const unsigned int* __restrict__ posting_doc_ids,
    const unsigned int* __restrict__ posting_tfs,
    const unsigned int* __restrict__ doc_lengths,
    float* __restrict__ scores,
    float idf, float k1, float b, float avgdl,
    long long offset,
    int num_entries)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_entries) {
        unsigned int doc_id = posting_doc_ids[offset + idx];
        float tf = (float)posting_tfs[offset + idx];
        float dl = (float)doc_lengths[doc_id];
        float norm = 1.0f - b + b * dl / avgdl;
        float tfc = tf / (k1 * norm + tf);
        atomicAdd(&scores[doc_id], idf * tfc);
    }
}

// Fused multi-term BM25 scoring: ALL query terms in ONE kernel launch.
// and a prefix-sum array to map global thread idx → term
extern "C" __global__ void bm25_score_fused_v2(
    const unsigned int* __restrict__ posting_doc_ids,
    const unsigned int* __restrict__ posting_tfs,
    const unsigned int* __restrict__ doc_lengths,
    float* __restrict__ scores,
    const long long* __restrict__ flat_offsets,    // [num_terms] offset in flat posting arrays
    const long long* __restrict__ virtual_starts,  // [num_terms] prefix sum of counts
    const float* __restrict__ idfs,                // [num_terms] IDF values
    float k1, float b, float avgdl,
    int num_terms,
    int total_entries)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_entries) return;

    // Binary search: find which term this thread belongs to
    int lo = 0, hi = num_terms - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (virtual_starts[mid] <= (long long)idx) lo = mid;
        else hi = mid - 1;
    }

    int local_pos = idx - (int)virtual_starts[lo];
    long long flat_pos = flat_offsets[lo] + (long long)local_pos;

    unsigned int doc_id = posting_doc_ids[flat_pos];
    float tf = (float)posting_tfs[flat_pos];
    float dl = (float)doc_lengths[doc_id];
    float norm = 1.0f - b + b * dl / avgdl;
    float tfc = tf / (k1 * norm + tf);
    atomicAdd(&scores[doc_id], idfs[lo] * tfc);
}

// Top-k extraction: each thread checks one doc, writes to output if score > 0
// max_results prevents out-of-bounds writes
extern "C" __global__ void collect_nonzero(
    const float* __restrict__ scores,
    unsigned int* __restrict__ out_doc_ids,
    float* __restrict__ out_scores,
    unsigned int* __restrict__ count,
    int n,
    int max_results)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && scores[idx] > 0.0f) {
        unsigned int pos = atomicAdd(count, 1);
        if (pos < (unsigned int)max_results) {
            out_doc_ids[pos] = (unsigned int)idx;
            out_scores[pos] = scores[idx];
        }
    }
}

// ── Correct GPU top-k: per-thread private heaps, then block merge ──
//
// Each thread maintains a private top-k (insertion sort in registers).
// After scanning, threads write their private results to shared memory.
// Thread 0 does a final merge over all 256*k candidates → block top-k.
// No races: each thread's private top-k is exact, merge is sequential.

#define TOPK_K 10

extern "C" __global__ void topk_per_block(
    const float* __restrict__ scores,
    unsigned int* __restrict__ out_doc_ids,  // [num_blocks * TOPK_K]
    float* __restrict__ out_scores,          // [num_blocks * TOPK_K]
    int n,
    int k)
{
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Phase 1: each thread scans its chunk, maintains private top-k in registers
    float my_scores[TOPK_K];
    unsigned int my_ids[TOPK_K];
    for (int i = 0; i < TOPK_K; i++) {
        my_scores[i] = -1.0f;
        my_ids[i] = 0xFFFFFFFF;
    }
    float my_min = -1.0f;
    int my_min_idx = 0;

    // Grid-stride: this block handles a contiguous chunk of the scores array
    int chunk_size = (n + gridDim.x - 1) / gridDim.x;
    int chunk_start = blockIdx.x * chunk_size;
    int chunk_end = min(chunk_start + chunk_size, n);

    int actual_k = min(k, TOPK_K);

    for (int i = chunk_start + tid; i < chunk_end; i += block_size) {
        float val = scores[i];
        if (val > my_min) {
            // Replace minimum in private top-k
            my_scores[my_min_idx] = val;
            my_ids[my_min_idx] = (unsigned int)i;
            // Recompute minimum
            my_min = my_scores[0];
            my_min_idx = 0;
            for (int j = 1; j < actual_k; j++) {
                if (my_scores[j] < my_min) {
                    my_min = my_scores[j];
                    my_min_idx = j;
                }
            }
        }
    }

    // Phase 2: write private results to shared memory
    // Shared memory: 256 threads * TOPK_K entries * 8 bytes = 20KB (fits easily)
    __shared__ float s_scores[256 * TOPK_K];
    __shared__ unsigned int s_ids[256 * TOPK_K];

    for (int i = 0; i < actual_k; i++) {
        s_scores[tid * TOPK_K + i] = my_scores[i];
        s_ids[tid * TOPK_K + i] = my_ids[i];
    }
    __syncthreads();

    // Phase 3: thread 0 merges all 256*k candidates into block's top-k
    if (tid == 0) {
        float final_scores[TOPK_K];
        unsigned int final_ids[TOPK_K];
        for (int i = 0; i < actual_k; i++) {
            final_scores[i] = -1.0f;
            final_ids[i] = 0xFFFFFFFF;
        }
        float fmin = -1.0f;
        int fmin_idx = 0;

        int total_candidates = block_size * actual_k;
        for (int i = 0; i < total_candidates; i++) {
            float val = s_scores[i];
            if (val > fmin) {
                final_scores[fmin_idx] = val;
                final_ids[fmin_idx] = s_ids[i];
                fmin = final_scores[0];
                fmin_idx = 0;
                for (int j = 1; j < actual_k; j++) {
                    if (final_scores[j] < fmin) {
                        fmin = final_scores[j];
                        fmin_idx = j;
                    }
                }
            }
        }

        // Write block's top-k to global memory
        int block_offset = blockIdx.x * actual_k;
        for (int i = 0; i < actual_k; i++) {
            out_doc_ids[block_offset + i] = final_ids[i];
            out_scores[block_offset + i] = final_scores[i];
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// Global CUDA context (lazy, panic-safe)
// ---------------------------------------------------------------------------

static CUDA_BROKEN: AtomicBool = AtomicBool::new(false);
static GLOBAL_CONTEXT: OnceLock<Mutex<Option<Arc<CudaIndexer>>>> = OnceLock::new();

pub struct CudaIndexer {
    pub stream: Arc<CudaStream>,
    histogram_fn: CudaFunction,
    scatter_fn: CudaFunction,
    zero_fn: CudaFunction,
    // Search kernels
    zero_f32_fn: CudaFunction,
    #[allow(dead_code)]
    bm25_score_fn: CudaFunction,
    bm25_score_fused_fn: CudaFunction,
    #[allow(dead_code)]
    collect_nonzero_fn: CudaFunction,
    topk_per_block_fn: CudaFunction,
}

pub fn is_cuda_available() -> bool {
    !CUDA_BROKEN.load(Ordering::Relaxed) && get_global_context().is_some()
}

pub fn mark_cuda_broken() {
    CUDA_BROKEN.store(true, Ordering::Relaxed);
}

pub fn get_global_context() -> Option<Arc<CudaIndexer>> {
    if CUDA_BROKEN.load(Ordering::Relaxed) {
        return None;
    }
    let mutex = GLOBAL_CONTEXT.get_or_init(|| Mutex::new(None));
    let mut guard = mutex.lock().ok()?;
    if let Some(ref ctx) = *guard {
        return Some(Arc::clone(ctx));
    }
    // catch_unwind: cudarc may panic if CUDA driver is missing, too old,
    // or has incompatible symbols. This ensures we gracefully fall back to CPU.
    // Suppress panic output for clean fallback.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let init_result = std::panic::catch_unwind(|| CudaIndexer::new(0));
    std::panic::set_hook(prev_hook);
    match init_result {
        Ok(Ok(ctx)) => {
            let arc = Arc::new(ctx);
            *guard = Some(Arc::clone(&arc));
            Some(arc)
        }
        Ok(Err(_)) | Err(_) => {
            // Silent fallback to CPU. When cuda=True, the caller raises an error.
            CUDA_BROKEN.store(true, Ordering::Relaxed);
            None
        }
    }
}

#[allow(dead_code)]
pub struct GpuScatterResult {
    pub out_doc_ids: Vec<u32>,
    pub out_tfs: Vec<u32>,
    pub histogram: Vec<u32>,
    pub offsets: Vec<u64>,
}

impl CudaIndexer {
    fn new(device_id: usize) -> Result<Self, String> {
        let device = CudarcContext::new(device_id).map_err(|e| format!("device: {:?}", e))?;
        let stream = device.default_stream();

        let ptx =
            compile_ptx(CUDA_KERNELS).map_err(|e| format!("NVRTC compile failed: {:?}", e))?;
        let module = device
            .load_module(ptx)
            .map_err(|e| format!("module load: {:?}", e))?;

        let histogram_fn = module
            .load_function("compute_histogram")
            .map_err(|e| format!("load compute_histogram: {:?}", e))?;
        let scatter_fn = module
            .load_function("scatter_postings")
            .map_err(|e| format!("load scatter_postings: {:?}", e))?;
        let zero_fn = module
            .load_function("zero_u32")
            .map_err(|e| format!("load zero_u32: {:?}", e))?;
        let zero_f32_fn = module
            .load_function("zero_f32")
            .map_err(|e| format!("load zero_f32: {:?}", e))?;
        let bm25_score_fn = module
            .load_function("bm25_score_lucene")
            .map_err(|e| format!("load bm25_score_lucene: {:?}", e))?;
        let collect_nonzero_fn = module
            .load_function("collect_nonzero")
            .map_err(|e| format!("load collect_nonzero: {:?}", e))?;
        let bm25_score_fused_fn = module
            .load_function("bm25_score_fused_v2")
            .map_err(|e| format!("load bm25_score_fused_v2: {:?}", e))?;

        let topk_per_block_fn = module
            .load_function("topk_per_block")
            .map_err(|e| format!("load topk_per_block: {:?}", e))?;

        Ok(CudaIndexer {
            stream,
            histogram_fn,
            scatter_fn,
            zero_fn,
            zero_f32_fn,
            bm25_score_fn,
            bm25_score_fused_fn,
            collect_nonzero_fn,
            topk_per_block_fn,
        })
    }

    /// GPU scatter: histogram + atomic scatter on device.
    #[allow(dead_code)]
    pub fn gpu_scatter(
        &self,
        term_ids: &[u32],
        doc_ids: &[u32],
        tfs: &[u32],
        vocab_size: usize,
    ) -> Result<GpuScatterResult, String> {
        let n = term_ids.len();
        if n == 0 {
            return Ok(GpuScatterResult {
                out_doc_ids: vec![],
                out_tfs: vec![],
                histogram: vec![0; vocab_size],
                offsets: vec![0; vocab_size],
            });
        }

        let block = 256u32;
        let grid_n = (n as u32).div_ceil(block);
        let grid_v = (vocab_size as u32).div_ceil(block);
        let n_i32 = n as i32;
        let v_i32 = vocab_size as i32;

        let d_term_ids: CudaSlice<u32> = self
            .stream
            .clone_htod(term_ids)
            .map_err(|e| format!("{:?}", e))?;
        let d_doc_ids: CudaSlice<u32> = self
            .stream
            .clone_htod(doc_ids)
            .map_err(|e| format!("{:?}", e))?;
        let d_tfs: CudaSlice<u32> = self
            .stream
            .clone_htod(tfs)
            .map_err(|e| format!("{:?}", e))?;

        let mut d_histogram: CudaSlice<u32> = self
            .stream
            .alloc_zeros(vocab_size)
            .map_err(|e| format!("{:?}", e))?;

        unsafe {
            self.stream
                .launch_builder(&self.zero_fn)
                .arg(&mut d_histogram)
                .arg(&v_i32)
                .launch(LaunchConfig {
                    block_dim: (block, 1, 1),
                    grid_dim: (grid_v, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| format!("{:?}", e))?;
            self.stream
                .launch_builder(&self.histogram_fn)
                .arg(&d_term_ids)
                .arg(&mut d_histogram)
                .arg(&n_i32)
                .launch(LaunchConfig {
                    block_dim: (block, 1, 1),
                    grid_dim: (grid_n, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| format!("{:?}", e))?;
        }

        let histogram: Vec<u32> = self
            .stream
            .clone_dtoh(&d_histogram)
            .map_err(|e| format!("{:?}", e))?;
        let mut offsets: Vec<u64> = Vec::with_capacity(vocab_size);
        let mut running: u64 = 0;
        for &count in &histogram {
            offsets.push(running);
            running += count as u64;
        }

        let d_offsets: CudaSlice<u64> = self
            .stream
            .clone_htod(&offsets)
            .map_err(|e| format!("{:?}", e))?;
        let mut d_counters: CudaSlice<u32> = self
            .stream
            .alloc_zeros(vocab_size)
            .map_err(|e| format!("{:?}", e))?;

        unsafe {
            self.stream
                .launch_builder(&self.zero_fn)
                .arg(&mut d_counters)
                .arg(&v_i32)
                .launch(LaunchConfig {
                    block_dim: (block, 1, 1),
                    grid_dim: (grid_v, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| format!("{:?}", e))?;
        }

        let mut d_out_doc_ids: CudaSlice<u32> =
            self.stream.alloc_zeros(n).map_err(|e| format!("{:?}", e))?;
        let mut d_out_tfs: CudaSlice<u32> =
            self.stream.alloc_zeros(n).map_err(|e| format!("{:?}", e))?;

        unsafe {
            self.stream
                .launch_builder(&self.scatter_fn)
                .arg(&d_term_ids)
                .arg(&d_doc_ids)
                .arg(&d_tfs)
                .arg(&d_offsets)
                .arg(&mut d_counters)
                .arg(&mut d_out_doc_ids)
                .arg(&mut d_out_tfs)
                .arg(&n_i32)
                .launch(LaunchConfig {
                    block_dim: (block, 1, 1),
                    grid_dim: (grid_n, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| format!("{:?}", e))?;
        }

        let out_doc_ids = self
            .stream
            .clone_dtoh(&d_out_doc_ids)
            .map_err(|e| format!("{:?}", e))?;
        let out_tfs = self
            .stream
            .clone_dtoh(&d_out_tfs)
            .map_err(|e| format!("{:?}", e))?;

        Ok(GpuScatterResult {
            out_doc_ids,
            out_tfs,
            histogram,
            offsets,
        })
    }
}

// ---------------------------------------------------------------------------
// GPU-resident search index
// ---------------------------------------------------------------------------

/// Index data uploaded to GPU for fast search. Created once, reused across queries.
pub struct GpuSearchIndex {
    /// All posting list doc_ids concatenated
    flat_doc_ids: CudaSlice<u32>,
    /// All posting list TFs concatenated
    flat_tfs: CudaSlice<u32>,
    /// Start offset per term_id into flat arrays
    offsets: Vec<u64>,
    /// Entry count per term_id
    counts: Vec<u32>,
    /// Document lengths on GPU
    doc_lengths: CudaSlice<u32>,
    /// Score accumulator (reused across queries)
    scores: CudaSlice<f32>,
    /// Buffer for non-zero doc_ids (for top-k) — reserved for future GPU top-k
    #[allow(dead_code)]
    result_doc_ids: CudaSlice<u32>,
    /// Buffer for non-zero scores (for top-k)
    #[allow(dead_code)]
    result_scores: CudaSlice<f32>,
    /// Atomic counter for collect_nonzero
    #[allow(dead_code)]
    result_count: CudaSlice<u32>,
    /// Top-k per-block output buffers
    topk_block_doc_ids: CudaSlice<u32>,
    topk_block_scores: CudaSlice<f32>,
    topk_num_blocks: u32,
    /// Number of documents
    num_docs: u32,
}

impl GpuSearchIndex {
    /// Upload a BM25 index to GPU memory.
    pub fn from_index(
        ctx: &CudaIndexer,
        postings: &[Vec<(u32, u32)>],
        doc_lengths: &[u32],
        num_docs: u32,
    ) -> Result<Self, String> {
        // Flatten posting lists into contiguous arrays
        let total_entries: usize = postings.iter().map(|p| p.len()).sum();
        let mut flat_doc_ids = Vec::with_capacity(total_entries);
        let mut flat_tfs = Vec::with_capacity(total_entries);
        let mut offsets = Vec::with_capacity(postings.len());
        let mut counts = Vec::with_capacity(postings.len());
        let mut offset = 0u64;

        for plist in postings {
            offsets.push(offset);
            counts.push(plist.len() as u32);
            for &(doc_id, tf) in plist {
                flat_doc_ids.push(doc_id);
                flat_tfs.push(tf);
            }
            offset += plist.len() as u64;
        }

        let stream = &ctx.stream;

        // Upload to GPU
        let d_flat_doc_ids = stream
            .clone_htod(&flat_doc_ids)
            .map_err(|e| format!("{:?}", e))?;
        let d_flat_tfs = stream
            .clone_htod(&flat_tfs)
            .map_err(|e| format!("{:?}", e))?;
        let d_doc_lengths = stream
            .clone_htod(doc_lengths)
            .map_err(|e| format!("{:?}", e))?;

        // Allocate score buffer + result buffers
        let d_scores: CudaSlice<f32> = stream
            .alloc_zeros(num_docs as usize)
            .map_err(|e| format!("{:?}", e))?;
        // For common English queries on 8.8M docs, up to 5M docs may be scored
        let max_results = num_docs as usize;
        let d_result_doc_ids: CudaSlice<u32> = stream
            .alloc_zeros(max_results)
            .map_err(|e| format!("{:?}", e))?;
        let d_result_scores: CudaSlice<f32> = stream
            .alloc_zeros(max_results)
            .map_err(|e| format!("{:?}", e))?;
        let d_result_count: CudaSlice<u32> =
            stream.alloc_zeros(1).map_err(|e| format!("{:?}", e))?;

        // Top-k buffers: use 256 blocks, each producing up to 1024 top-k candidates
        let topk_num_blocks = 256u32;
        let topk_buf_size = topk_num_blocks as usize * 1024; // max k=1024
        let d_topk_doc_ids: CudaSlice<u32> = stream
            .alloc_zeros(topk_buf_size)
            .map_err(|e| format!("{:?}", e))?;
        let d_topk_scores: CudaSlice<f32> = stream
            .alloc_zeros(topk_buf_size)
            .map_err(|e| format!("{:?}", e))?;

        Ok(GpuSearchIndex {
            flat_doc_ids: d_flat_doc_ids,
            flat_tfs: d_flat_tfs,
            offsets,
            counts,
            doc_lengths: d_doc_lengths,
            scores: d_scores,
            result_doc_ids: d_result_doc_ids,
            result_scores: d_result_scores,
            result_count: d_result_count,
            topk_block_doc_ids: d_topk_doc_ids,
            topk_block_scores: d_topk_scores,
            topk_num_blocks,
            num_docs,
        })
    }

    /// Search on GPU. Returns (doc_id, score) pairs sorted by descending score.
    ///
    /// Uses a fused kernel that scores ALL query terms in a single launch,
    /// eliminating per-term kernel launch overhead (~0.1ms each).
    pub fn search(
        &mut self,
        ctx: &CudaIndexer,
        query_term_ids: &[(u32, f32)], // (term_id, idf)
        k1: f32,
        b: f32,
        avgdl: f32,
        k: usize,
    ) -> Result<Vec<(u32, f32)>, String> {
        let block = 256u32;
        let grid_docs = (self.num_docs).div_ceil(block);
        let n_docs = self.num_docs as i32;

        // 1. Zero scores array
        unsafe {
            ctx.stream
                .launch_builder(&ctx.zero_f32_fn)
                .arg(&mut self.scores)
                .arg(&n_docs)
                .launch(LaunchConfig {
                    block_dim: (block, 1, 1),
                    grid_dim: (grid_docs, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| format!("zero scores: {:?}", e))?;
        }

        // 2. Build fused kernel params: flat_offsets, virtual_starts, idfs
        let mut flat_offsets: Vec<i64> = Vec::new();
        let mut virtual_starts: Vec<i64> = Vec::new();
        let mut idfs: Vec<f32> = Vec::new();
        let mut total_entries: i64 = 0;

        for &(term_id, idf_val) in query_term_ids {
            let tid = term_id as usize;
            if tid >= self.counts.len() || self.counts[tid] == 0 {
                continue;
            }
            flat_offsets.push(self.offsets[tid] as i64);
            virtual_starts.push(total_entries);
            idfs.push(idf_val);
            total_entries += self.counts[tid] as i64;
        }

        if total_entries == 0 {
            return Ok(Vec::new());
        }

        let num_terms = flat_offsets.len() as i32;
        let total = total_entries as i32;
        let grid = (total as u32).div_ceil(block);

        // Upload small per-query arrays (typically <100 bytes)
        let d_flat_offsets = ctx
            .stream
            .clone_htod(&flat_offsets)
            .map_err(|e| format!("{:?}", e))?;
        let d_virtual_starts = ctx
            .stream
            .clone_htod(&virtual_starts)
            .map_err(|e| format!("{:?}", e))?;
        let d_idfs = ctx
            .stream
            .clone_htod(&idfs)
            .map_err(|e| format!("{:?}", e))?;

        // Single fused kernel launch for ALL query terms
        unsafe {
            ctx.stream
                .launch_builder(&ctx.bm25_score_fused_fn)
                .arg(&self.flat_doc_ids)
                .arg(&self.flat_tfs)
                .arg(&self.doc_lengths)
                .arg(&mut self.scores)
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
                .map_err(|e| format!("bm25_score_fused: {:?}", e))?;
        }

        // 3. GPU top-k: each block finds local top-k, download block results, merge on CPU
        let k_clamped = k.min(1024) as i32; // kernel supports max k=1024
        let topk_blocks = self.topk_num_blocks;
        let topk_block_size = 256u32; // threads per block

        unsafe {
            ctx.stream
                .launch_builder(&ctx.topk_per_block_fn)
                .arg(&self.scores)
                .arg(&mut self.topk_block_doc_ids)
                .arg(&mut self.topk_block_scores)
                .arg(&n_docs)
                .arg(&k_clamped)
                .launch(LaunchConfig {
                    block_dim: (topk_block_size, 1, 1),
                    grid_dim: (topk_blocks, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| format!("topk_per_block: {:?}", e))?;
        }

        // 4. Download only num_blocks * k results (small: 256 * k * 8 bytes)
        let download_count = topk_blocks as usize * k_clamped as usize;
        let block_doc_ids: Vec<u32> = ctx
            .stream
            .clone_dtoh(&self.topk_block_doc_ids.slice(..download_count))
            .map_err(|e| format!("{:?}", e))?;
        let block_scores: Vec<f32> = ctx
            .stream
            .clone_dtoh(&self.topk_block_scores.slice(..download_count))
            .map_err(|e| format!("{:?}", e))?;

        // 5. CPU merge: top-k across all block results (small: 256*k entries)
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
