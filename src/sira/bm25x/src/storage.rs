// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write as IoWrite};
use std::path::Path;

use rustc_hash::FxHashMap;

use memmap2::Mmap;

use crate::index::BM25;
use crate::scoring::Method;
use crate::tokenizer::TokenizerMode;

const MAGIC: u64 = 0x424D32355253; // "BM25RS" in hex
/// Bumped from 1 → 2 to capture `max_n`, `n_features`, `tokenizer_mode_id`,
/// and `use_stopwords` — without these, `BM25::load` previously hardcoded
/// `TokenizerMode::UnicodeStem` + `use_stopwords=true`, silently breaking
/// queries on indices built with any other config. With multi-gram, this
/// becomes critical (a Plain-built index queried through a UnicodeStem load
/// would miss every n-gram lookup). v1 indices keep their 56-byte layout;
/// `load` dispatches on `version` and back-fills defaults for v1.
///
/// Bumped from 2 → 3 to capture `ngram_mask` (bit `n-1` set ⇒ n contributes
/// to BM25 scoring). v2 had only `max_n`, which implied a contiguous
/// `1..=max_n` set; v3 supports arbitrary subsets like `{1, 4}` (unigrams +
/// 4-grams, skipping bigrams + trigrams). v3 fits in the same 56-byte header
/// by stealing one byte from the v2 trailing `_pad: [u8; 4]`. v2 indices
/// are read with `ngram_mask` derived from `max_n`.
const VERSION: u32 = 3;

/// On-disk posting entry: (doc_id, tf)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PostingEntry {
    pub doc_id: u32,
    pub tf: u32,
}

/// On-disk term offset: (byte_offset_in_postings, count)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TermOffset {
    pub offset: u64,
    pub count: u32,
    _pad: u32,
}

/// Header stored on disk (v3 layout). With `#[repr(C)]` and natural 8-byte
/// alignment (due to `u64` fields), the actual size is **56 bytes** —
/// unchanged from v2 (the new `ngram_mask` byte steals from v2's 4-byte tail
/// pad, leaving 3 bytes of padding to keep struct alignment).
///
/// ```text
///   0..8   magic (u64)
///   8..12  version (u32)
///   12..16 num_terms (u32)
///   16..20 next_doc_id (u32)
///   20..24 num_docs (u32)
///   24..32 total_tokens (u64, 8-aligned)
///   32..36 k1_bits
///   36..40 b_bits
///   40..44 delta_bits
///   44..45 method_id
///   45..46 max_n              [v2 NEW] — derived from ngram_mask in v3
///   46..47 tokenizer_mode_id  [v2 NEW]
///   47..48 use_stopwords      [v2 NEW]
///   48..52 n_features (u32)   [v2 NEW]
///   52..53 ngram_mask         [v3 NEW] — bit `n-1` set ⇒ n contributes to score
///   53..56 _pad (3 bytes, struct alignment to 8)
/// ```
///
/// `tokenizer_mode_id` is encoded via [`TokenizerMode::to_id`] — keep that
/// numbering stable forever (it's on disk). v1 → v2 grew by 8 bytes (the four
/// new fields including their natural padding); v2 → v3 is byte-compatible at
/// the struct level — the version field disambiguates how to interpret byte
/// 52 (v2: pad/zero; v3: mask). This assumes the on-disk reader/writer are
/// on the same endianness, which is fine for our targets.
#[repr(C)]
struct Header {
    magic: u64,
    version: u32,
    num_terms: u32,
    next_doc_id: u32,
    num_docs: u32,
    total_tokens: u64,
    k1_bits: u32,
    b_bits: u32,
    delta_bits: u32,
    method_id: u8,
    max_n: u8,                   // v2: NEW
    tokenizer_mode_id: u8,       // v2: NEW
    use_stopwords: u8,           // v2: NEW (0 = off, 1 = on)
    n_features: u32,             // v2: NEW (0 when no n-gram tier allocated)
    ngram_mask: u8,              // v3: NEW (bit `n-1` set ⇒ n in score set)
    _pad: [u8; 3],
}

/// v1 header layout (48 bytes after `#[repr(C)]` natural alignment). Read-only
/// — `save` always writes v3 now. Kept verbatim so `BM25::load` can detect via
/// `version` and back-fill defaults (`max_n = 1`, `tokenizer_mode =
/// UnicodeStem`, `use_stopwords = true`, `n_features = 0`) — the latter two
/// preserve the exact legacy behavior.
#[repr(C)]
struct HeaderV1 {
    magic: u64,
    version: u32,
    num_terms: u32,
    next_doc_id: u32,
    num_docs: u32,
    total_tokens: u64,
    k1_bits: u32,
    b_bits: u32,
    delta_bits: u32,
    method_id: u8,
    _pad: [u8; 3],
}

/// Memory-mapped data backing an index.
pub struct MmapData {
    doc_lens_mmap: Mmap,
    postings_mmap: Mmap,
    offsets_mmap: Mmap,
    num_terms: u32,
    _max_doc_id: u32,
}

impl MmapData {
    pub fn get_doc_length(&self, doc_id: u32) -> u32 {
        let idx = doc_id as usize;
        let slice = unsafe {
            std::slice::from_raw_parts(
                self.doc_lens_mmap.as_ptr() as *const u32,
                self.doc_lens_mmap.len() / 4,
            )
        };
        *slice.get(idx).unwrap_or(&0)
    }

    pub fn all_doc_lengths(&self) -> Vec<u32> {
        let slice = unsafe {
            std::slice::from_raw_parts(
                self.doc_lens_mmap.as_ptr() as *const u32,
                self.doc_lens_mmap.len() / 4,
            )
        };
        slice.to_vec()
    }

    pub fn posting_count(&self, term_id: u32) -> u32 {
        if term_id >= self.num_terms {
            return 0;
        }
        let offsets = unsafe {
            std::slice::from_raw_parts(
                self.offsets_mmap.as_ptr() as *const TermOffset,
                self.num_terms as usize,
            )
        };
        offsets[term_id as usize].count
    }

    pub fn for_each_posting<F: FnMut(u32, u32)>(&self, term_id: u32, f: &mut F) {
        if term_id >= self.num_terms {
            return;
        }
        let offsets = unsafe {
            std::slice::from_raw_parts(
                self.offsets_mmap.as_ptr() as *const TermOffset,
                self.num_terms as usize,
            )
        };
        let to = &offsets[term_id as usize];
        let entries = unsafe {
            let base = self.postings_mmap.as_ptr().add(to.offset as usize) as *const PostingEntry;
            std::slice::from_raw_parts(base, to.count as usize)
        };
        for entry in entries {
            f(entry.doc_id, entry.tf);
        }
    }

    pub fn doc_lens_bytes(&self) -> &[u8] {
        &self.doc_lens_mmap
    }

    pub fn postings_bytes(&self) -> &[u8] {
        &self.postings_mmap
    }

    pub fn offsets_bytes(&self) -> &[u8] {
        &self.offsets_mmap
    }

    /// Binary search for a specific doc_id in a term's posting list.
    pub fn get_tf(&self, term_id: u32, doc_id: u32) -> Option<u32> {
        if term_id >= self.num_terms {
            return None;
        }
        let offsets = unsafe {
            std::slice::from_raw_parts(
                self.offsets_mmap.as_ptr() as *const TermOffset,
                self.num_terms as usize,
            )
        };
        let to = &offsets[term_id as usize];
        let entries = unsafe {
            let base = self.postings_mmap.as_ptr().add(to.offset as usize) as *const PostingEntry;
            std::slice::from_raw_parts(base, to.count as usize)
        };
        entries
            .binary_search_by_key(&doc_id, |e| e.doc_id)
            .ok()
            .map(|idx| entries[idx].tf)
    }
}

impl BM25 {
    /// Save the index to a directory.
    pub fn save<P: AsRef<Path>>(&self, dir: P) -> io::Result<()> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;

        // Write header (v3 — captures ngram_mask in addition to v2 fields).
        // ngram_mask: bit `n-1` set ⇒ n contributes to BM25 scoring.
        let mut ngram_mask: u8 = 0;
        for &n in self.ngram_set() {
            ngram_mask |= 1u8 << (n - 1);
        }
        let header = Header {
            magic: MAGIC,
            version: VERSION,
            num_terms: self.get_vocab().len() as u32,
            next_doc_id: self.get_num_docs(),
            num_docs: self.get_num_docs(),
            total_tokens: self.get_total_tokens(),
            k1_bits: self.k1.to_bits(),
            b_bits: self.b.to_bits(),
            delta_bits: self.delta.to_bits(),
            method_id: self.method.to_id(),
            max_n: self.max_n(),
            tokenizer_mode_id: self.tokenizer_mode().to_id(),
            use_stopwords: if self.use_stopwords() { 1 } else { 0 },
            n_features: self.n_features(),
            ngram_mask,
            _pad: [0; 3],
        };
        write_bytes(dir.join("header.bin"), as_bytes(&header))?;

        if self.get_mmap_data().is_some() {
            // Mmap-backed: the binary files on disk are already up to date.
            // Only metadata (header, vocab) needs rewriting — done below.
        } else {
            // In-memory path: serialize from Vec data
            write_bytes(
                dir.join("doc_lens.bin"),
                as_slice_bytes(self.get_doc_lengths_slice()),
            )?;

            let num_terms = self.get_vocab().len();
            let postings = self.get_postings();
            let mut flat_postings: Vec<PostingEntry> = Vec::new();
            let mut offsets: Vec<TermOffset> = Vec::with_capacity(num_terms);

            for entries in postings.iter().take(num_terms) {
                let byte_offset = flat_postings.len() * std::mem::size_of::<PostingEntry>();
                let count = entries.len() as u32;
                for &(doc_id, tf) in entries {
                    flat_postings.push(PostingEntry { doc_id, tf });
                }
                offsets.push(TermOffset {
                    offset: byte_offset as u64,
                    count,
                    _pad: 0,
                });
            }

            write_bytes(dir.join("postings.bin"), as_slice_bytes(&flat_postings))?;
            write_bytes(dir.join("offsets.bin"), as_slice_bytes(&offsets))?;

            // Write positions (parallel to postings)
            let positions = self.get_positions();
            let mut flat_positions: Vec<u32> = Vec::new();
            let mut pos_offsets: Vec<TermOffset> = Vec::with_capacity(num_terms);

            for (term_id, term_positions) in positions.iter().enumerate().take(num_terms) {
                let postings_count = if term_id < postings.len() {
                    postings[term_id].len()
                } else {
                    0
                };
                // For each posting in this term, write its positions
                for posting_idx in 0..postings_count {
                    let byte_offset = flat_positions.len() * std::mem::size_of::<u32>();
                    let pos_list = if posting_idx < term_positions.len() {
                        &term_positions[posting_idx]
                    } else {
                        &[] as &[u32]
                    };
                    let count = pos_list.len() as u32;
                    flat_positions.extend_from_slice(pos_list);
                    pos_offsets.push(TermOffset {
                        offset: byte_offset as u64,
                        count,
                        _pad: 0,
                    });
                }
            }

            write_bytes(dir.join("positions.bin"), as_slice_bytes(&flat_positions))?;
            write_bytes(dir.join("pos_offsets.bin"), as_slice_bytes(&pos_offsets))?;
        }

        // Persist the hashed n-gram tier to its own family of files
        // (`ngram_*.bin`). Mirrors the unigram tier's flat layout:
        // - ngram_postings.bin    : flat Vec<PostingEntry>
        // - ngram_offsets.bin     : Vec<TermOffset>, length n_features (one per slot)
        // - ngram_positions.bin   : flat Vec<u32> of unigram start positions
        // - ngram_pos_offsets.bin : Vec<TermOffset>, one per posting across all slots
        //
        // Skipped for mmap-backed indices for the same reason the unigram
        // mmap path is skipped: the on-disk files are the source of truth
        // (and there's no incremental n-gram tier mutation in this path).
        if let Some(side) = self.ngram_side.as_ref() {
            if self.get_mmap_data().is_none() {
                let n_features = side.n_features as usize;

                // Flatten n-gram postings in slot order.
                let mut flat_postings: Vec<PostingEntry> = Vec::new();
                let mut offsets: Vec<TermOffset> = Vec::with_capacity(n_features);
                for plist in side.postings.iter() {
                    let byte_offset = flat_postings.len() * std::mem::size_of::<PostingEntry>();
                    let count = plist.len() as u32;
                    for &(doc_id, tf) in plist {
                        flat_postings.push(PostingEntry { doc_id, tf });
                    }
                    offsets.push(TermOffset {
                        offset: byte_offset as u64,
                        count,
                        _pad: 0,
                    });
                }
                write_bytes(dir.join("ngram_postings.bin"), as_slice_bytes(&flat_postings))?;
                write_bytes(dir.join("ngram_offsets.bin"), as_slice_bytes(&offsets))?;

                // Flatten positions in (slot, posting_idx) order — emit one
                // TermOffset per posting (mirrors the unigram pos_offsets layout).
                let mut flat_positions: Vec<u32> = Vec::new();
                let mut pos_offsets: Vec<TermOffset> = Vec::new();
                for (slot, term_positions) in side.positions.iter().enumerate() {
                    let posting_count = side.postings[slot].len();
                    for posting_idx in 0..posting_count {
                        let byte_offset = flat_positions.len() * std::mem::size_of::<u32>();
                        let pos_list: &[u32] = if posting_idx < term_positions.len() {
                            &term_positions[posting_idx]
                        } else {
                            &[]
                        };
                        flat_positions.extend_from_slice(pos_list);
                        pos_offsets.push(TermOffset {
                            offset: byte_offset as u64,
                            count: pos_list.len() as u32,
                            _pad: 0,
                        });
                    }
                }
                write_bytes(dir.join("ngram_positions.bin"), as_slice_bytes(&flat_positions))?;
                write_bytes(dir.join("ngram_pos_offsets.bin"), as_slice_bytes(&pos_offsets))?;
            }
        }

        // Write vocab (convert FxHashMap → HashMap for bincode compatibility)
        let std_vocab: HashMap<&str, u32> = self.get_vocab().iter()
            .map(|(k, &v)| (k.as_str(), v))
            .collect();
        let vocab_bytes = bincode::serialize(&std_vocab).map_err(io::Error::other)?;
        fs::write(dir.join("vocab.bin"), &vocab_bytes)?;

        Ok(())
    }

    /// Load an index from a directory. If `mmap` is true, postings and doc_lengths
    /// are memory-mapped instead of loaded into RAM.
    ///
    /// Dispatches on header `version`:
    /// - v1 (56-byte header): tokenizer config not persisted; back-fills
    ///   `TokenizerMode::UnicodeStem` + `use_stopwords=true` + `max_n=1` to
    ///   preserve the legacy hardcoded behavior bit-for-bit.
    /// - v2 (64-byte header): reads `max_n`, `n_features`, `tokenizer_mode_id`,
    ///   `use_stopwords` directly; the n-gram set is the contiguous `1..=max_n`.
    /// - v3 (64-byte header): as v2, plus `ngram_mask`, which allows a
    ///   non-contiguous n-gram set (e.g. unigrams + 4-grams, skipping 2 and 3).
    ///
    /// For v2 and v3 the n-gram tier files (`ngram_*.bin`) are read back into
    /// the [`NgramSide`] as well, except under `mmap` (see below).
    pub fn load<P: AsRef<Path>>(dir: P, mmap: bool) -> io::Result<Self> {
        let dir = dir.as_ref();

        // Read header — peek magic + version BEFORE assuming a layout, since
        // v1 and v2 differ in size (56 vs 64 bytes).
        let header_bytes = fs::read(dir.join("header.bin"))?;
        if header_bytes.len() < 16 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "header too small"));
        }
        let magic = u64::from_ne_bytes(header_bytes[0..8].try_into().unwrap());
        let version = u32::from_ne_bytes(header_bytes[8..12].try_into().unwrap());
        if magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic number"));
        }

        // Decode the layout-specific fields. We bind into one tuple to keep the
        // downstream code identical for both versions. `ngram_mask` is a bitmask
        // (bit `n-1` set ⇒ n contributes to score); for v1/v2 it's derived from
        // `max_n` to preserve the old contiguous-`1..=max_n` semantic.
        #[allow(clippy::type_complexity)]
        let (num_terms, next_doc_id, num_docs, total_tokens, k1, b, delta,
             method_id, max_n, tokenizer_mode, use_stopwords, n_features, ngram_mask):
            (u32, u32, u32, u64, f32, f32, f32, u8, u8, TokenizerMode, bool, u32, u8) = match version {
            1 => {
                if header_bytes.len() < std::mem::size_of::<HeaderV1>() {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "v1 header truncated"));
                }
                let h: HeaderV1 = unsafe { std::ptr::read_unaligned(header_bytes.as_ptr() as *const HeaderV1) };
                // v1 didn't persist tokenizer config — back-fill to UnicodeStem +
                // stopwords=true to match the old `BM25::load` hardcode exactly.
                (h.num_terms, h.next_doc_id, h.num_docs, h.total_tokens,
                 f32::from_bits(h.k1_bits), f32::from_bits(h.b_bits),
                 f32::from_bits(h.delta_bits), h.method_id,
                 1u8, TokenizerMode::UnicodeStem, true, 0u32,
                 mask_from_max_n(1))
            }
            2 => {
                if header_bytes.len() < std::mem::size_of::<Header>() {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "v2 header truncated"));
                }
                let h: Header = unsafe { std::ptr::read_unaligned(header_bytes.as_ptr() as *const Header) };
                let tok_mode = TokenizerMode::from_id(h.tokenizer_mode_id).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "unknown tokenizer mode")
                })?;
                // v2 implied contiguous `1..=max_n`. Ignore the byte that v3
                // calls `ngram_mask` (it's structurally the v2 _pad, may hold
                // garbage from old writers — though the saver always wrote 0).
                (h.num_terms, h.next_doc_id, h.num_docs, h.total_tokens,
                 f32::from_bits(h.k1_bits), f32::from_bits(h.b_bits),
                 f32::from_bits(h.delta_bits), h.method_id,
                 h.max_n, tok_mode, h.use_stopwords != 0, h.n_features,
                 mask_from_max_n(h.max_n))
            }
            3 => {
                if header_bytes.len() < std::mem::size_of::<Header>() {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "v3 header truncated"));
                }
                let h: Header = unsafe { std::ptr::read_unaligned(header_bytes.as_ptr() as *const Header) };
                let tok_mode = TokenizerMode::from_id(h.tokenizer_mode_id).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "unknown tokenizer mode")
                })?;
                if h.ngram_mask == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "v3 header has empty ngram_mask (no n's selected)",
                    ));
                }
                (h.num_terms, h.next_doc_id, h.num_docs, h.total_tokens,
                 f32::from_bits(h.k1_bits), f32::from_bits(h.b_bits),
                 f32::from_bits(h.delta_bits), h.method_id,
                 h.max_n, tok_mode, h.use_stopwords != 0, h.n_features, h.ngram_mask)
            }
            v => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported header version: {}", v),
                ))
            }
        };
        let ngrams = ngrams_from_mask(ngram_mask);
        // `max_n` is read-only debug context here — the canonical source of
        // truth on disk for the n-gram structure is `ngram_mask` (v3) or the
        // derived contiguous set (v1/v2). Bind to suppress the unused warning
        // without losing the value for any future debug logging.
        let _max_n_legacy = max_n;

        let method = Method::from_id(method_id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "unknown method"))?;

        // Read vocab (deserialize as HashMap, convert to FxHashMap for fast lookups)
        let vocab_bytes = fs::read(dir.join("vocab.bin"))?;
        let std_vocab: HashMap<String, u32> =
            bincode::deserialize(&vocab_bytes).map_err(io::Error::other)?;
        let vocab: FxHashMap<String, u32> = std_vocab.into_iter().collect();

        // Construct using the persisted tokenizer config + the (possibly
        // non-contiguous) ngram set. v3 indices may have e.g. `ngrams=[1, 4]`;
        // v1/v2 always have a contiguous `1..=max_n` set. The empty NgramSide
        // is allocated when any n>=2 is selected; the n-gram tier files
        // populate it below.
        let mut index = BM25::with_options_ngrams(
            method, k1, b, delta, tokenizer_mode, use_stopwords, false, ngrams, n_features,
        );

        if mmap {
            let doc_lens_file = File::open(dir.join("doc_lens.bin"))?;
            let postings_file = File::open(dir.join("postings.bin"))?;
            let offsets_file = File::open(dir.join("offsets.bin"))?;

            let doc_lens_mmap = unsafe { Mmap::map(&doc_lens_file)? };
            let postings_mmap = unsafe { Mmap::map(&postings_file)? };
            let offsets_mmap = unsafe { Mmap::map(&offsets_file)? };

            let mmap_data = MmapData {
                doc_lens_mmap,
                postings_mmap,
                offsets_mmap,
                num_terms,
                _max_doc_id: next_doc_id,
            };

            index.set_mmap_internals(vocab, total_tokens, num_docs, mmap_data);
        } else {
            // Read everything into memory
            let doc_lens_bytes = fs::read(dir.join("doc_lens.bin"))?;
            let doc_lengths: Vec<u32> = bytes_to_vec(&doc_lens_bytes);

            let postings_bytes = fs::read(dir.join("postings.bin"))?;
            let flat_postings: Vec<PostingEntry> = bytes_to_vec_posting(&postings_bytes);

            let offsets_bytes = fs::read(dir.join("offsets.bin"))?;
            let flat_offsets: Vec<TermOffset> = bytes_to_vec_offset(&offsets_bytes);

            // Reconstruct posting lists
            let num_terms = num_terms as usize;
            let mut postings = Vec::with_capacity(num_terms);
            for to in flat_offsets.iter().take(num_terms) {
                let start = to.offset as usize / std::mem::size_of::<PostingEntry>();
                let entries: Vec<(u32, u32)> = flat_postings[start..start + to.count as usize]
                    .iter()
                    .map(|e| (e.doc_id, e.tf))
                    .collect();
                postings.push(entries);
            }

            // Load positions if available
            let positions = if dir.join("positions.bin").exists()
                && dir.join("pos_offsets.bin").exists()
            {
                let pos_bytes = fs::read(dir.join("positions.bin"))?;
                let flat_pos: Vec<u32> = bytes_to_vec(&pos_bytes);
                let pos_off_bytes = fs::read(dir.join("pos_offsets.bin"))?;
                let flat_pos_offsets: Vec<TermOffset> = bytes_to_vec_offset(&pos_off_bytes);

                let mut positions: Vec<Vec<Vec<u32>>> = Vec::with_capacity(num_terms);
                let mut pos_off_idx = 0;
                for term_id in 0..num_terms {
                    let posting_count = postings[term_id].len();
                    let mut term_positions = Vec::with_capacity(posting_count);
                    for _ in 0..posting_count {
                        if pos_off_idx < flat_pos_offsets.len() {
                            let po = &flat_pos_offsets[pos_off_idx];
                            let start = po.offset as usize / std::mem::size_of::<u32>();
                            let count = po.count as usize;
                            let pos_list = flat_pos[start..start + count].to_vec();
                            term_positions.push(pos_list);
                            pos_off_idx += 1;
                        } else {
                            term_positions.push(Vec::new());
                        }
                    }
                    positions.push(term_positions);
                }
                positions
            } else {
                vec![Vec::new(); num_terms]
            };

            index.set_internals(
                vocab,
                doc_lengths,
                postings,
                positions,
                total_tokens,
                num_docs,
            );
        }

        // Populate the hashed n-gram tier from disk. The empty side was already
        // allocated by `with_options_ngrams` above when any n >= 2 is selected.
        // Skipped for mmap-backed loads — TODO(follow-up): mmap support for the
        // n-gram tier (the in-memory branch above gates this anyway).
        if !mmap && max_n >= 2 && n_features > 0 {
            let ngram_postings_path = dir.join("ngram_postings.bin");
            let ngram_offsets_path = dir.join("ngram_offsets.bin");
            if ngram_postings_path.exists() && ngram_offsets_path.exists() {
                let postings_bytes = fs::read(&ngram_postings_path)?;
                let flat_postings: Vec<PostingEntry> = bytes_to_vec_posting(&postings_bytes);
                let offsets_bytes = fs::read(&ngram_offsets_path)?;
                let flat_offsets: Vec<TermOffset> = bytes_to_vec_offset(&offsets_bytes);

                // Direct mutation OK: NgramSide fields are pub(crate) and we
                // share the bm25x crate boundary.
                let side = index
                    .ngram_side
                    .as_mut()
                    .expect("max_n>=2 must allocate side via with_options_full");
                for (slot_idx, to) in flat_offsets.iter().enumerate().take(n_features as usize) {
                    let start = to.offset as usize / std::mem::size_of::<PostingEntry>();
                    let entries: Vec<(u32, u32)> = flat_postings
                        [start..start + to.count as usize]
                        .iter()
                        .map(|e| (e.doc_id, e.tf))
                        .collect();
                    side.doc_freqs[slot_idx] = entries.len() as u32;
                    side.postings[slot_idx] = entries;
                }

                // Load positions if persisted (parallel to postings, one
                // TermOffset per posting across all slots).
                let ngram_positions_path = dir.join("ngram_positions.bin");
                let ngram_pos_offsets_path = dir.join("ngram_pos_offsets.bin");
                if ngram_positions_path.exists() && ngram_pos_offsets_path.exists() {
                    let pos_bytes = fs::read(&ngram_positions_path)?;
                    let flat_positions: Vec<u32> = bytes_to_vec(&pos_bytes);
                    let pos_off_bytes = fs::read(&ngram_pos_offsets_path)?;
                    let flat_pos_offsets: Vec<TermOffset> = bytes_to_vec_offset(&pos_off_bytes);
                    let mut pos_off_idx = 0;
                    for slot_idx in 0..(n_features as usize) {
                        let posting_count = side.postings[slot_idx].len();
                        let mut term_positions: Vec<Vec<u32>> = Vec::with_capacity(posting_count);
                        for _ in 0..posting_count {
                            if pos_off_idx < flat_pos_offsets.len() {
                                let po = &flat_pos_offsets[pos_off_idx];
                                let start = po.offset as usize / std::mem::size_of::<u32>();
                                let count = po.count as usize;
                                term_positions.push(flat_positions[start..start + count].to_vec());
                                pos_off_idx += 1;
                            } else {
                                term_positions.push(Vec::new());
                            }
                        }
                        side.positions[slot_idx] = term_positions;
                    }
                }
            }
        }

        Ok(index)
    }
}

// --- Helper functions ---

/// Bit `n-1` set ⇒ n contributes to BM25 score. Convert a contiguous
/// `1..=max_n` set (v1/v2 implicit semantic) into the v3 mask.
fn mask_from_max_n(max_n: u8) -> u8 {
    if max_n == 0 {
        return 0;
    }
    let m = max_n.min(8);
    // Bits 0..m-1 set: e.g. max_n=4 → 0b0000_1111 = 15.
    ((1u16 << m) - 1) as u8
}

/// Inverse of `mask_from_max_n` for arbitrary subsets — returns the sorted
/// list of n values (each 1..=8) encoded by the mask.
fn ngrams_from_mask(mask: u8) -> Vec<u8> {
    (1u8..=8).filter(|&n| (mask & (1u8 << (n - 1))) != 0).collect()
}

fn write_bytes<P: AsRef<Path>>(path: P, data: &[u8]) -> io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(data)?;
    f.flush()?;
    Ok(())
}

fn as_bytes<T>(val: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(val as *const T as *const u8, std::mem::size_of::<T>()) }
}

fn as_slice_bytes<T>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

fn bytes_to_vec(bytes: &[u8]) -> Vec<u32> {
    let count = bytes.len() / 4;
    let mut v = Vec::with_capacity(count);
    for i in 0..count {
        let b = [
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ];
        v.push(u32::from_ne_bytes(b));
    }
    v
}

fn bytes_to_vec_posting(bytes: &[u8]) -> Vec<PostingEntry> {
    let size = std::mem::size_of::<PostingEntry>();
    let count = bytes.len() / size;
    let mut v = Vec::with_capacity(count);
    for i in 0..count {
        let offset = i * size;
        let doc_id = u32::from_ne_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        let tf = u32::from_ne_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);
        v.push(PostingEntry { doc_id, tf });
    }
    v
}

fn bytes_to_vec_offset(bytes: &[u8]) -> Vec<TermOffset> {
    let size = std::mem::size_of::<TermOffset>();
    let count = bytes.len() / size;
    let mut v = Vec::with_capacity(count);
    for i in 0..count {
        let offset = i * size;
        let o = u64::from_ne_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);
        let c = u32::from_ne_bytes([
            bytes[offset + 8],
            bytes[offset + 9],
            bytes[offset + 10],
            bytes[offset + 11],
        ]);
        v.push(TermOffset {
            offset: o,
            count: c,
            _pad: 0,
        });
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_save_and_load() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&[
            "the quick brown fox",
            "the lazy dog",
            "a brown dog and a quick fox",
        ]);

        let dir = TempDir::new().unwrap();
        index.save(dir.path()).unwrap();

        let loaded = BM25::load(dir.path(), false).unwrap();
        assert_eq!(loaded.len(), 3);

        let results = loaded.search("quick fox", 10);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_save_and_load_mmap() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&[
            "the quick brown fox",
            "the lazy dog",
            "a brown dog and a quick fox",
        ]);

        let dir = TempDir::new().unwrap();
        index.save(dir.path()).unwrap();

        let loaded = BM25::load(dir.path(), true).unwrap();
        assert_eq!(loaded.len(), 3);

        let results = loaded.search("quick fox", 10);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_save_load_with_deletions() {
        let mut index = BM25::new(Method::Lucene, 1.5, 0.75, 0.5, false);
        index.add(&["hello world", "foo bar", "hello foo"]);
        // Delete doc 1 ("foo bar") — compacts to ["hello world", "hello foo"]
        index.delete(&[1]);

        let dir = TempDir::new().unwrap();
        index.save(dir.path()).unwrap();

        let loaded = BM25::load(dir.path(), false).unwrap();
        assert_eq!(loaded.len(), 2);

        // "foo" is now only in doc 1 (was "hello foo" at old index 2)
        let results = loaded.search("foo", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].index, 1);
    }

    /// Locks down the v3 header byte layout — same total size as v2 (56 bytes,
    /// `#[repr(C)]` natural alignment) since `ngram_mask` reused one byte from
    /// the v2 trailing `_pad: [u8; 4]`. The on-disk format is part of the
    /// public contract; if this changes, bump VERSION and add a HeaderVN.
    #[test]
    fn header_v3_byte_size_is_56() {
        assert_eq!(std::mem::size_of::<Header>(), 56);
    }

    /// Verify mask helpers round-trip for the common cases.
    #[test]
    fn mask_helpers_round_trip() {
        let cases: &[(u8, &[u8])] = &[
            (0b0000_0001u8, &[1u8]),
            (0b0000_0011u8, &[1u8, 2u8]),
            (0b0000_1111u8, &[1u8, 2u8, 3u8, 4u8]),
            (0b0000_1001u8, &[1u8, 4u8]),     // skip bigrams + trigrams
            (0b0000_1010u8, &[2u8, 4u8]),     // bigram + 4-gram only
            (0b0000_0010u8, &[2u8]),          // bigram-only score
        ];
        for &(mask, expected_ns) in cases {
            let got = ngrams_from_mask(mask);
            assert_eq!(got.as_slice(), expected_ns, "ngrams_from_mask({:#010b})", mask);
        }
        assert_eq!(mask_from_max_n(1), 0b0000_0001);
        assert_eq!(mask_from_max_n(4), 0b0000_1111);
        assert_eq!(mask_from_max_n(8), 0b1111_1111);
    }

    /// `#[repr(C)]` packs the v1 layout to 48 bytes, not the 56 the field
    /// widths suggest; this pins that down so a layout change cannot pass
    /// silently and break v1 reads.
    #[test]
    fn header_v1_byte_size_is_48() {
        assert_eq!(std::mem::size_of::<HeaderV1>(), 48);
    }

    #[test]
    fn save_load_v2_preserves_max_n_and_tokenizer_mode() {
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Plain, false, false, /*max_n*/ 3, /*n_features*/ 1 << 16,
        );
        idx.add(&["Hello World Foo Bar"]).unwrap();
        let dir = TempDir::new().unwrap();
        idx.save(dir.path()).unwrap();
        let loaded = BM25::load(dir.path(), false).unwrap();
        assert_eq!(loaded.max_n(), 3);
        assert_eq!(loaded.n_features(), 1 << 16);
        assert_eq!(loaded.tokenizer_mode(), TokenizerMode::Plain);
        assert!(!loaded.use_stopwords());
        // A v2 index carries no `ngram_*.bin`, so only the unigram tier is
        // populated; search must still work off that tier alone.
        let r = loaded.search("hello", 5);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn save_load_v2_max_n_1_no_ngram_side() {
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::UnicodeStem, true, false, 1, 0,
        );
        idx.add(&["alpha beta"]).unwrap();
        let dir = TempDir::new().unwrap();
        idx.save(dir.path()).unwrap();
        let loaded = BM25::load(dir.path(), false).unwrap();
        assert_eq!(loaded.max_n(), 1);
        assert_eq!(loaded.n_features(), 0);
        assert_eq!(loaded.tokenizer_mode(), TokenizerMode::UnicodeStem);
        assert!(loaded.use_stopwords());
    }

    /// End-to-end multi-gram persistence: save → load must reproduce identical
    /// search rankings AND preserve the n-gram side's posting lists. This is
    /// the integration test for n-gram tier persistence.
    #[test]
    fn save_load_round_trip_preserves_ngram_postings() {
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 2, 1 << 16,
        );
        idx.add(&[
            "alpha beta gamma",
            "beta gamma delta",
            "gamma delta epsilon",
        ]).unwrap();
        let r0 = idx.search("alpha beta", 3);
        let dir = TempDir::new().unwrap();
        idx.save(dir.path()).unwrap();
        let loaded = BM25::load(dir.path(), false).unwrap();
        let r1 = loaded.search("alpha beta", 3);
        assert_eq!(r0.len(), r1.len());
        for (a, b) in r0.iter().zip(r1.iter()) {
            assert_eq!(a.index, b.index);
            assert!((a.score - b.score).abs() < 1e-5);
        }
        // n-gram side preserved end-to-end.
        let side = loaded.ngram_side().expect("loaded side present for max_n>=2");
        let slot_ab = side.slot_for("alpha beta");
        assert!(side.df(slot_ab) >= 1, "bigram 'alpha beta' DF preserved");
    }

    /// v3 arbitrary-subset round trip: build a `ngrams=[1, 4]` index (skip
    /// bigrams + trigrams), save, load, and verify both ngram_set and the
    /// 4-gram tier survive — and that the bigram/trigram slots are still
    /// empty after reload.
    #[test]
    fn save_load_v3_arbitrary_subset_round_trip() {
        let mut idx = BM25::with_options_ngrams(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false,
            vec![1u8, 4u8], 1 << 14,
        );
        idx.add(&["alpha beta gamma delta epsilon"]).unwrap();
        let dir = TempDir::new().unwrap();
        idx.save(dir.path()).unwrap();
        let loaded = BM25::load(dir.path(), false).unwrap();

        assert_eq!(loaded.ngram_set(), &[1u8, 4u8]);
        assert_eq!(loaded.max_n(), 4, "max_n derived from largest n in set");
        assert!(loaded.score_unigram(), "n=1 in set ⇒ unigram contributes");

        let side = loaded.ngram_side().expect("side allocated for n>=2 in set");
        assert_eq!(side.ns(), &[4u8], "only 4-grams emitted into side");
        // 4-gram present.
        assert!(side.df(side.slot_for("alpha beta gamma delta")) >= 1);
        // Bigrams must NOT be in the side — they were excluded at index time.
        assert_eq!(
            side.df(side.slot_for("alpha beta")),
            0,
            "bigrams not indexed when 2 ∉ ngram_set",
        );
    }

    /// v2 → v3 backward-compat: an old v2 index with `max_n=2` should load
    /// with `ngram_set = [1, 2]`. (We construct a v2 index by patching the
    /// version field on disk after a v3 save; same bytes since the layout
    /// is unchanged.)
    #[test]
    fn load_v2_derives_contiguous_ngram_set() {
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, /*max_n*/ 2, /*n_features*/ 1 << 14,
        );
        idx.add(&["alpha beta gamma"]).unwrap();
        let dir = TempDir::new().unwrap();
        idx.save(dir.path()).unwrap();
        // Patch on-disk version 3 → 2 to simulate a legacy file. The byte
        // layout is identical; the `ngram_mask` byte will be reinterpreted
        // as the v2 _pad and ignored on load (mask derived from max_n=2).
        let header_path = dir.path().join("header.bin");
        let mut bytes = fs::read(&header_path).unwrap();
        // version is at bytes 8..12 (after the 8-byte magic).
        bytes[8..12].copy_from_slice(&2u32.to_ne_bytes());
        fs::write(&header_path, &bytes).unwrap();

        let loaded = BM25::load(dir.path(), false).unwrap();
        assert_eq!(loaded.ngram_set(), &[1u8, 2u8], "v2 max_n=2 ⇒ contiguous {{1, 2}}");
        assert!(loaded.score_unigram());
    }

    /// 4-gram + stopword + stemming combo: stresses position persistence and
    /// makes sure the (slot, posting_idx) flatten/unflatten loop is symmetric.
    #[test]
    fn save_load_max_n_4_preserves_ngram_tier() {
        let mut idx = BM25::with_options_full(
            Method::Lucene, 1.5, 0.75, 0.5,
            TokenizerMode::Stem, true, false, 4, 1 << 14,
        );
        idx.add(&["the quick brown fox jumps over the lazy dog"]).unwrap();
        let dir = TempDir::new().unwrap();
        idx.save(dir.path()).unwrap();
        let loaded = BM25::load(dir.path(), false).unwrap();
        assert_eq!(loaded.max_n(), 4);
        let side = loaded.ngram_side().expect("loaded side present for max_n>=2");
        // 4-gram "quick brown fox jump" survives (after Stem + stopword removal:
        // the/over → dropped; jumps → "jump").
        let slot = side.slot_for("quick brown fox jump");
        assert!(side.df(slot) >= 1, "4-gram preserved");
    }
}
