// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Hashed n-gram inverted index for DF + co-occurrence lookups.
//!
//! Independent of [`crate::index::BM25`]: this is a side index used by SIRA
//! to (a) filter LLM-suggested keywords down to rare n-grams (DF < threshold)
//! and (b) propose co-occurring n-grams given a small seed query.
//!
//! ## Storage: hashed feature space (mirrors [`crate::tfidf::Tfidf`])
//!
//! Every n-gram is hashed via MurmurHash3 into a fixed-size slot space of
//! `n_features` (default 1<<23 = 8M). Posting lists live in a **dense
//! `Vec<Mutex<Vec<u32>>>` of length `n_features`** — same shape as Tfidf's
//! `Vec<AtomicU32>`, but with a Mutex around each `Vec` so multiple threads
//! can append doc IDs concurrently without a global merge phase.
//!
//! Memory upfront: `n_features × ~48 bytes` (≈ 384 MB at default). Heavy
//! for tests but bounded and consistent at MS-MARCO scale where actual
//! posting data dwarfs this. Per-ngram cost during build is one Mutex
//! lock + a `Vec::push` ≈ 70 ns, vs. ~250 ns for the previous sparse
//! `HashMap.entry().push` approach.
//!
//! ## Collision tradeoff
//!
//! With `n_features` slots and `V` distinct n-grams, every slot holds on
//! average `V / n_features` ngrams. Practical effects:
//!
//! - `df(ngram)` returns the *slot's* DF, which is `>=` the true DF of the
//!   queried n-gram (over-estimate, never under).
//! - `cooccur` counts may include intersection contributions from other
//!   n-grams that hash to the same slot.
//! - At `n_features = 8M` and MS-MARCO vocab ~50M, average collision factor
//!   is ~6×. Increase `n_features` (e.g. to 1<<26 = 64M) to drop noise to
//!   ~1× at the cost of ~8× more memory.
//!
//! ## API
//!
//! `add(&[String])` — index a batch (parallel via rayon, GIL released by Py).
//! `df(ngram)` / `contains(ngram)` — slot-DF lookups.
//! `cooccur(query, top_k, df_max)` — return top-k slots with the highest
//! intersection count, surfaced as the lex-smallest n-gram seen for each slot.
//! `save` / `load` — atomic bincode round-trip (Mutexes stripped on save,
//! re-wrapped on load).

use std::sync::Mutex;

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use crate::ngram::{iter_ngram_slots_with_str, slot_for_ngram_str};
use crate::tokenizer::{Tokenizer, TokenizerMode};

/// Default hash space — 1<<23 = 8M slots. Matches Tfidf's `n_features`
/// default. At this size the dense Mutex array costs ~384 MB upfront;
/// callers can override via `NGramIndex::new`.
pub const DEFAULT_N_FEATURES: usize = 1 << 23;

/// On-disk representation: same as in-memory, but `Vec` instead of
/// `Mutex<Vec>`. Used only for save/load via custom Serialize/Deserialize.
#[derive(Serialize, Deserialize)]
struct NGramIndexOnDisk {
    max_n: usize,
    n_features: usize,
    tokenizer_mode: TokenizerMode,
    use_stopwords: bool,
    num_docs: u32,
    /// Posting lists, indexed by slot. Empty slots stored as empty Vec.
    /// At MS-MARCO scale most slots are filled, so dense storage is fine.
    posting: Vec<Vec<u32>>,
    /// Slot → lex-smallest n-gram string ever inserted there. Sparse;
    /// only populated slots have an entry. ~40-byte overhead per entry.
    vocab_repr: FxHashMap<u32, String>,
}

pub struct NGramIndex {
    max_n: usize,
    n_features: usize,
    mask: u32,
    tokenizer_mode: TokenizerMode,
    use_stopwords: bool,
    num_docs: u32,
    /// Per-slot posting list, protected by a Mutex so concurrent `add`
    /// calls from rayon tasks can append without a merge phase. Posting
    /// lists are kept sorted: `add` only ever appends doc IDs greater than
    /// any previously-stored ID, but parallel writers within a single
    /// `add` may interleave, so we sort the new tail at end of `add`.
    posting: Vec<Mutex<Vec<u32>>>,
    /// Slot → lex-smallest n-gram string. Read-only borrow lets parallel
    /// `add` skip String allocs for already-known slots; writes happen at
    /// end of `add` under exclusive `&mut self` access.
    vocab_repr: FxHashMap<u32, String>,
}

impl NGramIndex {
    pub fn new(
        max_n: usize,
        n_features: usize,
        tokenizer_mode: TokenizerMode,
        use_stopwords: bool,
    ) -> Self {
        assert!(max_n >= 1, "max_n must be >= 1");
        assert!(
            n_features.is_power_of_two(),
            "n_features must be a power of 2 (got {n_features})"
        );
        assert!(
            n_features <= u32::MAX as usize + 1,
            "n_features must fit in u32"
        );
        let posting: Vec<Mutex<Vec<u32>>> =
            (0..n_features).map(|_| Mutex::new(Vec::new())).collect();
        Self {
            max_n,
            n_features,
            mask: (n_features - 1) as u32,
            tokenizer_mode,
            use_stopwords,
            num_docs: 0,
            posting,
            vocab_repr: FxHashMap::default(),
        }
    }

    /// Hash a pre-stemmed n-gram string (space-separated tokens) to its slot.
    #[inline]
    fn hash_ngram_str(&self, ngram: &str) -> u32 {
        slot_for_ngram_str(ngram, self.mask)
    }

    /// Index a batch of documents. Doc IDs are assigned sequentially starting
    /// from the current `num_docs`.
    ///
    /// Strategy: each rayon task tokenizes its docs, hashes every unique
    /// n-gram per doc, and pushes the doc ID into the slot's `Mutex<Vec<u32>>`.
    /// Repr strings are accumulated in a per-thread `FxHashMap` and merged
    /// into `self.vocab_repr` at the end.
    ///
    /// After all parallel pushes, the new tail of each touched slot is
    /// sorted in parallel. Old entries are guaranteed `< start_id` (they
    /// were appended by a previous `add`) and new entries are all
    /// `>= start_id`, so sorting just the new tail keeps the whole posting
    /// list sorted.
    pub fn add(&mut self, docs: &[String]) {
        if docs.is_empty() {
            return;
        }
        let start_id = self.num_docs;
        let mode = self.tokenizer_mode;
        let use_sw = self.use_stopwords;
        let max_n = self.max_n;
        let mask = self.mask;
        let tok = Tokenizer::with_mode(mode, use_sw);
        let known_reprs: &FxHashMap<u32, String> = &self.vocab_repr;
        let posting: &Vec<Mutex<Vec<u32>>> = &self.posting;

        // Per-thread accumulator: only the lex-smallest repr string we
        // built for slots not already in `known_reprs`. We deliberately
        // do NOT track which slots were touched — instead, after the
        // parallel push, we sweep every slot's posting list in parallel
        // and sort just the new tail (empty tails are no-ops). This
        // avoids ~N-pushes worth of per-thread HashSet inserts.
        let partials: Vec<FxHashMap<u32, String>> = docs
            .par_iter()
            .enumerate()
            .fold(
                FxHashMap::<u32, String>::default,
                |mut reprs, (i, text)| {
                    let doc_id = start_id + i as u32;
                    let tokens = tok.tokenize_owned(text);
                    let mut doc_seen: FxHashSet<u32> = FxHashSet::default();
                    // Fused (slot, joined_string, _start) walk — single window
                    // pass, no lockstep zip of two separate iterators.
                    for (h, ngram, _) in
                        iter_ngram_slots_with_str(&tokens, 1, max_n, mask)
                    {
                        if !doc_seen.insert(h) {
                            continue;
                        }
                        // Push into the global slot — no merge phase,
                        // no per-thread posting maps. Mutex lock is
                        // ~30-50 ns under low collision rates.
                        posting[h as usize].lock().unwrap().push(doc_id);
                        // String alloc only for slots that no thread
                        // (here or globally) has built a repr for yet.
                        if !known_reprs.contains_key(&h)
                            && !reprs.contains_key(&h)
                        {
                            reprs.insert(h, ngram);
                        }
                    }
                    reprs
                },
            )
            .collect();

        // Sort the new tail of every slot in parallel. Pre-existing entries
        // are guaranteed < start_id (they were appended before this call);
        // new entries are all >= start_id. partition_point finds the
        // boundary in O(log n), and sort_unstable is a no-op on empty
        // slices, so unmodified slots cost only one lock + one binary search.
        self.posting.par_iter().for_each(|m| {
            let mut v = m.lock().unwrap();
            let new_start = v.partition_point(|&d| d < start_id);
            if new_start < v.len() {
                v[new_start..].sort_unstable();
            }
        });

        // Merge per-thread reprs (lex-smallest wins).
        for reprs in partials {
            for (slot, ng) in reprs {
                self.vocab_repr
                    .entry(slot)
                    .and_modify(|cur| {
                        if ng < *cur {
                            *cur = ng.clone();
                        }
                    })
                    .or_insert(ng);
            }
        }

        self.num_docs = start_id + docs.len() as u32;
    }

    pub fn df(&self, ngram: &str) -> u32 {
        let h = self.hash_ngram_str(ngram);
        // lock() is needed because pushers also lock; but `df` is a quick
        // read of the Vec's len, so contention is tiny.
        self.posting[h as usize].lock().unwrap().len() as u32
    }

    pub fn contains(&self, ngram: &str) -> bool {
        self.df(ngram) > 0
    }

    /// Tokenize `text` with the same pipeline used at build time, generate
    /// every contiguous n-gram with `n` in `[n_min, n_max]`, and return the
    /// rarest (lowest-DF, DF > 0) one as `(df, ngram_string, n)`. `None`
    /// when the text is too short for any `n_min`-gram, or every generated
    /// n-gram has DF=0 (slot empty — i.e. truly unseen at build time).
    ///
    /// The dropped DF=0 case is the right call for the post-hoc filter use
    /// case: a hashed-feature index reports DF=0 only for unseen ngrams,
    /// and we cannot distinguish "never seen" from "would have hashed to
    /// an empty slot but is genuinely rare" — so we treat them as
    /// uninformative and fall back to the next-rarest seen n-gram.
    fn min_df_ngram_with(
        &self,
        text: &str,
        tok: &Tokenizer,
        n_min: usize,
        n_max: usize,
    ) -> Option<(u32, String, usize)> {
        let tokens = tok.tokenize_owned(text);
        let upper = n_max.min(tokens.len());
        let mut best: Option<(u32, String, usize)> = None;
        // Iterate n outer / window inner so we know `n` per yielded window
        // (the flat fused iterator drops it). Per-n calls walk the same
        // `tokens.windows(n)` the original loop did.
        for n in n_min..=upper {
            for (h, ngram, _) in iter_ngram_slots_with_str(&tokens, n, n, self.mask) {
                let df = self.posting[h as usize].lock().unwrap().len() as u32;
                if df == 0 {
                    continue;
                }
                let beat = match &best {
                    None => true,
                    Some((b, _, _)) => df < *b,
                };
                if beat {
                    best = Some((df, ngram, n));
                }
            }
        }
        best
    }

    pub fn min_df_ngram(
        &self,
        text: &str,
        n_min: usize,
        n_max: usize,
    ) -> Option<(u32, String, usize)> {
        let tok = Tokenizer::with_mode(self.tokenizer_mode, self.use_stopwords);
        self.min_df_ngram_with(text, &tok, n_min, n_max)
    }

    /// Batched min-DF n-gram lookup. Each input text is tokenized,
    /// n-grams are generated in `[n_min, n_max]`, and the rarest seen
    /// n-gram is returned per text. Parallelized across texts via rayon;
    /// per-thread tokenizer is reused to amortize allocation.
    ///
    /// Designed for the per-phrase enrichment filter: one Python ↔ Rust
    /// call replaces millions of `df()` round-trips.
    pub fn min_df_ngram_batch(
        &self,
        texts: &[String],
        n_min: usize,
        n_max: usize,
    ) -> Vec<Option<(u32, String, usize)>> {
        let mode = self.tokenizer_mode;
        let use_sw = self.use_stopwords;
        texts
            .par_iter()
            .map_init(
                || Tokenizer::with_mode(mode, use_sw),
                |tok, text| self.min_df_ngram_with(text, tok, n_min, n_max),
            )
            .collect()
    }

    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    /// Number of *occupied* hash slots. With high collision rates this is a
    /// lower bound on the true distinct-n-gram count.
    pub fn vocab_size(&self) -> usize {
        // Walk the dense posting array and count non-empty slots in parallel.
        self.posting
            .par_iter()
            .filter(|m| !m.lock().unwrap().is_empty())
            .count()
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Slot a given n-gram string would hash to. Exposed for diagnostics
    /// and Python-side correctness oracles.
    pub fn slot_of(&self, ngram: &str) -> u32 {
        self.hash_ngram_str(ngram)
    }

    /// Atomic bincode write: temp file + rename.
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let on_disk = NGramIndexOnDisk {
            max_n: self.max_n,
            n_features: self.n_features,
            tokenizer_mode: self.tokenizer_mode,
            use_stopwords: self.use_stopwords,
            num_docs: self.num_docs,
            posting: self
                .posting
                .iter()
                .map(|m| m.lock().unwrap().clone())
                .collect(),
            vocab_repr: self.vocab_repr.clone(),
        };
        let bytes = bincode::serialize(&on_disk)?;
        let tmp = format!("{}.tmp.{}", path, std::process::id());
        std::fs::write(&tmp, bytes).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            e
        })?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let on_disk: NGramIndexOnDisk = bincode::deserialize(&bytes)?;
        Ok(Self {
            max_n: on_disk.max_n,
            n_features: on_disk.n_features,
            mask: (on_disk.n_features - 1) as u32,
            tokenizer_mode: on_disk.tokenizer_mode,
            use_stopwords: on_disk.use_stopwords,
            num_docs: on_disk.num_docs,
            posting: on_disk.posting.into_iter().map(Mutex::new).collect(),
            vocab_repr: on_disk.vocab_repr,
        })
    }

    /// One-shot enrichment-candidate filter for LLM-proposed phrases.
    ///
    /// Designed so ``py.allow_threads`` releases the GIL for the entire
    /// call, allowing true parallelism across async workers.
    ///
    /// Multi-word phrases are accepted: ``BM25::enrich()`` tokenizes
    /// the kept phrase and adds all sliding-window sub-n-grams.
    ///
    /// ## Filter pipeline (per candidate, first match wins)
    ///
    /// 1. ``empty``          — trim is empty.
    /// 2. ``no_stems``       — tokenize+stem produces nothing (all stopwords).
    /// 3. ``not_in_vocab``   — ``require_in_vocab`` is true and every
    ///                         sub-n-gram has DF=0 (unseen in corpus).
    /// 4. ``too_common``     — **all** sub-n-grams (1..=max_n sliding window)
    ///                         have DF > ``max_df``. A phrase is kept as long
    ///                         as at least one sub-n-gram is rare enough.
    /// 5. ``kept``           — passes all filters.
    pub fn filter_candidates(
        &self,
        candidates: &[String],
        _doc_text: &str,
        _prior_enrichments: &[String],
        max_df: u32,
        max_n: usize,
        require_in_vocab: bool,
        collect_verdicts: bool,
    ) -> (
        Vec<String>,
        Vec<(&'static str, u32)>,
        Vec<(String, String, Option<u32>, bool, &'static str)>,
    ) {
        let tok = Tokenizer::with_mode(self.tokenizer_mode, self.use_stopwords);

        let mut kept: Vec<String> = Vec::with_capacity(candidates.len());
        let mut counts: [u32; 5] = [0; 5];
        const REASONS: [&str; 5] = [
            "kept",
            "not_in_vocab",
            "too_common",
            "no_stems",
            "empty",
        ];
        let mut verdicts: Vec<(String, String, Option<u32>, bool, &'static str)> =
            if collect_verdicts {
                Vec::with_capacity(candidates.len())
            } else {
                Vec::new()
            };

        for raw in candidates {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                counts[4] += 1;
                if collect_verdicts {
                    verdicts.push((raw.clone(), String::new(), None, false, REASONS[4]));
                }
                continue;
            }
            let lowered = trimmed.to_lowercase();
            let stems = tok.tokenize_owned(&lowered);
            if stems.is_empty() {
                counts[3] += 1;
                if collect_verdicts {
                    verdicts.push((raw.clone(), String::new(), None, false, REASONS[3]));
                }
                continue;
            }
            let key = stems.join(" ");
            let rarest = self.min_df_ngram_with(&lowered, &tok, 1, max_n);
            match rarest {
                None if require_in_vocab => {
                    counts[1] += 1;
                    if collect_verdicts {
                        verdicts.push((raw.clone(), key, None, false, REASONS[1]));
                    }
                    continue;
                }
                Some((min_df, _, _)) if min_df > max_df => {
                    counts[2] += 1;
                    if collect_verdicts {
                        verdicts.push((raw.clone(), key, Some(min_df), false, REASONS[2]));
                    }
                    continue;
                }
                _ => {}
            }
            counts[0] += 1;
            if collect_verdicts {
                verdicts.push((raw.clone(), key.clone(), rarest.map(|(d, _, _)| d), true, REASONS[0]));
            }
            kept.push(raw.clone());
        }

        let stats: Vec<(&'static str, u32)> = REASONS
            .iter()
            .zip(counts.iter())
            .filter(|(_, &c)| c > 0)
            .map(|(&r, &c)| (r, c))
            .collect();
        (kept, stats, verdicts)
    }

    /// One-shot per-doc context prep. Replaces
    /// ``sira.optimize.enrichment.prepare_doc_context``: extracts the
    /// first 30 unique 1..=max_n grams, runs ``cooccur`` on them for
    /// ``top_k=20`` hints under ``df_max=max_df``, and builds the
    /// surface-form ban list. All in one GIL-released Rust call —
    /// previously the ban-list loop and the dedup-set construction were
    /// pure Python, holding the GIL on the executor thread.
    ///
    /// The 30 / 20 / max_df constants are inlined to match the Python
    /// caller exactly (``prepare_doc_context`` slices ``[:30]`` then
    /// passes ``top_k=20, df_max=max_df`` to cooccur). Hard-coding here
    /// keeps the public Rust signature small; if these become tunable
    /// at the Python layer in future, lift them to params.
    pub fn prepare_doc(
        &self,
        doc_text: &str,
        max_n: usize,
        max_df: u32,
    ) -> (Vec<String>, Vec<(String, u32)>, Vec<String>) {
        let tok = Tokenizer::with_mode(self.tokenizer_mode, self.use_stopwords);

        // Step 1: doc_ngrams (first 30 unique, 1..=max_n).
        let stems = tok.tokenize_owned(doc_text);
        let mut doc_ngrams: Vec<String> = Vec::with_capacity(30);
        let mut seen: FxHashSet<String> = FxHashSet::default();
        'outer: for n in 1..=max_n {
            if n > stems.len() {
                break;
            }
            for i in 0..=(stems.len() - n) {
                let key = stems[i..i + n].join(" ");
                if seen.insert(key.clone()) {
                    doc_ngrams.push(key);
                    if doc_ngrams.len() >= 30 {
                        break 'outer;
                    }
                }
            }
        }

        // Step 2: cooccur hints (skip if no ngrams — cooccur returns
        // empty anyway, but the early exit avoids the par_iter sweep).
        let cooccur_hints = if doc_ngrams.is_empty() {
            Vec::new()
        } else {
            self.cooccur(&doc_ngrams, 20, max_df)
        };

        // Step 3: surface-form ban list (original-case, dedup case-insensitive).
        // Mirrors Python ``w.strip(".,;:!?\"'()[]{}")`` then lowercase dedup.
        const PUNCT: &[char] = &['.', ',', ';', ':', '!', '?', '"', '\'', '(', ')', '[', ']', '{', '}'];
        let mut ban_tokens: Vec<String> = Vec::new();
        let mut ban_seen: FxHashSet<String> = FxHashSet::default();
        for word in doc_text.split_whitespace() {
            let trimmed = word.trim_matches(PUNCT);
            if trimmed.is_empty() {
                continue;
            }
            // ``to_lowercase`` allocates; do it once per word and reuse for
            // dedup membership. ASCII-only docs would benefit from
            // ``make_ascii_lowercase`` on a stack buffer, but FiQA has
            // enough Unicode (e.g. €, ñ) that the safe path wins.
            let key = trimmed.to_lowercase();
            if ban_seen.insert(key) {
                ban_tokens.push(trimmed.to_string());
            }
        }

        (doc_ngrams, cooccur_hints, ban_tokens)
    }

    /// Top-k slots whose posting lists best intersect with the
    /// intersection of the query n-grams' posting lists. Excludes the
    /// query slots themselves and slots with `df > df_max`.
    pub fn cooccur(
        &self,
        query_ngrams: &[String],
        top_k: usize,
        df_max: u32,
    ) -> Vec<(String, u32)> {
        if query_ngrams.is_empty() || top_k == 0 {
            return Vec::new();
        }

        // Step 1: hash queries → slot IDs → posting list snapshots.
        let query_slots: Vec<u32> =
            query_ngrams.iter().map(|q| self.hash_ngram_str(q)).collect();
        let lists: Vec<Vec<u32>> = query_slots
            .iter()
            .filter_map(|s| {
                let v = self.posting[*s as usize].lock().unwrap();
                if v.is_empty() {
                    None
                } else {
                    Some(v.clone())
                }
            })
            .collect();
        if lists.is_empty() {
            return Vec::new();
        }

        // Step 2: intersect, smallest list first.
        let mut order: Vec<usize> = (0..lists.len()).collect();
        order.sort_by_key(|&i| lists[i].len());
        let mut docs: Vec<u32> = lists[order[0]].clone();
        for &i in &order[1..] {
            docs = sorted_intersect(&docs, &lists[i]);
            if docs.is_empty() {
                return Vec::new();
            }
        }
        let exclude: FxHashSet<u32> = query_slots.into_iter().collect();

        // Step 3: parallel score every candidate slot. Iterate the dense
        // Vec<Mutex<Vec<u32>>> directly — no FxHashMap iteration overhead.
        let mut scores: Vec<(u32, u32)> = self
            .posting
            .par_iter()
            .enumerate()
            .filter_map(|(slot_usize, m)| {
                let slot = slot_usize as u32;
                if exclude.contains(&slot) {
                    return None;
                }
                let plist = m.lock().unwrap();
                if plist.is_empty() || plist.len() as u32 > df_max {
                    return None;
                }
                let count = sorted_intersect_count(&plist, &docs);
                if count == 0 {
                    None
                } else {
                    Some((slot, count))
                }
            })
            .collect();

        // Step 4: top-k by count desc, ties by representative string asc.
        scores.sort_by(|a, b| {
            b.1.cmp(&a.1).then_with(|| {
                let ra = self.vocab_repr.get(&a.0);
                let rb = self.vocab_repr.get(&b.0);
                ra.cmp(&rb)
            })
        });
        scores.truncate(top_k);
        scores
            .into_iter()
            .map(|(slot, c)| {
                let s = self
                    .vocab_repr
                    .get(&slot)
                    .cloned()
                    .unwrap_or_else(|| format!("<slot:{slot}>"));
                (s, c)
            })
            .collect()
    }
}

/// Intersect two sorted Vec<u32>s into a new sorted Vec<u32>.
fn sorted_intersect(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len().min(b.len()));
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    out
}

/// Count the intersection size of two sorted slices without allocating.
fn sorted_intersect_count(a: &[u32], b: &[u32]) -> u32 {
    let mut count = 0u32;
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer_mode() -> TokenizerMode {
        TokenizerMode::UnicodeStem
    }

    /// Use a generous n_features for unit tests so collisions don't perturb
    /// exact-equality assertions on small vocabularies.
    const NF: usize = 1 << 20; // 1M slots ≈ 48 MB upfront — still fine for tests.

    fn new_idx(max_n: usize) -> NGramIndex {
        NGramIndex::new(max_n, NF, tokenizer_mode(), true)
    }

    #[test]
    fn test_build_and_df_unigram() {
        let mut idx = new_idx(4);
        idx.add(&[
            "the cat sat on the mat".to_string(),
            "the dog sat on the floor".to_string(),
            "the bird flew".to_string(),
        ]);
        assert_eq!(idx.df("sat"), 2);
        assert_eq!(idx.df("bird"), 1);
        assert_eq!(idx.df("rocket"), 0);
        assert_eq!(idx.num_docs(), 3);
    }

    #[test]
    fn test_df_bigram_trigram() {
        let mut idx = new_idx(4);
        idx.add(&[
            "machine learning is fun".to_string(),
            "deep learning is hard".to_string(),
            "machine learning models".to_string(),
        ]);
        assert_eq!(idx.df("machin learn"), 2);
        assert_eq!(idx.df("machin learn model"), 1);
        assert_eq!(idx.df("hello world"), 0);
        assert!(idx.vocab_size() > 0);
    }

    #[test]
    fn test_cooccur_basic() {
        let mut idx = new_idx(4);
        idx.add(&[
            "machine learning models are powerful".to_string(),
            "machine learning is a subfield of ai".to_string(),
            "deep learning models perform well".to_string(),
            "ai is changing the world".to_string(),
        ]);
        let result = idx.cooccur(&["machin learn".to_string()], 200, 10);
        let names: FxHashSet<_> = result.iter().map(|(s, _)| s.clone()).collect();
        assert!(names.contains("model"), "expected 'model' in {:?}", names);
        assert!(names.contains("ai"), "expected 'ai' in {:?}", names);
        assert!(!names.contains("machin learn"));
    }

    #[test]
    fn test_cooccur_unknown_query() {
        let mut idx = new_idx(4);
        idx.add(&["hello world".to_string()]);
        let r = idx.cooccur(&["nonexistent".to_string()], 10, 100);
        assert!(r.is_empty());
    }

    #[test]
    fn test_cooccur_df_max_filter() {
        let mut idx = new_idx(4);
        idx.add(&[
            "common term rare1".to_string(),
            "common term rare2".to_string(),
            "common term rare3".to_string(),
        ]);
        let r = idx.cooccur(&["term".to_string()], 10, 2);
        let names: FxHashSet<_> = r.iter().map(|(s, _)| s.clone()).collect();
        assert!(!names.contains("common"));
    }

    #[test]
    fn test_add_large_parallel() {
        let docs: Vec<String> = (0..1000)
            .map(|i| {
                format!(
                    "document {} contains some words like alpha beta gamma {}",
                    i,
                    i % 50
                )
            })
            .collect();
        let mut idx = new_idx(4);
        let t = std::time::Instant::now();
        idx.add(&docs);
        eprintln!("add 1000 docs: {:?}", t.elapsed());
        assert_eq!(idx.num_docs(), 1000);
        assert!(idx.vocab_size() > 100);
        assert_eq!(idx.df("alpha"), 1000);
        assert_eq!(idx.df("alpha beta"), 1000);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap();

        let mut idx = NGramIndex::new(3, NF, tokenizer_mode(), true);
        idx.add(&[
            "alpha beta gamma".to_string(),
            "beta gamma delta".to_string(),
        ]);
        idx.save(path).unwrap();

        let loaded = NGramIndex::load(path).unwrap();
        assert_eq!(loaded.num_docs(), 2);
        assert_eq!(loaded.df("beta"), 2);
        assert_eq!(loaded.df("alpha beta"), 1);
        assert_eq!(loaded.n_features(), NF);
        let r = loaded.cooccur(&["beta".to_string()], 10, 100);
        let names: FxHashSet<_> = r.iter().map(|(s, _)| s.clone()).collect();
        assert!(names.contains("gamma"));
    }

    #[test]
    fn test_add_empty_corpus() {
        let mut idx = new_idx(4);
        idx.add(&[]);
        assert_eq!(idx.num_docs(), 0);
        assert_eq!(idx.vocab_size(), 0);
        assert_eq!(idx.df("anything"), 0);
        assert!(!idx.contains(""));
    }

    #[test]
    fn test_add_empty_and_blank_strings() {
        let mut idx = new_idx(4);
        idx.add(&["".to_string(), "   ".to_string(), "real content".to_string()]);
        assert_eq!(idx.num_docs(), 3);
        assert_eq!(idx.df("real"), 1);
        assert_eq!(idx.df("content"), 1);
        assert_eq!(idx.df(""), 0);
    }

    #[test]
    fn test_incremental_add_continues_doc_ids() {
        let mut idx = NGramIndex::new(2, NF, tokenizer_mode(), true);
        idx.add(&["alpha".to_string(), "beta".to_string()]);
        assert_eq!(idx.num_docs(), 2);
        idx.add(&["alpha".to_string(), "gamma".to_string()]);
        assert_eq!(idx.num_docs(), 4);

        assert_eq!(idx.df("alpha"), 2);
        assert_eq!(idx.df("beta"), 1);
        assert_eq!(idx.df("gamma"), 1);

        // Posting list for "alpha" must remain sorted across the boundary
        // (doc 0 from first add, doc 2 from second add).
        let slot = idx.hash_ngram_str("alpha") as usize;
        let plist = idx.posting[slot].lock().unwrap();
        assert_eq!(*plist, vec![0u32, 2]);
    }

    #[test]
    fn test_max_n_one_only_unigrams() {
        let mut idx = NGramIndex::new(1, NF, tokenizer_mode(), true);
        idx.add(&["alpha beta gamma".to_string()]);
        assert_eq!(idx.df("alpha"), 1);
        assert_eq!(idx.df("beta"), 1);
        assert_eq!(idx.df("alpha beta"), 0);
        assert_eq!(idx.df("alpha beta gamma"), 0);
    }

    #[test]
    fn test_use_stopwords_false_keeps_them() {
        let mut idx = NGramIndex::new(2, NF, tokenizer_mode(), false);
        idx.add(&["the cat sat".to_string()]);
        assert_eq!(idx.df("the"), 1);
        assert_eq!(idx.df("the cat"), 1);
    }

    #[test]
    fn test_cooccur_empty_query() {
        let mut idx = new_idx(2);
        idx.add(&["alpha beta".to_string()]);
        assert!(idx.cooccur(&[], 10, 100).is_empty());
    }

    #[test]
    fn test_cooccur_top_k_zero() {
        let mut idx = new_idx(2);
        idx.add(&["alpha beta gamma".to_string()]);
        assert!(idx.cooccur(&["alpha".to_string()], 0, 100).is_empty());
    }

    #[test]
    fn test_cooccur_df_max_zero_filters_everything() {
        let mut idx = new_idx(2);
        idx.add(&["alpha beta".to_string()]);
        assert!(idx.cooccur(&["alpha".to_string()], 10, 0).is_empty());
    }

    #[test]
    fn test_cooccur_disjoint_query_intersection_empty() {
        let mut idx = new_idx(2);
        idx.add(&["alpha".to_string(), "beta".to_string()]);
        let r = idx.cooccur(&["alpha".to_string(), "beta".to_string()], 10, 100);
        assert!(r.is_empty());
    }

    #[test]
    fn test_cooccur_is_deterministic() {
        let mut idx = new_idx(2);
        idx.add(&[
            "alpha beta gamma".to_string(),
            "alpha gamma delta".to_string(),
            "alpha beta delta".to_string(),
        ]);
        let r1 = idx.cooccur(&["alpha".to_string()], 50, 100);
        let r2 = idx.cooccur(&["alpha".to_string()], 50, 100);
        assert_eq!(r1, r2);
        let counts: Vec<u32> = r1.iter().map(|(_, c)| *c).collect();
        let mut sorted = counts.clone();
        sorted.sort_by(|a, b| b.cmp(a));
        assert_eq!(counts, sorted);
    }

    #[test]
    fn test_unicode_tokens_indexed() {
        let mut idx = new_idx(2);
        idx.add(&["café résumé".to_string()]);
        assert_eq!(idx.df("cafe"), 1);
        assert_eq!(idx.df("resum"), 1);
    }

    #[test]
    fn test_save_load_empty_index() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap();
        let idx = new_idx(4);
        idx.save(path).unwrap();
        let loaded = NGramIndex::load(path).unwrap();
        assert_eq!(loaded.num_docs(), 0);
        assert_eq!(loaded.vocab_size(), 0);
        assert!(loaded.cooccur(&["foo".to_string()], 10, 100).is_empty());
    }

    #[test]
    fn test_load_nonexistent_path_errors() {
        assert!(NGramIndex::load("/tmp/this_does_not_exist_98237492.idx").is_err());
    }

    #[test]
    fn test_save_atomicity_no_temp_left_on_failure() {
        let bad = "/proc/this_is_read_only_path_does_not_exist/idx";
        let idx = new_idx(2);
        assert!(idx.save(bad).is_err());
    }

    #[test]
    fn test_save_overwrite_existing_file() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap();

        let mut a = new_idx(2);
        a.add(&["alpha".to_string()]);
        a.save(path).unwrap();

        let mut b = new_idx(2);
        b.add(&["beta".to_string(), "gamma".to_string()]);
        b.save(path).unwrap();

        let loaded = NGramIndex::load(path).unwrap();
        assert_eq!(loaded.num_docs(), 2);
        assert_eq!(loaded.df("beta"), 1);
        assert_eq!(loaded.df("alpha"), 0);
    }

    #[test]
    fn test_cooccur_intersection_of_query_ngrams() {
        let mut idx = new_idx(4);
        idx.add(&[
            "alpha beta gamma".to_string(),
            "alpha delta epsilon".to_string(),
            "beta zeta eta".to_string(),
        ]);
        let r = idx.cooccur(&["alpha".to_string(), "beta".to_string()], 10, 100);
        let names: FxHashSet<_> = r.iter().map(|(s, _)| s.clone()).collect();
        assert!(names.contains("gamma"));
        assert!(!names.contains("delta"));
        assert!(!names.contains("zeta"));
    }

    #[test]
    fn test_n_features_must_be_power_of_two() {
        let result = std::panic::catch_unwind(|| {
            NGramIndex::new(2, 1000, tokenizer_mode(), true)
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_min_df_ngram_picks_rarest_seen() {
        let mut idx = new_idx(4);
        idx.add(&[
            "401k retirement plan match employer".to_string(),
            "401k contribution limits".to_string(),
            "401k match".to_string(),
            "match the score".to_string(),
            "match the score now".to_string(),
            "match the goal".to_string(),
        ]);
        // unigram DFs: 401k=3, match=4 (much more common). bigram
        // "401k match" = 1 (much rarer than either unigram). The rarest
        // n-gram in the phrase should be the bigram, not "401k".
        let r = idx.min_df_ngram("401k match", 1, 4);
        let (df, gram, n) = r.expect("phrase has tokens");
        assert_eq!(n, 2, "expected the bigram to be the rarest, got n={n}");
        assert_eq!(df, 1);
        assert_eq!(gram, "401k match");
    }

    #[test]
    fn test_min_df_ngram_batch_matches_serial() {
        let mut idx = new_idx(4);
        idx.add(&[
            "alpha beta gamma".to_string(),
            "alpha gamma".to_string(),
            "beta gamma delta".to_string(),
        ]);
        let phrases = vec![
            "alpha beta".to_string(),
            "gamma delta".to_string(),
            "alpha beta gamma".to_string(),
            "lonely".to_string(),  // not in the index
        ];
        let serial: Vec<_> = phrases
            .iter()
            .map(|p| idx.min_df_ngram(p, 1, 4))
            .collect();
        let batched = idx.min_df_ngram_batch(&phrases, 1, 4);
        assert_eq!(serial, batched);
    }

    #[test]
    fn test_min_df_ngram_short_phrase_below_n_min() {
        let mut idx = new_idx(4);
        idx.add(&["alpha beta".to_string()]);
        // single-token phrase; n_min=2 → no eligible n-grams → None.
        assert!(idx.min_df_ngram("alpha", 2, 4).is_none());
    }

    #[test]
    fn test_filter_candidates_long_phrase_not_rejected() {
        // too_long filter was removed — phrases longer than max_n are kept
        // and indexed via their sub-n-grams (up to max_n).
        let mut idx = new_idx(4);
        idx.add(&["seed corpus".to_string()]);
        let candidates = vec![
            "this phrase has five words".to_string(),  // 5 raw words, still kept
            "short bigram".to_string(),                // 2 raw words
        ];
        let (kept, _stats, _) = idx.filter_candidates(
            &candidates, "doc text",
            &[],     // no priors
            10_000,  // max_df permissive
            4,       // max_n=4
            false,   // require_in_vocab=false (doc-side)
            true,    // collect_verdicts
        );
        assert_eq!(kept.len(), 2, "both phrases should be kept");
        assert!(kept.contains(&"this phrase has five words".to_string()));
        assert!(kept.contains(&"short bigram".to_string()));
    }

    #[test]
    fn test_collisions_at_small_n_features_overestimate_only() {
        let mut idx = NGramIndex::new(1, 16, tokenizer_mode(), true);
        idx.add(&[
            "alpha".to_string(),
            "beta".to_string(),
            "gamma".to_string(),
            "delta".to_string(),
        ]);
        for ng in ["alpha", "beta", "gamma", "delta"] {
            let df = idx.df(ng);
            assert!(df >= 1, "{ng} df={df} < truth (1)");
        }
    }
}
