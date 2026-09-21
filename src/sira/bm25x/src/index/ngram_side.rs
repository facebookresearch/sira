// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Hashed n-gram tier for [`crate::index::BM25`].
//!
//! Mirrors the unigram tier's flat `Vec<Vec<...>>` shape, but indexed by
//! slot ∈ `[0, n_features)` instead of vocab term_id. Only allocated when
//! the parent index has `max_n >= 2`.
//!
//! Hash collisions cause `df()` to be an over-estimate of the true n-gram
//! DF (never under). IDF computed from `df()` is correspondingly under-
//! estimated. Document this limitation; raise `n_features` to mitigate.

use crate::ngram::{hash_ngram_window, slot_for_ngram_str, slot_tf_pos_map_in};

/// Dense per-slot inverted index for n-grams.
///
/// Tracks an explicit set of allowed `n` values (`ns`, all >= 2), so callers
/// can build a non-contiguous tier — e.g. just bigrams + 4-grams, skipping
/// trigrams. `max_n` is derived (`*ns.last().unwrap()`) and exposed only for
/// back-compat of older call sites and storage v2.
pub(crate) struct NgramSide {
    /// Sorted, deduped n values (each >= 2). E.g. `[2, 4]` means only bigrams
    /// and 4-grams populate the side; trigrams are never emitted or scored.
    pub(crate) ns: Vec<u8>,
    pub(crate) n_features: u32,
    pub(crate) mask: u32,

    /// Per-slot posting list: sorted by doc_id.
    /// `postings[slot][i] = (doc_id, tf)`.
    pub(crate) postings: Vec<Vec<(u32, u32)>>,

    /// Per-slot positions, parallel to postings.
    /// `positions[slot][i]` = sorted unigram start positions for postings[slot][i].
    pub(crate) positions: Vec<Vec<Vec<u32>>>,

    /// Per-slot DF (cached `postings[slot].len()`).
    pub(crate) doc_freqs: Vec<u32>,
}

impl NgramSide {
    /// Legacy constructor: contiguous `2..=max_n` n-gram tier.
    #[allow(dead_code)] // kept for tests + back-compat; new code uses `with_ns`
    pub(crate) fn new(max_n: u8, n_features: u32) -> Self {
        assert!(max_n >= 2, "NgramSide requires max_n >= 2 (n=1 lives in unigram tier)");
        Self::with_ns((2..=max_n).collect(), n_features)
    }

    /// Build a side that emits only the n values in `ns` (sorted ascending,
    /// each >= 2). Empty `ns` is rejected — callers should set
    /// `BM25::ngram_side = None` instead.
    pub(crate) fn with_ns(mut ns: Vec<u8>, n_features: u32) -> Self {
        ns.sort_unstable();
        ns.dedup();
        assert!(!ns.is_empty(), "NgramSide requires at least one n value");
        assert!(ns.iter().all(|&n| n >= 2), "NgramSide n values must be >= 2");
        assert!(n_features.is_power_of_two(), "n_features must be a power of 2");
        let nf = n_features as usize;
        Self {
            ns,
            n_features,
            mask: n_features - 1,
            postings: vec![Vec::new(); nf],
            positions: vec![Vec::new(); nf],
            doc_freqs: vec![0u32; nf],
        }
    }

    #[inline] pub(crate) fn max_n(&self) -> u8 { *self.ns.last().expect("ns non-empty by construction") }
    #[inline] pub(crate) fn ns(&self) -> &[u8] { &self.ns }
    #[allow(dead_code)] // Parity accessor for the unigram tier; no caller yet.
    #[inline] pub(crate) fn n_features(&self) -> u32 { self.n_features }
    #[inline] pub(crate) fn mask(&self) -> u32 { self.mask }

    /// DF for a slot.
    #[inline]
    pub(crate) fn df(&self, slot: u32) -> u32 {
        self.doc_freqs[slot as usize]
    }

    /// Number of postings (== DF, exposed for parity with unigram tier).
    #[allow(dead_code)] // Parity accessor for the unigram tier; no caller yet.
    #[inline]
    pub(crate) fn posting_count(&self, slot: u32) -> u32 {
        self.postings[slot as usize].len() as u32
    }

    /// Look up TF for a (slot, doc_id) pair via binary search.
    pub(crate) fn posting_tf(&self, slot: u32, doc_id: u32) -> Option<u32> {
        let plist = &self.postings[slot as usize];
        plist.binary_search_by_key(&doc_id, |&(d, _)| d).ok().map(|i| plist[i].1)
    }

    /// Positions for a single posting.
    #[allow(dead_code)] // Parity accessor for the unigram tier; no caller yet.
    pub(crate) fn posting_positions(&self, slot: u32, posting_idx: usize) -> &[u32] {
        &self.positions[slot as usize][posting_idx]
    }

    /// Resolve a query n-gram string to its slot. The slot may be empty
    /// (caller checks `df()`).
    #[inline]
    pub(crate) fn slot_for(&self, ngram: &str) -> u32 {
        slot_for_ngram_str(ngram, self.mask)
    }

    /// Iterate `(doc_id, tf)` for a slot's postings. Mirrors `BM25::for_each_posting`.
    pub(crate) fn for_each_posting<F: FnMut(u32, u32)>(&self, slot: u32, mut f: F) {
        for &(d, tf) in &self.postings[slot as usize] {
            f(d, tf);
        }
    }

    /// Index a single document's n-grams (called sequentially during merge).
    /// `tokens` are the unigram tokens already produced by the tokenizer.
    ///
    /// Caller contract (debug-checked invariant): `doc_id` MUST be strictly
    /// greater than every prior `doc_id` passed to `add_doc`. This is what
    /// keeps each slot's posting list sorted by simple append. Violations
    /// silently corrupt postings in release builds and will fail subsequent
    /// `binary_search_by_key` lookups; the `debug_assert!` below pins the
    /// contract for tests.
    pub(crate) fn add_doc(&mut self, doc_id: u32, tokens: &[String]) {
        let map = slot_tf_pos_map_in(tokens, &self.ns, self.mask);
        for (slot, (tf, pos)) in map {
            // doc_id is monotonically increasing across calls — append (sorted).
            debug_assert!(
                self.postings[slot as usize]
                    .last()
                    .map_or(true, |&(prev, _)| prev < doc_id),
                "add_doc requires monotonically increasing doc_id (caller contract)",
            );
            self.postings[slot as usize].push((doc_id, tf));
            self.positions[slot as usize].push(pos);
            self.doc_freqs[slot as usize] += 1;
        }
    }

    /// Re-index a document's n-grams (for `BM25::update`).
    /// First drops the old entries via `remove_doc`, then inserts sorted.
    pub(crate) fn replace_doc(&mut self, doc_id: u32, tokens: &[String]) {
        self.remove_doc(doc_id);
        let map = slot_tf_pos_map_in(tokens, &self.ns, self.mask);
        for (slot, (tf, pos)) in map {
            let plist = &mut self.postings[slot as usize];
            let i = plist.partition_point(|&(d, _)| d < doc_id);
            plist.insert(i, (doc_id, tf));
            self.positions[slot as usize].insert(i, pos);
            self.doc_freqs[slot as usize] += 1;
        }
    }

    /// Add n-gram contributions from extra tokens at offset `pos_offset`,
    /// merging with any existing posting for `doc_id` (for `BM25::enrich`).
    pub(crate) fn enrich_doc(&mut self, doc_id: u32, extra_tokens: &[String], pos_offset: u32) {
        let map = slot_tf_pos_map_in(extra_tokens, &self.ns, self.mask);
        for (slot, (extra_tf, mut extra_pos)) in map {
            for p in extra_pos.iter_mut() { *p += pos_offset; }
            let plist = &mut self.postings[slot as usize];
            match plist.binary_search_by_key(&doc_id, |&(d, _)| d) {
                Ok(i) => {
                    plist[i].1 += extra_tf;
                    self.positions[slot as usize][i].extend(extra_pos);
                }
                Err(i) => {
                    plist.insert(i, (doc_id, extra_tf));
                    self.positions[slot as usize].insert(i, extra_pos);
                    self.doc_freqs[slot as usize] += 1;
                }
            }
        }
    }

    /// Subtract n-gram contributions, removing the posting if TF reaches 0
    /// (for `BM25::unenrich`). Positions are NOT mutated — same caveat as
    /// the unigram tier (`BM25::unenrich` doc-comment).
    pub(crate) fn unenrich_doc(&mut self, doc_id: u32, extra_tokens: &[String]) {
        // TF-only map (positions discarded for symmetry with the unigram path).
        let mut tf_map: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for (slot, _pos) in crate::ngram::iter_ngram_slots_in(
            extra_tokens, &self.ns, self.mask,
        ) {
            *tf_map.entry(slot).or_insert(0) += 1;
        }
        for (slot, sub_tf) in tf_map {
            let plist = &mut self.postings[slot as usize];
            if let Ok(i) = plist.binary_search_by_key(&doc_id, |&(d, _)| d) {
                if plist[i].1 <= sub_tf {
                    plist.remove(i);
                    self.positions[slot as usize].remove(i);
                    self.doc_freqs[slot as usize] = self.doc_freqs[slot as usize].saturating_sub(1);
                } else {
                    plist[i].1 -= sub_tf;
                }
            }
        }
    }

    /// Add the WHOLE token sequence as a single n-gram (no sliding sub-windows).
    /// Mirrors [`Self::enrich_doc`] but writes exactly ONE slot — the hash of
    /// the full `extra_tokens` joined as a single n-gram.
    ///
    /// Used by `BM25::enrich_exact` for LLM-proposed enrichment phrases where
    /// the caller wants the LLM's full phrase to be the only signal in the
    /// n-gram tier (no sub-window pollution of common bigrams' df). See the
    /// "How `add()` propagates" section in
    /// `sira.search.bm25.EnrichmentAdapter` for the rationale.
    ///
    /// No-op when `extra_tokens.len() < 2` (n-gram tier holds n>=2 only) or
    /// when `extra_tokens.len()` is not in `self.ns` (the whole phrase's
    /// length must be one of the configured n values; otherwise the slot
    /// would never be queried at retrieve time).
    pub(crate) fn enrich_doc_exact(&mut self, doc_id: u32, extra_tokens: &[String], pos_offset: u32) {
        let n = extra_tokens.len();
        if n < 2 || !self.ns.iter().any(|&allowed| allowed as usize == n) {
            return;
        }
        let slot = hash_ngram_window(extra_tokens, self.mask);
        let plist = &mut self.postings[slot as usize];
        match plist.binary_search_by_key(&doc_id, |&(d, _)| d) {
            Ok(i) => {
                plist[i].1 += 1;
                self.positions[slot as usize][i].push(pos_offset);
            }
            Err(i) => {
                plist.insert(i, (doc_id, 1));
                self.positions[slot as usize].insert(i, vec![pos_offset]);
                self.doc_freqs[slot as usize] += 1;
            }
        }
    }

    /// Reverse [`Self::enrich_doc_exact`]. Decrements TF on the single slot
    /// hashed from the full token sequence; removes the posting if TF reaches
    /// 0 (and decrements `doc_freqs`). Positions are NOT mutated — same
    /// caveat as [`Self::unenrich_doc`].
    pub(crate) fn unenrich_doc_exact(&mut self, doc_id: u32, extra_tokens: &[String]) {
        let n = extra_tokens.len();
        if n < 2 || !self.ns.iter().any(|&allowed| allowed as usize == n) {
            return;
        }
        let slot = hash_ngram_window(extra_tokens, self.mask);
        let plist = &mut self.postings[slot as usize];
        if let Ok(i) = plist.binary_search_by_key(&doc_id, |&(d, _)| d) {
            if plist[i].1 <= 1 {
                plist.remove(i);
                self.positions[slot as usize].remove(i);
                self.doc_freqs[slot as usize] = self.doc_freqs[slot as usize].saturating_sub(1);
            } else {
                plist[i].1 -= 1;
            }
        }
    }

    /// Drop every posting for `doc_id` across all slots (for `BM25::update`'s
    /// re-index path).
    ///
    /// Cost: O(n_features × log(avg_posting_len)) — at default n_features=8M and
    /// average posting length ~2, this is ~50ms. Acceptable for low-frequency
    /// `update` calls. If hot, switch to a doc→slot map.
    ///
    /// Reachable transitively via [`Self::replace_doc`]; not called directly
    /// by BM25 because the unigram path inlines its own removal.
    pub(crate) fn remove_doc(&mut self, doc_id: u32) {
        for (slot, plist) in self.postings.iter_mut().enumerate() {
            if let Ok(i) = plist.binary_search_by_key(&doc_id, |&(d, _)| d) {
                plist.remove(i);
                self.positions[slot].remove(i);
                self.doc_freqs[slot] = self.doc_freqs[slot].saturating_sub(1);
            }
        }
    }

    /// Compact every slot's postings against an `id_map` (for `BM25::delete`):
    /// `id_map[old_id] == u32::MAX` means deleted; else the new compacted ID.
    pub(crate) fn compact(&mut self, id_map: &[u32]) {
        for (slot, plist) in self.postings.iter_mut().enumerate() {
            let pos_list = &mut self.positions[slot];
            let mut new_p: Vec<(u32, u32)> = Vec::with_capacity(plist.len());
            let mut new_pos: Vec<Vec<u32>> = Vec::with_capacity(pos_list.len());
            for (i, &(old_id, tf)) in plist.iter().enumerate() {
                let new_id = id_map[old_id as usize];
                if new_id != u32::MAX {
                    new_p.push((new_id, tf));
                    new_pos.push(std::mem::take(&mut pos_list[i]));
                }
            }
            self.doc_freqs[slot] = new_p.len() as u32;
            *plist = new_p;
            *pos_list = new_pos;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_side() -> NgramSide {
        NgramSide::new(/* max_n */ 4, /* n_features */ 1 << 16)
    }

    #[test]
    fn new_allocates_dense_slots() {
        let s = small_side();
        assert_eq!(s.n_features(), 1 << 16);
        assert_eq!(s.max_n(), 4);
        assert_eq!(s.posting_count(0), 0);
        assert_eq!(s.df(0), 0);
        assert_eq!(s.mask(), (1u32 << 16) - 1);
    }

    #[test]
    fn add_doc_appends_postings_and_positions() {
        let mut s = small_side();
        let toks = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        s.add_doc(0, &toks);  // bigrams "a b", "b c"; trigram "a b c"; 4-gram absent (only 3 tokens).

        let slot_ab = s.slot_for("a b");
        assert_eq!(s.df(slot_ab), 1);
        assert_eq!(s.posting_tf(slot_ab, 0), Some(1));
        assert_eq!(s.posting_positions(slot_ab, 0), &[0u32][..]);

        let slot_abc = s.slot_for("a b c");
        assert_eq!(s.df(slot_abc), 1);
        assert_eq!(s.posting_positions(slot_abc, 0), &[0u32][..]);
    }

    #[test]
    fn add_doc_aggregates_repeated_ngram_tf() {
        let mut s = small_side();
        s.add_doc(0, &["a".into(), "b".into(), "a".into(), "b".into()]);
        let slot = s.slot_for("a b");
        assert_eq!(s.df(slot), 1);
        assert_eq!(s.posting_tf(slot, 0), Some(2));
        assert_eq!(s.posting_positions(slot, 0), &[0u32, 2][..]);
    }

    #[test]
    fn add_doc_two_docs_keeps_postings_sorted() {
        let mut s = small_side();
        s.add_doc(0, &["a".into(), "b".into()]);
        s.add_doc(1, &["a".into(), "b".into()]);
        let slot = s.slot_for("a b");
        assert_eq!(s.df(slot), 2);
        assert_eq!(s.posting_tf(slot, 0), Some(1));
        assert_eq!(s.posting_tf(slot, 1), Some(1));
    }

    #[test]
    fn replace_doc_removes_old_and_inserts_new_sorted() {
        let mut s = small_side();
        s.add_doc(0, &["a".into(), "b".into()]);
        s.add_doc(1, &["c".into(), "d".into()]);
        s.add_doc(2, &["e".into(), "f".into()]);

        // Replace doc 1 with new text whose bigram doesn't overlap any existing slot.
        s.replace_doc(1, &["x".into(), "y".into()]);

        let slot_cd = s.slot_for("c d");
        assert_eq!(s.df(slot_cd), 0, "old bigram for doc 1 must be gone");
        let slot_xy = s.slot_for("x y");
        assert_eq!(s.df(slot_xy), 1);
        assert_eq!(s.posting_tf(slot_xy, 1), Some(1));

        // Other docs untouched.
        assert_eq!(s.df(s.slot_for("a b")), 1);
        assert_eq!(s.df(s.slot_for("e f")), 1);
    }

    #[test]
    fn enrich_doc_merges_into_existing_posting() {
        let mut s = small_side();
        s.add_doc(0, &["a".into(), "b".into()]);
        // Enrich doc 0 with extra tokens whose own bigram is "c d";
        // pos_offset = original doc length (2), so positions land at 2.
        s.enrich_doc(0, &["c".into(), "d".into()], 2);

        let slot_cd = s.slot_for("c d");
        assert_eq!(s.df(slot_cd), 1);
        assert_eq!(s.posting_tf(slot_cd, 0), Some(1));
        assert_eq!(s.posting_positions(slot_cd, 0), &[2u32][..]);

        // Original bigram untouched.
        assert_eq!(s.df(s.slot_for("a b")), 1);
        assert_eq!(s.posting_positions(s.slot_for("a b"), 0), &[0u32][..]);
    }

    #[test]
    fn enrich_doc_accumulates_tf_for_existing_slot() {
        let mut s = small_side();
        s.add_doc(0, &["a".into(), "b".into()]);
        // Re-enrich with tokens whose bigram is the SAME slot: "a b".
        s.enrich_doc(0, &["a".into(), "b".into()], 2);

        let slot_ab = s.slot_for("a b");
        assert_eq!(s.df(slot_ab), 1, "doc count unchanged");
        assert_eq!(s.posting_tf(slot_ab, 0), Some(2), "TF accumulated");
        assert_eq!(s.posting_positions(slot_ab, 0), &[0u32, 2][..]);
    }

    #[test]
    fn unenrich_doc_subtracts_tf_and_drops_zero_posting() {
        let mut s = small_side();
        s.add_doc(0, &["a".into(), "b".into(), "a".into(), "b".into()]);
        let slot = s.slot_for("a b");
        assert_eq!(s.posting_tf(slot, 0), Some(2));

        s.unenrich_doc(0, &["a".into(), "b".into()]);
        assert_eq!(s.posting_tf(slot, 0), Some(1), "TF decremented");
        assert_eq!(s.df(slot), 1, "DF unchanged while TF > 0");

        // Subtract again — TF goes to 0, posting must drop.
        s.unenrich_doc(0, &["a".into(), "b".into()]);
        assert_eq!(s.posting_tf(slot, 0), None);
        assert_eq!(s.df(slot), 0);
    }

    #[test]
    fn enrich_doc_exact_writes_only_one_slot_for_full_phrase() {
        // Full 4-gram phrase enriched into doc 0 must hit ONE slot
        // (the 4-gram slot) — no sub-window slots for the contained
        // bigrams / trigrams.
        let mut s = small_side();
        let toks: Vec<String> = vec!["w1".into(), "w2".into(), "w3".into(), "w4".into()];
        s.enrich_doc_exact(0, &toks, 50);

        // The full 4-gram slot got the write.
        let slot_4g = s.slot_for("w1 w2 w3 w4");
        assert_eq!(s.df(slot_4g), 1);
        assert_eq!(s.posting_tf(slot_4g, 0), Some(1));
        assert_eq!(s.posting_positions(slot_4g, 0), &[50u32][..]);

        // Sub-window slots must NOT have been touched. Use distinct
        // string slots so we know we're not just observing a hash
        // collision with the 4-gram slot above.
        for sub in ["w1 w2", "w2 w3", "w3 w4", "w1 w2 w3", "w2 w3 w4"] {
            let slot = s.slot_for(sub);
            if slot == slot_4g {
                continue; // accidental hash collision — skip; would need a
                          // different test phrase to verify
            }
            assert_eq!(s.df(slot), 0, "sub-window {:?} must not be written", sub);
        }
    }

    #[test]
    fn enrich_doc_exact_noop_for_unigram() {
        // Single-token phrase has no n-gram-tier representation
        // (n=1 lives in the unigram tier of BM25, not in NgramSide).
        let mut s = small_side();
        s.enrich_doc_exact(0, &["lonely".into()], 0);
        // Walk every slot; total df must remain 0.
        let total_df: u64 = (0..s.n_features()).map(|i| s.df(i) as u64).sum();
        assert_eq!(total_df, 0);
    }

    #[test]
    fn enrich_doc_exact_noop_when_phrase_length_exceeds_max_n() {
        // 5-token phrase with default max_n=4 → no slot should fire.
        let mut s = small_side();
        let toks: Vec<String> = (0..5).map(|i| format!("t{i}")).collect();
        s.enrich_doc_exact(0, &toks, 0);
        let total_df: u64 = (0..s.n_features()).map(|i| s.df(i) as u64).sum();
        assert_eq!(total_df, 0, "over-long phrase must be a defence-in-depth no-op");
    }

    #[test]
    fn unenrich_doc_exact_reverses_enrich_doc_exact() {
        let mut s = small_side();
        let toks: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        s.enrich_doc_exact(0, &toks, 0);
        let slot = s.slot_for("a b c");
        assert_eq!(s.df(slot), 1);

        s.unenrich_doc_exact(0, &toks);
        assert_eq!(s.df(slot), 0);
        assert_eq!(s.posting_tf(slot, 0), None);
    }

    #[test]
    fn enrich_doc_exact_accumulates_tf_on_repeat() {
        // Repeating the same exact phrase keeps DF=1 but doubles TF.
        let mut s = small_side();
        let toks: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        s.enrich_doc_exact(0, &toks, 0);
        s.enrich_doc_exact(0, &toks, 3);  // pos_offset advances on the doc

        let slot = s.slot_for("a b c");
        assert_eq!(s.df(slot), 1);
        assert_eq!(s.posting_tf(slot, 0), Some(2));
        assert_eq!(s.posting_positions(slot, 0), &[0u32, 3][..]);
    }

    #[test]
    fn remove_doc_drops_every_posting_for_id() {
        let mut s = small_side();
        s.add_doc(0, &["a".into(), "b".into(), "c".into()]);
        s.add_doc(1, &["d".into(), "e".into()]);

        s.remove_doc(0);
        assert_eq!(s.df(s.slot_for("a b")), 0);
        assert_eq!(s.df(s.slot_for("b c")), 0);
        assert_eq!(s.df(s.slot_for("a b c")), 0);
        // Doc 1 untouched.
        assert_eq!(s.df(s.slot_for("d e")), 1);
    }

    #[test]
    fn compact_remaps_doc_ids_per_id_map() {
        let mut s = small_side();
        s.add_doc(0, &["a".into(), "b".into()]);
        s.add_doc(1, &["c".into(), "d".into()]);
        s.add_doc(2, &["e".into(), "f".into()]);

        // Simulate `delete(&[1])`: doc 1 is removed, doc 2 shifts down to ID 1.
        let id_map: Vec<u32> = vec![0, u32::MAX, 1];
        s.compact(&id_map);

        let slot_cd = s.slot_for("c d");
        assert_eq!(s.df(slot_cd), 0);
        let slot_ef = s.slot_for("e f");
        assert_eq!(s.df(slot_ef), 1);
        assert_eq!(s.posting_tf(slot_ef, 1), Some(1));  // remapped ID
    }

    #[test]
    fn new_panics_on_max_n_below_2() {
        let r = std::panic::catch_unwind(|| NgramSide::new(1, 1 << 16));
        assert!(r.is_err());
    }

    #[test]
    fn new_panics_on_non_power_of_two_n_features() {
        let r = std::panic::catch_unwind(|| NgramSide::new(4, 12345));
        assert!(r.is_err());
    }

    #[test]
    fn with_ns_subset_skips_unlisted_orders() {
        // Only bigrams and 4-grams. Trigrams must NOT enter any slot, even when
        // present in the token stream.
        let mut s = NgramSide::with_ns(vec![2, 4], 1 << 16);
        s.add_doc(0, &["a".into(), "b".into(), "c".into(), "d".into()]);
        // Bigram "a b" present.
        assert_eq!(s.df(s.slot_for("a b")), 1);
        // 4-gram "a b c d" present.
        assert_eq!(s.df(s.slot_for("a b c d")), 1);
        // Trigram "a b c" must be absent.
        assert_eq!(s.df(s.slot_for("a b c")), 0);
        // max_n() reflects the largest n in the set.
        assert_eq!(s.max_n(), 4);
        assert_eq!(s.ns(), &[2u8, 4u8]);
    }

    #[test]
    fn with_ns_panics_on_n_below_2() {
        let r = std::panic::catch_unwind(|| NgramSide::with_ns(vec![1, 2], 1 << 16));
        assert!(r.is_err(), "n=1 lives in unigram tier; rejected here");
    }

    #[test]
    fn with_ns_panics_on_empty() {
        let r = std::panic::catch_unwind(|| NgramSide::with_ns(vec![], 1 << 16));
        assert!(r.is_err());
    }

    #[test]
    fn replace_doc_preserves_sort_order_in_mid_slot() {
        // 3 docs sharing the same bigram "a b". Replace doc 1 with new tokens
        // that ALSO contain "a b" — exercises partition_point's mid-slot insert.
        let mut s = small_side();
        s.add_doc(0, &["a".into(), "b".into()]);
        s.add_doc(1, &["a".into(), "b".into()]);
        s.add_doc(2, &["a".into(), "b".into()]);

        let slot = s.slot_for("a b");
        assert_eq!(s.df(slot), 3);

        // Replace doc 1's tokens; new TF for "a b" is 2 (bigram appears twice).
        s.replace_doc(1, &["a".into(), "b".into(), "a".into(), "b".into()]);

        assert_eq!(s.df(slot), 3, "DF unchanged — same doc replaced");
        let plist = &s.postings[slot as usize];
        assert_eq!(plist.len(), 3);
        assert_eq!(plist[0].0, 0, "doc 0 first");
        assert_eq!(plist[1].0, 1, "doc 1 stays at sorted position 1");
        assert_eq!(plist[2].0, 2, "doc 2 last");
        assert_eq!(plist[1].1, 2, "new TF for doc 1 is 2");
        // Positions for doc 1 reflect the new tokens.
        assert_eq!(s.posting_positions(slot, 1), &[0u32, 2][..]);
    }

    #[test]
    fn compact_preserves_survivors_within_same_slot() {
        // 3 docs all sharing bigram "a b". Delete doc 1 — surviving docs 0 and
        // 2 must remap to compacted IDs 0 and 1, in order, with positions intact.
        let mut s = small_side();
        s.add_doc(0, &["a".into(), "b".into()]);
        s.add_doc(1, &["a".into(), "b".into(), "a".into(), "b".into()]);
        s.add_doc(2, &["a".into(), "b".into()]);

        let slot = s.slot_for("a b");
        assert_eq!(s.df(slot), 3);
        // Pre-compact: doc 1 has TF=2 with positions [0, 2].
        assert_eq!(s.posting_tf(slot, 1), Some(2));
        let pre_doc2_pos = s.posting_positions(slot, 2).to_vec();

        // Simulate `delete(&[1])`: doc 1 removed; doc 2 shifts to ID 1.
        let id_map: Vec<u32> = vec![0, u32::MAX, 1];
        s.compact(&id_map);

        assert_eq!(s.df(slot), 2);
        let plist = &s.postings[slot as usize];
        assert_eq!(plist.len(), 2);
        assert_eq!(plist[0].0, 0, "doc 0 stays at compacted ID 0");
        assert_eq!(plist[1].0, 1, "old doc 2 shifts to compacted ID 1");
        assert_eq!(plist[1].1, 1, "TF for the surviving posting unchanged");
        // Positions for surviving doc 2 (now ID 1) are intact.
        assert_eq!(s.posting_positions(slot, 1), pre_doc2_pos.as_slice());
    }
}
