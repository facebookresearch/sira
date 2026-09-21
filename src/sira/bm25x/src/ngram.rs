// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Shared n-gram generation and hashing.
//!
//! Single source of truth used by:
//! - `crate::index::BM25` (hashed n-gram tier for n >= 2)
//! - `crate::tfidf::Tfidf` (n-gram TF/DF in vocab and hashing modes)
//! - `crate::ngram_index::NGramIndex` (full hashed n-gram side index)
//!
//! N-grams are space-joined (`"machin learn"`) so vocab strings, debugging
//! output, and Python-side tooling stay consistent across the three indices.
//!
//! ## Hashing
//!
//! [`hash_ngram_window`] mirrors sklearn's `HashingVectorizer`:
//! `abs(murmurhash3_x86_32(...) as i32) & mask`. Required for slot-space
//! parity with `Tfidf` / `NGramIndex` (which Python tests already assume).

use murmurhash3::murmurhash3_x86_32;
use std::collections::HashMap;

/// Iterate every n-gram with `n_min <= n <= n_max`, yielding `(joined, start_pos)`.
/// Requires `n_min >= 1`.
pub(crate) fn iter_ngrams<'a>(
    tokens: &'a [String],
    n_min: usize,
    n_max: usize,
) -> impl Iterator<Item = (String, u32)> + 'a {
    debug_assert!(n_min >= 1, "n_min must be >= 1 (windows(0) panics)");
    let n_max = n_max.min(tokens.len());
    (n_min..=n_max).flat_map(move |n| {
        tokens.windows(n).enumerate()
            .map(move |(start, w)| (join_ngram(w), start as u32))
    })
}

/// Iterate n-grams whose order `n` is in `ns`, yielding `(slot_hash, start_pos)`.
/// `ns` must contain only values >= 1; ordering of `ns` controls emission order
/// (callers pass sorted lists so emission stays deterministic across runs).
///
/// Use this when you need a non-contiguous subset (e.g. just bigrams + 4-grams,
/// skipping trigrams). For contiguous ranges, prefer `iter_ngram_slots` — both
/// are equivalent when `ns == (n_min..=n_max).collect()`.
pub(crate) fn iter_ngram_slots_in<'a>(
    tokens: &'a [String],
    ns: &'a [u8],
    mask: u32,
) -> impl Iterator<Item = (u32, u32)> + 'a {
    let token_len = tokens.len();
    ns.iter().filter_map(move |&n| {
        debug_assert!(n >= 1, "n must be >= 1 (windows(0) panics)");
        let n_usize = n as usize;
        if n_usize == 0 || n_usize > token_len {
            return None;
        }
        Some((n_usize, tokens.windows(n_usize).enumerate()))
    }).flat_map(move |(_n, iter)| {
        iter.map(move |(start, w)| (hash_ngram_window(w, mask), start as u32))
    })
}

/// Iterate every n-gram in `[n_min, n_max]`, yielding `(slot_hash, start_pos)`.
/// Requires `n_min >= 1`.
pub(crate) fn iter_ngram_slots<'a>(
    tokens: &'a [String],
    n_min: usize,
    n_max: usize,
    mask: u32,
) -> impl Iterator<Item = (u32, u32)> + 'a {
    debug_assert!(n_min >= 1, "n_min must be >= 1 (windows(0) panics)");
    let n_max = n_max.min(tokens.len());
    (n_min..=n_max).flat_map(move |n| {
        tokens.windows(n).enumerate()
            .map(move |(start, w)| (hash_ngram_window(w, mask), start as u32))
    })
}

/// Iterate every n-gram in `[n_min, n_max]`, yielding `(slot_hash, joined_string, start_pos)`
/// in the same order as `iter_ngrams` and `iter_ngram_slots`. Single window pass — the
/// fused yield avoids the lockstep-zip fragility of using the two iterators separately.
///
/// Requires `n_min >= 1`.
pub(crate) fn iter_ngram_slots_with_str<'a>(
    tokens: &'a [String],
    n_min: usize,
    n_max: usize,
    mask: u32,
) -> impl Iterator<Item = (u32, String, u32)> + 'a {
    debug_assert!(n_min >= 1, "n_min must be >= 1 (windows(0) panics)");
    let n_max = n_max.min(tokens.len());
    (n_min..=n_max).flat_map(move |n| {
        tokens.windows(n).enumerate().map(move |(start, w)| {
            let s = join_ngram(w);
            // Hash the joined string directly — `slot_for_ngram_str` agrees with
            // `hash_ngram_window`'s composed hash byte-for-byte (both feed the
            // same sklearn-parity formula).
            let slot = slot_for_ngram_str(&s, mask);
            (slot, s, start as u32)
        })
    })
}

/// Build TF + start-position map (used by unigram tier of `BM25` and by mutation paths).
/// Requires `n_min >= 1`.
#[allow(dead_code)] // Consumed in Task 2 (BM25 hashed-ngram tier).
pub(crate) fn ngram_tf_pos_map(
    tokens: &[String],
    n_min: usize,
    n_max: usize,
) -> HashMap<String, (u32, Vec<u32>)> {
    debug_assert!(n_min >= 1, "n_min must be >= 1 (windows(0) panics)");
    // upper-bound; collisions across n-orders make actual count smaller
    let mut out: HashMap<String, (u32, Vec<u32>)> =
        HashMap::with_capacity(tokens.len() * (n_max.saturating_sub(n_min) + 1).max(1));
    for (ng, pos) in iter_ngrams(tokens, n_min, n_max) {
        let entry = out.entry(ng).or_insert((0, Vec::new()));
        entry.0 += 1;
        entry.1.push(pos);
    }
    out
}

/// Build a slot-keyed TF + position map for an arbitrary subset of n values.
/// See `iter_ngram_slots_in` for the contract on `ns`.
pub(crate) fn slot_tf_pos_map_in(
    tokens: &[String],
    ns: &[u8],
    mask: u32,
) -> HashMap<u32, (u32, Vec<u32>)> {
    let mut out: HashMap<u32, (u32, Vec<u32>)> =
        HashMap::with_capacity(tokens.len() * ns.len().max(1));
    for (slot, pos) in iter_ngram_slots_in(tokens, ns, mask) {
        let entry = out.entry(slot).or_insert((0, Vec::new()));
        entry.0 += 1;
        entry.1.push(pos);
    }
    out
}

/// Build TF-only map. Used by `Tfidf` (no positions needed).
/// Requires `n_min >= 1`.
pub(crate) fn ngram_tf_map(tokens: &[String], n_min: usize, n_max: usize) -> HashMap<String, u32> {
    debug_assert!(n_min >= 1, "n_min must be >= 1 (windows(0) panics)");
    // upper-bound; collisions across n-orders make actual count smaller
    let mut out: HashMap<String, u32> =
        HashMap::with_capacity(tokens.len() * (n_max.saturating_sub(n_min) + 1).max(1));
    for (ng, _) in iter_ngrams(tokens, n_min, n_max) {
        *out.entry(ng).or_insert(0) += 1;
    }
    out
}

/// Hash an n-gram window to its slot. Sklearn-compatible (signed → abs → mask).
#[inline]
pub(crate) fn hash_ngram_window(window: &[String], mask: u32) -> u32 {
    let raw = if window.len() == 1 {
        murmurhash3_x86_32(window[0].as_bytes(), 0)
    } else {
        let cap: usize = window.iter().map(|w| w.len()).sum::<usize>() + window.len() - 1;
        let mut buf = String::with_capacity(cap);
        for (i, t) in window.iter().enumerate() {
            if i > 0 { buf.push(' '); }
            buf.push_str(t);
        }
        murmurhash3_x86_32(buf.as_bytes(), 0)
    };
    (raw as i32).unsigned_abs() & mask
}

/// Hash a pre-joined n-gram string. Used at query time when we already have
/// the joined form (e.g. from user query analysis).
#[inline]
pub(crate) fn slot_for_ngram_str(ngram: &str, mask: u32) -> u32 {
    let raw = murmurhash3_x86_32(ngram.as_bytes(), 0);
    (raw as i32).unsigned_abs() & mask
}

/// Single-allocation `window.join(" ")`.
#[inline]
pub(crate) fn join_ngram(window: &[String]) -> String {
    if window.len() == 1 {
        return window[0].clone();
    }
    let cap: usize = window.iter().map(|w| w.len()).sum::<usize>() + window.len() - 1;
    let mut buf = String::with_capacity(cap);
    for (i, t) in window.iter().enumerate() {
        if i > 0 { buf.push(' '); }
        buf.push_str(t);
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iter_ngrams_emits_uni_bi_tri_with_start_positions() {
        let toks = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let got: Vec<(String, u32)> = iter_ngrams(&toks, 1, 3).collect();
        assert_eq!(
            got,
            vec![
                ("a".into(), 0), ("b".into(), 1), ("c".into(), 2),
                ("a b".into(), 0), ("b c".into(), 1),
                ("a b c".into(), 0),
            ]
        );
    }

    #[test]
    fn iter_ngram_slots_yields_slot_and_start() {
        let toks = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mask = (1u32 << 16) - 1;
        let got: Vec<(u32, u32)> = iter_ngram_slots(&toks, 2, 3, mask).collect();
        assert_eq!(got.len(), 3);
        for (slot, _) in &got {
            assert!(*slot < (mask + 1));
        }
    }

    #[test]
    fn slot_for_ngram_str_matches_iter_ngram_slots_for_window() {
        let mask = (1u32 << 16) - 1;
        let toks = vec!["machin".to_string(), "learn".to_string()];
        let from_iter: Vec<u32> = iter_ngram_slots(&toks, 2, 2, mask)
            .map(|(s, _)| s).collect();
        assert_eq!(from_iter[0], slot_for_ngram_str("machin learn", mask));
    }

    #[test]
    fn empty_or_short_input_safe() {
        assert_eq!(iter_ngrams(&[], 1, 4).count(), 0);
        let toks = vec!["a".to_string()];
        assert_eq!(iter_ngrams(&toks, 2, 4).count(), 0);
        assert_eq!(iter_ngram_slots(&toks, 2, 4, 0xFFFF).count(), 0);
    }

    #[test]
    fn iter_ngram_slots_in_emits_only_listed_ns() {
        let toks: Vec<String> = "alpha beta gamma delta".split(' ').map(String::from).collect();
        let mask = (1u32 << 16) - 1;
        // Subset {2, 4} skips trigrams.
        let got: Vec<(u32, u32)> = iter_ngram_slots_in(&toks, &[2u8, 4u8], mask).collect();
        // 4 tokens → 3 bigrams + 1 four-gram = 4 emissions.
        assert_eq!(got.len(), 4);
        // Compare to the contiguous {2, 3, 4} version: it would emit 3 + 2 + 1 = 6.
        let full: Vec<(u32, u32)> = iter_ngram_slots(&toks, 2, 4, mask).collect();
        assert_eq!(full.len(), 6);
    }

    #[test]
    fn iter_ngram_slots_in_with_n_above_token_count_skips_silently() {
        let toks: Vec<String> = "a b".split(' ').map(String::from).collect();
        let mask = (1u32 << 16) - 1;
        let got: Vec<(u32, u32)> = iter_ngram_slots_in(&toks, &[2u8, 3u8, 4u8], mask).collect();
        // Only the bigram fits — trigram and 4-gram are skipped.
        assert_eq!(got.len(), 1);
    }

    #[test]
    fn iter_ngram_slots_with_str_matches_separate_iterators() {
        let toks: Vec<String> = "alpha beta gamma delta".split(' ').map(String::from).collect();
        let mask = (1u32 << 16) - 1;
        let fused: Vec<(u32, String, u32)> =
            iter_ngram_slots_with_str(&toks, 1, 3, mask).collect();
        let separate: Vec<(u32, String, u32)> =
            iter_ngram_slots(&toks, 1, 3, mask)
                .zip(iter_ngrams(&toks, 1, 3))
                .map(|((slot, _), (s, pos))| (slot, s, pos))
                .collect();
        assert_eq!(fused, separate);
    }
}
