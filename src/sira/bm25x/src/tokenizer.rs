// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use std::collections::HashMap;

use rust_stemmers::{Algorithm, Stemmer};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

/// Tokenizer mode controls text preprocessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenizerMode {
    /// Lowercase + split on non-alphanumeric. Fast, no stemming.
    Plain,
    /// Plain + Unicode NFKD normalization (diacritics folded to ASCII).
    Unicode,
    /// Plain + Snowball stemming (English).
    Stem,
    /// Unicode normalization + Snowball stemming. Best accuracy.
    UnicodeStem,
    /// Character n-gram tokenizer: emits every overlapping char window of
    /// size `min_n..=max_n`. No stemming; spaces are included as word-boundary
    /// markers. Pair with `ngrams=[1]` (word n-gram order 1 only).
    CharNgram { min_n: u8, max_n: u8 },
}

impl TokenizerMode {
    /// Stable on-disk encoding used by the storage v2 header.
    /// CHANGE THE NUMBERS, BREAK ALL SAVED INDICES — only ever append new variants.
    pub fn to_id(self) -> u8 {
        match self {
            TokenizerMode::Plain => 0,
            TokenizerMode::Unicode => 1,
            TokenizerMode::Stem => 2,
            TokenizerMode::UnicodeStem => 3,
            TokenizerMode::CharNgram { min_n: 4, max_n: 5 } => 4,
            TokenizerMode::CharNgram { min_n: 7, max_n: 8 } => 5,
            TokenizerMode::CharNgram { min_n, max_n } => {
                panic!("CharNgram({min_n},{max_n}) has no assigned storage ID")
            }
        }
    }

    /// Inverse of [`Self::to_id`]; returns `None` for unknown IDs (forward-compat
    /// guard: a future writer may emit a variant this reader doesn't understand).
    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(Self::Plain),
            1 => Some(Self::Unicode),
            2 => Some(Self::Stem),
            3 => Some(Self::UnicodeStem),
            4 => Some(Self::CharNgram { min_n: 4, max_n: 5 }),
            5 => Some(Self::CharNgram { min_n: 7, max_n: 8 }),
            _ => None,
        }
    }
}

/// A configurable tokenizer: lowercase, split, optional unicode folding,
/// optional stemming, optional stopword removal.
pub struct Tokenizer {
    stopwords: Option<std::collections::HashSet<String>>,
    mode: TokenizerMode,
    stemmer: Option<Stemmer>,
}

pub(crate) const ENGLISH_STOPWORDS: &[&str] = &[
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "its",
    "itself",
    "them",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "am",
    "been",
    "being",
    "do",
    "does",
    "did",
    "doing",
    "would",
    "should",
    "could",
    "ought",
    "might",
    "shall",
    "can",
    "need",
    "dare",
    "had",
    "has",
    "have",
    "having",
    "about",
    "above",
    "after",
    "again",
    "against",
    "below",
    "between",
    "during",
    "from",
    "further",
    "here",
    "once",
    "only",
    "out",
    "over",
    "same",
    "so",
    "than",
    "too",
    "under",
    "until",
    "up",
    "very",
    "own",
    "just",
    "don",
    "now",
    "d",
    "ll",
    "m",
    "o",
    "re",
    "s",
    "t",
    "ve",
];

impl Tokenizer {
    /// Create a tokenizer. For backward compatibility, `new(use_stopwords)` uses `Plain` mode.
    pub fn new(use_stopwords: bool) -> Self {
        Self::with_mode(TokenizerMode::Plain, use_stopwords)
    }

    /// Create a tokenizer with a specific mode.
    pub fn with_mode(mode: TokenizerMode, use_stopwords: bool) -> Self {
        let stopwords = if use_stopwords {
            Some(ENGLISH_STOPWORDS.iter().map(|s| s.to_string()).collect())
        } else {
            None
        };
        let stemmer = match mode {
            TokenizerMode::Stem | TokenizerMode::UnicodeStem => {
                Some(Stemmer::create(Algorithm::English))
            }
            _ => None,
        };
        Tokenizer {
            stopwords,
            mode,
            stemmer,
        }
    }

    /// The tokenizer mode (Plain / Unicode / Stem / UnicodeStem). Persisted by
    /// storage v2 so `BM25::load` can faithfully reconstruct the index instead
    /// of hardcoding `UnicodeStem` (the legacy v1 bug that silently broke
    /// queries on Plain-built indices).
    #[inline]
    pub fn mode(&self) -> TokenizerMode { self.mode }

    /// Whether English stopwords are filtered. Persisted by storage v2 alongside
    /// `mode()` for the same root-cause reason.
    #[inline]
    pub fn use_stopwords(&self) -> bool { self.stopwords.is_some() }

    /// Lowercase, normalize, and optionally stem a single token (no splitting).
    ///
    /// Used by the query rewrite layer to stem wildcard prefixes without
    /// splitting on non-alphanumeric boundaries or filtering stopwords.
    pub fn stem_single(&self, token: &str) -> String {
        let lowered = token.to_lowercase();
        // CharNgram has no stemming concept; return lowercased token as-is
        if matches!(self.mode, TokenizerMode::CharNgram { .. }) {
            return lowered;
        }
        let normalized = match self.mode {
            TokenizerMode::Unicode | TokenizerMode::UnicodeStem => fold_to_ascii(&lowered),
            _ => lowered,
        };
        if let Some(ref stemmer) = self.stemmer {
            stemmer.stem(&normalized).into_owned()
        } else {
            normalized
        }
    }

    /// Tokenize and return owned lowercase (+ optionally stemmed/normalized) tokens.
    pub fn tokenize_owned(&self, text: &str) -> Vec<String> {
        // Char n-gram modes bypass word-level tokenization entirely
        if let TokenizerMode::CharNgram { min_n, max_n } = self.mode {
            return char_ngrams_from_text(text, min_n as usize, max_n as usize);
        }

        // Step 1: Lowercase
        let lowered = text.to_lowercase();

        // Step 2: Optional Unicode NFKD normalization → strip diacritics
        let normalized: String = match self.mode {
            TokenizerMode::Unicode | TokenizerMode::UnicodeStem => fold_to_ascii(&lowered),
            _ => lowered,
        };

        // Step 3: Split on non-alphanumeric
        let raw_tokens = split_alphanumeric(&normalized);

        // Step 4: Stopword filter + optional stemming
        let mut tokens = Vec::with_capacity(raw_tokens.len());
        for token in raw_tokens {
            if !self.should_keep(&token) {
                continue;
            }
            let final_token = if let Some(ref stemmer) = self.stemmer {
                stemmer.stem(&token).into_owned()
            } else {
                token
            };
            if !final_token.is_empty() {
                tokens.push(final_token);
            }
        }
        tokens
    }

    /// Tokenize with a thread-local stem cache for amortized O(1) stemming.
    ///
    /// The cache maps pre-stem tokens → stemmed form. For large corpora this
    /// avoids redundant Snowball stemming of repeated words across documents.
    /// ~500K unique English tokens means ~500K stems cached, vs ~500M stem calls.
    /// Tokenize with std HashMap stem cache (for non-CUDA paths).
    pub fn tokenize_cached(
        &self,
        text: &str,
        stem_cache: &mut HashMap<String, String>,
    ) -> Vec<String> {
        self.tokenize_with_cache_impl(text, stem_cache)
    }

    /// Tokenize with FxHashMap stem cache (faster hashing, used by CUDA path).
    pub fn tokenize_cached_fx(
        &self,
        text: &str,
        stem_cache: &mut FxHashMap<String, String>,
    ) -> Vec<String> {
        self.tokenize_with_cache_impl(text, stem_cache)
    }

    fn tokenize_with_cache_impl<S: std::hash::BuildHasher>(
        &self,
        text: &str,
        stem_cache: &mut HashMap<String, String, S>,
    ) -> Vec<String> {
        // Char n-gram modes don't use stem cache
        if let TokenizerMode::CharNgram { min_n, max_n } = self.mode {
            return char_ngrams_from_text(text, min_n as usize, max_n as usize);
        }

        let lowered = text.to_lowercase();

        let normalized: String = match self.mode {
            TokenizerMode::Unicode | TokenizerMode::UnicodeStem => fold_to_ascii(&lowered),
            _ => lowered,
        };

        let raw_tokens = split_alphanumeric(&normalized);

        let mut tokens = Vec::with_capacity(raw_tokens.len());
        for token in raw_tokens {
            if !self.should_keep(&token) {
                continue;
            }
            let final_token = if let Some(ref stemmer) = self.stemmer {
                if let Some(cached) = stem_cache.get(&token) {
                    cached.clone()
                } else {
                    let stemmed = stemmer.stem(&token).into_owned();
                    stem_cache.insert(token, stemmed.clone());
                    stemmed
                }
            } else {
                token
            };
            if !final_token.is_empty() {
                tokens.push(final_token);
            }
        }
        tokens
    }

    /// Fused tokenize + TF count: single-pass ASCII fast path, zero intermediate allocations.
    ///
    /// For ASCII text (99%+ of English corpora), this:
    /// - Does in-place lowercase in a reusable buffer (no String alloc)
    /// - Skips fold_to_ascii entirely (text is already ASCII)
    /// - Splits + stopword-filters + stem-caches + TF-counts in ONE pass
    /// - Avoids allocating Vec<String> for intermediate tokens
    ///
    /// For non-ASCII text, falls back to the standard 3-pass pipeline.
    pub fn tokenize_and_count<S: std::hash::BuildHasher>(
        &self,
        text: &str,
        stem_cache: &mut HashMap<String, String, S>,
        tf_map: &mut HashMap<String, u32, S>,
        buf: &mut Vec<u8>,
    ) -> u32 {
        tf_map.clear();

        // Char n-gram modes bypass word tokenization + stemming entirely
        if let TokenizerMode::CharNgram { min_n, max_n } = self.mode {
            return count_char_ngrams(text, min_n as usize, max_n as usize, tf_map);
        }

        if text.is_ascii() {
            self.tokenize_count_ascii(text, stem_cache, tf_map, buf)
        } else {
            self.tokenize_count_unicode(text, stem_cache, tf_map)
        }
    }

    /// ASCII fast path: zero-heap-alloc single-pass tokenization.
    ///
    /// Lowercases on-the-fly into a stack-allocated 128-byte token buffer.
    /// Never touches the heap for text processing — only for stem cache
    /// misses and new TF map entries.
    fn tokenize_count_ascii<S: std::hash::BuildHasher>(
        &self,
        text: &str,
        stem_cache: &mut HashMap<String, String, S>,
        tf_map: &mut HashMap<String, u32, S>,
        _buf: &mut Vec<u8>,
    ) -> u32 {
        let bytes = text.as_bytes();
        let mut doc_len = 0u32;
        // Stack-allocated token buffer — no heap allocation per token.
        // 128 bytes covers all English words (longest common word ~30 chars).
        let mut token_buf: [u8; 128] = [0; 128];
        let mut token_len: usize = 0;

        for &b in bytes {
            if b.is_ascii_alphanumeric() {
                if token_len < 128 {
                    // Branchless ASCII lowercase: 'A'-'Z' → 'a'-'z', others unchanged.
                    // b | 0x20 works for letters but corrupts digits, so use conditional.
                    token_buf[token_len] = b.to_ascii_lowercase();
                    token_len += 1;
                }
                // Tokens > 128 chars: silently truncate (never happens in practice)
            } else if token_len > 0 {
                // Safety: buffer only contains ASCII lowercase bytes from to_ascii_lowercase.
                let token = std::str::from_utf8(&token_buf[..token_len]).unwrap();
                doc_len += self.count_token(token, stem_cache, tf_map);
                token_len = 0;
            }
        }
        if token_len > 0 {
            let token = std::str::from_utf8(&token_buf[..token_len]).unwrap();
            doc_len += self.count_token(token, stem_cache, tf_map);
        }
        doc_len
    }

    /// Unicode fallback: standard 3-pass pipeline fused with TF counting.
    fn tokenize_count_unicode<S: std::hash::BuildHasher>(
        &self,
        text: &str,
        stem_cache: &mut HashMap<String, String, S>,
        tf_map: &mut HashMap<String, u32, S>,
    ) -> u32 {
        let lowered = text.to_lowercase();
        let normalized: String = match self.mode {
            TokenizerMode::Unicode | TokenizerMode::UnicodeStem => fold_to_ascii(&lowered),
            _ => lowered,
        };
        // After fold_to_ascii, result is ASCII — use the fast split+count loop
        let bytes = normalized.as_bytes();
        let mut doc_len = 0u32;
        let mut start: Option<usize> = None;

        for i in 0..bytes.len() {
            if bytes[i].is_ascii_alphanumeric() {
                if start.is_none() {
                    start = Some(i);
                }
            } else if let Some(s) = start {
                doc_len += self.count_token(&normalized[s..i], stem_cache, tf_map);
                start = None;
            }
        }
        if let Some(s) = start {
            doc_len += self.count_token(&normalized[s..], stem_cache, tf_map);
        }
        doc_len
    }

    /// Process a single token: stopword check → stem cache → TF map increment.
    /// Returns 1 if token was counted, 0 if filtered.
    #[inline]
    fn count_token<S: std::hash::BuildHasher>(
        &self,
        token: &str,
        stem_cache: &mut HashMap<String, String, S>,
        tf_map: &mut HashMap<String, u32, S>,
    ) -> u32 {
        if !self.should_keep(token) {
            return 0;
        }

        if let Some(ref stemmer) = self.stemmer {
            if let Some(stem) = stem_cache.get(token) {
                // Hot path: stem cached. Try TF map lookup first to avoid clone.
                if let Some(count) = tf_map.get_mut(stem.as_str()) {
                    *count += 1;
                } else {
                    tf_map.insert(stem.clone(), 1);
                }
            } else {
                // Cold path: new token, compute stem
                let stemmed = stemmer.stem(token).into_owned();
                if !stemmed.is_empty() {
                    stem_cache.insert(token.to_string(), stemmed.clone());
                    *tf_map.entry(stemmed).or_insert(0) += 1;
                } else {
                    return 0;
                }
            }
        } else if let Some(count) = tf_map.get_mut(token) {
            *count += 1;
        } else {
            tf_map.insert(token.to_string(), 1);
        }
        1
    }

    #[inline]
    fn should_keep(&self, token: &str) -> bool {
        if token.is_empty() {
            return false;
        }
        if let Some(ref sw) = self.stopwords {
            !sw.contains(token)
        } else {
            true
        }
    }
}

/// Split a string on non-alphanumeric boundaries, returning owned tokens.
fn split_alphanumeric(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut start = None;

    for (i, c) in text.char_indices() {
        if c.is_alphanumeric() {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start {
            tokens.push(text[s..i].to_string());
            start = None;
        }
    }
    if let Some(s) = start {
        tokens.push(text[s..].to_string());
    }
    tokens
}

/// NFKD normalize then strip non-ASCII (removes diacritics: "café" → "cafe").
fn fold_to_ascii(text: &str) -> String {
    text.nfkd().filter(|c| c.is_ascii()).collect()
}

/// Generate all overlapping char n-grams (min_n..=max_n) from `text`.
///
/// Pipeline: lowercase → collapse non-alphanumeric runs to single space,
/// trim leading/trailing spaces → slide windows. Spaces are kept as
/// word-boundary markers so "hello world" at n=5 yields "hello", "ello ",
/// "llo w", "lo wo", "o wor", " worl", "world".
fn normalize_for_char_ngrams(text: &str) -> Vec<char> {
    let lowered = text.to_lowercase();
    let mut normalized = String::with_capacity(lowered.len());
    let mut prev_space = true;
    for c in lowered.chars() {
        if c.is_alphanumeric() {
            normalized.push(c);
            prev_space = false;
        } else if !prev_space {
            normalized.push(' ');
            prev_space = true;
        }
    }
    if normalized.ends_with(' ') {
        normalized.pop();
    }
    normalized.chars().collect()
}

fn char_ngrams_from_text(text: &str, min_n: usize, max_n: usize) -> Vec<String> {
    let chars = normalize_for_char_ngrams(text);
    let mut tokens = Vec::new();
    for n in min_n..=max_n {
        for window in chars.windows(n) {
            tokens.push(window.iter().collect::<String>());
        }
    }
    tokens
}

/// Fused char n-gram generation + TF counting.
///
/// Returns doc_len (total char n-grams generated, with multiplicity).
fn count_char_ngrams<S: std::hash::BuildHasher>(
    text: &str,
    min_n: usize,
    max_n: usize,
    tf_map: &mut HashMap<String, u32, S>,
) -> u32 {
    let chars = normalize_for_char_ngrams(text);
    let mut doc_len = 0u32;
    for n in min_n..=max_n {
        for window in chars.windows(n) {
            let ngram: String = window.iter().collect();
            *tf_map.entry(ngram).or_insert(0) += 1;
            doc_len += 1;
        }
    }
    doc_len
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plain_basic() {
        let tok = Tokenizer::new(false);
        assert_eq!(tok.tokenize_owned("Hello WORLD"), vec!["hello", "world"]);
    }

    #[test]
    fn test_plain_stopwords() {
        let tok = Tokenizer::new(true);
        let tokens = tok.tokenize_owned("the quick brown fox");
        assert_eq!(tokens, vec!["quick", "brown", "fox"]);
    }

    #[test]
    fn test_plain_punctuation() {
        let tok = Tokenizer::new(false);
        let tokens = tok.tokenize_owned("hello, world! how's it going?");
        assert_eq!(tokens, vec!["hello", "world", "how", "s", "it", "going"]);
    }

    #[test]
    fn test_unicode_diacritics() {
        let tok = Tokenizer::with_mode(TokenizerMode::Unicode, false);
        let tokens = tok.tokenize_owned("café résumé naïve");
        assert_eq!(tokens, vec!["cafe", "resume", "naive"]);
    }

    #[test]
    fn test_unicode_fullwidth() {
        let tok = Tokenizer::with_mode(TokenizerMode::Unicode, false);
        // NFKD normalizes fullwidth chars
        let tokens = tok.tokenize_owned("Ｈｅｌｌｏ");
        assert_eq!(tokens, vec!["hello"]);
    }

    #[test]
    fn test_stem_basic() {
        let tok = Tokenizer::with_mode(TokenizerMode::Stem, false);
        let tokens = tok.tokenize_owned("running cancellation connections");
        assert_eq!(tokens, vec!["run", "cancel", "connect"]);
    }

    #[test]
    fn test_stem_with_stopwords() {
        let tok = Tokenizer::with_mode(TokenizerMode::Stem, true);
        let tokens = tok.tokenize_owned("the cats are running quickly");
        // "the" and "are" removed, then stemmed
        // Snowball English: "cats"→"cat", "running"→"run", "quickly"→"quick"
        assert_eq!(tokens, vec!["cat", "run", "quick"]);
    }

    #[test]
    fn test_unicode_stem() {
        let tok = Tokenizer::with_mode(TokenizerMode::UnicodeStem, false);
        let tokens = tok.tokenize_owned("café résumés naïvely");
        assert_eq!(tokens, vec!["cafe", "resum", "naiv"]);
    }

    #[test]
    fn test_unicode_lowercase() {
        let tok = Tokenizer::with_mode(TokenizerMode::Unicode, false);
        // Full Unicode lowercase (not just ASCII)
        let tokens = tok.tokenize_owned("Ünïcödé STRASSE");
        assert_eq!(tokens, vec!["unicode", "strasse"]);
    }

    #[test]
    fn tokenizer_mode_id_round_trip() {
        for m in [
            TokenizerMode::Plain,
            TokenizerMode::Unicode,
            TokenizerMode::Stem,
            TokenizerMode::UnicodeStem,
        ] {
            assert_eq!(TokenizerMode::from_id(m.to_id()), Some(m));
        }
        assert_eq!(TokenizerMode::from_id(99), None);
    }

    #[test]
    fn tokenizer_exposes_mode_and_use_stopwords() {
        let t = Tokenizer::with_mode(TokenizerMode::Plain, true);
        assert_eq!(t.mode(), TokenizerMode::Plain);
        assert!(t.use_stopwords());
        let t = Tokenizer::with_mode(TokenizerMode::UnicodeStem, false);
        assert_eq!(t.mode(), TokenizerMode::UnicodeStem);
        assert!(!t.use_stopwords());
    }

    #[test]
    fn char_ngram_4_5_basic() {
        let tok = Tokenizer::with_mode(TokenizerMode::CharNgram { min_n: 4, max_n: 5 }, false);
        let tokens = tok.tokenize_owned("hello");
        // 4-grams: "hell", "ello"; 5-grams: "hello"
        assert!(tokens.contains(&"hell".to_string()));
        assert!(tokens.contains(&"ello".to_string()));
        assert!(tokens.contains(&"hello".to_string()));
        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn char_ngram_space_boundary() {
        let tok = Tokenizer::with_mode(TokenizerMode::CharNgram { min_n: 4, max_n: 4 }, false);
        let tokens = tok.tokenize_owned("hi world");
        // "hi" is only 2 chars — no 4-grams from it alone; cross-boundary 4-grams include space
        assert!(tokens.contains(&"worl".to_string()));
        assert!(tokens.contains(&"orld".to_string()));
    }

    #[test]
    fn char_ngram_uppercase_folded() {
        let tok = Tokenizer::with_mode(TokenizerMode::CharNgram { min_n: 4, max_n: 4 }, false);
        let t1 = tok.tokenize_owned("HELLO");
        let t2 = tok.tokenize_owned("hello");
        assert_eq!(t1, t2);
    }

    #[test]
    fn char_ngram_mode_id_roundtrip() {
        for m in [
            TokenizerMode::CharNgram { min_n: 4, max_n: 5 },
            TokenizerMode::CharNgram { min_n: 7, max_n: 8 },
        ] {
            assert_eq!(TokenizerMode::from_id(m.to_id()), Some(m));
        }
    }

    #[test]
    fn char_ngram_empty_text() {
        let tok = Tokenizer::with_mode(TokenizerMode::CharNgram { min_n: 4, max_n: 5 }, false);
        assert_eq!(tok.tokenize_owned(""), Vec::<String>::new());
        assert_eq!(tok.tokenize_owned("   "), Vec::<String>::new());
    }

    #[test]
    fn char_ngram_short_text_no_panic() {
        // text shorter than min_n should produce empty output, not panic
        let tok = Tokenizer::with_mode(TokenizerMode::CharNgram { min_n: 4, max_n: 5 }, false);
        assert_eq!(tok.tokenize_owned("hi"), Vec::<String>::new());
    }

    #[test]
    fn char_ngram_stopwords_flag_ignored() {
        // use_stopwords=true must not crash for CharNgram (stopword list unused)
        let tok = Tokenizer::with_mode(TokenizerMode::CharNgram { min_n: 4, max_n: 5 }, true);
        let tokens = tok.tokenize_owned("the quick brown fox");
        // "the" is 3 chars — no 4-grams; other words produce 4/5-grams
        assert!(tokens.contains(&"quic".to_string()));
    }

    #[test]
    fn test_modes_different_output() {
        let text = "The résumés are running";
        let plain = Tokenizer::with_mode(TokenizerMode::Plain, true).tokenize_owned(text);
        let unicode = Tokenizer::with_mode(TokenizerMode::Unicode, true).tokenize_owned(text);
        let stem = Tokenizer::with_mode(TokenizerMode::Stem, true).tokenize_owned(text);
        let us = Tokenizer::with_mode(TokenizerMode::UnicodeStem, true).tokenize_owned(text);

        // Plain keeps diacritics
        assert!(plain.contains(&"résumés".to_string()));
        // Unicode folds them
        assert!(unicode.contains(&"resumes".to_string()));
        // Stem stems but keeps diacritics
        assert!(stem.iter().any(|t| t.contains("résum")));
        // UnicodeStem folds + stems
        assert!(us.contains(&"resum".to_string()));
    }
}
