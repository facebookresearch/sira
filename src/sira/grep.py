# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Fast grep-like search using an inverted index with trigram acceleration.

Builds a word-level inverted index and a character-trigram index at
construction time. Substring lookups use trigram intersection to narrow
candidates, reducing from O(vocab_size) to O(candidate_size) per word.
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

_TRIGRAM_MIN_LEN = 3


class GrepIndex:
    """Inverted index for fast case-insensitive substring search."""

    def __init__(self, corpus_texts: list[str]) -> None:
        self.corpus_lower = [t.lower() for t in corpus_texts]
        self._inv: dict[str, set[int]] = defaultdict(set)
        for idx, text in enumerate(self.corpus_lower):
            for word in set(text.split()):
                self._inv[word].add(idx)

        # Character trigram index: trigram → set of vocab words containing it
        self._tri: dict[str, set[str]] = defaultdict(set)
        for word in self._inv:
            for i in range(len(word) - _TRIGRAM_MIN_LEN + 1):
                self._tri[word[i : i + _TRIGRAM_MIN_LEN]].add(word)

        logger.info(
            "Built grep index: %d docs, %d unique words, %d trigrams",
            len(corpus_texts), len(self._inv), len(self._tri),
        )

    def __len__(self) -> int:
        return len(self.corpus_lower)

    def _substr_vocab_words(self, word: str) -> list[str]:
        """Find vocab words that contain `word` as a substring (excluding exact match)."""
        if len(word) >= _TRIGRAM_MIN_LEN:
            trigrams = [word[i : i + _TRIGRAM_MIN_LEN] for i in range(len(word) - _TRIGRAM_MIN_LEN + 1)]
            candidates: set[str] | None = None
            for tri in trigrams:
                tri_words = self._tri.get(tri)
                if tri_words is None:
                    return []
                candidates = tri_words.copy() if candidates is None else candidates & tri_words
                if not candidates:
                    return []
            return [vw for vw in candidates if vw != word and word in vw]
        else:
            # Very short words (1-2 chars): trigram index can't help, linear scan
            return [vw for vw in self._inv if vw != word and word in vw]

    def _candidates_for_word(self, word: str) -> set[int]:
        """Find docs containing word as a substring of any vocab word."""
        exact = self._inv.get(word, set())
        extra: set[int] = set()
        for vw in self._substr_vocab_words(word):
            extra |= self._inv[vw]
        return exact | extra

    def search(self, pattern: str, max_matches: int = 1000) -> list[int]:
        """Case-insensitive substring search. Returns doc indices."""
        pat = pattern.lower().strip()
        if not pat:
            return []
        words = pat.split()

        candidates: set[int] | None = None
        for w in words:
            docs = self._candidates_for_word(w)
            if not docs:
                return []
            candidates = docs.copy() if candidates is None else candidates & docs
            if not candidates:
                return []

        return [i for i in candidates if pat in self.corpus_lower[i]][:max_matches]

    def search_multi(self, patterns: list[str], max_matches: int = 1000) -> set[int]:
        """Search multiple patterns and pool all results."""
        pooled: set[int] = set()
        for pat in patterns:
            pooled.update(self.search(pat, max_matches))
        return pooled

    def search_or(
        self, pattern: str, max_matches: int = 1000
    ) -> list[tuple[int, int]]:
        """Search with pipe-separated OR patterns.

        Returns list of (doc_index, match_position) tuples.
        """
        parts = [p.strip() for p in pattern.split("|") if p.strip()]
        if not parts:
            return []
        matched: list[tuple[int, int]] = []
        seen: set[int] = set()
        for p in parts:
            for idx in self.search(p, max_matches):
                if idx not in seen:
                    seen.add(idx)
                    pos = self.corpus_lower[idx].find(p.lower())
                    matched.append((idx, max(pos, 0)))
                    if len(matched) >= max_matches:
                        return matched
        return matched
