# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Type stubs for bm25x — Rust BM25 search engine with search expression support."""

from typing import Optional, Union

import numpy as np
import numpy.typing as npt

class BM25:
    """BM25 search index with search expression support.

    The BM25 score sums contributions from every n in ``ngrams`` (default
    ``[1, 2, 3, 4]``). Pass ``ngrams=`` to pick an arbitrary subset of
    ``{1..8}`` — e.g. ``[1]`` for pure unigram, ``[1, 4]`` to skip bigrams
    and trigrams, or ``[2]`` for bigram-only score. The unigram tier is
    always BUILT (search_expr / term DF / prefix scan depend on it); when
    ``1 ∉ ngrams`` it just doesn't contribute to the score.

    Legacy ``max_n`` is honored when ``ngrams`` is omitted: it implies a
    contiguous ``1..=max_n`` set, equivalent to passing
    ``ngrams=list(range(1, max_n + 1))``.
    """

    def __new__(
        cls,
        index: Optional[str] = None,
        method: str = "lucene",
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
        tokenizer: str = "unicode_stem",
        use_stopwords: bool = True,
        cuda: bool = False,
        max_n: int = 4,
        ngrams: Optional[list[int]] = None,
        n_features: int = 8388608,
    ) -> "BM25": ...

    def add(self, documents: list[str]) -> list[int]: ...
    def add_bytes(self, data: bytes) -> list[int]: ...

    def search(
        self,
        query: Union[str, list[str]],
        k: int,
        subset: Optional[Union[list[int], list[list[int]]]] = None,
    ) -> Union[list[tuple[int, float]], list[list[tuple[int, float]]]]: ...

    def search_expr(
        self,
        query: Union[str, list[str]],
        k: int,
    ) -> Union[list[tuple[int, float]], list[list[tuple[int, float]]]]: ...

    def search_expr_numpy(
        self,
        queries: list[str],
        k: int,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32], npt.NDArray[np.int64]]: ...

    def search_expr_numpy_with_extras(
        self,
        queries: list[str],
        extra_ngrams: list[list[str]],
        k: int,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32], npt.NDArray[np.int64]]: ...

    def ngram_score_breakdown_for_docs(
        self,
        doc_indices: list[int],
        ngram_keys: list[str],
    ) -> list[tuple[int, int, float, int, float]]:
        """Returns (slot_idx, doc_pos, idf, tf_raw, score) for non-zero entries."""
        ...

    def delete(self, doc_ids: list[int]) -> None: ...
    def update(self, doc_id: int, new_text: str) -> None: ...
    def enrich(self, doc_id: int, extra_text: str) -> None: ...
    def unenrich(self, doc_id: int, extra_text: str) -> None: ...
    def enrich_exact(self, doc_id: int, extra_text: str) -> None: ...
    def unenrich_exact(self, doc_id: int, extra_text: str) -> None: ...
    def enrich_batch(self, items: list[tuple[int, list[str]]]) -> None: ...
    def enrich_exact_batch(self, items: list[tuple[int, list[str]]]) -> None: ...
    def disable_auto_save(self) -> None: ...
    def save(self, index: str) -> None: ...

    @staticmethod
    def load(index: str, mmap: bool = False, cuda: bool = False) -> "BM25": ...

    def score(
        self,
        query: Union[str, list[str]],
        documents: Union[list[str], list[list[str]]],
    ) -> Union[list[float], list[list[float]]]: ...

    def upload_to_gpu(self) -> None: ...
    def __len__(self) -> int: ...

    # Index exploration
    def vocab_size(self) -> int: ...
    def tokenize(self, text: str) -> list[str]: ...
    def get_term_df(self, term: str) -> Optional[int]: ...
    def get_term_idf(self, term: str) -> Optional[float]: ...
    def get_vocab_matches(self, prefix: str, limit: int = 20) -> list[tuple[str, int]]: ...
    def cooccurring_terms(self, term: str, limit: int = 20) -> list[tuple[str, int, int]]: ...

    # Multi-gram tier
    def max_n(self) -> int: ...
    def ngrams(self) -> list[int]: ...
    def score_unigram(self) -> bool: ...
    def n_features(self) -> int: ...
    def ngram_df(self, ngram: str) -> int: ...

class Tfidf:
    """TF-IDF keyword extraction with configurable tokenizer and n-grams."""

    def __new__(
        cls,
        top_k: int = 20,
        ngram_range: tuple[int, int] = (1, 1),
        sublinear_tf: bool = True,
        min_df: int = 1,
        tokenizer: str = "unicode_stem",
        use_stopwords: bool = True,
        use_hashing: bool = False,
        n_features: int = 8388608,
        dedup: bool = False,
    ) -> "Tfidf": ...

    def fit_transform(
        self,
        texts: list[str],
    ) -> tuple[
        list[list[str]],
        Union[npt.NDArray[np.float32], list[list[float]]],
    ]: ...

    def fit_transform_ndjson(
        self,
        texts: list[str],
        ids: list[str],
        batch_size: int = 500_000,
    ) -> bytes: ...

    def vocab_size(self) -> int: ...
    def num_docs(self) -> int: ...

class NGramIndex:
    """Hashed inverted index over 1..=max_n grams for DF + cooccur lookups.

    Memory is bounded by `n_features` (default 1<<23 = 8M slots), independent
    of corpus size. Hash collisions cause `df()`/`cooccur()` counts to be
    *over-estimates* of the true value (never under). Increase `n_features`
    to reduce noise at the cost of memory.
    """

    def __new__(
        cls,
        max_n: int = 4,
        n_features: int = 1 << 23,
        tokenizer: str = "unicode_stem",
        use_stopwords: bool = True,
    ) -> "NGramIndex": ...

    def add(self, docs: list[str]) -> None: ...
    def df(self, ngram: str) -> int: ...
    def contains(self, ngram: str) -> bool: ...
    def num_docs(self) -> int: ...
    def vocab_size(self) -> int: ...
    def n_features(self) -> int: ...
    def slot(self, ngram: str) -> int: ...

    def cooccur(
        self,
        query_ngrams: list[str],
        top_k: int = 20,
        df_max: int = 0xFFFFFFFF,
    ) -> list[tuple[str, int]]: ...

    def min_df_ngram(
        self,
        text: str,
        n_min: int = 1,
        n_max: int = 4,
    ) -> Optional[tuple[int, str, int]]: ...

    def min_df_ngram_batch(
        self,
        texts: list[str],
        n_min: int = 1,
        n_max: int = 4,
    ) -> list[Optional[tuple[int, str, int]]]: ...

    def filter_candidates(
        self,
        candidates: list[str],
        doc_text: str,
        prior_enrichments: list[str],
        max_df: int,
        max_n: int = 4,
        require_in_vocab: bool = True,
        collect_verdicts: bool = False,
    ) -> tuple[
        list[str],
        list[tuple[str, int]],
        list[tuple[str, str, Optional[int], bool, str]],
    ]: ...

    def prepare_doc(
        self,
        doc_text: str,
        max_n: int = 4,
        max_df: int = 0xFFFFFFFF,
    ) -> tuple[list[str], list[tuple[str, int]], list[str]]: ...

    def save(self, path: str) -> None: ...

    @staticmethod
    def load(path: str) -> "NGramIndex": ...

class Tokenizer:
    """Lowercase + (optional) Unicode normalize + (optional) stem + (optional) stopword filter."""

    def __new__(
        cls,
        mode: str = "unicode_stem",
        use_stopwords: bool = True,
    ) -> "Tokenizer": ...

    def tokenize(self, text: str) -> list[str]: ...

def is_gpu_available() -> bool: ...
