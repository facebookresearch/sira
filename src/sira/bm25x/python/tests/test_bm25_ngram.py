# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the multi-gram BM25 Python bindings.

Covers the new constructor knobs (`max_n`, `n_features`), the three accessor
methods (`max_n()`, `n_features()`, `ngram_df()`), the n-gram tier round-trip
through save/load, and constructor input validation.
"""

import pytest

import bm25x


def test_default_max_n_is_4():
    idx = bm25x.BM25()
    assert idx.max_n() == 4
    assert idx.n_features() == 1 << 23


def test_max_n_2_indexes_bigrams_via_hash():
    idx = bm25x.BM25(max_n=2, n_features=1 << 16, tokenizer="stem")
    idx.add(["machine learning is fun", "deep learning rocks"])
    # Stemmed bigrams "machin learn" and "deep learn" both appear in the corpus.
    assert idx.ngram_df("machin learn") >= 1
    assert idx.ngram_df("deep learn") >= 1
    # Unknown bigram returns 0 (slot empty under low collision).
    assert idx.ngram_df("nothing here") == 0


def test_search_prefers_ngram_match():
    idx = bm25x.BM25(max_n=4, n_features=1 << 16, tokenizer="stem")
    idx.add([
        "machine learning models are powerful tools",
        "vision models for medical imaging",
    ])
    res = idx.search("machine learning models", 2)
    # doc 0 has bigram + trigram match; should rank first.
    assert res[0][0] == 0


def test_save_load_round_trip(tmp_path):
    p = str(tmp_path / "idx")
    a = bm25x.BM25(index=p, max_n=3, n_features=1 << 16, tokenizer="stem")
    a.add(["alpha beta gamma delta"])
    # Save explicitly to ensure the on-disk layout is up-to-date even if
    # auto-save is enabled (defensive — `add` already auto-saves).
    a.save(p)
    del a
    b = bm25x.BM25.load(p)
    assert b.max_n() == 3
    assert b.n_features() == 1 << 16
    assert b.ngram_df("alpha beta gamma") >= 1


def test_max_n_1_does_not_allocate_side(tmp_path):
    idx = bm25x.BM25(max_n=1)
    assert idx.max_n() == 1
    assert idx.n_features() == 0
    # ngram_df should still work (returns 0 for any input — no n-gram tier).
    assert idx.ngram_df("anything at all") == 0


def test_constructor_validates_max_n():
    with pytest.raises(ValueError, match="max_n must be >= 1"):
        bm25x.BM25(max_n=0)
    with pytest.raises(ValueError, match="impractical"):
        bm25x.BM25(max_n=9)


def test_constructor_validates_n_features_power_of_two():
    with pytest.raises(ValueError, match="power of 2"):
        bm25x.BM25(max_n=2, n_features=12345)
