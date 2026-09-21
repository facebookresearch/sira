# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for NGramIndex Python bindings."""

import json
import subprocess
import sys

import pytest

import bm25x


def test_basic_df_and_cooccur():
    idx = bm25x.NGramIndex(max_n=4, tokenizer="unicode_stem", use_stopwords=True)
    idx.add(
        [
            "machine learning models are powerful",
            "machine learning is a subfield of ai",
            "deep learning models perform well",
        ]
    )
    assert idx.num_docs() == 3
    # bigram "machine learning" stems to "machin learn"
    assert idx.df("machin learn") == 2
    assert idx.contains("learn")
    assert not idx.contains("rocket science")

    # Cooccurring with the bigram should surface "model" and "ai".
    cooccur = idx.cooccur(["machin learn"], top_k=200, df_max=10)
    names = {n for n, _ in cooccur}
    assert "model" in names
    assert "machin learn" not in names


def test_save_load_roundtrip(tmp_path):
    idx = bm25x.NGramIndex(max_n=3)
    idx.add(["alpha beta gamma", "beta gamma delta"])
    p = str(tmp_path / "ngram.idx")
    idx.save(p)
    loaded = bm25x.NGramIndex.load(p)
    assert loaded.num_docs() == 2
    assert loaded.df("beta") == 2
    assert loaded.df("alpha beta") == 1
    # cooccur survives the round-trip
    out = loaded.cooccur(["beta"], top_k=10)
    assert any(name == "gamma" for name, _ in out)


def test_constructor_validation():
    with pytest.raises(ValueError):
        bm25x.NGramIndex(max_n=0)
    with pytest.raises(ValueError):
        bm25x.NGramIndex(tokenizer="bogus")


def test_add_empty_corpus():
    idx = bm25x.NGramIndex(max_n=4)
    idx.add([])
    assert idx.num_docs() == 0
    assert idx.vocab_size() == 0
    assert idx.df("anything") == 0
    assert not idx.contains("anything")


def test_add_empty_and_blank_strings():
    idx = bm25x.NGramIndex(max_n=4)
    idx.add(["", "   ", "real content"])
    assert idx.num_docs() == 3
    assert idx.df("real") == 1
    assert idx.df("content") == 1
    # the empty string is never a valid n-gram
    assert idx.df("") == 0


def test_incremental_add_doc_ids_continue():
    # Co-occurrence is per-document, so the fixture must put the seed term and
    # its co-occurring term in the SAME doc. Single-term docs (doc 0={alpha},
    # doc 1={beta}) never co-occur and would make cooccur() return [].
    idx = bm25x.NGramIndex(max_n=2)
    idx.add(["alpha solo"])           # doc 0: alpha + solo
    idx.add(["beta only", "alpha beta"])  # doc 1: beta+only, doc 2: alpha+beta
    assert idx.num_docs() == 3
    # alpha is in docs 0 and 2 -> df=2.
    assert idx.df("alpha") == 2
    # beta is in docs 1 and 2 -> df=2.
    assert idx.df("beta") == 2
    # cooccur on "alpha": intersection of alpha's docs {0,2} with each
    # candidate's docs. "beta" -> {1,2} intersect {0,2} = {2} -> count=1.
    out = idx.cooccur(["alpha"], top_k=20)
    counts = dict(out)
    assert counts.get("beta") == 1
    # "only" is in doc 1 only -> intersect with {0,2} = empty -> not in output.
    assert "only" not in counts


def test_max_n_one_only_unigrams():
    idx = bm25x.NGramIndex(max_n=1)
    idx.add(["alpha beta gamma"])
    assert idx.df("alpha") == 1
    # bigrams must NOT exist at max_n=1
    assert idx.df("alpha beta") == 0


def test_use_stopwords_false_keeps_them():
    idx = bm25x.NGramIndex(max_n=2, use_stopwords=False)
    idx.add(["the cat sat"])
    assert idx.df("the") == 1
    assert idx.df("the cat") == 1


def test_cooccur_empty_query():
    idx = bm25x.NGramIndex(max_n=2)
    idx.add(["alpha beta"])
    assert idx.cooccur([], top_k=10) == []


def test_cooccur_top_k_zero():
    idx = bm25x.NGramIndex(max_n=2)
    idx.add(["alpha beta gamma"])
    assert idx.cooccur(["alpha"], top_k=0) == []


def test_cooccur_df_max_zero_returns_empty():
    idx = bm25x.NGramIndex(max_n=2)
    idx.add(["alpha beta"])
    assert idx.cooccur(["alpha"], top_k=10, df_max=0) == []


def test_cooccur_disjoint_query_returns_empty():
    idx = bm25x.NGramIndex(max_n=2)
    idx.add(["alpha", "beta"])
    assert idx.cooccur(["alpha", "beta"], top_k=10) == []


def test_cooccur_results_are_sorted_count_desc():
    idx = bm25x.NGramIndex(max_n=2)
    idx.add(
        [
            "alpha beta gamma",
            "alpha gamma delta",
            "alpha beta delta",
        ]
    )
    out = idx.cooccur(["alpha"], top_k=50)
    counts = [c for _, c in out]
    assert counts == sorted(counts, reverse=True)
    # rerunning must give an identical, deterministic result
    assert out == idx.cooccur(["alpha"], top_k=50)


def test_unicode_tokens_indexed():
    idx = bm25x.NGramIndex(max_n=2)
    idx.add(["café résumé"])
    # unicode_stem normalizes diacritics: "café" -> "cafe"
    assert idx.df("cafe") == 1


def test_save_load_empty_index(tmp_path):
    idx = bm25x.NGramIndex(max_n=4)
    p = str(tmp_path / "empty.idx")
    idx.save(p)
    loaded = bm25x.NGramIndex.load(p)
    assert loaded.num_docs() == 0
    assert loaded.vocab_size() == 0
    assert loaded.cooccur(["foo"]) == []


def test_load_nonexistent_path_raises(tmp_path):
    with pytest.raises(ValueError):
        bm25x.NGramIndex.load(str(tmp_path / "does_not_exist.idx"))


def test_save_load_in_fresh_process(tmp_path):
    """A separately spawned interpreter must be able to load the index and
    reproduce DF + cooccur results — the critical real-world property."""
    p = str(tmp_path / "fresh.idx")
    builder = bm25x.NGramIndex(max_n=3)
    builder.add(
        [
            "alpha beta gamma",
            "alpha beta delta",
            "alpha epsilon zeta",
        ]
    )
    builder.save(p)
    expected_df = builder.df("alpha beta")
    expected_cooccur = builder.cooccur(["alpha"], top_k=10)

    # Read it back from a brand-new Python process.
    code = (
        "import json, sys; from bm25x import NGramIndex; "
        f"idx = NGramIndex.load({p!r}); "
        "print(json.dumps({"
        "'num_docs': idx.num_docs(), "
        "'df': idx.df('alpha beta'), "
        "'cooccur': idx.cooccur(['alpha'], top_k=10)}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(out.stdout.strip())
    assert payload["num_docs"] == 3
    assert payload["df"] == expected_df
    # JSON deserializes tuples as lists; compare element-wise.
    assert [list(t) for t in expected_cooccur] == payload["cooccur"]


def test_large_corpus_cooccur_correctness():
    """End-to-end sanity: 200 synthetic docs, verify hand-computed counts."""
    docs = []
    for i in range(200):
        # Every doc has "anchor". Half have "left", the other half have "right".
        side = "left" if i < 100 else "right"
        docs.append(f"anchor word {side} extra{i}")
    idx = bm25x.NGramIndex(max_n=2)
    idx.add(docs)
    # df sanity
    assert idx.df("anchor") == 200
    assert idx.df("left") == 100
    assert idx.df("right") == 100
    # cooccur(anchor, df_max=150) should include "left" (100) and "right" (100)
    # but exclude very-frequent "anchor"-co-occurrences if their df > 150.
    out = idx.cooccur(["anchor"], top_k=20, df_max=150)
    names = {n for n, _ in out}
    assert "left" in names
    assert "right" in names
    # Counts for left/right against intersection (== anchor's docs == 200) are 100.
    counts = dict(out)
    assert counts["left"] == 100
    assert counts["right"] == 100


def test_releases_gil():
    """Smoke test that add() releases the GIL — two threads can call add()
    on independent indices and finish faster than sequential.

    Uses a small `n_features` (1<<14 = 16K slots) so the constructor's
    dense Mutex-array allocation doesn't dominate the timing. add() is
    where the GIL release matters in practice; the constructor runs
    once per index and doesn't repeatedly compete for the GIL.
    """
    import threading
    import time

    docs = ["the quick brown fox jumps over the lazy dog " * 20] * 200

    def build():
        idx = bm25x.NGramIndex(max_n=3, n_features=1 << 14)
        idx.add(docs)
        assert idx.num_docs() == 200

    # Sequential baseline
    t0 = time.perf_counter()
    build()
    build()
    seq = time.perf_counter() - t0

    # Threaded
    t0 = time.perf_counter()
    threads = [threading.Thread(target=build) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    par = time.perf_counter() - t0

    # If the GIL were held the whole time, par ≈ seq. Allow a generous
    # 25% margin for noise — true parallelism gives ~50%.
    assert par < seq * 0.85, f"GIL apparently held: seq={seq:.3f}s par={par:.3f}s"
