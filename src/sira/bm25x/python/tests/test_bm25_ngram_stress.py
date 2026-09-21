# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Stress test: build max_n=4 index over 50k synthetic docs, round-trip
through save/load/search, verify a known bigram appears in a top-10 result
for queries containing it. Also exercises the n-gram tier disk persistence.

Marked `@pytest.mark.slow` so fast CI runs can skip via `-m 'not slow'`.
Typical wall time on a modern x86 box: 30-90s.
"""

import random
import string

import pytest

import bm25x


def gen_corpus(n_docs=50_000, seed=42):
    """Synthetic corpus of 4-letter pseudowords drawn from a 2k vocabulary.

    Uniform sampling guarantees each unigram has DF >> 1 across 50k docs, so
    bigram/trigram matches are the discriminative signal — exactly what the
    n-gram tier is supposed to amplify.
    """
    rng = random.Random(seed)
    base_words = [''.join(rng.choices(string.ascii_lowercase, k=4)) for _ in range(2_000)]
    docs = []
    for _ in range(n_docs):
        n_tokens = rng.randint(20, 80)
        docs.append(' '.join(rng.choices(base_words, k=n_tokens)))
    return docs


@pytest.mark.slow
def test_large_corpus_max_n_4_round_trip(tmp_path):
    docs = gen_corpus()
    idx = bm25x.BM25(
        index=str(tmp_path / "idx"),
        max_n=4,
        n_features=1 << 20,  # 1M slots — enough for 50k docs without too much collision
        tokenizer="stem",
        use_stopwords=True,
    )
    idx.add(docs)
    idx.save(str(tmp_path / "idx"))

    # Pick a real bigram from doc 0 and confirm doc 0 appears in the top-10.
    tokens = idx.tokenize(docs[0])
    bigram = ' '.join(tokens[:2])
    res = idx.search(bigram, k=10)
    assert any(r[0] == 0 for r in res), \
        f"doc 0 missing from top-10 for its own bigram {bigram!r}"

    # Round-trip: save and reload preserves vocab + search behavior.
    idx2 = bm25x.BM25.load(str(tmp_path / "idx"))
    assert idx2.max_n() == 4
    assert idx2.n_features() == 1 << 20
    res2 = idx2.search(bigram, k=10)
    assert [r[0] for r in res] == [r[0] for r in res2]
