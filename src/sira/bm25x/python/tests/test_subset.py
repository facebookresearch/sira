# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for subset (pre-filtered) search — single and batch mode."""

import pytest

import bm25x


@pytest.fixture
def index():
    """Create and populate a test index."""
    idx = bm25x.BM25(method="lucene", k1=1.5, b=0.75, use_stopwords=False)
    idx.add(
        [
            "the quick brown fox jumps over the lazy dog",  # 0
            "a lazy brown dog sits on the mat",  # 1
            "the quick red car drives fast",  # 2
            "a slow brown truck on the highway",  # 3
            "the fox is quick and clever",  # 4
        ]
    )
    return idx


class TestSingleSubset:
    """Single query with subset."""

    def test_subset_limits_results(self, index):
        """Only documents in the subset should appear in results."""
        results = index.search("brown", k=10, subset=[0, 2])
        doc_ids = [r[0] for r in results]
        assert all(d in [0, 2] for d in doc_ids)

    def test_subset_excludes_docs(self, index):
        """Documents NOT in the subset should never appear."""
        # "brown" appears in docs 0, 1, 3 — but subset only allows [0, 3]
        results = index.search("brown", k=10, subset=[0, 3])
        doc_ids = [r[0] for r in results]
        assert 1 not in doc_ids
        assert 2 not in doc_ids
        assert 4 not in doc_ids

    def test_subset_empty(self, index):
        """Empty subset should return no results."""
        results = index.search("brown", k=10, subset=[])
        assert results == []

    def test_subset_single_doc(self, index):
        """Subset with one document."""
        results = index.search("brown", k=10, subset=[1])
        assert len(results) == 1
        assert results[0][0] == 1

    def test_subset_no_match(self, index):
        """Subset that doesn't overlap with query matches."""
        # "fox" appears in docs 0 and 4, but subset is [1, 2, 3]
        results = index.search("fox", k=10, subset=[1, 2, 3])
        assert results == []

    def test_subset_scores_match_unfiltered(self, index):
        """Scores for a document should be the same whether filtered or not."""
        unfiltered = index.search("quick fox", k=10)
        filtered = index.search("quick fox", k=10, subset=[0])

        score_unfiltered = next(s for d, s in unfiltered if d == 0)
        score_filtered = next(s for d, s in filtered if d == 0)
        assert abs(score_unfiltered - score_filtered) < 1e-6

    def test_subset_respects_k(self, index):
        """k should limit results even within subset."""
        results = index.search("the", k=1, subset=[0, 1, 2, 3, 4])
        assert len(results) <= 1

    def test_subset_out_of_range(self, index):
        """Out-of-range doc IDs in subset should be silently ignored."""
        results = index.search("brown", k=10, subset=[0, 999])
        doc_ids = [r[0] for r in results]
        assert 999 not in doc_ids
        assert 0 in doc_ids

    def test_subset_results_sorted_by_score(self, index):
        """Results should be sorted by descending score."""
        results = index.search("brown", k=10, subset=[0, 1, 3])
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


class TestBatchSubset:
    """Batch queries with per-query subsets."""

    def test_batch_subset_basic(self, index):
        """Each query should respect its own subset."""
        results = index.search(
            ["brown", "quick"],
            k=10,
            subset=[[0, 1], [2, 4]],
        )
        assert len(results) == 2

        # "brown" with subset [0, 1]
        brown_ids = [r[0] for r in results[0]]
        assert all(d in [0, 1] for d in brown_ids)

        # "quick" with subset [2, 4]
        quick_ids = [r[0] for r in results[1]]
        assert all(d in [2, 4] for d in quick_ids)

    def test_batch_subset_empty_subsets(self, index):
        """Empty subsets should return empty results."""
        results = index.search(
            ["brown", "quick"],
            k=10,
            subset=[[], []],
        )
        assert results == [[], []]

    def test_batch_subset_mixed(self, index):
        """Mix of matching and non-matching subsets."""
        results = index.search(
            ["fox", "fox"],
            k=10,
            subset=[[0, 4], [1, 2, 3]],
        )
        # First query: fox in docs 0 and 4, subset allows both
        assert len(results[0]) == 2
        # Second query: fox not in docs 1, 2, 3
        assert results[1] == []

    def test_batch_subset_scores_match_single(self, index):
        """Batch subset scores should match individual subset calls."""
        batch = index.search(
            ["brown", "quick"],
            k=10,
            subset=[[0, 1, 3], [0, 2, 4]],
        )
        single_brown = index.search("brown", k=10, subset=[0, 1, 3])
        single_quick = index.search("quick", k=10, subset=[0, 2, 4])

        assert len(batch[0]) == len(single_brown)
        assert len(batch[1]) == len(single_quick)

        for (bd, bs), (sd, ss) in zip(batch[0], single_brown):
            assert bd == sd
            assert abs(bs - ss) < 1e-6

        for (bd, bs), (sd, ss) in zip(batch[1], single_quick):
            assert bd == sd
            assert abs(bs - ss) < 1e-6

    def test_batch_subset_different_k(self, index):
        """k applies to all queries in the batch."""
        results = index.search(
            ["the", "the"],
            k=2,
            subset=[[0, 1, 2, 3, 4], [0, 1]],
        )
        assert len(results[0]) <= 2
        assert len(results[1]) <= 2


class TestBatchWithoutSubset:
    """Batch queries without subsets (full index search)."""

    def test_batch_matches_sequential(self, index):
        """Batch results should match individual search calls."""
        queries = ["brown fox", "lazy dog", "quick car"]
        batch = index.search(queries, k=10)

        for i, q in enumerate(queries):
            single = index.search(q, k=10)
            assert len(batch[i]) == len(single)
            for (bd, bs), (sd, ss) in zip(batch[i], single):
                assert bd == sd
                assert abs(bs - ss) < 1e-6

    def test_batch_empty_queries(self, index):
        """Batch with queries that match nothing."""
        results = index.search(["xyznonexistent", "abcnotfound"], k=10)
        assert results == [[], []]
