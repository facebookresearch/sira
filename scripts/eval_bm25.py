# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Build bm25x indices and evaluate BM25 Recall@K on an MTEB dataset.

For each n-gram candidate, builds an index, retrieves top-K, computes
NDCG + Recall, and symlinks the chosen config as ``index/best`` and
``eval/best.json``.

The n-gram order is chosen on ``selection_split`` (default ``dev``, falling back
to ``train``) and never on ``data.split``, which is reported as-is. A dataset
that ships no split other than the one being reported gets no sweep at all: the
fixed ``bm25.max_n`` is used, so the reported numbers stay selection-free.

Usage::

    source sandbox.sh
    python scripts/eval_bm25.py
    python scripts/eval_bm25.py data=fiqa
    python scripts/eval_bm25.py bm25.max_n_candidates='[1]' data.k_values='[10,200]'

Config groups::

    global  — db_root (shared across all scripts)
    data/   — dataset name, k_values
    bm25/   — index parameters (max_n_candidates, tokenizer, method, k1, b, cuda)

Outputs under ``{db_root}/{data.name}/``::

    index/bm25-n{ngrams}-{tokenizer}/    — persisted bm25x index
    index/best -> best config            — symlink
    eval/bm25-n{ngrams}-{tokenizer}.json — evaluation metrics
    eval/best.json -> best eval          — symlink
"""

import json
import logging
import os
import time

import hydra
from bm25x import BM25
from omegaconf import DictConfig
from sira.schema.mteb import (
    COL_ID,
    COL_TEXT,
    DatasetDir,
    load_qrels_dict,
    read_corpus_texts,
    read_queries,
)

logger = logging.getLogger(__name__)


def _ds(cfg: DictConfig) -> DatasetDir:
    return DatasetDir(root=cfg.db_root, name=cfg.data.name)


def _load_split(
    ds: DatasetDir, split: str
) -> tuple[list[str], list[str], dict[str, dict[str, int]]]:
    """Read one split's queries + qrels. Returns (query_ids, query_texts, qrels)."""
    queries_df = read_queries(ds.queries(split))
    return (
        queries_df.get_column(COL_ID).to_list(),
        queries_df.get_column(COL_TEXT).to_list(),
        load_qrels_dict(ds.qrels(split)),
    )


def _load_or_build_index(
    ds: DatasetDir, bm25_cfg: DictConfig, max_n: int, texts: list[str]
) -> BM25:
    """Load a persisted bm25x index for this max_n, building it if absent."""
    index_dir = ds.bm25_index(max_n, bm25_cfg.tokenizer)
    if os.path.exists(os.path.join(index_dir, "header.bin")):
        logger.info("Loading existing index from %s", index_dir)
        return BM25.load(index_dir, cuda=bm25_cfg.cuda)

    logger.info("Building bm25x index (max_n=%d) over %d docs …", max_n, len(texts))
    bm25 = BM25(
        index=index_dir,
        cuda=bm25_cfg.cuda,
        max_n=max_n,
        tokenizer=bm25_cfg.tokenizer,
        method=bm25_cfg.method,
        k1=bm25_cfg.k1,
        b=bm25_cfg.b,
    )
    bm25.add(texts)
    logger.info("Index built: %d docs", len(bm25))
    return bm25


def _search_and_evaluate(
    bm25: BM25,
    index_dir: str,
    doc_ids: list[str],
    query_ids: list[str],
    query_texts: list[str],
    qrels: dict[str, dict[str, int]],
    k_values: list[int],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Search one query set and score it. Returns (metrics, beir_results)."""
    max_k = max(k_values)
    logger.info("Searching %d queries @ k=%d …", len(query_texts), max_k)
    try:
        batch_results = bm25.search(query_texts, k=max_k)
    except (ValueError, RuntimeError) as e:
        if "CUDA_ERROR_OUT_OF_MEMORY" not in str(e) and "out of memory" not in str(e):
            raise
        logger.warning("GPU OOM, falling back to CPU: %s", e)
        bm25 = BM25.load(index_dir, cuda=False)
        batch_results = bm25.search(query_texts, k=max_k)

    beir_results: dict[str, dict[str, float]] = {}
    for qid, hits in zip(query_ids, batch_results):
        beir_results[qid] = {doc_ids[idx]: float(score) for idx, score in hits}

    from beir.retrieval.evaluation import EvaluateRetrieval

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, beir_results, k_values
    )
    return {**ndcg, **recall, **precision}, beir_results


def build_and_evaluate(cfg: DictConfig) -> None:
    """Build indices for the n-gram candidates, evaluate, and pick a config.

    The n-gram order is chosen on a held-out split, never on the split we
    report. Datasets that ship no held-out split get no sweep at all: they use
    the fixed ``bm25.max_n`` so the reported numbers stay selection-free.
    """
    ds = _ds(cfg)
    bm25_cfg = cfg.bm25
    k_values = list(cfg.data.k_values)
    selection_metric = cfg.selection_metric
    eval_split = cfg.data.split

    sel_split = ds.resolve_selection_split(cfg.selection_split, eval_split)

    doc_ids, texts = read_corpus_texts(ds.corpus)
    query_ids, query_texts, qrels = _load_split(ds, eval_split)

    if sel_split is None:
        candidates = [int(bm25_cfg.max_n)]
        logger.warning(
            "[%s] No held-out split (only %r) — skipping the n-gram sweep and "
            "using the fixed bm25.max_n=%d. Selecting on %r would bias the "
            "reported numbers.",
            ds.name, eval_split, candidates[0], eval_split,
        )
        sel_query_ids: list[str] = []
        sel_query_texts: list[str] = []
        sel_qrels: dict[str, dict[str, int]] = {}
    else:
        candidates = [int(n) for n in bm25_cfg.max_n_candidates]
        sel_query_ids, sel_query_texts, sel_qrels = _load_split(ds, sel_split)
        logger.info(
            "[%s] Selecting n-gram order on %r (%d queries); reporting on %r.",
            ds.name, sel_split, len(sel_query_ids), eval_split,
        )

    results: dict[int, dict[str, float]] = {}
    sel_results: dict[int, dict[str, float]] = {}
    beir_results_all: dict[int, dict[str, dict[str, float]]] = {}

    for max_n in candidates:
        tag = ds._bm25_tag(max_n, bm25_cfg.tokenizer)
        logger.info("--- %s ---", tag)
        index_dir = ds.bm25_index(max_n, bm25_cfg.tokenizer)
        eval_path = ds.eval_baseline(max_n, bm25_cfg.tokenizer)

        # Only reusable when there is nothing to sweep; a real sweep needs the
        # selection-split metrics, which the cached eval json does not hold.
        if len(candidates) == 1 and os.path.exists(eval_path):
            logger.info("Eval already done at %s — loading.", eval_path)
            with open(eval_path) as f:
                results[max_n] = json.load(f)
            continue

        bm25 = _load_or_build_index(ds, bm25_cfg, max_n, texts)

        if sel_split is not None:
            sel_metrics, _ = _search_and_evaluate(
                bm25, index_dir, doc_ids,
                sel_query_ids, sel_query_texts, sel_qrels, k_values,
            )
            sel_results[max_n] = sel_metrics

        metrics, beir_res = _search_and_evaluate(
            bm25, index_dir, doc_ids, query_ids, query_texts, qrels, k_values
        )
        results[max_n] = metrics
        beir_results_all[max_n] = beir_res

        os.makedirs(os.path.dirname(eval_path), exist_ok=True)
        with open(eval_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved to %s", eval_path)

    if not results:
        return

    logger.info("Summary for %s (%s split):", ds.name, eval_split)
    for max_n, metrics in sorted(results.items()):
        tag = ds._bm25_tag(max_n, bm25_cfg.tokenizer)
        parts = [f"  {tag}:"]
        for k in k_values:
            parts.append(
                f"NDCG@{k}={metrics.get(f'NDCG@{k}', 0):.4f} "
                f"Recall@{k}={metrics.get(f'Recall@{k}', 0):.4f}"
            )
        logger.info("  ".join(parts))

    if sel_results:
        best_n = max(sel_results, key=lambda n: sel_results[n].get(selection_metric, 0))
        logger.info(
            "Selected %s on %r (%s=%.4f)",
            ds._bm25_tag(best_n, bm25_cfg.tokenizer),
            sel_split,
            selection_metric,
            sel_results[best_n].get(selection_metric, 0),
        )
    else:
        best_n = candidates[0]
        logger.info(
            "Using %s (fixed, no selection performed)",
            ds._bm25_tag(best_n, bm25_cfg.tokenizer),
        )
    best_tag = ds._bm25_tag(best_n, bm25_cfg.tokenizer)

    # Index best symlink (directory, not json)
    best_index = ds.bm25_index(best_n, bm25_cfg.tokenizer)
    if os.path.islink(ds.bm25_index_best):
        os.remove(ds.bm25_index_best)
    os.symlink(
        os.path.relpath(best_index, os.path.dirname(ds.bm25_index_best)),
        ds.bm25_index_best,
    )
    logger.info("Linked %s → %s", ds.bm25_index_best, os.readlink(ds.bm25_index_best))

    meta = {
        "stage": "baseline",
        "dataset": ds.name,
        "best_config": best_tag,
        "metrics": results[best_n],
        "all_configs": {
            ds._bm25_tag(n, bm25_cfg.tokenizer): m for n, m in results.items()
        },
        "bm25_params": {
            "method": bm25_cfg.method,
            "tokenizer": bm25_cfg.tokenizer,
            "k1": bm25_cfg.k1,
            "b": bm25_cfg.b,
            "max_n": best_n,
        },
        "eval_split": eval_split,
        "selection_split": sel_split,
        "timestamp": int(time.time()),
    }
    # Eval + index best.meta.json
    ds.update_best(
        best_links=[ds.eval_baseline_best],
        target_name=f"{best_tag}.json",
        meta=meta,
        selection_metric=selection_metric,
        selection_metrics=sel_results.get(best_n),
    )
    # Also write meta next to index best symlink
    with open(
        os.path.join(os.path.dirname(ds.bm25_index_best), "best.meta.json"), "w"
    ) as f:
        json.dump(meta, f, indent=2)

    # Save baseline retrieval JSONL
    baseline_path = os.path.join(ds.retrieval_dir, "baseline.jsonl")
    os.makedirs(ds.retrieval_dir, exist_ok=True)
    best_beir = beir_results_all.get(best_n)
    if not best_beir:
        logger.info("Re-searching best config for baseline.jsonl …")
        bm25 = BM25.load(best_index)
        batch_results = bm25.search(query_texts, k=max(k_values))
        best_beir = {}
        for qid, hits in zip(query_ids, batch_results):
            best_beir[qid] = {doc_ids[idx]: float(score) for idx, score in hits}
    with open(baseline_path, "w") as f:
        for qid in query_ids:
            scored = best_beir.get(qid, {})
            ranked = [
                {"doc_id": did, "score": s, "rank": r}
                for r, (did, s) in enumerate(
                    sorted(scored.items(), key=lambda x: -x[1]), 1
                )
            ]
            f.write(json.dumps({"query_id": qid, "candidates": ranked}) + "\n")
    logger.info("Saved baseline retrieval: %s", baseline_path)


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="eval_bm25",
)
def main(cfg: DictConfig) -> None:
    build_and_evaluate(cfg)


if __name__ == "__main__":
    main()
