# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Rerank BM25 retrieval candidates with an LLM, then evaluate.

Loads top-K candidates from ``retrieval/best.jsonl``, scores each
(query, document) pair with an LLM relevance prompt, reranks by
LLM score, and evaluates Recall@K / NDCG@K.

Usage::

    source sandbox.sh
    python scripts/llm_reranking.py
    python scripts/llm_reranking.py data=fiqa rerank.top_n=50

Config groups::

    global   — db_root
    data/    — dataset name, k_values
    rerank/  — top_n, concurrency, max_tokens, temperature, prompt_file
    sglang/  — LLM server params (model, port, chat_template_kwargs)

Requires::

    python scripts/enrich_query_and_retrieve.py data=<name>
    python scripts/serve_llm.py
"""

import asyncio
import json
import logging
import os
import re
import time

import aiohttp
import hydra
from omegaconf import DictConfig, OmegaConf
from sira.llm import post_chat
from sira.schema.mteb import (
    COL_ID,
    COL_TEXT,
    DatasetDir,
    load_qrels_dict,
    read_corpus_texts,
    read_queries,
)

logger = logging.getLogger(__name__)

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def _ds(cfg: DictConfig) -> DatasetDir:
    return DatasetDir(root=cfg.db_root, name=cfg.data.name)


def _parse_score(raw: str) -> float | None:
    """Extract a numeric score from LLM response."""
    idx = raw.find("{")
    if idx >= 0:
        end = raw.rfind("}")
        if end > idx:
            try:
                obj = json.loads(raw[idx : end + 1])
                if isinstance(obj, dict) and "score" in obj:
                    return float(obj["score"])
            except (json.JSONDecodeError, ValueError):
                pass
    m = re.search(r"(\d+(?:\.\d+)?)", raw)
    if m:
        return float(m.group(1))
    return None


async def run(cfg: DictConfig) -> None:
    ds = _ds(cfg)
    rerank_cfg = cfg.rerank
    sglang_cfg = cfg.sglang

    # Load prompt
    prompt_path = os.path.join(SCRIPTS_DIR, rerank_cfg.prompt_file)
    with open(prompt_path) as f:
        prompt_template = f.read()
    logger.info("Prompt: %s", rerank_cfg.prompt_file)

    chat_template_kwargs = None
    if getattr(sglang_cfg, "chat_template_kwargs", None):
        chat_template_kwargs = OmegaConf.to_container(sglang_cfg.chat_template_kwargs)

    llm_url = f"http://127.0.0.1:{sglang_cfg.port}/v1/chat/completions"
    model = sglang_cfg.model

    # Load retrieval candidates
    retrieval_path = ds.retrieval_best("query-enrich")
    if not os.path.exists(retrieval_path):
        raise SystemExit(
            f"Retrieval results not found at {retrieval_path}. "
            "Run enrich_query_and_retrieve.py first."
        )
    candidates_by_query: dict[str, list[dict]] = {}
    with open(retrieval_path) as f:
        for line in f:
            record = json.loads(line)
            qid = record["query_id"]
            candidates_by_query[qid] = record["candidates"][: rerank_cfg.top_n]
    logger.info(
        "Loaded %d queries, top-%d candidates each",
        len(candidates_by_query),
        rerank_cfg.top_n,
    )

    # Load corpus and queries
    doc_ids, doc_texts = read_corpus_texts(ds.corpus)
    doc_id_to_text = dict(zip(doc_ids, doc_texts))
    del doc_ids, doc_texts

    split = cfg.data.split
    queries_df = read_queries(ds.queries(split))
    query_id_to_text = dict(
        zip(
            queries_df.get_column(COL_ID).to_list(),
            queries_df.get_column(COL_TEXT).to_list(),
        )
    )
    qrels = load_qrels_dict(ds.qrels(split))

    shard_rank = int(cfg.get("shard_rank", 0))
    num_shards = int(cfg.get("num_shards", 1))
    if num_shards > 1:
        all_qids = list(candidates_by_query.keys())
        my_qids = set(all_qids[shard_rank::num_shards])
        candidates_by_query = {
            q: c for q, c in candidates_by_query.items() if q in my_qids
        }
        logger.info(
            "Shard %d/%d: reranking %d/%d queries",
            shard_rank,
            num_shards,
            len(candidates_by_query),
            len(all_qids),
        )

    # Run dir
    run_name = cfg.get("run_name", f"rerank-{int(time.time())}")
    run_dir = os.path.join(ds.rerank_runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger.info("Run dir: %s", run_dir)

    if shard_rank == 0:
        with open(os.path.join(run_dir, "prompt.txt"), "w") as f:
            f.write(prompt_template)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

    # Resume: skip already-scored pairs
    trace_suffix = f".shard{shard_rank}" if num_shards > 1 else ""
    trace_kept_path = os.path.join(run_dir, f"trace.kept{trace_suffix}.jsonl")
    trace_failed_path = os.path.join(run_dir, f"trace.failed{trace_suffix}.jsonl")

    done_pairs: set[tuple[str, str]] = set()
    for path in [trace_kept_path, trace_failed_path]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        done_pairs.add((rec["query_id"], rec["doc_id"]))
                    except (json.JSONDecodeError, KeyError):
                        pass
    if done_pairs:
        logger.info("Resuming: %d pairs already scored", len(done_pairs))

    # Score all (query, doc) pairs
    connector = aiohttp.TCPConnector(limit=rerank_cfg.concurrency)
    timeout = aiohttp.ClientTimeout(total=300.0, sock_read=60.0)
    sem = asyncio.Semaphore(rerank_cfg.concurrency)

    total_pairs = sum(len(cands) for cands in candidates_by_query.values())
    scores: dict[str, dict[str, float]] = {}
    stats = {"total": total_pairs, "scored": 0, "parse_errors": 0, "llm_errors": 0}
    done_count = 0

    trace_kept_file = open(trace_kept_path, "a")
    trace_failed_file = open(trace_failed_path, "a")
    trace_lock = asyncio.Lock()

    async def _write_trace(record: dict, success: bool) -> None:
        f = trace_kept_file if success else trace_failed_file
        async with trace_lock:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def _score_one(
        session: aiohttp.ClientSession,
        qid: str,
        doc_id: str,
        query_text: str,
        doc_text: str,
    ) -> None:
        nonlocal done_count
        async with sem:
            t0 = time.time()
            prompt = prompt_template.format(
                query=query_text, document=doc_text[: rerank_cfg.max_doc_chars]
            )
            payload: dict = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": rerank_cfg.max_tokens,
                "temperature": rerank_cfg.temperature,
            }
            if rerank_cfg.seed is not None:
                payload["seed"] = rerank_cfg.seed
            if chat_template_kwargs:
                payload["chat_template_kwargs"] = chat_template_kwargs

            try:
                data = await post_chat(session, llm_url, payload)
                raw = data["choices"][0]["message"]["content"] or ""
            except Exception as e:
                stats["llm_errors"] += 1
                done_count += 1
                await _write_trace(
                    {
                        "query_id": qid,
                        "doc_id": doc_id,
                        "status": "error",
                        "error": str(e),
                    },
                    success=False,
                )
                return

            ms = int((time.time() - t0) * 1000)
            done_count += 1
            score = _parse_score(raw)

            if score is None:
                stats["parse_errors"] += 1
                await _write_trace(
                    {
                        "query_id": qid,
                        "doc_id": doc_id,
                        "status": "parse_error",
                        "raw_response": raw,
                        "ms": ms,
                    },
                    success=False,
                )
                return

            stats["scored"] += 1
            scores.setdefault(qid, {})[doc_id] = score

            if done_count % 1000 == 0:
                logger.info(
                    "Progress: %d/%d (%.0f%%)",
                    done_count,
                    total_pairs,
                    done_count / total_pairs * 100,
                )

            await _write_trace(
                {
                    "query_id": qid,
                    "doc_id": doc_id,
                    "status": "ok",
                    "score": score,
                    "raw_response": raw,
                    "ms": ms,
                },
                success=True,
            )

    logger.info(
        "Scoring %d (query, doc) pairs (concurrency=%d) …",
        total_pairs,
        rerank_cfg.concurrency,
    )
    t0 = time.time()

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = []
        for qid, cands in candidates_by_query.items():
            query_text = query_id_to_text.get(qid, "")
            for cand in cands:
                doc_id = cand["doc_id"]
                if (qid, doc_id) in done_pairs:
                    continue
                doc_text = doc_id_to_text.get(doc_id, "")
                tasks.append((qid, doc_id, query_text, doc_text))
        for batch_start in range(0, len(tasks), 10_000):
            batch = tasks[batch_start : batch_start + 10_000]
            await asyncio.gather(
                *[_score_one(session, qid, did, qt, dt) for qid, did, qt, dt in batch]
            )

    trace_kept_file.close()
    trace_failed_file.close()
    elapsed = time.time() - t0
    logger.info(
        "Scoring done in %.1fs (%.0f pairs/s): %d scored, %d parse errors, %d llm errors",
        elapsed,
        total_pairs / max(elapsed, 0.01),
        stats["scored"],
        stats["parse_errors"],
        stats["llm_errors"],
    )

    if num_shards > 1:
        shard_suffix = f".shard{shard_rank}"
        with open(os.path.join(run_dir, f"scores{shard_suffix}.jsonl"), "w") as f:
            for qid, doc_scores in scores.items():
                for doc_id, score in doc_scores.items():
                    f.write(
                        json.dumps(
                            {"query_id": qid, "doc_id": doc_id, "score": score},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        logger.info("Shard %d done → %s", shard_rank, run_dir)
        return

    actual_attempted = total_pairs - len(done_pairs)
    if actual_attempted > 0 and stats["scored"] == 0:
        raise SystemExit(
            f"All {actual_attempted} pairs failed (0 scored). Check LLM server."
        )

    # Build reranked results: sort by LLM score desc, BM25 rank asc as tiebreaker.
    # Candidates without an LLM score keep their original BM25 rank at the bottom.
    reranked_results: dict[str, dict[str, float]] = {}
    for qid, cands in candidates_by_query.items():
        qscores = scores.get(qid, {})
        scored = [
            (c["doc_id"], qscores[c["doc_id"]], c["rank"])
            for c in cands
            if c["doc_id"] in qscores
        ]
        unscored = [
            (c["doc_id"], -1e6, c["rank"]) for c in cands if c["doc_id"] not in qscores
        ]
        all_ranked = sorted(scored + unscored, key=lambda x: (-x[1], x[2]))
        reranked_results[qid] = {
            doc_id: float(len(all_ranked) - rank)
            for rank, (doc_id, _, _) in enumerate(all_ranked)
        }

    # Evaluate
    k_values = list(cfg.data.k_values)
    if qrels:
        from beir.retrieval.evaluation import EvaluateRetrieval

        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
            qrels, reranked_results, k_values
        )
        metrics = {**ndcg, **recall, **precision}
        logger.info("Results (reranked):")
        for k in k_values:
            logger.info(
                "  k=%d  NDCG=%.4f  Recall=%.4f",
                k,
                ndcg.get(f"NDCG@{k}", 0),
                recall.get(f"Recall@{k}", 0),
            )
    else:
        metrics = {}
        logger.info("No qrels — skipping evaluation")

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(run_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # Save reranked retrieval results
    with open(os.path.join(run_dir, "reranked.jsonl"), "w") as f:
        for qid, cands in candidates_by_query.items():
            qscores = scores.get(qid, {})
            scored = [
                (c["doc_id"], qscores.get(c["doc_id"]), c["score"]) for c in cands
            ]
            scored.sort(key=lambda x: -(x[1] if x[1] is not None else -1e6))
            record = {
                "query_id": qid,
                "candidates": [
                    {
                        "doc_id": doc_id,
                        "llm_score": llm_s,
                        "bm25_score": bm25_s,
                        "rank": rank,
                    }
                    for rank, (doc_id, llm_s, bm25_s) in enumerate(scored, 1)
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Save eval
    eval_path = ds.eval_rerank(run_name)
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Update best
    selection_metric = cfg.selection_metric
    meta = {
        "stage": "rerank",
        "dataset": ds.name,
        "run_name": run_name,
        "run_dir": run_dir,
        "metrics": metrics,
        "stats": stats,
        "retrieval_source": retrieval_path,
        "rerank_params": {
            "top_n": rerank_cfg.top_n,
            "temperature": rerank_cfg.temperature,
            "seed": rerank_cfg.seed,
            "prompt_file": rerank_cfg.prompt_file,
        },
        "model": model,
        "num_queries": len(candidates_by_query),
        "total_pairs": total_pairs,
        "elapsed_s": round(elapsed, 1),
        "timestamp": int(time.time()),
    }
    ds.update_best(
        best_links=[ds.eval_rerank_best],
        target_name=f"{run_name}.json",
        meta=meta,
        selection_metrics=None,
    )
    logger.info(
        "Best → %s (%s=%.4f). No held-out split at this stage, so the latest "
        "run wins; runs are not ranked against each other on the eval split.",
        run_name,
        selection_metric,
        metrics.get(selection_metric, 0),
    )
    logger.info("Done → %s", run_dir)


def merge_rerank_shards(cfg: DictConfig) -> None:
    """Merge shard scores, then rerank + eval + best-update."""
    ds = _ds(cfg)
    rerank_cfg = cfg.rerank
    sglang_cfg = cfg.sglang
    num_shards = int(cfg.get("num_shards", 1))
    run_name = cfg.get("run_name", "")
    run_dir = os.path.join(ds.rerank_runs_dir, run_name)

    # Load retrieval candidates
    retrieval_path = ds.retrieval_best("query-enrich")
    candidates_by_query: dict[str, list[dict]] = {}
    with open(retrieval_path) as f:
        for line in f:
            record = json.loads(line)
            qid = record["query_id"]
            candidates_by_query[qid] = record["candidates"][: rerank_cfg.top_n]

    # Merge scores from all shards (JSONL)
    scores: dict[str, dict[str, float]] = {}
    for i in range(num_shards):
        path = os.path.join(run_dir, f"scores.shard{i}.jsonl")
        if not os.path.exists(path):
            logger.warning("Missing rerank shard %d: %s", i, path)
            continue
        shard_count = 0
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                scores.setdefault(rec["query_id"], {})[rec["doc_id"]] = rec["score"]
                shard_count += 1
        logger.info("Loaded rerank shard %d: %d scores", i, shard_count)

    total_scored = sum(len(v) for v in scores.values())
    logger.info(
        "Merged %d scores across %d queries from %d shards",
        total_scored,
        len(scores),
        num_shards,
    )

    # Build reranked results: sort by LLM score desc, BM25 rank asc as tiebreaker.
    reranked_results: dict[str, dict[str, float]] = {}
    for qid, cands in candidates_by_query.items():
        qscores = scores.get(qid, {})
        scored = [
            (c["doc_id"], qscores[c["doc_id"]], c["rank"])
            for c in cands
            if c["doc_id"] in qscores
        ]
        unscored = [
            (c["doc_id"], -1e6, c["rank"]) for c in cands if c["doc_id"] not in qscores
        ]
        all_ranked = sorted(scored + unscored, key=lambda x: (-x[1], x[2]))
        reranked_results[qid] = {
            doc_id: float(len(all_ranked) - rank)
            for rank, (doc_id, _, _) in enumerate(all_ranked)
        }

    # Evaluate
    split = cfg.data.split
    qrels = load_qrels_dict(ds.qrels(split))
    k_values = list(cfg.data.k_values)
    if qrels:
        from beir.retrieval.evaluation import EvaluateRetrieval

        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
            qrels, reranked_results, k_values
        )
        metrics = {**ndcg, **recall, **precision}
        logger.info("Results (reranked):")
        for k in k_values:
            logger.info(
                "  k=%d  NDCG=%.4f  Recall=%.4f",
                k,
                ndcg.get(f"NDCG@{k}", 0),
                recall.get(f"Recall@{k}", 0),
            )
    else:
        metrics = {}
        logger.info("No qrels — skipping evaluation")

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save reranked results
    with open(os.path.join(run_dir, "reranked.jsonl"), "w") as f:
        for qid, cands in candidates_by_query.items():
            qscores = scores.get(qid, {})
            scored_list = [
                (c["doc_id"], qscores.get(c["doc_id"]), c["score"]) for c in cands
            ]
            scored_list.sort(key=lambda x: -(x[1] if x[1] is not None else -1e6))
            record = {
                "query_id": qid,
                "candidates": [
                    {
                        "doc_id": doc_id,
                        "llm_score": llm_s,
                        "bm25_score": bm25_s,
                        "rank": rank,
                    }
                    for rank, (doc_id, llm_s, bm25_s) in enumerate(scored_list, 1)
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Save eval + update best
    eval_path = ds.eval_rerank(run_name)
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)

    selection_metric = cfg.selection_metric
    meta = {
        "stage": "rerank",
        "dataset": ds.name,
        "run_name": run_name,
        "run_dir": run_dir,
        "metrics": metrics,
        "rerank_params": {
            "top_n": rerank_cfg.top_n,
            "temperature": rerank_cfg.temperature,
            "seed": rerank_cfg.seed,
            "prompt_file": rerank_cfg.prompt_file,
        },
        "model": sglang_cfg.model,
        "num_queries": len(candidates_by_query),
        "total_scored": total_scored,
        "num_shards": num_shards,
        "timestamp": int(time.time()),
    }
    ds.update_best(
        best_links=[ds.eval_rerank_best],
        target_name=f"{run_name}.json",
        meta=meta,
        selection_metrics=None,
    )
    logger.info(
        "Best → %s (%s=%.4f). No held-out split at this stage, so the latest "
        "run wins; runs are not ranked against each other on the eval split.",
        run_name,
        selection_metric,
        metrics.get(selection_metric, 0),
    )
    logger.info("Rerank merge done → %s", run_dir)


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="llm_reranking",
)
def main(cfg: DictConfig) -> None:
    if cfg.get("merge_shards", False):
        merge_rerank_shards(cfg)
    else:
        asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
