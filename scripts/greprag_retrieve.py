# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""GrepRAG-style retrieval: LLM generates grep patterns, exact match, Jaccard rank.

Faithful to the GrepRAG paper (Naive GrepRAG):
1. LLM generates N grep patterns (keywords, phrases) for each query
2. Exact case-insensitive substring match against the corpus (like ripgrep)
3. All grep hits pooled together (no per-pattern ranking or RRF)
4. Pool ranked once by token-level Jaccard similarity against the original query

Usage::

    source sandbox.sh
    python scripts/greprag_retrieve.py
    python scripts/greprag_retrieve.py data=fiqa

Config groups::

    global   — db_root
    data/    — dataset name, k_values
    greprag/ — num_patterns, rrf_k, concurrency, prompt
    sglang/  — LLM server params (model, port, chat_template_kwargs)

Requires::

    python scripts/prepare_mteb_data.py data=<name>
    python scripts/eval_bm25.py data=<name>
    python scripts/serve_llm.py
"""

import asyncio
import json
import logging
import os
import time

import aiohttp
import hydra
from omegaconf import DictConfig, OmegaConf
from sira.grep import GrepIndex
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


def _parse_patterns(raw: str) -> list[str]:
    """Extract grep patterns from LLM response."""
    for key in ("patterns", "queries", "keywords"):
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(raw[start : end + 1])
                if isinstance(obj, dict) and key in obj:
                    return [p for p in obj[key] if isinstance(p, str) and p.strip()]
            except json.JSONDecodeError:
                pass
    start = raw.find("[")
    end = raw.rfind("]")
    if start >= 0 and end > start:
        try:
            arr = json.loads(raw[start : end + 1])
            if isinstance(arr, list):
                return [p for p in arr if isinstance(p, str) and p.strip()]
        except json.JSONDecodeError:
            pass
    return []


def jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def tokenize_simple(text: str) -> set[str]:
    """Lowercase whitespace tokenization for Jaccard."""
    return set(text.lower().split())



def rank_by_jaccard(
    matched_indices: set[int],
    query_tokens: set[str],
    corpus_texts: list[str],
) -> list[tuple[int, float]]:
    """Rank matched documents by Jaccard similarity against the query."""
    scored = []
    for idx in matched_indices:
        doc_tokens = tokenize_simple(corpus_texts[idx])
        score = jaccard_similarity(query_tokens, doc_tokens)
        scored.append((idx, score))
    scored.sort(key=lambda x: -x[1])
    return scored


async def run(cfg: DictConfig) -> None:
    ds = _ds(cfg)
    greprag_cfg = cfg.greprag
    sglang_cfg = cfg.sglang

    prompt_path = os.path.join(SCRIPTS_DIR, greprag_cfg.prompt_file)
    with open(prompt_path) as f:
        prompt_template = f.read()
    logger.info("Prompt: %s", greprag_cfg.prompt_file)

    chat_template_kwargs = None
    if getattr(sglang_cfg, "chat_template_kwargs", None):
        chat_template_kwargs = OmegaConf.to_container(sglang_cfg.chat_template_kwargs)

    llm_url = f"http://127.0.0.1:{sglang_cfg.port}/v1/chat/completions"
    model = sglang_cfg.model

    # Load corpus + build grep index
    doc_ids, corpus_texts = read_corpus_texts(ds.corpus)
    grep_idx = GrepIndex(corpus_texts)

    # Load eval-split queries + qrels
    split = cfg.data.split
    queries_df = read_queries(ds.queries(split))
    query_ids = queries_df.get_column(COL_ID).to_list()
    query_texts = queries_df.get_column(COL_TEXT).to_list()
    qrels = load_qrels_dict(ds.qrels(split))

    shard_rank = int(cfg.get("shard_rank", 0))
    num_shards = int(cfg.get("num_shards", 1))
    if num_shards > 1:
        shard_qids = set(query_ids[shard_rank::num_shards])
        logger.info(
            "Shard %d/%d: processing %d/%d queries",
            shard_rank, num_shards, len(shard_qids), len(query_ids),
        )

    # Run dir
    run_name = cfg.get("run_name", f"greprag-{int(time.time())}")
    run_dir = os.path.join(ds.greprag_runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger.info("Run dir: %s", run_dir)

    if shard_rank == 0:
        with open(os.path.join(run_dir, "prompt.txt"), "w") as f:
            f.write(prompt_template)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

    # Resume
    trace_suffix = f".shard{shard_rank}" if num_shards > 1 else ""
    trace_kept_path = os.path.join(run_dir, f"trace.kept{trace_suffix}.jsonl")
    trace_failed_path = os.path.join(run_dir, f"trace.failed{trace_suffix}.jsonl")

    done_qids: set[str] = set()
    for path in [trace_kept_path, trace_failed_path]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        done_qids.add(json.loads(line)["query_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    if done_qids:
        logger.info("Resuming: %d queries already done", len(done_qids))

    # Async setup
    connector = aiohttp.TCPConnector(limit=greprag_cfg.concurrency)
    timeout = aiohttp.ClientTimeout(total=300.0, sock_read=60.0)
    sem = asyncio.Semaphore(greprag_cfg.concurrency)

    num_patterns = greprag_cfg.num_sub_queries
    top_k = greprag_cfg.top_k_per_query
    max_grep_matches = top_k * 5

    all_fused: dict[str, list[tuple[int, float]]] = {}
    stats = {
        "total": 0, "ok": 0, "errors": 0,
        "total_patterns": 0, "total_grep_hits": 0,
    }
    trace_kept_file = open(trace_kept_path, "a")
    trace_failed_file = open(trace_failed_path, "a")
    trace_lock = asyncio.Lock()

    async def _write_trace(record: dict, success: bool) -> None:
        f = trace_kept_file if success else trace_failed_file
        async with trace_lock:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def _process_one(
        session: aiohttp.ClientSession, qid: str, qtext: str
    ) -> None:
        async with sem:
            t0 = time.time()
            stats["total"] += 1
            try:
                prompt = prompt_template.format(
                    query=qtext, num_sub_queries=num_patterns
                )
                payload: dict = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": greprag_cfg.max_tokens,
                    "temperature": greprag_cfg.temperature,
                }
                if greprag_cfg.seed is not None:
                    payload["seed"] = greprag_cfg.seed
                if chat_template_kwargs:
                    payload["chat_template_kwargs"] = chat_template_kwargs

                data = await post_chat(session, llm_url, payload)
                raw = data["choices"][0]["message"]["content"] or ""
                patterns = _parse_patterns(raw)
            except Exception as e:
                logger.warning("LLM error for query %s: %s", qid, e)
                stats["errors"] += 1
                await _write_trace(
                    {
                        "query_id": qid,
                        "status": "error",
                        "error": str(e),
                        "ms": int((time.time() - t0) * 1000),
                    },
                    success=False,
                )
                return

            if not patterns:
                stats["errors"] += 1
                await _write_trace(
                    {
                        "query_id": qid,
                        "status": "no_patterns",
                        "raw_response": raw,
                        "ms": int((time.time() - t0) * 1000),
                    },
                    success=False,
                )
                return

            stats["total_patterns"] += len(patterns)

            # Step 1: Grep — exact substring match, pool all results
            # Run in thread pool to avoid blocking the event loop
            all_matched = await asyncio.to_thread(
                grep_idx.search_multi, patterns, max_grep_matches
            )
            stats["total_grep_hits"] += len(all_matched)

            if not all_matched:
                stats["ok"] += 1
                await _write_trace(
                    {
                        "query_id": qid,
                        "status": "ok_no_grep_hits",
                        "original_query": qtext,
                        "patterns": patterns,
                        "grep_hits": 0,
                        "raw_response": raw,
                        "ms": int((time.time() - t0) * 1000),
                    },
                    success=True,
                )
                return

            # Step 2: Rank pooled results by token-level Jaccard similarity
            query_tokens = tokenize_simple(qtext)
            ranked = await asyncio.to_thread(
                rank_by_jaccard, all_matched, query_tokens, corpus_texts
            )
            all_fused[qid] = ranked[:top_k]

            ms = int((time.time() - t0) * 1000)
            stats["ok"] += 1
            await _write_trace(
                {
                    "query_id": qid,
                    "status": "ok",
                    "original_query": qtext,
                    "patterns": patterns,
                    "grep_hits": len(all_matched),
                    "num_ranked_docs": len(ranked),
                    "raw_response": raw,
                    "ms": ms,
                },
                success=True,
            )

    # Build work items
    work_items = [
        (qid, qtext)
        for qid, qtext in zip(query_ids, query_texts)
        if (num_shards <= 1 or qid in shard_qids) and qid not in done_qids
    ]
    logger.info("Processing %d queries …", len(work_items))
    t0 = time.time()

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for batch_start in range(0, len(work_items), 10_000):
            batch = work_items[batch_start : batch_start + 10_000]
            await asyncio.gather(
                *[_process_one(session, qid, qtext) for qid, qtext in batch]
            )

    trace_kept_file.close()
    trace_failed_file.close()
    elapsed = time.time() - t0
    logger.info(
        "GrepRAG done in %.1fs: %d ok / %d errors / %d patterns / %d grep hits",
        elapsed, stats["ok"], stats["errors"],
        stats["total_patterns"], stats["total_grep_hits"],
    )

    if num_shards > 1:
        shard_path = os.path.join(run_dir, f"fused_results.shard{shard_rank}.jsonl")
        with open(shard_path, "w") as f:
            for qid, fused in all_fused.items():
                record = {
                    "query_id": qid,
                    "candidates": [
                        {
                            "doc_id": doc_ids[idx],
                            "score": round(score, 6),
                            "rank": rank,
                        }
                        for rank, (idx, score) in enumerate(fused, 1)
                    ],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Shard %d done → %s", shard_rank, run_dir)
        return

    # Queries with no results get empty candidate lists
    for qid in query_ids:
        if qid not in all_fused:
            all_fused[qid] = []

    # Build BEIR results
    beir_results: dict[str, dict[str, float]] = {}
    for qid in query_ids:
        fused = all_fused.get(qid, [])
        beir_results[qid] = {doc_ids[idx]: score for idx, score in fused}

    # Save retrieval results
    retrieval_path = ds.retrieval_results(run_name, stage="greprag")
    os.makedirs(os.path.dirname(retrieval_path), exist_ok=True)
    with open(retrieval_path, "w") as f:
        for qid in query_ids:
            fused = all_fused.get(qid, [])
            record = {
                "query_id": qid,
                "candidates": [
                    {
                        "doc_id": doc_ids[idx],
                        "score": round(score, 6),
                        "rank": rank,
                    }
                    for rank, (idx, score) in enumerate(fused, 1)
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Saved %d retrieval results to %s", len(query_ids), retrieval_path)

    # Evaluate
    k_values = list(cfg.data.k_values)
    if qrels:
        from beir.retrieval.evaluation import EvaluateRetrieval

        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
            qrels, beir_results, k_values
        )
        metrics = {**ndcg, **recall, **precision}
        logger.info("GrepRAG results:")
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

    # Save eval + update best
    eval_path = ds.eval_greprag(run_name)
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)

    selection_metric = cfg.selection_metric
    meta = {
        "stage": "greprag",
        "dataset": ds.name,
        "run_name": run_name,
        "run_dir": run_dir,
        "metrics": metrics,
        "stats": stats,
        "greprag_params": {
            "num_patterns": num_patterns,
            "top_k_per_query": top_k,
            "temperature": greprag_cfg.temperature,
            "seed": greprag_cfg.seed,
        },
        "prompt_file": greprag_cfg.prompt_file,
        "model": model,
        "num_queries": len(query_ids),
        "elapsed_s": round(elapsed, 1),
        "timestamp": int(time.time()),
    }
    ds.update_best(
        best_links=[ds.eval_greprag_best],
        target_name=f"{run_name}.json",
        meta=meta,
        selection_metrics=None,
    )
    ds.update_best(
        best_links=[ds.retrieval_best("greprag")],
        target_name=f"{run_name}.jsonl",
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


def merge_greprag_shards(cfg: DictConfig) -> None:
    """Merge shard results, then evaluate + best-update."""
    ds = _ds(cfg)
    sglang_cfg = cfg.sglang
    num_shards = int(cfg.get("num_shards", 1))
    run_name = cfg.get("run_name", "")
    run_dir = os.path.join(ds.greprag_runs_dir, run_name)

    split = cfg.data.split
    queries_df = read_queries(ds.queries(split))
    query_ids = queries_df.get_column(COL_ID).to_list()
    qrels = load_qrels_dict(ds.qrels(split))
    doc_ids, _ = read_corpus_texts(ds.corpus)

    all_results: dict[str, list[dict]] = {}
    for i in range(num_shards):
        shard_path = os.path.join(run_dir, f"fused_results.shard{i}.jsonl")
        if os.path.exists(shard_path):
            with open(shard_path) as f:
                for line in f:
                    rec = json.loads(line)
                    all_results[rec["query_id"]] = rec["candidates"]
        logger.info("Loaded greprag shard %d", i)

    logger.info("Merged %d query results from %d shards", len(all_results), num_shards)

    beir_results: dict[str, dict[str, float]] = {}
    for qid in query_ids:
        candidates = all_results.get(qid, [])
        beir_results[qid] = {c["doc_id"]: c["score"] for c in candidates}

    retrieval_path = ds.retrieval_results(run_name, stage="greprag")
    os.makedirs(os.path.dirname(retrieval_path), exist_ok=True)
    with open(retrieval_path, "w") as f:
        for qid in query_ids:
            candidates = all_results.get(qid, [])
            record = {"query_id": qid, "candidates": candidates}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Saved %d retrieval results to %s", len(query_ids), retrieval_path)

    k_values = list(cfg.data.k_values)
    if qrels:
        from beir.retrieval.evaluation import EvaluateRetrieval

        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
            qrels, beir_results, k_values
        )
        metrics = {**ndcg, **recall, **precision}
        logger.info("GrepRAG results (merged):")
        for k in k_values:
            logger.info(
                "  k=%d  NDCG=%.4f  Recall=%.4f",
                k, ndcg.get(f"NDCG@{k}", 0), recall.get(f"Recall@{k}", 0),
            )
    else:
        metrics = {}

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    eval_path = ds.eval_greprag(run_name)
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)

    selection_metric = cfg.selection_metric
    meta = {
        "stage": "greprag",
        "dataset": ds.name,
        "run_name": run_name,
        "run_dir": run_dir,
        "metrics": metrics,
        "model": sglang_cfg.model,
        "num_queries": len(query_ids),
        "num_shards": num_shards,
        "timestamp": int(time.time()),
    }
    ds.update_best(
        best_links=[ds.eval_greprag_best],
        target_name=f"{run_name}.json",
        meta=meta,
        selection_metrics=None,
    )
    ds.update_best(
        best_links=[ds.retrieval_best("greprag")],
        target_name=f"{run_name}.jsonl",
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
    logger.info("GrepRAG merge done → %s", run_dir)


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="greprag_retrieve",
)
def main(cfg: DictConfig) -> None:
    if cfg.get("merge_shards", False):
        merge_greprag_shards(cfg)
    else:
        asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
