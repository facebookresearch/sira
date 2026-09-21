# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Iterative agentic retrieval (ReAct-style keyword search loop).

For each test query, the LLM iteratively generates keyword search patterns,
performs exact substring matching (like rga/pdfgrep), reviews results,
and refines its search strategy over multiple rounds.

Usage::

    source sandbox.sh
    python scripts/iterative_retrieve.py
    python scripts/iterative_retrieve.py data=fiqa

Config groups::

    global     — db_root
    data/      — dataset name, k_values
    iterative/ — max_iterations, top_k_per_iter, concurrency, prompt
    sglang/    — LLM server params (model, port, chat_template_kwargs)

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


def _parse_action(raw: str) -> dict:
    """Extract action JSON from LLM response.

    Returns {"action": "search", "query": "..."} or {"action": "done"}.
    Falls back to {"action": "error"} on parse failure.
    """
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
            if isinstance(obj, dict) and "action" in obj:
                return obj
        except json.JSONDecodeError:
            pass
    raw_lower = raw.lower()
    if '"done"' in raw_lower or "'done'" in raw_lower:
        return {"action": "done"}
    if '"search"' in raw_lower or "'search'" in raw_lower:
        import re
        m = re.search(r'["\']query["\']\s*:\s*["\'](.+?)["\']', raw)
        if m:
            return {"action": "search", "query": m.group(1)}
    return {"action": "error", "raw": raw}



def _format_search_results(
    hits: list[tuple[int, int]],
    doc_ids: list[str],
    corpus_texts: list[str],
    snippet_chars: int,
    top_k: int,
) -> str:
    """Format grep results with context around the match point (like grep -C 5)."""
    shown = hits[:top_k]
    lines = [f"Search returned {len(shown)} results:\n"]
    half = snippet_chars // 2
    for rank, (idx, match_pos) in enumerate(shown, 1):
        did = doc_ids[idx]
        text = corpus_texts[idx]
        start = max(0, match_pos - half)
        end = min(len(text), match_pos + half)
        snippet = text[start:end]
        if start > 0:
            snippet = "…" + snippet
        if end < len(text):
            snippet = snippet + "…"
        lines.append(f"[{rank}] (id={did}) {snippet}\n")
    return "\n".join(lines)


def _score_documents(
    doc_records: dict[int, dict],
    scoring: str,
) -> list[tuple[int, float]]:
    """Score accumulated documents using the chosen strategy.

    doc_records maps doc_index -> {"best_rank": int, "count": int, "ranks": [int, ...]}.
    """
    scored: list[tuple[int, float]] = []
    for doc_idx, info in doc_records.items():
        if scoring == "frequency":
            score = float(info["count"])
        elif scoring == "rrf":
            score = sum(1.0 / (60 + r) for r in info["ranks"])
        else:  # best_rank
            score = 1.0 / (1 + info["best_rank"])
        scored.append((doc_idx, score))
    return sorted(scored, key=lambda x: -x[1])


async def run(cfg: DictConfig) -> None:
    ds = _ds(cfg)
    iter_cfg = cfg.iterative
    sglang_cfg = cfg.sglang

    prompt_path = os.path.join(SCRIPTS_DIR, iter_cfg.prompt_file)
    with open(prompt_path) as f:
        system_prompt = f.read()
    system_prompt = system_prompt.format(max_iterations=iter_cfg.max_iterations)
    logger.info("Prompt: %s", iter_cfg.prompt_file)

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
    run_name = cfg.get("run_name", f"iterative-{int(time.time())}")
    run_dir = os.path.join(ds.iterative_runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger.info("Run dir: %s", run_dir)

    if shard_rank == 0:
        with open(os.path.join(run_dir, "prompt.txt"), "w") as f:
            f.write(system_prompt)
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

    # Config values
    max_iterations = iter_cfg.max_iterations
    top_k_per_iter = iter_cfg.top_k_per_iter
    snippet_chars = iter_cfg.snippet_chars
    scoring = iter_cfg.scoring
    final_top_k = iter_cfg.final_top_k

    # Async setup
    connector = aiohttp.TCPConnector(limit=iter_cfg.concurrency)
    per_iter_timeout = 120.0
    timeout = aiohttp.ClientTimeout(
        total=per_iter_timeout * max_iterations * 2,
        sock_read=per_iter_timeout,
    )
    sem = asyncio.Semaphore(iter_cfg.concurrency)

    all_scored: dict[str, list[tuple[int, float]]] = {}
    stats = {
        "total": 0, "ok": 0, "errors": 0,
        "total_iterations": 0, "total_searches": 0,
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

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Find documents relevant to: {qtext}"},
            ]

            doc_records: dict[int, dict] = {}
            search_queries_used: list[str] = []
            iterations_done = 0

            try:
                for iteration in range(max_iterations):
                    iterations_done += 1
                    payload: dict = {
                        "model": model,
                        "messages": messages,
                        "max_tokens": iter_cfg.max_tokens,
                        "temperature": iter_cfg.temperature,
                    }
                    if iter_cfg.seed is not None:
                        payload["seed"] = iter_cfg.seed
                    if chat_template_kwargs:
                        payload["chat_template_kwargs"] = chat_template_kwargs

                    data = await post_chat(session, llm_url, payload)
                    raw = data["choices"][0]["message"]["content"] or ""
                    action = _parse_action(raw)

                    messages.append({"role": "assistant", "content": raw})

                    if action["action"] == "done":
                        break
                    elif action["action"] == "search" and "query" in action:
                        search_query = action["query"]
                        search_queries_used.append(search_query)
                        stats["total_searches"] += 1

                        hits = await asyncio.to_thread(
                            grep_idx.search_or,
                            search_query,
                            top_k_per_iter * 5,
                        )

                        for rank, (idx, _pos) in enumerate(hits[:top_k_per_iter], 1):
                            if idx not in doc_records:
                                doc_records[idx] = {
                                    "best_rank": rank,
                                    "count": 0,
                                    "ranks": [],
                                }
                            doc_records[idx]["best_rank"] = min(
                                doc_records[idx]["best_rank"], rank
                            )
                            doc_records[idx]["count"] += 1
                            doc_records[idx]["ranks"].append(rank)

                        result_text = _format_search_results(
                            hits, doc_ids, corpus_texts,
                            snippet_chars, top_k_per_iter,
                        )
                        remaining = max_iterations - iteration - 1
                        if remaining > 0:
                            result_text += (
                                f"\nYou have {remaining} search(es) remaining. "
                                "Search again with different terms or respond with "
                                '{"action": "done"} if you have found enough.'
                            )
                        else:
                            result_text += "\nNo more searches remaining."
                        messages.append({"role": "user", "content": result_text})
                    else:
                        break

            except Exception as e:
                logger.warning("Error for query %s: %s", qid, e)
                stats["errors"] += 1
                await _write_trace(
                    {
                        "query_id": qid,
                        "status": "error",
                        "error": str(e),
                        "search_queries": search_queries_used,
                        "iterations": iterations_done,
                        "ms": int((time.time() - t0) * 1000),
                    },
                    success=False,
                )
                return

            stats["total_iterations"] += iterations_done

            if not doc_records:
                stats["errors"] += 1
                await _write_trace(
                    {
                        "query_id": qid,
                        "status": "no_results",
                        "search_queries": search_queries_used,
                        "iterations": iterations_done,
                        "ms": int((time.time() - t0) * 1000),
                    },
                    success=False,
                )
                return

            scored = _score_documents(doc_records, scoring)[:final_top_k]
            all_scored[qid] = scored
            stats["ok"] += 1

            ms = int((time.time() - t0) * 1000)
            await _write_trace(
                {
                    "query_id": qid,
                    "status": "ok",
                    "original_query": qtext,
                    "search_queries": search_queries_used,
                    "iterations": iterations_done,
                    "num_docs_found": len(doc_records),
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
        "Iterative retrieval done in %.1fs: %d ok / %d errors / "
        "%d total iterations / %d total searches",
        elapsed, stats["ok"], stats["errors"],
        stats["total_iterations"], stats["total_searches"],
    )

    if num_shards > 1:
        shard_path = os.path.join(run_dir, f"fused_results.shard{shard_rank}.jsonl")
        with open(shard_path, "w") as f:
            for qid, scored in all_scored.items():
                record = {
                    "query_id": qid,
                    "candidates": [
                        {
                            "doc_id": doc_ids[idx],
                            "score": round(score, 6),
                            "rank": rank,
                        }
                        for rank, (idx, score) in enumerate(scored, 1)
                    ],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Shard %d done → %s", shard_rank, run_dir)
        return

    # Queries with no results get empty candidate lists
    for qid in query_ids:
        if qid not in all_scored:
            all_scored[qid] = []

    # Build BEIR results
    beir_results: dict[str, dict[str, float]] = {}
    for qid in query_ids:
        scored = all_scored.get(qid, [])
        beir_results[qid] = {doc_ids[idx]: score for idx, score in scored}

    # Save retrieval results
    retrieval_path = ds.retrieval_results(run_name, stage="iterative")
    os.makedirs(os.path.dirname(retrieval_path), exist_ok=True)
    with open(retrieval_path, "w") as f:
        for qid in query_ids:
            scored = all_scored.get(qid, [])
            record = {
                "query_id": qid,
                "candidates": [
                    {
                        "doc_id": doc_ids[idx],
                        "score": round(score, 6),
                        "rank": rank,
                    }
                    for rank, (idx, score) in enumerate(scored, 1)
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
        logger.info("Iterative retrieval results:")
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
    eval_path = ds.eval_iterative(run_name)
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)

    selection_metric = cfg.selection_metric
    meta = {
        "stage": "iterative",
        "dataset": ds.name,
        "run_name": run_name,
        "run_dir": run_dir,
        "metrics": metrics,
        "stats": stats,
        "iterative_params": {
            "max_iterations": max_iterations,
            "top_k_per_iter": top_k_per_iter,
            "snippet_chars": snippet_chars,
            "scoring": scoring,
            "final_top_k": final_top_k,
            "temperature": iter_cfg.temperature,
            "seed": iter_cfg.seed,
        },
        "prompt_file": iter_cfg.prompt_file,
        "model": model,
        "num_queries": len(query_ids),
        "elapsed_s": round(elapsed, 1),
        "timestamp": int(time.time()),
    }
    ds.update_best(
        best_links=[ds.eval_iterative_best],
        target_name=f"{run_name}.json",
        meta=meta,
        selection_metrics=None,
    )
    ds.update_best(
        best_links=[ds.retrieval_best("iterative")],
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


def merge_iterative_shards(cfg: DictConfig) -> None:
    """Merge shard results, then evaluate + best-update."""
    ds = _ds(cfg)
    sglang_cfg = cfg.sglang
    num_shards = int(cfg.get("num_shards", 1))
    run_name = cfg.get("run_name", "")
    run_dir = os.path.join(ds.iterative_runs_dir, run_name)

    split = cfg.data.split
    queries_df = read_queries(ds.queries(split))
    query_ids = queries_df.get_column(COL_ID).to_list()
    qrels = load_qrels_dict(ds.qrels(split))
    doc_ids, _ = read_corpus_texts(ds.corpus)

    # Merge shard retrieval files
    all_results: dict[str, list[dict]] = {}
    for i in range(num_shards):
        shard_path = os.path.join(run_dir, f"fused_results.shard{i}.jsonl")
        if os.path.exists(shard_path):
            with open(shard_path) as f:
                for line in f:
                    rec = json.loads(line)
                    all_results[rec["query_id"]] = rec["candidates"]
        logger.info("Loaded iterative shard %d", i)

    logger.info("Merged %d query results from %d shards", len(all_results), num_shards)

    beir_results: dict[str, dict[str, float]] = {}
    for qid in query_ids:
        candidates = all_results.get(qid, [])
        beir_results[qid] = {c["doc_id"]: c["score"] for c in candidates}

    retrieval_path = ds.retrieval_results(run_name, stage="iterative")
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
        logger.info("Iterative results (merged):")
        for k in k_values:
            logger.info(
                "  k=%d  NDCG=%.4f  Recall=%.4f",
                k, ndcg.get(f"NDCG@{k}", 0), recall.get(f"Recall@{k}", 0),
            )
    else:
        metrics = {}

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    eval_path = ds.eval_iterative(run_name)
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)

    selection_metric = cfg.selection_metric
    meta = {
        "stage": "iterative",
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
        best_links=[ds.eval_iterative_best],
        target_name=f"{run_name}.json",
        meta=meta,
        selection_metrics=None,
    )
    ds.update_best(
        best_links=[ds.retrieval_best("iterative")],
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
    logger.info("Iterative merge done → %s", run_dir)


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="iterative_retrieve",
)
def main(cfg: DictConfig) -> None:
    if cfg.get("merge_shards", False):
        merge_iterative_shards(cfg)
    else:
        asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
