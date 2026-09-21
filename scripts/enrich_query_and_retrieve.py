# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Expand eval-split queries with LLM-proposed phrases, then retrieve and evaluate.

Loads the best BM25 index + best doc enrichments, then for each test
query asks the LLM for expansion phrases. The expanded query (original +
kept phrases) is used for BM25 retrieval.

Usage::

    source sandbox.sh
    python scripts/enrich_query_and_retrieve.py
    python scripts/enrich_query_and_retrieve.py data=fiqa

Config groups::

    global   — db_root
    data/    — dataset name, k_values
    enrich/  — concurrency, max_tokens, temperature, seed, max_df_ratio
    sglang/  — LLM server params (model, port, chat_template_kwargs)

Requires::

    python scripts/prepare_mteb_data.py data=<name>
    python scripts/eval_bm25.py data=<name>
    python scripts/add_doc_index_adapter.py data=<name>
    python scripts/serve_llm.py
"""

import asyncio
import json
import logging
import os
import time

import aiohttp
import hydra
from bm25x import BM25
from omegaconf import DictConfig, OmegaConf
from sira.llm import parse_phrases, post_chat
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


async def _expand_query(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    query_text: str,
    max_tokens: int,
    temperature: float,
    prompt_template: str,
    max_n: int,
    seed: int | None = None,
    chat_template_kwargs: dict | None = None,
) -> tuple[str, list[str]]:
    """Returns (raw_response, parsed_phrases)."""
    prompt = prompt_template.format(doc_text=query_text, max_n=max_n)
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if seed is not None:
        payload["seed"] = seed
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs

    data = await post_chat(session, url, payload)
    raw = data["choices"][0]["message"]["content"] or ""
    return raw, parse_phrases(raw)


async def run(cfg: DictConfig) -> None:
    ds = _ds(cfg)
    enrich_cfg = cfg.enrich
    sglang_cfg = cfg.sglang

    # Load query prompt
    query_prompt_path = os.path.join(SCRIPTS_DIR, enrich_cfg.query_prompt_file)
    with open(query_prompt_path) as f:
        query_prompt = f.read()
    logger.info("Query prompt: %s", enrich_cfg.query_prompt_file)

    chat_template_kwargs = None
    if getattr(sglang_cfg, "chat_template_kwargs", None):
        chat_template_kwargs = OmegaConf.to_container(sglang_cfg.chat_template_kwargs)

    llm_url = f"http://127.0.0.1:{sglang_cfg.port}/v1/chat/completions"
    model = sglang_cfg.model

    # Load BM25 index + apply doc enrichments
    index_dir = ds.bm25_index_best
    if not os.path.exists(index_dir):
        raise SystemExit(f"BM25 index not found at {index_dir}.")
    bm25 = BM25.load(index_dir)
    bm25.disable_auto_save()
    doc_ids, texts = read_corpus_texts(ds.corpus)
    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}
    logger.info("Loaded BM25 index: %d docs", len(bm25))

    max_df = (
        int(len(doc_ids) * enrich_cfg.max_df_ratio)
        if enrich_cfg.max_df_ratio > 0
        else 0
    )
    if max_df > 0:
        logger.info(
            "Unigram df filter: max_df=%d (%.1f%% of %d docs)",
            max_df,
            enrich_cfg.max_df_ratio * 100,
            len(doc_ids),
        )
    del texts

    shard_rank = int(cfg.get("shard_rank", 0))
    num_shards = int(cfg.get("num_shards", 1))

    if num_shards <= 1:
        if os.path.exists(ds.doc_enrichments_best):
            items = []
            with open(ds.doc_enrichments_best) as f:
                for line in f:
                    rec = json.loads(line)
                    idx = doc_id_to_idx.get(rec["doc_id"])
                    if idx is not None:
                        items.append((idx, rec["phrases"]))
            bm25.enrich_batch(items)
            logger.info("Applied %d doc enrichments (batch)", len(items))
        else:
            logger.info("No doc enrichments found, using base index")
    else:
        logger.info("Shard mode: skipping doc enrichments (applied during merge)")

    # Load eval-split queries + qrels
    split = cfg.data.split
    queries_df = read_queries(ds.queries(split))
    query_ids = queries_df.get_column(COL_ID).to_list()
    query_texts = queries_df.get_column(COL_TEXT).to_list()
    qrels = load_qrels_dict(ds.qrels(split))

    if num_shards > 1:
        shard_qids = set(query_ids[shard_rank::num_shards])
        logger.info(
            "Shard %d/%d: expanding %d/%d queries",
            shard_rank,
            num_shards,
            len(shard_qids),
            len(query_ids),
        )

    # Run dir
    run_name = cfg.get("run_name", f"query-enrich-{int(time.time())}")
    run_dir = os.path.join(ds.query_enrich_runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger.info("Run dir: %s", run_dir)

    if shard_rank == 0:
        with open(os.path.join(run_dir, "query_prompt.txt"), "w") as f:
            f.write(query_prompt)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

    # Resume: skip already-processed queries
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

    # Expand queries
    connector = aiohttp.TCPConnector(limit=enrich_cfg.concurrency)
    timeout = aiohttp.ClientTimeout(total=300.0, sock_read=60.0)
    sem = asyncio.Semaphore(enrich_cfg.concurrency)

    expansion_terms: list[str] = [""] * len(query_ids)
    query_enrichments: dict[str, list[str]] = {}
    failed: dict[str, str] = {}
    trace_kept_file = open(trace_kept_path, "a")
    trace_failed_file = open(trace_failed_path, "a")
    trace_lock = asyncio.Lock()
    stats = {"proposed": 0, "kept": 0, "filtered": 0, "errors": 0}

    async def _write_trace(record: dict, success: bool) -> None:
        f = trace_kept_file if success else trace_failed_file
        async with trace_lock:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def _expand_one(
        session: aiohttp.ClientSession, i: int, qid: str, qtext: str
    ) -> None:
        async with sem:
            t0 = time.time()
            try:
                raw_response, phrases = await _expand_query(
                    session,
                    llm_url,
                    model,
                    qtext,
                    enrich_cfg.query_max_tokens,
                    enrich_cfg.temperature,
                    query_prompt,
                    enrich_cfg.max_n,
                    enrich_cfg.seed,
                    chat_template_kwargs,
                )
            except Exception as e:
                logger.warning("LLM error for query %s: %s", qid, e)
                stats["errors"] += 1
                failed[qid] = f"error: {e}"
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

            ms = int((time.time() - t0) * 1000)
            stats["proposed"] += len(phrases)

            # Sliding-window DF filter: keep phrase if any 1..=max_n
            # sub-n-gram has 0 < DF <= max_df in the BM25 index.
            kept_phrases, rejected_pairs = bm25.filter_query_expansion(
                qtext,
                phrases,
                max_df,
            )

            stats["kept"] += len(kept_phrases)
            stats["filtered"] += len(rejected_pairs)

            # Tokenize kept phrases into stems for the expansion scoring
            # pass. search_with_expansion will re-tokenize and generate
            # sliding-window n-gram targets from this string.
            kept_stems = []
            for p in kept_phrases:
                kept_stems.extend(bm25.tokenize(p))
            expansion_terms[i] = " ".join(kept_stems) if kept_stems else ""

            if kept_phrases:
                query_enrichments[qid] = kept_phrases
                await _write_trace(
                    {
                        "query_id": qid,
                        "status": "ok",
                        "original": qtext,
                        "proposed": phrases,
                        "kept_phrases": kept_phrases,
                        "kept_stems": kept_stems,
                        "rejected": [
                            {"phrase": p, "reason": r} for p, r in rejected_pairs
                        ],
                        "raw_response": raw_response,
                        "ms": ms,
                    },
                    success=True,
                )
            else:
                reason = "no_phrases" if not phrases else "all_filtered"
                failed[qid] = reason
                await _write_trace(
                    {
                        "query_id": qid,
                        "status": reason,
                        "original": qtext,
                        "proposed": phrases,
                        "rejected": [
                            {"phrase": p, "reason": r} for p, r in rejected_pairs
                        ],
                        "raw_response": raw_response,
                        "ms": ms,
                    },
                    success=False,
                )

    expand_items = [
        (i, qid, qtext)
        for i, (qid, qtext) in enumerate(zip(query_ids, query_texts))
        if (num_shards <= 1 or qid in shard_qids) and qid not in done_qids
    ]
    logger.info("Expanding %d queries …", len(expand_items))
    t0 = time.time()

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for batch_start in range(0, len(expand_items), 10_000):
            batch = expand_items[batch_start : batch_start + 10_000]
            await asyncio.gather(
                *[_expand_one(session, i, qid, qtext) for i, qid, qtext in batch]
            )

    trace_kept_file.close()
    trace_failed_file.close()
    elapsed = time.time() - t0
    logger.info(
        "Query expansion done in %.1fs: %d kept / %d proposed / %d filtered / %d errors",
        elapsed,
        stats["kept"],
        stats["proposed"],
        stats["filtered"],
        stats["errors"],
    )

    if num_shards > 1:
        shard_suffix = f".shard{shard_rank}"
        with open(
            os.path.join(run_dir, f"enrichments.kept{shard_suffix}.jsonl"), "w"
        ) as f:
            for qid, phrases in query_enrichments.items():
                f.write(
                    json.dumps(
                        {"query_id": qid, "phrases": phrases}, ensure_ascii=False
                    )
                    + "\n"
                )
        with open(
            os.path.join(run_dir, f"expansion_terms{shard_suffix}.jsonl"), "w"
        ) as f:
            for i, t in enumerate(expansion_terms):
                if t:
                    f.write(
                        json.dumps({"index": i, "terms": t}, ensure_ascii=False) + "\n"
                    )
        logger.info("Shard %d done → %s", shard_rank, run_dir)
        return

    if stats["errors"] == len(query_ids):
        raise SystemExit(
            f"All {len(query_ids)} queries failed (0 enriched). Check LLM server."
        )

    # Retrieve: base + weighted expansion in one Rust call
    k_values = list(cfg.data.k_values)
    max_k = max(k_values)
    w = cfg.expansion_weight

    logger.info(
        "Searching %d queries (expansion_weight=%.2f) @ k=%d …",
        len(query_texts),
        w,
        max_k,
    )
    batch_results = bm25.search_with_expansion(
        query_texts,
        expansion_terms,
        k=max_k,
        weight=w,
    )

    beir_results: dict[str, dict[str, float]] = {}
    for qid, hits in zip(query_ids, batch_results):
        beir_results[qid] = {doc_ids[idx]: float(score) for idx, score in hits}

    # Save retrieval results for reranking
    retrieval_path = ds.retrieval_results(run_name, stage="query-enrich")
    os.makedirs(os.path.dirname(retrieval_path), exist_ok=True)
    with open(retrieval_path, "w") as f:
        for qid, hits in zip(query_ids, batch_results):
            record = {
                "query_id": qid,
                "candidates": [
                    {
                        "doc_id": doc_ids[idx],
                        "score": round(float(score), 6),
                        "rank": rank,
                    }
                    for rank, (idx, score) in enumerate(hits, 1)
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Saved %d retrieval results to %s", len(query_ids), retrieval_path)

    if qrels:
        from beir.retrieval.evaluation import EvaluateRetrieval

        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
            qrels, beir_results, k_values
        )
        metrics = {**ndcg, **recall, **precision}
        logger.info("Results (doc enrich + query expand):")
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

    # Save query enrichments
    with open(os.path.join(run_dir, "enrichments.kept.jsonl"), "w") as f:
        for qid, phrases in query_enrichments.items():
            f.write(
                json.dumps({"query_id": qid, "phrases": phrases}, ensure_ascii=False)
                + "\n"
            )
    with open(os.path.join(run_dir, "enrichments.failed.jsonl"), "w") as f:
        for qid, err in failed.items():
            f.write(
                json.dumps({"query_id": qid, "error": err}, ensure_ascii=False) + "\n"
            )
    os.makedirs(ds.query_enrichments_dir, exist_ok=True)
    with open(ds.query_enrichments(run_name), "w") as f:
        for qid, phrases in query_enrichments.items():
            f.write(
                json.dumps({"query_id": qid, "phrases": phrases}, ensure_ascii=False)
                + "\n"
            )
    logger.info(
        "Coverage: %d/%d enriched, %d failed",
        len(query_enrichments),
        len(query_ids),
        len(failed),
    )

    # Save eval to eval/query-enrich/
    eval_path = ds.eval_query_enrich(run_name)
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Update best symlink
    selection_metric = cfg.selection_metric

    doc_enrich_run = None
    doc_enrich_meta_path = os.path.join(ds.doc_enrichments_dir, "best.meta.json")
    if os.path.exists(doc_enrich_meta_path):
        with open(doc_enrich_meta_path) as f:
            doc_enrich_run = json.load(f).get("run_name")

    meta = {
        "stage": "query-enrich",
        "dataset": ds.name,
        "run_name": run_name,
        "run_dir": run_dir,
        "metrics": metrics,
        "stats": stats,
        "enrich_params": {
            "max_n": enrich_cfg.max_n,
            "max_df_ratio": enrich_cfg.max_df_ratio,
            "max_df": max_df,
            "temperature": enrich_cfg.temperature,
            "seed": enrich_cfg.seed,
        },
        "expansion_weight": w,
        "query_prompt_file": enrich_cfg.query_prompt_file,
        "model": model,
        "doc_enrich_run": doc_enrich_run,
        "num_queries": len(query_ids),
        "num_enriched": len(query_enrichments),
        "num_failed": len(failed),
        "elapsed_s": round(elapsed, 1),
        "timestamp": int(time.time()),
    }
    ds.update_best(
        best_links=[ds.eval_query_enrich_best],
        target_name=f"{run_name}.json",
        meta=meta,
        selection_metrics=None,
    )
    ds.update_best(
        best_links=[
            ds.query_enrichments_best,
            ds.retrieval_best("query-enrich"),
        ],
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


def merge_query_shards(cfg: DictConfig) -> None:
    """Merge shard expansion terms, then run retrieval + eval + best-update."""
    ds = _ds(cfg)
    sglang_cfg = cfg.sglang
    num_shards = int(cfg.get("num_shards", 1))
    run_name = cfg.get("run_name", "")
    run_dir = os.path.join(ds.query_enrich_runs_dir, run_name)

    # Load queries
    split = cfg.data.split
    queries_df = read_queries(ds.queries(split))
    query_ids = queries_df.get_column(COL_ID).to_list()
    query_texts = queries_df.get_column(COL_TEXT).to_list()
    qrels = load_qrels_dict(ds.qrels(split))

    # Merge expansion terms from all shards (JSONL)
    expansion_terms: list[str] = [""] * len(query_ids)
    merged_enrichments: dict[str, list[str]] = {}
    for i in range(num_shards):
        terms_path = os.path.join(run_dir, f"expansion_terms.shard{i}.jsonl")
        if os.path.exists(terms_path):
            with open(terms_path) as f:
                for line in f:
                    rec = json.loads(line)
                    expansion_terms[rec["index"]] = rec["terms"]
        enrich_path = os.path.join(run_dir, f"enrichments.kept.shard{i}.jsonl")
        if os.path.exists(enrich_path):
            with open(enrich_path) as f:
                for line in f:
                    rec = json.loads(line)
                    merged_enrichments[rec["query_id"]] = rec["phrases"]
        logger.info("Loaded query shard %d", i)

    logger.info(
        "Merged %d query enrichments from %d shards",
        len(merged_enrichments),
        num_shards,
    )

    # Save merged enrichments
    with open(os.path.join(run_dir, "enrichments.kept.jsonl"), "w") as f:
        for qid, phrases in merged_enrichments.items():
            f.write(
                json.dumps({"query_id": qid, "phrases": phrases}, ensure_ascii=False)
                + "\n"
            )
    os.makedirs(ds.query_enrichments_dir, exist_ok=True)
    with open(ds.query_enrichments(run_name), "w") as f:
        for qid, phrases in merged_enrichments.items():
            f.write(
                json.dumps({"query_id": qid, "phrases": phrases}, ensure_ascii=False)
                + "\n"
            )

    # Load BM25 index + apply doc enrichments
    index_dir = ds.bm25_index_best
    bm25 = BM25.load(index_dir)
    bm25.disable_auto_save()
    doc_ids, _ = read_corpus_texts(ds.corpus)
    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    if os.path.exists(ds.doc_enrichments_best):
        items = []
        with open(ds.doc_enrichments_best) as f:
            for line in f:
                rec = json.loads(line)
                idx = doc_id_to_idx.get(rec["doc_id"])
                if idx is not None:
                    items.append((idx, rec["phrases"]))
        bm25.enrich_batch(items)
        logger.info("Applied %d doc enrichments", len(items))

    # Retrieve
    k_values = list(cfg.data.k_values)
    max_k = max(k_values)
    w = cfg.expansion_weight

    logger.info(
        "Searching %d queries (expansion_weight=%.2f) @ k=%d …",
        len(query_texts),
        w,
        max_k,
    )
    batch_results = bm25.search_with_expansion(
        query_texts, expansion_terms, k=max_k, weight=w
    )

    beir_results: dict[str, dict[str, float]] = {}
    for qid, hits in zip(query_ids, batch_results):
        beir_results[qid] = {doc_ids[idx]: float(score) for idx, score in hits}

    # Save retrieval results
    retrieval_path = ds.retrieval_results(run_name, stage="query-enrich")
    os.makedirs(os.path.dirname(retrieval_path), exist_ok=True)
    with open(retrieval_path, "w") as f:
        for qid, hits in zip(query_ids, batch_results):
            record = {
                "query_id": qid,
                "candidates": [
                    {
                        "doc_id": doc_ids[idx],
                        "score": round(float(score), 6),
                        "rank": rank,
                    }
                    for rank, (idx, score) in enumerate(hits, 1)
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Saved %d retrieval results to %s", len(query_ids), retrieval_path)

    # Evaluate
    if qrels:
        from beir.retrieval.evaluation import EvaluateRetrieval

        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
            qrels, beir_results, k_values
        )
        metrics = {**ndcg, **recall, **precision}
        logger.info("Results (doc enrich + query expand):")
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

    # Save eval + update best
    eval_path = ds.eval_query_enrich(run_name)
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)

    selection_metric = cfg.selection_metric
    meta = {
        "stage": "query-enrich",
        "dataset": ds.name,
        "run_name": run_name,
        "run_dir": run_dir,
        "metrics": metrics,
        "model": sglang_cfg.model,
        "num_queries": len(query_ids),
        "num_enriched": len(merged_enrichments),
        "num_shards": num_shards,
        "timestamp": int(time.time()),
    }
    ds.update_best(
        best_links=[ds.eval_query_enrich_best],
        target_name=f"{run_name}.json",
        meta=meta,
        selection_metrics=None,
    )
    ds.update_best(
        best_links=[
            ds.query_enrichments_best,
            ds.retrieval_best("query-enrich"),
        ],
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
    logger.info("Query merge done → %s", run_dir)


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="enrich_query_and_retrieve",
)
def main(cfg: DictConfig) -> None:
    if cfg.get("merge_shards", False):
        merge_query_shards(cfg)
    else:
        asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
