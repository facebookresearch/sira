# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Enrich a BM25 index with LLM-proposed keywords, then evaluate.

For each document in the corpus, asks an LLM to propose keyword phrases
that would help retrieve the document. Non-empty phrases are applied to
the BM25 index via ``enrich``, then Recall@K is evaluated.

Usage::

    source sandbox.sh
    python scripts/add_doc_index_adapter.py
    python scripts/add_doc_index_adapter.py data=fiqa enrich.concurrency=512

Config groups::

    global   — db_root
    data/    — dataset name, k_values
    enrich/  — concurrency, max_tokens, temperature, seed, max_df_ratio, prompt_file
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
import polars as pl
from bm25x import BM25, NGramIndex
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
GATHER_BATCH_SIZE = 10_000


def _load_prompt(cfg: DictConfig) -> str:
    prompt_path = os.path.join(SCRIPTS_DIR, cfg.enrich.doc_prompt_file)
    with open(prompt_path) as f:
        return f.read()


def _ds(cfg: DictConfig) -> DatasetDir:
    return DatasetDir(root=cfg.db_root, name=cfg.data.name)


def _read_bm25_max_n(ds: DatasetDir) -> int | None:
    meta_path = os.path.join(os.path.dirname(ds.bm25_index_best), "best.meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f).get("bm25_params", {}).get("max_n")
    return None


async def _propose(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    doc_text: str,
    max_tokens: int,
    temperature: float,
    prompt_template: str,
    max_n: int,
    max_doc_chars: int = 4000,
    seed: int | None = None,
    chat_template_kwargs: dict | None = None,
) -> tuple[str, list[str]]:
    """Returns (raw_response, parsed_phrases)."""
    prompt = prompt_template.format(doc_text=doc_text[:max_doc_chars], max_n=max_n)
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


async def enrich_corpus(cfg: DictConfig) -> None:
    ds = _ds(cfg)
    enrich_cfg = cfg.enrich
    sglang_cfg = cfg.sglang
    prompt_template = _load_prompt(cfg)
    logger.info("Loaded prompt from %s", enrich_cfg.doc_prompt_file)

    chat_template_kwargs = None
    if getattr(sglang_cfg, "chat_template_kwargs", None):
        chat_template_kwargs = OmegaConf.to_container(sglang_cfg.chat_template_kwargs)
        logger.info("chat_template_kwargs: %s", chat_template_kwargs)

    llm_url = f"http://127.0.0.1:{sglang_cfg.port}/v1/chat/completions"
    model = sglang_cfg.model

    # Load BM25 index
    index_dir = ds.bm25_index_best
    if not os.path.exists(index_dir):
        raise SystemExit(
            f"BM25 index not found at {index_dir}. Run eval_bm25.py first."
        )
    bm25 = BM25.load(index_dir)
    bm25.disable_auto_save()
    logger.info("Loaded BM25 index: %d docs", len(bm25))

    # Auto-detect max_n from BM25 index
    max_n = _read_bm25_max_n(ds)
    if max_n is None:
        max_n = enrich_cfg.max_n
        logger.info("No BM25 meta found, using enrich.max_n=%d", max_n)
    else:
        logger.info("Auto max_n=%d from BM25 index", max_n)

    doc_ids, texts = read_corpus_texts(ds.corpus)
    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    # Build NGramIndex for Rust-side filtering
    logger.info("Building NGramIndex for df filter …")
    ngram_index = NGramIndex(max_n=max_n, tokenizer=enrich_cfg.tokenizer)
    ngram_index.add(texts)
    max_df_ratio = getattr(enrich_cfg, "max_df_ratio", 0)
    max_df = int(len(doc_ids) * max_df_ratio) if max_df_ratio > 0 else 0xFFFFFFFF
    logger.info("NGramIndex: vocab=%d, max_df=%d", ngram_index.vocab_size(), max_df)

    target_ids = doc_ids
    target_texts = texts
    if enrich_cfg.max_docs > 0:
        target_ids = doc_ids[: enrich_cfg.max_docs]
        target_texts = texts[: enrich_cfg.max_docs]
    del texts

    shard_rank = int(cfg.get("shard_rank", 0))
    num_shards = int(cfg.get("num_shards", 1))
    if num_shards > 1:
        target_ids = target_ids[shard_rank::num_shards]
        target_texts = target_texts[shard_rank::num_shards]
        logger.info(
            "Shard %d/%d: processing %d docs", shard_rank, num_shards, len(target_ids)
        )

    run_name = cfg.get("run_name", f"enrich-{int(time.time())}")
    run_dir = os.path.join(ds.doc_enrich_runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ds.doc_enrichments_dir, exist_ok=True)
    logger.info("Run dir: %s", run_dir)

    if shard_rank == 0:
        with open(os.path.join(run_dir, "prompt.txt"), "w") as f:
            f.write(prompt_template)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

    # Resume: skip already-processed docs
    shard_suffix = f".shard{shard_rank}" if num_shards > 1 else ""
    enrich_path = os.path.join(run_dir, f"enrichments.kept{shard_suffix}.jsonl")
    trace_kept_path = os.path.join(run_dir, f"trace.kept{shard_suffix}.jsonl")
    trace_failed_path = os.path.join(run_dir, f"trace.failed{shard_suffix}.jsonl")

    done_ids: set[str] = set()
    for path in [trace_kept_path, trace_failed_path]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        done_ids.add(json.loads(line)["doc_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    if done_ids:
        before = len(target_ids)
        pairs = [
            (did, txt)
            for did, txt in zip(target_ids, target_texts)
            if did not in done_ids
        ]
        target_ids = [p[0] for p in pairs]
        target_texts = [p[1] for p in pairs]
        logger.info(
            "Resuming: %d/%d already done, %d remaining",
            len(done_ids),
            before,
            len(target_ids),
        )

    connector = aiohttp.TCPConnector(limit=enrich_cfg.concurrency)
    timeout = aiohttp.ClientTimeout(total=300.0, sock_read=60.0)
    sem = asyncio.Semaphore(enrich_cfg.concurrency)

    stats = {"proposed": 0, "kept": 0, "filtered": 0, "errors": 0}
    enriched_count = 0
    failed_count = 0
    done_count = 0
    total_count = len(target_ids)

    trace_kept_file = open(trace_kept_path, "a")
    trace_failed_file = open(trace_failed_path, "a")
    enrich_file = open(enrich_path, "a")
    file_lock = asyncio.Lock()

    async def _write_line(f, line: str) -> None:
        async with file_lock:
            f.write(line + "\n")

    async def _enrich_one(
        session: aiohttp.ClientSession, doc_id: str, doc_text: str
    ) -> None:
        nonlocal done_count, enriched_count, failed_count
        async with sem:
            t0_doc = time.time()
            try:
                raw_response, phrases = await _propose(
                    session,
                    llm_url,
                    model,
                    doc_text,
                    enrich_cfg.doc_max_tokens,
                    enrich_cfg.temperature,
                    prompt_template,
                    max_n,
                    enrich_cfg.max_doc_chars,
                    enrich_cfg.seed,
                    chat_template_kwargs,
                )
            except Exception as e:
                logger.warning("LLM error for %s: %s", doc_id, e)
                stats["errors"] += 1
                failed_count += 1
                done_count += 1
                await _write_line(
                    trace_failed_file,
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "status": "error",
                            "error": str(e),
                            "ms": int((time.time() - t0_doc) * 1000),
                        },
                        ensure_ascii=False,
                    ),
                )
                return

            ms_elapsed = int((time.time() - t0_doc) * 1000)
            done_count += 1
            if done_count % 1000 == 0:
                logger.info(
                    "Progress: %d/%d (%.0f%%), %d kept so far",
                    done_count,
                    total_count,
                    done_count / total_count * 100,
                    stats["kept"],
                )

            if not phrases:
                failed_count += 1
                await _write_line(
                    trace_failed_file,
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "status": "no_phrases",
                            "raw_response": raw_response,
                            "ms": ms_elapsed,
                        },
                        ensure_ascii=False,
                    ),
                )
                return

            stats["proposed"] += len(phrases)

            kept, filter_stats, _ = ngram_index.filter_candidates(
                phrases,
                "",
                [],
                max_df,
                max_n,
                require_in_vocab=False,
                collect_verdicts=False,
            )
            filter_stats_dict = dict(filter_stats)
            n_kept = filter_stats_dict.get("kept", 0)
            stats["kept"] += n_kept
            stats["filtered"] += len(phrases) - n_kept

            if kept:
                enriched_count += 1
                await _write_line(
                    enrich_file,
                    json.dumps({"doc_id": doc_id, "phrases": kept}, ensure_ascii=False),
                )
                await _write_line(
                    trace_kept_file,
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "status": "ok",
                            "raw_response": raw_response,
                            "proposed": phrases,
                            "kept": kept,
                            "filter_stats": filter_stats_dict,
                            "ms": ms_elapsed,
                        },
                        ensure_ascii=False,
                    ),
                )
            else:
                failed_count += 1
                await _write_line(
                    trace_failed_file,
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "status": "all_filtered",
                            "raw_response": raw_response,
                            "proposed": phrases,
                            "filter_stats": filter_stats_dict,
                            "ms": ms_elapsed,
                        },
                        ensure_ascii=False,
                    ),
                )

    logger.info(
        "Enriching %d docs (concurrency=%d) …", total_count, enrich_cfg.concurrency
    )
    t0 = time.time()

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        items = list(zip(target_ids, target_texts))
        for batch_start in range(0, len(items), GATHER_BATCH_SIZE):
            batch = items[batch_start : batch_start + GATHER_BATCH_SIZE]
            await asyncio.gather(
                *[_enrich_one(session, did, txt) for did, txt in batch]
            )

    # Apply enrichments to BM25 index (single-node only)
    if num_shards <= 1 and enriched_count > 0:
        enrich_file.close()
        enrichments = {}
        with open(os.path.join(run_dir, "enrichments.kept.jsonl")) as f:
            for line in f:
                rec = json.loads(line)
                enrichments.setdefault(rec["doc_id"], []).extend(rec["phrases"])
        items = [(doc_id_to_idx[did], phrases) for did, phrases in enrichments.items()]
        bm25.enrich_batch(items)
        logger.info("Applied enrichments to %d docs via enrich_batch", len(items))
        # Also write as single JSONL for downstream (doc_enrichments)
        with open(ds.doc_enrichments(run_name), "w") as f:
            for doc_id, phrases in enrichments.items():
                f.write(
                    json.dumps(
                        {"doc_id": doc_id, "phrases": phrases}, ensure_ascii=False
                    )
                    + "\n"
                )
    else:
        enrich_file.close()

    trace_kept_file.close()
    trace_failed_file.close()
    elapsed = time.time() - t0
    logger.info(
        "Enrichment done in %.1fs (%.1f doc/s): %d docs enriched, "
        "%d kept / %d proposed / %d filtered / %d errors",
        elapsed,
        total_count / max(elapsed, 0.01),
        enriched_count,
        stats["kept"],
        stats["proposed"],
        stats["filtered"],
        stats["errors"],
    )

    if total_count > 0 and stats["errors"] == total_count:
        raise SystemExit(
            f"All {total_count} docs failed (0 enriched). Check LLM server."
        )

    with open(os.path.join(run_dir, f"stats{shard_suffix}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(
        "Coverage: %d/%d enriched, %d failed", enriched_count, total_count, failed_count
    )

    if num_shards > 1:
        logger.info("Shard %d done → %s", shard_rank, run_dir)
        return

    # Evaluate
    k_values = list(cfg.data.k_values)
    split = cfg.data.split
    queries_df = read_queries(ds.queries(split))
    query_ids = queries_df.get_column(COL_ID).to_list()
    query_texts = queries_df.get_column(COL_TEXT).to_list()
    qrels = load_qrels_dict(ds.qrels(split))

    max_k = max(k_values)
    logger.info("Evaluating %d queries @ k=%d …", len(query_texts), max_k)
    batch_results = bm25.search(query_texts, k=max_k)

    beir_results: dict[str, dict[str, float]] = {}
    for qid, hits in zip(query_ids, batch_results):
        beir_results[qid] = {doc_ids[idx]: float(score) for idx, score in hits}

    from beir.retrieval.evaluation import EvaluateRetrieval

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, beir_results, k_values
    )

    metrics = {**ndcg, **recall, **precision}
    logger.info("Results (enriched):")
    for k in k_values:
        logger.info(
            "  k=%d  NDCG=%.4f  Recall=%.4f",
            k,
            ndcg.get(f"NDCG@{k}", 0),
            recall.get(f"Recall@{k}", 0),
        )

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    eval_path = ds.eval_doc_enrich(run_name)
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)

    retrieval_path = ds.retrieval_results(run_name, stage="doc-enrich")
    os.makedirs(os.path.dirname(retrieval_path), exist_ok=True)
    with open(retrieval_path, "w") as f:
        for qid in query_ids:
            scored = beir_results.get(qid, {})
            candidates = [
                {"doc_id": did, "score": s, "rank": r}
                for r, (did, s) in enumerate(
                    sorted(scored.items(), key=lambda x: -x[1]), 1
                )
            ]
            f.write(json.dumps({"query_id": qid, "candidates": candidates}) + "\n")

    selection_metric = cfg.selection_metric
    max_df_ratio = getattr(enrich_cfg, "max_df_ratio", 0)
    meta = {
        "stage": "doc-enrich",
        "dataset": ds.name,
        "run_name": run_name,
        "run_dir": run_dir,
        "metrics": metrics,
        "stats": stats,
        "enrich_params": {
            "max_n": max_n,
            "max_df_ratio": max_df_ratio,
            "max_df": max_df,
            "temperature": enrich_cfg.temperature,
            "seed": enrich_cfg.seed,
            "prompt_file": enrich_cfg.doc_prompt_file,
        },
        "model": model,
        "num_docs_enriched": enriched_count,
        "num_failed": failed_count,
        "corpus_size": len(doc_ids),
        "elapsed_s": round(elapsed, 1),
        "timestamp": int(time.time()),
    }
    ds.update_best(
        best_links=[ds.eval_doc_enrich_best],
        target_name=f"{run_name}.json",
        meta=meta,
        selection_metrics=None,
    )
    ds.update_best(
        best_links=[
            os.path.join(ds.doc_enrichments_dir, "best.jsonl"),
            ds.retrieval_best("doc-enrich"),
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


def merge_shards(cfg: DictConfig) -> None:
    """Merge shard enrichments, then run eval + best-update."""
    ds = _ds(cfg)
    num_shards = int(cfg.get("num_shards", 1))
    run_name = cfg.get("run_name", "")
    run_dir = os.path.join(ds.doc_enrich_runs_dir, run_name)
    enrich_cfg = cfg.enrich
    sglang_cfg = cfg.sglang
    model = sglang_cfg.model

    index_dir = ds.bm25_index_best
    bm25 = BM25.load(index_dir)
    bm25.disable_auto_save()

    # Vectorized merge: polars for JSONL reading + join, shutil for file concat
    os.makedirs(ds.doc_enrichments_dir, exist_ok=True)
    merged_path = ds.doc_enrichments(run_name)

    corpus_idx = pl.read_ndjson(ds.corpus, schema={"_id": pl.Utf8}).with_row_index(
        "idx"
    )

    shard_paths = []
    for i in range(num_shards):
        path = os.path.join(run_dir, f"enrichments.kept.shard{i}.jsonl")
        if not os.path.exists(path):
            logger.warning("Missing shard %d: %s", i, path)
            continue
        shard_paths.append(path)

    # Read shards with polars, join to get corpus idx
    df = pl.concat(
        [
            pl.read_ndjson(p, schema={"doc_id": pl.Utf8, "phrases": pl.List(pl.Utf8)})
            for p in shard_paths
        ]
    )
    joined = df.join(corpus_idx, left_on="doc_id", right_on="_id", how="inner")

    # Write merged JSONL (single pass from polars, no re-read)
    df.write_ndjson(merged_path)
    n_items = len(joined)
    logger.info("Read %d enrichments from %d shards", n_items, len(shard_paths))

    if n_items > 0:
        t0 = time.time()
        if hasattr(bm25, "enrich_batch_columnar"):
            import numpy as np

            indices = joined["idx"].to_numpy().astype(np.uint32)
            lengths = joined["phrases"].list.len().to_numpy()
            doc_offsets = np.zeros(n_items + 1, dtype=np.uint64)
            np.cumsum(lengths, out=doc_offsets[1:])
            arrow_array = joined["phrases"].explode().to_arrow()
            buffers = arrow_array.buffers()
            str_offsets = np.frombuffer(buffers[1], dtype=np.int64)[
                : len(arrow_array) + 1
            ]
            str_data = bytes(buffers[2])
            logger.info(
                "Arrow zero-copy: %d docs, %d phrases, %d MB str_data",
                n_items,
                len(arrow_array),
                len(str_data) // 1_000_000,
            )
            bm25.enrich_batch_columnar(indices, doc_offsets, str_data, str_offsets)
        else:
            items = list(zip(joined["idx"].to_list(), joined["phrases"].to_list()))
            logger.info("Fallback enrich_batch: %d items", len(items))
            bm25.enrich_batch(items)
        logger.info("enrich_batch done in %.1fs", time.time() - t0)

    logger.info(
        "Merged %d docs from %d shards, applied to BM25 index",
        n_items,
        len(shard_paths),
    )

    # Evaluate
    k_values = list(cfg.data.k_values)
    split = cfg.data.split
    queries_df = read_queries(ds.queries(split))
    query_ids = queries_df.get_column(COL_ID).to_list()
    query_texts = queries_df.get_column(COL_TEXT).to_list()
    qrels = load_qrels_dict(ds.qrels(split))

    max_k = max(k_values)
    logger.info("Evaluating %d queries @ k=%d …", len(query_texts), max_k)
    t0 = time.time()
    batch_results = bm25.search(query_texts, k=max_k)
    logger.info("BM25 search done in %.1fs", time.time() - t0)

    doc_id_arr = corpus_idx["_id"]
    beir_results: dict[str, dict[str, float]] = {}
    for qid, hits in zip(query_ids, batch_results):
        beir_results[qid] = {doc_id_arr[idx]: float(score) for idx, score in hits}

    from beir.retrieval.evaluation import EvaluateRetrieval

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, beir_results, k_values
    )

    metrics = {**ndcg, **recall, **precision}
    logger.info("Results (enriched):")
    for k in k_values:
        logger.info(
            "  k=%d  NDCG=%.4f  Recall=%.4f",
            k,
            ndcg.get(f"NDCG@{k}", 0),
            recall.get(f"Recall@{k}", 0),
        )

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    eval_path = ds.eval_doc_enrich(run_name)
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)

    retrieval_path = ds.retrieval_results(run_name, stage="doc-enrich")
    os.makedirs(os.path.dirname(retrieval_path), exist_ok=True)
    with open(retrieval_path, "w") as f:
        for qid in query_ids:
            scored = beir_results.get(qid, {})
            candidates = [
                {"doc_id": did, "score": s, "rank": r}
                for r, (did, s) in enumerate(
                    sorted(scored.items(), key=lambda x: -x[1]), 1
                )
            ]
            f.write(json.dumps({"query_id": qid, "candidates": candidates}) + "\n")

    max_n = _read_bm25_max_n(ds) or enrich_cfg.max_n
    selection_metric = cfg.selection_metric
    max_df_ratio = getattr(enrich_cfg, "max_df_ratio", 0)
    meta = {
        "stage": "doc-enrich",
        "dataset": ds.name,
        "run_name": run_name,
        "run_dir": run_dir,
        "metrics": metrics,
        "enrich_params": {
            "max_n": max_n,
            "max_df_ratio": max_df_ratio,
            "temperature": enrich_cfg.temperature,
            "seed": enrich_cfg.seed,
            "prompt_file": enrich_cfg.doc_prompt_file,
        },
        "model": model,
        "num_docs_enriched": n_items,
        "corpus_size": len(corpus_idx),
        "num_shards": num_shards,
        "timestamp": int(time.time()),
    }
    ds.update_best(
        best_links=[ds.eval_doc_enrich_best],
        target_name=f"{run_name}.json",
        meta=meta,
        selection_metrics=None,
    )
    ds.update_best(
        best_links=[
            os.path.join(ds.doc_enrichments_dir, "best.jsonl"),
            ds.retrieval_best("doc-enrich"),
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
    logger.info("Merge done → %s", run_dir)


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="add_doc_index_adapter",
)
def main(cfg: DictConfig) -> None:
    if cfg.get("merge_shards", False):
        merge_shards(cfg)
    else:
        asyncio.run(enrich_corpus(cfg))


if __name__ == "__main__":
    main()
