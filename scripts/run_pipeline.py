# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Run the full SIRA pipeline.

Stages: prepare → bm25 → enrich_corpus → enrich_query → rerank

Usage::

    source sandbox.sh
    python scripts/run_pipeline.py                                    # scifact, all stages
    python scripts/run_pipeline.py data=fiqa                          # single dataset
    python scripts/run_pipeline.py stages='[enrich_query,rerank]'     # subset of stages
    python scripts/run_pipeline.py datasets='[scifact,fiqa,arguana]'  # multi-dataset

Multi-node (MAST) — auto-detects RANK/WORLD_SIZE from environment::

    python scripts/run_pipeline.py datasets='[scifact,fiqa]'

LLM server is auto-detected: if one is already running on the configured
port it will be reused, otherwise a new one is started and stopped
automatically.

Stages::

    prepare        — download and prepare dataset (skips if already done)
    bm25           — build BM25 index and evaluate (skips if eval exists)
    enrich_corpus  — LLM doc enrichment (sharded across nodes)
    enrich_query   — LLM query expansion + retrieval (sharded across nodes)
    rerank         — LLM reranking (sharded across nodes)
"""

import asyncio
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf
from sira.schema.mteb import DatasetDir
from sira.schema.sglang import SGLangServerConfig

logger = logging.getLogger(__name__)

ALL_STAGES = ["prepare", "bm25", "enrich_corpus", "enrich_query", "greprag", "iterative", "rerank"]
LLM_STAGES = {"enrich_corpus", "enrich_query", "greprag", "iterative", "rerank"}

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(SCRIPTS_DIR, "configs")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _with_dataset(cfg: DictConfig, ds_name: str) -> DictConfig:
    """Return a copy of cfg with data.* replaced by the named dataset config.

    Also applies per-dataset enrich/rerank overrides if they exist
    (e.g. configs/enrich/quora.yaml, configs/rerank/quora.yaml).
    """
    data_path = os.path.join(CONFIGS_DIR, "data", f"{ds_name}.yaml")
    with open(data_path) as f:
        data_cfg = yaml.safe_load(f)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg["data"] = data_cfg

    for section in ("enrich", "greprag", "iterative", "rerank"):
        override_path = os.path.join(CONFIGS_DIR, section, f"{ds_name}.yaml")
        if os.path.exists(override_path):
            with open(override_path) as f:
                override_cfg = yaml.safe_load(f)
            if override_cfg:
                cfg[section].update(override_cfg)
                logger.info("Using %s/%s.yaml override for %s", section, ds_name, ds_name)

    return OmegaConf.create(cfg)


def _with_overrides(cfg: DictConfig, **overrides) -> DictConfig:
    """Return a copy of cfg with extra keys merged in."""
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg.update(overrides)
    return OmegaConf.create(cfg)


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def _run_prepare(cfg: DictConfig) -> None:
    from prepare_mteb_data import download

    download(cfg)


def _run_bm25(cfg: DictConfig) -> None:
    from eval_bm25 import build_and_evaluate

    build_and_evaluate(cfg)


def _run_enrich_corpus(cfg: DictConfig) -> None:
    from add_doc_index_adapter import enrich_corpus

    asyncio.run(enrich_corpus(cfg))


def _run_merge_corpus_shards(cfg: DictConfig) -> None:
    from add_doc_index_adapter import merge_shards

    merge_shards(cfg)


def _run_enrich_query(cfg: DictConfig) -> None:
    from enrich_query_and_retrieve import run

    asyncio.run(run(cfg))


def _run_merge_query_shards(cfg: DictConfig) -> None:
    from enrich_query_and_retrieve import merge_query_shards

    merge_query_shards(cfg)


def _run_greprag(cfg: DictConfig) -> None:
    from greprag_retrieve import run

    asyncio.run(run(cfg))


def _run_merge_greprag_shards(cfg: DictConfig) -> None:
    from greprag_retrieve import merge_greprag_shards

    merge_greprag_shards(cfg)


def _run_iterative(cfg: DictConfig) -> None:
    from iterative_retrieve import run

    asyncio.run(run(cfg))


def _run_merge_iterative_shards(cfg: DictConfig) -> None:
    from iterative_retrieve import merge_iterative_shards

    merge_iterative_shards(cfg)


def _run_rerank(cfg: DictConfig) -> None:
    from llm_reranking import run

    asyncio.run(run(cfg))


def _run_merge_rerank_shards(cfg: DictConfig) -> None:
    from llm_reranking import merge_rerank_shards

    merge_rerank_shards(cfg)


STAGE_RUNNERS = {
    "prepare": _run_prepare,
    "bm25": _run_bm25,
    "enrich_corpus": _run_enrich_corpus,
    "enrich_query": _run_enrich_query,
    "greprag": _run_greprag,
    "iterative": _run_iterative,
    "rerank": _run_rerank,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gpu_info() -> list[dict]:
    try:
        import torch

        return [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_mb": torch.cuda.get_device_properties(i).total_memory // (1024 * 1024),
            }
            for i in range(torch.cuda.device_count())
        ]
    except Exception:
        return []


def _find_resumable_run(ds: DatasetDir, stage: str, num_shards: int) -> str | None:
    """Find an existing run dir with partial shard progress but no merged output."""
    runs_dir = {
        "enrich_corpus": ds.doc_enrich_runs_dir,
        "enrich_query": ds.query_enrich_runs_dir,
        "greprag": ds.greprag_runs_dir,
        "iterative": ds.iterative_runs_dir,
        "rerank": ds.rerank_runs_dir,
    }.get(stage)
    if not runs_dir or not os.path.isdir(runs_dir):
        return None
    for name in sorted(os.listdir(runs_dir), reverse=True):
        run_dir = os.path.join(runs_dir, name)
        if not os.path.isdir(run_dir):
            continue
        has_shards = any(
            f.startswith("trace.kept.shard") for f in os.listdir(run_dir)
        )
        has_merged = any(
            f in ("metrics.json", "enrichments.kept.json", "enrichments.kept.jsonl", "reranked.jsonl")
            for f in os.listdir(run_dir)
        )
        if has_shards and not has_merged:
            return name
    return None


def _should_skip(stage: str, ds: DatasetDir, skip_existing: bool = False) -> bool:
    if stage == "prepare":
        return os.path.exists(ds.metadata)
    if stage == "bm25":
        return os.path.exists(ds.bm25_index_best) and os.path.exists(
            ds.eval_baseline_best
        )
    if skip_existing and stage == "enrich_corpus":
        return os.path.exists(ds.eval_doc_enrich_best)
    return False


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------


def _start_server(cfg: DictConfig) -> subprocess.Popen:
    sglang_cfg = SGLangServerConfig()
    for k, v in cfg.sglang.items():
        if hasattr(sglang_cfg, k):
            setattr(sglang_cfg, k, v)

    cmd = [sys.executable, "-m", "sglang.launch_server", *sglang_cfg.to_cli_args()]
    logger.info("Starting sglang server...")
    proc = subprocess.Popen(cmd)

    port = cfg.sglang.port
    timeout = int(cfg.server.get("wait_timeout", 900))
    urls = [f"http://127.0.0.1:{port}/v1/models", f"http://[::1]:{port}/v1/models"]

    for i in range(1, timeout + 1):
        if proc.poll() is not None:
            raise RuntimeError("sglang server process died during startup")
        for url in urls:
            try:
                urllib.request.urlopen(url, timeout=2)
                logger.info("sglang server ready after %ds (PID=%d)", i, proc.pid)
                return proc
            except Exception:
                pass
        if i % 30 == 0:
            logger.info("  Waiting for server... (%ds)", i)
        time.sleep(1)

    proc.kill()
    proc.wait()
    raise RuntimeError(f"sglang server not ready after {timeout}s")


def _stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    logger.info("Stopping sglang server (PID=%d)...", proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------------------
# Multi-node sync (shared-storage signal files)
# ---------------------------------------------------------------------------


def _detect_node_info() -> tuple[int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world


def _signal_dir(cfg: DictConfig) -> str:
    job_name = os.environ.get("SIRA_JOB_NAME", "local")
    return os.path.join(cfg.db_root, f".signals-{job_name}")


def _signal_done(sig_dir: str, label: str, rank: int) -> None:
    os.makedirs(sig_dir, exist_ok=True)
    open(os.path.join(sig_dir, f"{label}-{rank}-done"), "w").close()


def _wait_all(sig_dir: str, label: str, num_nodes: int) -> None:
    for r in range(num_nodes):
        path = os.path.join(sig_dir, f"{label}-{r}-done")
        while not os.path.exists(path):
            time.sleep(5)
        logger.info("  Node %d: %s done", r, label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _resolve_stages(cfg: DictConfig, ds_name: str) -> list[str]:
    """Return the stage list for *ds_name*.

    Uses ``dataset_stages.<ds_name>`` if present, otherwise falls back
    to the global ``stages`` list.  Example CLI usage::

        python run_pipeline.py \\
            datasets='[quora,nq,fever,climate-fever]' \\
            stages='[enrich_corpus,enrich_query,rerank]' \\
            '+dataset_stages.quora=[rerank]' \\
            '+dataset_stages.nq=[rerank]'
    """
    per_ds = cfg.get("dataset_stages")
    if per_ds and ds_name in per_ds:
        return list(per_ds[ds_name])
    return list(cfg.stages)


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="run_pipeline",
)
def main(cfg: DictConfig) -> None:
    datasets = list(cfg.datasets) if cfg.get("datasets") else [cfg.data.name]

    all_stages: set[str] = set()
    ds_stage_map: dict[str, list[str]] = {}
    for ds_name in datasets:
        ds_stages = _resolve_stages(cfg, ds_name)
        for s in ds_stages:
            if s not in STAGE_RUNNERS:
                logger.error("Unknown stage %r, valid: %s", s, ALL_STAGES)
                sys.exit(1)
        ds_stage_map[ds_name] = ds_stages
        all_stages.update(ds_stages)

    node_rank, num_nodes = _detect_node_info()
    multi_node = num_nodes > 1
    job_name = os.environ.get("SIRA_JOB_NAME", "local")

    logger.info(
        "Pipeline: datasets=%s, stages=%s, node=%d/%d",
        datasets,
        {ds: ds_stage_map[ds] for ds in datasets},
        node_rank,
        num_nodes,
    )

    # --- Ensure LLM server is available ---
    server_proc = None
    gpus = _gpu_info()
    if all_stages & LLM_STAGES:
        port = cfg.sglang.port
        server_running = False
        for url in [f"http://127.0.0.1:{port}/v1/models", f"http://[::1]:{port}/v1/models"]:
            try:
                urllib.request.urlopen(url, timeout=2)
                server_running = True
                logger.info("Using existing LLM server on port %d", port)
                break
            except Exception:
                pass
        if not server_running:
            server_proc = _start_server(cfg)
            if not gpus:
                gpus = _gpu_info()

    sig_dir = _signal_dir(cfg) if multi_node else None

    try:
        # Phase 1: offline stages (rank 0 only)
        if node_rank == 0:
            for ds_name in datasets:
                stages = ds_stage_map[ds_name]
                ds_cfg = _with_dataset(cfg, ds_name)
                ds = DatasetDir(root=cfg.db_root, name=ds_name)
                for stage in stages:
                    if stage in LLM_STAGES:
                        break
                    if _should_skip(stage, ds, skip_existing=cfg.get("skip_existing", False)):
                        logger.info("=== %s / %s: skipped ===", ds_name, stage)
                        continue
                    logger.info("=== %s / %s ===", ds_name, stage)
                    STAGE_RUNNERS[stage](ds_cfg)

        if multi_node:
            _signal_done(sig_dir, "phase1", node_rank)
            _wait_all(sig_dir, "phase1", num_nodes)

        # Phase 2+: LLM stages (all nodes, sharded if multi-node)
        llm_t0 = time.time()
        phase_times: dict[str, float] = {}

        llm_stage_config = {
            "enrich_corpus": (_run_enrich_corpus, _run_merge_corpus_shards),
            "enrich_query": (_run_enrich_query, _run_merge_query_shards),
            "greprag": (_run_greprag, _run_merge_greprag_shards),
            "iterative": (_run_iterative, _run_merge_iterative_shards),
            "rerank": (_run_rerank, _run_merge_rerank_shards),
        }

        for ds_name in datasets:
            t_ds = time.time()
            llm_stages = [
                s for s in ALL_STAGES
                if s in ds_stage_map[ds_name] and s in llm_stage_config
            ]
            for stage in llm_stages:
                run_fn, merge_fn = llm_stage_config[stage]

                ds_obj = DatasetDir(root=cfg.db_root, name=ds_name)
                if _should_skip(stage, ds_obj, skip_existing=cfg.get("skip_existing", False)):
                    logger.info("=== %s / %s: skipped (results exist) ===", ds_name, stage)
                    continue

                ds_cfg = _with_dataset(cfg, ds_name)
                if multi_node:
                    resumable = _find_resumable_run(ds_obj, stage, num_nodes)
                    if resumable:
                        run_name = resumable
                        logger.info("Resuming %s/%s from run %s", ds_name, stage, run_name)
                    else:
                        run_name = f"{ds_name}-{job_name}"
                    ds_cfg = _with_overrides(
                        ds_cfg,
                        shard_rank=node_rank,
                        num_shards=num_nodes,
                        run_name=run_name,
                    )
                logger.info(
                    "=== %s / %s (node %d/%d) ===",
                    ds_name,
                    stage,
                    node_rank,
                    num_nodes,
                )
                run_fn(ds_cfg)

                if multi_node:
                    sig_label = f"{ds_name}-{stage}"
                    _signal_done(sig_dir, sig_label, node_rank)
                    logger.info("Node %d: %s/%s done, waiting...", node_rank, ds_name, stage)
                    _wait_all(sig_dir, sig_label, num_nodes)

                    if node_rank == 0:
                        ds_cfg = _with_dataset(cfg, ds_name)
                        ds_cfg = _with_overrides(
                            ds_cfg,
                            merge_shards=True,
                            num_shards=num_nodes,
                            run_name=run_name,
                        )
                        logger.info("=== %s / %s merge ===", ds_name, stage)
                        merge_fn(ds_cfg)

                    merge_label = f"{ds_name}-{stage}-merge"
                    if node_rank == 0:
                        _signal_done(sig_dir, merge_label, 0)
                    else:
                        while not os.path.exists(os.path.join(sig_dir, f"{merge_label}-0-done")):
                            time.sleep(5)

            ds_elapsed = time.time() - t_ds
            phase_times[ds_name] = ds_elapsed
            if node_rank == 0:
                logger.info("=== %s done in %.1fs (%.1f min) ===", ds_name, ds_elapsed, ds_elapsed / 60)

        # Timing summary + save (rank 0 only)
        if node_rank == 0:
            llm_elapsed = time.time() - llm_t0
            logger.info("=== Timing summary ===")
            for phase, elapsed in phase_times.items():
                logger.info("  %-15s %7.1fs  (%4.1f min)", phase, elapsed, elapsed / 60)
            logger.info("  %-15s %7.1fs  (%4.1f min)", "TOTAL", llm_elapsed, llm_elapsed / 60)

            summary = {
                "datasets": datasets,
                "stages": {ds: ds_stage_map[ds] for ds in datasets},
                "num_nodes": num_nodes,
                "gpus_per_node": len(gpus),
                "total_gpus": num_nodes * len(gpus),
                "job_name": job_name,
                "model": cfg.sglang.model,
                "gpu": gpus,
                "hostname": platform.node(),
                "timing_s": {**phase_times, "total": round(llm_elapsed, 1)},
                "timestamp": int(time.time()),
            }
            summary_path = os.path.join(cfg.db_root, "pipeline-runs")
            os.makedirs(summary_path, exist_ok=True)
            summary_file = os.path.join(
                summary_path, f"{job_name}-{int(time.time())}.json"
            )
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info("Summary saved to %s", summary_file)
            logger.info("Pipeline complete for %s", datasets)

        # Multi-node: signal completion and wait for all nodes to ack
        if multi_node:
            if node_rank == 0:
                _signal_done(sig_dir, "all", 0)
            else:
                logger.info("Node %d: waiting for pipeline completion...", node_rank)
                while not os.path.exists(os.path.join(sig_dir, "all-0-done")):
                    time.sleep(5)
            _signal_done(sig_dir, "ack", node_rank)
            if node_rank == 0:
                _wait_all(sig_dir, "ack", num_nodes)

    finally:
        _stop_server(server_proc)
        if multi_node and node_rank == 0 and sig_dir:
            shutil.rmtree(sig_dir, ignore_errors=True)
        # Sync DeepGEMM cache back to shared storage (rank 0 only)
        if node_rank == 0:
            local_dg = os.environ.get("SGLANG_DG_CACHE_DIR", "")
            shared_dg = os.environ.get("SIRA_SHARED_DG_CACHE", "")
            if local_dg and shared_dg and os.path.isdir(os.path.join(local_dg, "cache")):
                os.makedirs(os.path.join(shared_dg, "cache"), exist_ok=True)
                for f in os.listdir(os.path.join(local_dg, "cache")):
                    src = os.path.join(local_dg, "cache", f)
                    dst = os.path.join(shared_dg, "cache", f)
                    if not os.path.exists(dst):
                        if os.path.isdir(src):
                            shutil.copytree(src, dst)
                        else:
                            shutil.copy2(src, dst)
                logger.info("Synced DeepGEMM cache to shared storage")


if __name__ == "__main__":
    main()
