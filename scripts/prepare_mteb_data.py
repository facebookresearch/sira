# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Download and prepare an MTEB dataset for retrieval evaluation.

Downloads corpus/queries/qrels from HuggingFace, filters by corpus
validity and minimum query length, writes per-split JSONL files,
and runs data quality checks.

Usage::

    source sandbox.sh
    python scripts/prepare_mteb_data.py
    python scripts/prepare_mteb_data.py data=fiqa

Config groups::

    global  — db_root (shared across all scripts)
    data/   — dataset name, HF repo, query filtering, k_values

Outputs under ``{db_root}/{data.name}/raw/``::

    corpus.jsonl, queries-{split}.jsonl, qrels-{split}.jsonl, metadata.json
"""

import json
import logging
import os

import hydra
import polars as pl
from huggingface_hub import snapshot_download
from omegaconf import DictConfig
from sira.schema.mteb import (
    COL_CORPUS_ID,
    COL_ID,
    COL_QUERY_ID,
    COL_SCORE,
    COL_TEXT,
    CORPUS_SCHEMA,
    DatasetDir,
    QUERY_SCHEMA,
)

logger = logging.getLogger(__name__)


def _ds(cfg: DictConfig) -> DatasetDir:
    return DatasetDir(root=cfg.db_root, name=cfg.data.name)


def download(cfg: DictConfig) -> None:
    """Download corpus/queries/qrels from HuggingFace."""
    ds = _ds(cfg)
    if os.path.exists(ds.metadata):
        logger.info("Already prepared at %s — skipping download.", ds.dir)
        return

    logger.info("Downloading %s …", cfg.data.repo)
    snap = snapshot_download(
        cfg.data.repo,
        repo_type="dataset",
        allow_patterns=["corpus.jsonl", "queries.jsonl", "qrels/*.jsonl"],
    )

    corpus = pl.read_ndjson(os.path.join(snap, "corpus.jsonl"), schema=CORPUS_SCHEMA)
    logger.info("Corpus: %d documents", corpus.height)
    os.makedirs(ds.raw_dir, exist_ok=True)
    corpus.write_ndjson(ds.corpus)

    corpus_ids = corpus.get_column(COL_ID)
    queries = pl.read_ndjson(os.path.join(snap, "queries.jsonl"), schema=QUERY_SCHEMA)

    qrels_dir = os.path.join(snap, "qrels")
    splits = [
        f.removesuffix(".jsonl")
        for f in sorted(os.listdir(qrels_dir))
        if f.endswith(".jsonl")
    ]
    qrels = pl.concat(
        [
            pl.read_ndjson(os.path.join(qrels_dir, f"{s}.jsonl"))
            .select(
                pl.col(COL_QUERY_ID).cast(pl.String),
                pl.col(COL_CORPUS_ID).cast(pl.String),
                pl.col(COL_SCORE).cast(pl.Int32),
            )
            .with_columns(pl.lit(s).alias("split"))
            for s in splits
        ]
    )

    corpus_id_list = corpus_ids.to_list()
    qrels = qrels.filter(pl.col(COL_CORPUS_ID).is_in(corpus_id_list))
    valid_qids = qrels.select(COL_QUERY_ID).unique()
    queries = queries.filter(
        pl.col(COL_TEXT).str.strip_chars().str.len_chars() >= cfg.data.min_query_len
    ).join(valid_qids, left_on=COL_ID, right_on=COL_QUERY_ID, how="semi")
    qrels = qrels.join(
        queries.select(COL_ID), left_on=COL_QUERY_ID, right_on=COL_ID, how="semi"
    )

    stats: dict[str, dict] = {}
    for (split,), s_qrels in qrels.partition_by("split", as_dict=True).items():
        s_qrels = s_qrels.select(COL_QUERY_ID, COL_CORPUS_ID, COL_SCORE)
        s_queries = queries.join(
            s_qrels.select(COL_QUERY_ID).unique(),
            left_on=COL_ID,
            right_on=COL_QUERY_ID,
            how="semi",
        )
        s_queries.write_ndjson(ds.queries(split))
        s_qrels.write_ndjson(ds.qrels(split))
        stats[split] = {"num_queries": s_queries.height, "num_qrels": s_qrels.height}
        logger.info(
            "[%s] %d queries, %d qrels", split, s_queries.height, s_qrels.height
        )

    metadata = {
        "name": cfg.data.name,
        "source": cfg.data.repo,
        "num_corpus": corpus.height,
        "splits": stats,
    }
    with open(ds.metadata, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Download done → %s", ds.dir)

    for split in stats:
        report = ds.validate(split)
        if not report.ok:
            raise RuntimeError(f"Data quality check failed for {ds.name}/{split}")


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="prepare_mteb_data",
)
def main(cfg: DictConfig) -> None:
    download(cfg)


if __name__ == "__main__":
    main()
