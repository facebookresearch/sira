<!-- Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree. -->
# Scripts

All scripts run from the repo root with `source sandbox.sh` first.

## Pipeline

`run_pipeline.py` is the single entry point for local and MAST runs.

```bash
# Local — single dataset
python scripts/run_pipeline.py data=scifact server.auto_start=true

# Local — multiple datasets, shared server
python scripts/run_pipeline.py datasets='[scifact,arguana]' server.auto_start=true

# Local — specific stages only
python scripts/run_pipeline.py data=fiqa stages='[enrich_query,rerank]'

# Local — manual server (start separately with serve_llm.py)
python scripts/run_pipeline.py data=scifact
```

Stages: `prepare` → `bm25` → `enrich_corpus` → `enrich_query` → `rerank`

### Multi-node

Multi-node is auto-detected from `RANK`/`WORLD_SIZE` environment variables.
Each node starts its own sglang server and processes its shard of `enrich_corpus`.
Node 0 merges shards, then runs `enrich_query` and `rerank`.

### Timing & hardware

Each pipeline run saves a summary JSON to `{db_root}/pipeline-runs/`:

```json
{
  "datasets": ["scifact", "arguana"],
  "num_nodes": 2,
  "model": "Qwen/Qwen3.6-35B-A3B-FP8",
  "gpu": [{"name": "NVIDIA H100 80GB HBM3", "memory_mb": 81079}, ...],
  "timing_s": {"enrich_corpus": 131.2, "enrich_query": 20.5, "rerank": 469.1, "total": 621.0},
  ...
}
```

## Individual Scripts

Each script can be run standalone (has its own `@hydra.main`):

| Script | Purpose | Example |
|---|---|---|
| `prepare_mteb_data.py` | Download MTEB dataset, run quality checks | `python scripts/prepare_mteb_data.py data=scifact` |
| `eval_bm25.py` | Build BM25 indices, evaluate, symlink best | `python scripts/eval_bm25.py data=scifact` |
| `add_doc_index_adapter.py` | LLM doc enrichment → DF filter → evaluate | `python scripts/add_doc_index_adapter.py data=scifact` |
| `enrich_query_and_retrieve.py` | LLM query expansion → weighted retrieval → evaluate | `python scripts/enrich_query_and_retrieve.py data=scifact` |
| `llm_reranking.py` | LLM pointwise reranking → evaluate | `python scripts/llm_reranking.py data=scifact` |
| `serve_llm.py` | Launch sglang server (tmux background mode) | `python scripts/serve_llm.py` |

## Config Layout

```
configs/
├── global.yaml                          # db_root, selection_metric, selection_split
├── run_pipeline.yaml                    # stages, datasets, server, hydra overrides
├── {script_name}.yaml                   # per-script Hydra configs
├── data/                                # dataset configs
│   ├── scifact.yaml, fiqa.yaml, ...
├── bm25/default.yaml                    # max_n_candidates, tokenizer, k1, b
├── enrich/
│   ├── default.yaml                     # concurrency, max_tokens, max_df_ratio
│   └── prompts/doc_v06.txt, query_v06.txt
├── rerank/
│   ├── default.yaml                     # top_n, concurrency, max_tokens
│   └── prompts/relevance_v02.txt
└── sglang/
    └── qwen3.6-35b-a3b-fp8:h100.yaml   # model, tp, dp, mem, context_length
```

## Data Layout

```
{db_root}/
├── pipeline-runs/                         # pipeline run summaries
│   └── {job_name}-{timestamp}.json        #   timing, GPU, nodes, model
│
└── {dataset}/
    ├── raw/                               # [prepare] downloaded data
    │   ├── corpus.jsonl
    │   ├── queries-{split}.jsonl
    │   ├── qrels-{split}.jsonl
    │   └── metadata.json
    ├── index/                             # [bm25] BM25 indices
    │   ├── bm25-n{X}-{tokenizer}/
    │   ├── best -> ...
    │   └── best.meta.json
    ├── eval/                              # metrics per stage
    │   ├── baseline/
    │   ├── doc-enrich/
    │   ├── query-enrich/
    │   └── rerank/
    │       └── {run_name}.json, best.json -> ..., best.meta.json
    ├── enrichments/                       # LLM-generated phrases
    │   ├── doc/{run_name}.json            #   {doc_id: [phrases]}
    │   └── query/{run_name}.json          #   {query_id: [phrases]}
    │       └── best.json -> ..., best.meta.json
    ├── retrieval/                         # BM25 candidates for reranker
    │   ├── {run_name}.jsonl               #   per-query top-K
    │   └── best.jsonl -> ...
    └── runs/                              # detailed per-run artifacts
        ├── doc-enrich/{run_name}/
        │   ├── config.json, prompt.txt
        │   ├── enrichments.kept.json, enrichments.failed.json
        │   ├── metrics.json, stats.json
        │   └── trace.kept.jsonl, trace.failed.jsonl
        ├── query-enrich/{run_name}/
        │   └── (same structure)
        └── rerank/{run_name}/
            ├── config.json, prompt.txt
            ├── reranked.jsonl, metrics.json, stats.json
            └── trace.kept.jsonl, trace.failed.jsonl
```

Best symlinks are updated automatically, and each downstream stage reads `best.*`
from the previous stage. Selection never looks at the evaluation split: where a
config is swept (currently only the BM25 n-gram order), candidates are ranked on
`selection_split` (default `dev`, falling back to `train`) by `selection_metric`
(default `Recall@10`). Datasets that ship no held-out split get no sweep at all —
the fixed `bm25.max_n` is used instead. The remaining stages compute no held-out
metrics, so their latest run simply becomes `best`. Each `best.meta.json` records
`eval_split`, `selection_split`, and the `selection_score` it was chosen on.
