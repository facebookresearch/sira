# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass

import polars as pl

logger = logging.getLogger(__name__)

# JSONL column names (BEIR / MTEB convention)
COL_ID = "_id"
COL_TITLE = "title"
COL_TEXT = "text"
COL_QUERY_ID = "query-id"
COL_CORPUS_ID = "corpus-id"
COL_SCORE = "score"

# Polars schemas for reading JSONL files
CORPUS_SCHEMA = pl.Schema({COL_ID: pl.String, COL_TITLE: pl.String, COL_TEXT: pl.String})
QUERY_SCHEMA = pl.Schema({COL_ID: pl.String, COL_TEXT: pl.String})
QREL_SCHEMA = pl.Schema({COL_QUERY_ID: pl.String, COL_CORPUS_ID: pl.String, COL_SCORE: pl.Int32})


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CorpusItem:
    id: str
    title: str
    text: str

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.text}" if self.title else self.text

    @staticmethod
    def from_row(row: dict[str, str]) -> "CorpusItem":
        return CorpusItem(id=row[COL_ID], title=row[COL_TITLE], text=row[COL_TEXT])


@dataclass
class QueryItem:
    id: str
    text: str


@dataclass
class QrelItem:
    query_id: str
    corpus_id: str
    score: int


@dataclass
class GroupedQrelItem:
    query_id: str
    corpus_ids: list[str]
    scores: list[int]


# ---------------------------------------------------------------------------
# ID index builder (shared by stores)
# ---------------------------------------------------------------------------


def _build_id_index(df: pl.DataFrame, col: str = COL_ID) -> dict[str, int]:
    """Map ID values to row indices. Warns on duplicates (last-writer wins)."""
    ids = df[col].to_list()
    out = dict(zip(ids, range(len(ids))))
    n_dup = len(ids) - len(out)
    if n_dup > 0:
        seen: set[str] = set()
        dups: list[str] = []
        for did in ids:
            if did in seen and did not in dups:
                dups.append(did)
                if len(dups) >= 5:
                    break
            seen.add(did)
        logger.warning(
            "Column %r has %d duplicate values (e.g. %s). "
            "Last-row-wins; lookups at dropped indices return wrong data.",
            col, n_dup, dups,
        )
    return out


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def read_corpus(path: str) -> pl.DataFrame:
    """Read a corpus JSONL file."""
    return pl.read_ndjson(path, schema=CORPUS_SCHEMA)


def read_corpus_texts(path: str) -> tuple[list[str], list[str]]:
    """Read corpus and return (doc_ids, combined_texts).

    Combines ``title. text`` when title is non-empty, otherwise just ``text``.
    """
    corpus = read_corpus(path)
    doc_ids = corpus.get_column(COL_ID).to_list()
    texts = (
        corpus.select(
            pl.when(
                pl.col(COL_TITLE).is_not_null()
                & (pl.col(COL_TITLE).str.len_chars() > 0)
            )
            .then(pl.col(COL_TITLE) + ". " + pl.col(COL_TEXT).fill_null(""))
            .otherwise(pl.col(COL_TEXT).fill_null(""))
            .alias("combined")
        )
        .get_column("combined")
        .to_list()
    )
    return doc_ids, texts


def read_queries(path: str) -> pl.DataFrame:
    """Read a queries JSONL file."""
    return pl.read_ndjson(path, schema=QUERY_SCHEMA)


def read_qrels(path: str) -> pl.DataFrame:
    """Read a qrels JSONL file."""
    return pl.read_ndjson(path, schema=QREL_SCHEMA)


def load_qrels_dict(path: str) -> dict[str, dict[str, int]]:
    """Read qrels JSONL into BEIR-format dict: ``{qid: {did: score}}``."""
    df = read_qrels(path)
    out: dict[str, dict[str, int]] = {}
    for row in df.iter_rows(named=True):
        out.setdefault(row[COL_QUERY_ID], {})[row[COL_CORPUS_ID]] = row[COL_SCORE]
    return out


# ---------------------------------------------------------------------------
# Data stores (O(1) access by ID, with quality checks)
# ---------------------------------------------------------------------------


class CorpusStore:
    """O(1) access to corpus items by ID. Checks for duplicates on init."""

    def __init__(self, path: str) -> None:
        self._df = read_corpus(path)
        self._id_to_idx = _build_id_index(self._df)

    def __getitem__(self, id: str) -> CorpusItem:
        return CorpusItem.from_row(self._df.row(self._id_to_idx[id], named=True))

    def __contains__(self, id: str) -> bool:
        return id in self._id_to_idx

    def __len__(self) -> int:
        return len(self._df)

    def doc_ids(self) -> list[str]:
        return self._df[COL_ID].to_list()


class QueryStore:
    """O(1) access to query items by ID."""

    def __init__(self, path: str) -> None:
        self._df = read_queries(path)
        self._id_to_idx = _build_id_index(self._df)

    def __getitem__(self, id: str) -> QueryItem:
        row = self._df.row(self._id_to_idx[id], named=True)
        return QueryItem(id=row[COL_ID], text=row[COL_TEXT])

    def __contains__(self, id: str) -> bool:
        return id in self._id_to_idx

    def __len__(self) -> int:
        return len(self._df)


class QrelStore:
    """Access to qrel items. Supports flat or grouped-by-query modes."""

    def __init__(self, path: str, *, grouped: bool = False) -> None:
        self._grouped = grouped
        df = read_qrels(path)
        if grouped:
            self._df = df.group_by(COL_QUERY_ID).agg(COL_CORPUS_ID, COL_SCORE)
        else:
            self._df = df

    def __getitem__(self, idx: int) -> QrelItem | GroupedQrelItem:
        r = self._df.row(idx, named=True)
        if not self._grouped:
            return QrelItem(
                query_id=r[COL_QUERY_ID],
                corpus_id=r[COL_CORPUS_ID],
                score=r[COL_SCORE],
            )
        return GroupedQrelItem(
            query_id=r[COL_QUERY_ID],
            corpus_ids=r[COL_CORPUS_ID],
            scores=r[COL_SCORE],
        )

    def __len__(self) -> int:
        return len(self._df)


# ---------------------------------------------------------------------------
# Data quality validation
# ---------------------------------------------------------------------------


@dataclass
class DataQualityReport:
    """Summary of quality checks on a prepared dataset split."""

    dataset: str
    split: str
    num_corpus: int
    num_queries: int
    num_qrels: int
    duplicate_corpus_ids: int
    duplicate_query_ids: int
    orphan_qrel_queries: int
    orphan_qrel_docs: int
    negative_scores: int
    corpus_coverage: float

    @property
    def ok(self) -> bool:
        return (
            self.num_corpus > 0
            and self.num_queries > 0
            and self.num_qrels > 0
            and self.duplicate_corpus_ids == 0
            and self.duplicate_query_ids == 0
            and self.orphan_qrel_queries == 0
            and self.orphan_qrel_docs == 0
            and self.negative_scores == 0
        )


def validate_split(
    corpus_path: str,
    queries_path: str,
    qrels_path: str,
    dataset: str = "",
    split: str = "",
) -> DataQualityReport:
    """Run quality checks on a prepared corpus/queries/qrels triple."""
    corpus = read_corpus(corpus_path)
    queries = read_queries(queries_path)
    qrels = read_qrels(qrels_path)

    corpus_ids = set(corpus[COL_ID].to_list())
    query_ids = set(queries[COL_ID].to_list())

    qrel_qids = set(qrels[COL_QUERY_ID].to_list())
    qrel_dids = set(qrels[COL_CORPUS_ID].to_list())

    n_corpus = corpus.height
    n_queries = queries.height
    n_qrels = qrels.height

    dup_corpus = n_corpus - len(corpus_ids)
    dup_queries = n_queries - len(query_ids)
    orphan_queries = len(qrel_qids - query_ids)
    orphan_docs = len(qrel_dids - corpus_ids)
    neg_scores = int(qrels.filter(pl.col(COL_SCORE) < 0).height)
    coverage = len(qrel_dids & corpus_ids) / max(n_corpus, 1)

    report = DataQualityReport(
        dataset=dataset,
        split=split,
        num_corpus=n_corpus,
        num_queries=n_queries,
        num_qrels=n_qrels,
        duplicate_corpus_ids=dup_corpus,
        duplicate_query_ids=dup_queries,
        orphan_qrel_queries=orphan_queries,
        orphan_qrel_docs=orphan_docs,
        negative_scores=neg_scores,
        corpus_coverage=coverage,
    )

    if report.ok:
        logger.info(
            "[%s/%s] Quality OK: %d corpus, %d queries, %d qrels, %.1f%% coverage",
            dataset, split, n_corpus, n_queries, n_qrels, coverage * 100,
        )
    else:
        logger.warning(
            "[%s/%s] Quality issues: dup_corpus=%d, dup_queries=%d, "
            "orphan_qrel_queries=%d, orphan_qrel_docs=%d, neg_scores=%d",
            dataset, split, dup_corpus, dup_queries,
            orphan_queries, orphan_docs, neg_scores,
        )

    return report


# ---------------------------------------------------------------------------
# Dataset directory layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetDir:
    """Paths for a prepared MTEB dataset.

    File format convention:
      .json  = single object (config, metrics, metadata, eval)
      .jsonl = multi-record stream (corpus, queries, enrichments, retrieval, traces)

    Layout::

        {root}/{name}/
        ├── raw/                               # downloaded data
        │   ├── corpus.jsonl
        │   ├── queries-{split}.jsonl
        │   ├── qrels-{split}.jsonl
        │   └── metadata.json
        ├── index/                             # BM25 indices
        │   ├── bm25-n{ngrams}-{tok}/
        │   └── best -> bm25-n{best}-{tok}
        ├── eval/                              # metrics per stage
        │   ├── baseline/                      #   BM25 baseline
        │   │   ├── bm25-n{ngrams}-{tok}.json
        │   │   └── best.json, best.meta.json
        │   ├── doc-enrich/                    #   after doc enrichment
        │   │   └── best.json, best.meta.json
        │   ├── query-enrich/                  #   after doc + query enrichment
        │   │   └── best.json, best.meta.json
        │   ├── rerank/                        #   after LLM reranking
        │   │   └── best.json, best.meta.json
        │   ├── greprag/                      #   multi-query RRF retrieval
        │   │   └── best.json, best.meta.json
        │   └── iterative/                    #   iterative agentic retrieval
        │       └── best.json, best.meta.json
        ├── enrichments/                       # kept phrases (best only)
        │   ├── doc/                           #   JSONL: {doc_id, phrases} per line
        │   │   └── best.jsonl, best.meta.json
        │   └── query/                         #   JSONL: {query_id, phrases} per line
        │       └── best.jsonl, best.meta.json
        ├── retrieval/                         # per-query candidates
        │   ├── baseline.jsonl                 #   BM25 baseline retrieval
        │   ├── doc-enrich/                    #   after doc enrichment
        │   │   ├── best.jsonl -> {best-run}.jsonl
        │   │   └── best.meta.json
        │   └── query-enrich/                  #   after query enrichment (rerank input)
        │       ├── best.jsonl -> {best-run}.jsonl
        │       └── best.meta.json
        └── runs/                              # per-run artifacts
            ├── doc-enrich/{run}/
            │   ├── config.json, prompt.txt
            │   ├── enrichments.kept.jsonl
            │   ├── metrics.json, stats.json
            │   └── trace.kept.jsonl, trace.failed.jsonl
            ├── query-enrich/{run}/
            │   ├── config.json, query_prompt.txt
            │   ├── enrichments.kept.jsonl, enrichments.failed.jsonl
            │   ├── metrics.json, stats.json
            │   └── trace.kept.jsonl, trace.failed.jsonl
            ├── rerank/{run}/
            │   ├── config.json, prompt.txt
            │   ├── reranked.jsonl, metrics.json, stats.json
            │   └── trace.kept.jsonl, trace.failed.jsonl
            ├── greprag/{run}/
            │   ├── config.json, prompt.txt, metrics.json, stats.json
            │   └── trace.kept.jsonl, trace.failed.jsonl
            └── iterative/{run}/
                ├── config.json, prompt.txt, metrics.json, stats.json
                └── trace.kept.jsonl, trace.failed.jsonl
    """

    root: str
    name: str

    @property
    def dir(self) -> str:
        return os.path.join(self.root, self.name)

    # -- raw/ --

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.dir, "raw")

    @property
    def corpus(self) -> str:
        return os.path.join(self.raw_dir, "corpus.jsonl")

    @property
    def metadata(self) -> str:
        return os.path.join(self.raw_dir, "metadata.json")

    def queries(self, split: str) -> str:
        return os.path.join(self.raw_dir, f"queries-{split}.jsonl")

    def qrels(self, split: str) -> str:
        return os.path.join(self.raw_dir, f"qrels-{split}.jsonl")

    def has_split(self, split: str) -> bool:
        return os.path.exists(self.queries(split)) and os.path.exists(self.qrels(split))

    def resolve_selection_split(
        self, preferred: str, eval_split: str
    ) -> str | None:
        """Pick a held-out split for model selection, or None if there is none.

        Configuration choices (n-gram order, which enrichment or rerank run
        becomes ``best``) must never be made on the split we report, or the
        reported numbers are selection-biased. Tries ``preferred`` (normally
        ``dev``), then ``train``; returns None when the dataset ships only the
        evaluation split, in which case callers must disable selection rather
        than fall back to the evaluation split.
        """
        for candidate in (preferred, "train"):
            if candidate and candidate != eval_split and self.has_split(candidate):
                return candidate
        return None

    # -- index/ --

    @staticmethod
    def _bm25_tag(max_n: int = 4, tokenizer: str = "unicode_stem") -> str:
        ngrams = "".join(str(n) for n in range(1, max_n + 1))
        return f"bm25-n{ngrams}-{tokenizer}"

    def bm25_index(self, max_n: int = 4, tokenizer: str = "unicode_stem") -> str:
        return os.path.join(self.dir, "index", self._bm25_tag(max_n, tokenizer))

    @property
    def bm25_index_best(self) -> str:
        return os.path.join(self.dir, "index", "best")

    # -- enrichments/ --

    @property
    def doc_enrichments_dir(self) -> str:
        return os.path.join(self.dir, "enrichments", "doc")

    def doc_enrichments(self, run_name: str) -> str:
        return os.path.join(self.doc_enrichments_dir, f"{run_name}.jsonl")

    @property
    def doc_enrichments_best(self) -> str:
        return os.path.join(self.doc_enrichments_dir, "best.jsonl")

    @property
    def query_enrichments_dir(self) -> str:
        return os.path.join(self.dir, "enrichments", "query")

    def query_enrichments(self, run_name: str) -> str:
        return os.path.join(self.query_enrichments_dir, f"{run_name}.jsonl")

    @property
    def query_enrichments_best(self) -> str:
        return os.path.join(self.query_enrichments_dir, "best.jsonl")

    # -- runs/ --

    @property
    def runs_dir(self) -> str:
        return os.path.join(self.dir, "runs")

    @property
    def doc_enrich_runs_dir(self) -> str:
        return os.path.join(self.runs_dir, "doc-enrich")

    @property
    def query_enrich_runs_dir(self) -> str:
        return os.path.join(self.runs_dir, "query-enrich")

    # -- eval/ --

    def eval_baseline(self, max_n: int = 4, tokenizer: str = "unicode_stem") -> str:
        return os.path.join(self.dir, "eval", "baseline", f"{self._bm25_tag(max_n, tokenizer)}.json")

    @property
    def eval_baseline_best(self) -> str:
        return os.path.join(self.dir, "eval", "baseline", "best.json")

    def eval_doc_enrich(self, run_name: str) -> str:
        return os.path.join(self.dir, "eval", "doc-enrich", f"{run_name}.json")

    @property
    def eval_doc_enrich_best(self) -> str:
        return os.path.join(self.dir, "eval", "doc-enrich", "best.json")

    def eval_query_enrich(self, run_name: str) -> str:
        return os.path.join(self.dir, "eval", "query-enrich", f"{run_name}.json")

    @property
    def eval_query_enrich_best(self) -> str:
        return os.path.join(self.dir, "eval", "query-enrich", "best.json")

    # -- retrieval results --

    @property
    def retrieval_dir(self) -> str:
        return os.path.join(self.dir, "retrieval")

    def retrieval_stage_dir(self, stage: str) -> str:
        return os.path.join(self.retrieval_dir, stage)

    def retrieval_results(self, run_name: str, stage: str = "query-enrich") -> str:
        return os.path.join(self.retrieval_stage_dir(stage), f"{run_name}.jsonl")

    def retrieval_best(self, stage: str = "query-enrich") -> str:
        return os.path.join(self.retrieval_stage_dir(stage), "best.jsonl")

    # -- eval/rerank --

    def eval_rerank(self, run_name: str) -> str:
        return os.path.join(self.dir, "eval", "rerank", f"{run_name}.json")

    @property
    def eval_rerank_best(self) -> str:
        return os.path.join(self.dir, "eval", "rerank", "best.json")

    @property
    def rerank_runs_dir(self) -> str:
        return os.path.join(self.runs_dir, "rerank")

    # -- eval/greprag --

    def eval_greprag(self, run_name: str) -> str:
        return os.path.join(self.dir, "eval", "greprag", f"{run_name}.json")

    @property
    def eval_greprag_best(self) -> str:
        return os.path.join(self.dir, "eval", "greprag", "best.json")

    @property
    def greprag_runs_dir(self) -> str:
        return os.path.join(self.runs_dir, "greprag")

    # -- eval/iterative --

    def eval_iterative(self, run_name: str) -> str:
        return os.path.join(self.dir, "eval", "iterative", f"{run_name}.json")

    @property
    def eval_iterative_best(self) -> str:
        return os.path.join(self.dir, "eval", "iterative", "best.json")

    @property
    def iterative_runs_dir(self) -> str:
        return os.path.join(self.runs_dir, "iterative")

    # -- best symlink management --

    @staticmethod
    def update_best(
        best_links: list[str],
        target_name: str,
        meta: dict,
        *,
        selection_metric: str = "",
        selection_metrics: dict[str, float] | None = None,
    ) -> bool:
        """Point the ``best`` symlinks at this run, unless a held-out split says otherwise.

        Selection is only ever scored on a held-out split. When
        ``selection_metrics`` is None the dataset has no held-out split, so
        there is no legitimate basis for ranking runs against each other and
        the newest run simply wins — deliberately, so that re-running a stage
        cannot be used to cherry-pick whichever config scored best on the
        evaluation split.

        Args:
            best_links: list of best.* symlink paths to update.
            target_name: filename the symlinks should point to (e.g. "run-123.json").
            meta: metadata dict to write as best.meta.json next to each symlink.
                Any evaluation-split metrics belong here, for reporting only.
            selection_metric: key to compare within ``selection_metrics``.
            selection_metrics: metrics computed on the held-out selection split,
                or None to disable comparison (latest run wins).

        Returns:
            True if this run became the new best.
        """
        import json
        import tempfile

        if selection_metrics is None:
            meta["selection_basis"] = "latest (no held-out split available)"
            meta["selection_metric"] = None
            meta["selection_score"] = None
        else:
            current_score = selection_metrics.get(selection_metric, 0.0)
            prev_best_score = float("-inf")
            for link in best_links:
                meta_path = os.path.join(os.path.dirname(link), "best.meta.json")
                if not os.path.exists(meta_path):
                    continue
                with open(meta_path) as f:
                    prev = json.load(f).get("selection_score")
                if prev is not None:
                    prev_best_score = float(prev)
                break

            if current_score < prev_best_score:
                return False

            meta["selection_basis"] = "held-out split"
            meta["selection_metric"] = selection_metric
            meta["selection_score"] = current_score
            meta["prev_selection_score"] = (
                None if prev_best_score == float("-inf") else prev_best_score
            )

        for link in best_links:
            link_dir = os.path.dirname(link)
            os.makedirs(link_dir, exist_ok=True)

            prev_target = None
            if os.path.islink(link):
                prev_target = os.readlink(link)

            fd, tmp = tempfile.mkstemp(dir=link_dir)
            os.close(fd)
            os.remove(tmp)
            os.symlink(target_name, tmp)
            os.rename(tmp, link)

            if prev_target and prev_target != target_name:
                prev_file = os.path.join(link_dir, prev_target)
                if os.path.exists(prev_file):
                    os.remove(prev_file)

            if link.endswith((".json", ".jsonl")):
                meta_path = os.path.join(link_dir, "best.meta.json")
                fd, tmp = tempfile.mkstemp(dir=link_dir, suffix=".tmp")
                with os.fdopen(fd, "w") as f:
                    json.dump(meta, f, indent=2)
                os.rename(tmp, meta_path)

        return True

    # -- validation --

    def validate(self, split: str = "test") -> DataQualityReport:
        """Run quality checks on this dataset's given split."""
        return validate_split(
            self.corpus, self.queries(split), self.qrels(split),
            dataset=self.name, split=split,
        )
