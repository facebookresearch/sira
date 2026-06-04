# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end Wikipedia dump pipeline: download + parse XML + parse SQL + category index.

Mirrors the English Wikipedia dump for ``DUMP_DATE`` (default ``20260501``)
into ``localdb_dir() / "wikipedia-monthly" / <DATE> / "en"`` and converts the
article XML + 5 SQL tables to parquet for downstream polars use.

Usage (from the sira/ folder)::

    ./run.sh python3 scripts/prerequisites/prereq_wikidump.py
    ./run.sh python3 scripts/prerequisites/prereq_wikidump.py --probe

``pip install mwxml mwsql`` first if missing — but ONLY Phase 1-3 (download +
parse) import them. Phase 4-9 work off the parsed parquets, so re-running just
the derived phases on an already-populated dump needs neither package nor
network. Every phase skips its output if it already exists.

PIPELINE (9 phases; ~90 min full run)
-------------------------------------
1. DOWNLOAD   — fetch ``dumpstatus.json``; download 70 multistream XML shards
                + 5 SQL dumps to ``raw/`` (parallel HTTP, ``Range:`` resume).
2. PARSE      — XML (mwxml) and SQL (mwsql) pools run concurrently. mwsql is
                monkey-patched to open with ``errors='replace'`` (varbinary
                cols hold occasional non-UTF-8 bytes) and pigz for fast gunzip;
                pagelinks uses a vectorized regex fast path. Raw mwsql strings
                are cast to typed columns in batch via polars.
3. MERGE      — stream the per-shard XML parquets into ``pages.parquet``.
4. CATEGORY INDEX  — ``category_index.parquet``: one row per Category page with
                a BFS-derived ``level`` (distance from ``LEVEL_ROOTS`` via
                subcat edges; NULL = unreachable admin/hidden cat).
5. SUBCAT DAG      — ``category_dag.parquet``: long-form ``(child, parent)``
                subcat edges (delegates to ``WikiDump.write_full_category_dag``).
6. DOC CATEGORIES  — ``doc_categories.parquet``: long-form ``(page_id,
                category)`` ns=0 article tags (``WikiDump.write_doc_categories``).
7. CAT STRUCTURE   — ``cat_structure.parquet`` (per-cat direct subcats + pages)
                plus a seekable members blob (``cat_member_offsets.parquet`` +
                ``cat_members.bin``) for ~3 ms mmap member lookups.
8. PAGE CATEGORIES — ``page_categories.parquet``: all-namespace long-form
                ``(page_id, namespace, category)``. LEFT join onto pages keeps
                categorylinks rows for pages absent from the XML dump (Talk /
                User) with ``namespace = null``.
9. PAGES WITH CATEGORIES — ``pages_with_categories.parquet``: one row per page,
                denormalized (page_id, namespace, title, description,
                categories list, text). One ``scan_parquet`` for BM25 / RAG.

``--probe`` downloads + parses only the smallest XML shard and prints samples;
skips SQL, merge, and the derived phases. Artifacts persist for a full re-run.

ON-DISK LAYOUT
--------------
    <localdb>/wikipedia-monthly/<DATE>/en/
    ├── pages.parquet            — merged article corpus (XML, Phase 3)
    ├── categorylinks / redirect / page_props / pagelinks / linktarget.parquet
    │                            — from SQL (Phase 2)
    ├── category_index.parquet   — derived (Phase 4)
    ├── category_dag.parquet     — derived (Phase 5)
    ├── doc_categories.parquet   — derived (Phase 6)
    ├── cat_structure.parquet + cat_member_offsets.parquet + cat_members.bin
    │                            — derived (Phase 7)
    ├── page_categories.parquet  — derived (Phase 8)
    ├── pages_with_categories.parquet — derived (Phase 9)
    ├── raw/                     — original downloads (KEEP for re-parse)
    └── shards/                  — per-bz2 XML parquets (input to merge)

Only the top-level parquets are downstream artifacts; ``raw/`` and ``shards/``
are intermediates safe to delete if you don't need to re-parse.

KEY SCHEMAS
-----------
``pages.parquet`` (one row per page, ALL namespaces; filter ``namespace == 0``
for real articles):

    page_id:         Int64    -- MediaWiki page_id
    namespace:       Int64    -- 0=Main, 6=File, 10=Template, 14=Category, ...
    title:           Utf8     -- with spaces (not underscores)
    redirect_target: Utf8?    -- target title if redirect, else null
    revision_id:     Int64
    timestamp:       Utf8     -- ISO 8601
    text:            Utf8     -- raw MediaWiki wikitext

SQL parquets are 1:1 with their MediaWiki source tables (peek a raw dump with
``zcat raw/<file> | head -50`` for the CREATE TABLE). Modern MediaWiki (1.39+)
keeps link/category titles in ``linktarget``: both ``categorylinks.cl_target_id``
and ``pagelinks.pl_target_id`` are foreign keys into ``linktarget.lt_id``. To
resolve a category name, join on ``lt_id`` where ``lt_namespace == 14``. Title
strings keep the DB underscore form; convert downstream if needed.

To add a SQL table, append its suffix to ``SQL_FILE_SUFFIXES`` — its parquet is
named ``<table>.parquet`` automatically.

Source: https://dumps.wikimedia.org/enwiki/
License: CC-BY-SA-4.0 (article text); CC0 (database/metadata).
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import io
import re
import shutil
import subprocess
import time
from collections import deque
from concurrent.futures import as_completed, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any

import polars as pl
import requests
from sira.src.python.thirdeye.config import localdb_dir
from sira.src.python.thirdeye.data.wikidump import DEFAULT_DUMP_DATE, DEFAULT_LANG
from tqdm import tqdm

# mwsql / mwxml are only needed by Phase 1-3 (download + XML/SQL parse).
# Phase 4-9 work off the already-parsed parquets and don't import either.
# Imported lazily inside parse_xml_shard / parse_sql_file / _ensure_mwsql_patched
# so users running just the derived phases don't need them in their conda env.

# ============================================================================
# CONSTANTS
# ============================================================================

DUMP_DATE = DEFAULT_DUMP_DATE
WIKI = "enwiki"
LANG = DEFAULT_LANG
DUMP_HOST = "https://dumps.wikimedia.org"
DUMPSTATUS_URL = f"{DUMP_HOST}/{WIKI}/{DUMP_DATE}/dumpstatus.json"
ARTICLES_JOB = "articlesmultistreamdump"

SQL_FILE_SUFFIXES: tuple[str, ...] = (
    f"-{DUMP_DATE}-categorylinks.sql.gz",
    f"-{DUMP_DATE}-redirect.sql.gz",
    f"-{DUMP_DATE}-page_props.sql.gz",
    f"-{DUMP_DATE}-pagelinks.sql.gz",
    f"-{DUMP_DATE}-linktarget.sql.gz",
)

PARALLEL_DOWNLOADS = 4
PARALLEL_XML_PARSERS = 8
PARALLEL_SQL_PARSERS = 5  # one per SQL file; they run independently
DOWNLOAD_CHUNK = 1 << 20  # 1 MiB
HTTP_TIMEOUT = (60, 600)  # (connect, read)
SQL_BATCH = 2_000_000  # rows per intermediate parquet chunk

CATEGORY_NAMESPACE = 14
LEVEL_ROOTS: tuple[str, ...] = (
    "Main_topic_classifications",
    "Fundamental_categories",
    "Articles",
)

PAGE_SCHEMA = pl.Schema(
    {
        "page_id": pl.Int64,
        "namespace": pl.Int64,
        "title": pl.Utf8,
        "redirect_target": pl.Utf8,
        "revision_id": pl.Int64,
        "timestamp": pl.Utf8,
        "text": pl.Utf8,
    }
)


def _parquet_len(path: Path) -> int:
    """Row count of a parquet via a metadata-only lazy scan."""
    return pl.scan_parquet(path).select(pl.len()).collect().item()


# ============================================================================
# mwsql MONKEY-PATCH: tolerate non-UTF-8 bytes in varbinary columns
# ============================================================================


_PIGZ = shutil.which("pigz")


@contextmanager
def _binary_safe_open_file(file_path, encoding=None):
    """Patched ``mwsql._open_file`` — uses pigz subprocess + ``errors='replace'``.

    Two fixes over mwsql's default:

    1. ``errors='replace'`` instead of strict UTF-8. MediaWiki ``varbinary``
       columns (e.g. ``cl_to``) hold raw bytes — usually UTF-8 but occasionally
       legacy non-UTF-8 sequences. Strict mode crashes on the first bad byte;
       replace turns it into U+FFFD so the parse runs to completion.

    2. ``pigz -dc`` (parallel gunzip via subprocess) instead of Python's
       single-threaded ``gzip`` module. pigz uses N threads internally and is
       ~5x faster on multi-core machines (Python gzip ≈ 30 MB/s, pigz on 4
       threads ≈ 150 MB/s). Falls back to Python ``gzip`` if pigz is missing.
    """
    enc = encoding or "utf-8"
    is_gz = str(file_path).endswith(".gz")
    if is_gz and _PIGZ:
        proc = subprocess.Popen([_PIGZ, "-dc", str(file_path)], stdout=subprocess.PIPE)
        assert proc.stdout is not None
        fh = io.TextIOWrapper(proc.stdout, encoding=enc, errors="replace")
        try:
            yield fh
        finally:
            fh.close()
            proc.terminate()
            proc.wait()
    else:
        # No pigz (or a plain, non-gz file): single-threaded text reader. Both
        # gzip.open and the builtin open accept mode="rt" + errors="replace".
        opener = gzip.open if is_gz else open
        fh = opener(file_path, mode="rt", encoding=enc, errors="replace")
        try:
            yield fh
        finally:
            fh.close()


def _ensure_mwsql_patched() -> None:
    """Apply the binary-safe ``_open_file`` monkey patch to mwsql.

    Idempotent (Python caches the import; reassigning the same callable
    is a no-op) and lazy — only called from ``parse_sql_file``, so
    Phase 4-6 never trigger the mwsql import.
    """
    import mwsql.dump
    import mwsql.utils

    mwsql.utils._open_file = _binary_safe_open_file
    mwsql.dump._open_file = _binary_safe_open_file


# ============================================================================
# MANIFEST
# ============================================================================


def fetch_status() -> dict:
    print(f"Fetching dump manifest: {DUMPSTATUS_URL}")
    resp = requests.get(DUMPSTATUS_URL, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_multistream_files(status: dict) -> list[tuple[str, int]]:
    """Return ``[(url, size)]`` for the multistream split bz2 files."""
    job = status.get("jobs", {}).get(ARTICLES_JOB)
    if job is None:
        raise RuntimeError(f"Job '{ARTICLES_JOB}' not found in {DUMPSTATUS_URL}")
    if job.get("status") != "done":
        raise RuntimeError(
            f"Job '{ARTICLES_JOB}' status is {job.get('status')!r} — not ready"
        )
    out: list[tuple[str, int]] = []
    for name, meta in job.get("files", {}).items():
        if "multistream-index" in name or not name.endswith(".bz2"):
            continue
        out.append((f"{DUMP_HOST}{meta['url']}", int(meta["size"])))
    if not out:
        raise RuntimeError(f"No multistream files in {DUMPSTATUS_URL}")
    out.sort()
    return out


def find_file_by_suffix(status: dict, suffix: str) -> tuple[str, int]:
    """Locate a single file ending in ``suffix`` across all completed jobs."""
    for job in status.get("jobs", {}).values():
        if job.get("status") != "done":
            continue
        for name, meta in job.get("files", {}).items():
            if name.endswith(suffix):
                return f"{DUMP_HOST}{meta['url']}", int(meta["size"])
    raise RuntimeError(f"No file ending in {suffix} found in {DUMPSTATUS_URL}")


# ============================================================================
# PHASE 1: DOWNLOAD
# ============================================================================


def _download_one(
    url: str, expected_size: int, dest: Path, pbar: tqdm, lock: Lock
) -> None:
    have = dest.stat().st_size if dest.exists() else 0
    if have > expected_size:
        dest.unlink()
        have = 0
    if have == expected_size:
        with lock:
            pbar.update(have)
        return
    with lock:
        pbar.update(have)
    headers = {"Range": f"bytes={have}-"} if have else {}
    mode = "ab" if have else "wb"
    with requests.get(url, stream=True, headers=headers, timeout=HTTP_TIMEOUT) as r:
        r.raise_for_status()
        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                with lock:
                    pbar.update(len(chunk))
    actual = dest.stat().st_size
    if actual != expected_size:
        raise IOError(
            f"Size mismatch for {dest.name}: got {actual}, expected {expected_size}"
        )


def download_all(files: list[tuple[str, int]], raw_dir: Path) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = sum(s for _, s in files)
    paths: list[Path] = []
    pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc="download")
    lock = Lock()
    with ThreadPoolExecutor(max_workers=PARALLEL_DOWNLOADS) as ex:
        futs = {}
        for url, size in files:
            dest = raw_dir / url.rsplit("/", 1)[-1]
            paths.append(dest)
            futs[ex.submit(_download_one, url, size, dest, pbar, lock)] = dest
        for fut in as_completed(futs):
            fut.result()
    pbar.close()
    return paths


# ============================================================================
# PHASE 2a: XML PARSE (mwxml + bz2.open + ProcessPool)
# ============================================================================


def parse_xml_shard(bz2_path: Path, parquet_path: Path) -> int:
    """Stream one bz2 XML shard via mwxml into a typed parquet (all namespaces)."""
    if parquet_path.exists():
        return _parquet_len(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    import mwxml

    rows: list[dict[str, Any]] = []
    with bz2.open(bz2_path, "rb") as fh:
        dump = mwxml.Dump.from_file(fh)
        for page in dump:
            # page.redirect is the target title as a plain str (or None).
            redirect_target = page.redirect if page.redirect else None
            rev = next(iter(page), None)  # articles dump has one current revision
            rows.append(
                {
                    "page_id": page.id,
                    "namespace": page.namespace,
                    "title": page.title,
                    "redirect_target": redirect_target,
                    "revision_id": rev.id if rev else None,
                    "timestamp": (
                        str(rev.timestamp) if rev and rev.timestamp else None
                    ),
                    "text": (rev.text or "") if rev else "",
                }
            )

    df = pl.DataFrame(rows, schema=PAGE_SCHEMA)
    df.write_parquet(parquet_path)
    return df.height


def parse_xml_all(bz2_paths: list[Path], shard_dir: Path) -> list[Path]:
    """Parse all XML shards in parallel; return per-shard parquet paths."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [shard_dir / (bz2p.stem + ".parquet") for bz2p in bz2_paths]
    print(f"\n[XML] parsing {len(bz2_paths)} shards ({PARALLEL_XML_PARSERS} workers)")
    with ProcessPoolExecutor(max_workers=PARALLEL_XML_PARSERS) as ex:
        futs = {
            ex.submit(parse_xml_shard, bz2p, shp): (bz2p, shp)
            for bz2p, shp in zip(bz2_paths, shard_paths)
        }
        done = 0
        for fut in as_completed(futs):
            bz2p, shp = futs[fut]
            n = fut.result()
            done += 1
            print(f"[XML {done}/{len(bz2_paths)}] {bz2p.name}: {n:,} pages")
    return shard_paths


# ============================================================================
# PHASE 2b: SQL PARSE (mwsql + ProcessPool)
# ============================================================================

_PL_FOR_PY: dict[type, type[pl.DataType]] = {
    int: pl.Int64,
    float: pl.Float64,
    str: pl.Utf8,
}

# Fast-path regex for pagelinks: 3-int tuples, no strings/NULLs/escapes.
# Schema-verified safe: pagelinks CREATE TABLE declares all 3 fields NOT NULL int.
_PAGELINKS_TUPLE_RE = re.compile(rb"\((\d+),(-?\d+),(\d+)\)")
_PAGELINKS_INSERT_PREFIX = b"INSERT INTO `pagelinks` VALUES "
_PAGELINKS_SCHEMA = pl.Schema(
    {
        "pl_from": pl.Int64,
        "pl_from_namespace": pl.Int64,
        "pl_target_id": pl.Int64,
    }
)
_PAGELINKS_BATCH = 10_000_000  # bigger than SQL_BATCH; rows are tiny (3 ints)


def _sql_out_parquet(gz_path: Path, target_dir: Path) -> Path:
    """``enwiki-DATE-categorylinks.sql.gz`` -> ``categorylinks.parquet``."""
    name = gz_path.name.removeprefix(f"{WIKI}-{DUMP_DATE}-").removesuffix(".sql.gz")
    return target_dir / f"{name}.parquet"


def _write_chunks(
    chunk_paths: list[Path], out_parquet: Path, tmp_dir: Path, src_path: Path
) -> None:
    """Merge numbered chunk parquets into ``out_parquet`` and clean up ``tmp_dir``.

    Shared finalization for the streaming SQL parsers (``parse_sql_file`` and
    ``parse_pagelinks_fast``): streamed-merge the per-batch chunk parquets via
    ``scan_parquet().sink_parquet()``, then unlink each chunk and remove the
    now-empty tmp dir. Raises if no chunks were written — an empty parse is
    always a bug for these tables. ``src_path`` only feeds the error message.
    """
    if not chunk_paths:
        raise RuntimeError(f"No rows parsed from {src_path}")
    pl.scan_parquet(chunk_paths).sink_parquet(out_parquet)
    for p in chunk_paths:
        p.unlink()
    tmp_dir.rmdir()


def parse_sql_file(gz_path: Path, out_parquet: Path) -> int:
    """Stream one MediaWiki SQL dump into a typed parquet via mwsql.

    Cast is done in polars (vectorized C++) rather than per-row Python — we
    collect raw mwsql ``list[str]`` rows into batches, build a string-typed
    DataFrame, then ``.cast()`` the whole column at once. ~5-10x faster than
    Python ``int(v)`` per cell.
    """
    if out_parquet.exists():
        n = _parquet_len(out_parquet)
        print(f"[SQL] {out_parquet.name}: already exists ({n:,} rows) — skipping")
        return n

    _ensure_mwsql_patched()
    from mwsql import Dump

    dump = Dump.from_file(str(gz_path))
    print(f"[SQL] {gz_path.name} -> {out_parquet.name}")
    print(f"      cols: {dump.col_names}")

    cols = list(dump.col_names)
    py_dtypes = dump.dtypes
    str_schema = pl.Schema({c: pl.Utf8 for c in cols})
    # For numeric columns, cast strict=False so empty strings ('' = NULL) become null.
    cast_exprs = [
        pl.col(c).cast(_PL_FOR_PY[py_dtypes[c]], strict=False)
        for c in cols
        if py_dtypes[c] is not str
    ]

    tmp_dir = out_parquet.parent / f"_chunks_{out_parquet.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    batch: list[list[str]] = []
    chunk_paths: list[Path] = []
    total = 0

    def flush() -> None:
        nonlocal batch, total
        if not batch:
            return
        df = pl.DataFrame(batch, schema=str_schema, orient="row")
        if cast_exprs:
            df = df.with_columns(*cast_exprs)
        chunk_path = tmp_dir / f"chunk_{len(chunk_paths):04d}.parquet"
        df.write_parquet(chunk_path)
        chunk_paths.append(chunk_path)
        total += len(batch)
        batch = []

    for row in dump.rows():
        batch.append(row)  # keep raw strings; cast in batch at flush time
        if len(batch) >= SQL_BATCH:
            flush()
    flush()

    _write_chunks(chunk_paths, out_parquet, tmp_dir, gz_path)
    print(f"[SQL] {out_parquet.name}: {total:,} rows done")
    return total


def parse_pagelinks_fast(gz_path: Path, out_parquet: Path) -> int:
    """Vectorized pagelinks parser: bulk regex findall + polars batch construction.

    Bypasses mwsql's per-row Python generator (which dominates wall time at
    ~25 min for 1.5B rows) by reading each ``INSERT INTO pagelinks VALUES ...``
    line whole, running one ``re.findall`` (C-level) per line to extract ALL
    tuples at once, then constructing typed polars columns from string lists.
    ~3-5x faster than the generic mwsql path for this schema.

    Safe to use ONLY because pagelinks has a dead-simple schema (3 NOT NULL
    integers, no strings, no escapes, no NULLs). Verified empirically against
    10K INSERT lines of the actual dump: zero NULL/quote/backslash bytes.
    Other tables (categorylinks/page_props/redirect/linktarget) have strings
    and must keep using the mwsql path.
    """
    if out_parquet.exists():
        n = _parquet_len(out_parquet)
        print(f"[FAST] {out_parquet.name}: already exists ({n:,} rows) — skipping")
        return n

    print(f"[FAST] {gz_path.name} -> {out_parquet.name} (vectorized)")
    tmp_dir = out_parquet.parent / f"_chunks_{out_parquet.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    a: list[bytes] = []
    b: list[bytes] = []
    c: list[bytes] = []
    chunk_paths: list[Path] = []
    total = 0

    def flush() -> None:
        nonlocal a, b, c, total
        if not a:
            return
        df = pl.DataFrame(
            {"pl_from": a, "pl_from_namespace": b, "pl_target_id": c}
        ).with_columns(
            pl.col("pl_from").cast(pl.Int64),
            pl.col("pl_from_namespace").cast(pl.Int64),
            pl.col("pl_target_id").cast(pl.Int64),
        )
        chunk_path = tmp_dir / f"chunk_{len(chunk_paths):04d}.parquet"
        df.write_parquet(chunk_path)
        chunk_paths.append(chunk_path)
        total += df.height
        a, b, c = [], [], []

    if not _PIGZ:
        raise RuntimeError("pigz required for pagelinks fast path")
    proc = subprocess.Popen([_PIGZ, "-dc", str(gz_path)], stdout=subprocess.PIPE)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if not line.startswith(_PAGELINKS_INSERT_PREFIX):
                continue
            matches = _PAGELINKS_TUPLE_RE.findall(line)
            if not matches:
                continue
            # zip(*matches) is faster than 3 list comprehensions
            ai, bi, ci = zip(*matches, strict=True)
            a.extend(ai)
            b.extend(bi)
            c.extend(ci)
            if len(a) >= _PAGELINKS_BATCH:
                flush()
        flush()
    finally:
        proc.terminate()
        proc.wait()

    _write_chunks(chunk_paths, out_parquet, tmp_dir, gz_path)
    print(f"[FAST] {out_parquet.name}: {total:,} rows done")
    return total


# Per-suffix dispatcher: pagelinks gets the fast path; everything else mwsql.
_SQL_SPECIAL_PARSERS = {
    f"-{DUMP_DATE}-pagelinks.sql.gz": parse_pagelinks_fast,
}


def _dispatch_sql_parser(gz_path: Path, out_parquet: Path) -> int:
    for suffix, parser in _SQL_SPECIAL_PARSERS.items():
        if gz_path.name.endswith(suffix):
            return parser(gz_path, out_parquet)
    return parse_sql_file(gz_path, out_parquet)


def parse_sql_all(sql_paths: list[Path], target_dir: Path) -> list[Path]:
    """Parse all SQL dumps in parallel (one process per file).

    Dispatches each file to its parser: ``parse_pagelinks_fast`` (vectorized
    3-int regex) for pagelinks, ``parse_sql_file`` (mwsql) for everything else.
    """
    print(f"\n[SQL] parsing {len(sql_paths)} dumps ({PARALLEL_SQL_PARSERS} workers)")
    tasks = [(p, _sql_out_parquet(p, target_dir)) for p in sql_paths]
    with ProcessPoolExecutor(max_workers=min(PARALLEL_SQL_PARSERS, len(tasks))) as ex:
        futs = {ex.submit(_dispatch_sql_parser, gz, out): out for gz, out in tasks}
        for fut in as_completed(futs):
            fut.result()
    return [out for _, out in tasks]


# ============================================================================
# PHASE 3: MERGE
# ============================================================================


def merge_shards(shard_paths: list[Path], out_path: Path) -> None:
    if out_path.exists():
        print(f"[MERGE] {out_path.name} already exists — skipping")
        return
    print(f"[MERGE] merging {len(shard_paths)} XML shards -> {out_path.name}")
    pl.scan_parquet(shard_paths).sink_parquet(out_path)
    print(f"[MERGE] done: {out_path}")


# ============================================================================
# PHASE 4: CATEGORY INDEX (derived from pages + categorylinks + linktarget)
# ============================================================================


def _page_id_to_lt_id(pages_path: Path, lt_path: Path) -> pl.DataFrame:
    """Map: category page_id -> its lt_id.

    page.title is 'Category:Foo Bar' (with prefix, spaces); lt.lt_title is
    'Foo_Bar' (no prefix, underscores). Normalize and inner-join.
    """
    cat_pages = (
        pl.scan_parquet(pages_path)
        .filter(pl.col("namespace") == CATEGORY_NAMESPACE)
        .select(["page_id", "title"])
        .with_columns(
            pl.col("title")
            .str.strip_prefix("Category:")
            .str.replace_all(" ", "_")
            .alias("norm_title")
        )
        .select(["page_id", "norm_title"])
        .collect()
    )
    cat_lt = (
        pl.scan_parquet(lt_path)
        .filter(pl.col("lt_namespace") == CATEGORY_NAMESPACE)
        .select(["lt_id", "lt_title"])
        .collect()
    )
    joined = cat_pages.join(
        cat_lt, left_on="norm_title", right_on="lt_title", how="inner"
    ).select(["page_id", "lt_id"])
    print(
        f"[CAT] page->lt: {cat_pages.height:,} category pages, "
        f"{cat_lt.height:,} lt entries -> {joined.height:,} resolved"
    )
    return joined


def _subcat_adjacency(cl_path: Path, page_to_lt: pl.DataFrame) -> dict[int, list[int]]:
    """Return parent_lt_id -> [child_lt_id]."""
    edges = (
        pl.scan_parquet(cl_path)
        .filter(pl.col("cl_type") == "subcat")
        .select(["cl_from", "cl_target_id"])
        .collect()
        .join(page_to_lt, left_on="cl_from", right_on="page_id", how="inner")
        .group_by("cl_target_id")
        .agg(pl.col("lt_id").alias("children"))
    )
    adj: dict[int, list[int]] = {
        int(p): [int(c) for c in cs]
        for p, cs in zip(edges["cl_target_id"], edges["children"], strict=True)
    }
    print(f"[CAT] subcat adjacency: {len(adj):,} parents")
    return adj


def _multi_root_bfs(adj: dict[int, list[int]], roots: list[int]) -> dict[int, int]:
    """BFS from all roots simultaneously; lt_id -> min distance."""
    dist: dict[int, int] = dict.fromkeys(roots, 0)
    q: deque[int] = deque(roots)
    while q:
        cur = q.popleft()
        d_next = dist[cur] + 1
        for child in adj.get(cur, ()):
            if child not in dist:
                dist[child] = d_next
                q.append(child)
    print(f"[CAT] BFS reached {len(dist):,} categories from {len(roots)} roots")
    return dist


def _aggregate_members(cl_path: Path) -> pl.DataFrame:
    """Return one row per cat_lt_id: n_pages, n_subcats, page_ids."""
    cl = pl.scan_parquet(cl_path).select(["cl_from", "cl_target_id", "cl_type"])
    pages = (
        cl.filter(pl.col("cl_type") == "page")
        .group_by("cl_target_id")
        .agg(pl.len().alias("n_pages"), pl.col("cl_from").alias("page_ids"))
        .collect()
    )
    subcats = (
        cl.filter(pl.col("cl_type") == "subcat")
        .group_by("cl_target_id")
        .agg(pl.len().alias("n_subcats"))
        .collect()
    )
    return pages.join(subcats, on="cl_target_id", how="full", coalesce=True)


def _resolve_roots(lt_path: Path, names: list[str]) -> list[int]:
    df = (
        pl.scan_parquet(lt_path)
        .filter(pl.col("lt_namespace") == CATEGORY_NAMESPACE)
        .filter(pl.col("lt_title").is_in(names))
        .select(["lt_id", "lt_title"])
        .collect()
    )
    missing = set(names) - set(df["lt_title"].to_list())
    if missing:
        print(f"[CAT] WARN roots not found in linktarget: {sorted(missing)}")
    return [int(x) for x in df["lt_id"].to_list()]


def build_category_index(target_dir: Path) -> Path:
    """Derive category_index.parquet from the SQL + XML parquets in target_dir."""
    cl_path = target_dir / "categorylinks.parquet"
    lt_path = target_dir / "linktarget.parquet"
    pages_path = target_dir / "pages.parquet"
    out_path = target_dir / "category_index.parquet"

    if out_path.exists():
        print(f"[CAT] {out_path.name} already exists — skipping")
        return out_path
    for p in (cl_path, lt_path, pages_path):
        if not p.exists():
            raise FileNotFoundError(f"category index missing input: {p}")

    print(f"\n[CAT] building {out_path.name}")
    t0 = time.monotonic()
    page_to_lt = _page_id_to_lt_id(pages_path, lt_path)
    adj = _subcat_adjacency(cl_path, page_to_lt)
    root_ids = _resolve_roots(lt_path, list(LEVEL_ROOTS))
    dist = _multi_root_bfs(adj, root_ids)
    members = _aggregate_members(cl_path).rename({"cl_target_id": "cat_lt_id"})

    all_cats = (
        pl.scan_parquet(lt_path)
        .filter(pl.col("lt_namespace") == CATEGORY_NAMESPACE)
        .select(
            pl.col("lt_id").alias("cat_lt_id"),
            pl.col("lt_title").alias("cat_title"),
        )
        .collect()
    )
    levels = pl.DataFrame(
        {"cat_lt_id": list(dist.keys()), "level": list(dist.values())},
        schema={"cat_lt_id": pl.Int64, "level": pl.Int32},
    )
    df = (
        all_cats.join(levels, on="cat_lt_id", how="left")
        .join(members, on="cat_lt_id", how="left")
        .with_columns(
            pl.col("n_pages").fill_null(0).cast(pl.Int32),
            pl.col("n_subcats").fill_null(0).cast(pl.Int32),
        )
        .select(["cat_lt_id", "cat_title", "level", "n_pages", "n_subcats", "page_ids"])
        .sort(["level", "cat_title"], nulls_last=True)
    )
    df.write_parquet(out_path, compression="zstd")

    n_reachable = int(df["level"].is_not_null().sum())
    elapsed = time.monotonic() - t0
    print(
        f"[CAT] wrote {out_path} in {elapsed:.1f}s: {df.height:,} categories "
        f"({n_reachable:,} reachable, {df.height - n_reachable:,} orphan)"
    )
    return out_path


# ============================================================================
# PHASE 5: FULL SUBCAT-EDGE DAG (long-form parquet for thirdeye indexing)
# ============================================================================
#
# Wikipedia's category DAG is a property of the dump, not of any one
# corpus shard — but until Phase 5 was added, every indexing run
# re-derived it from scratch (~17 s of polars join + sort over 10 M
# rows). Phase 5 computes it once at dump-prep time and persists the
# long-form (child, parent) edge table next to the raw parquets.
# :meth:`thirdeye.data.wikidump.WikiDump.load_full_category_dag` then
# reads it directly (~0.5 s) on every downstream indexing run.


def _build_via_wikidump(target_dir: Path, tag: str, label: str, method: str) -> Path:
    """Run ``WikiDump(target_dir).<method>()`` with a timed log line.

    Shared boilerplate for the derived phases that delegate to a
    ``WikiDump.write_*`` method so the exact same polars chain feeds both
    the prereq write path and the fallback compute path inside ``WikiDump``
    itself (single source of truth, no risk of the two diverging).
    """
    from sira.src.python.thirdeye.data.wikidump import WikiDump

    print(f"\n[{tag}] building {label}")
    t0 = time.monotonic()
    out_path = getattr(WikiDump(target_dir), method)()
    print(f"[{tag}] wrote {out_path} in {time.monotonic() - t0:.1f}s")
    return out_path


def build_full_category_dag(target_dir: Path) -> Path:
    """Persist ``category_dag.parquet`` — long-form ``(child, parent)``
    subcat edges, sorted by ``(child, parent)``."""
    return _build_via_wikidump(
        target_dir, "DAG", "category_dag.parquet", "write_full_category_dag"
    )


# ============================================================================
# PHASE 6: DOC CATEGORIES (long-form ns=0 page_id -> category tags)
# ============================================================================
#
# Wikipedia's article-to-direct-category mapping is a property of
# the dump, not of any one corpus shard — but until Phase 6 was
# added, every BM25X indexing run re-derived it from scratch
# (~2 parquet scans + a per-row dict build in
# :meth:`WikiDump.batch_page_categories`). Phase 6 computes it once
# at dump-prep time and persists the long-form
# ``(page_id, category)`` table next to the raw parquets;
# :meth:`batch_page_categories` then filters that table by
# ``page_ids`` in a single scan on every downstream indexing run.


def build_doc_categories(target_dir: Path) -> Path:
    """Persist ``doc_categories.parquet`` — long-form
    ``(page_id, category)`` ns=0 article tags, sorted by
    ``(page_id, category)``."""
    return _build_via_wikidump(
        target_dir, "DOC_CATS", "doc_categories.parquet", "write_doc_categories"
    )


# ============================================================================
# PHASE 7: CAT STRUCTURE (per-cat direct subcats + direct pages)
# ============================================================================
#
# A category's direct subcats and direct page members are properties
# of the dump, not of any particular indexed corpus. Phase 7 emits
# this whole-dump, corpus-independent direct-children shape once at
# dump-prep time. Consumed by the ``CategoryNav`` navigation tool
# (``tools/category_nav.py``) for direct-children lookups without
# walking raw parquets.


def build_cat_structure(target_dir: Path) -> Path:
    """Persist ``cat_structure.parquet`` — per-cat direct subcats +
    direct pages — plus the seekable members blob
    (``cat_member_offsets.parquet`` + ``cat_members.bin``) derived from it.
    Delegates to :meth:`WikiDump.write_cat_structure` /
    :meth:`WikiDump.write_cat_members` so the polars chains live in one place
    (single source of truth with the fallback compute path inside ``WikiDump``).
    The blob is the fast member-lookup form of cat_structure's ``page_ids``: a
    category-restricted search resolves members by an offsets pushdown read + an
    mmap slice (~3 ms) instead of decompressing a shared cat_structure column
    chunk (60-300+ ms).
    """
    from sira.src.python.thirdeye.data.wikidump import WikiDump

    print("\n[CAT_STRUCT] building cat_structure.parquet")
    t0 = time.monotonic()
    dump = WikiDump(target_dir)
    out_path = dump.write_cat_structure()
    print(f"[CAT_STRUCT] wrote {out_path} in {time.monotonic() - t0:.1f}s")

    print("[CAT_STRUCT] building seekable members blob")
    t0 = time.monotonic()
    off_path, bin_path = dump.write_cat_members()
    print(
        f"[CAT_STRUCT] wrote {off_path.name} + {bin_path.name} "
        f"in {time.monotonic() - t0:.1f}s"
    )
    return out_path


# ============================================================================
# PHASE 8: ALL-NAMESPACE PAGE CATEGORIES (long-form page_id -> category)
# ============================================================================
#
# All-namespace sibling of Phase 6's ``doc_categories.parquet``.
# Long-form ``(page_id, namespace, category)`` with the ``namespace``
# column so callers can slice by ns without re-joining to the 29 GB
# ``pages.parquet``. Built inline (not via ``WikiDump.write_*``)
# because no downstream method currently consumes this artifact —
# there is no fallback compute path to keep in sync with. Use it
# only when you genuinely need non-article tags (ns=14 Category-page
# navigation, ns=4 Wikipedia-project policy lookups); for the common
# RAG path stay on ``doc_categories.parquet``.


def build_page_categories(target_dir: Path) -> Path:
    """Persist ``page_categories.parquet`` — all-namespace long-form
    ``(page_id, namespace, category)``, sorted by ``(namespace,
    page_id, category)``.
    """
    out_path = target_dir / "page_categories.parquet"
    if out_path.exists():
        print(f"[PAGE_CATS] {out_path.name} already exists — skipping")
        return out_path

    cl_path = target_dir / "categorylinks.parquet"
    lt_path = target_dir / "linktarget.parquet"
    pages_path = target_dir / "pages.parquet"
    for p in (cl_path, lt_path, pages_path):
        if not p.exists():
            raise FileNotFoundError(f"page_categories missing input: {p}")

    print("\n[PAGE_CATS] building page_categories.parquet")
    t0 = time.monotonic()

    cl = pl.scan_parquet(cl_path).select(
        pl.col("cl_from").alias("page_id"),
        "cl_target_id",
    )
    lt = (
        pl.scan_parquet(lt_path)
        .filter(pl.col("lt_namespace") == CATEGORY_NAMESPACE)
        .select("lt_id", pl.col("lt_title").alias("category"))
    )
    pages_ns = pl.scan_parquet(pages_path).select("page_id", "namespace")

    # LEFT join with pages_ns (not inner) so we keep every categorylinks
    # row even if cl_from refers to a page absent from pages.parquet —
    # mostly Talk / User pages excluded by the pages-articles-multistream
    # XML dump but still tagged in the categorylinks SQL dump. Those rows
    # land here with ``namespace = null``; downstream code can ignore them
    # by filtering ``namespace.is_not_null()`` if it doesn't want noise.
    (
        cl.join(lt, left_on="cl_target_id", right_on="lt_id", how="inner")
        .join(pages_ns, on="page_id", how="left")
        .select("page_id", "namespace", "category")
        .unique()
        .sort("namespace", "page_id", "category", nulls_last=True)
        .sink_parquet(out_path, compression="zstd")
    )
    n = _parquet_len(out_path)
    print(f"[PAGE_CATS] wrote {out_path} in {time.monotonic() - t0:.1f}s: {n:,} rows")
    return out_path


# ============================================================================
# PHASE 9: PAGES WITH CATEGORIES (page-centric view: title + desc + categories)
# ============================================================================
#
# One-stop denormalized view: one row per ``pages.parquet`` page,
# enriched with its short description (from ``page_props``, key
# ``wikibase-shortdesc``) and its categories (aggregated to ``list[Utf8]``
# from ``page_categories``). LEFT joins from ``pages.parquet`` so
# every page survives — pages with no ``{{Short description}}``
# template get ``description = null``; pages with no categorylinks
# entries get ``categories = null``. Saves downstream code from
# doing the same two left joins on every read.


def build_pages_with_categories(target_dir: Path) -> Path:
    """Persist ``pages_with_categories.parquet`` — one row per page in
    ``pages.parquet``, enriched with description + categories list + text.

    Includes ``text`` (raw wikitext) so a single parquet read powers
    full-text BM25 / RAG without needing a second join back to
    ``pages.parquet``. File grows to ~30 GB because ``text`` is the bulk
    of ``pages.parquet``.
    """
    out_path = target_dir / "pages_with_categories.parquet"
    if out_path.exists():
        print(f"[PWC] {out_path.name} already exists — skipping")
        return out_path

    pages_path = target_dir / "pages.parquet"
    page_cats_path = target_dir / "page_categories.parquet"
    page_props_path = target_dir / "page_props.parquet"
    for p in (pages_path, page_cats_path, page_props_path):
        if not p.exists():
            raise FileNotFoundError(f"pages_with_categories missing input: {p}")

    print("\n[PWC] building pages_with_categories.parquet")
    t0 = time.monotonic()

    cats_by_page = (
        pl.scan_parquet(page_cats_path)
        .group_by("page_id")
        .agg(pl.col("category").sort().alias("categories"))
    )
    # The MediaWiki page_props key for short description is
    # ``wikibase-shortdesc``, not ``description`` — verified empirically
    # against the 20260501 enwiki dump (no ``description`` key exists).
    descriptions = (
        pl.scan_parquet(page_props_path)
        .filter(pl.col("pp_propname") == "wikibase-shortdesc")
        .select(
            pl.col("pp_page").alias("page_id"),
            pl.col("pp_value").alias("description"),
        )
    )
    pages = pl.scan_parquet(pages_path).select("page_id", "namespace", "title", "text")

    (
        pages.join(descriptions, on="page_id", how="left")
        .join(cats_by_page, on="page_id", how="left")
        .select("page_id", "namespace", "title", "description", "categories", "text")
        .sort("page_id")
        .sink_parquet(out_path, compression="zstd")
    )
    n = _parquet_len(out_path)
    print(f"[PWC] wrote {out_path} in {time.monotonic() - t0:.1f}s: {n:,} rows")
    return out_path


# ============================================================================
# PROBE
# ============================================================================


def probe_summary(parquet_path: Path) -> None:
    df = pl.read_parquet(parquet_path)
    print(f"\n=== {df.height:,} pages in {parquet_path.name}")
    print("\n=== per-namespace counts:")
    for row in (
        df.group_by("namespace")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(20)
        .iter_rows(named=True)
    ):
        print(f"  ns={row['namespace']:>4}: {row['n']:>6,}")
    redirects = df.filter(pl.col("redirect_target").is_not_null()).height
    print(f"\n=== redirects: {redirects:,} ({redirects / df.height:.2%})")
    print("\n=== first 3 rows:")
    for row in df.head(3).iter_rows(named=True):
        text = row["text"] or ""
        print("-" * 80)
        print(f"  page_id:         {row['page_id']}")
        print(f"  namespace:       {row['namespace']}")
        print(f"  title:           {row['title']}")
        print(f"  redirect_target: {row['redirect_target']}")
        print(f"  text len:        {len(text):,} chars")
        print(f"  text[:300]:      {text[:300]!r}")


# ============================================================================
# MAIN
# ============================================================================


_PHASE_1_3_OUTPUTS: tuple[str, ...] = (
    "pages.parquet",
    "categorylinks.parquet",
    "linktarget.parquet",
    "redirect.parquet",
    "page_props.parquet",
    "pagelinks.parquet",
)


def _phase_1_3_complete(target_dir: Path) -> bool:
    """Return True iff every Phase 1-3 output parquet already exists.

    When this is True, ``main()`` skips the entire download + parse +
    merge block — no manifest fetch, no mwsql import, no network. This
    lets a contributor add a new derived Phase N+1 (e.g. Phase 6's
    ``doc_categories.parquet``) and re-run the script on an already-
    populated dump without re-downloading 37 GB and without needing
    mwsql/mwxml in the conda env or HTTP access to dumps.wikimedia.org.
    """
    return all((target_dir / name).exists() for name in _PHASE_1_3_OUTPUTS)


def main(probe: bool = False) -> None:
    target_dir = localdb_dir() / "wikipedia-monthly" / DUMP_DATE / LANG
    raw_dir = target_dir / "raw"
    shard_dir = target_dir / "shards"
    pages_parquet = target_dir / "pages.parquet"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Probe mode always exercises the download + parse path (it's the whole
    # point of probe — sanity-check the XML parser end-to-end on a small shard).
    skip_phase_1_3 = (not probe) and _phase_1_3_complete(target_dir)
    if skip_phase_1_3:
        print(
            f"=== Phase 1-3: all expected parquets present under {target_dir} — "
            f"skipping download + parse + merge ==="
        )
    else:
        # === Phase 1: discover + download ===
        status = fetch_status()
        xml_files = get_multistream_files(status)
        xml_gb = sum(s for _, s in xml_files) / 1e9
        print(f"Found {len(xml_files)} XML shards; total = {xml_gb:.1f} GB")

        if probe:
            xml_files = [min(xml_files, key=lambda x: x[1])]
            name = xml_files[0][0].rsplit("/", 1)[-1]
            print(
                f"PROBE MODE: only downloading + parsing smallest XML shard: {name}"
                f" ({xml_files[0][1] / 1e6:.0f} MB); skipping SQL + merge"
            )
            bz2_paths = download_all(xml_files, raw_dir)
            shard_paths = parse_xml_all(bz2_paths, shard_dir)
            probe_summary(shard_paths[0])
            print(f"\nArtifacts kept under: {target_dir}")
            return

        sql_locs: list[tuple[str, int]] = []
        for suffix in SQL_FILE_SUFFIXES:
            url, size = find_file_by_suffix(status, suffix)
            sql_locs.append((url, size))
            print(f"Found {url.rsplit('/', 1)[-1]}: {size / 1e9:.2f} GB")

        all_files = xml_files + sql_locs
        all_paths = download_all(all_files, raw_dir)
        bz2_paths = all_paths[: len(xml_files)]
        sql_paths = all_paths[len(xml_files) :]

        # === Phase 2: parse XML + SQL CONCURRENTLY (two inner ProcessPools) ===
        print(
            f"\n=== Phase 2: parsing XML ({PARALLEL_XML_PARSERS} workers) and "
            f"SQL ({PARALLEL_SQL_PARSERS} workers) in parallel ==="
        )
        shard_paths: list[Path] = []
        with ThreadPoolExecutor(max_workers=2) as outer:
            xml_fut = outer.submit(parse_xml_all, bz2_paths, shard_dir)
            sql_fut = outer.submit(parse_sql_all, sql_paths, target_dir)
            shard_paths = xml_fut.result()
            sql_fut.result()

        # === Phase 3: merge XML shards ===
        merge_shards(shard_paths, pages_parquet)

    # === Phase 4: derive category index ===
    build_category_index(target_dir)

    # === Phase 5: derive full subcat-edge DAG ===
    build_full_category_dag(target_dir)

    # === Phase 6: derive ns=0 article direct-category tags ===
    build_doc_categories(target_dir)

    # === Phase 7: derive per-cat direct subcats + direct pages ===
    build_cat_structure(target_dir)

    # === Phase 8: derive all-namespace (page_id, namespace, category) tags ===
    build_page_categories(target_dir)

    # === Phase 9: page-centric view (title + description + categories) ===
    build_pages_with_categories(target_dir)

    print(f"\nDone. Outputs under: {target_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--probe",
        action="store_true",
        help=(
            "Download + parse only the smallest XML shard, print samples; skip "
            "SQL + merge. Artifacts persist; re-run without --probe to do everything."
        ),
    )
    main(probe=parser.parse_args().probe)
