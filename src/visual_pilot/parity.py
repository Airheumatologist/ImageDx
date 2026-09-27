"""Parity harness (docs/visual_pilot_plan.md §6.2, workstream W0).

Usage::

    python3 -m src.visual_pilot.parity prepare --out DIR [--source DB]
    python3 -m src.visual_pilot.parity run --data-dir DIR \
        [--seed-llm-calls BASELINE_DB] [-- extra run-all args ...]
    python3 -m src.visual_pilot.parity compare BASELINE_DIR CANDIDATE_DIR
    python3 -m src.visual_pilot.parity selection-dump --data-dir DIR --out FILE

``prepare`` builds a scratch VP_DATA_DIR seeded from the main DB: diseases +
findings_vocab copied wholesale, the parity set's article rows with status
reset to ``relevant``, and no figures/panels/disease_findings/llm_calls.

``run`` executes ``run-all --skip-select --disease all --pmcids <parity set>``
against that dir (subprocess; extra args after ``--`` pass through). With
``--seed-llm-calls`` the baseline ``llm_calls`` rows are copied in first and
``VP_LLM_CACHE_ONLY=1`` makes any changed model input fail loudly.

``compare`` exits nonzero when the candidate differs from the baseline on
any artifact the pipeline is required to keep identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path, PurePosixPath

from . import config, db, diseases

DEFAULT_SOURCE_DB = "/Volumes/Vibing/Turborag/data/visual_pilot/visual_pilot.sqlite"
PARITY_PMCIDS_PATH = (
    Path(__file__).resolve().parent / "data" / "parity_pmcids.json"
)
CACHE_MISS_MARKER = "cache miss:"

# Columns compared for figures rows (timestamps and bookkeeping excluded).
FIGURE_COMPARE_COLS = (
    "figure_id",
    "pmcid",
    "status",
    "triage_json",
    "vision_json",
    "sha256",
    "image_url",
    "image_format",
    "effective_license",
    "attempts",
    "error",
)
# Directories whose files must be byte-identical between baseline/candidate.
FILE_DIRS = ("figures", "panels", "thumbs")
# Statuses for which store is allowed to write figure files.
FILE_ALLOWED_STATUSES = frozenset({"vision_accepted", "stored"})


def parity_pmcids() -> list[str]:
    """Load the frozen parity PMCID set (src/visual_pilot/data/parity_pmcids.json)."""
    data = json.loads(PARITY_PMCIDS_PATH.read_text())
    if isinstance(data, dict):
        return [str(p) for p in data["pmcids"]]
    return [str(p) for p in data]


def _connect_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------
def prepare(out_dir: Path, source: Path) -> dict:
    """Create a scratch VP_DATA_DIR seeded from the main DB. Returns stats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / config.DB_FILENAME
    for suffix in ("", "-wal", "-shm"):
        (target.parent / (target.name + suffix)).unlink(missing_ok=True)

    conn = db.init_db(db.connect(target))
    src = _connect_ro(source)
    pmcids = parity_pmcids()
    stats: dict = {"out": str(out_dir), "source": str(source)}

    for table in ("diseases", "findings_vocab"):
        cols = db.table_columns(conn, table)
        col_list = ", ".join(cols)
        placeholders = ", ".join("?" for _ in cols)
        rows = src.execute(f"SELECT {col_list} FROM {table}").fetchall()
        conn.executemany(
            f"INSERT OR REPLACE INTO {table} ({col_list}) "
            f"VALUES ({placeholders})",
            [tuple(row) for row in rows],
        )
        stats[table] = len(rows)

    cols = db.table_columns(conn, "articles")
    col_list = ", ".join(cols)
    marks = ", ".join("?" for _ in pmcids)
    rows = src.execute(
        f"SELECT {col_list} FROM articles WHERE pmcid IN ({marks})", pmcids
    ).fetchall()
    missing = set(pmcids) - {row["pmcid"] for row in rows}
    # All columns preserved except the pipeline status, reset to 'relevant'.
    status_idx = cols.index("status")
    conn.executemany(
        f"INSERT OR REPLACE INTO articles ({col_list}) "
        f"VALUES ({', '.join('?' for _ in cols)})",
        [
            tuple("relevant" if i == status_idx else v
                  for i, v in enumerate(tuple(row)))
            for row in rows
        ],
    )
    conn.commit()
    stats["articles"] = len(rows)
    stats["missing_pmcids"] = sorted(missing)
    src.close()

    for table in ("figures", "panels", "disease_findings", "llm_calls"):
        n = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        stats[table] = n  # must stay 0
    conn.close()
    return stats


def cmd_prepare(args: argparse.Namespace) -> int:
    stats = prepare(Path(args.out), Path(args.source))
    print(json.dumps(stats, indent=1))
    if stats["missing_pmcids"]:
        print(
            f"prepare: WARNING {len(stats['missing_pmcids'])} parity PMCIDs "
            f"absent from source: {stats['missing_pmcids']}"
        )
        return 1
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def _seed_llm_calls(data_dir: Path, baseline_db: Path) -> int:
    """Copy the baseline llm_calls rows into the scratch DB (dedup on pk)."""
    target = data_dir / config.DB_FILENAME
    conn = db.init_db(db.connect(target))
    src = _connect_ro(baseline_db)
    cols = db.table_columns(conn, "llm_calls")
    col_list = ", ".join(cols)
    placeholders = ", ".join("?" for _ in cols)
    rows = src.execute(f"SELECT {col_list} FROM llm_calls").fetchall()
    conn.executemany(
        f"INSERT OR IGNORE INTO llm_calls ({col_list}) VALUES ({placeholders})",
        [tuple(row) for row in rows],
    )
    conn.commit()
    n = conn.execute("SELECT COUNT(*) AS n FROM llm_calls").fetchone()["n"]
    src.close()
    conn.close()
    return n


def cmd_run(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    extra = list(getattr(args, "extra", []) or [])
    if extra and extra[0] == "--":
        extra = extra[1:]

    seeded = 0
    if args.seed_llm_calls:
        seeded = _seed_llm_calls(data_dir, Path(args.seed_llm_calls))
        print(f"run: seeded {seeded} llm_calls rows; VP_LLM_CACHE_ONLY=1")

    env = dict(os.environ)
    env["VP_DATA_DIR"] = str(data_dir)
    if args.seed_llm_calls:
        env["VP_LLM_CACHE_ONLY"] = "1"

    cmd = [
        sys.executable,
        "-m",
        "src.visual_pilot.cli",
        "run-all",
        "--skip-select",
        "--disease",
        "all",
    ]
    if "--pmcids" not in extra:
        cmd += ["--pmcids", *parity_pmcids()]
    cmd += extra
    print("run:", " ".join(cmd))
    proc = subprocess.run(cmd, env=env, cwd=config.REPO_ROOT)
    return proc.returncode


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
def _rows(conn: sqlite3.Connection, table: str, cols: tuple[str, ...]) -> list[tuple]:
    col_list = ", ".join(cols)
    # Sorted as tuples: insertion order (and therefore ORDER BY ties) differs
    # between sequential and streamed apply — compare content, not rowids.
    # repr() key keeps the sort deterministic across mixed str/None/float.
    return sorted(
        (tuple(row) for row in conn.execute(f"SELECT {col_list} FROM {table}")),
        key=lambda t: tuple(repr(v) for v in t),
    )


def _file_manifest(root: Path) -> dict[str, str]:
    """relpath -> sha256 for every file under root (empty dict if missing)."""
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            rel = path.relative_to(root.parent).as_posix()
            out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _compare_table(
    failures: list[str],
    bconn: sqlite3.Connection,
    cconn: sqlite3.Connection,
    table: str,
    cols: tuple[str, ...],
) -> None:
    brows = _rows(bconn, table, cols)
    crows = _rows(cconn, table, cols)
    if brows != crows:
        bset, cset = set(brows), set(crows)
        failures.append(
            f"{table}: {len(bset - cset)} baseline-only, "
            f"{len(cset - bset)} candidate-only row(s) "
            f"(of {len(brows)}/{len(crows)})"
        )
        for row in sorted(bset - cset)[:5]:
            failures.append(f"  baseline-only {table}: {row[0]!r}")
        for row in sorted(cset - bset)[:5]:
            failures.append(f"  candidate-only {table}: {row[0]!r}")


def compare(baseline_dir: Path, candidate_dir: Path) -> list[str]:
    """All §6.2 checks; returns a list of failure strings ([] = identical)."""
    failures: list[str] = []
    bconn = _connect_ro(baseline_dir / config.DB_FILENAME)
    cconn = _connect_ro(candidate_dir / config.DB_FILENAME)

    # 1. Cache-miss markers in the candidate (any changed model input).
    #    Errored baseline rows legitimately miss: failed calls are never
    #    ledgered, so a cache-only replay cannot reproduce them.
    def _baseline_errored(table: str, keycol: str) -> set:
        return {
            r[0]
            for r in bconn.execute(
                f"SELECT {keycol} FROM {table} WHERE error IS NOT NULL"
            )
        }

    for table, col, keycol in (
        ("figures", "error", "figure_id"),
        ("articles", "error", "pmcid"),
    ):
        errored = _baseline_errored(table, keycol)
        flagged = [
            r[0]
            for r in cconn.execute(
                f"SELECT {keycol} FROM {table} "
                f"WHERE {col} LIKE '%' || ? || '%'",
                (CACHE_MISS_MARKER,),
            )
            if r[0] not in errored
        ]
        if flagged:
            failures.append(
                f"{table}: {len(flagged)} row(s) contain "
                f"'{CACHE_MISS_MARKER}' errors: {flagged[:5]}"
            )

    # 2. llm_calls row count equal (no new live calls).
    bn = bconn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0]
    cn = cconn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0]
    if bn != cn:
        failures.append(f"llm_calls: baseline has {bn} rows, candidate has {cn}")

    # 3. Same input_hash set per stage.
    def hash_sets(conn):
        out: dict[str, set[str]] = {}
        for row in conn.execute("SELECT stage, input_hash FROM llm_calls"):
            out.setdefault(row["stage"], set()).add(row["input_hash"])
        return out

    bhash, chash = hash_sets(bconn), hash_sets(cconn)
    for stage in sorted(set(bhash) | set(chash)):
        if bhash.get(stage, set()) != chash.get(stage, set()):
            failures.append(
                f"llm_calls stage {stage!r}: input_hash sets differ "
                f"({len(bhash.get(stage, set()))} baseline vs "
                f"{len(chash.get(stage, set()))} candidate)"
            )

    # 4. figures rows on the compared columns. Error text is volatile across
    #    runs (transport message vs replayed cache miss), so it is reduced to
    #    presence; status still pins the outcome. `attempts` on an errored row
    #    is equally volatile: W9's in-run judge retries legitimately raise it
    #    between the recorded baseline and a cache-only replay, so it is
    #    normalized alongside error (non-errored rows still compare attempts).
    def figure_rows(conn):
        err_idx = FIGURE_COMPARE_COLS.index("error")
        att_idx = FIGURE_COMPARE_COLS.index("attempts")
        col_list = ", ".join(FIGURE_COMPARE_COLS)
        rows = []
        for row in conn.execute(f"SELECT {col_list} FROM figures"):
            t = list(row)
            if t[err_idx]:
                t[err_idx] = "(set)"
                t[att_idx] = "(any)"
            rows.append(tuple(t))
        return sorted(rows, key=lambda t: tuple(repr(v) for v in t))

    brows, crows = figure_rows(bconn), figure_rows(cconn)
    if brows != crows:
        bset, cset = set(brows), set(crows)
        failures.append(
            f"figures: {len(bset - cset)} baseline-only, "
            f"{len(cset - bset)} candidate-only row(s) "
            f"(of {len(brows)}/{len(crows)})"
        )
        for row in sorted(bset - cset)[:5]:
            failures.append(f"  baseline-only figures: {row[0]!r}")
        for row in sorted(cset - bset)[:5]:
            failures.append(f"  candidate-only figures: {row[0]!r}")

    # 5. panels rows, every column except timestamps.
    panel_cols = tuple(
        c for c in db.table_columns(bconn, "panels")
        if c not in ("created_at", "updated_at")
    )
    _compare_table(failures, bconn, cconn, "panels", panel_cols)

    # 6. disease_findings rows (content only; autoincrement id and created_at
    #    are bookkeeping — streamed apply inserts in a different order).
    df_cols = tuple(
        c
        for c in db.table_columns(bconn, "disease_findings")
        if c not in ("id", "created_at")
    )
    _compare_table(failures, bconn, cconn, "disease_findings", df_cols)

    # 7. findings_vocab proposal counts.
    _compare_table(
        failures, bconn, cconn, "findings_vocab", ("finding_key", "proposal_count")
    )

    # 8. Byte-identical files under figures/, panels/, thumbs/.
    for sub in FILE_DIRS:
        bfiles = _file_manifest(baseline_dir / sub)
        cfiles = _file_manifest(candidate_dir / sub)
        if bfiles != cfiles:
            only_b = sorted(set(bfiles) - set(cfiles))
            only_c = sorted(set(cfiles) - set(bfiles))
            changed = sorted(
                k for k in set(bfiles) & set(cfiles) if bfiles[k] != cfiles[k]
            )
            failures.append(
                f"{sub}/: {len(only_b)} baseline-only, {len(only_c)} "
                f"candidate-only, {len(changed)} content-changed file(s)"
            )
            for rel in (only_b[:3] + only_c[:3] + changed[:3]):
                failures.append(f"  {sub} file diff: {rel}")

    # 9. No files written for figures that never reached vision_accepted.
    for path in sorted((candidate_dir / "figures").rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        pmcid = path.parent.name
        name = path.name
        fig_rows = cconn.execute(
            "SELECT figure_id, status, image_url FROM figures WHERE pmcid = ?",
            (pmcid,),
        ).fetchall()
        owner = None
        for row in fig_rows:
            base = PurePosixPath((row["image_url"] or "").split("?")[0]).name
            names = {base, PurePosixPath(base).with_suffix(".png").name}
            if name in names:
                owner = row
                break
        if owner is None:
            failures.append(f"figures/{pmcid}/{name}: no owning figure row")
        elif owner["status"] not in FILE_ALLOWED_STATUSES:
            failures.append(
                f"figures/{pmcid}/{name}: figure {owner['figure_id']} has "
                f"status {owner['status']!r} (never vision_accepted)"
            )

    bconn.close()
    cconn.close()
    return failures


def cmd_compare(args: argparse.Namespace) -> int:
    failures = compare(Path(args.baseline), Path(args.candidate))
    if failures:
        print("parity compare: FAILED")
        for line in failures:
            print(f"  {line}")
        return 1
    print("parity compare: identical")
    return 0


# ---------------------------------------------------------------------------
# selection-dump (W4's selection-parity check, §6.2)
# ---------------------------------------------------------------------------
def cmd_selection_dump(args: argparse.Namespace) -> int:
    """Dump per-disease ranked (pmcid, score) lists produced by retrieval."""
    os.environ["VP_DATA_DIR"] = args.data_dir
    from . import select_articles

    conn = db.init_db()
    diseases_data = diseases.load_diseases()
    visual_findings = (
        select_articles._visual_findings_from_db(conn)
        or diseases.load_findings_vocab()
    )
    retriever = select_articles._make_retriever()
    embed_fn = getattr(retriever, "_embed_query", None)

    out: dict[str, list] = {}
    disease_keys = (
        list(diseases.DISEASE_KEYS) if args.disease == "all" else [args.disease]
    )
    for key in disease_keys:
        visual_queries = select_articles.visual_queries_for_disease(
            key,
            diseases_data,
            findings=visual_findings,
            coverage_counts=select_articles._stored_panel_counts(conn, key),
        )
        articles = select_articles.retrieve_for_disease(
            retriever.ns_pmc,
            embed_fn,
            diseases_data[key]["synonyms"],
            visual_queries,
            disease_key=key,
        )
        ranked = sorted(
            articles.items(), key=lambda kv: (-kv[1]["score"], kv[0])
        )
        out[key] = [[pmcid, info["score"]] for pmcid, info in ranked]
        print(f"selection-dump: {key} {len(ranked)} ranked candidates")
    conn.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"selection-dump: wrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="src.visual_pilot.parity",
        description="Parity harness (docs/visual_pilot_plan.md §6.2)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="seed a scratch VP_DATA_DIR from the main DB")
    p.add_argument("--out", required=True, help="scratch data dir to create")
    p.add_argument("--source", default=DEFAULT_SOURCE_DB, help="baseline DB path")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("run", help="run-all on a prepared scratch dir")
    p.add_argument("--data-dir", required=True, help="prepared VP_DATA_DIR")
    p.add_argument(
        "--seed-llm-calls",
        default=None,
        metavar="BASELINE_DB",
        help="copy baseline llm_calls into the DB and set VP_LLM_CACHE_ONLY=1",
    )
    p.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="extra run-all args (after --), e.g. -- --pmcids PMC1 --limit 2",
    )
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("compare", help="compare baseline and candidate dirs")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser(
        "selection-dump", help="dump per-disease ranked (pmcid, score) lists"
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--disease", choices=["sle", "dm", "as", "all"], default="all")
    p.set_defaults(fn=cmd_selection_dump)
    return parser


def main(argv: list[str] | None = None) -> int:
    args, unknown = build_parser().parse_known_args(argv)
    if args.command == "run":
        args.extra = [*getattr(args, "extra", []), *unknown]
    elif unknown:
        build_parser().error(f"unrecognized arguments: {' '.join(unknown)}")
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
