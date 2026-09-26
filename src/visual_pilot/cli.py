"""CLI entry point: ``python -m src.visual_pilot.cli <stage> [flags]``.

Subcommands dispatch through the ``COMMANDS`` registry: each maps to a
``run(args)`` function (later workstreams plug in their stage modules here).
Unimplemented stages print "not implemented yet" and exit with code 2.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

from . import config, db, diseases

CommandFn = Callable[[argparse.Namespace], int]


def _cmd_init(args: argparse.Namespace) -> int:
    conn = db.init_db()
    try:
        counts = diseases.seed(conn)
        vocab_per_disease = {
            row["disease_key"]: row["n"]
            for row in conn.execute(
                "SELECT je.value AS disease_key, COUNT(*) AS n "
                "FROM findings_vocab fv, json_each(fv.disease_keys_json) je "
                "GROUP BY je.value ORDER BY je.value"
            )
        }
        row_counts = {
            table: conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
            for table in ("articles", "figures", "panels", "disease_findings", "llm_calls")
        }
    finally:
        conn.close()

    print(f"DB initialized: {config.db_path()}")
    print(f"diseases: {counts['diseases']} ({', '.join(diseases.DISEASE_KEYS)})")
    per_disease = ", ".join(f"{k}={v}" for k, v in vocab_per_disease.items())
    print(f"findings_vocab: {counts['findings_vocab']} approved entries ({per_disease})")
    print("rows: " + ", ".join(f"{k}={v}" for k, v in row_counts.items()))
    return 0


def _cmd_not_implemented(args: argparse.Namespace) -> int:
    print(f"{args.command}: not implemented yet")
    return 2


def _lazy(module: str, attr: str = "run") -> CommandFn:
    """Resolve a stage module at call time so `cli --help` stays fast."""

    def fn(args: argparse.Namespace) -> int:
        import importlib

        mod = importlib.import_module(f".{module}", package=__package__)
        return getattr(mod, attr)(args)

    return fn


def _cmd_run_all(args: argparse.Namespace) -> int:
    """Select once, then expand visual-yield batches until marginal yield falls."""
    from . import parse as parse_stage

    started = time.monotonic()
    max_runtime = max(1, int(getattr(args, "max_runtime_seconds", 900) or 900))
    max_articles = max(1, int(getattr(args, "max_articles", 600) or 600))
    batch_size = max(1, int(getattr(args, "batch_size", 50) or 50))
    zero_yield_limit = max(1, int(getattr(args, "zero_yield_batches", 2) or 2))
    if args.dry_run:
        print(
            "run-all dry-run: no database writes, network retrieval, JATS fetch, "
            f"or LLM calls; planned batch size={batch_size}, "
            f"article safety limit={max_articles}/disease, runtime={max_runtime}s"
        )
        return 0
    _cmd_init(args)
    conn = db.connect()
    start_call_id = conn.execute(
        "SELECT COALESCE(MAX(call_id), 0) AS m FROM llm_calls"
    ).fetchone()["m"]
    conn.close()
    budget0 = args.budget_usd

    def _spent_since_start() -> float:
        c = db.connect()
        try:
            return c.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) AS s FROM llm_calls WHERE call_id > ?",
                (start_call_id,),
            ).fetchone()["s"]
        finally:
            c.close()

    def _remaining_budget():
        return None if budget0 is None else budget0 - _spent_since_start()

    def _snapshot(disease_key: str) -> tuple[set[str], set[str]]:
        c = db.connect()
        try:
            hashes = {
                r["sha256"] for r in c.execute(
                    "SELECT DISTINCT sha256 FROM panels WHERE disease_key=? "
                    "AND sha256 IS NOT NULL AND sha256 != ''", (disease_key,)
                )
            }
            approved = {
                r["finding_key"] for r in c.execute(
                    "SELECT finding_key FROM findings_vocab WHERE approved=1 "
                    "AND EXISTS (SELECT 1 FROM json_each(disease_keys_json) je "
                    "WHERE je.value=?)", (disease_key,)
                )
            }
            findings: set[str] = set()
            for r in c.execute(
                "SELECT findings_json FROM panels WHERE disease_key=?", (disease_key,)
            ):
                for value in db.from_json(r["findings_json"], []) or []:
                    key = value.get("finding_key") if isinstance(value, dict) else value
                    if key and str(key) in approved:
                        findings.add(str(key))
            return hashes, findings
        finally:
            c.close()

    def _unfinished_figures(pmcids=None) -> int:
        c = db.connect()
        try:
            placeholders = ""
            values: tuple = (
                "pending", "caption_kept", "caption_uncertain", "vision_accepted", 3
            )
            if pmcids:
                marks = ",".join("?" for _ in pmcids)
                placeholders = f" AND pmcid IN ({marks})"
                values += tuple(pmcids)
            return c.execute(
                "SELECT COUNT(*) AS n FROM figures WHERE "
                "(status IN (?,?,?,?) OR (status='vision_error' AND attempts < ?))"
                + placeholders,
                values,
            ).fetchone()["n"]
        finally:
            c.close()

    if COMMANDS["select"] is None:
        return 2
    requested_pmcids = set(args.pmcids) if args.pmcids else None
    if getattr(args, "skip_select", False):
        print("run-all: using committed article selections; selection skipped")
    elif requested_pmcids is None:
        print("run-all: select")
        args.budget_usd = _remaining_budget()
        rc = COMMANDS["select"](args)
        if rc != 0:
            return rc
    else:
        print(f"run-all: using {len(requested_pmcids)} existing requested PMCIDs; selection skipped")
    if args.dry_run:
        return 0

    disease_keys = list(diseases.DISEASE_KEYS) if args.disease == "all" else [args.disease]
    scoped = argparse.Namespace(**vars(args))
    per_disease_cap = min(max_articles, args.limit) if args.limit is not None else max_articles

    # Resume any figure work left by an interrupted earlier invocation before
    # measuring marginal yield from new article batches.
    for disease_key in disease_keys:
        scoped.disease = disease_key
        scoped.pmcids = sorted(requested_pmcids) if requested_pmcids is not None else None
        for name in ("triage", "judge", "store"):
            remaining = _remaining_budget()
            if remaining is not None and remaining <= 0:
                print(f"run-all: budget ${budget0:.2f} exhausted while resuming {name}")
                return 4
            scoped.budget_usd = remaining
            print(f"run-all: resume queued figures for {disease_key}: {name}")
            rc = COMMANDS[name](scoped) if COMMANDS[name] else 2
            if rc != 0:
                return rc

    for disease_key in disease_keys:
        processed = 0
        zero_yield_batches = 0
        while processed < per_disease_cap:
            if time.monotonic() - started >= max_runtime:
                print(f"run-all: runtime safety limit ({max_runtime}s) reached; stopping expansion")
                break
            remaining = _remaining_budget()
            if remaining is not None and remaining <= 0:
                print(f"run-all: budget ${budget0:.2f} exhausted; stopping expansion")
                return 4
            conn = db.connect()
            try:
                selected = parse_stage.select_batch(
                    conn,
                    disease_key,
                    min(batch_size, per_disease_cap - processed),
                    pmcids=requested_pmcids,
                )
            finally:
                conn.close()
            if requested_pmcids is not None:
                selected = [r for r in selected if r["pmcid"] in requested_pmcids]
            if not selected:
                break
            pmcids = [r["pmcid"] for r in selected]
            before_images, before_findings = _snapshot(disease_key)
            scoped.disease = disease_key
            scoped.pmcids = pmcids
            scoped.limit = None
            for name in ("parse", "triage", "judge", "store"):
                remaining = _remaining_budget()
                if remaining is not None and remaining <= 0:
                    print(f"run-all: budget ${budget0:.2f} exhausted before {name}; stopping")
                    return 4
                scoped.budget_usd = remaining
                print(f"run-all: {disease_key} batch {processed // batch_size + 1}: {name} ({len(pmcids)} articles)")
                rc = COMMANDS[name](scoped) if COMMANDS[name] else 2
                if rc != 0:
                    return rc
            after_images, after_findings = _snapshot(disease_key)
            new_images = after_images - before_images
            new_findings = after_findings - before_findings
            processed += len(pmcids)
            unfinished = _unfinished_figures(pmcids)
            if unfinished:
                print(
                    f"run-all: {disease_key} batch has {unfinished} unfinished figures; "
                    "leaving it resumable and pausing disease expansion"
                )
                break
            print(
                f"run-all: {disease_key} batch yield: {len(new_images)} distinct images, "
                f"{len(new_findings)} newly covered findings"
            )
            if new_images or new_findings:
                zero_yield_batches = 0
            else:
                zero_yield_batches += 1
                if zero_yield_batches >= zero_yield_limit:
                    print(f"run-all: {disease_key} stopped after {zero_yield_batches} consecutive zero-yield batches")
                    break

    scoped.disease = args.disease
    scoped.pmcids = None
    for name in ("extract", "report"):
        remaining = _remaining_budget()
        if remaining is not None and remaining <= 0:
            print(f"run-all: budget ${budget0:.2f} exhausted before {name}; stopping")
            return 4
        scoped.budget_usd = remaining
        print(f"run-all: {name}")
        rc = COMMANDS[name](scoped) if COMMANDS[name] else 2
        if rc != 0:
            return rc
    return 0


# Registry: later workstreams replace the None entries with their module's
# run(args) function (e.g. select_articles.run, triage.run, ...).
COMMANDS: dict[str, CommandFn | None] = {
    "init": _cmd_init,
    "select": _lazy("select_articles"),
    "parse": _lazy("parse"),
    "triage": _lazy("triage"),
    "judge": _lazy("judge"),
    "store": _lazy("store"),
    "extract": _lazy("extract_findings"),
    "report": _lazy("report"),
    "serve": _lazy("viewer"),
    "run-all": _cmd_run_all,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="src.visual_pilot.cli",
        description="Visual Findings Library pilot pipeline",
    )
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--disease",
        choices=["sle", "dm", "as", "all"],
        default="all",
        help="disease scope (default: all)",
    )
    shared.add_argument("--limit", type=int, default=None, help="max items to process")
    shared.add_argument(
        "--dry-run",
        action="store_true",
        help="plan only; no writes or LLM calls",
    )
    shared.add_argument(
        "--budget-usd",
        type=float,
        default=None,
        help="stop LLM spend when the cumulative cost reaches this amount",
    )
    shared.add_argument(
        "--cap",
        type=int,
        default=None,
        help="legacy stage-2 report threshold; parse uses yield batches",
    )
    shared.add_argument(
        "--batch-size", type=int, default=50,
        help="articles per disease in a visual-yield expansion batch (default: 50)",
    )
    shared.add_argument(
        "--max-articles", type=int, default=600,
        help="safety limit on articles processed per disease per run (default: 600)",
    )
    shared.add_argument(
        "--max-runtime-seconds", type=int, default=900,
        help="runtime safety limit for parse/run-all expansion (default: 900)",
    )
    shared.add_argument(
        "--zero-yield-batches", type=int, default=2,
        help="stop a disease after this many consecutive batches add no images or findings",
    )
    shared.add_argument(
        "--pmcids",
        nargs="*",
        default=None,
        help="restrict the stage to these PMCIDs",
    )
    shared.add_argument("--port", type=int, default=8765, help="viewer port")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subs = {}
    for name in COMMANDS:
        subs[name] = subparsers.add_parser(name, parents=[shared])
    subs["store"].add_argument(
        "--refresh",
        action="store_true",
        help="recompute panel attribution/subtype from stored vision_json",
    )
    for name in ("parse", "run-all"):
        subs[name].add_argument(
            "--accept-cap",
            action="store_true",
            help="deprecated compatibility flag; expansion is yield-based",
        )
    subs["run-all"].add_argument(
        "--skip-select",
        action="store_true",
        help="resume from committed article selections without new retrieval or relevance calls",
    )
    subs["extract"].add_argument(
        "--force",
        action="store_true",
        help="re-extract articles that already have source='text' findings",
    )
    for name in ("extract", "run-all"):
        subs[name].add_argument(
            "--image-only",
            action="store_true",
            help="refresh panel-derived finding links without text extraction",
        )
    subs["select"].add_argument(
        "--recheck-title-rule",
        action="store_true",
        help="skip retrieval/licensing; re-run P1 on articles the title rule "
        "passed (relevance_reason='title_rule', status relevant|parsed)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fn = COMMANDS.get(args.command)
    if fn is None:
        return _cmd_not_implemented(args)
    return fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
