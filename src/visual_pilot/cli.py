"""CLI entry point: ``python -m src.visual_pilot.cli <stage> [flags]``.

Subcommands dispatch through the ``COMMANDS`` registry: each maps to a
``run(args)`` function (later workstreams plug in their stage modules here).
Unimplemented stages print "not implemented yet" and exit with code 2.
"""

from __future__ import annotations

import argparse
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
    """init -> select -> parse -> triage -> judge -> store -> extract -> report.

    Shared cumulative budget: each stage receives the remaining allowance
    (llm_calls spend recorded since this invocation started). Aborts after
    select when any disease exceeds the cap unless --accept-cap was given.
    """
    import json as _json

    from . import config

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

    stages = ["select", "parse", "triage", "judge", "store", "extract", "report"]
    for name in stages:
        if budget0 is not None:
            remaining = budget0 - _spent_since_start()
            if remaining <= 0:
                print(f"run-all: budget ${budget0:.2f} exhausted before {name}; stopping.")
                return 4
            args.budget_usd = remaining
        print(f"run-all: {name} (remaining budget {args.budget_usd})")
        rc = COMMANDS[name](args) if COMMANDS[name] else 2
        if name == "select":
            counts_path = config.reports_dir() / "stage2_counts.json"
            over = []
            if counts_path.exists():
                counts = _json.loads(counts_path.read_text())
                over = [k for k, v in counts.items() if (v or {}).get("over_cap")]
            if over and not getattr(args, "accept_cap", False):
                print(
                    "run-all: relevant articles exceed the stage-2 cap for: "
                    + ", ".join(over)
                    + ". Human confirmation required — rerun with --accept-cap."
                )
                return 3
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
        help="per-disease article cap (default: 150)",
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
            help="confirm the stage-2 per-disease cap; parse only the top-capped "
            "relevant articles per disease",
        )
    subs["extract"].add_argument(
        "--force",
        action="store_true",
        help="re-extract articles that already have source='text' findings",
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
