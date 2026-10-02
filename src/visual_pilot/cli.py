"""CLI entry point: ``python -m src.visual_pilot.cli <stage> [flags]``.

Subcommands dispatch through the ``COMMANDS`` registry: each maps to a
``run(args)`` function (later workstreams plug in their stage modules here).
Unimplemented stages print "not implemented yet" and exit with code 2.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone

from . import config, db, diseases, timing

logger = logging.getLogger(__name__)

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


def _write_timings_report() -> None:
    """W0/C7: write reports/timings_<utc>.json at the end of run-all."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        path = timing.write_report(config.reports_dir() / f"timings_{stamp}.json")
        print(f"run-all: timings written to {path}")
    except Exception as exc:  # noqa: BLE001 - reporting must never fail run-all
        print(f"run-all: could not write timings report: {exc}")


def _lazy(module: str, attr: str = "run") -> CommandFn:
    """Resolve a stage module at call time so `cli --help` stays fast."""

    def fn(args: argparse.Namespace) -> int:
        import importlib

        mod = importlib.import_module(f".{module}", package=__package__)
        return getattr(mod, attr)(args)

    return fn


def _cmd_run_all(args: argparse.Namespace) -> int:
    """Figure-first loop: discover → triage → judge → store, then extract/report.

    Discovery runs broad sources first: one ``overview`` round (narrative
    reviews and case series surveying each disease), one ``manifestation``
    round (reviews and case series about each under-target finding), then
    ``backfill`` rounds over any article type, case reports included. Every
    round re-plans the pairs still under the image target (fewest images
    first), parses the matching articles and runs the LLM stages on just
    those articles, so case reports only fill what reviews left short.
    Backfill rounds repeat until no pair is under target, a round discovers
    nothing new, or the runtime/budget limit is reached. ``--pmcids`` skips discovery and only
    drains the stages for those articles. Figures an interrupted run left
    unfinished are drained before the first discovery round.
    """
    from . import discover, judge

    started = time.monotonic()
    max_runtime = int(args.max_runtime_seconds)
    if args.dry_run:
        print(f"run-all dry-run: no writes; per-pair={args.per_pair}, runtime={max_runtime}s")
        return 0
    _cmd_init(args)
    read_conn = db.connect()
    try:
        start_call_id = read_conn.execute(
            "SELECT COALESCE(MAX(call_id), 0) AS m FROM llm_calls"
        ).fetchone()["m"]

        def _remaining_budget():
            if args.budget_usd is None:
                return None
            spent = read_conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) AS s FROM llm_calls WHERE call_id > ?",
                (start_call_id,),
            ).fetchone()["s"]
            return args.budget_usd - spent

        scoped = argparse.Namespace(**vars(args))

        def _stage(name: str, pmcids, label: str) -> int:
            remaining = _remaining_budget()
            if remaining is not None and remaining <= 0:
                print(f"run-all: budget ${args.budget_usd:.2f} exhausted before {name}; stopping")
                return 4
            scoped.budget_usd = remaining
            scoped.pmcids = pmcids
            print(f"run-all: {label}: {name}")
            with timing.stage(name):
                return COMMANDS[name](scoped)

        def _drain(pmcids, label: str) -> int:
            for name in ("triage", "judge"):
                rc = _stage(name, pmcids, label)
                if rc != 0:
                    return rc
            for _ in range(judge.MAX_ATTEMPTS - 1):
                marks = ",".join("?" for _ in pmcids)
                retry = read_conn.execute(
                    "SELECT COUNT(*) AS n FROM figures WHERE status='vision_error' "
                    f"AND attempts < ? AND pmcid IN ({marks})",
                    (judge.MAX_ATTEMPTS, *pmcids),
                ).fetchone()["n"] if pmcids else 0
                if not retry:
                    break
                rc = _stage("judge", pmcids, label)
                if rc != 0:
                    return rc
            return _stage("store", pmcids, label)

        if args.pmcids:
            rc = _drain(list(args.pmcids), "explicit PMCIDs")
            if rc != 0:
                return rc
        else:
            disease_keys = list(diseases.DISEASE_KEYS) if args.disease == "all" else [args.disease]
            # Resume: finish figures an interrupted run left mid-pipeline
            # before discovering more articles.
            leftover = [
                row["pmcid"] for row in read_conn.execute(
                    "SELECT DISTINCT f.pmcid FROM figures f JOIN articles a USING(pmcid) "
                    "WHERE (f.status IN ('pending','caption_kept','caption_uncertain','vision_accepted') "
                    "OR (f.status='vision_error' AND f.attempts < ?)) "
                    "AND EXISTS (SELECT 1 FROM json_each(a.primary_disease_keys_json) je "
                    f"WHERE je.value IN ({','.join('?' for _ in disease_keys)})) "
                    "ORDER BY f.pmcid",
                    (judge.MAX_ATTEMPTS, *disease_keys),
                )
            ]
            for i in range(0, len(leftover), args.batch_size):
                chunk = leftover[i : i + args.batch_size]
                rc = _drain(chunk, f"resume batch {i // args.batch_size + 1}")
                if rc != 0:
                    return rc
            write_conn = db.connect()
            try:
                schedule = (
                    [] if args.skip_review_passes else ["overview", "manifestation"]
                ) + ["backfill"] * int(args.max_rounds)
                for round_no, pass_name in enumerate(schedule, 1):
                    left = max_runtime - (time.monotonic() - started)
                    if left <= 0:
                        print("run-all: runtime limit reached")
                        break
                    print(f"run-all: round {round_no}: discover ({pass_name})")
                    with timing.stage("discover"):
                        stats = discover.discover(
                            write_conn, disease_keys, per_pair=args.per_pair,
                            target=int(config.VP_FINDING_IMAGE_TARGET), max_runtime=left,
                            pass_name=pass_name,
                        )
                    pmcids = stats["pmcids"]
                    if not pmcids:
                        if pass_name != "backfill":
                            print(f"run-all: {pass_name} pass found no new articles")
                            continue
                        print("run-all: no new articles for any under-target pair; done")
                        break
                    for i in range(0, len(pmcids), args.batch_size):
                        chunk = pmcids[i : i + args.batch_size]
                        rc = _drain(chunk, f"round {round_no} batch {i // args.batch_size + 1}")
                        if rc != 0:
                            return rc
            finally:
                write_conn.close()

        scoped.disease = args.disease
        scoped.pmcids = None
        for name in ("extract", "report"):
            rc = _stage(name, None, "final")
            if rc != 0:
                return rc
        return 0
    finally:
        read_conn.close()
        _write_timings_report()


# Registry: later workstreams replace the None entries with their module's
# run(args) function (e.g. discover.run, triage.run, ...).
COMMANDS: dict[str, CommandFn | None] = {
    "init": _cmd_init,
    "discover": _lazy("discover"),
    "triage": _lazy("triage"),
    "judge": _lazy("judge"),
    "requeue-plates": _lazy("judge", "run_requeue_plates"),
    "requeue-age-vetoes": _lazy("judge", "run_requeue_age_vetoes"),
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
        choices=[*diseases.disease_keys_from_catalog(), "all"],
        default="all",
        help="disease scope (default: all)",
    )
    shared.add_argument(
        "--limit",
        type=int,
        default=None,
        help="max items to process (discover: pairs)",
    )
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
        "--batch-size", type=int, default=50,
        help="articles per triage/judge/store batch in run-all (default: 50)",
    )
    shared.add_argument(
        "--per-pair", type=int, default=25,
        help="new Europe PMC articles fetched per under-target pair per round (default: 25)",
    )
    shared.add_argument(
        "--max-rounds", type=int, default=3,
        help="backfill discovery rounds in run-all, after the overview and manifestation "
        "rounds; each round re-plans from current coverage (default: 3)",
    )
    shared.add_argument(
        "--skip-review-passes", action="store_true",
        help="run-all: skip the overview and manifestation rounds and only backfill",
    )
    shared.add_argument(
        "--max-runtime-seconds", type=int, default=14400,
        help="runtime safety limit for run-all (default: 14400)",
    )
    shared.add_argument(
        "--finding-image-floor", type=int, default=None,
        help="minimum published images before a pair is considered staffed "
        "(default: VP_FINDING_IMAGE_FLOOR=3)",
    )
    shared.add_argument(
        "--finding-image-target", type=int, default=None,
        help="distinct published images targeted per approved (disease, finding) "
        "pair (default: VP_FINDING_IMAGE_TARGET=10)",
    )
    shared.add_argument(
        "--finding-gallery-cap", type=int, default=None,
        help="maximum published distinct representatives per pair; eligible "
        "surplus stays stored as reserves (default: VP_FINDING_GALLERY_CAP=20)",
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
    subs["discover"].add_argument(
        "--pass", dest="discovery_pass", choices=["overview", "manifestation", "backfill"],
        default="backfill",
        help="overview: reviews/case series surveying each disease; manifestation: "
        "reviews/case series about each under-target finding; backfill: any article "
        "type (default: backfill)",
    )
    subs["triage"].add_argument(
        "--retriage-montages",
        action="store_true",
        help="requeue caption_rejected montage/collage figures for the "
        "whole-figure plate policy before triaging pending rows",
    )
    subs["store"].add_argument(
        "--refresh",
        action="store_true",
        help="recompute panel attribution/subtype from stored vision_json",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overridden = False
    for flag, name in (
        ("finding_image_floor", "VP_FINDING_IMAGE_FLOOR"),
        ("finding_image_target", "VP_FINDING_IMAGE_TARGET"),
        ("finding_gallery_cap", "VP_FINDING_GALLERY_CAP"),
    ):
        value = getattr(args, flag, None)
        if value is not None:
            setattr(config, name, value)
            overridden = True
    if overridden:
        try:
            config.validate_coverage_settings()
        except ValueError as exc:
            print(f"error: {exc}")
            return 2
    fn = COMMANDS.get(args.command)
    if fn is None:
        return _cmd_not_implemented(args)
    return fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
