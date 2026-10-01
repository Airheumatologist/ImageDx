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
    """Drain queued figure work, then expand coverage deficit-first per pair.

    No broad all-pair retrieval stage runs here; expansion is orchestrated
    through ``search_policy``/``scheduling_policy``: sync lanes, take one
    global gallery snapshot, replenish only actionable lanes that lack
    selectable candidates, reserve a shared global batch, and run the
    existing parse → triage → judge → store stages once per shared PMCID.
    ``--skip-select`` and explicit ``--pmcids`` never trigger replenishment,
    retrieval, or P1 calls. Retained ``vision_rejected`` plates are never
    requeued wholesale — drain covers only already-pending/retryable work.
    """
    from . import gallery, judge
    from . import manifestation_queue, search_policy, select_articles
    from . import parse as parse_stage

    started = time.monotonic()

    def _limit(name: str, default: int) -> int:
        # Explicit zero limits pause expansion; they are never reset to a
        # default by an ``or`` fallback.
        value = getattr(args, name, None)
        return default if value is None else int(value)

    max_runtime = _limit("max_runtime_seconds", 14400)
    max_articles = _limit("max_articles", 6000)
    batch_size = max(1, _limit("batch_size", 200))
    if args.dry_run:
        print(
            "run-all dry-run: no database writes, network retrieval, JATS fetch, "
            f"or LLM calls; planned batch size={batch_size}, "
            f"article safety limit={max_articles}, runtime={max_runtime}s"
        )
        return 0
    _cmd_init(args)
    # W9 task 5: one long-lived read connection for every bookkeeping query
    # (budget, snapshots, unfinished counts). WAL autocommit sees fresh
    # commits per statement; queue/replenishment writes commit explicitly.
    read_conn = db.connect()
    try:
        start_call_id = read_conn.execute(
            "SELECT COALESCE(MAX(call_id), 0) AS m FROM llm_calls"
        ).fetchone()["m"]
        budget0 = args.budget_usd
        processed = {"articles": 0}

        def _spent_since_start() -> float:
            return read_conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) AS s FROM llm_calls WHERE call_id > ?",
                (start_call_id,),
            ).fetchone()["s"]

        def _remaining_budget():
            return None if budget0 is None else budget0 - _spent_since_start()

        def _expansion_block() -> str | None:
            """Global caps checked before every expansion callback."""
            if time.monotonic() - started >= max_runtime:
                return "runtime_limit"
            remaining = _remaining_budget()
            if remaining is not None and remaining <= 0:
                return "budget"
            if processed["articles"] >= max_articles:
                return "article_limit"
            return None

        def _retriable_vision_errors(pmcids) -> int:
            """Cheap count of ``vision_error`` figures judge would still retry.

            Mirrors the ``attempts < judge.MAX_ATTEMPTS`` filter in ``judge.run``
            so the in-batch retry below only re-invokes judge when a pass could
            actually do work.
            """
            if not pmcids:
                return 0
            marks = ",".join("?" for _ in pmcids)
            return read_conn.execute(
                "SELECT COUNT(*) AS n FROM figures WHERE status='vision_error' "
                f"AND attempts < ? AND pmcid IN ({marks})",
                (judge.MAX_ATTEMPTS, *pmcids),
            ).fetchone()["n"]

        requested_pmcids = set(args.pmcids) if args.pmcids else None
        skip_select = bool(getattr(args, "skip_select", False))
        if skip_select:
            print("run-all: using committed article selections; no new retrieval")
        elif requested_pmcids is not None:
            print(
                f"run-all: {len(requested_pmcids)} explicit PMCIDs; "
                "no new retrieval or relevance calls"
            )

        disease_keys = list(diseases.DISEASE_KEYS) if args.disease == "all" else [args.disease]
        scoped = argparse.Namespace(**vars(args))

        def _run_batch_stages(disease_key: str, batch_no: int, pmcids: list[str]) -> int:
            """Run one batch through parse → triage → judge → store.

            ``vision_error`` figures that judge will still retry
            (``attempts < judge.MAX_ATTEMPTS``) get up to ``MAX_ATTEMPTS - 1``
            extra judge passes so transiently-failed figures recover and are
            stored in-batch instead of pausing the lane.
            """

            def _stage(name: str) -> int:
                remaining = _remaining_budget()
                if remaining is not None and remaining <= 0:
                    print(f"run-all: budget ${budget0:.2f} exhausted before {name}; stopping")
                    return 4
                scoped.budget_usd = remaining
                print(
                    f"run-all: {disease_key} batch {batch_no}: {name} "
                    f"({len(pmcids)} articles)"
                )
                with timing.stage(name):
                    return COMMANDS[name](scoped) if COMMANDS[name] else 2

            for name in ("parse", "triage", "judge"):
                rc = _stage(name)
                if rc != 0:
                    return rc
            for _ in range(judge.MAX_ATTEMPTS - 1):
                if not _retriable_vision_errors(pmcids):
                    break
                rc = _stage("judge")
                if rc != 0:
                    return rc
            return _stage("store")

        # Lane bookkeeping: transient pauses apply to this invocation only;
        # a persisted search_plan_exhausted block lasts until the policy
        # version changes (sync_candidates reopens it then). Clearing is
        # scoped to this run's diseases — a single --disease run must not
        # clear another disease's pauses.
        read_conn.execute(
            "UPDATE manifestation_lanes SET blocked_reason=NULL "
            "WHERE blocked_reason IS NOT NULL "
            "AND blocked_reason != 'search_plan_exhausted' "
            f"AND disease_key IN ({','.join('?' for _ in disease_keys)})",
            disease_keys,
        )
        read_conn.commit()

        def _pause_lane(disease_key: str, finding_key: str, reason: str) -> None:
            read_conn.execute(
                "UPDATE manifestation_lanes SET blocked_reason=?, "
                "last_outcome=?, updated_at=datetime('now') "
                "WHERE disease_key=? AND finding_key=?",
                (f"paused_{reason}", reason, disease_key, finding_key),
            )
            read_conn.commit()

        def _exhaust_lane(disease_key: str, finding_key: str) -> None:
            read_conn.execute(
                "UPDATE manifestation_lanes SET blocked_reason='search_plan_exhausted', "
                "search_policy_version=?, last_outcome='search_plan_exhausted', "
                "updated_at=datetime('now') WHERE disease_key=? AND finding_key=?",
                (config.PAIR_SEARCH_POLICY_VERSION, disease_key, finding_key),
            )
            read_conn.commit()

        # Resume downstream work left by an interrupted earlier invocation —
        # only already-pending/retryable figures; run-all never requeues
        # retained vision_rejected plates wholesale.
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
                with timing.stage(name):
                    rc = COMMANDS[name](scoped) if COMMANDS[name] else 2
                if rc != 0:
                    return rc

        batches = {"n": 0}

        def _execute_batch(disease_key: str, pmcids: list[str]) -> int:
            batches["n"] += 1
            scoped.disease = disease_key
            scoped.pmcids = pmcids
            scoped.limit = None
            rc = _run_batch_stages(disease_key, batches["n"], pmcids)
            if rc == 0:
                manifestation_queue.record_published_outcomes(read_conn, disease_key)
                processed["articles"] += len(pmcids)
            return rc

        if requested_pmcids is not None:
            # Explicit articles bypass the lane scheduler entirely — normal
            # publication checks still apply downstream. No retrieval/P1 here.
            wanted = set(requested_pmcids)
            for disease_key in disease_keys:
                while wanted and processed["articles"] < max_articles:
                    block = _expansion_block()
                    if block:
                        print(f"run-all: {block} reached; stopping expansion")
                        if block == "budget":
                            return 4
                        break
                    ranked = parse_stage.ranked_pending_articles(
                        read_conn, disease_key, pmcids=wanted
                    )
                    rows = [
                        row for row in ranked
                        if row["pmcid"] in wanted
                    ][:batch_size]
                    if not rows:
                        break
                    pmcids = [row["pmcid"] for row in rows]
                    wanted -= set(pmcids)
                    rc = _execute_batch(disease_key, pmcids)
                    if rc != 0:
                        return rc
        else:
            # Pair-replenishment orchestration: highest actionable tiers
            # first; empty lanes replenish before lower tiers expand.
            while True:
                block = _expansion_block()
                if block:
                    print(f"run-all: {block} reached; stopping expansion")
                    break
                for disease_key in disease_keys:
                    manifestation_queue.sync_candidates(read_conn, disease_key)
                snapshot = gallery.coverage_snapshot(read_conn)
                lanes = manifestation_queue.actionable_lanes(snapshot, disease_keys)
                if not lanes:
                    print("run-all: no actionable lanes; coverage work is done")
                    break
                replenished = False
                lane_outcomes: dict[tuple[str, str], str] = {}
                if not skip_select:
                    for lane in lanes:
                        block = _expansion_block()
                        if block:
                            break
                        pair = (lane["disease_key"], lane["finding_key"])
                        pending_rows = search_policy.pending_candidates(
                            read_conn, *pair
                        )
                        relevant_pending = any(
                            row["status"] == "relevant" for row in pending_rows
                        )
                        if relevant_pending:
                            continue  # reserve_global_batch picks it up
                        round_no = search_policy.next_round(read_conn, *pair)
                        if round_no is None and not pending_rows:
                            _exhaust_lane(*pair)
                            continue
                        remaining = _remaining_budget()
                        # round_no=0 is a drain-only call: no unattempted spec
                        # fires, but undrained pending candidates still get
                        # processed before the lane can be called exhausted.
                        result = select_articles.replenish_pair(
                            read_conn, *pair,
                            round_no=round_no if round_no is not None else 0,
                            budget_usd=remaining,
                            max_runtime_seconds=(
                                max_runtime - (time.monotonic() - started)
                            ),
                            max_articles=max_articles - processed["articles"],
                        )
                        replenished = True
                        status = result["status"]
                        lane_outcomes[pair] = status
                        if status == "search_plan_exhausted":
                            _exhaust_lane(*pair)
                        elif status == "retrieval_error":
                            print(
                                f"run-all: {pair[0]}/{pair[1]} retrieval error "
                                f"({result['reason']}); lane paused for this run"
                            )
                            _pause_lane(*pair, "retrieval_error")
                        elif status == "paused":
                            _pause_lane(*pair, result["reason"] or "limits")
                    if block:
                        print(f"run-all: {block} reached; stopping expansion")
                        break
                snapshot = gallery.coverage_snapshot(read_conn)
                lanes = manifestation_queue.actionable_lanes(snapshot, disease_keys)
                active_diseases = sorted({lane["disease_key"] for lane in lanes})
                ranked_by_disease = {
                    key: parse_stage.ranked_pending_articles(
                        read_conn,
                        key,
                        peek_limit=min(max(0, batch_size * 2), 100),
                    )
                    for key in active_diseases
                }
                selected = manifestation_queue.reserve_global_batch(
                    read_conn,
                    ranked_by_disease,
                    min(batch_size, max(0, max_articles - processed["articles"])),
                    disease_keys=active_diseases,
                    snapshot=snapshot,
                )
                if not selected:
                    # Continue only when this pass left provably-outstanding
                    # bounded work: a lane that completed a round still owns
                    # deeper unattempted rounds (300→600→1200), or a
                    # pending_work lane still holds real pending candidates.
                    # Termination is guaranteed: every issued attempt is
                    # ledgered and never refires (a round_complete lane's
                    # next_round strictly advances), exhausted/paused lanes
                    # leave actionable_lanes, and a lane merely awaiting an
                    # empty reservation produced no outcome this pass.
                    outstanding = False
                    if replenished:
                        snapshot = gallery.coverage_snapshot(read_conn)
                        for lane in manifestation_queue.actionable_lanes(
                            snapshot, disease_keys
                        ):
                            pair = (lane["disease_key"], lane["finding_key"])
                            outcome = lane_outcomes.get(pair)
                            if outcome == "round_complete" and (
                                search_policy.next_round(read_conn, *pair)
                                is not None
                            ):
                                outstanding = True
                            elif outcome == "pending_work" and (
                                search_policy.pending_candidates(
                                    read_conn, *pair
                                )
                            ):
                                outstanding = True
                            if outstanding:
                                break
                    if outstanding:
                        continue
                    print(
                        "run-all: no selectable candidates remain in "
                        "actionable lanes; stopping"
                    )
                    break
                groups: dict[str, list[str]] = {}
                for row in selected:
                    groups.setdefault(row["batch_disease_key"], []).append(row["pmcid"])
                for disease_key, pmcids in groups.items():
                    block = _expansion_block()
                    if block:
                        print(f"run-all: {block} reached; stopping expansion")
                        break
                    rc = _execute_batch(disease_key, pmcids)
                    if rc != 0:
                        return rc
                if block:
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
            with timing.stage(name):
                rc = COMMANDS[name](scoped) if COMMANDS[name] else 2
            if rc != 0:
                return rc
        return 0
    finally:
        read_conn.close()
        _write_timings_report()


# Registry: later workstreams replace the None entries with their module's
# run(args) function (e.g. select_articles.run, triage.run, ...).
COMMANDS: dict[str, CommandFn | None] = {
    "init": _cmd_init,
    "select": _lazy("select_articles"),
    "resume-selection": _lazy("select_articles", "run_resume"),
    "audit-licenses": _lazy("select_articles", "run_license_audit"),
    "parse": _lazy("parse"),
    "triage": _lazy("triage"),
    "judge": _lazy("judge"),
    "requeue-plates": _lazy("judge", "run_requeue_plates"),
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
        help="max items to process (select: license-passing articles per disease)",
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
        "--cap",
        type=int,
        default=None,
        help="legacy stage-2 report threshold; parse uses yield batches",
    )
    shared.add_argument(
        "--batch-size", type=int, default=200,
        help="articles per disease in a visual-yield expansion batch (default: 200)",
    )
    shared.add_argument(
        "--max-articles", type=int, default=6000,
        help="safety limit on articles processed per disease per run (default: 6000)",
    )
    shared.add_argument(
        "--max-runtime-seconds", type=int, default=14400,
        help="runtime safety limit for parse/run-all expansion (default: 14400)",
    )
    shared.add_argument(
        "--zero-yield-batches", type=int, default=2,
        help="stop a disease after this many consecutive batches add no images "
        "or findings, only when no approved pair remains under "
        "--finding-image-target images",
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
    for name in ('resume-selection', 'audit-licenses'):
        subs[name].add_argument('--finding', help='restrict to a persisted manifestation lane')
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
