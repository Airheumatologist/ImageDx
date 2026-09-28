"""CLI entry point: ``python -m src.visual_pilot.cli <stage> [flags]``.

Subcommands dispatch through the ``COMMANDS`` registry: each maps to a
``run(args)`` function (later workstreams plug in their stage modules here).
Unimplemented stages print "not implemented yet" and exit with code 2.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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


def _snapshot(conn, disease_key: str) -> tuple[set[str], set[str]]:
    """Yield baseline for one disease: distinct image hashes + covered findings.

    W9 task 5: the per-row ``findings_json`` Python scan is now a single
    aggregate ``json_each`` query. ``je.type='object'`` mirrors the old
    ``isinstance(value, dict)`` check and ``json_extract(...,'$.finding_key')``
    mirrors ``value.get("finding_key")`` (NULL when absent — never truthy).
    The nested ``CASE`` feeds ``json_each`` only valid array JSON: ``WHERE``
    filters cannot stop a table-valued function from evaluating invalid or
    non-array input, and ``json_type`` raises on malformed JSON, so the
    ``json_valid`` arm guards it (``AND`` is not guaranteed to short-circuit).
    The Python-side ``k and str(k) in approved`` filter applies the same
    truthiness/str() rules as the old loop, so results are identical.
    """
    hashes = {
        r["sha256"]
        for r in conn.execute(
            "SELECT DISTINCT sha256 FROM published_panels WHERE disease_key=? "
            "AND sha256 IS NOT NULL AND sha256 != ''",
            (disease_key,),
        )
    }
    approved = {
        r["finding_key"]
        for r in conn.execute(
            "SELECT finding_key FROM findings_vocab WHERE approved=1 "
            "AND EXISTS (SELECT 1 FROM json_each(disease_keys_json) je "
            "WHERE je.value=?)",
            (disease_key,),
        )
    }
    findings = {
        str(r["k"])
        for r in conn.execute(
            "SELECT DISTINCT CASE WHEN je.type='object' "
            "THEN json_extract(je.value,'$.finding_key') "
            "ELSE je.value END AS k "
            "FROM published_panels p, json_each(CASE WHEN json_valid(p.findings_json) "
            "THEN CASE WHEN json_type(p.findings_json)='array' "
            "THEN p.findings_json ELSE '[]' END ELSE '[]' END) je "
            "WHERE p.disease_key=?",
            (disease_key,),
        )
        if r["k"] and str(r["k"]) in approved
    }
    return hashes, findings


def _warm_jats_entry(article_row: dict) -> None:
    """Fetch + JATS-parse one article into ``parse._JATS_CACHE``.

    Prefetch worker step (W9 task 3): touches the network and the locked,
    bounded in-memory LRU only — no DB access, no disk writes. Best-effort: a
    failed warm just means the later caption peek/parse fetches it anyway.
    """
    from . import parse as parse_stage

    pmcid = article_row["pmcid"]
    try:
        entry = parse_stage._bundle_and_parsed(article_row)
    except Exception as exc:  # noqa: BLE001 - warm-up must never fail run-all
        logger.debug("run-all prefetch: %s not warmed (%s)", pmcid, exc)
        return
    parse_stage._jats_put(pmcid, entry)


def _prefetch_candidates(
    disease_key: str,
    batch_size: int,
    *,
    exclude_pmcids: frozenset[str],
    allowed_pmcids: frozenset[str] | None,
    pool: ThreadPoolExecutor,
) -> None:
    """Warm ``parse._JATS_CACHE`` for the *next* batch's likely candidates.

    W9 task 3 (fetch-only overlap): while batch N runs its stages, this ranks
    the disease's ``relevant`` articles the same way ``select_batch`` does and
    warms the top ``min(batch_size * 2, 100)`` (the peek limit ``select_batch``
    would use). It is 100% write-free — its own ``db.connect()`` is read-only
    (WAL sees fresh commits per statement) and it never calls
    ``ranked_pending_articles``, whose rescue path commits writes — so the real
    ``select_batch`` still runs on the main thread after batch N's ``store``
    (coverage gaps feed ranking; early selection is a proven parity hazard).
    """
    from . import article_rank
    from . import parse as parse_stage

    try:
        conn = db.connect()
        try:
            rows = [
                dict(row)
                for row in db.rows_with_status(
                    conn, "articles", "relevant", disease=disease_key
                )
            ]
            gaps = parse_stage.coverage_gaps(conn, disease_key)
        finally:
            conn.close()
        if allowed_pmcids is not None:
            rows = [row for row in rows if row["pmcid"] in allowed_pmcids]
        for row in rows:
            row["matched_passages"] = (
                db.from_json(row.get("retrieval_evidence_json"), []) or []
            )
        ranked = article_rank.rank_articles(
            rows, disease_key, coverage_gaps=gaps
        )
        candidates = [
            row
            for row, _score in ranked
            if row["pmcid"] not in exclude_pmcids
        ][: min(batch_size * 2, 100)]
    except Exception as exc:  # noqa: BLE001 - prefetch is pure warm-up
        logger.debug("run-all prefetch: candidate scan failed for %s (%s)", disease_key, exc)
        return
    for row in candidates:
        pool.submit(_warm_jats_entry, row)


def _cmd_run_all(args: argparse.Namespace) -> int:
    """Select once, then expand visual-yield batches until marginal yield falls."""
    from . import judge
    from . import parse as parse_stage

    started = time.monotonic()
    max_runtime = max(1, int(getattr(args, "max_runtime_seconds", 900) or 900))
    max_articles = max(1, int(getattr(args, "max_articles", 1200) or 1200))
    batch_size = max(1, int(getattr(args, "batch_size", 100) or 100))
    zero_yield_limit = max(1, int(getattr(args, "zero_yield_batches", 2) or 2))
    if args.dry_run:
        print(
            "run-all dry-run: no database writes, network retrieval, JATS fetch, "
            f"or LLM calls; planned batch size={batch_size}, "
            f"article safety limit={max_articles}/disease, runtime={max_runtime}s"
        )
        return 0
    _cmd_init(args)
    # W9 task 5: one long-lived read connection for every bookkeeping query
    # (budget, snapshots, unfinished counts, select_batch). WAL autocommit
    # sees fresh commits per statement; stage modules write on their own
    # connections, and select_batch's internal rescue commits explicitly.
    read_conn = db.connect()
    # W9 task 3: one small executor for the whole run's write-free prefetch
    # warm-up of parse._JATS_CACHE (submitted per batch, cancelled on exit).
    prefetch_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="vp-prefetch")
    try:
        start_call_id = read_conn.execute(
            "SELECT COALESCE(MAX(call_id), 0) AS m FROM llm_calls"
        ).fetchone()["m"]
        budget0 = args.budget_usd

        def _spent_since_start() -> float:
            return read_conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) AS s FROM llm_calls WHERE call_id > ?",
                (start_call_id,),
            ).fetchone()["s"]

        def _remaining_budget():
            return None if budget0 is None else budget0 - _spent_since_start()

        def _unfinished_figures(pmcids=None) -> int:
            placeholders = ""
            values: tuple = (
                "pending", "caption_kept", "caption_uncertain", "vision_accepted",
                judge.MAX_ATTEMPTS,
            )
            if pmcids:
                marks = ",".join("?" for _ in pmcids)
                placeholders = f" AND pmcid IN ({marks})"
                values += tuple(pmcids)
            return read_conn.execute(
                "SELECT COUNT(*) AS n FROM figures WHERE "
                "(status IN (?,?,?,?) OR (status='vision_error' AND attempts < ?))"
                + placeholders,
                values,
            ).fetchone()["n"]

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

        if COMMANDS["select"] is None:
            return 2
        requested_pmcids = set(args.pmcids) if args.pmcids else None
        if getattr(args, "skip_select", False):
            print("run-all: using committed article selections; selection skipped")
        elif requested_pmcids is None:
            print("run-all: select")
            args.budget_usd = _remaining_budget()
            with timing.stage("select"):
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

        def _run_batch_stages(disease_key: str, batch_no: int, pmcids: list[str]) -> int:
            """Run one batch through parse → triage → judge → store.

            W9: between judge and store, ``vision_error`` figures that judge will
            still retry (``attempts < judge.MAX_ATTEMPTS``) get up to
            ``MAX_ATTEMPTS - 1`` extra judge passes so transiently-failed figures
            recover and are stored in-batch instead of pausing the disease.
            ``scoped`` is already narrowed to this batch's disease and pmcids.
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

        # W0/C7: every stage call below is wrapped in timing.stage(...) and the
        # report is written on every exit path. Observation only.
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
                with timing.stage(name):
                    rc = COMMANDS[name](scoped) if COMMANDS[name] else 2
                if rc != 0:
                    return rc

        # W9: round-robin expansion across in-scope diseases. Each disease
        # keeps its own {processed, zero_yield, batch count} state and yields
        # the slot after every batch, so a paused/exhausted/zero-yielding
        # disease cannot starve the others on the shared runtime + budget.
        pending_diseases = deque(disease_keys)
        progress = {
            key: {"processed": 0, "zero_yield_batches": 0, "batches": 0}
            for key in disease_keys
        }
        while pending_diseases:
            if time.monotonic() - started >= max_runtime:
                print(f"run-all: runtime safety limit ({max_runtime}s) reached; stopping expansion")
                break
            remaining = _remaining_budget()
            if remaining is not None and remaining <= 0:
                print(f"run-all: budget ${budget0:.2f} exhausted; stopping expansion")
                return 4
            disease_key = pending_diseases.popleft()
            state = progress[disease_key]
            selected = parse_stage.select_batch(
                read_conn,
                disease_key,
                min(batch_size, per_disease_cap - state["processed"]),
                pmcids=requested_pmcids,
            )
            if requested_pmcids is not None:
                selected = [r for r in selected if r["pmcid"] in requested_pmcids]
            if not selected:
                continue  # finished: nothing left to select for this disease
            pmcids = [r["pmcid"] for r in selected]
            before_images, before_findings = _snapshot(read_conn, disease_key)
            scoped.disease = disease_key
            scoped.pmcids = pmcids
            scoped.limit = None
            state["batches"] += 1
            # W9 task 3: while this batch runs its stage sequence, warm
            # parse._JATS_CACHE for the next rotation's likely candidates on a
            # prefetch worker. Pure cache warming on the worker's own
            # connection — selection itself stays here, after this batch's
            # store, because coverage gaps feed the ranking.
            if pending_diseases:
                prefetch_target = pending_diseases[0]
            elif state["processed"] + len(pmcids) < per_disease_cap:
                prefetch_target = disease_key  # this disease can still continue
            else:
                prefetch_target = None
            if prefetch_target is not None:
                prefetch_pool.submit(
                    _prefetch_candidates,
                    prefetch_target,
                    batch_size,
                    exclude_pmcids=frozenset(pmcids),
                    allowed_pmcids=(
                        frozenset(requested_pmcids)
                        if requested_pmcids is not None
                        else None
                    ),
                    pool=prefetch_pool,
                )
            rc = _run_batch_stages(disease_key, state["batches"], pmcids)
            if rc != 0:
                return rc
            after_images, after_findings = _snapshot(read_conn, disease_key)
            new_images = after_images - before_images
            new_findings = after_findings - before_findings
            state["processed"] += len(pmcids)
            unfinished = _unfinished_figures(pmcids)
            if unfinished:
                print(
                    f"run-all: {disease_key} batch has {unfinished} unfinished figures; "
                    "leaving it resumable and pausing disease expansion"
                )
                continue  # finished: paused, stays resumable for the next run
            print(
                f"run-all: {disease_key} batch yield: {len(new_images)} distinct images, "
                f"{len(new_findings)} newly covered findings"
            )
            if new_images or new_findings:
                state["zero_yield_batches"] = 0
            else:
                state["zero_yield_batches"] += 1
                if state["zero_yield_batches"] >= zero_yield_limit:
                    print(f"run-all: {disease_key} stopped after {state['zero_yield_batches']} consecutive zero-yield batches")
                    continue  # finished: marginal yield fell
            if state["processed"] < per_disease_cap:
                pending_diseases.append(disease_key)

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
        # Pending warm-up jobs are pure cache warming — cancel rather than
        # wait; anything already running finishes harmlessly on its own conn.
        prefetch_pool.shutdown(wait=False, cancel_futures=True)
        read_conn.close()
        _write_timings_report()


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
        choices=[*diseases.disease_keys_from_catalog(), "all"],
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
        "--batch-size", type=int, default=100,
        help="articles per disease in a visual-yield expansion batch (default: 100)",
    )
    shared.add_argument(
        "--max-articles", type=int, default=1200,
        help="safety limit on articles processed per disease per run (default: 1200)",
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
