"""CLI entry point: ``python -m src.visual_pilot.cli <stage> [flags]``.

Subcommands dispatch through the ``COMMANDS`` registry: each maps to a
``run(args)`` function (later workstreams plug in their stage modules here).
Unimplemented stages print "not implemented yet" and exit with code 2.
"""

from __future__ import annotations

import argparse
import io
import logging
import queue
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
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


def _write_timings_report(path=None, quiet: bool = False):
    """W0/C7: write the run's timings JSON (reports/timings_<utc>.json).

    run-all passes one path per run and rewrites it after every batch, so an
    interrupted or still-running run has a current report.
    """
    if path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = config.reports_dir() / f"timings_{stamp}.json"
    try:
        written = timing.write_report(path)
        if not quiet:
            print(f"run-all: timings written to {written}")
    except Exception as exc:  # noqa: BLE001 - reporting must never fail run-all
        print(f"run-all: could not write timings report: {exc}")
    return path


class _TimestampedStream(io.TextIOBase):
    """Prefix every complete output line with a local timestamp and flush.

    Stages print from several threads in run-all; partial writes are buffered
    per thread so a line is emitted whole.
    """

    def __init__(self, stream) -> None:
        self._stream = stream
        self._lock = threading.Lock()
        self._partial: dict[int, str] = {}

    def write(self, text: str) -> int:
        tid = threading.get_ident()
        *lines, rest = (self._partial.get(tid, "") + text).split("\n")
        self._partial[tid] = rest
        if lines:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with self._lock:
                self._stream.write("".join(f"[{stamp}] {line}\n" for line in lines))
                self._stream.flush()
        return len(text)

    def flush(self) -> None:
        with self._lock:
            for tid, rest in list(self._partial.items()):
                if rest:
                    self._stream.write(rest)
                    self._partial[tid] = ""
            self._stream.flush()


@contextmanager
def _timestamped_output():
    original = sys.stdout
    sys.stdout = _TimestampedStream(original)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stdout = original


def _lazy(module: str, attr: str = "run") -> CommandFn:
    """Resolve a stage module at call time so `cli --help` stays fast."""

    def fn(args: argparse.Namespace) -> int:
        import importlib

        mod = importlib.import_module(f".{module}", package=__package__)
        return getattr(mod, attr)(args)

    return fn


def _cmd_run_all(args: argparse.Namespace) -> int:
    with _timestamped_output():
        return _run_all(args)


# Seconds the triage worker waits for more discovered articles before it
# triages a partial batch, so judging never idles behind a slow search.
_FILL_WAIT_SECONDS = 20.0
_DONE = object()


def _run_all(args: argparse.Namespace) -> int:
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

    Within a round the stages overlap (``_pipeline``): discovery streams
    newly parsed articles to a triage thread, which hands batches of up to
    ``--batch-size`` articles to this thread for judge → store → describe.
    The next round plans only after the current one is fully drained.
    ``--max-runtime-seconds`` stops discovery and the next batch, leaving
    unfinished figures for the resume step of the next run.
    """
    from . import discover, judge

    started = time.monotonic()
    deadline = started + int(args.max_runtime_seconds)
    if args.dry_run:
        print(f"run-all dry-run: no writes; per-pair={args.per_pair}, "
              f"runtime={int(args.max_runtime_seconds)}s")
        return 0
    _cmd_init(args)
    timings_path = _write_timings_report(quiet=True)
    print(f"run-all: timings report: {timings_path}")
    read_conn = db.connect()
    read_lock = threading.Lock()  # read_conn is shared by the pipeline threads
    stop = threading.Event()

    def out_of_time() -> bool:
        return time.monotonic() >= deadline

    def should_stop() -> bool:
        return stop.is_set() or out_of_time()

    try:
        with read_lock:
            start_call_id = read_conn.execute(
                "SELECT COALESCE(MAX(call_id), 0) AS m FROM llm_calls"
            ).fetchone()["m"]

        def _remaining_budget():
            if args.budget_usd is None:
                return None
            with read_lock:
                spent = read_conn.execute(
                    "SELECT COALESCE(SUM(cost_usd), 0) AS s FROM llm_calls WHERE call_id > ?",
                    (start_call_id,),
                ).fetchone()["s"]
            return args.budget_usd - spent

        def _stage(name: str, pmcids, label: str) -> int:
            remaining = _remaining_budget()
            if remaining is not None and remaining <= 0:
                print(f"run-all: budget ${args.budget_usd:.2f} exhausted before {name}; stopping")
                return 4
            # One namespace per call: stages run on more than one thread.
            scoped = argparse.Namespace(**vars(args))
            scoped.budget_usd = remaining
            scoped.pmcids = pmcids
            print(f"run-all: {label}: {name}" + (f" ({len(pmcids)} article(s))" if pmcids else ""))
            with timing.stage(name):
                return COMMANDS[name](scoped)

        def _finish(pmcids, label: str) -> int:
            """judge (with vision_error retries) → store → describe."""
            rc = _stage("judge", pmcids, label)
            if rc != 0:
                return rc
            for _ in range(judge.MAX_ATTEMPTS - 1):
                marks = ",".join("?" for _ in pmcids)
                with read_lock:
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
            rc = _stage("store", pmcids, label)
            if rc != 0:
                return rc
            return _stage("describe", pmcids, label)

        def _pipeline(feed, prefix: str) -> int:
            """Run ``feed(put)`` on a producer thread and drain what it puts.

            ``put(pmcids)`` hands newly parsed articles to the triage thread;
            this thread judges, stores and describes each triaged batch.
            """
            arrivals: queue.Queue = queue.Queue()
            triaged: queue.Queue = queue.Queue(maxsize=2)
            errors: list[BaseException] = []
            failed_rc: list[int] = []

            def produce() -> None:
                try:
                    feed(arrivals.put)
                except BaseException as exc:  # noqa: BLE001 - re-raised below
                    errors.append(exc)
                    stop.set()
                finally:
                    arrivals.put(_DONE)

            def triage_worker() -> None:
                batch_no = 0
                seen: set[str] = set()
                finished = False
                try:
                    while not finished:
                        item = arrivals.get()
                        if item is _DONE:
                            break
                        chunk = list(item)
                        while len(chunk) < args.batch_size:
                            try:
                                item = arrivals.get(timeout=_FILL_WAIT_SECONDS)
                            except queue.Empty:
                                break
                            if item is _DONE:
                                finished = True
                                break
                            chunk.extend(item)
                        chunk = [p for p in dict.fromkeys(chunk) if p not in seen]
                        seen.update(chunk)
                        for i in range(0, len(chunk), args.batch_size):
                            if should_stop():
                                return
                            part = chunk[i : i + args.batch_size]
                            batch_no += 1
                            label = f"{prefix} batch {batch_no}"
                            rc = _stage("triage", part, label)
                            if rc != 0:
                                failed_rc.append(rc)
                                stop.set()
                                return
                            triaged.put((part, label))
                except BaseException as exc:  # noqa: BLE001 - re-raised below
                    errors.append(exc)
                    stop.set()
                finally:
                    triaged.put(_DONE)

            threads = [
                threading.Thread(target=produce, name="discover", daemon=True),
                threading.Thread(target=triage_worker, name="triage", daemon=True),
            ]
            for thread in threads:
                thread.start()
            rc = 0
            while True:
                item = triaged.get()
                if item is _DONE:
                    break
                if rc != 0 or should_stop():
                    continue  # keep draining so the triage thread can exit
                part, label = item
                rc = _finish(part, label)
                if rc != 0:
                    stop.set()
                _write_timings_report(timings_path, quiet=True)
            for thread in threads:
                thread.join()
            if errors:
                raise errors[0]
            if out_of_time() and not stop.is_set():
                print("run-all: runtime limit reached; unfinished figures resume next run")
            return rc or (failed_rc[0] if failed_rc else 0)

        def _feed_list(pmcids):
            def feed(put):
                for i in range(0, len(pmcids), args.batch_size):
                    if should_stop():
                        return
                    put(pmcids[i : i + args.batch_size])
            return feed

        if args.pmcids:
            rc = _pipeline(_feed_list(list(args.pmcids)), "explicit PMCIDs")
            if rc != 0:
                return rc
        else:
            disease_keys = list(diseases.DISEASE_KEYS) if args.disease == "all" else [args.disease]
            # Resume: finish figures an interrupted run left mid-pipeline
            # before discovering more articles.
            with read_lock:
                leftover = [
                    row["pmcid"] for row in read_conn.execute(
                        "SELECT DISTINCT f.pmcid FROM figures f JOIN articles a USING(pmcid) "
                        "WHERE (f.status IN ('pending','caption_kept','caption_uncertain','vision_accepted') "
                        "OR (f.status='vision_error' AND f.attempts < ?)) "
                        "AND EXISTS (SELECT 1 FROM json_each(a.primary_disease_keys_json) je "
                        "WHERE je.value IN (SELECT value FROM json_each(?))) "
                        "ORDER BY f.pmcid",
                        (judge.MAX_ATTEMPTS, db.to_json(disease_keys)),
                    )
                ]
            if leftover:
                print(f"run-all: resuming {len(leftover)} article(s) left mid-pipeline")
                rc = _pipeline(_feed_list(leftover), "resume")
                if rc != 0:
                    return rc
            schedule = (
                [] if args.skip_review_passes else ["overview", "manifestation"]
            ) + ["backfill"] * int(args.max_rounds)
            for round_no, pass_name in enumerate(schedule, 1):
                if should_stop():
                    print("run-all: runtime limit reached")
                    break
                print(f"run-all: round {round_no}: discover ({pass_name})")
                stats: dict = {}

                def feed(put, pass_name=pass_name, stats=stats):
                    sent: set[str] = set()

                    def stream(pmcids):
                        sent.update(pmcids)
                        put(pmcids)

                    conn = db.connect()
                    try:
                        with timing.stage("discover"):
                            stats.update(discover.discover(
                                conn, disease_keys, per_pair=args.per_pair,
                                target=int(config.VP_FINDING_IMAGE_TARGET),
                                max_runtime=deadline - time.monotonic(),
                                pass_name=pass_name, on_articles=stream,
                                should_stop=should_stop,
                            ))
                    finally:
                        conn.close()
                    unsent = [p for p in stats.get("pmcids") or [] if p not in sent]
                    if unsent:
                        put(unsent)

                rc = _pipeline(feed, f"round {round_no}")
                if rc != 0:
                    return rc
                if not stats.get("pmcids"):
                    if pass_name != "backfill":
                        print(f"run-all: {pass_name} pass found no new articles")
                        continue
                    print("run-all: no new articles for any under-target pair; done")
                    break

        if stop.is_set():
            return 0
        for name in ("extract", "report"):
            scoped_args = argparse.Namespace(**vars(args))
            scoped_args.pmcids = None
            remaining = _remaining_budget()
            if remaining is not None and remaining <= 0:
                print(f"run-all: budget ${args.budget_usd:.2f} exhausted before {name}; stopping")
                return 4
            scoped_args.budget_usd = remaining
            print(f"run-all: final: {name}")
            with timing.stage(name):
                rc = COMMANDS[name](scoped_args)
            if rc != 0:
                return rc
        return 0
    finally:
        read_conn.close()
        _write_timings_report(timings_path)


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
    "describe": _lazy("describe"),
    "extract": _lazy("extract_findings"),
    "report": _lazy("report"),
    "serve": _lazy("viewer"),
    "export-site": _lazy("site_export"),
    "build-vocab": _lazy("topic_vocab"),
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
        "--batch-size", type=int, default=150,
        help="articles per triage/judge/store batch in run-all; each batch waits for "
        "its slowest vision call, so larger batches waste fewer idle slots (default: 150)",
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
    subs["describe"].add_argument(
        "--force",
        action="store_true",
        help="rewrite display captions that already exist",
    )
    subs["extract"].add_argument(
        "--force",
        action="store_true",
        help="re-extract articles that already have source='text' findings",
    )
    subs["build-vocab"].add_argument(
        "--topics", nargs="*", default=None,
        help="topic_ids to (re)build; default: every topic without findings yet",
    )
    subs["build-vocab"].add_argument(
        "--force", action="store_true",
        help="rebuild topics that already have generated findings",
    )
    subs["export-site"].add_argument(
        "--out",
        default=None,
        help="static site output directory (default: site/ at the repo root)",
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
