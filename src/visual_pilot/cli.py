"""CLI entry point: ``python -m src.visual_pilot.cli <stage> [flags]``.

Subcommands dispatch through the ``COMMANDS`` registry: each maps to a
``run(args)`` function (later workstreams plug in their stage modules here).
Unimplemented stages print "not implemented yet" and exit with code 2.
"""

from __future__ import annotations

import argparse
import faulthandler
import io
import queue
import signal
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone

from . import config, db, diseases, timing


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
    round (reviews and case series about each under-target finding), one
    ``disease`` round (any article titled with a topic still under
    ``VP_TOPIC_IMAGE_FLOOR`` published images), then ``backfill`` rounds over any article type, case reports included. Every
    round re-plans the pairs still under the image target (fewest images
    first), parses the matching articles and runs the LLM stages on just
    those articles, so case reports only fill what reviews left short.
    Backfill rounds repeat until no pair is under target, a round discovers
    nothing new, or the runtime/budget limit is reached. ``--pmcids`` skips discovery and only
    drains the stages for those articles. Figures an interrupted run left
    unfinished are drained before the first discovery round.

    Within a round the stages overlap (``_pipeline``): discovery streams
    newly parsed articles to a triage thread, which hands batches of up to
    ``--batch-size`` articles to this thread for judge → store → describe → audit-mimics.
    The next round plans only after the current one is fully drained.
    ``--max-runtime-seconds`` stops discovery and the next batch, leaving
    unfinished figures for the resume step of the next run.
    """
    started = time.monotonic()
    deadline = started + int(args.max_runtime_seconds)
    if args.dry_run:
        print(f"run-all dry-run: no writes; per-pair={args.per_pair}, "
              f"runtime={int(args.max_runtime_seconds)}s")
        return 0
    _cmd_init(args)
    timings_path = _write_timings_report(quiet=True)
    print(f"run-all: timings report: {timings_path}")
    run = _RunAll(args, deadline, timings_path)
    try:
        return run.execute()
    finally:
        run.close()
        _write_timings_report(timings_path)


_LEFTOVER_SQL = (
    "SELECT DISTINCT f.pmcid FROM figures f JOIN articles a USING(pmcid) "
    "WHERE (f.status IN ('pending','caption_kept','caption_uncertain','vision_accepted') "
    "OR (f.status='vision_error' AND f.attempts < ?)) "
    "AND EXISTS (SELECT 1 FROM json_each(a.primary_disease_keys_json) je "
    "WHERE je.value IN (SELECT value FROM json_each(?))) "
    "ORDER BY f.pmcid"
)


class _RunAll:
    """State shared by one ``run-all`` invocation's rounds and pipeline threads."""

    def __init__(self, args: argparse.Namespace, deadline: float, timings_path) -> None:
        self.args = args
        self.deadline = deadline
        self.timings_path = timings_path
        self.read_conn = db.connect()
        self.read_lock = threading.Lock()  # read_conn is shared by the pipeline threads
        self.stop = threading.Event()
        self.start_call_id = 0

    def close(self) -> None:
        self.read_conn.close()

    def out_of_time(self) -> bool:
        return time.monotonic() >= self.deadline

    def should_stop(self) -> bool:
        return self.stop.is_set() or self.out_of_time()

    def _read(self, sql: str, params=()) -> list:
        with self.read_lock:
            return self.read_conn.execute(sql, params).fetchall()

    def remaining_budget(self):
        if self.args.budget_usd is None:
            return None
        spent = self._read(
            "SELECT COALESCE(SUM(cost_usd), 0) AS s FROM llm_calls WHERE call_id > ?",
            (self.start_call_id,),
        )[0]["s"]
        return self.args.budget_usd - spent

    def _scoped_args(self, name: str, pmcids):
        """A per-call copy of args with the remaining budget; None once it is spent."""
        remaining = self.remaining_budget()
        if remaining is not None and remaining <= 0:
            print(f"run-all: budget ${self.args.budget_usd:.2f} exhausted before {name}; stopping")
            return None
        # One namespace per call: stages run on more than one thread.
        scoped = argparse.Namespace(**vars(self.args))
        scoped.budget_usd = remaining
        scoped.pmcids = pmcids
        return scoped

    def stage(self, name: str, pmcids, label: str) -> int:
        scoped = self._scoped_args(name, pmcids)
        if scoped is None:
            return 4
        print(f"run-all: {label}: {name}" + (f" ({len(pmcids)} article(s))" if pmcids else ""))
        with timing.stage(name):
            return COMMANDS[name](scoped)

    def _retryable(self, pmcids) -> int:
        """Figures among ``pmcids`` whose vision call failed but may be retried."""
        from . import judge

        if not pmcids:
            return 0
        marks = ",".join("?" for _ in pmcids)
        return self._read(
            "SELECT COUNT(*) AS n FROM figures WHERE status='vision_error' "
            f"AND attempts < ? AND pmcid IN ({marks})",
            (judge.MAX_ATTEMPTS, *pmcids),
        )[0]["n"]

    def finish(self, pmcids, label: str) -> int:
        """judge (with vision_error retries) → store → describe → audit-mimics."""
        from . import judge

        rc = self.stage("judge", pmcids, label)
        for _ in range(judge.MAX_ATTEMPTS - 1):
            if rc != 0 or not self._retryable(pmcids):
                break
            rc = self.stage("judge", pmcids, label)
        if rc != 0:
            return rc
        return (
            self.stage("store", pmcids, label)
            or self.stage("describe", pmcids, label)
            or self.stage("audit-mimics", pmcids, label)
        )

    def pipeline(self, feed, prefix: str) -> int:
        return _Pipeline(self, prefix).run(feed)

    def feed_list(self, pmcids):
        def feed(put):
            for i in range(0, len(pmcids), self.args.batch_size):
                if self.should_stop():
                    return
                put(pmcids[i : i + self.args.batch_size])
        return feed

    def execute(self) -> int:
        self.start_call_id = self._read(
            "SELECT COALESCE(MAX(call_id), 0) AS m FROM llm_calls"
        )[0]["m"]
        if self.args.pmcids:
            rc = self.pipeline(self.feed_list(list(self.args.pmcids)), "explicit PMCIDs")
        else:
            rc = self.discover_rounds(diseases.selected_keys(self.args))
        if rc != 0:
            return rc
        if self.stop.is_set():
            return 0
        return self.final_stages()

    def resume(self, disease_keys) -> int:
        """Finish figures an interrupted run left mid-pipeline before discovering more."""
        from . import judge

        leftover = [
            row["pmcid"]
            for row in self._read(_LEFTOVER_SQL, (judge.MAX_ATTEMPTS, db.to_json(disease_keys)))
        ]
        if not leftover:
            return 0
        if self.args.skip_resume:
            print(f"run-all: skipping {len(leftover)} article(s) left mid-pipeline (--skip-resume)")
            return 0
        print(f"run-all: resuming {len(leftover)} article(s) left mid-pipeline")
        return self.pipeline(self.feed_list(leftover), "resume")

    def _schedule(self) -> list[str]:
        reviews = [] if self.args.skip_review_passes else ["overview", "manifestation"]
        return reviews + ["disease"] + ["backfill"] * int(self.args.max_rounds)

    def _topics_under_target(self, disease_keys) -> list[str]:
        from . import pair_reporting

        target = self.args.topic_image_target
        with self.read_lock:
            counts = pair_reporting.topic_library_counts(self.read_conn)
        keys = [k for k in disease_keys if counts.get(k, 0) < target]
        print(f"run-all: {len(keys)} topic(s) still under {target} images")
        return keys

    def discover_rounds(self, disease_keys) -> int:
        rc = self.resume(disease_keys)
        if rc != 0:
            return rc
        for round_no, pass_name in enumerate(self._schedule(), 1):
            if self.should_stop():
                print("run-all: runtime limit reached")
                break
            if self.args.topic_image_target:
                disease_keys = self._topics_under_target(disease_keys)
                if not disease_keys:
                    break
            print(f"run-all: round {round_no}: discover ({pass_name})")
            stats: dict = {}
            rc = self.pipeline(self._discover_feed(disease_keys, pass_name, stats), f"round {round_no}")
            if rc != 0:
                return rc
            if stats.get("pmcids"):
                continue
            if pass_name != "backfill":
                print(f"run-all: {pass_name} pass found no new articles")
                continue
            print("run-all: no new articles for any under-target pair; done")
            break
        return 0

    def _discover_feed(self, disease_keys, pass_name: str, stats: dict):
        """A feed that runs one discovery pass, streaming parsed articles to ``put``."""
        from . import discover

        def feed(put):
            sent: set[str] = set()

            def stream(pmcids):
                sent.update(pmcids)
                put(pmcids)

            conn = db.connect()
            try:
                with timing.stage("discover"):
                    stats.update(discover.discover(
                        conn, disease_keys, per_pair=self.args.per_pair,
                        target=int(config.VP_FINDING_IMAGE_TARGET),
                        max_runtime=self.deadline - time.monotonic(),
                        pass_name=pass_name, on_articles=stream,
                        should_stop=self.should_stop,
                    ))
            finally:
                conn.close()
            unsent = [p for p in stats.get("pmcids") or [] if p not in sent]
            if unsent:
                put(unsent)

        return feed

    def final_stages(self) -> int:
        for name in ("extract", "report"):
            if name == "extract" and self.args.skip_extract:
                print("run-all: final: extract skipped (--skip-extract)")
                continue
            rc = self.stage(name, None, "final")
            if rc != 0:
                return rc
        return 0


class _Pipeline:
    """One overlapped round: a producer thread feeds a triage thread, and the
    calling thread judges, stores and describes each triaged batch.

    ``feed(put)`` runs on the producer thread; ``put(pmcids)`` hands newly
    parsed articles to the triage thread.
    """

    def __init__(self, run: _RunAll, prefix: str) -> None:
        self.run_all = run
        self.prefix = prefix
        self.arrivals: queue.Queue = queue.Queue()
        self.triaged: queue.Queue = queue.Queue(maxsize=2)
        self.errors: list[BaseException] = []
        self.failed_rc: list[int] = []
        self.batch_no = 0
        self.seen: set[str] = set()

    def _fail(self, exc: BaseException) -> None:
        self.errors.append(exc)
        self.run_all.stop.set()

    def _produce(self, feed) -> None:
        try:
            feed(self.arrivals.put)
        except BaseException as exc:  # noqa: BLE001 - re-raised in run()
            self._fail(exc)
        finally:
            self.arrivals.put(_DONE)

    def _gather(self, first) -> tuple[list[str], bool]:
        """Top ``first`` up to a batch, waiting briefly; True once the feed is done."""
        chunk = list(first)
        while len(chunk) < self.run_all.args.batch_size:
            try:
                item = self.arrivals.get(timeout=_FILL_WAIT_SECONDS)
            except queue.Empty:
                break
            if item is _DONE:
                return chunk, True
            chunk.extend(item)
        return chunk, False

    def _triage_chunk(self, chunk: list[str]) -> bool:
        """Triage the unseen articles batch by batch; False stops the worker."""
        chunk = [p for p in dict.fromkeys(chunk) if p not in self.seen]
        self.seen.update(chunk)
        size = self.run_all.args.batch_size
        for i in range(0, len(chunk), size):
            if self.run_all.should_stop():
                return False
            part = chunk[i : i + size]
            self.batch_no += 1
            label = f"{self.prefix} batch {self.batch_no}"
            rc = self.run_all.stage("triage", part, label)
            if rc != 0:
                self.failed_rc.append(rc)
                self.run_all.stop.set()
                return False
            self.triaged.put((part, label))
        return True

    def _triage_worker(self) -> None:
        try:
            finished = False
            while not finished:
                item = self.arrivals.get()
                if item is _DONE:
                    break
                chunk, finished = self._gather(item)
                if not self._triage_chunk(chunk):
                    return
        except BaseException as exc:  # noqa: BLE001 - re-raised in run()
            self._fail(exc)
        finally:
            self.triaged.put(_DONE)

    def _drain(self) -> int:
        rc = 0
        while (item := self.triaged.get()) is not _DONE:
            if rc != 0 or self.run_all.should_stop():
                continue  # keep draining so the triage thread can exit
            part, label = item
            rc = self.run_all.finish(part, label)
            if rc != 0:
                self.run_all.stop.set()
            _write_timings_report(self.run_all.timings_path, quiet=True)
        return rc

    def run(self, feed) -> int:
        threads = [
            threading.Thread(target=self._produce, args=(feed,), name="discover", daemon=True),
            threading.Thread(target=self._triage_worker, name="triage", daemon=True),
        ]
        for thread in threads:
            thread.start()
        rc = self._drain()
        for thread in threads:
            thread.join()
        if self.errors:
            raise self.errors[0]
        if self.run_all.out_of_time() and not self.run_all.stop.is_set():
            print("run-all: runtime limit reached; unfinished figures resume next run")
        return rc or (self.failed_rc[0] if self.failed_rc else 0)


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
    "audit-mimics": _lazy("mimic_audit"),
    "extract": _lazy("extract_findings"),
    "report": _lazy("report"),
    "serve": _lazy("viewer"),
    "export-site": _lazy("site_export"),
    "build-vocab": _lazy("topic_vocab"),
    "expand-terms": _lazy("topic_vocab", "run_expand"),
    "requeue-terms": _lazy("discover", "run_requeue"),
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
        "--diseases",
        nargs="+",
        default=None,
        help="several disease keys; overrides --disease (run-all, discover, "
        "expand-terms, requeue-terms)",
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
        help="run-all: skip the overview and manifestation rounds; the disease "
        "round and backfill still run",
    )
    shared.add_argument(
        "--topic-image-target", type=int, default=None,
        help="run-all: drop a topic from later rounds once its default library "
        "shows this many distinct images (default: every round runs every topic)",
    )
    shared.add_argument(
        "--skip-resume", action="store_true",
        help="run-all: leave figures an earlier run left mid-pipeline (e.g. to "
        "another run working on them) and go straight to discovery",
    )
    shared.add_argument(
        "--skip-extract", action="store_true",
        help="run-all: skip the final library-wide text extraction (report still runs)",
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
        "--pass", dest="discovery_pass", choices=["overview", "manifestation", "disease", "backfill"],
        default="backfill",
        help="overview: reviews/case series surveying each disease; manifestation: "
        "reviews/case series about each under-target finding; backfill: any article "
        "type; disease: case reports naming a topic under VP_TOPIC_IMAGE_FLOOR "
        "images in the title or abstract, and any article titled with it "
        "(default: backfill)",
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
    subs["expand-terms"].add_argument(
        "--force", action="store_true",
        help="re-expand topics already expanded with the current prompt",
    )
    subs["requeue-terms"].add_argument(
        "--retry-errors", action="store_true",
        help="also reset retryable vision_error figures (not provider content refusals)",
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
    subs["export-site"].add_argument(
        "--merge",
        action="store_true",
        help="keep every image the existing site shows and add the database's "
        "new ones per topic, up to VP_TOPIC_IMAGE_CAP (for a database that holds "
        "only some topics or images)",
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
    # `kill -USR1 <pid>` dumps every thread's stack to stderr (hung runs).
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)
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
