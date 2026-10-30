"""Thin-topic run: every catalog topic under the image target, with the thin-topic levers on.

Usage: venv/bin/python scripts/run_under10_all.py --database {main,new} [--target 10]
       [--max-rounds 4] [--tag NAME] [--dry-run]

``main`` covers the established topics in data/visual_pilot/visual_pilot.sqlite;
``new`` covers the 2026-10-28 additions in their isolated database and
vocabulary. Run one process per database; both may run at the same time.

Each run selects the database's topics below the target (default 10) in merged
(site + database) images, then:
1. returns ``no_finding_term`` caption drops that now pass the gate
   (every pass) and resets retryable ``vision_error`` figures;
2. re-plates ``vision_rejected`` collages with unlabeled plates allowed;
3. finishes those figures, then runs disease-pass discovery rounds, with the
   license-free second search tier, until every topic reaches the target, a round
   finds no new article, or the round or time limit is reached.
Library-wide text extraction is skipped, as in the earlier under-10 run.
"""

import argparse
import fcntl
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "data/visual_pilot"
NEW_DIR = RUN_DIR / "new_topics_20261028"
MAX_RUNTIME_SECONDS = 8 * 3600
BATCH_SIZE = 300
PER_PAIR = 25
# Same models and limits as the new-topic run, plus the two thin-topic levers.
OVERRIDES = {"VP_RELAXED_LICENSE_SEARCH": "1", "VP_ALLOW_UNLABELED_PLATES": "1"}

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--database", choices=["main", "new"], required=True)
parser.add_argument("--dry-run", action="store_true",
                    help="report the topics and requeue counts without writes, searches or LLM calls")
parser.add_argument("--target", type=int, default=10, help="raise every selected topic to this many merged images")
parser.add_argument("--max-rounds", type=int, default=4, help="discovery rounds per run")
parser.add_argument("--tag", help="status/log name prefix (default under<target>_all_<today>)")
ARGS = parser.parse_args()
TARGET = ARGS.target
MAX_ROUNDS = ARGS.max_rounds
TAG = ARGS.tag or f"under{TARGET}_all_{datetime.now().strftime('%Y%m%d')}"

NEW_MANIFEST = json.loads((ROOT / "data/new_topics_20261028.json").read_text())
os.environ.update(NEW_MANIFEST["environment"])
os.environ.update(OVERRIDES)
if ARGS.database == "new":
    os.environ["VP_DATA_DIR"] = str(NEW_DIR)

sys.path.insert(0, str(ROOT))
from src.visual_pilot import cli, config, db, diseases, discover, judge, published  # noqa: E402

if ARGS.database == "new":
    # Before anything builds the catalog: the isolated run's own vocabulary.
    diseases.TOPIC_VOCAB_PATH = NEW_DIR / "topic_findings_vocab.json"

NEW_IDS = set(NEW_MANIFEST["topics"])
NAME = f"{TAG}_{ARGS.database}"
STATUS = RUN_DIR / f"{NAME}_status.json"
status = {
    "pid": os.getpid(), "state": "running", "stage": "initializing", "database": ARGS.database,
    "db_path": str(config.db_path()),
    "started_at_utc": datetime.now(timezone.utc).isoformat(), "target": TARGET,
    "settings": OVERRIDES, "rounds": [],
}


def save():
    if ARGS.dry_run:
        return
    status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps(status, indent=2) + "\n")
    temp.replace(STATUS)


def select_topics(conn) -> dict[str, int]:
    keys = [r[0] for r in conn.execute("SELECT disease_key FROM diseases")]
    scope = [k for k in keys if (k in NEW_IDS) == (ARGS.database == "new")]
    counts = published.merged_topic_counts(conn, scope)
    return {k: n for k, n in counts.items() if n < TARGET}


def requeue(conn, topics: list[str], dry_run: bool) -> dict:
    return {
        "caption_drops_returned": discover.requeue_terms(conn, topics, dry_run=dry_run, all_passes=True),
        "vision_errors_reset": discover.requeue_errors(conn, topics, dry_run=dry_run),
        "plates": {k: v for k, v in judge.requeue_plates(conn, dry_run=dry_run, disease_keys=topics).items()
                   if k in ("examined", "requeued")},
        # Judgments the deterministic gate vetoed, re-validated under the
        # current attribution and plate policy (no LLM calls).
        "gates": {k: v for k, v in judge.requeue_gate_vetoes(conn, dry_run=dry_run, disease_keys=topics).items()
                  if k in ("examined", "requeued")},
    }


def execute(baseline: dict[str, int]) -> int:
    topics = sorted(baseline)
    args = cli.build_parser().parse_args([
        "run-all", "--diseases", *topics,
        "--topic-image-target", str(TARGET), "--batch-size", str(BATCH_SIZE),
        "--max-runtime-seconds", str(MAX_RUNTIME_SECONDS),
        "--skip-review-passes", "--skip-extract",
    ])
    cli._cmd_init(args)
    conn = db.init_db()
    try:
        status.update(stage="requeue", requeue=requeue(conn, topics, dry_run=False))
    finally:
        conn.close()
    save()
    print("UNDER10 requeue: " + json.dumps(status["requeue"]), flush=True)

    deadline = time.monotonic() + MAX_RUNTIME_SECONDS
    timing_path = RUN_DIR / f"reports/timings_{NAME}.json"
    if ARGS.database == "new":
        timing_path = NEW_DIR / f"reports/timings_{NAME}.json"
    run = cli._RunAll(args, deadline, timing_path)
    try:
        status["stage"] = "finish_requeued"
        save()
        rc = run.resume(topics)
        if rc:
            return rc
        for round_no in range(1, MAX_ROUNDS + 1):
            if run.should_stop():
                status["stop_reason"] = "runtime limit"
                break
            current = published.merged_topic_counts(run.read_conn, topics)
            active = sorted((k for k in topics if current[k] < TARGET), key=lambda k: (current[k], k))
            status.update(stage=f"discovery_round_{round_no}", **{f"topics_still_below_{TARGET}": len(active)},
                          current_counts=current)
            save()
            if not active:
                status["stop_reason"] = f"all selected topics reached {TARGET}"
                break
            print(f"UNDER10 round={round_no} topics={len(active)} "
                  f"zero={sum(current[k] == 0 for k in active)}", flush=True)
            stats: dict = {}

            def feed(put):
                feed_conn = db.connect()
                try:
                    with cli.timing.stage("discover"):
                        discovery = discover._DiscoveryPass(
                            feed_conn, "disease", per_pair=PER_PAIR,
                            max_runtime=deadline - time.monotonic(),
                            log=print, on_articles=put, should_stop=run.should_stop,
                        )
                        # Plan from the merged gallery, not the database-only count.
                        discovery.topic_counts = current
                        discovery.known = discover._known_by_disease(feed_conn)
                        stats.update(discovery.execute([(k, None, current[k]) for k in active]))
                finally:
                    feed_conn.close()

            rc = run.pipeline(feed, f"under10 round {round_no}")
            cli._write_timings_report(timing_path)
            status["rounds"].append({"round": round_no, "topics_searched": len(active),
                                     "new_articles": len(set(stats.get("pmcids", []))),
                                     "fetch_errors": stats.get("fetch_errors", 0)})
            save()
            if rc:
                return rc
            if not stats.get("pmcids"):
                status["stop_reason"] = "no new articles in the last round"
                break
        after = published.merged_topic_counts(run.read_conn, topics)
        summary = {
            "topics": len(topics),
            f"below_{TARGET}_before": len(topics), f"below_{TARGET}_after": sum(n < TARGET for n in after.values()),
            "zero_before": sum(n == 0 for n in baseline.values()),
            "zero_after": sum(n == 0 for n in after.values()),
            "images_added": sum(after.values()) - sum(baseline.values()),
            "before_counts": baseline, "after_counts": after,
        }
        status.update(stage="finished", summary=summary,
                      stop_reason=status.get("stop_reason", "round limit reached"))
        print("UNDER10 finished: " + json.dumps({k: v for k, v in summary.items()
                                                 if not k.endswith("_counts")}), flush=True)
        return 0
    finally:
        run.close()
        cli._write_timings_report(timing_path)


def dry_run() -> int:
    conn = db.connect()
    try:
        baseline = select_topics(conn)
        topics = sorted(baseline)
        print(f"Database: {ARGS.database} ({config.db_path()})")
        print(f"Models: text={NEW_MANIFEST['models']['text']}; vision={NEW_MANIFEST['models']['vision']}; "
              f"settings={OVERRIDES}")
        print(f"Topics below {TARGET}: {len(topics)} ({sum(n == 0 for n in baseline.values())} at zero)")
        print("Would requeue: " + json.dumps(requeue(conn, topics, dry_run=True)))
        print(f"Then up to {MAX_ROUNDS} discovery rounds within {MAX_RUNTIME_SECONDS // 3600} h. "
              "Dry run: no writes, searches or LLM calls.")
    finally:
        conn.close()
    return 0


def main() -> int:
    if ARGS.dry_run:
        return dry_run()
    with (RUN_DIR / f"{NAME}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"{NAME} is already running", file=sys.stderr)
            return 1
        conn = db.connect()
        try:
            baseline = select_topics(conn)
        finally:
            conn.close()
        status["topics"] = len(baseline)
        save()
        try:
            with cli._timestamped_output():
                rc = execute(baseline)
        except BaseException as exc:
            status.update(state="failed", error=f"{type(exc).__name__}: {exc}", exit_code=1)
            raise
        else:
            status.update(state="completed" if rc == 0 else "failed", exit_code=rc)
        finally:
            save()
        return rc


if __name__ == "__main__":
    raise SystemExit(main())
