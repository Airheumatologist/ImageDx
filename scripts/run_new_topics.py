"""Run only the approved 493 additions with the saved October run settings.

Usage: venv/bin/python scripts/run_new_topics.py [--dry-run] [--isolated]
No discovery or LLM work is performed by --dry-run.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/new_topics_20261028.json"
RUN_DIR = ROOT / "data/visual_pilot"
STATUS = RUN_DIR / "new_topics_20261028_status.json"
LOG = RUN_DIR / "run_new_topics_20261028.log"
VOCAB = ROOT / "src/visual_pilot/data/topic_findings_vocab.json"


def load_manifest():
    manifest = json.loads(MANIFEST.read_text())
    keys = manifest["topics"]
    catalog = json.loads((ROOT / manifest["catalog_source"]).read_text())
    existing = {t["topic_id"] for rows in catalog["specialties"].values() for t in rows}
    approved = json.loads((ROOT / manifest["review_source"]).read_text())
    approved_keys = {t["topic_id"] for t in approved["topics"]}
    if len(keys) != 493 or len(set(keys)) != 493 or set(keys) != approved_keys:
        raise ValueError("Run manifest must contain exactly the 493 approved additions.")
    if not set(keys) <= existing:
        raise ValueError("Run manifest names topics missing from the main catalog.")
    if manifest["openrouter_only"] != ["open-inference"]:
        raise ValueError("Run must retain the recorded OpenInference provider restriction.")
    return manifest


def missing_vocab(keys):
    from src.visual_pilot import diseases
    valid = set(diseases.load_diseases())
    rows = json.loads(VOCAB.read_text()) if VOCAB.exists() else []
    have = set()
    for row in rows:
        try:
            diseases._validate_vocab_item(row, valid)
        except (ValueError, KeyError, TypeError):
            continue
        if row.get("caption_terms"):
            have.update(row["disease_keys"])
    return [key for key in keys if key not in have]


def pipeline_args(manifest):
    return ["run-all", "--diseases", *manifest["topics"],
            "--topic-image-target", str(manifest["topic_image_target"]),
            "--batch-size", str(manifest["batch_size"]),
            "--max-runtime-seconds", str(manifest["max_runtime_seconds"]),
            "--per-pair", str(manifest["per_pair"]),
            "--max-rounds", str(manifest["max_rounds"]),
            *(["--skip-review-passes"] if manifest["skip_review_passes"] else [])]


def save_status(status):
    status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps(status, indent=2) + "\n")
    temp.replace(STATUS)


def run_stage(name, args, environment, status, log):
    status.update(stage=name, state="running")
    save_status(status)
    header = f"NEW TOPICS stage={name} at {datetime.now(timezone.utc).isoformat()}\n"
    print(header, end="", flush=True)
    log.write(header)
    log.flush()
    if environment.get("VP_RUN_TOPIC_VOCAB"):
        # Override this process's vocabulary before CLI stages import it. The
        # shared source file and other running interpreters remain untouched.
        entry = ["-c", "import os; from pathlib import Path; "
                 "from src.visual_pilot import diseases; "
                 "diseases.TOPIC_VOCAB_PATH = Path(os.environ['VP_RUN_TOPIC_VOCAB']); "
                 "from src.visual_pilot.cli import main; raise SystemExit(main())"]
    else:
        entry = ["-m", "src.visual_pilot.cli"]
    child = subprocess.Popen([sys.executable, "-u", *entry, *args],
                             cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
    status["child_pid"] = child.pid
    save_status(status)
    try:
        for line in child.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        rc = child.wait()
    except BaseException:
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        raise
    finally:
        child.stdout.close()
        status.pop("child_pid", None)
    if rc:
        raise RuntimeError(f"Stage {name} exited with code {rc}; rerun the launcher to resume.")


def main(argv=None):
    global VOCAB
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="validate scope and print the plan without writes or API calls")
    parser.add_argument("--isolated", action="store_true", help="use a private database and vocabulary while another job is running")
    args = parser.parse_args(argv)
    manifest = load_manifest()
    # Apply these before importing pipeline modules, which cache config on import.
    environment = {**os.environ, **manifest["environment"]}
    if args.isolated:
        private = RUN_DIR / "new_topics_20261028"
        VOCAB = private / "topic_findings_vocab.json"
        environment.update(VP_DATA_DIR=str(private), VP_RUN_TOPIC_VOCAB=str(VOCAB))
    os.environ.update(manifest["environment"])
    missing = missing_vocab(manifest["topics"])
    print(f"Scope: {len(manifest['topics'])} approved new topics; {len(missing)} need vocabulary.")
    print(f"Models: text={manifest['models']['text']}; vision={manifest['models']['vision']}; provider=open-inference.")
    print(f"Run: target={manifest['topic_image_target']} images/topic; batch={manifest['batch_size']}; "
          f"runtime={manifest['max_runtime_seconds']}s; per-pair={manifest['per_pair']}; "
          f"backfill-rounds={manifest['max_rounds']}; skip-review-passes={manifest['skip_review_passes']}.")
    print("Stages: build missing vocabularies → validate coverage → expand new-topic terms → run-all.")
    if args.isolated:
        print(f"Isolated database: {private / 'visual_pilot.sqlite'}; vocabulary: {VOCAB}.")
    if args.dry_run:
        print("Dry run complete: no database writes, network requests or LLM calls.")
        return 0
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with (RUN_DIR / "new_topics_20261028.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This new-topic launcher is already running.")
        if args.isolated:
            private.mkdir(parents=True, exist_ok=True)
            if not VOCAB.exists():
                VOCAB.write_text("[]\n")
        status = {"pid": os.getpid(), "state": "running", "topic_count": len(manifest["topics"]),
                  "manifest": str(MANIFEST), "log": str(LOG),
                  "isolated": args.isolated,
                  "database": str(private / "visual_pilot.sqlite") if args.isolated else str(RUN_DIR / "visual_pilot.sqlite"),
                  "vocabulary": str(VOCAB),
                  "started_at_utc": datetime.now(timezone.utc).isoformat()}
        with LOG.open("a", buffering=1) as log:
            try:
                if missing:
                    run_stage("build_vocab", ["build-vocab", "--topics", *missing], environment, status, log)
                missing = missing_vocab(manifest["topics"])
                if missing:
                    raise RuntimeError(f"Vocabulary incomplete for {len(missing)} topics: {', '.join(missing)}. "
                                       "Acquisition was not started. Rerun to retry missing topics.")
                run_stage("expand_terms", ["expand-terms", "--diseases", *manifest["topics"]], environment, status, log)
                run_stage("pipeline", pipeline_args(manifest), environment, status, log)
            except BaseException as exc:
                status.update(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
                save_status(status)
                raise
            else:
                status.update(state="completed", exit_code=0)
                save_status(status)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
