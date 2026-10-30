"""Remove topics whose published gallery holds too few images.

Usage: venv/bin/python scripts/prune_thin_topics.py [--max-images 4] [--apply]

Without --apply the script only reports what it would remove. With --apply it
removes the topics from the topic catalog, the 2026-10-28 manifest and id list,
the search-name and vocabulary files, the exported site (topic folders, the
disease index and now-unreferenced articles) and from both pipeline databases
(every row keyed by the topic, and the curation rows of its panels). Both
databases are copied to data/visual_pilot/backup_prune_<date>/ first, and the
removed catalog entries are written next to them so the deletion can be undone.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site" / "data"
CATALOG = ROOT / "data" / "main_database_topics.json"
MANIFEST = ROOT / "data" / "new_topics_20261028.json"
ID_LIST = ROOT / "data" / "new_topic_ids_20261028.txt"
REVIEW = ROOT / "outputs" / "final_new_topics" / "final_new_topics.json"
SEARCH_NAMES = ROOT / "src" / "visual_pilot" / "data" / "topic_search_names.json"
BASE_VOCAB = ROOT / "src" / "visual_pilot" / "data" / "findings_vocab.json"
TOPIC_VOCABS = [ROOT / "src" / "visual_pilot" / "data" / "topic_findings_vocab.json",
                ROOT / "data" / "visual_pilot" / "new_topics_20261028" / "topic_findings_vocab.json"]
DATABASES = [ROOT / "data" / "visual_pilot" / "visual_pilot.sqlite",
             ROOT / "data" / "visual_pilot" / "new_topics_20261028" / "visual_pilot.sqlite"]


def load(path: Path):
    return json.loads(path.read_text())


def dump(path: Path, data, apply: bool) -> None:
    if apply:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def site_counts() -> dict[str, int]:
    counts = {}
    for folder in SITE.iterdir():
        panels = folder / "panels.json"
        if panels.exists():
            counts[folder.name] = load(panels)["count"]
    return counts


def prune_catalog(thin: set[str], apply: bool) -> list[dict]:
    catalog = load(CATALOG)
    removed = []
    for specialty, topics in catalog["specialties"].items():
        keep = []
        for topic in topics:
            (removed if topic["topic_id"] in thin else keep).append(topic)
        catalog["specialties"][specialty] = keep
    catalog["specialties"] = {k: v for k, v in catalog["specialties"].items() if v}
    meta = catalog["metadata"]
    meta["total_topics"] = sum(len(v) for v in catalog["specialties"].values())
    meta["total_specialties"] = len(catalog["specialties"])
    meta["audited_on"] = dt.date.today().isoformat()
    meta["latest_prune"] = {"date": dt.date.today().isoformat(), "removed_topics": len(removed),
                            "reason": "fewer than five published images"}
    dump(CATALOG, catalog, apply)
    return removed


def prune_manifest(thin: set[str], apply: bool, catalog_total: int) -> int:
    manifest = load(MANIFEST)
    before = len(manifest["topics"])
    manifest["topics"] = [t for t in manifest["topics"] if t not in thin]
    manifest["total_topics"] = len(manifest["topics"])
    manifest["catalog_topic_count"] = catalog_total
    kept = set(manifest["topics"])
    review = {t["topic_id"]: t for t in load(REVIEW)["topics"]} if REVIEW.exists() else {}
    if review:
        priorities, specialties = {}, {}
        for key in manifest["topics"]:
            entry = review.get(key, {})
            priorities[entry.get("priority", "?")] = priorities.get(entry.get("priority", "?"), 0) + 1
            specialties[entry.get("specialty", "?")] = specialties.get(entry.get("specialty", "?"), 0) + 1
        manifest["priority_counts"] = dict(sorted(priorities.items()))
        manifest["specialty_counts"] = dict(sorted(specialties.items()))
    dump(MANIFEST, manifest, apply)
    if apply:
        ID_LIST.write_text("\n".join(t for t in ID_LIST.read_text().split() if t in kept) + "\n")
    return before - len(manifest["topics"])


def prune_json_files(thin: set[str], apply: bool) -> dict[str, int]:
    out = {}
    names = load(SEARCH_NAMES)
    out["search_names"] = sum(k in thin for k in names)
    dump(SEARCH_NAMES, {k: v for k, v in names.items() if k not in thin}, apply)
    for path in [BASE_VOCAB, *TOPIC_VOCABS]:
        rows = load(path)
        kept = []
        for row in rows:
            keys = [k for k in row["disease_keys"] if k not in thin]
            if keys:
                row["disease_keys"] = keys
                kept.append(row)
        out[path.name if path != TOPIC_VOCABS[1] else "new_topics/" + path.name] = len(rows) - len(kept)
        dump(path, kept, apply)
    return out


def prune_site(thin: set[str], apply: bool) -> dict[str, int]:
    out = {"folders": 0}
    for key in thin:
        folder = SITE / key
        if folder.is_dir():
            out["folders"] += 1
            if apply:
                shutil.rmtree(folder)
    diseases = load(SITE / "diseases.json")
    out["diseases"] = sum(d["key"] in thin for d in diseases)
    dump(SITE / "diseases.json", [d for d in diseases if d["key"] not in thin], apply)
    referenced = set()
    for folder in SITE.iterdir():
        panels = folder / "panels.json"
        if folder.name not in thin and panels.exists():
            referenced.update(p["pmcid"] for p in load(panels)["panels"])
    articles = load(SITE / "articles.json")
    kept = [a for a in articles if a["pmcid"] in referenced]
    out["articles"] = len(articles) - len(kept)
    dump(SITE / "articles.json", kept, apply)
    return out


def prune_database(path: Path, thin: set[str], apply: bool, backup_dir: Path) -> dict[str, int]:
    if apply:
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_dir / f"{path.parent.name}_{path.name}")
    conn = sqlite3.connect(path)
    marks = ",".join("?" * len(thin))
    keys = sorted(thin)
    out = {}
    tables = [r[0] for r in conn.execute("select name from sqlite_master where type='table'")]
    columns = {t: [r[1] for r in conn.execute(f"pragma table_info({t})")] for t in tables}
    panel_ids = [r[0] for r in conn.execute(f"select panel_id from panels where disease_key in ({marks})", keys)]
    for table, cols in columns.items():
        if "disease_key" in cols:
            out[table] = conn.execute(f"select count(*) from {table} where disease_key in ({marks})", keys).fetchone()[0]
            if apply:
                conn.execute(f"delete from {table} where disease_key in ({marks})", keys)
        elif "panel_id" in cols and panel_ids:
            pm = ",".join("?" * len(panel_ids))
            out[table] = conn.execute(f"select count(*) from {table} where panel_id in ({pm})", panel_ids).fetchone()[0]
            if apply:
                conn.execute(f"delete from {table} where panel_id in ({pm})", panel_ids)
    removed = edited = 0
    for finding_key, raw in conn.execute("select finding_key, disease_keys_json from findings_vocab").fetchall():
        current = json.loads(raw)
        kept = [k for k in current if k not in thin]
        if kept == current:
            continue
        if kept:
            edited += 1
            if apply:
                conn.execute("update findings_vocab set disease_keys_json=? where finding_key=?", (json.dumps(kept), finding_key))
        else:
            removed += 1
            if apply:
                conn.execute("delete from findings_vocab where finding_key=?", (finding_key,))
    out["findings_vocab removed"], out["findings_vocab edited"] = removed, edited
    if apply:
        conn.commit()
        conn.execute("vacuum")
    conn.close()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--max-images", type=int, default=4, help="remove topics with this many published images or fewer")
    parser.add_argument("--apply", action="store_true", help="perform the deletion (default: report only)")
    args = parser.parse_args()
    counts = site_counts()
    thin = {k for k, n in counts.items() if n <= args.max_images}
    print(f"site topics: {len(counts)} | removing {len(thin)} with <= {args.max_images} images "
          f"({sum(counts[k] == 0 for k in thin)} at zero, {sum(counts[k] for k in thin)} published panels)")
    stamp = dt.date.today().strftime("%Y%m%d")
    backup_dir = ROOT / "data" / "visual_pilot" / f"backup_prune_{stamp}"
    removed = prune_catalog(thin, args.apply)
    catalog_total = len(counts) - len(thin)
    print(f"catalog entries removed: {len(removed)} -> {catalog_total} topics remain")
    print(f"manifest topics removed: {prune_manifest(thin, args.apply, catalog_total)}")
    print("json files:", prune_json_files(thin, args.apply))
    print("site:", prune_site(thin, args.apply))
    for db in DATABASES:
        print(f"{db.relative_to(ROOT)}:", prune_database(db, thin, args.apply, backup_dir))
    if args.apply:
        backup_dir.mkdir(parents=True, exist_ok=True)
        (backup_dir / "removed_catalog_entries.json").write_text(json.dumps(
            {"removed_on": dt.date.today().isoformat(), "max_images": args.max_images,
             "site_counts": {k: counts[k] for k in sorted(thin)}, "entries": removed}, indent=2, ensure_ascii=False) + "\n")
        print(f"backups written to {backup_dir.relative_to(ROOT)}")
    else:
        print("dry run: nothing changed (pass --apply to delete)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
