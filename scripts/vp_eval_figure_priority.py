"""Offline, equal-call-budget audit of the figure-priority heuristic.

Uses only the local pilot SQLite DB and the manually reviewed metadata labels
in vp_figure_eval_labels.json. No image downloads or model calls are made.
Run from the repository root: python3 scripts/vp_eval_figure_priority.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.visual_pilot import config, judge  # noqa: E402

LABEL_FILE = ROOT / "scripts" / "vp_figure_eval_labels.json"
K = 3  # equal vision-call budget per disease / method


def main() -> int:
    label_data = json.loads(LABEL_FILE.read_text(encoding="utf-8"))
    labels = label_data["labels"]
    # sqlite URI construction is explicit so this audit cannot create/migrate DB.
    import sqlite3

    conn = sqlite3.connect(f"file:{config.db_path()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    print(f"DB: {config.db_path()} (read-only); budget: {K} figures/disease/method")
    all_results = {}
    for disease, disease_labels in labels.items():
        placeholders = ",".join("?" for _ in disease_labels)
        rows = [dict(r) for r in conn.execute(
            f"SELECT f.*, a.title, a.primary_disease_keys_json "
            f"FROM figures f JOIN articles a USING(pmcid) WHERE f.figure_id IN ({placeholders})",
            list(disease_labels),
        )]
        by_id = {r["figure_id"]: r for r in rows}
        missing = set(disease_labels) - set(by_id)
        if missing:
            print(f"{disease}: labels absent from local DB: {', '.join(sorted(missing))}")
            continue
        # Restrict both strategies to the same labeled candidate set and legal
        # statuses used by the vision stage. This is a rank comparison, not an
        # estimate of performance on every rejected candidate.
        candidates = [r for r in rows if r["status"] in {"caption_kept", "caption_uncertain"}]
        baseline = sorted(candidates, key=lambda r: (r["status"] != "caption_kept", r["figure_id"]))
        article_by_id = {r["pmcid"]: {"pmcid": r["pmcid"], "title": r["title"],
                                      "primary_disease_keys_json": r["primary_disease_keys_json"]}
                         for r in candidates}
        proposed = judge.rank_figures(conn, [dict(r) for r in candidates], article_by_id)
        gold = {fid: item["useful"] for fid, item in disease_labels.items()}
        positive_count = sum(gold.get(r["figure_id"], False) for r in candidates)
        disease_result = {}
        print(f"\n{disease.upper()}: {len(candidates)} labeled eligible figures; {positive_count} caption-positive")
        for name, order in (("baseline", baseline), ("priority", proposed)):
            top = order[:K]
            hits = sum(gold.get(r["figure_id"], False) for r in top)
            precision = hits / K
            recall = hits / positive_count if positive_count else 0.0
            ids = [r["figure_id"] for r in top]
            disease_result[name] = {"precision_at_3": precision, "recall_at_3": recall, "top": ids}
            print(f"  {name:8s} P@{K}={precision:.2f} R@{K}={recall:.2f}: {', '.join(ids)}")
        all_results[disease] = disease_result
        rejected_labels = label_data.get("rejected_sample", {}).get(disease, {})
        rejected_rows = {}
        for fid, label in rejected_labels.items():
            row = conn.execute(
                "SELECT status,triage_json FROM figures WHERE figure_id=?", (fid,)
            ).fetchone()
            if row:
                rejected_rows[fid] = {"useful": bool(label["useful"]), "status": row["status"],
                                      "triage": json.loads(row["triage_json"] or "{}")}
        false_rejects = [fid for fid, item in rejected_rows.items() if item["useful"]]
        all_results[disease]["caption_rejected_audit"] = {
            "sampled": len(rejected_rows), "caption_positive_rejected": false_rejects,
        }
        print(f"  rejected-candidate audit: {len(false_rejects)} caption-positive figure(s) in "
              f"{len(rejected_rows)} manually reviewed rejected sample(s)")
    conn.close()
    out = ROOT / "docs" / "visual_pilot_figure_priority_eval_results.json"
    out.write_text(json.dumps({"budget_per_disease": K, "results": all_results}, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
