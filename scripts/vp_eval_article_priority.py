"""Offline equal-budget article ranking audit using local captions only."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.visual_pilot import config, pmc  # noqa: E402
from src.visual_pilot.article_rank import score_article  # noqa: E402

LABEL_FILE = ROOT / "scripts" / "vp_figure_eval_labels.json"
K = 2  # equal article-budget per disease / method


def main() -> int:
    labels = json.loads(LABEL_FILE.read_text(encoding="utf-8"))["article_labels"]
    conn = sqlite3.connect(f"file:{config.db_path()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    print(f"DB: {config.db_path()} (read-only); budget: {K} articles/disease/method")
    results = {}
    for disease, disease_labels in labels.items():
        candidates = []
        missing = []
        for pmcid, label in disease_labels.items():
            row = conn.execute(
                "SELECT pmcid,title,status,retrieval_score,primary_disease_keys_json "
                "FROM articles WHERE pmcid=?", (pmcid,),
            ).fetchone()
            if row is None:
                missing.append(pmcid)
                continue
            captions = []
            for fig in conn.execute(
                "SELECT caption,effective_license,fig_permissions_text,triage_json,status "
                "FROM figures WHERE pmcid=?", (pmcid,),
            ):
                triage = json.loads(fig["triage_json"] or "{}")
                allowed = pmc.license_allows(fig["effective_license"]) is not None
                allowed = allowed and not bool(triage.get("third_party"))
                captions.append({"caption": fig["caption"] or "", "eligible": allowed})
            article = {
                "pmcid": pmcid,
                "title": row["title"] or "",
                "retrieval_score": row["retrieval_score"] or 0.0,
                # Abstract and retained retrieval passages do not exist in the
                # local baseline DB. Do not substitute P1's relevance rationale.
                "abstract": "",
                "status": row["status"],
            }
            score = score_article(article, disease, captions=captions)
            candidates.append({"article": article, "gold": bool(label["useful"]),
                               "note": label["note"], "score": score})
        if missing:
            print(f"{disease}: missing article IDs: {', '.join(missing)}")
        old = sorted(candidates, key=lambda x: (x["article"]["retrieval_score"], x["article"]["pmcid"]), reverse=True)
        new = sorted(candidates, key=lambda x: (x["score"]["score"], x["article"]["retrieval_score"], x["article"]["pmcid"]), reverse=True)
        positives = sum(item["gold"] for item in candidates)
        row_result = {"labeled": len(candidates), "caption_positive": positives,
                      "workflow_status_counts": dict(Counter(x["article"]["status"] for x in candidates))}
        print(f"\n{disease.upper()}: {len(candidates)} labeled articles; {positives} caption-positive")
        print(f"  workflow statuses: {row_result['workflow_status_counts']}")
        for name, order in (("retrieval_score", old), ("visual_priority", new)):
            top = order[:K]
            hits = sum(item["gold"] for item in top)
            row_result[name] = {
                "precision_at_2": hits / K,
                "recall_at_2": hits / positives if positives else 0.0,
                "top": [item["article"]["pmcid"] for item in top],
            }
            print(f"  {name:16s} P@{K}={hits / K:.2f} R@{K}={row_result[name]['recall_at_2']:.2f}: "
                  + ", ".join(row_result[name]["top"]))
        results[disease] = row_result
    conn.close()
    out = ROOT / "docs" / "visual_pilot_article_priority_eval_results.json"
    out.write_text(json.dumps({"budget_per_disease": K, "results": results}, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
