"""Stage 4c: mimic audit (prompt P8).

The vision judge (P3) attributes an image from what it shows and its
caption, so a look-alike disease can pass as the topic: "Langerhans cell
histiocytosis presented as persistent diaper dermatitis", "Becker nevus
presenting as pseudofolliculitis barbae", a tuberculous abscess filed under
furuncle. P8 reads the topic (with its synonyms and subtypes), the source
article's title and abstract, the caption and the citing sentences, and
returns ``topic``, ``other_disease`` or ``uncertain``. Text-only, one call
per panel, cached in ``llm_calls``.

``other_disease`` and ``uncertain`` panels get a reversible ``panel_curation``
exclusion whose reason starts with ``mimic_audit``; a later ``topic`` verdict
lifts it. Exclusions from other stages or from manual review are never
touched. A failed call leaves the panel as it was; rerun to retry.

Usage: ``python -m src.visual_pilot.cli audit-mimics [--disease X] [--pmcids ...]``
"""

from __future__ import annotations

import json

from . import config, db, llm, prompts
from .diseases import load_diseases

REASON_PREFIX = "mimic_audit"
EXCLUDED_VERDICTS = ("other_disease", "uncertain")
MAX_ABSTRACT_CHARS = 3000
MAX_MENTIONS = 2
MAX_MENTION_CHARS = 600


def user_content(panel: dict, disease: dict, abstract: str) -> str:
    mentions = [
        str(m)[:MAX_MENTION_CHARS]
        for m in (db.from_json(panel.get("in_text_mentions_json"), []) or [])[:MAX_MENTIONS]
    ]
    return json.dumps(
        {
            "topic": disease.get("name"),
            "topic_synonyms": disease.get("synonyms") or [],
            "topic_subtypes": [s.get("label") or s.get("key") for s in disease.get("subtypes") or []
                               if isinstance(s, dict)],
            "article_title": panel.get("title") or "",
            "abstract": (abstract or "")[:MAX_ABSTRACT_CHARS],
            "figure_caption": panel.get("caption") or "",
            "in_text_mentions": mentions,
        },
        ensure_ascii=False,
    )


def exclusion_reason(parsed: dict) -> str | None:
    """The ``panel_curation`` reason for a P8 verdict, or None to publish."""
    verdict = (parsed or {}).get("verdict")
    if verdict not in EXCLUDED_VERDICTS:
        return None
    diagnosis = " ".join(str((parsed or {}).get("actual_diagnosis") or "").split())[:120]
    return f"{REASON_PREFIX}: {verdict}" + (f": {diagnosis}" if diagnosis else "")


def _panels(conn, args) -> list[dict]:
    where, params = [], []
    if getattr(args, "disease", "all") != "all":
        where.append("p.disease_key = ?")
        params.append(args.disease)
    if getattr(args, "pmcids", None):
        where.append(f"p.pmcid IN ({','.join('?' for _ in args.pmcids)})")
        params.extend(args.pmcids)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    meta = ("LEFT JOIN article_source_metadata m ON m.pmcid = p.pmcid"
            if "article_source_metadata" in tables else "")
    sql = (
        "SELECT p.panel_id, p.pmcid, p.sha256, p.disease_key, a.title, f.caption, "
        f"f.in_text_mentions_json{', m.metadata_json' if meta else ''} "
        "FROM panels p JOIN figures f USING(figure_id) JOIN articles a ON a.pmcid = p.pmcid "
        f"{meta}" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY p.panel_id"
    )
    rows = [dict(r) for r in conn.execute(sql, params)]
    return rows[: args.limit] if getattr(args, "limit", None) else rows


def run(args) -> int:
    conn = db.init_db()
    panels = _panels(conn, args)
    print(f"audit-mimics: {len(panels)} panel(s) to audit")
    if args.dry_run or not panels:
        conn.close()
        return 0
    catalog = load_diseases()
    client = llm.LLMClient(db_conn=conn, budget_usd=args.budget_usd)
    requests = (
        {
            "stage": "mimic_audit",
            "model": config.VP_DESCRIBE_MODEL,
            "system": prompts.P8.system,
            "user_content": user_content(
                p, catalog.get(p["disease_key"], {"name": p["disease_key"]}),
                (db.from_json(p.get("metadata_json"), {}) or {}).get("abstract") or "",
            ),
            "schema": prompts.P8.schema,
            "prompt_version": prompts.P8.version,
            "reasoning_effort": config.VP_DESCRIBE_REASONING_EFFORT,
        }
        for p in panels
    )
    counts = {"topic": 0, "other_disease": 0, "uncertain": 0}
    errors = 0
    budget_hit = False
    for res in client.iter_many(requests):
        with client.db_lock:  # iter_many workers commit on this connection
            if res.error is not None:
                if isinstance(res.error, llm.BudgetExceeded):
                    budget_hit = True
                else:
                    errors += 1
                continue
            verdict = (res.parsed or {}).get("verdict")
            if verdict not in counts:
                errors += 1
                continue
            counts[verdict] += 1
            panel = panels[res.index]
            reason = exclusion_reason(res.parsed)
            with conn:
                if reason is None:
                    conn.execute(
                        "DELETE FROM panel_curation WHERE panel_id = ? AND reason LIKE ?",
                        (panel["panel_id"], REASON_PREFIX + "%"),
                    )
                    continue
                # Never overwrite another stage's or a reviewer's exclusion.
                conn.execute(
                    "INSERT INTO panel_curation "
                    "(panel_id, image_sha256, decision, reason, policy_version) "
                    "VALUES (?, ?, 'exclude', ?, ?) ON CONFLICT(panel_id) DO UPDATE SET "
                    "image_sha256=excluded.image_sha256, reason=excluded.reason, "
                    "policy_version=excluded.policy_version, reviewed_at=datetime('now') "
                    "WHERE panel_curation.reason LIKE ?",
                    (panel["panel_id"], panel["sha256"] or "", reason, prompts.P8.version,
                     REASON_PREFIX + "%"),
                )
    if budget_hit:
        print(f"LLM budget exhausted (${client.spent_usd:.4f}); stopping cleanly. Rerun to resume.")
    print(
        f"audit-mimics: {counts['topic']} topic, {counts['other_disease']} other disease, "
        f"{counts['uncertain']} uncertain ({counts['other_disease'] + counts['uncertain']} "
        f"excluded), {errors} error(s), spend=${client.spent_usd:.4f}"
    )
    conn.close()
    return 0
