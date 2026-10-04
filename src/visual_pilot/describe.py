"""Stage 4b: standalone display captions (prompt P5).

Article captions are written for the article: they carry figure and panel
letters, citation marks ("tendon.19 A"), cross-references and copyright
notes. For every stored panel without one, P5 rewrites the caption into a
short ``display_title`` and a 1–2 sentence ``display_description`` that
read on their own, and picks the viewer section (tab) and subsection the
image belongs in. The viewer shows those, keeps the original caption under
Source and falls back to rule routing for an invalid section. Text-only,
one call per panel, cached in ``llm_calls``.

Treatment images (before/after, drug response, postoperative views) are
kept out of the library: P5 flags them and the panel gets a reversible
``panel_curation`` exclusion with reason ``treatment_related`` (rows stay,
so the figure is not stored again). Caption triage
(P2) drops treatment figures before download, so this is a safety net.
"""

from __future__ import annotations

import json

from . import config, db, llm, prompts
from .diseases import load_diseases

MAX_MENTIONS = 3
MAX_MENTION_CHARS = 600
MAX_TITLE_CHARS = 90
MAX_DESCRIPTION_CHARS = 400
TREATMENT_REASON = "treatment_related"


def user_content(
    panel: dict,
    disease_names: dict[str, str],
    finding_labels: dict[str, str],
    sections: list[dict],
) -> str:
    whole = panel.get("crop_mode") == "whole_figure" or (panel.get("panel_label") or "").lower() == "whole"
    findings = [
        {"finding_key": f["finding_key"], "label": finding_labels.get(f["finding_key"], f["finding_key"])}
        for f in db.from_json(panel.get("findings_json"), []) or []
        if isinstance(f, dict) and f.get("finding_key")
    ]
    mentions = [
        str(m)[:MAX_MENTION_CHARS]
        for m in (db.from_json(panel.get("in_text_mentions_json"), []) or [])[:MAX_MENTIONS]
    ]
    payload = {
        "image_is": "the whole figure" if whole else f"panel {panel.get('panel_label')} of the figure",
        "figure_label": panel.get("figure_label"),
        "figure_caption": panel.get("figure_caption") or "",
        "in_text_mentions": mentions,
        "disease": disease_names.get(panel.get("disease_key"), panel.get("disease_key")),
        "subtype": panel.get("subtype"),
        "modality": panel.get("modality"),
        "body_site": panel.get("body_site"),
        "findings": findings,
        "sections": sections,
    }
    return json.dumps(payload, ensure_ascii=False)


def _clean(text, limit: int) -> str | None:
    text = " ".join(str(text or "").split()).strip()
    if not text:
        return None
    if len(text) > limit:
        text = text[: limit - 1].rsplit(" ", 1)[0] + "…"
    return text


def _section_choice(parsed: dict, panel: dict, sections: list[dict]) -> tuple[str | None, str | None]:
    """Keep the model's section/subsection only when they are listed options."""
    by_key = {s["key"]: s for s in sections}
    section = by_key.get((parsed or {}).get("section"))
    if section is None:
        return None, None
    subsection = (parsed or {}).get("subsection")
    allowed = section.get("subsections") or []
    if allowed == ["finding"]:
        keys = {f.get("finding_key") for f in db.from_json(panel.get("findings_json"), []) or [] if isinstance(f, dict)}
        allowed = sorted(k for k in keys if k)
    return section["key"], subsection if subsection in allowed else None


def _panels(conn, args) -> list[dict]:
    where, params = [], []
    if args.disease != "all":
        where.append("p.disease_key = ?")
        params.append(args.disease)
    if args.pmcids:
        where.append(f"p.pmcid IN ({','.join('?' for _ in args.pmcids)})")
        params.extend(args.pmcids)
    if not getattr(args, "force", False):
        where.append("(p.display_description IS NULL OR p.display_description = '')")
    sql = (
        "SELECT p.panel_id, p.figure_id, p.pmcid, p.sha256, p.panel_label, p.crop_mode, p.disease_key, p.subtype, p.modality, "
        "p.body_site, p.findings_json, f.label AS figure_label, f.caption AS figure_caption, "
        "f.in_text_mentions_json FROM panels p JOIN figures f USING(figure_id)"
        + (" WHERE " + " AND ".join(where) if where else "")
        + " ORDER BY p.panel_id"
    )
    rows = [dict(r) for r in conn.execute(sql, params)]
    return rows[: args.limit] if args.limit else rows


def run(args) -> int:
    conn = db.init_db()
    panels = _panels(conn, args)
    print(f"describe: {len(panels)} panel(s) need a display caption")
    if args.dry_run or not panels:
        conn.close()
        return 0

    disease_names = {k: d.get("name", k) for k, d in load_diseases().items()}
    finding_labels = {
        r["finding_key"]: r["label"]
        for r in conn.execute("SELECT finding_key, label FROM findings_vocab")
    }
    from .viewer.app import section_options

    sections = {key: section_options(key) for key in {p["disease_key"] for p in panels}}
    client = llm.LLMClient(db_conn=conn, budget_usd=args.budget_usd)
    requests = (
        {
            "stage": "describe",
            "model": config.VP_DESCRIBE_MODEL,
            "system": prompts.P5.system,
            "user_content": user_content(
                p, disease_names, finding_labels, sections[p["disease_key"]]
            ),
            "schema": prompts.P5.schema,
            "prompt_version": prompts.P5.version,
            "reasoning_effort": config.VP_DESCRIBE_REASONING_EFFORT,
        }
        for p in panels
    )
    written = errors = 0
    excluded: list[str] = []
    budget_hit = False
    for res in client.iter_many(requests):
        with client.db_lock:  # iter_many workers commit on this connection
            if res.error is not None:
                if isinstance(res.error, llm.BudgetExceeded):
                    budget_hit = True
                else:
                    errors += 1
                continue
            title = _clean((res.parsed or {}).get("title"), MAX_TITLE_CHARS)
            description = _clean((res.parsed or {}).get("description"), MAX_DESCRIPTION_CHARS)
            if not description:
                errors += 1
                continue
            panel = panels[res.index]
            section, subsection = _section_choice(res.parsed, panel, sections[panel["disease_key"]])
            with conn:
                conn.execute(
                    "UPDATE panels SET display_title = ?, display_description = ?, "
                    "display_section = ?, display_subsection = ?, "
                    "updated_at = datetime('now') WHERE panel_id = ?",
                    (title, description, section, subsection, panel["panel_id"]),
                )
                if (res.parsed or {}).get("treatment_related") is True:
                    conn.execute(
                        "INSERT INTO panel_curation "
                        "(panel_id, image_sha256, decision, reason, policy_version) "
                        "VALUES (?, ?, 'exclude', ?, ?) ON CONFLICT(panel_id) DO UPDATE SET "
                        "image_sha256=excluded.image_sha256, decision=excluded.decision, "
                        "reason=excluded.reason, policy_version=excluded.policy_version, "
                        "reviewed_at=datetime('now')",
                        (panel["panel_id"], panel["sha256"] or "", TREATMENT_REASON,
                         prompts.P5.version),
                    )
                    excluded.append(panel["panel_id"])
                else:
                    conn.execute(
                        "DELETE FROM panel_curation WHERE panel_id = ? AND reason = ?",
                        (panel["panel_id"], TREATMENT_REASON),
                    )
            written += 1
    if budget_hit:
        print(f"LLM budget exhausted (${client.spent_usd:.4f}); stopping cleanly. Rerun to resume.")
    print(
        f"describe: {written} written, {len(excluded)} treatment image(s) excluded, "
        f"{errors} error(s), spend=${client.spent_usd:.4f}"
    )
    conn.close()
    return 0
