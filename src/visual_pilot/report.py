"""Stage 9: build the pilot report (no LLM calls).

Writes ``reports/pilot_report.md`` and ``reports/pilot_report.json`` with the
per-disease funnel (select -> parse -> triage -> vision -> stored), panel
distributions, zero-image vocab findings, skin-tone distribution, cost
ledger, access failures and TIFF conversions, plus human spot-check sheets
(``spot_accepted.html``, ``spot_caption_rejected.html`` + ``.csv``).

The module deliberately never instantiates an ``LLMClient`` — a test guards
that.
"""

from __future__ import annotations

import html
import json
from collections import Counter

from . import config, coverage, db, diseases, gallery, pair_reporting, publication
from . import representatives

_COUNTS_FILE = "stage2_counts.json"


def _rows(conn, sql, params=()):
    return [dict(r) for r in conn.execute(sql, params)]


def _triaged_categories(conn) -> dict[str, int]:
    out: Counter[str] = Counter()
    for row in conn.execute("SELECT triage_json, status FROM figures"):
        triage = db.from_json(row["triage_json"], {}) or {}
        cat = triage.get("category")
        if cat:
            out[f"{cat}:{row['status']}"] += 1
    return dict(out.most_common())


def _triage_funnel(conn, fig_where: str, params: tuple) -> tuple[dict, dict]:
    """Cumulative caption-triage outcome per figure, derived from stored
    triage_json (a figure keeps its triage verdict even after advancing to
    vision_accepted/stored, unlike the status column)."""
    out = {"kept": 0, "uncertain": 0, "rejected": 0, "pending": 0}
    reasons: Counter[str] = Counter()
    rows = conn.execute(
        "SELECT f.triage_json, f.status FROM figures f "
        "JOIN articles a ON a.pmcid = f.pmcid WHERE 1=1" + fig_where,
        params,
    )
    for row in rows:
        triage = db.from_json(row["triage_json"], {}) or {}
        route = triage.get("route")
        if triage.get("third_party") or route in ("drop", "third_party"):
            out["rejected"] += 1
            reason = triage.get("reason") or route or "unknown"
            source = triage.get("source") or "p2"
            reasons[f"{reason} ({source})"] += 1
        elif route == "keep":
            out["kept"] += 1
        elif route == "uncertain":
            out["uncertain"] += 1
        else:
            out["pending"] += 1
    return out, dict(reasons.most_common())


def _vision_funnel(conn, fig_where: str, params: tuple) -> tuple[dict, dict]:
    """Cumulative vision outcome per figure, derived from vision_json."""
    out = {"accepted": 0, "rejected": 0, "errors": 0, "pending": 0}
    exclusions: Counter[str] = Counter()
    plate_classes: Counter[str] = Counter()
    rows = conn.execute(
        "SELECT f.vision_json, f.status FROM figures f "
        "JOIN articles a ON a.pmcid = f.pmcid WHERE 1=1" + fig_where,
        params,
    )
    for row in rows:
        vision = db.from_json(row["vision_json"], {}) or {}
        if not vision:
            if row["status"] == "vision_error":
                out["errors"] += 1
            elif row["status"] in ("caption_kept", "caption_uncertain", "vision_error"):
                out["pending"] += 1
            continue
        panels = vision.get("panels") or []
        plate = vision.get("plate") or {}
        if plate.get("plate_class"):
            plate_classes[plate["plate_class"]] += 1
        if any(p.get("include") for p in panels) or plate.get("include"):
            out["accepted"] += 1
        else:
            out["rejected"] += 1
        for panel in panels:
            if not panel.get("include"):
                exclusions[panel.get("exclusion_reason") or "unspecified"] += 1
    out["plate_classes"] = dict(plate_classes.most_common())
    return out, dict(exclusions.most_common())


def funnel(conn, disease_key: str | None = None) -> dict:
    """The stage funnel for one disease (or all when key is None)."""
    where = ""
    params = ()
    if disease_key:
        where = " AND EXISTS (SELECT 1 FROM json_each(a.primary_disease_keys_json) je WHERE je.value = ?)"
        params = (disease_key,)
    articles = {
        r["status"]: r["n"]
        for r in conn.execute(
            f"SELECT a.status, COUNT(*) n FROM articles a WHERE 1=1{where} GROUP BY a.status",
            params,
        )
    }
    fig_where = (
        " AND EXISTS (SELECT 1 FROM json_each(a.primary_disease_keys_json) je WHERE je.value = ?)"
        if disease_key
        else ""
    )
    figures = {
        r["status"]: r["n"]
        for r in conn.execute(
            "SELECT f.status, COUNT(*) n FROM figures f JOIN articles a ON a.pmcid = f.pmcid "
            f"WHERE 1=1{fig_where} GROUP BY f.status",
            params,
        )
    }
    triage, reject_reasons = _triage_funnel(conn, fig_where, params)
    vision, exclusions = _vision_funnel(conn, fig_where, params)
    panels_where = "WHERE p.disease_key = ?" if disease_key else ""
    panel_params = (disease_key,) if disease_key else ()
    panel_rows = _rows(
        conn,
        f"SELECT p.sha256 FROM published_panels p {panels_where}",
        panel_params,
    )
    stored_panels = conn.execute(
        f"SELECT COUNT(*) FROM panels p {panels_where}", panel_params,
    ).fetchone()[0]
    return {
        "articles": articles,
        "figures": figures,
        "triage": triage,
        "vision": vision,
        "figure_reject_reasons": reject_reasons,
        "triaged_categories": _triaged_categories(conn),
        "vision_exclusions": exclusions,
        "panels": len(panel_rows),
        "stored_panels": stored_panels,
        "excluded_panels": stored_panels - len(panel_rows),
        "unique_images": len({r["sha256"] for r in panel_rows if r["sha256"]}),
    }


def panel_distribution(conn, *, snapshot=None, records=None) -> dict:
    """Distribution over the default published library (gallery rows).

    ``by_finding`` counts ``published_distinct`` per approved pair from the
    same frozen gallery snapshot the scheduler and viewer use — never raw
    finding tags on stored rows. ``by_modality``/``by_subtype``/
    ``by_disease``/``skin_tone`` count the published library rows (selected
    pair representatives plus Combined views), not reserves.
    """
    if snapshot is None:
        snapshot = gallery.coverage_snapshot(conn)
    if records is None:
        records = publication.panel_records(conn)
    rows = pair_reporting.library_rows(conn, snapshot=snapshot, records=records)
    return pair_reporting.distribution(rows, snapshot)


def pair_coverage(conn, *, snapshot=None) -> dict:
    """Distinct published images per approved (disease, finding) pair.

    Pair counts, histogram buckets, and floor/target/cap milestones come
    from ``pair_reporting.pair_summary`` over the same frozen gallery
    snapshot the scheduler and viewer consume, so all three agree on pair
    counts. The legacy per-disease same_finding/combined plate counts are
    folded into the summary rather than dropped.
    """
    if snapshot is None:
        snapshot = gallery.coverage_snapshot(conn)
    summary = pair_reporting.pair_summary(snapshot)
    plates: dict[str, Counter] = {}
    for row in conn.execute(
        "SELECT disease_key, plate_kind, COUNT(*) n FROM published_panels "
        "WHERE plate_kind IS NOT NULL GROUP BY disease_key, plate_kind"
    ):
        plates.setdefault(row["disease_key"] or "", Counter())[row["plate_kind"]] += row["n"]
    for dk, entry in summary["per_disease"].items():
        entry["same_finding_plates"] = plates.get(dk, Counter()).get("same_finding", 0)
        entry["combined_plates"] = plates.get(dk, Counter()).get("combined", 0)
    return summary


def zero_image_findings(conn) -> dict[str, list[str]]:
    """Approved vocab findings with no panel per disease."""
    used: dict[str, set[str]] = {}
    for row in conn.execute("SELECT disease_key, findings_json FROM published_panels"):
        for f in db.from_json(row["findings_json"], []):
            key = f.get("finding_key") if isinstance(f, dict) else f
            if key:
                used.setdefault(row["disease_key"] or "", set()).add(key)
    out: dict[str, list[str]] = {}
    for row in conn.execute("SELECT finding_key, disease_keys_json FROM findings_vocab WHERE approved = 1"):
        for dk in db.from_json(row["disease_keys_json"], []):
            if row["finding_key"] not in used.get(dk, set()):
                out.setdefault(dk, []).append(row["finding_key"])
    return out


def cost_summary(conn) -> dict:
    stages = _rows(
        conn,
        "SELECT stage, COUNT(*) calls, COALESCE(SUM(input_tokens),0) in_tok, "
        "COALESCE(SUM(output_tokens),0) out_tok, COALESCE(SUM(cost_usd),0) cost "
        "FROM llm_calls GROUP BY stage ORDER BY stage",
    )
    total = sum(s["cost"] for s in stages)
    panels = conn.execute("SELECT COUNT(*) n FROM published_panels").fetchone()["n"]
    return {
        "per_stage": stages,
        "total_usd": round(total, 6),
        "panels_stored": panels,
        "cost_per_panel": round(total / panels, 6) if panels else None,
        "note": "cache hits return stored responses without writing ledger rows",
    }


def access_failures(conn) -> dict:
    errors: Counter[str] = Counter()
    for row in conn.execute(
        "SELECT error FROM figures WHERE status = 'vision_error' OR error IS NOT NULL"
    ):
        text = (row["error"] or "").split(":", 1)[0] or "unknown"
        errors[text] += 1
    tiff = conn.execute(
        "SELECT COUNT(*) n FROM figures WHERE image_format LIKE '%->%'"
    ).fetchone()["n"]
    return {"vision_error_classes": dict(errors.most_common()), "tiff_conversions": tiff}


def representative_summary(conn) -> dict:
    rows = _rows(conn, "SELECT disease_key, selection_source, locked FROM manifestation_representatives")
    by_disease: Counter[str] = Counter(r["disease_key"] for r in rows)
    by_source: Counter[str] = Counter(r["selection_source"] for r in rows)
    return {
        "covered_pairs": len(rows),
        "by_disease": dict(sorted(by_disease.items())),
        "by_selection_source": dict(sorted(by_source.items())),
        "locked": sum(bool(r["locked"]) for r in rows),
    }


# ---------------------------------------------------------------------------
# Spot-check sheets
# ---------------------------------------------------------------------------
def _spot_accepted_html(conn) -> str:
    panels = _rows(
        conn,
        "SELECT pp.*, f.image_url FROM published_panels pp "
        "LEFT JOIN figures f USING(figure_id) ORDER BY pp.disease_key, pp.panel_id",
    )
    cards = []
    for p in panels:
        findings = ", ".join(
            (f.get("finding_key") if isinstance(f, dict) else str(f)) or ""
            for f in db.from_json(p["findings_json"], [])
        )
        cards.append(
            "<div class='card'>"
            f"<img src='{html.escape(p['image_url'] or '')}' loading='lazy'>"
            f"<div><b>{html.escape(p['panel_id'])}</b> "
            f"<i>{html.escape(p['disease_key'] or '')}/{html.escape(p['subtype'] or '')}</i><br>"
            f"{html.escape(p['modality'] or '')} · {html.escape(p['body_site'] or '')} · "
            f"{html.escape(p['typicality'] or '')} · skin_tone={html.escape(str(p['skin_tone']))}<br>"
            f"findings: {html.escape(findings)}<br>"
            f"stage: {html.escape(str(p['stage']))} · age: {html.escape(str(p['age_group']))} · "
            f"conf: {p['confidence']}<br>"
            f"<small>{html.escape(p['rationale'] or '')}</small><br>"
            f"<small class='att'>{html.escape(p['attribution_text'] or '')}</small></div></div>"
        )
    return _page("Accepted panels", "".join(cards))


def _spot_rejected(conn):
    rows = _rows(
        conn,
        "SELECT figure_id, pmcid, label, caption, image_url, triage_json "
        "FROM figures WHERE status = 'caption_rejected' ORDER BY figure_id",
    )
    cards, csv_lines = [], ["figure_id,pmcid,reason,caption,image_url"]
    for r in rows:
        triage = db.from_json(r["triage_json"], {}) or {}
        reason = triage.get("reason") or triage.get("route") or ""
        cards.append(
            "<div class='card'>"
            f"<div><b>{html.escape(r['figure_id'])}</b> — {html.escape(reason)}<br>"
            f"{html.escape(r['label'] or '')}<br>"
            f"<small>{html.escape((r['caption'] or '')[:300])}</small><br>"
            f"<small class='att'>{html.escape(r['image_url'] or '')}</small></div></div>"
        )
        csv_lines.append(
            ",".join(
                '"{}"'.format(str(v or "").replace('"', '""'))
                for v in (r["figure_id"], r["pmcid"], reason, (r["caption"] or "")[:300], r["image_url"])
            )
        )
    return _page("Caption-rejected figures", "".join(cards)), "\n".join(csv_lines) + "\n"


def _page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html><meta charset='utf-8'><title>"
        + html.escape(title)
        + "</title><style>body{font-family:sans-serif;margin:24px}"
        ".card{display:flex;gap:14px;border:1px solid #ddd;border-radius:8px;"
        "padding:10px;margin-bottom:10px}.card img{max-width:200px;max-height:160px}"
        ".att{color:#678}</style>"
        f"<h1>{html.escape(title)}</h1>{body}"
    )


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(args) -> int:
    conn = db.init_db()
    representative_rebuild = representatives.rebuild(conn)
    reports = config.reports_dir()
    reports.mkdir(parents=True, exist_ok=True)

    stage2 = {}
    path = config.reports_dir() / _COUNTS_FILE
    if path.exists():
        stage2 = json.loads(path.read_text())

    funnels = {key: funnel(conn, key) for key in diseases.DISEASE_KEYS}
    funnels["all"] = funnel(conn)
    zero = zero_image_findings(conn)
    costs = cost_summary(conn)
    failures = access_failures(conn)
    representative_stats = representative_summary(conn)

    # All pair metrics read the same frozen gallery snapshot the scheduler
    # and viewer consume, taken inside one consistent read transaction.
    with coverage.consistent_read(conn):
        snapshot = gallery.coverage_snapshot(conn)
        records = publication.panel_records(conn)
        pairs = pair_coverage(conn, snapshot=snapshot)
        dist = panel_distribution(conn, snapshot=snapshot, records=records)
        pair_funnels = pair_reporting.pair_funnel(conn, snapshot=snapshot, records=records)

    report = {
        "stage2_counts": stage2,
        "funnel": funnels,
        "panel_distribution": dist,
        "zero_image_findings": zero,
        "pair_coverage": pairs,
        "pair_funnel": pair_funnels,
        "costs": costs,
        "access_failures": failures,
        "representatives": {**representative_stats, "rebuild": representative_rebuild},
    }
    (reports / "pilot_report.json").write_text(json.dumps(report, indent=1) + "\n")

    accepted_html = _spot_accepted_html(conn)
    rejected_html, rejected_csv = _spot_rejected(conn)
    (reports / "spot_accepted.html").write_text(accepted_html)
    (reports / "spot_caption_rejected.html").write_text(rejected_html)
    (reports / "spot_caption_rejected.csv").write_text(rejected_csv)

    md = ["# Visual Findings Library — pilot report", ""]
    md.append("## Per-disease funnel")
    md.append("| disease | relevant | parsed | figures | kept | uncertain | rejected | vision_ok | stored_panels | unique_images |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for key, f in funnels.items():
        a, fig = f["articles"], f["figures"]
        t, v = f["triage"], f["vision"]
        md.append(
            f"| {key} | {a.get('relevant',0)+a.get('parsed',0)+a.get('parse_error',0)} | "
            f"{a.get('parsed',0)} | {sum(fig.values())} | {t['kept']} | "
            f"{t['uncertain']} | {t['rejected']} | "
            f"{v['accepted']} | {f['panels']} | {f['unique_images']} |"
        )
    md += ["", "## Panel distribution", ""]
    for k, v in dist.items():
        md.append(f"### {k}\n```json\n{json.dumps(v, indent=1)}\n```")
    md += ["", "## Per-pair image coverage", ""]
    md.append(
        "Distinct published images per approved (disease, finding) pair; "
        f"floor = {pairs['floor']}, target = {pairs['target']}, cap = {pairs['cap']}. "
        "Counts come from the same gallery snapshot the scheduler and viewer use."
    )
    md.append("")
    md.append("| images/pair | pairs |")
    md.append("|---|---|")
    for bucket, n in pairs["histogram"].items():
        md.append(f"| {bucket} | {n} |")
    md += ["", "| disease | pairs | ≥ floor | ≥ target | full | zero | same-finding plates | combined plates |",
           "|---|---|---|---|---|---|---|---|"]
    for dk, entry in pairs["per_disease"].items():
        md.append(
            f"| {dk} | {entry['pairs']} | {entry['pairs_at_floor']} | "
            f"{entry['pairs_at_target']} | {entry['pairs_full']} | "
            f"{entry['zero_image_pairs']} | {entry['same_finding_plates']} | "
            f"{entry['combined_plates']} |"
        )
    md += ["", "### Under-target pairs", f"```json\n{json.dumps({dk: e['under_target'] for dk, e in pairs['per_disease'].items()}, indent=1)}\n```"]
    md += ["", "## Per-pair funnel", ""]
    md.append(
        "Every approved pair is listed, even with zero articles or images. "
        "Rejection categories count eligibility failures only; selection "
        "reserves are eligible images retained beyond the gallery cap — "
        "stored, never rejected or deleted."
    )
    for dk in sorted(pair_funnels):
        findings = pair_funnels[dk]
        md += ["", f"### {dk}", ""]
        md.append(
            "| finding | retrieved | licensed | supported figs | published | "
            "eligible | reserves | tier / milestone | blocked | next action |"
        )
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        rejected_totals: Counter[str] = Counter()
        reserve_groups = reserve_rows = 0
        for fk, rec in sorted(findings.items()):
            md.append(
                f"| {fk} | {rec['retrieved_unique_articles']} | "
                f"{rec['licensed_articles']} | "
                f"{rec['pair_supported_caption_figures']} | "
                f"{rec['published_distinct']} | {rec['eligible_distinct']} | "
                f"{rec['reserve_distinct']} | "
                f"{rec['tier']} / {rec['milestone']} | "
                f"{rec['blocked_reason'] or '—'} | {rec['next_action']} |"
            )
            rejected_totals.update(rec["rejection_categories"])
            reserve_groups += rec["selection_reserves"]["distinct_groups"]
            reserve_rows += rec["selection_reserves"]["duplicate_or_diversity_rows"]
        md += ["", "Rejection categories (eligibility failures):",
               f"```json\n{json.dumps(dict(rejected_totals), indent=1)}\n```"]
        md.append(
            f"Selection reserves: {reserve_groups} distinct eligible groups held "
            f"beyond the cap; {reserve_rows} duplicate/diversity member rows "
            "collapse into published or reserve groups (not rejections)."
        )
    md += ["", "## Zero-image vocab findings", f"```json\n{json.dumps(zero, indent=1)}\n```"]
    md += ["", "## Costs", f"```json\n{json.dumps(costs, indent=1)}\n```"]
    md += ["", "## Access failures", f"```json\n{json.dumps(failures, indent=1)}\n```"]
    md += ["", "## Primary representatives", f"```json\n{json.dumps(report['representatives'], indent=1)}\n```"]
    md += ["", "## Figure rejection reasons", f"```json\n{json.dumps(funnels['all']['figure_reject_reasons'], indent=1)}\n```"]
    (reports / "pilot_report.md").write_text("\n".join(md) + "\n")

    print(f"report: {reports/'pilot_report.md'} + json + 2 spot sheets")
    conn.close()
    return 0
