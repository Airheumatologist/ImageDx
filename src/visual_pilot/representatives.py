"""Deterministic primary image selection for covered finding pairs.

Publication and curation remain the hard eligibility gates. This module only
chooses which already-published panel leads each approved disease/finding
pair; it never changes the panel collection itself.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from . import db


def _finding_keys(raw) -> set[str]:
    value = db.from_json(raw, []) if isinstance(raw, (str, bytes)) else raw
    out = set()
    for finding in value or []:
        key = finding.get("finding_key") if isinstance(finding, dict) else finding
        if key:
            out.add(str(key))
    return out


def score_panel(panel: dict) -> tuple[float, dict]:
    """Return a 0–100-ish auditable score and its individual components.

    Formula: 40×confidence + typicality (classic 25, variant 12) + up to 15
    for pixel area (linear to 1024²) + crop integrity (+10 for whole figure)
    + annotation clarity (+10 absent, −10 present). Unknown evidence earns 0.
    """
    try:
        confidence = float(panel.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))
    confidence_points = round(40.0 * confidence, 4)

    typicality = str(panel.get("typicality") or "").strip().lower()
    typicality_points = {"classic": 25.0, "variant": 12.0}.get(typicality, 0.0)

    try:
        width, height = int(panel.get("width") or 0), int(panel.get("height") or 0)
    except (TypeError, ValueError):
        width = height = 0
    pixel_area = max(0, width) * max(0, height)
    resolution_points = round(15.0 * min(1.0, pixel_area / (1024 * 1024)), 4)

    crop_mode = str(panel.get("crop_mode") or "").strip().lower()
    crop_points = 10.0 if crop_mode == "whole_figure" else 0.0
    annotations = panel.get("annotations_present")
    if annotations in (0, False, "0", "false", "False"):
        annotation_points = 10.0
    elif annotations in (1, True, "1", "true", "True"):
        annotation_points = -10.0
    else:
        annotation_points = 0.0

    parts = {
        "confidence": {"value": confidence, "weight": 40, "points": confidence_points},
        "typicality": {"value": typicality or None, "points": typicality_points},
        "resolution": {
            "width": width, "height": height, "pixel_area": pixel_area,
            "full_area_reference": 1024 * 1024, "points": resolution_points,
        },
        "crop_integrity": {"value": crop_mode or None, "points": crop_points},
        "annotations": {"value": annotations, "points": annotation_points},
    }
    total = round(confidence_points + typicality_points + resolution_points + crop_points + annotation_points, 4)
    return total, {"formula": "confidence*40 + typicality + resolution + crop_integrity + annotations", "components": parts, "total": total}


_REPRESENTATIVE_COLUMNS = (
    "disease_key", "finding_key", "panel_id", "score",
    "scoring_json", "selection_source", "locked", "updated_at",
)


def _is_locked(row: dict) -> bool:
    return bool(row.get("locked") or row.get("selection_source") == "manual")


def rebuild(conn: sqlite3.Connection, *, snapshot=None) -> dict:
    """Rebuild the persisted table from the current gallery selection.

    The primary for each approved pair is the first selected gallery entry —
    never a reserve. A still-valid lock keeps its stored source/lock flags;
    an invalid lock is replaced by the elected lead and its full original row
    is preserved under ``scoring_json.replaced_selection``. That replacement
    history is propagated unchanged through later rebuilds, so a second
    rebuild cannot erase the evidence. Locked rows whose pair loses every
    image — or is no longer approved at all — are retained verbatim as
    inactive records; ``mapping_for_disease`` derives primaries from the
    current gallery snapshot so they can never surface as a primary.
    ``covered_pairs`` counts only pairs with a live primary.
    """
    tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "manifestation_representatives" not in tables:
        raise sqlite3.OperationalError("manifestation_representatives table is not initialized")
    # Lazy: publication -> gallery -> representatives.score_panel would cycle.
    from . import coverage, gallery

    # Snapshot and stored rows are read inside one consistent transaction;
    # a caller-supplied frozen snapshot (even an empty one) is honored.
    with coverage.consistent_read(conn):
        if snapshot is None:
            snapshot = gallery.coverage_snapshot(conn)
        current = {
            (r["disease_key"], r["finding_key"]): dict(r)
            for r in conn.execute("SELECT * FROM manifestation_representatives")
        }
        panel_rows = {
            str(r["panel_id"]): dict(r)
            for r in conn.execute("SELECT * FROM panels")
        }
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    inactive = 0
    processed: set[tuple[str, str]] = set()
    for disease_key in sorted(snapshot):
        for finding_key in sorted(snapshot[disease_key]):
            pair = (disease_key, finding_key)
            processed.add(pair)
            existing = current.get(pair)
            published = snapshot[disease_key][finding_key].get("published_panel_ids") or []
            if not published:
                if existing is not None and _is_locked(existing):
                    rows.append({col: existing.get(col) for col in _REPRESENTATIVE_COLUMNS})
                    inactive += 1
                continue
            panel_id = published[0]
            score, scoring = score_panel(panel_rows.get(panel_id) or {})
            if existing is not None and _is_locked(existing) and str(existing["panel_id"]) != panel_id:
                source, is_locked = "auto_replaced_invalid_lock", 0
                scoring["replaced_selection"] = {
                    **existing,
                    "reason": "locked panel is no longer the selected gallery lead "
                    "for this approved disease/finding pair",
                }
            elif existing is not None:
                is_locked = int(bool(existing.get("locked"))) if _is_locked(existing) else 0
                source = existing["selection_source"] or (
                    "manual" if _is_locked(existing) else "auto"
                )
                prior = db.from_json(existing.get("scoring_json"), {}) or {}
                if isinstance(prior, dict) and prior.get("replaced_selection"):
                    scoring["replaced_selection"] = prior["replaced_selection"]
            else:
                source, is_locked = "auto", 0
            rows.append({
                "disease_key": disease_key, "finding_key": finding_key,
                "panel_id": panel_id, "score": score,
                "scoring_json": json.dumps(scoring, sort_keys=True),
                "selection_source": source, "locked": is_locked, "updated_at": now,
            })
    # Locked rows for pairs no longer in the approved snapshot stay as
    # inactive records — the lock evidence is preserved, not deleted.
    for pair, row in current.items():
        if pair in processed or not _is_locked(row):
            continue
        rows.append({col: row.get(col) for col in _REPRESENTATIVE_COLUMNS})
        inactive += 1
    with conn:
        conn.execute("DELETE FROM manifestation_representatives")
        conn.executemany(
            "INSERT INTO manifestation_representatives "
            "(disease_key, finding_key, panel_id, score, scoring_json, selection_source, locked, updated_at) "
            "VALUES (:disease_key, :finding_key, :panel_id, :score, :scoring_json, :selection_source, :locked, :updated_at)",
            rows,
        )
    return {
        "covered_pairs": len(rows) - inactive,
        "locked_preserved": sum(r["locked"] for r in rows),
        "auto_replaced": sum(
            r["selection_source"] == "auto_replaced_invalid_lock" for r in rows
        ),
        "inactive_preserved": inactive,
    }


def mapping_for_disease(
    conn, disease_key: str, *, snapshot=None, panels=None
) -> dict[str, dict]:
    """Return finding-key → representative metadata for API serialization.

    The primary is derived from the current gallery snapshot — the first
    selected gallery entry — even when the stored row is stale or missing.
    Stored rows supply ``selection_source``/``locked`` flags only when they
    still match that lead, so curation, config, file, or lock changes can
    never surface a stale or inactive primary. Scores are recomputed from
    the selected panel row rather than served from cache. Callers holding a
    frozen snapshot (and optionally ``{panel_id: row}`` panels) pass them in
    so no live reads are needed at all.
    """
    # Lazy: gallery -> representatives.score_panel would cycle.
    from . import coverage, gallery

    try:
        with coverage.consistent_read(conn):
            records = (
                snapshot if snapshot is not None
                else gallery.coverage_snapshot(conn, disease_key)
            ).get(disease_key, {})
            stored = {
                r["finding_key"]: dict(r)
                for r in conn.execute(
                    "SELECT finding_key, panel_id, score, selection_source, locked "
                    "FROM manifestation_representatives WHERE disease_key=?",
                    (disease_key,),
                )
            }
            out = {}
            for finding_key, record in records.items():
                published = record.get("published_panel_ids") or []
                if not published:
                    continue
                panel_id = published[0]
                if panels is not None:
                    panel_row = panels.get(panel_id)
                else:
                    panel_row = conn.execute(
                        "SELECT * FROM panels WHERE panel_id=?", (panel_id,)
                    ).fetchone()
                if panel_row is None:
                    continue
                existing = stored.get(finding_key)
                if existing is not None and existing["panel_id"] == panel_id:
                    source = existing["selection_source"]
                    locked = bool(existing["locked"])
                else:
                    source, locked = "auto", False
                out[finding_key] = {
                    "panel_id": panel_id,
                    "score": score_panel(dict(panel_row))[0],
                    "selection_source": source,
                    "locked": locked,
                }
            return out
    except sqlite3.OperationalError:
        return {}
