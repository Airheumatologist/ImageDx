"""Deterministic primary image selection for covered finding pairs.

Publication and curation remain the hard eligibility gates. This module only
chooses which already-published panel leads each approved disease/finding
pair; it never changes the panel collection itself.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
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


def _approved_pairs(vocab_rows) -> set[tuple[str, str]]:
    pairs = set()
    for row in vocab_rows:
        if not row["approved"]:
            continue
        for disease in db.from_json(row["disease_keys_json"], []) or []:
            pairs.add((str(disease), str(row["finding_key"])))
    return pairs


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


def elect_representatives(panel_rows, vocab_rows, existing_rows=()) -> list[dict]:
    """Elect one panel for every covered approved pair, preserving valid locks.

    The input mappings need only contain the columns used by ``score_panel``
    plus panel_id, disease_key and findings_json. Ties resolve by panel_id.
    """
    approved = _approved_pairs(vocab_rows)
    candidates: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for panel in panel_rows:
        disease = panel.get("disease_key")
        panel_id = panel.get("panel_id")
        if not disease or not panel_id:
            continue
        for finding in _finding_keys(panel.get("findings_json")):
            pair = (str(disease), finding)
            if pair in approved:
                candidates[pair][str(panel_id)] = dict(panel)

    existing = {(r["disease_key"], r["finding_key"]): dict(r) for r in existing_rows}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    selected = []
    for pair in sorted(candidates):
        options = candidates[pair]
        scored = {pid: score_panel(panel) for pid, panel in options.items()}
        locked = existing.get(pair)
        keep = None
        if locked and (locked.get("locked") or locked.get("selection_source") == "manual"):
            keep = locked.get("panel_id") if locked.get("panel_id") in options else None
        if keep is not None:
            panel = options[keep]
            score, scoring = scored[keep]
            source = locked.get("selection_source") or "manual"
            is_locked = int(bool(locked.get("locked")))
        else:
            keep = min(options, key=lambda pid: (-scored[pid][0], pid))
            panel = options[keep]
            score, scoring = scored[keep]
            source = "auto_replaced_invalid_lock" if locked and (locked.get("locked") or locked.get("selection_source") == "manual") else "auto"
            is_locked = 0
            if source == "auto_replaced_invalid_lock":
                scoring["replaced_selection"] = {
                    "panel_id": locked.get("panel_id"),
                    "reason": "selected panel is no longer published or no longer supports this approved disease/finding pair",
                }
        selected.append({
            "disease_key": pair[0], "finding_key": pair[1], "panel_id": keep,
            "score": score, "scoring_json": json.dumps(scoring, sort_keys=True),
            "selection_source": source, "locked": is_locked, "updated_at": now,
        })
    return selected


def rebuild(conn: sqlite3.Connection) -> dict:
    """Rebuild the persisted table from published panels and approved vocab."""
    tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "manifestation_representatives" not in tables:
        raise sqlite3.OperationalError("manifestation_representatives table is not initialized")
    panels = [dict(r) for r in conn.execute("SELECT * FROM published_panels")]
    vocab = [dict(r) for r in conn.execute(
        "SELECT finding_key, disease_keys_json, approved FROM findings_vocab WHERE approved=1"
    )]
    current = [dict(r) for r in conn.execute("SELECT * FROM manifestation_representatives")]
    rows = elect_representatives(panels, vocab, current)
    with conn:
        conn.execute("DELETE FROM manifestation_representatives")
        conn.executemany(
            "INSERT INTO manifestation_representatives "
            "(disease_key, finding_key, panel_id, score, scoring_json, selection_source, locked, updated_at) "
            "VALUES (:disease_key, :finding_key, :panel_id, :score, :scoring_json, :selection_source, :locked, :updated_at)",
            rows,
        )
    return {"covered_pairs": len(rows), "locked_preserved": sum(r["locked"] for r in rows),
            "auto_replaced": sum(r["selection_source"] == "auto_replaced_invalid_lock" for r in rows)}


def mapping_for_disease(conn, disease_key: str) -> dict[str, dict]:
    """Return finding-key → representative metadata for API serialization."""
    try:
        rows = conn.execute(
            "SELECT finding_key, panel_id, score, selection_source, locked "
            "FROM manifestation_representatives WHERE disease_key=? ORDER BY finding_key",
            (disease_key,),
        )
    except sqlite3.OperationalError:
        return {}
    return {
        r["finding_key"]: {
            "panel_id": r["panel_id"], "score": r["score"],
            "selection_source": r["selection_source"], "locked": bool(r["locked"]),
        }
        for r in rows
    }
