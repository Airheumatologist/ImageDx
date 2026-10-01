"""Coordinator-defined coverage measurements from frozen gallery selections."""

from __future__ import annotations

from contextlib import contextmanager

from . import config, db, publication


@contextmanager
def consistent_read(conn):
    """Own only the read transaction we start; never commit a caller's writes."""
    started = not conn.in_transaction
    if started:
        conn.execute("BEGIN")
    try:
        yield
    finally:
        if started:
            conn.rollback()


def coverage_tier(count: int) -> str:
    floor, target, cap = config.validate_coverage_settings()
    if count == 0:
        return "empty"
    if count < floor:
        return "below_floor"
    if count < target:
        return "below_target"
    if count < cap:
        return "expanding"
    return "full"


def approved_pairs(vocab_rows, disease_key=None) -> list[tuple[str, str]]:
    pairs = set()
    for row in vocab_rows:
        if not row["approved"]:
            continue
        for disease in db.from_json(row["disease_keys_json"], []) or []:
            if disease_key is None or disease == disease_key:
                pairs.add((str(disease), str(row["finding_key"])))
    return sorted(pairs)


def assemble_snapshot(pairs, selections: dict, lanes: dict) -> dict:
    """Only representative groups count; duplicate row aliases never add credit."""
    floor, target, cap = config.validate_coverage_settings()
    out = {}
    for disease, finding in pairs:
        selection = selections[(disease, finding)]
        published = len(selection["published_panel_ids"])
        reserves = len(selection["reserve_panel_ids"])
        eligible = selection["eligible_distinct"]
        if eligible != published + reserves:
            raise ValueError("Gallery distinct groups must partition into publication and reserves")
        lane = lanes.get((disease, finding), {})
        out.setdefault(disease, {})[finding] = {
            "eligible_distinct": eligible,
            "published_distinct": published,
            "reserve_distinct": reserves,
            "floor_deficit": max(0, floor - published),
            "target_deficit": max(0, target - published),
            "cap_remaining": max(0, cap - published),
            "tier": coverage_tier(published),
            "blocked_reason": lane.get("blocked_reason"),
            "published_panel_ids": list(selection["published_panel_ids"]),
            "reserve_panel_ids": list(selection["reserve_panel_ids"]),
            "selection_reasons": selection["selection_reasons"],
            "identity_unknown_count": selection["identity_unknown_count"],
            "last_served_sequence": lane.get("last_served_sequence", 0),
            "last_deficit_reduction_at": lane.get("last_deficit_reduction_at"),
            "search_policy_version": lane.get("search_policy_version"),
        }
    return out


def snapshot(conn, selector, disease_key=None, *, panels=None) -> dict:
    with consistent_read(conn):
        vocab = list(conn.execute("SELECT * FROM findings_vocab WHERE approved=1"))
        pairs = approved_pairs(vocab, disease_key)
        panels = publication.eligible_panels(conn, disease_key) if panels is None else panels
        lanes = {
            (row["disease_key"], row["finding_key"]): dict(row)
            for row in conn.execute("SELECT * FROM manifestation_lanes")
        }
        locks = {
            (row["disease_key"], row["finding_key"]): row["panel_id"]
            for row in conn.execute(
                "SELECT * FROM manifestation_representatives WHERE locked=1 OR selection_source='manual'"
            )
        }
        selections = {
            pair: selector(
                panels, *pair, cap=config.VP_FINDING_GALLERY_CAP,
                locked_panel_id=locks.get(pair),
            )
            for pair in pairs
        }
        return assemble_snapshot(pairs, selections, lanes)
