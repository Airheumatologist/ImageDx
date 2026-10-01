"""Frozen deficit-tier and one-article-per-lane allocation policy."""

from __future__ import annotations

TIER_ORDER = {"empty": 0, "below_floor": 1, "below_target": 2, "expanding": 3, "full": 4}


def actionable_lanes(snapshot: dict, disease_keys=None) -> list[dict]:
    allowed = set(disease_keys) if disease_keys is not None else None
    lanes = []
    for disease, findings in snapshot.items():
        if allowed is not None and disease not in allowed:
            continue
        for finding, record in findings.items():
            if record["tier"] == "full" or record.get("blocked_reason"):
                continue
            lanes.append({**record, "disease_key": disease, "finding_key": finding})
    if not lanes:
        return []
    highest = min(TIER_ORDER[row["tier"]] for row in lanes)
    return sorted(
        (row for row in lanes if TIER_ORDER[row["tier"]] == highest),
        key=lambda row: (
            row["published_distinct"], row.get("last_served_sequence", 0),
            row["disease_key"], row["finding_key"],
        ),
    )


def reserve_plan(lanes: list[dict], candidates: dict, capacity: int) -> list[dict]:
    """Candidates are already pair-ranked; shared articles consume one slot."""
    if capacity <= 0:
        return []
    pairs = [(row["disease_key"], row["finding_key"]) for row in lanes]
    pools = {pair: list(candidates.get(pair, [])) for pair in pairs}
    selected, ids = [], set()
    while len(selected) < capacity:
        served = set()
        progressed = False
        for pair in pairs:
            if pair in served:
                continue
            row = next((row for row in pools[pair] if row["pmcid"] not in ids), None)
            if row is None:
                continue
            pmcid = row["pmcid"]
            matching = [
                key for key in pairs if any(item["pmcid"] == pmcid for item in pools[key])
            ]
            selected.append({
                **row, "service_pairs": matching, "batch_disease_key": pair[0],
            })
            ids.add(pmcid)
            served.update(matching)
            progressed = True
            if len(selected) >= capacity:
                break
        if not progressed:
            break
    return selected
