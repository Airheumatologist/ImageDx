"""Deterministic published-gallery selection for one approved pair.

``select_gallery`` is pure: callers pass the current eligible panel dicts
(from ``publication.eligible_panels``, decorated with
``supported_finding_keys`` and any evidence-backed ``identity_review``) and a
published-gallery cap. Eligible images beyond the cap are reported as
reserves; nothing is deleted, truncated, or written here.
"""

from __future__ import annotations


def _validated_identity(panel: dict) -> dict | None:
    """Return the panel's identity review only when it binds to current bytes.

    A review counts as evidence only when ``reviewed_image_sha256`` equals the
    panel's current nonempty ``sha256`` and both ``source_quote`` and
    ``review_provenance`` are nonblank. Stale-hash or undocumented reviews
    supplied to this pure selector are ignored.
    """
    review = panel.get("identity_review")
    if not isinstance(review, dict):
        return None
    sha = str(panel.get("sha256") or "")
    if not sha or str(review.get("reviewed_image_sha256") or "") != sha:
        return None
    if not str(review.get("source_quote") or "").strip():
        return None
    if not str(review.get("review_provenance") or "").strip():
        return None
    return review


def _panel_keys(panel: dict) -> tuple[set[tuple[str, str]], bool]:
    """Grouping keys a panel contributes, plus documented-patient flag."""
    keys: set[tuple[str, str]] = set()
    review = _validated_identity(panel)
    documented = False
    if review is not None:
        patient = str(review.get("patient_group_key") or "").strip()
        reuse = str(review.get("reuse_group_key") or "").strip()
        if patient:
            keys.add(("patient", patient))
            documented = True
        if reuse:
            keys.add(("reuse", reuse))
    sha = str(panel.get("sha256") or "")
    # Panels without a hash can never byte-merge; key them by panel_id so each
    # stays its own group unless evidence links it elsewhere.
    keys.add(("hash", sha) if sha else ("hash", f"panel:{panel['panel_id']}"))
    # Same-figure panels form one source family only while identity is
    # undocumented; two documented distinct patients never merge by figure.
    if not documented:
        figure = str(panel.get("figure_id") or "")
        if figure:
            keys.add(("figure", figure))
    return keys, documented


def _find(parent: dict[str, str], node: str) -> str:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def _group_panels(panels: list[dict]) -> list[dict]:
    """Union panels into transitive groups over shared evidence keys."""
    parent = {str(p["panel_id"]): str(p["panel_id"]) for p in panels}
    keys_of: dict[str, set[tuple[str, str]]] = {}
    documented: dict[str, bool] = {}
    owners: dict[tuple[str, str], str] = {}
    for panel in panels:
        panel_id = str(panel["panel_id"])
        keys, documented[panel_id] = _panel_keys(panel)
        keys_of[panel_id] = keys
        for key in keys:
            owner = owners.get(key)
            if owner is None:
                owners[key] = panel_id
            else:
                parent[_find(parent, panel_id)] = _find(parent, owner)
    grouped: dict[str, list[dict]] = {}
    for panel in panels:
        grouped.setdefault(_find(parent, str(panel["panel_id"])), []).append(panel)
    return [
        {
            "members": members,
            "keys_of": keys_of,
            "documented": any(documented[str(m["panel_id"])] for m in members),
        }
        for members in grouped.values()
    ]


def _alias_status(
    panel_id: str,
    member_ids: list[str],
    keys_of: dict[str, set[tuple[str, str]]],
) -> str:
    """Reason a non-representative member shares the group's identity."""
    shared = set()
    for key in keys_of[panel_id]:
        if key == ("hash", f"panel:{panel_id}"):
            continue
        if any(key in keys_of[other] for other in member_ids if other != panel_id):
            shared.add(key[0])
    for kind, status in (
        ("patient", "same_patient"),
        ("reuse", "reused_image"),
        ("hash", "duplicate"),
        ("figure", "source_family"),
    ):
        if kind in shared:
            return status
    return "duplicate"


def select_gallery(
    panels: list[dict],
    disease_key: str,
    finding_key: str,
    *,
    cap: int,
    locked_panel_id: str | None = None,
) -> dict:
    """Select the published gallery for one (disease, finding) pair.

    Qualifying panels collapse into distinct groups by identical hashes,
    documented patient/reuse evidence, and undocumented source-figure
    families. A valid lock is its group's representative and the first
    gallery entry; otherwise the group lead is the best
    ``representatives.score_panel`` score, ties ascending by panel_id.
    Selection is round-robin over source articles (at most two
    representatives per article, counting the lock), then a second pass
    fills the cap in score order since the article preference is soft.
    Unselected group representatives are reserves; every other group member
    is retained in ``selection_reasons`` and earns no distinct credit.
    """
    from .representatives import score_panel

    if cap < 1:
        raise ValueError("cap must be >= 1")
    qualified = sorted(
        (
            dict(panel)
            for panel in panels
            if panel.get("eligible")
            and str(panel.get("disease_key") or "") == disease_key
            and finding_key in (panel.get("supported_finding_keys") or [])
        ),
        key=lambda panel: str(panel["panel_id"]),
    )
    scores = {
        str(panel["panel_id"]): score_panel(panel)[0] for panel in qualified
    }
    groups = _group_panels(qualified)

    def group_key(group: dict) -> tuple[float, str]:
        rep_id = str(group["rep"]["panel_id"])
        return (-scores[rep_id], rep_id)

    locked = str(locked_panel_id) if locked_panel_id is not None else None
    locked_group = None
    for group in groups:
        members = group["members"]
        if locked is not None and any(
            str(member["panel_id"]) == locked for member in members
        ):
            group["rep"] = next(
                member for member in members if str(member["panel_id"]) == locked
            )
            locked_group = group
        else:
            group["rep"] = min(
                members, key=lambda m: (-scores[str(m["panel_id"])], str(m["panel_id"]))
            )
    groups.sort(key=group_key)

    reasons: dict[str, dict] = {}
    published: list[str] = []
    published_groups: list[dict] = []
    selected: set[int] = set()

    def select(group: dict, status: str) -> None:
        rep_id = str(group["rep"]["panel_id"])
        published.append(rep_id)
        published_groups.append(group)
        selected.add(id(group))
        reasons[rep_id] = {"status": status}

    if locked_group is not None:
        select(locked_group, "locked")

    # First pass: round-robin articles by their top group's (-score, panel_id),
    # one group per article per round, at most two per article (counting the
    # already-selected lock).
    by_article: dict[str, list[dict]] = {}
    for group in groups:
        if id(group) in selected:
            continue
        pmcid = str(group["rep"].get("pmcid") or "")
        by_article.setdefault(pmcid, []).append(group)

    def article_key(pmcid: str) -> tuple[float, str]:
        return group_key(by_article[pmcid][0])

    article_counts: dict[str, int] = {}
    if locked_group is not None:
        pmcid = str(locked_group["rep"].get("pmcid") or "")
        article_counts[pmcid] = article_counts.get(pmcid, 0) + 1
    while len(published) < cap:
        progressed = False
        for pmcid in sorted(by_article, key=article_key):
            if len(published) >= cap:
                break
            if article_counts.get(pmcid, 0) >= 2:
                continue
            remaining = [g for g in by_article[pmcid] if id(g) not in selected]
            if not remaining:
                continue
            select(remaining[0], "published")
            article_counts[pmcid] = article_counts.get(pmcid, 0) + 1
            progressed = True
        if not progressed:
            break

    # Second pass: the per-article preference is soft; fill remaining capacity
    # from the rest of the distinct groups in score order.
    for group in groups:
        if len(published) >= cap:
            break
        if id(group) not in selected:
            select(group, "published")

    reserves = [g for g in groups if id(g) not in selected]
    reserve_ids = [str(g["rep"]["panel_id"]) for g in reserves]
    for group in reserves:
        reasons[str(group["rep"]["panel_id"])] = {"status": "gallery_full"}

    for group in groups:
        rep_id = str(group["rep"]["panel_id"])
        member_ids = [str(m["panel_id"]) for m in group["members"]]
        for member in group["members"]:
            member_id = str(member["panel_id"])
            if member_id == rep_id:
                continue
            reasons[member_id] = {
                "status": _alias_status(member_id, member_ids, group["keys_of"]),
                "representative_id": rep_id,
            }

    return {
        "published_panel_ids": published,
        "reserve_panel_ids": reserve_ids,
        "selection_reasons": reasons,
        "identity_unknown_count": sum(
            1 for group in published_groups if not group["documented"]
        ),
        "eligible_distinct": len(groups),
    }


def coverage_snapshot(conn, disease_key: str | None = None) -> dict:
    """Coverage snapshot whose selections come from ``select_gallery``."""
    from . import coverage

    return coverage.snapshot(conn, select_gallery, disease_key)
