"""Deterministic published-gallery selection for one approved pair.

``select_gallery`` is pure: callers pass the current eligible panel dicts
(from ``publication.eligible_panels``, decorated with
``supported_finding_keys`` and any evidence-backed ``identity_review``) and a
published-gallery cap. Eligible images beyond the cap are reported as
reserves; nothing is deleted, truncated, or written here.
"""

from __future__ import annotations

from . import config
from .source_quality import TIER_SERIES

# Panels built without article metadata rank as an ordinary study.
TIER_DEFAULT = TIER_SERIES


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


def _choose_representatives(groups: list[dict], locked: str | None, rank) -> dict | None:
    """Set each group's ``rep`` (the lock, else its best-ranked member).

    Returns the group holding the lock, if any.
    """
    locked_group = None
    for group in groups:
        lock = next(
            (m for m in group["members"] if locked is not None and str(m["panel_id"]) == locked),
            None,
        )
        if lock is not None:
            group["rep"] = lock
            locked_group = group
        else:
            group["rep"] = min(group["members"], key=rank)
    return locked_group


def _rep_pmcid(group: dict) -> str:
    return str(group["rep"].get("pmcid") or "")


class _Selection:
    """Published/reserve bookkeeping for one ``select_gallery`` call."""

    def __init__(self, groups: list[dict], cap: int, rank) -> None:
        self.groups = groups  # sorted by their representative's rank
        self.cap = cap
        self.rank = rank
        self.reasons: dict[str, dict] = {}
        self.published: list[str] = []
        self.published_groups: list[dict] = []
        self.selected: set[int] = set()
        self.article_counts: dict[str, int] = {}

    def full(self) -> bool:
        return len(self.published) >= self.cap

    def select(self, group: dict, status: str) -> None:
        rep_id = str(group["rep"]["panel_id"])
        self.published.append(rep_id)
        self.published_groups.append(group)
        self.selected.add(id(group))
        self.reasons[rep_id] = {"status": status}
        pmcid = _rep_pmcid(group)
        self.article_counts[pmcid] = self.article_counts.get(pmcid, 0) + 1

    def round_robin(self) -> None:
        """Round-robin articles by their top group's (tier, -score, panel_id),
        one group per article per round, at most two per article (counting
        an already-selected lock)."""
        by_article: dict[str, list[dict]] = {}
        for group in self.groups:
            if id(group) not in self.selected:
                by_article.setdefault(_rep_pmcid(group), []).append(group)
        order = sorted(by_article, key=lambda pmcid: self.rank(by_article[pmcid][0]["rep"]))
        while not self.full() and self._round(order, by_article):
            pass

    def _round(self, order: list[str], by_article: dict[str, list[dict]]) -> bool:
        progressed = False
        for pmcid in order:
            if self.full():
                break
            if self.article_counts.get(pmcid, 0) >= 2:
                continue
            remaining = [g for g in by_article[pmcid] if id(g) not in self.selected]
            if remaining:
                self.select(remaining[0], "published")
                progressed = True
        return progressed

    def fill(self) -> None:
        """The per-article preference is soft; fill remaining capacity from
        the rest of the distinct groups in tier, then score order."""
        for group in self.groups:
            if self.full():
                break
            if id(group) not in self.selected:
                self.select(group, "published")

    def reserve_ids(self) -> list[str]:
        reserves = [str(g["rep"]["panel_id"]) for g in self.groups if id(g) not in self.selected]
        for rep_id in reserves:
            self.reasons[rep_id] = {"status": "gallery_full"}
        return reserves

    def record_aliases(self) -> None:
        for group in self.groups:
            rep_id = str(group["rep"]["panel_id"])
            member_ids = [str(m["panel_id"]) for m in group["members"]]
            for member_id in member_ids:
                if member_id != rep_id:
                    self.reasons[member_id] = {
                        "status": _alias_status(member_id, member_ids, group["keys_of"]),
                        "representative_id": rep_id,
                    }


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
    gallery entry; otherwise the group lead is the best source tier
    (``article_tier``: review, series/study, case report, atypical), then
    the best ``representatives.score_panel`` score, ties ascending by panel_id.
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

    def rank(panel: dict) -> tuple[int, float, str]:
        panel_id = str(panel["panel_id"])
        return (int(panel.get("article_tier", TIER_DEFAULT)), -scores[panel_id], panel_id)

    locked = str(locked_panel_id) if locked_panel_id is not None else None
    locked_group = _choose_representatives(groups, locked, rank)
    groups.sort(key=lambda group: rank(group["rep"]))

    selection = _Selection(groups, cap, rank)
    if locked_group is not None:
        selection.select(locked_group, "locked")
    selection.round_robin()
    selection.fill()
    reserve_ids = selection.reserve_ids()
    selection.record_aliases()
    return {
        "published_panel_ids": selection.published,
        "reserve_panel_ids": reserve_ids,
        "selection_reasons": selection.reasons,
        "identity_unknown_count": sum(
            1 for group in selection.published_groups if not group["documented"]
        ),
        "eligible_distinct": len(groups),
    }


# Pseudo finding key that routes unlabeled candidates through select_gallery.
_UNLABELED = "__unlabeled__"


def select_unlabeled(
    records: list[dict],
    disease_key: str,
    shown: set[str],
    *,
    target: int | None = None,
) -> list[str]:
    """Unlabeled images that top a topic's default view up to ``target``.

    Candidates passed every publication gate except naming their finding
    (``unlabeled_publishable``). They fill only the slots left after the
    ``shown`` panels (distinct by hash), ranked and grouped exactly as a pair
    gallery is, and earn no pair credit. Bytes already eligible as finding
    evidence, published or reserve, never re-enter through this path.
    """
    target = config.VP_UNLABELED_FILL_TARGET if target is None else target
    hash_of = {str(r["panel_id"]): str(r.get("sha256") or r["panel_id"]) for r in records}
    slots = target - len({hash_of.get(pid, pid) for pid in shown})
    if slots < 1:
        return []
    taken = {str(r["sha256"]) for r in records if r.get("eligible") and r.get("sha256")}
    candidates = [
        {**r, "eligible": True, "supported_finding_keys": [_UNLABELED]}
        for r in records
        if r.get("unlabeled_publishable")
        and str(r.get("disease_key") or "") == disease_key
        and not (r.get("sha256") and str(r["sha256"]) in taken)
    ]
    if not candidates:
        return []
    return select_gallery(candidates, disease_key, _UNLABELED, cap=slots)["published_panel_ids"]


def library_panel_ids(
    snapshot: dict, records: list[dict], disease_key: str, *, cap: int | None = None,
) -> set[str]:
    """One topic's default library: the panel ids every default view shows.

    The union of each pair's selected gallery, eligible whole-figure combined
    plates (visible, no per-finding credit), then the unlabeled fill. A
    ``cap`` (default ``VP_TOPIC_IMAGE_CAP``; 0 = none) keeps that many
    distinct images, taking pair galleries round-robin (each pair's best
    first), then combined plates, then the fill.
    """
    cap = config.VP_TOPIC_IMAGE_CAP if cap is None else cap
    records = [r for r in records if str(r.get("disease_key") or "") == disease_key]
    pairs = snapshot.get(disease_key) or {}
    galleries = [
        [str(pid) for pid in pairs[key].get("published_panel_ids") or []]
        for key in sorted(pairs)
    ]
    ordered = [
        g[i]
        for i in range(max((len(g) for g in galleries), default=0))
        for g in galleries if i < len(g)
    ]
    ordered += sorted(
        str(r["panel_id"]) for r in records
        if r.get("eligible") and r.get("plate_kind") == "combined"
    )
    ordered += select_unlabeled(records, disease_key, set(ordered))
    if cap < 1:
        return set(ordered)
    hash_of = {str(r["panel_id"]): str(r.get("sha256") or r["panel_id"]) for r in records}
    selected: set[str] = set()
    hashes: set[str] = set()
    for pid in ordered:
        sha = hash_of.get(pid, pid)
        if sha not in hashes and len(hashes) >= cap:
            continue
        hashes.add(sha)
        selected.add(pid)
    return selected


def published_coverage(conn, disease_key: str) -> dict[str, int]:
    """Distinct *published* gallery images per approved finding.

    Counts the same grouped representatives the viewer publishes — eligible
    reserves and duplicate aliases never inflate a pair's count. Every
    approved pair is present, including empty ones.
    """
    snapshot = coverage_snapshot(conn, disease_key)
    return {
        finding_key: int(record.get("published_distinct") or 0)
        for finding_key, record in (snapshot.get(disease_key) or {}).items()
    }


def coverage_snapshot(conn, disease_key: str | None = None) -> dict:
    """Coverage snapshot whose selections come from ``select_gallery``."""
    from . import coverage

    return coverage.snapshot(conn, select_gallery, disease_key)
