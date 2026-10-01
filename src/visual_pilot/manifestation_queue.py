"""Persistent finding-specific article queues and deficit-first batch selection."""

from __future__ import annotations

from . import config, db


def published_coverage(conn, disease_key: str) -> dict[str, int]:
    """Distinct *published* gallery images per approved finding.

    Delegates to the frozen gallery snapshot so coverage counts the same
    grouped representatives the viewer publishes — eligible reserves and
    duplicate aliases never inflate a lane's count. Every approved pair is
    present, including empty ones.
    """
    from . import gallery

    snapshot = gallery.coverage_snapshot(conn, disease_key)
    return {
        finding_key: int(record.get("published_distinct") or 0)
        for finding_key, record in (snapshot.get(disease_key) or {}).items()
    }


def sync_candidates(conn, disease_key: str) -> dict[str, int]:
    """Refresh lane definitions and import finding-keyed retrieval evidence.

    Only evidence carrying explicit ``disease_key`` provenance activates a
    lane candidate — legacy ambiguous rows stay retained but unresolved and
    are never inferred from query text or article membership. Lane status is
    ``covered`` only at the published-gallery cap; tiers between target and
    cap remain open as expanding. Returns ``{finding_key:
    published_distinct}`` for findings under the cap, zero included.
    """
    from . import coverage as coverage_mod

    _floor, _target, cap = config.validate_coverage_settings()
    policy_version = config.PAIR_SEARCH_POLICY_VERSION
    vocab = {
        row["finding_key"]: set(db.from_json(row["disease_keys_json"], []) or [])
        for row in conn.execute(
            "SELECT finding_key, disease_keys_json FROM findings_vocab WHERE approved=1"
        )
    }
    coverage = published_coverage(conn, disease_key)
    findings = sorted(key for key, diseases in vocab.items() if disease_key in diseases)
    lanes = {
        row["finding_key"]: dict(row)
        for row in conn.execute(
            "SELECT * FROM manifestation_lanes WHERE disease_key=?", (disease_key,)
        )
    }
    for finding_key in findings:
        count = coverage.get(finding_key, 0)
        status = "covered" if count >= cap else "open"
        outcome = "published_panel_exists" if status == "covered" else None
        tier = coverage_mod.coverage_tier(count)
        lane = lanes.get(finding_key)
        if lane is None:
            conn.execute(
                "INSERT INTO manifestation_lanes"
                "(disease_key,finding_key,status,last_outcome,tier,"
                " last_published_distinct,search_policy_version) "
                "VALUES(?,?,?,?,?,?,?)",
                (disease_key, finding_key, status, outcome, tier, count,
                 policy_version),
            )
            continue
        # A search plan exhausted under an older policy version reopens;
        # other persisted blocks (paused/exhausted) survive the refresh.
        blocked = lane.get("blocked_reason")
        if blocked == "search_plan_exhausted" and (
            lane.get("search_policy_version") or ""
        ) != policy_version:
            blocked = None
        # last_published_distinct tracks the CURRENT count (a curation drop
        # reopens the lane); the reduction timestamp only moves forward.
        prior_count = int(lane.get("last_published_distinct") or 0)
        last_published, reduced_at = count, count > prior_count
        conn.execute(
            "UPDATE manifestation_lanes SET status=?, "
            "last_outcome=CASE WHEN ?='covered' THEN 'published_panel_exists' "
            "ELSE last_outcome END, tier=?, last_published_distinct=?, "
            "last_deficit_reduction_at=CASE WHEN ? THEN datetime('now') "
            "ELSE last_deficit_reduction_at END, blocked_reason=?, "
            "updated_at=datetime('now') "
            "WHERE disease_key=? AND finding_key=?",
            (
                status, status, tier, last_published, reduced_at, blocked,
                disease_key, finding_key,
            ),
        )

    for finding_key, n_panels in coverage.items():
        if n_panels >= cap:
            conn.execute(
                "UPDATE manifestation_candidates SET status='covered',last_outcome='published_panel_exists', "
                "updated_at=datetime('now') WHERE disease_key=? AND finding_key=? "
                "AND provenance_status='explicit'",
                (disease_key, finding_key),
            )

    for article in conn.execute(
        "SELECT pmcid,retrieval_score,retrieval_evidence_json FROM articles"
    ):
        evidence = db.from_json(article["retrieval_evidence_json"], []) or []
        best: dict[str, tuple[int | None, str | None, float | None]] = {}
        for item in evidence:
            if not isinstance(item, dict):
                continue
            # Explicit pair provenance only; legacy items without a disease
            # key stay unresolved rather than activating this disease's lane.
            if str(item.get("disease_key") or "") != disease_key:
                continue
            finding_key = str(item.get("finding_key") or "")
            if not finding_key or disease_key not in vocab.get(finding_key, set()):
                continue
            rank_value = item.get("best_rank", item.get("rank"))
            try:
                rank = int(rank_value) if rank_value is not None else None
            except (TypeError, ValueError):
                rank = None
            score_value = item.get("retrieval_score", item.get("score"))
            try:
                score = float(score_value) if score_value is not None else article["retrieval_score"]
            except (TypeError, ValueError):
                score = article["retrieval_score"]
            prior = best.get(finding_key)
            if prior is None or (
                rank is not None and (prior[0] is None or rank < prior[0])
            ) or (rank == prior[0] and (score or 0) > (prior[2] or 0)):
                best[finding_key] = (rank, item.get("query"), score)
        for finding_key, (rank, query, score) in best.items():
            if coverage.get(finding_key, 0) >= cap:
                status = "covered"
                outcome = "published_panel_exists"
            else:
                status = "pending"
                outcome = None
            conn.execute(
                "INSERT INTO manifestation_candidates "
                "(disease_key,finding_key,pmcid,query,best_rank,retrieval_score,status,last_outcome,"
                "provenance_status,provenance_disease_key) "
                "VALUES(?,?,?,?,?,?,?,?,'explicit',?) ON CONFLICT(disease_key,finding_key,pmcid) "
                "DO UPDATE SET query=COALESCE(excluded.query,manifestation_candidates.query), "
                "best_rank=CASE WHEN excluded.best_rank IS NULL THEN manifestation_candidates.best_rank "
                "WHEN manifestation_candidates.best_rank IS NULL THEN excluded.best_rank "
                "ELSE MIN(manifestation_candidates.best_rank,excluded.best_rank) END, "
                "retrieval_score=MAX(COALESCE(manifestation_candidates.retrieval_score,0), "
                "COALESCE(excluded.retrieval_score,0)), "
                "status=CASE WHEN manifestation_candidates.status IN ('parsed','parse_error','covered') "
                "THEN manifestation_candidates.status ELSE excluded.status END, "
                "provenance_status='explicit', "
                "provenance_disease_key=excluded.provenance_disease_key, "
                "updated_at=datetime('now')",
                (disease_key, finding_key, article["pmcid"], query, rank, score, status, outcome,
                 disease_key),
            )

    # Under-cap findings reopen their 'covered' candidates first, so the
    # durable-outcome updates below still re-terminate parsed/errored rows.
    for finding_key in findings:
        if coverage.get(finding_key, 0) >= cap:
            continue
        conn.execute(
            "UPDATE manifestation_candidates SET status='pending', "
            "last_outcome='reopened_below_cap', updated_at=datetime('now') "
            "WHERE disease_key=? AND finding_key=? AND status='covered' "
            "AND provenance_status='explicit'",
            (disease_key, finding_key),
        )

    # Reflect durable article outcomes on explicit candidates only —
    # unresolved legacy rows stay retained and untouched, never reset.
    conn.execute(
        "UPDATE manifestation_candidates SET status='parsed',last_outcome='article_already_parsed', "
        "updated_at=datetime('now') WHERE disease_key=? AND provenance_status='explicit' "
        "AND pmcid IN "
        "(SELECT pmcid FROM articles WHERE status='parsed') AND status NOT IN ('covered')",
        (disease_key,),
    )
    conn.execute(
        "UPDATE manifestation_candidates SET status='parse_error',last_outcome='article_parse_error', "
        "updated_at=datetime('now') WHERE disease_key=? AND provenance_status='explicit' AND pmcid IN "
        "(SELECT pmcid FROM articles WHERE status='parse_error') AND status NOT IN ('covered')",
        (disease_key,),
    )
    for article_status in ("license_rejected", "irrelevant"):
        conn.execute(
            "UPDATE manifestation_candidates SET status='exhausted',last_outcome=?, "
            "updated_at=datetime('now') WHERE disease_key=? AND provenance_status='explicit' "
            "AND pmcid IN "
            "(SELECT pmcid FROM articles WHERE status=?) "
            "AND status NOT IN ('covered','parsed','parse_error')",
            (f"article_{article_status}", disease_key, article_status),
        )
    # A relevant article whose primary disease keys form a NON-EMPTY set
    # lacking this pair's disease can never satisfy the lane — terminate it.
    # A NULL/empty set is unlabeled (attribution not yet recorded), not an
    # affirmative exclusion, so it does not terminate the candidate.
    conn.execute(
        "UPDATE manifestation_candidates SET status='exhausted', "
        "last_outcome='article_other_disease', updated_at=datetime('now') "
        "WHERE disease_key=? AND provenance_status='explicit' "
        "AND status IN ('pending','selected') AND pmcid IN "
        "(SELECT pmcid FROM articles WHERE status='relevant' "
        "AND EXISTS (SELECT 1 FROM json_each(primary_disease_keys_json)) "
        "AND NOT EXISTS "
        "(SELECT 1 FROM json_each(primary_disease_keys_json) WHERE value=?))",
        (disease_key, disease_key),
    )
    conn.commit()
    return {
        key: coverage.get(key, 0)
        for key in findings
        if coverage.get(key, 0) < cap
    }


def actionable_lanes(snapshot, disease_keys=None) -> list[dict]:
    """Global deficit-first lane order from the settled scheduling policy."""
    from . import scheduling_policy

    return scheduling_policy.actionable_lanes(snapshot, disease_keys)


def _lane_pool(conn, disease_key: str, finding_key: str, ranked: list[dict]) -> list[dict]:
    """Relevant, explicitly-provenanced candidates for one lane, pair-ranked."""
    candidate_ids = {
        row["pmcid"]
        for row in conn.execute(
            "SELECT mc.pmcid FROM manifestation_candidates mc JOIN articles a USING(pmcid) "
            "WHERE mc.disease_key=? AND mc.finding_key=? AND a.status='relevant' "
            "AND mc.status IN ('pending','selected') "
            "AND mc.provenance_status='explicit' "
            "AND mc.provenance_disease_key=mc.disease_key "
            # A non-empty primary-disease set that lacks this disease is an
            # affirmative exclusion; NULL/empty is unlabeled, not an
            # exclusion, so it does not disqualify the pair. Explicit pair
            # provenance stays strictly required either way.
            "AND (NOT EXISTS (SELECT 1 FROM json_each(a.primary_disease_keys_json)) "
            "OR EXISTS (SELECT 1 FROM json_each(a.primary_disease_keys_json) "
            "WHERE value=?))",
            (disease_key, finding_key, disease_key),
        )
    }
    rows = [row for row in ranked if row["pmcid"] in candidate_ids]
    # Pair ordering replaces the old disease-global order *inside* each lane.
    if any(finding_key in (row.get("pair_rankings") or {}) for row in rows):
        from . import pair_rank

        rows = sorted(
            rows, key=lambda row: pair_rank.sort_key(row, finding_key),
            reverse=True,
        )
    return rows


def _persist_global_reservation(conn, selected: list[dict]) -> None:
    """Mark served candidates/lanes; shared articles advance every lane.

    Each chosen article gets the next global service sequence, applied to
    every lane it serves, so batch-size-one calls and cross-run fairness
    share one durable ordering.
    """
    for row in selected:
        sequence = conn.execute(
            "SELECT COALESCE(MAX(last_served_sequence), 0) + 1 AS n "
            "FROM manifestation_lanes"
        ).fetchone()["n"]
        pmcid = row["pmcid"]
        for disease_key, finding_key in row.get("service_pairs") or []:
            conn.execute(
                "UPDATE manifestation_candidates SET status='selected', "
                "last_outcome='selected_for_lane', updated_at=datetime('now') "
                "WHERE disease_key=? AND finding_key=? AND pmcid=? "
                "AND status IN ('pending','selected') "
                "AND provenance_status='explicit'",
                (disease_key, finding_key, pmcid),
            )
            conn.execute(
                "UPDATE manifestation_lanes SET status='open', "
                "last_outcome='batch_reserved', last_served_sequence=?, "
                "selected_count=selected_count+1, updated_at=datetime('now') "
                "WHERE disease_key=? AND finding_key=?",
                (sequence, disease_key, finding_key),
            )
    conn.commit()


def reserve_global_batch(
    conn,
    ranked_by_disease: dict[str, list[dict]],
    batch_size: int,
    *,
    persist: bool = True,
    disease_keys=None,
    snapshot=None,
) -> list[dict]:
    """Reserve one global batch across the highest-tier actionable lanes.

    Candidate pools contain only relevant articles whose lane candidate rows
    carry explicit provenance matching the pair's disease. A shared PMCID
    consumes one slot and serves every matching lane; spare capacity is
    never filled with articles unrelated to an actionable lane.
    """
    from . import gallery, scheduling_policy, source_quality

    capacity = max(0, int(batch_size))
    if not capacity:
        return []
    snapshot = snapshot if snapshot is not None else gallery.coverage_snapshot(conn)
    lanes = scheduling_policy.actionable_lanes(snapshot, disease_keys)
    if not lanes:
        return []
    pools: dict[tuple[str, str], list[dict]] = {}
    for lane in lanes:
        disease_key, finding_key = lane["disease_key"], lane["finding_key"]
        ranked = [
            row for row in (ranked_by_disease.get(disease_key) or [])
            if not source_quality.quality_signal(row.get("source_metadata"))["retracted"]
        ]
        pool = _lane_pool(conn, disease_key, finding_key, ranked)
        if pool:
            pools[(disease_key, finding_key)] = pool
    selected = scheduling_policy.reserve_plan(lanes, pools, capacity)
    if persist and selected:
        _persist_global_reservation(conn, selected)
    return selected


def reserve_batch(conn, disease_key: str, ranked: list[dict], batch_size: int,
                  uncovered: set[str] | dict[str, int] | None, *, persist: bool = True) -> list[dict]:
    """Single-disease compatibility wrapper over the global reservation.

    ``uncovered`` is a set of under-cap findings or the
    ``{finding_key: count}`` dict from ``sync_candidates``/``coverage_gaps``;
    it scopes which of the disease's actionable lanes may reserve. Spare
    capacity is never filled with articles unrelated to an actionable lane.
    """
    from . import gallery, scheduling_policy, source_quality

    capacity = max(0, int(batch_size))
    if not capacity:
        return []
    snapshot = gallery.coverage_snapshot(conn)
    lanes = actionable_lanes(snapshot, [disease_key])
    if uncovered is not None:
        wanted = set(uncovered)
        lanes = [lane for lane in lanes if lane["finding_key"] in wanted]
    clean = [
        row for row in ranked
        if not source_quality.quality_signal(row.get("source_metadata"))["retracted"]
    ]
    pools: dict[tuple[str, str], list[dict]] = {}
    for lane in lanes:
        pool = _lane_pool(conn, disease_key, lane["finding_key"], clean)
        if pool:
            pools[(disease_key, lane["finding_key"])] = pool
    selected = scheduling_policy.reserve_plan(lanes, pools, capacity)
    if persist and selected:
        _persist_global_reservation(conn, selected)
    return selected


def record_article_outcome(conn, pmcid: str, outcome: str) -> None:
    """Persist parse outcomes for every lane associated with an article."""
    status = "parsed" if outcome == "parsed" else "parse_error" if outcome == "parse_error" else "pending"
    conn.execute(
        "UPDATE manifestation_candidates SET status=?,last_outcome=?,updated_at=datetime('now') "
        "WHERE pmcid=? AND status!='covered' AND provenance_status='explicit'",
        (status, f"article_{outcome}", pmcid),
    )
    conn.commit()


def record_published_outcomes(conn, disease_key: str) -> None:
    """Mark finding lanes covered from the publication view after store."""
    _floor, _target, cap = config.validate_coverage_settings()
    coverage = published_coverage(conn, disease_key)
    for finding_key, count in coverage.items():
        if count < cap:
            continue
        conn.execute(
            "UPDATE manifestation_candidates SET status='covered',last_outcome='published_panel_created', "
            "updated_at=datetime('now') WHERE disease_key=? AND finding_key=? "
            "AND provenance_status='explicit'",
            (disease_key, finding_key),
        )
        conn.execute(
            "UPDATE manifestation_lanes SET status='covered',last_outcome='published_panel_created', "
            "tier='full',updated_at=datetime('now') WHERE disease_key=? AND finding_key=?",
            (disease_key, finding_key),
        )
    conn.commit()
