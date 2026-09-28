"""Persistent finding-specific article queues and deficit-first batch selection."""

from __future__ import annotations

from collections import defaultdict

from . import db


def published_coverage(conn, disease_key: str) -> dict[str, int]:
    """Count published panels per finding for one disease."""
    counts: dict[str, int] = defaultdict(int)
    for row in conn.execute(
        "SELECT findings_json FROM published_panels WHERE disease_key=?",
        (disease_key,),
    ):
        for value in db.from_json(row["findings_json"], []) or []:
            key = value.get("finding_key") if isinstance(value, dict) else value
            if key:
                counts[str(key)] += 1
    return dict(counts)


def sync_candidates(conn, disease_key: str) -> set[str]:
    """Refresh lane definitions and import finding-keyed retrieval evidence."""
    vocab = {
        row["finding_key"]: set(db.from_json(row["disease_keys_json"], []) or [])
        for row in conn.execute(
            "SELECT finding_key, disease_keys_json FROM findings_vocab WHERE approved=1"
        )
    }
    coverage = published_coverage(conn, disease_key)
    findings = sorted(key for key, diseases in vocab.items() if disease_key in diseases)
    for finding_key in findings:
        covered = coverage.get(finding_key, 0) > 0
        conn.execute(
            "INSERT INTO manifestation_lanes(disease_key,finding_key,status,last_outcome) "
            "VALUES(?,?,?,?) ON CONFLICT(disease_key,finding_key) DO UPDATE SET "
            "status=excluded.status,last_outcome=CASE WHEN excluded.status='covered' "
            "THEN 'published_panel_exists' ELSE manifestation_lanes.last_outcome END, "
            "updated_at=datetime('now')",
            (disease_key, finding_key, "covered" if covered else "open",
             "published_panel_exists" if covered else None),
        )

    for finding_key, n_panels in coverage.items():
        if n_panels:
            conn.execute(
                "UPDATE manifestation_candidates SET status='covered',last_outcome='published_panel_exists', "
                "updated_at=datetime('now') WHERE disease_key=? AND finding_key=?",
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
            if coverage.get(finding_key, 0):
                status = "covered"
                outcome = "published_panel_exists"
            elif rank is not None or query or score is not None:
                status = "pending"
                outcome = None
            else:
                status = "pending"
                outcome = None
            conn.execute(
                "INSERT INTO manifestation_candidates "
                "(disease_key,finding_key,pmcid,query,best_rank,retrieval_score,status,last_outcome) "
                "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(disease_key,finding_key,pmcid) "
                "DO UPDATE SET query=COALESCE(excluded.query,manifestation_candidates.query), "
                "best_rank=CASE WHEN excluded.best_rank IS NULL THEN manifestation_candidates.best_rank "
                "WHEN manifestation_candidates.best_rank IS NULL THEN excluded.best_rank "
                "ELSE MIN(manifestation_candidates.best_rank,excluded.best_rank) END, "
                "retrieval_score=MAX(COALESCE(manifestation_candidates.retrieval_score,0), "
                "COALESCE(excluded.retrieval_score,0)), "
                "status=CASE WHEN manifestation_candidates.status IN ('parsed','parse_error','covered') "
                "THEN manifestation_candidates.status ELSE excluded.status END, "
                "updated_at=datetime('now')",
                (disease_key, finding_key, article["pmcid"], query, rank, score, status, outcome),
            )

    # Reflect durable article outcomes, and retire lanes whose candidate pool
    # contains no article that can be selected by the current parser.
    conn.execute(
        "UPDATE manifestation_candidates SET status='parsed',last_outcome='article_already_parsed', "
        "updated_at=datetime('now') WHERE disease_key=? AND pmcid IN "
        "(SELECT pmcid FROM articles WHERE status='parsed') AND status NOT IN ('covered')",
        (disease_key,),
    )
    conn.execute(
        "UPDATE manifestation_candidates SET status='parse_error',last_outcome='article_parse_error', "
        "updated_at=datetime('now') WHERE disease_key=? AND pmcid IN "
        "(SELECT pmcid FROM articles WHERE status='parse_error') AND status NOT IN ('covered')",
        (disease_key,),
    )
    for article_status in ("license_rejected", "irrelevant"):
        conn.execute(
            "UPDATE manifestation_candidates SET status='exhausted',last_outcome=?, "
            "updated_at=datetime('now') WHERE disease_key=? AND pmcid IN "
            "(SELECT pmcid FROM articles WHERE status=?) "
            "AND status NOT IN ('covered','parsed','parse_error')",
            (f"article_{article_status}", disease_key, article_status),
        )
    for finding_key in findings:
        if coverage.get(finding_key, 0):
            continue
        eligible = conn.execute(
            "SELECT COUNT(*) n FROM manifestation_candidates mc JOIN articles a USING(pmcid) "
            "WHERE mc.disease_key=? AND mc.finding_key=? AND a.status='relevant' "
            "AND mc.status IN ('pending','selected')",
            (disease_key, finding_key),
        ).fetchone()["n"]
        if not eligible:
            conn.execute(
                "UPDATE manifestation_lanes SET status='exhausted', "
                "last_outcome=CASE WHEN EXISTS (SELECT 1 FROM manifestation_candidates mc "
                "WHERE mc.disease_key=? AND mc.finding_key=?) THEN 'all_candidates_terminal' "
                "ELSE 'no_retrieved_candidates' END, updated_at=datetime('now') "
                "WHERE disease_key=? AND finding_key=? AND status!='covered'",
                (disease_key, finding_key, disease_key, finding_key),
            )
        else:
            conn.execute(
                "UPDATE manifestation_lanes SET status='open',last_outcome=COALESCE(last_outcome,'candidates_available'), "
                "updated_at=datetime('now') WHERE disease_key=? AND finding_key=? AND status!='covered'",
                (disease_key, finding_key),
            )
    conn.commit()
    return {key for key in findings if coverage.get(key, 0) == 0}


def reserve_batch(conn, disease_key: str, ranked: list[dict], batch_size: int,
                  uncovered: set[str], *, persist: bool = True) -> list[dict]:
    """Reserve lane slots round-robin, then fill spare capacity globally.

    An article appearing in several finding queues occupies one batch slot and
    marks each matching lane candidate selected, so it can advance several
    lanes without being parsed twice.
    """
    capacity = max(0, int(batch_size))
    if not capacity or not ranked:
        return []
    lanes = sorted(uncovered)
    candidate_sets: dict[str, set[str]] = {}
    for finding_key in lanes:
        candidate_sets[finding_key] = {
            row["pmcid"] for row in conn.execute(
                "SELECT mc.pmcid FROM manifestation_candidates mc JOIN articles a USING(pmcid) "
                "WHERE mc.disease_key=? AND mc.finding_key=? AND a.status='relevant' "
                "AND mc.status IN ('pending','selected','exhausted')",
                (disease_key, finding_key),
            )
        }

    selected: list[dict] = []
    selected_ids: set[str] = set()
    selected_lanes: dict[str, set[str]] = defaultdict(set)
    remaining_lanes = set(lanes)
    while len(selected) < capacity and remaining_lanes:
        progressed = False
        for finding_key in lanes:
            if finding_key not in remaining_lanes or len(selected) >= capacity:
                continue
            candidate = next((row for row in ranked
                              if row["pmcid"] in candidate_sets[finding_key]), None)
            if candidate is None:
                remaining_lanes.discard(finding_key)
                continue
            pmcid = candidate["pmcid"]
            if pmcid in selected_ids:
                selected_lanes[pmcid].add(finding_key)
                remaining_lanes.discard(finding_key)
                progressed = True
                continue
            selected.append(candidate)
            selected_ids.add(pmcid)
            selected_lanes[pmcid].add(finding_key)
            # One slot satisfies this lane's first-turn reservation; later
            # turns can allocate further articles if batch capacity remains.
            candidate_sets[finding_key].discard(pmcid)
            progressed = True
        if not progressed:
            break

    # Fill unused slots with the existing global ranking, preserving its
    # caption peek, rescue rules, and tie-break behavior.
    for row in ranked:
        if len(selected) >= capacity:
            break
        if row["pmcid"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["pmcid"])

    if persist and selected_ids:
        for pmcid in selected_ids:
            conn.execute(
                "UPDATE manifestation_candidates SET status='selected', "
                "last_outcome=CASE WHEN finding_key IN (%s) THEN 'selected_for_lane' "
                "ELSE 'shared_article_selected' END, updated_at=datetime('now') "
                "WHERE disease_key=? AND pmcid=? AND status IN ('pending','selected','exhausted')"
                % (",".join("?" for _ in lanes) or "NULL"),
                (*lanes, disease_key, pmcid),
            )
        for finding_key in lanes:
            if any(finding_key in selected_lanes[p] for p in selected_ids):
                conn.execute(
                    "UPDATE manifestation_lanes SET status='open',last_outcome='batch_reserved', "
                    "selected_count=selected_count+1,updated_at=datetime('now') "
                    "WHERE disease_key=? AND finding_key=?",
                    (disease_key, finding_key),
                )
        conn.commit()
    return selected


def record_article_outcome(conn, pmcid: str, outcome: str) -> None:
    """Persist parse outcomes for every lane associated with an article."""
    status = "parsed" if outcome == "parsed" else "parse_error" if outcome == "parse_error" else "pending"
    conn.execute(
        "UPDATE manifestation_candidates SET status=?,last_outcome=?,updated_at=datetime('now') "
        "WHERE pmcid=? AND status!='covered'",
        (status, f"article_{outcome}", pmcid),
    )
    conn.commit()


def record_published_outcomes(conn, disease_key: str) -> None:
    """Mark finding lanes covered from the publication view after store."""
    coverage = published_coverage(conn, disease_key)
    for finding_key in coverage:
        conn.execute(
            "UPDATE manifestation_candidates SET status='covered',last_outcome='published_panel_created', "
            "updated_at=datetime('now') WHERE disease_key=? AND finding_key=?",
            (disease_key, finding_key),
        )
        conn.execute(
            "UPDATE manifestation_lanes SET status='covered',last_outcome='published_panel_created', "
            "updated_at=datetime('now') WHERE disease_key=? AND finding_key=?",
            (disease_key, finding_key),
        )
    conn.commit()
