"""Pair-specific funnel definitions; article membership is not figure support."""

from __future__ import annotations

from collections import Counter

from . import config, coverage, curation, db, pmc, publication


def caption_supports_pair(figure: dict, article: dict, disease: str, finding: dict) -> bool:
    caption = str(figure.get("caption") or "")
    mentioned = curation._diseases_in_text(caption)
    target_in_caption = disease in mentioned
    if (mentioned - {disease} or curation._has_nonpilot_disease(caption)) and not target_in_caption:
        return False
    if not target_in_caption and disease not in curation._diseases_in_text(str(article.get("title") or "")):
        return False
    mentions = db.from_json(figure.get("in_text_mentions_json"), []) or []
    text = " ".join(" ".join([caption, *map(str, mentions)]).casefold().split())
    terms = publication.finding_terms([finding])[finding["finding_key"]]
    return publication.finding_supported({"key": finding["finding_key"]}, text, terms)


def next_action(record: dict, pending: int) -> str:
    if record["tier"] == "full":
        return "Gallery full; retain and inspect reserves"
    if record["blocked_reason"]:
        return f"Review block: {record['blocked_reason']}; deficit remains"
    if pending:
        return "Finish pending pair candidates before new retrieval"
    return "Attempt the next bounded disease-scoped search strategy"


def pair_funnel(conn, *, snapshot=None, records=None) -> dict:
    from . import gallery

    with coverage.consistent_read(conn):
        snapshot = snapshot if snapshot is not None else gallery.coverage_snapshot(conn)
        records = records if records is not None else publication.panel_records(conn)
        vocab = {row["finding_key"]: dict(row) for row in conn.execute(
            "SELECT * FROM findings_vocab WHERE approved=1"
        )}
        articles = {row["pmcid"]: dict(row) for row in conn.execute("SELECT * FROM articles")}
        figures = [dict(row) for row in conn.execute("SELECT * FROM figures")]
        candidates = [dict(row) for row in conn.execute("SELECT * FROM manifestation_candidates")]
        attempts = [dict(row) for row in conn.execute(
            "SELECT * FROM pair_search_attempts ORDER BY policy_version,round_no,attempt_id"
        )]
        out = {}
        for disease, findings in snapshot.items():
            disease_out = out.setdefault(disease, {})
            for finding, record in findings.items():
                matched = [
                    row for row in candidates
                    if row["disease_key"] == disease and row["finding_key"] == finding
                ]
                explicit = [
                    row for row in matched
                    if row["provenance_status"] == "explicit"
                    and row["provenance_disease_key"] == disease
                ]
                ids = {row["pmcid"] for row in explicit}
                licensed = {
                    pmcid for pmcid in ids if pmcid in articles
                    and pmc.license_allows(pmc.normalize_license(articles[pmcid].get("license_code"))) is not None
                }
                supported_figures = {
                    row["figure_id"] for row in figures
                    if row["pmcid"] in ids and caption_supports_pair(
                        row, articles.get(row["pmcid"], {}), disease, vocab[finding],
                    )
                }
                pending = {
                    row["pmcid"] for row in explicit
                    if articles.get(row["pmcid"], {}).get("status") in {"candidate", "license_ok", "relevant"}
                    and row["status"] in {"pending", "selected", "covered"}
                    and (
                        articles[row["pmcid"]]["status"] != "relevant"
                        or disease in db.from_json(articles[row["pmcid"]]["primary_disease_keys_json"], [])
                    )
                }
                rejected = Counter()
                for panel in records:
                    if panel.get("disease_key") != disease or panel["eligible"]:
                        continue
                    raw = [
                        *(db.from_json(panel.get("findings_json"), []) or []),
                        *(db.from_json(panel.get("plate_findings_json"), []) or []),
                    ]
                    keys = {item.get("finding_key") if isinstance(item, dict) else item for item in raw}
                    if finding in keys:
                        rejected.update(panel["rejection_categories"])
                for row in explicit:
                    status = articles.get(row["pmcid"], {}).get("status")
                    if status == "license_rejected":
                        rejected["license/third-party"] += 1
                strategies = []
                for attempt in attempts:
                    if attempt["disease_key"] != disease or attempt["finding_key"] != finding:
                        continue
                    strategies.append({
                        "policy_version": attempt["policy_version"],
                        "round_no": attempt["round_no"], "query": attempt["query"],
                        "query_filter_hash": attempt["query_filter_hash"],
                        "depth": attempt["depth"], "status": attempt["status"],
                        "returned_unique_articles": len(set(db.from_json(attempt["returned_pmcids_json"], []) or [])),
                        "new_unique_articles": len(set(db.from_json(attempt["new_pmcids_json"], []) or [])),
                        "error": attempt["error"], "completed_at": attempt["completed_at"],
                    })
                    if attempt["error"]:
                        rejected["retrieval error"] += 1
                rejection_categories = {
                    category: rejected.get(category, 0) for category in (
                        "license/third-party", "review type", "no patient image", "age unclear",
                        "attribution unclear/other disease", "unsupported finding", "mixed plate",
                        "quality", "retrieval error",
                    )
                }
                disease_out[finding] = {
                    **record,
                    "retrieved_unique_articles": len(ids),
                    "licensed_articles": len(licensed),
                    "pair_supported_caption_figures": len(supported_figures),
                    "unresolved_legacy_candidates": len(matched) - len(explicit),
                    "pending_work": len(pending),
                    "attempted_strategies": strategies,
                    "last_deficit_reduction": record["last_deficit_reduction_at"],
                    "next_action": next_action(record, len(pending)),
                    "rejection_categories": rejection_categories,
                    "selection_reserves": {
                        "distinct_groups": record["reserve_distinct"],
                        "duplicate_or_diversity_rows": sum(
                            item.get("status") in {"duplicate", "same_patient", "reused_image", "source_family"}
                            for item in record["selection_reasons"].values()
                        ),
                    },
                    "milestone": (
                        "full" if not record["cap_remaining"] else
                        "target_reached" if not record["target_deficit"] else
                        "floor_reached" if not record["floor_deficit"] else "floor_deficit"
                    ),
                }
        return out


def library_rows(conn, disease_key=None, *, snapshot=None, records=None) -> list[dict]:
    """Default library union: selected pair representatives plus Combined views."""
    from . import gallery

    with coverage.consistent_read(conn):
        snapshot = snapshot if snapshot is not None else gallery.coverage_snapshot(conn, disease_key)
        records = records if records is not None else publication.panel_records(conn, disease_key)
        selected = {
            panel_id for disease, findings in snapshot.items()
            if disease_key is None or disease == disease_key
            for record in findings.values() for panel_id in record["published_panel_ids"]
        }
        return [
            row for row in records
            if row["eligible"] and (disease_key is None or row["disease_key"] == disease_key)
            and (row["panel_id"] in selected or row.get("plate_kind") == "combined")
        ]


def pair_summary(snapshot: dict) -> dict:
    """Retain the legacy histogram while exposing all three coverage milestones."""
    floor, target, cap = config.validate_coverage_settings()
    buckets = ("0", "1", "2", "3", "4", "5-6", "7-9", ">=10")
    histogram = Counter()
    per_disease = {}
    for disease, findings in sorted(snapshot.items()):
        under = []
        tiers = Counter()
        for finding, record in sorted(findings.items()):
            count = record["published_distinct"]
            bucket = ">=10" if count >= 10 else "7-9" if count >= 7 else "5-6" if count >= 5 else str(count)
            histogram[bucket] += 1
            tiers[record["tier"]] += 1
            if count < target:
                under.append({"finding_key": finding, "images": count})
        per_disease[disease] = {
            "pairs": len(findings),
            "pairs_at_floor": sum(record["floor_deficit"] == 0 for record in findings.values()),
            "pairs_at_target": sum(record["target_deficit"] == 0 for record in findings.values()),
            "pairs_full": sum(record["cap_remaining"] == 0 for record in findings.values()),
            "zero_image_pairs": tiers["empty"],
            "tier_counts": dict(tiers),
            "under_target": sorted(under, key=lambda item: (item["images"], item["finding_key"])),
        }
    return {
        "floor": floor, "target": target, "cap": cap,
        "histogram": {bucket: histogram[bucket] for bucket in buckets},
        "per_disease": per_disease,
    }


def distribution(rows: list[dict], snapshot: dict) -> dict:
    """Pair finding counts use the gallery, not tags on every stored figure."""
    out = {
        "by_modality": Counter(), "by_subtype": Counter(), "by_disease": Counter(),
        "by_finding": Counter(), "skin_tone": {},
    }
    for row in rows:
        for name, field in (("by_modality", "modality"), ("by_subtype", "subtype"), ("by_disease", "disease_key")):
            out[name][str(row.get(field))] += 1
        if row.get("modality") in {"clinical_photo", "dermoscopy", "capillaroscopy"} or row.get("skin_tone") is not None:
            out["skin_tone"].setdefault(row.get("disease_key") or "?", Counter())[row.get("skin_tone") or "unknown"] += 1
    for findings in snapshot.values():
        for finding, record in findings.items():
            out["by_finding"][finding] += record["published_distinct"]
    return {
        **{name: dict(value) for name, value in out.items() if name != "skin_tone"},
        "skin_tone": {key: dict(value) for key, value in out["skin_tone"].items()},
    }
