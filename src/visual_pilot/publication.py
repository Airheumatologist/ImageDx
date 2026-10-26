"""Current publication eligibility, shared by galleries and their consumers."""

from __future__ import annotations

from . import curation, db, demographics, diseases, jats, pair_terms, pmc, source_quality


def finding_terms(rows) -> dict[str, list[str]]:
    """Source phrasings per finding: key, label, synonyms, and caption terms.

    Caption terms (``pair_terms.caption_terms``) drop the disease and
    modality words vocabulary labels carry ("Systemic-sclerosis digital
    ulcers" -> "digital ulcer"), the way figure captions name findings.
    """
    catalog = diseases.load_diseases()
    terms = {}
    for row in rows:
        row = dict(row)
        synonyms = row.get("synonyms")
        if synonyms is None:
            synonyms = db.from_json(row.get("synonyms_json"), [])
        disease_keys = row.get("disease_keys")
        if disease_keys is None:
            disease_keys = db.from_json(row.get("disease_keys_json"), []) or []
        values = [
            str(row["finding_key"]).replace("_", " "),
            str(row.get("label") or "").split("(", 1)[0],
            *(synonyms or []),
        ]
        finding = {**row, "synonyms": list(synonyms or [])}
        for disease_key in disease_keys:
            if disease_key in catalog:
                values += pair_terms.caption_terms(disease_key, finding, catalog[disease_key])
        terms[row["finding_key"]] = list(dict.fromkeys(
            " ".join(str(value).casefold().split()) for value in values if value
        ))
    return terms


def finding_supported(finding: dict, source_text: str, terms: list[str]) -> bool:
    """Copied model evidence alone is not source support for its assigned label.

    The source text must name the finding: a term as a substring, or a
    plural-tolerant caption match (every content word of a multi-word term
    in the same text, as discovery searches captions).
    """
    if any(term and term in source_text for term in terms):
        return True
    return pair_terms.caption_matches(source_text, terms, mode="words")


def _norm(value) -> str:
    return str(value or "").strip().lower()


# Automated visual tags with a clear modality/site mismatch are rejected.
_HISTOLOGY_MODALITIES = {"histology_he", "histology_ihc", "immunofluorescence"}
_IMAGING_MODALITIES = {"mri", "ct", "radiograph", "ultrasound", "echo", "pet"}
_SURFACE_CATEGORIES = {"skin", "nail", "mucosa", "clinical_msk"}


def modality_mismatch(modality: str, body_site: str, category: str) -> bool:
    """A finding category the panel's (normalized) modality and site rule out."""
    if modality == "ophthalmic":
        return category != "eye"
    if modality in _HISTOLOGY_MODALITIES and body_site in {"kidney", "renal"}:
        return category != "histology"
    return modality in _IMAGING_MODALITIES and category in _SURFACE_CATEGORIES


def _finding_lookups(figure: dict) -> tuple[dict, dict]:
    """(terms, categories) by finding key, from the figure's cache or the vocab."""
    terms = figure.get("_finding_terms")
    categories = figure.get("_finding_categories")
    vocab = diseases.load_findings_vocab() if terms is None or categories is None else []
    if terms is None:
        terms = finding_terms(vocab)
    if categories is None:
        categories = {row["finding_key"]: row["category"] for row in vocab}
    return terms, categories


def _source_findings(panel: dict, figure: dict, approved: set[str]) -> set[str]:
    terms, categories = _finding_lookups(figure)
    mentions = db.from_json(figure.get("in_text_mentions_json"), []) or []
    caption = str(figure.get("caption") or "")
    text = " ".join(" ".join([caption, *map(str, mentions)]).casefold().split())
    raw = db.from_json(panel.get("findings_json"), []) or []
    if panel.get("plate_kind") == "combined":
        raw = db.from_json(panel.get("plate_findings_json"), []) or raw
    supported = set()
    modality = _norm(panel.get("modality"))
    site = _norm(panel.get("body_site"))
    for item in raw:
        key = item.get("finding_key") if isinstance(item, dict) else item
        if key not in approved or modality_mismatch(modality, site, _norm(categories.get(key))):
            continue
        if finding_supported({"key": key}, text, terms.get(key, [])):
            supported.add(str(key))
    return supported


# Notices rather than articles; every real article type (case reports,
# original research, reviews, letters with images) may supply images.
_EXCLUDED_TYPE_PARTS = ("retraction", "retracted", "erratum", "correction", "expression of concern")


def passes_type_filter(publication_types, article_type=None) -> bool:
    labels = [str(t).strip().lower() for t in (publication_types or []) if str(t).strip()]
    labels.append(str(article_type or "").strip().lower())
    return not any(part in label for label in labels for part in _EXCLUDED_TYPE_PARTS)


def rejection_category(reason: str) -> str:
    text = reason.casefold()
    if any(term in text for term in ("license", "third-party", "third party")):
        return "license/third-party"
    if "notice" in text or "retracted" in text:
        return "source type"
    if "age" in text:
        return "age unclear"
    if any(term in text for term in ("plate", "collage")):
        return "mixed plate"
    if any(term in text for term in ("disease", "attribution", "association")):
        return "attribution unclear/other disease"
    if any(term in text for term in ("finding", "manifestation", "label")):
        return "unsupported finding"
    if any(term in text for term in ("patient image", "diagram", "chart", "veterinary", "control")):
        return "no patient image"
    return "quality"


NO_SUPPORTED_LABEL = "no approved source-supported finding label"


def _license_reasons(panel: dict, figure: dict, article: dict) -> list[str]:
    # Never replace an explicitly restrictive source license with a panel label.
    reasons = []
    for source, raw in (
        ("article", article.get("license_code")),
        ("figure", figure.get("effective_license")),
        ("panel", panel.get("license_code")),
    ):
        if not raw and source == "figure":
            continue
        allowed = pmc.license_allows(pmc.normalize_license(raw))
        if allowed is None:
            reasons.append(f"{source} license disallows publication")
        elif allowed == "whole_figure" and panel.get("crop_mode") != "whole_figure":
            reasons.append(f"{source} license permits whole figures only")
    return reasons


def _source_reasons(figure: dict, article: dict) -> list[str]:
    """Third-party figures, retracted sources and notices."""
    reasons = []
    triage = db.from_json(figure.get("triage_json"), {}) or {}
    permissions_third_party, _ = jats._third_party(figure.get("fig_permissions_text"), None, [])
    if triage.get("third_party") or triage.get("route") == "third_party" or permissions_third_party:
        reasons.append("third-party figure")
    metadata = article.get("source_metadata") or {}
    if metadata.get("retraction_status") == "retracted":
        reasons.append("source article is retracted")
    types = db.from_json(article.get("publication_types_json"), []) or metadata.get("publication_types") or []
    if isinstance(types, str):
        types = [types]
    if not passes_type_filter(types, article.get("article_type")):
        reasons.append("source is a notice, not an article")
    return reasons


def _retained_reason(panel: dict, figure: dict) -> str | None:
    """The stored judgment's own exclusion of this panel (or plate), if any."""
    vision = db.from_json(figure.get("vision_json"), {}) or {}
    retained = vision.get("plate") if panel.get("plate_kind") else next(
        (row for row in vision.get("panels", []) if row.get("panel_label") == panel.get("panel_label")),
        None,
    )
    if retained and retained.get("include") is False:
        return "retained judgment: " + str(retained.get("exclusion_reason") or "model excluded panel").replace("_", " ")
    return None


def _valid_size(panel: dict) -> bool:
    try:
        return int(panel.get("width") or 0) > 0 and int(panel.get("height") or 0) > 0
    except (TypeError, ValueError):
        return False


def panel_eligibility(
    panel: dict, figure: dict, article: dict, approved_findings: set[str],
) -> dict:
    """Evaluate retained evidence without changing judgments or audit records."""
    panel, figure, article = dict(panel), dict(figure), dict(article)
    supported = _source_findings(panel, figure, approved_findings)
    candidates = [
        "current-hash audit exclusion" if panel.get("_audit_excluded") else None,
        *_license_reasons(panel, figure, article),
        *_source_reasons(figure, article),
        _retained_reason(panel, figure),
        curation.exclusion_reason(panel, figure, article, allowed_findings=approved_findings),
        None if supported else NO_SUPPORTED_LABEL,
        None if _valid_size(panel) else "invalid stored image dimensions",
        # Pages draw the panel from the figure's public PMC S3 image.
        None if figure.get("image_url") else "missing figure image_url",
    ]
    reasons = list(dict.fromkeys(r for r in candidates if r))
    eligible = not reasons
    return {
        "eligible": eligible,
        # Passes every gate except naming its finding: publishable as an
        # unlabeled disease image (``gallery.library_panel_ids``), never as
        # finding evidence.
        "unlabeled_publishable": reasons == [NO_SUPPORTED_LABEL],
        "reasons": reasons,
        "rejection_categories": sorted({rejection_category(r) for r in reasons}),
        "supported_finding_keys": sorted(supported) if eligible and panel.get("plate_kind") != "combined" else [],
    }


def panel_records(conn, disease_key: str | None = None) -> list[dict]:
    """Return eligible and failed rows together for non-publication diagnostics."""
    vocab = [dict(row) for row in conn.execute("SELECT * FROM findings_vocab WHERE approved=1")]
    terms = finding_terms(vocab)
    categories = {row["finding_key"]: row["category"] for row in vocab}
    approved = curation.approved_findings_by_disease(conn)
    exclusions = {
        row["panel_id"]: row["image_sha256"]
        for row in conn.execute("SELECT panel_id,image_sha256 FROM panel_curation WHERE decision='exclude'")
    }
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    identities = {}
    if "panel_identity_reviews" in tables:
        identities = {row["panel_id"]: dict(row) for row in conn.execute("SELECT * FROM panel_identity_reviews")}
    # One topic's call reads only that topic's figures and articles.
    scope, params = "", ()
    if disease_key is not None:
        scope, params = " WHERE pmcid IN (SELECT pmcid FROM panels WHERE disease_key=?)", (disease_key,)
    figures = {row["figure_id"]: dict(row) for row in conn.execute("SELECT * FROM figures" + scope, params)}
    articles = {row["pmcid"]: dict(row) for row in conn.execute("SELECT * FROM articles" + scope, params)}
    if "article_source_metadata" in tables:
        for row in conn.execute("SELECT pmcid,metadata_json FROM article_source_metadata" + scope, params):
            metadata = db.from_json(row["metadata_json"], {}) or {}
            if row["pmcid"] in articles:
                articles[row["pmcid"]]["source_metadata"] = metadata
                articles[row["pmcid"]]["article_type"] = metadata.get("article_type")
    sql = "SELECT * FROM panels" + (" WHERE disease_key=?" if disease_key is not None else "")
    rows = conn.execute(sql, (disease_key,) if disease_key is not None else ())
    out = []
    for row in rows:
        panel = dict(row)
        panel["_audit_excluded"] = (
            panel["panel_id"] in exclusions
            and exclusions[panel["panel_id"]] == (panel.get("sha256") or "")
        )
        figure = dict(figures.get(panel["figure_id"], {}))
        figure["_finding_terms"], figure["_finding_categories"] = terms, categories
        article = articles.get(panel["pmcid"], {})
        result = panel_eligibility(panel, figure, article, approved.get(panel.get("disease_key"), set()))
        panel.update(result)
        panel["eligibility"] = result
        panel["source_age"] = demographics.resolve_age(figure)
        panel["article_tier"] = source_quality.article_tier(
            db.from_json(article.get("publication_types_json"), []), article.get("title"),
            (article.get("source_metadata") or {}).get("abstract"),
        )
        identity = identities.get(panel["panel_id"])
        if (
            identity and panel.get("sha256")
            and identity["reviewed_image_sha256"] == panel["sha256"]
            and identity["source_quote"].strip() and identity["review_provenance"].strip()
        ):
            panel["identity_review"] = identity
        out.append(panel)
    return out


def eligible_panels(conn, disease_key: str | None = None) -> list[dict]:
    return [panel for panel in panel_records(conn, disease_key) if panel["eligible"]]
