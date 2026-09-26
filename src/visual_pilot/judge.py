"""Stage 5: vision judge with prompt P3.

Input figures are those the caption stage kept or left uncertain. DeepInfra
cannot fetch the S3 image URLs (stage 0), so each figure's bytes are fetched
into memory, normalized via ``pmc.prepare_for_llm`` and sent as a base64
data URL. The cache identity is the sha256 of the *original* fetched bytes,
so a rerun with unchanged inputs hits the cache and makes zero LLM calls.

P3's output is post-validated (unknown finding keys -> proposed_findings,
non-pilot disease -> excluded, bbox clamped/swapped) and stored in
``figures.vision_json``; the figure becomes ``vision_accepted`` when at
least one panel is included, else ``vision_rejected``. Fetch/LLM failures
become ``vision_error`` (attempts-bounded retries on later runs).

Scheduling (plan §5 W6): a fetch pool of ``VP_FETCH_CONCURRENCY`` workers
prepares images in ``rank_figures`` priority order and streams them into
``LLMClient.iter_many`` (≤ ``VP_JUDGE_CONCURRENCY`` in flight); results are
applied on the main thread in completion order. Accepted figures' original
bytes are parked in ``originals`` (contract C5) for the store stage.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from PIL import Image

from . import config, db, diseases, llm, originals, pmc
from .prompts import P3

MAX_ATTEMPTS = 3
PILOT_KEYS = set(diseases.DISEASE_KEYS)

# P3 enum per disease (§6). Matching is case-insensitive with
# hyphen/space->underscore folding.
SUBTYPES = {
    "sle": {"acle", "scle", "dle", "lupus_nephritis", "npsle", "other_systemic"},
    "dm": {"classic", "cadm", "jdm", "anti_mda5", "cancer_associated"},
    "as": {"r_axspa", "nr_axspa"},
}

# Lightweight, deterministic pre-vision ranking. This changes call order only:
# it does not reject candidates. The caption model has already applied the
# existing third-party/license gate, and this stage checks the figure-level
# license again before giving an item a positive license score.
_MODALITY_TERMS = {
    "clinical_photo": ("clinical photograph", "clinical photo", "photograph", "skin lesion", "rash", "papule", "ulcer", "gottron", "heliotrope"),
    "mri": ("mri", "magnetic resonance", "stir", "t1-weighted", "t2-weighted"),
    "radiograph": ("radiograph", "x-ray", "x ray", "plain film"),
    "ct": ("computed tomography", " ct ", "ldct", "dect"),
    "ultrasound": ("ultrasound", "sonogram", "doppler"),
    "histology": ("histology", "histological", "biopsy", "stain", "h&e", "hematoxylin", "immunofluorescence"),
}
_VISUAL_CONTEXT = ("patient", "image", "photograph", "photo", "mri", "radiograph", "histolog", "biopsy", "ultrasound", "computed tomography", "ct scan")
_DIAGRAM_CUES = ("diagram", "schematic", "flowchart", "algorithm", "mechanism", "pathway", "prisma flow")
_ALTERNATIVE_DIAGNOSIS_CUES = ("degenerative", "osteitis condensans", "fracture", "brucellosis", "healthy control", "normal control")


def _contains(text: str, term: str) -> bool:
    """Phrase-aware case-insensitive match, with punctuation as boundaries."""
    term = term.strip().lower()
    if not term:
        return False
    if term.startswith(" ") or term.endswith(" "):
        return term in f" {text.lower()} "
    return re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text.lower()) is not None


@lru_cache(maxsize=8)
def _disease_terms(disease_key: str) -> tuple[str, ...]:
    item = diseases.load_diseases().get(disease_key, {})
    terms = [item.get("name", ""), *item.get("synonyms", [])]
    # Common short forms often occur with punctuation/qualifiers.
    terms += {"sle": ["lupus", "sle"], "dm": ["dermatomyositis", "jdm", "cadm", "anti-mda5"],
              "as": ["axspa", "ankylosing spondylitis", "sacroiliitis"]}.get(disease_key, [])
    return tuple(dict.fromkeys(t for t in terms if t))


def _load_priority_context(conn) -> dict:
    vocabulary = [dict(row) for row in conn.execute(
        "SELECT finding_key,label,synonyms_json,disease_keys_json FROM findings_vocab WHERE approved=1"
    )]
    coverage: dict[str, set[str]] = {}
    for row in conn.execute("SELECT disease_key,findings_json FROM panels"):
        coverage.setdefault(row["disease_key"], set()).update(
            f["finding_key"] for f in db.from_json(row["findings_json"], []) or []
            if isinstance(f, dict) and f.get("finding_key")
        )
    return {"vocabulary": vocabulary, "coverage": coverage}


def _figure_findings(conn, disease_key: str, text: str, vocabulary=None) -> set[str]:
    found = set()
    rows = vocabulary if vocabulary is not None else conn.execute(
        "SELECT finding_key,label,synonyms_json,disease_keys_json FROM findings_vocab WHERE approved=1"
    )
    for row in rows:
        keys = db.from_json(row["disease_keys_json"], [])
        if keys and disease_key not in keys:
            continue
        terms = [row["label"], row["finding_key"].replace("_", " "), *db.from_json(row["synonyms_json"], [])]
        # Caption prose frequently expands site abbreviations and inserts
        # anatomical words (e.g. "sacroiliac joint erosions" vs "SI erosions").
        key = row["finding_key"]
        if key.startswith("si_"):
            stem = key[3:].replace("_", " ")
            terms.extend((f"sacroiliac {stem}", f"sacroiliac joint {stem}"))
        if any(_contains(text, t) for t in terms):
            found.add(row["finding_key"])
    return found


def figure_priority(conn, figure: dict, article: dict, context: dict | None = None) -> tuple[float, dict]:
    """Return a metadata-based call-order score and its auditable components.

    This is intentionally a soft ranking signal. A low or uncertain score
    never removes a figure from the vision queue.
    """
    caption = figure.get("caption") or ""
    mentions = db.from_json(figure.get("in_text_mentions_json"), []) or []
    mention_text = " ".join(str(x) for x in mentions)
    title = article.get("title") or ""
    combined = f"{caption} {mention_text} {title}"
    keys = db.from_json(article.get("primary_disease_keys_json"), []) or []
    status = figure.get("status")
    context = context or _load_priority_context(conn)
    license_code = figure.get("effective_license") or article.get("license_code")
    license_mode = pmc.license_allows(license_code) if license_code else None

    status_score = 1.0 if status == "caption_kept" else 0.0
    caption_disease_hits = [key for key in keys if any(_contains(caption, term) for term in _disease_terms(key))]
    mention_disease_hits = [key for key in keys if any(_contains(mention_text, term) for term in _disease_terms(key))]
    title_disease_hits = [key for key in keys if any(_contains(title, term) for term in _disease_terms(key))]
    disease_hits = sorted(set(caption_disease_hits + mention_disease_hits + title_disease_hits))
    disease_score = (2.0 if caption_disease_hits else 0.0) + (0.75 if mention_disease_hits else 0.0) + (0.35 if title_disease_hits else 0.0)
    if not disease_score and keys:
        disease_score = 0.4  # parent article disease is weak context only
    alternative_penalty = -1.25 if any(_contains(caption, term) for term in _ALTERNATIVE_DIAGNOSIS_CUES) else 0.0
    modality_hits = [mode for mode, terms in _MODALITY_TERMS.items()
                     if any(_contains(f" {combined} ", term) for term in terms)]
    visual_context = any(_contains(combined, term) for term in _VISUAL_CONTEXT)
    modality_score = min(2.0, 0.8 * len(modality_hits))
    # In-text figure-specific passage support gives a modest boost; broad
    # article prose alone is weak evidence and cannot outweigh captions.
    mention_score = min(1.5, 0.35 * sum(_contains(mention_text, term) for term in _VISUAL_CONTEXT))
    license_score = 1.0 if license_mode is not None else -8.0

    coverage_bonus = 0.0
    matched_findings: set[str] = set()
    matched_finding_pairs: set[tuple[str, str]] = set()
    if keys:
        # Keep finding coverage disease-specific: a finding stored for SLE
        # must not make the same key look covered for DM on a mixed article.
        for key in keys:
            found_for_disease = _figure_findings(conn, key, combined, context["vocabulary"])
            matched_findings |= found_for_disease
            matched_finding_pairs |= {(key, finding) for finding in found_for_disease}
            known = context["coverage"].get(key, set())
            coverage_bonus += min(1.0, 0.7 * len(found_for_disease - known))
    coverage_bonus = min(2.0, coverage_bonus)
    finding_score = min(1.8, 0.9 * len(matched_finding_pairs))
    # Mechanism-only diagrams receive no visual-context boost; remain eligible
    # because some broad reviews contain a relevant figure despite generic text.
    visual_score = 0.6 if visual_context else 0.0
    diagram_penalty = -1.5 if any(_contains(caption, term) for term in _DIAGRAM_CUES) else 0.0
    total = status_score + disease_score + alternative_penalty + modality_score + mention_score + license_score + finding_score + coverage_bonus + visual_score + diagram_penalty
    return total, {
        "status": status_score, "disease": disease_score, "disease_hits": disease_hits,
        "alternative_diagnosis": alternative_penalty,
        "modality": modality_score, "modality_hits": modality_hits,
        "in_text": mention_score, "license": license_score,
        "coverage": coverage_bonus, "finding_match": finding_score,
        "diagram_penalty": diagram_penalty, "matched_findings": sorted(matched_findings),
        "matched_finding_disease_pairs": sorted([list(pair) for pair in matched_finding_pairs]),
        "visual_context": visual_context,
    }


def rank_figures(conn, figures: list[dict], articles: dict[str, dict]) -> list[dict]:
    """Stable priority order; uncertain figures stay in the returned queue."""
    context = _load_priority_context(conn)
    ranked = []
    for fig in figures:
        score, components = figure_priority(conn, fig, articles[fig["pmcid"]], context)
        ranked.append((score, fig["status"] == "caption_kept", fig["figure_id"], fig, components))
    ranked.sort(key=lambda x: (-x[0], -int(x[1]), x[2]))
    for score, _, _, fig, components in ranked:
        fig["_priority_score"] = score
        fig["_priority_components"] = components
    return [item[3] for item in ranked]


def normalize_subtype(disease_key: str | None, subtype) -> tuple[str | None, str | None]:
    """Map a raw model subtype to the P3 enum for the panel's disease.

    Returns (canonical_subtype_or_None, dropped_raw_or_None): the raw value is
    only non-None when it could not be mapped and must be preserved in the
    rationale."""
    if subtype is None or str(subtype).strip() == "":
        return None, None
    raw = str(subtype).strip()
    norm = re.sub(r"[-\s]+", "_", raw.lower())
    if norm in SUBTYPES.get(disease_key or "", set()):
        return norm, None
    return None, raw


# ---------------------------------------------------------------------------
# Vocabulary + user content
# ---------------------------------------------------------------------------
def vocab_for_diseases(conn, disease_keys: list[str]) -> list[dict]:
    """Approved vocab items relevant to the article's diseases."""
    rows = conn.execute("SELECT * FROM findings_vocab WHERE approved = 1")
    wanted = set(disease_keys)
    items = []
    for row in rows:
        keys = db.from_json(row["disease_keys_json"], [])
        if not wanted or wanted & set(keys):
            items.append(
                {
                    "finding_key": row["finding_key"],
                    "label": row["label"],
                    "synonyms": db.from_json(row["synonyms_json"], []),
                    "category": row["category"],
                }
            )
    return items


def user_content(figure: dict, article: dict, vocabulary: list[dict]) -> str:
    return json.dumps(
        {
            "figure_id": figure["figure_id"],
            "label": figure["label"] or "",
            "caption": figure["caption"] or "",
            "in_text_mentions": db.from_json(figure["in_text_mentions_json"], []),
            "article_title": article["title"] or "",
            "primary_disease_keys": db.from_json(article["primary_disease_keys_json"], []),
            "vocabulary": vocabulary,
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Image fetch + preparation
# ---------------------------------------------------------------------------
def fetch_and_prepare(figure: dict) -> tuple[str, bytes, bytes, str | None]:
    """(original_bytes, mime, prepared_bytes, format_note) — memory only."""
    ref = pmc.ImageRef(url=figure["image_url"], needs_bytes=False, format=figure["image_format"])
    original = pmc.fetch_image_bytes(ref)
    with Image.open(io.BytesIO(original)) as im:
        orig_fmt = (im.format or "").lower() or figure["image_format"]
    mime, prepared = pmc.prepare_for_llm(original)
    prep_fmt = mime.rsplit("/", 1)[-1]
    note = f"{orig_fmt}->{prep_fmt}" if orig_fmt and orig_fmt != prep_fmt else (orig_fmt or prep_fmt)
    return original, mime, prepared, note


# ---------------------------------------------------------------------------
# Post-validation of the P3 response (spec §5 stage 5)
# ---------------------------------------------------------------------------
def post_validate(result: dict, valid_keys: set[str]) -> dict:
    out = dict(result)
    panels = []
    for panel in result.get("panels") or []:
        panel = dict(panel)
        # bbox: clamp to [0,1]; swap inverted corners.
        bbox = list(panel.get("bbox") or [])
        if len(bbox) == 4:
            bbox = [min(1.0, max(0.0, float(v))) for v in bbox]
            x0, y0, x1, y1 = bbox
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0
            panel["bbox"] = [x0, y0, x1, y1]
        # Unknown finding keys -> proposed_findings.
        findings = []
        for finding in panel.get("findings") or []:
            if finding.get("finding_key") in valid_keys:
                findings.append(finding)
            else:
                panel.setdefault("proposed_findings", []).append(
                    finding.get("finding_key")
                )
        panel["findings"] = findings
        # Non-pilot / missing disease on an included panel -> excluded.
        if panel.get("include") and panel.get("disease_key") not in PILOT_KEYS:
            panel["include"] = False
            panel["disease_key"] = None
            panel["exclusion_reason"] = "other_disease"
        # Subtype -> P3 enum; unmappable raw values are preserved in the
        # rationale and the field set to null.
        canon, dropped = normalize_subtype(panel.get("disease_key"), panel.get("subtype"))
        if dropped is not None:
            panel["subtype"] = None
            note = f"[raw subtype: {dropped}]"
            panel["rationale"] = ((panel.get("rationale") or "") + " " + note).strip()
        else:
            panel["subtype"] = canon
        panels.append(panel)
    out["panels"] = panels
    return out


def figure_status(result: dict) -> str:
    if any(p.get("include") for p in result.get("panels") or []):
        return "vision_accepted"
    return "vision_rejected"


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(args) -> int:
    conn = db.init_db()
    disease = None if args.disease == "all" else args.disease
    rows = db.rows_with_status(
        conn,
        "figures",
        ["caption_kept", "caption_uncertain", "vision_error"],
        disease=disease,
    )
    if args.pmcids:
        wanted = set(args.pmcids)
        rows = [r for r in rows if r["pmcid"] in wanted]
    articles = {
        r["pmcid"]: dict(r)
        for r in conn.execute(
            "SELECT pmcid, title, primary_disease_keys_json, license_code FROM articles"
        )
    }
    figures = []
    for r in rows:
        fig = dict(r)
        if fig["status"] == "vision_error" and (fig["attempts"] or 0) >= MAX_ATTEMPTS:
            continue
        article = articles.get(fig["pmcid"], {})
        if not fig.get("effective_license") and article.get("license_code"):
            # Older/imported rows may have only the article-level license.
            # Parse normally copies this onto each figure; persist the same
            # inherited value so the later storage stage sees the same gate.
            fig["effective_license"] = article["license_code"]
            conn.execute(
                "UPDATE figures SET effective_license=? WHERE figure_id=?",
                (fig["effective_license"], fig["figure_id"]),
            )
        # Re-assert the figure-level commercial license gate at the call
        # boundary. Parse normally routes disallowed licenses out of this
        # queue; this protects resumed/imported rows as well.
        if pmc.license_allows(fig.get("effective_license")) is None:
            continue
        figures.append(fig)
    conn.commit()
    figures = rank_figures(conn, figures, articles)
    if args.limit:
        figures = figures[: args.limit]

    if args.dry_run:
        print(f"judge: {len(figures)} figures in scope (metadata-prioritized)")
        conn.close()
        return 0

    vocab_cache: dict[tuple, list[dict]] = {}

    def vocab_for(fig) -> list[dict]:
        keys = tuple(
            sorted(db.from_json(articles[fig["pmcid"]]["primary_disease_keys_json"], []))
        )
        if keys not in vocab_cache:
            vocab_cache[keys] = vocab_for_diseases(conn, list(keys))
        return vocab_cache[keys]

    # Let the figure-level retry queue handle transient vision timeouts. A
    # single request should not occupy a worker for multiple full timeouts.
    client = llm.LLMClient(
        db_conn=conn, budget_usd=args.budget_usd, max_retries=0,
        concurrency=config.VP_JUDGE_CONCURRENCY,
        timeout_seconds=config.VP_JUDGE_TIMEOUT_SECONDS,
    )
    totals = {"accepted": 0, "rejected": 0, "errors": 0}
    reasons: Counter = Counter()
    budget_hit = False

    # Streaming pipeline (plan §5 W6): a fetch pool prepares images just
    # ahead of the judge and client.iter_many pulls a new request only when
    # an LLM slot frees, so completed-but-unjudged images stay bounded to
    # ~2x the in-flight cap and the whole figure set's bytes are never held
    # in memory at once. Everything below runs on this thread: figures are
    # dispatched in rank_figures priority order, results are applied and
    # committed in completion order (call order is not an input; §1).
    fetch_window = max(1, 2 * config.VP_JUDGE_CONCURRENCY)
    submitted: list[dict] = []  # BatchResult.index -> figure

    def _requests():
        """Yield call_json kwargs in priority order, fetching just ahead."""
        pending: deque = deque()  # (figure, fetch future), priority order
        fig_iter = iter(figures)
        pool = ThreadPoolExecutor(max_workers=config.VP_FETCH_CONCURRENCY)
        try:
            while True:
                while len(pending) < fetch_window:
                    fig = next(fig_iter, None)
                    if fig is None:
                        break
                    if not fig["image_url"]:
                        db.set_status(
                            conn, "figures", fig["figure_id"], "vision_error",
                            error="needs_bytes",
                        )
                        conn.commit()
                        continue
                    pending.append((fig, pool.submit(fetch_and_prepare, fig)))
                if not pending:
                    return
                fig, fut = pending.popleft()
                try:
                    original, mime, prepared, note = fut.result()
                except Exception as exc:  # noqa: BLE001 - per-figure isolation
                    db.set_status(
                        conn, "figures", fig["figure_id"], "vision_error",
                        error=f"fetch: {exc}"[:500],
                        attempts=(fig["attempts"] or 0) + 1,
                    )
                    conn.commit()
                    totals["errors"] += 1
                    continue
                article = articles[fig["pmcid"]]
                vocab = vocab_for(fig)
                fig["_valid_keys"] = {v["finding_key"] for v in vocab}
                fig["_original"] = original
                fig["_format_note"] = note
                submitted.append(fig)
                yield {
                    "stage": "p3",
                    "model": config.VP_JUDGE_MODEL,
                    "system": P3.system,
                    "user_content": user_content(fig, article, vocab),
                    "schema": P3.schema,
                    "prompt_version": P3.version,
                    "images": [
                        llm.ImageInput(
                            data_url=pmc.to_data_url(mime, prepared),
                            sha256=hashlib.sha256(original).hexdigest(),
                        )
                    ],
                }
        finally:
            # Early stop (budget/exception): cancel not-yet-started fetches;
            # running ones finish in the background and are discarded.
            pool.shutdown(wait=False, cancel_futures=True)

    requests = _requests()
    results = client.iter_many(requests, max_in_flight=config.VP_JUDGE_CONCURRENCY)
    try:
        for res in results:
            fig = submitted[res.index]
            if res.error is not None:
                if isinstance(res.error, llm.BudgetExceeded):
                    budget_hit = True  # leave status unchanged, stop cleanly
                    break
                db.set_status(
                    conn, "figures", fig["figure_id"], "vision_error",
                    error=f"llm: {res.error}"[:500],
                    attempts=(fig["attempts"] or 0) + 1,
                )
                totals["errors"] += 1
                conn.commit()
                fig.pop("_original", None)
                fig.pop("_format_note", None)
                continue
            parsed = post_validate(res.parsed or {}, fig["_valid_keys"])
            status = figure_status(parsed)
            sha256 = hashlib.sha256(fig["_original"]).hexdigest()
            db.set_status(
                conn,
                "figures",
                fig["figure_id"],
                status,
                vision_json=db.to_json(parsed),
                image_format=fig["_format_note"],
                sha256=sha256,
                error=None,
            )
            if status == "vision_accepted":
                # C5: hand the original bytes to the store stage. Bytes for
                # every other outcome are dropped immediately below.
                originals.put(fig["figure_id"], sha256, fig["_original"])
                totals["accepted"] += 1
            else:
                totals["rejected"] += 1
                for panel in parsed.get("panels") or []:
                    if not panel.get("include"):
                        reasons[panel.get("exclusion_reason") or "not_relevant"] += 1
            conn.commit()
            fig.pop("_original", None)
            fig.pop("_format_note", None)
    finally:
        requests.close()
        results.close()

    if budget_hit:
        print(f"LLM budget exhausted (${client.spent_usd:.4f}); stopping cleanly. Rerun to resume.")

    summary = (
        f"judge: {totals['accepted']} accepted, {totals['rejected']} rejected, "
        f"{totals['errors']} errors, spend=${client.spent_usd:.4f}"
    )
    if reasons:
        summary += " | rejections: " + ", ".join(
            f"{k}={v}" for k, v in reasons.most_common()
        )
    print(summary)
    conn.close()
    return 0
