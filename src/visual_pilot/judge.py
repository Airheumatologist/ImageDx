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
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from . import config, db, diseases, llm, pmc
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
    figures = []
    for r in rows:
        fig = dict(r)
        if fig["status"] == "vision_error" and (fig["attempts"] or 0) >= MAX_ATTEMPTS:
            continue
        figures.append(fig)
    # caption_kept first, then figure_id.
    figures.sort(key=lambda f: (f["status"] != "caption_kept", f["figure_id"]))
    if args.limit:
        figures = figures[: args.limit]

    if args.dry_run:
        print(f"judge: {len(figures)} figures in scope")
        conn.close()
        return 0

    articles = {
        r["pmcid"]: dict(r)
        for r in conn.execute(
            "SELECT pmcid, title, primary_disease_keys_json FROM articles"
        )
    }
    vocab_cache: dict[tuple, list[dict]] = {}

    def vocab_for(fig) -> list[dict]:
        keys = tuple(
            sorted(db.from_json(articles[fig["pmcid"]]["primary_disease_keys_json"], []))
        )
        if keys not in vocab_cache:
            vocab_cache[keys] = vocab_for_diseases(conn, list(keys))
        return vocab_cache[keys]

    client = llm.LLMClient(db_conn=conn, budget_usd=args.budget_usd)
    totals = {"accepted": 0, "rejected": 0, "errors": 0}
    reasons: Counter = Counter()
    budget_hit = False
    chunk_size = config.VP_CONCURRENCY * 4

    for chunk_start in range(0, len(figures), chunk_size):
        if budget_hit:
            break
        chunk = figures[chunk_start : chunk_start + chunk_size]

        # Fetch the chunk's image bytes on a thread pool (memory stays bounded
        # to the chunk), then judge them in one call_many.
        ready: list[dict] = []
        with ThreadPoolExecutor(max_workers=config.VP_CONCURRENCY) as pool:
            futures = {}
            for fig in chunk:
                if not fig["image_url"]:
                    db.set_status(
                        conn, "figures", fig["figure_id"], "vision_error",
                        error="needs_bytes",
                    )
                    continue
                futures[pool.submit(fetch_and_prepare, fig)] = fig
            for fut in as_completed(futures):
                fig = futures[fut]
                try:
                    original, mime, prepared, note = fut.result()
                except Exception as exc:  # noqa: BLE001 - per-figure isolation
                    db.set_status(
                        conn, "figures", fig["figure_id"], "vision_error",
                        error=f"fetch: {exc}"[:500],
                        attempts=(fig["attempts"] or 0) + 1,
                    )
                    totals["errors"] += 1
                    continue
                fig["_original"] = original
                fig["_data_url"] = pmc.to_data_url(mime, prepared)
                fig["_format_note"] = note
                ready.append(fig)
        conn.commit()
        if not ready:
            continue

        requests = []
        for fig in ready:
            article = articles[fig["pmcid"]]
            vocab = vocab_for(fig)
            fig["_valid_keys"] = {v["finding_key"] for v in vocab}
            requests.append(
                {
                    "stage": "p3",
                    "model": config.VP_JUDGE_MODEL,
                    "system": P3.system,
                    "user_content": user_content(fig, article, vocab),
                    "schema": P3.schema,
                    "prompt_version": P3.version,
                    "images": [
                        llm.ImageInput(
                            data_url=fig["_data_url"],
                            sha256=hashlib.sha256(fig["_original"]).hexdigest(),
                        )
                    ],
                }
            )

        for fig, res in zip(ready, client.call_many(requests)):
            if res.error is not None:
                if isinstance(res.error, llm.BudgetExceeded):
                    budget_hit = True  # leave status unchanged, stop cleanly
                    continue
                db.set_status(
                    conn, "figures", fig["figure_id"], "vision_error",
                    error=f"llm: {res.error}"[:500],
                    attempts=(fig["attempts"] or 0) + 1,
                )
                totals["errors"] += 1
                continue
            parsed = post_validate(res.parsed or {}, fig["_valid_keys"])
            status = figure_status(parsed)
            db.set_status(
                conn,
                "figures",
                fig["figure_id"],
                status,
                vision_json=db.to_json(parsed),
                image_format=fig["_format_note"],
                sha256=hashlib.sha256(fig["_original"]).hexdigest(),
                error=None,
            )
            if status == "vision_accepted":
                totals["accepted"] += 1
            else:
                totals["rejected"] += 1
                for panel in parsed.get("panels") or []:
                    if not panel.get("include"):
                        reasons[panel.get("exclusion_reason") or "not_relevant"] += 1
        conn.commit()

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
