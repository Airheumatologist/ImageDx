"""Stage 6: crop panels, write image files, insert panel rows.

For each ``vision_accepted`` figure the original bytes are refetched into
memory; the original is written to ``figures/{pmcid}/{basename}`` (TIFF
converted to PNG on write) and every included panel is cropped from its
normalized bbox (2% padding of the original size, clamped) to
``panels/{disease}/{modality}/{panel_id}.png`` with a 400px WebP thumb in
``thumbs/``. ``whole_figure`` crop mode applies to ND licenses, missing or
tiny (<3%) bboxes and >30% overlaps between included panels.

Exact dedup: panels whose saved PNG sha256 already exists reuse the earlier
file but keep their own row, attribution and license. Proposed findings
upsert into findings_vocab exactly once per figure (the vision_accepted ->
stored transition is the only place they are counted).
"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path, PurePosixPath

from PIL import Image

from . import config, db, diseases, jats, pmc

PILOT_KEYS = set(diseases.DISEASE_KEYS)
PAD_FRAC = 0.02
MIN_BBOX_AREA = 0.03
OVERLAP_MAX = 0.30
THUMB_EDGE = 400

_LICENSE_NAMES = {
    "cc0": "CC0",
    "cc-by": "CC BY",
    "cc-by-sa": "CC BY-SA",
    "cc-by-nd": "CC BY-ND",
    "cc-by-nc": "CC BY-NC",
    "cc-by-nc-sa": "CC BY-NC-SA",
    "cc-by-nc-nd": "CC BY-NC-ND",
}

_MODALITY_CATEGORY = {
    "clinical_photo": "skin",
    "dermoscopy": "skin",
    "capillaroscopy": "capillaroscopy",
    "histology_he": "histology",
    "histology_ihc": "histology",
    "immunofluorescence": "histology",
    "radiograph": "radiology_xray",
    "ct": "ct",
    "mri": "mri",
    "ultrasound": "us",
    "echo": "echo",
    "ophthalmic": "eye",
}


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------
def sanitize_id(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", str(text))).strip("_")


def panel_id_for(pmcid: str, fig_xml_id: str, panel_label: str) -> str:
    return sanitize_id(f"{pmcid}_{fig_xml_id}_{panel_label}")


def bbox_to_pixels(
    bbox: list[float], width: int, height: int, pad: float = PAD_FRAC
) -> tuple[int, int, int, int]:
    """Normalized bbox -> padded, clamped pixel box on the original image."""
    x0, y0, x1, y1 = bbox
    pad_x, pad_y = pad * width, pad * height
    return (
        max(0, int(round(x0 * width - pad_x))),
        max(0, int(round(y0 * height - pad_y))),
        min(width, int(round(x1 * width + pad_x))),
        min(height, int(round(y1 * height + pad_y))),
    )


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def overlap_ratio(a: list[float], b: list[float]) -> float:
    """intersection / smaller box area (0 when either box is empty)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    smaller = min(bbox_area(a), bbox_area(b))
    return inter / smaller if smaller > 0 else 0.0


def decide_crop_modes(panels: list[dict], license_mode: str | None) -> list[str]:
    """§5 stage 6 crop mode per panel ("panel" | "whole_figure").

    ``panels`` is every panel of the figure (included or not) — the overlap
    rule compares against all of them. ``license_mode`` is the license's
    mode from ``pmc.license_allows`` ("whole_figure" for ND licenses).
    """
    modes = []
    for panel in panels:
        mode = "panel"
        if license_mode == "whole_figure":
            mode = "whole_figure"
        else:
            bbox = panel.get("bbox")
            if not bbox or len(bbox) != 4 or bbox_area(bbox) < MIN_BBOX_AREA:
                mode = "whole_figure"
            else:
                for other in panels:
                    if other is panel:
                        continue
                    obox = other.get("bbox")
                    if (
                        obox
                        and len(obox) == 4
                        and overlap_ratio(bbox, obox) > OVERLAP_MAX
                    ):
                        mode = "whole_figure"
                        break
        modes.append(mode)
    return modes


def license_name(code: str | None) -> str:
    return _LICENSE_NAMES.get(code or "", (code or "license").upper() or "license")


def attribution_text(
    authors: list[str],
    author_count: int,
    title: str,
    journal: str,
    year,
    doi: str | None,
    license_code: str | None,
    license_url: str | None,
    label: str,
    panel_label: str,
) -> str:
    if not authors or author_count <= 0:
        who = ""
    elif author_count == 1:
        who = f"{authors[0]}. "
    elif author_count == 2 and len(authors) >= 2:
        who = f"{authors[0]} and {authors[1]}. "
    else:
        who = f"{authors[0]} et al. "
    fig = re.sub(r"\.$", "", (label or "").strip())
    tail = re.sub(r"(?i)^fig(?:ure)?\.?\s*", "", fig) or fig or "?"
    doi_part = f"doi:{doi}. " if doi else ""
    url_part = f" ({license_url})" if license_url else ""
    return (
        f"{who}{title}. {journal} {year}. {doi_part}"
        f"{license_name(license_code)}{url_part}. Figure {tail}{panel_label or ''}."
    )


def panel_suffix(vision: dict, panel_label: str | None) -> str:
    """Letter appended to the figure citation: omitted only for a
    non-compound figure whose vision_json has exactly one panel."""
    panels = vision.get("panels") or []
    if not vision.get("figure_is_compound") and len(panels) == 1:
        return ""
    return panel_label or ""


def slugify(term: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", term.strip().lower())).strip("_")


def save_thumb(img: Image.Image, path: Path) -> None:
    thumb = img.copy()
    thumb.thumbnail((THUMB_EDGE, THUMB_EDGE))
    thumb.save(path, format="WEBP")


# ---------------------------------------------------------------------------
# Storage internals
# ---------------------------------------------------------------------------
def _original_basename(figure: dict) -> str:
    name = PurePosixPath((figure.get("image_url") or "").split("?")[0]).name
    return name or f"{figure['figure_id'].replace(':', '_')}.png"


def write_original(original_bytes: bytes, figure: dict, data_dir: Path) -> str:
    """Save the original figure bytes; TIFF originals are converted to PNG
    on write (and the saved name gets a .png suffix)."""
    name = _original_basename(figure)
    data = original_bytes
    with Image.open(io.BytesIO(original_bytes)) as im:
        if (im.format or "").upper() in {"TIFF", "TIF"}:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="PNG")
            data = buf.getvalue()
            name = PurePosixPath(name).with_suffix(".png").name
    rel = f"figures/{figure['pmcid']}/{name}"
    path = data_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return rel


def upsert_proposed(conn, term: str, disease_key: str, modality: str | None) -> bool:
    """One proposal upsert. Returns False when the slug is an approved key."""
    slug = slugify(term)
    if not slug:
        return False
    row = conn.execute(
        "SELECT approved, disease_keys_json FROM findings_vocab WHERE finding_key = ?",
        (slug,),
    ).fetchone()
    if row is not None:
        if row["approved"]:
            return False
        keys = set(db.from_json(row["disease_keys_json"], []))
        if disease_key:
            keys.add(disease_key)
        conn.execute(
            "UPDATE findings_vocab SET proposal_count = proposal_count + 1, "
            "proposed_by_llm = 1, disease_keys_json = ? WHERE finding_key = ?",
            (db.to_json(sorted(keys)), slug),
        )
        return True
    conn.execute(
        "INSERT INTO findings_vocab (finding_key, disease_keys_json, label, "
        "synonyms_json, category, approved, proposed_by_llm, proposal_count) "
        "VALUES (?, ?, ?, '[]', ?, 0, 1, 1)",
        (slug, db.to_json([disease_key] if disease_key else []), term, _MODALITY_CATEGORY.get(modality or "", "clinical_msk")),
    )
    return True


def store_figure(conn, figure: dict, article: dict, parsed: jats.ParsedArticle, data_dir: Path) -> dict:
    """Write files + panel rows for one vision_accepted figure."""
    ref = pmc.ImageRef(url=figure["image_url"], format=figure["image_format"])
    original_bytes = pmc.fetch_image_bytes(ref)
    img = Image.open(io.BytesIO(original_bytes)).convert("RGB")
    write_original(original_bytes, figure, data_dir)

    vision = db.from_json(figure["vision_json"], {}) or {}
    all_panels = vision.get("panels") or []
    fig_xml_id = figure["figure_id"].split(":", 1)[1]
    effective_license = figure["effective_license"]
    license_mode = pmc.license_allows(effective_license)
    license_url = (
        article["license_url"]
        if effective_license == article["license_code"]
        else pmc.license_url_for(effective_license, figure["fig_permissions_text"])
    )

    width, height = img.size
    stats = {"panels": 0, "whole_figure": 0, "dedup": 0, "proposed": 0}
    seen_sha: set[str] = set()

    for panel, mode in zip(all_panels, decide_crop_modes(all_panels, license_mode)):
        if not panel.get("include"):
            continue
        if mode == "whole_figure":
            stats["whole_figure"] += 1
            crop = img
        else:
            crop = img.crop(bbox_to_pixels(panel["bbox"], width, height))

        png_io = io.BytesIO()
        crop.save(png_io, format="PNG")
        png_bytes = png_io.getvalue()
        sha = hashlib.sha256(png_bytes).hexdigest()

        pid = panel_id_for(figure["pmcid"], fig_xml_id, panel.get("panel_label") or "A")
        modality = panel.get("modality") or "other"
        rel_img = f"panels/{panel['disease_key']}/{modality}/{pid}.png"
        rel_thumb = f"thumbs/{pid}.webp"

        existing = conn.execute(
            "SELECT image_path, thumb_path FROM panels WHERE sha256 = ?", (sha,)
        ).fetchone()
        if existing is not None or sha in seen_sha:
            if existing is not None:
                rel_img, rel_thumb = existing["image_path"], existing["thumb_path"]
            stats["dedup"] += 1
        else:
            out = data_dir / rel_img
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(png_bytes)
            thumb_path = data_dir / rel_thumb
            thumb_path.parent.mkdir(parents=True, exist_ok=True)
            save_thumb(crop, thumb_path)
            seen_sha.add(sha)

        findings = [
            {"finding_key": f["finding_key"], "evidence": f.get("evidence", "")}
            for f in (panel.get("findings") or [])
            if f.get("finding_key")
        ]
        proposed = set(panel.get("proposed_findings") or [])
        for term in proposed:
            stats["proposed"] += upsert_proposed(
                conn, term, panel.get("disease_key") or "", modality
            )

        attrib = attribution_text(
            parsed.authors,
            parsed.author_count,
            article["title"] or "",
            article["journal"] or parsed.journal_name or "",
            article["year"],
            article["doi"],
            effective_license,
            license_url,
            figure["label"] or fig_xml_id,
            panel_suffix(vision, panel.get("panel_label")),
        )
        conn.execute(
            "INSERT OR REPLACE INTO panels (panel_id, figure_id, pmcid, panel_label, "
            "disease_key, subtype, modality, body_site, findings_json, typicality, "
            "stage, age_group, skin_tone, stated_ethnicity, stated_ethnicity_quote, "
            "study_region, annotations_present, bbox_json, crop_mode, confidence, "
            "rationale, image_path, thumb_path, width, height, sha256, "
            "attribution_text, license_code, license_url, source_url) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                pid,
                figure["figure_id"],
                figure["pmcid"],
                panel.get("panel_label"),
                panel.get("disease_key"),
                panel.get("subtype"),
                modality,
                panel.get("body_site"),
                db.to_json(findings),
                panel.get("typicality"),
                panel.get("stage"),
                panel.get("age_group"),
                panel.get("skin_tone"),
                panel.get("stated_ethnicity"),
                panel.get("stated_ethnicity_quote"),
                article["study_region"],
                int(bool(panel.get("annotations_present"))),
                db.to_json(panel.get("bbox")),
                mode,
                panel.get("confidence"),
                panel.get("rationale"),
                rel_img,
                rel_thumb,
                crop.width,
                crop.height,
                sha,
                attrib,
                effective_license,
                license_url,
                f"https://pmc.ncbi.nlm.nih.gov/articles/{figure['pmcid']}/",
            ),
        )
        stats["panels"] += 1
    return stats


# ---------------------------------------------------------------------------
# Retroactive metadata refresh (attribution format + subtype enum fixes)
# ---------------------------------------------------------------------------
def refresh_panel_metadata(conn) -> dict:
    """Recompute attribution_text and subtype/rationale on existing panel
    rows from the parent figure's stored vision_json and the article bundle
    (re-fetched only for authors; image bytes are untouched)."""
    from . import judge

    stats = {"figures": 0, "panels": 0, "errors": 0}
    figure_ids = [r["figure_id"] for r in conn.execute("SELECT DISTINCT figure_id FROM panels")]
    for figure_id in figure_ids:
        figure = conn.execute(
            "SELECT * FROM figures WHERE figure_id = ?", (figure_id,)
        ).fetchone()
        article = (
            conn.execute("SELECT * FROM articles WHERE pmcid = ?", (figure["pmcid"],)).fetchone()
            if figure
            else None
        )
        if figure is None or article is None:
            stats["errors"] += 1
            continue
        try:
            parsed = jats.parse_article(pmc.get_article_bundle(figure["pmcid"]).xml_text)
        except Exception as exc:  # noqa: BLE001 - per-figure isolation
            print(f"{figure['pmcid']}: refresh bundle error {exc}")
            stats["errors"] += 1
            continue

        vision = db.from_json(figure["vision_json"], {}) or {}
        vision_by_label = {p.get("panel_label"): p for p in vision.get("panels") or []}
        effective_license = figure["effective_license"]
        license_url = (
            article["license_url"]
            if effective_license == article["license_code"]
            else pmc.license_url_for(effective_license, figure["fig_permissions_text"])
        )
        stats["figures"] += 1
        for prow in conn.execute(
            "SELECT * FROM panels WHERE figure_id = ?", (figure_id,)
        ).fetchall():
            vpanel = vision_by_label.get(prow["panel_label"], {})
            subtype, dropped = judge.normalize_subtype(
                prow["disease_key"], vpanel.get("subtype", prow["subtype"])
            )
            rationale = vpanel.get("rationale") or prow["rationale"]
            if dropped is not None and "[raw subtype:" not in (rationale or ""):
                rationale = ((rationale or "") + f" [raw subtype: {dropped}]").strip()
            attrib = attribution_text(
                parsed.authors,
                parsed.author_count,
                article["title"] or "",
                article["journal"] or parsed.journal_name or "",
                article["year"],
                article["doi"],
                effective_license,
                license_url,
                figure["label"] or figure["figure_id"].split(":", 1)[1],
                panel_suffix(vision, prow["panel_label"]),
            )
            conn.execute(
                "UPDATE panels SET attribution_text = ?, subtype = ?, rationale = ?, "
                "updated_at = datetime('now') WHERE panel_id = ?",
                (attrib, subtype, rationale, prow["panel_id"]),
            )
            stats["panels"] += 1
    conn.commit()
    return stats


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(args) -> int:
    conn = db.init_db()
    data_dir = config.data_dir()

    if getattr(args, "refresh", False):
        stats = refresh_panel_metadata(conn)
        print(
            f"refresh: {stats['panels']} panels re-tagged across "
            f"{stats['figures']} figures ({stats['errors']} errors)"
        )
        conn.close()
        return 0
    disease = None if args.disease == "all" else args.disease
    rows = db.rows_with_status(conn, "figures", "vision_accepted", disease=disease)
    if args.pmcids:
        wanted = set(args.pmcids)
        rows = [r for r in rows if r["pmcid"] in wanted]
    figures = [dict(r) for r in rows]
    if args.limit:
        figures = figures[: args.limit]

    if args.dry_run:
        print(f"store: {len(figures)} vision_accepted figures would be stored")
        conn.close()
        return 0

    article_cache: dict[str, dict] = {}
    parsed_cache: dict[str, jats.ParsedArticle] = {}
    totals = {"stored": 0, "panels": 0, "whole_figure": 0, "dedup": 0, "proposed": 0, "errors": 0}
    for figure in figures:
        pmcid = figure["pmcid"]
        try:
            if pmcid not in article_cache:
                article_cache[pmcid] = dict(
                    conn.execute("SELECT * FROM articles WHERE pmcid = ?", (pmcid,)).fetchone()
                )
                parsed_cache[pmcid] = jats.parse_article(
                    pmc.get_article_bundle(pmcid).xml_text
                )
            # One transaction per figure: panel rows, proposed-finding upserts
            # and the status flip commit together or roll back together, so a
            # rerun never double-counts proposals.
            with conn:
                stats = store_figure(
                    conn, figure, article_cache[pmcid], parsed_cache[pmcid], data_dir
                )
                db.set_status(conn, "figures", figure["figure_id"], "stored", error=None)
            for key in ("panels", "whole_figure", "dedup", "proposed"):
                totals[key] += stats[key]
            totals["stored"] += 1
        except Exception as exc:  # noqa: BLE001 - per-figure isolation
            # Keep vision_accepted so the figure is retried on the next run.
            db.set_status(
                conn, "figures", figure["figure_id"], "vision_accepted",
                error=f"store: {exc}"[:500],
            )
            totals["errors"] += 1
            print(f"{figure['figure_id']}: store error {exc}")
            conn.commit()

    print(
        f"store: {totals['stored']} figures stored, {totals['panels']} panels "
        f"({totals['whole_figure']} whole-figure, {totals['dedup']} deduped), "
        f"{totals['proposed']} proposed-finding upserts, {totals['errors']} errors"
    )
    conn.close()
    return 0
