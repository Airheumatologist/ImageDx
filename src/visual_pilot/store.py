"""Stage 6: crop panels, write image files, insert panel rows.

For each ``vision_accepted`` figure the original bytes come from the
judge->store handoff cache (contract C5, ``originals.take``) or are
refetched into memory; a display copy capped at ``VP_ORIGINAL_MAX_EDGE``
is written to ``figures/{pmcid}/{stem}.webp`` (raw bytes stay refetchable
from the PMC S3 bundle) and every included panel is cropped from its
normalized bbox (2% padding of the original size, clamped), capped at
``VP_PANEL_MAX_EDGE`` and encoded per ``VP_PANEL_FORMAT``/``VP_PANEL_QUALITY``
(default WebP q90) to ``panels/{disease}/{modality}/{panel_id}.{ext}``
with a 400px WebP thumb in ``thumbs/``. ``whole_figure`` crop mode applies
to ND licenses, missing or tiny (<3%) bboxes and >30% overlaps between
included panels, and to ``vision_json["plate"]`` — the v5 whole-figure
record for a publishable single-disease compound figure, stored once with
``panel_label='whole'`` and its findings in ``plate_findings_json``. When
``figures.sha256`` is set the bytes must match it; a
mismatch is a store error (the figure stays ``vision_accepted``) rather
than silently storing different pixels.

Attribution (contract C6) is built from the persisted
``articles.authors_json``/``author_count``/``journal_name``; a hinted JATS
refetch is used only when those fields are missing.

Exact dedup: panels whose saved image sha256 already exists reuse the
earlier file but keep their own row, attribution and license. Proposed findings
upsert into findings_vocab exactly once per figure (the vision_accepted ->
stored transition is the only place they are counted).

Scheduling (plan §5 W7): fetch + decode + crop + PNG/WebP encoding run on a
``VP_FETCH_CONCURRENCY`` worker pool; file writes, the sha256 dedup check,
panel inserts, proposal upserts and the ``vision_accepted -> stored`` flip
are applied on the main thread in deterministic figure order, one
transaction per figure — identical results to the sequential path.
"""

from __future__ import annotations

import hashlib
import io
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from PIL import Image

from . import config, curation, db, diseases, jats, originals, pmc, timing

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
    """Letter appended to the figure citation: omitted for a non-compound
    figure whose vision_json has exactly one panel, and for the
    whole-figure plate (cited as "Figure N.")."""
    if panel_label == curation.PLATE_LABEL:
        return ""
    panels = vision.get("panels") or []
    if not vision.get("figure_is_compound") and len(panels) == 1:
        return ""
    return panel_label or ""


def slugify(term: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", term.strip().lower())).strip("_")


def save_thumb(img: Image.Image, path: Path) -> None:
    thumb = img.copy()
    thumb.thumbnail((THUMB_EDGE, THUMB_EDGE))
    thumb.save(path, format="WEBP", quality=80, method=6)


def _cap_edge(img: Image.Image, max_edge: int) -> Image.Image:
    """Return ``img`` downscaled so the long edge is <= ``max_edge``
    (0 disables). Returns the same object when no resize is needed."""
    if max_edge and max(img.size) > max_edge:
        scale = max_edge / max(img.size)
        return img.resize(
            (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return img


_PANEL_ENCODINGS = {
    "webp": (".webp", "WEBP"),
    "jpeg": (".jpg", "JPEG"),
    "png": (".png", "PNG"),
}


def _encode_panel(img: Image.Image) -> tuple[bytes, str]:
    """(encoded_bytes, file_ext) for one panel per VP_PANEL_FORMAT/QUALITY."""
    try:
        ext, pil_fmt = _PANEL_ENCODINGS[config.VP_PANEL_FORMAT]
    except KeyError:
        raise ValueError(
            f"VP_PANEL_FORMAT must be one of {sorted(_PANEL_ENCODINGS)}"
        ) from None
    buf = io.BytesIO()
    if pil_fmt == "PNG":
        img.save(buf, format="PNG", optimize=True)
    elif pil_fmt == "JPEG":
        img.save(buf, format="JPEG", quality=config.VP_PANEL_QUALITY,
                 optimize=True, progressive=True)
    else:
        img.save(buf, format="WEBP", quality=config.VP_PANEL_QUALITY, method=6)
    return buf.getvalue(), ext


# ---------------------------------------------------------------------------
# Storage internals
# ---------------------------------------------------------------------------
def _original_basename(figure: dict) -> str:
    name = PurePosixPath((figure.get("image_url") or "").split("?")[0]).name
    return name or f"{figure['figure_id'].replace(':', '_')}.webp"


def _encode_original(img: Image.Image, figure: dict) -> tuple[str, bytes]:
    """(rel_path, file_bytes) for the stored figure original: a display copy
    re-encoded as WebP capped at ``VP_ORIGINAL_MAX_EDGE``. The verbatim bytes
    are never written to disk — they stay refetchable from the PMC S3 bundle
    (``figures.sha256`` verifies the pixels on refetch)."""
    buf = io.BytesIO()
    _cap_edge(img, config.VP_ORIGINAL_MAX_EDGE).save(
        buf, format="WEBP", quality=config.VP_PANEL_QUALITY, method=6
    )
    return original_rel_path(figure), buf.getvalue()


def original_rel_path(figure: dict) -> str:
    """Data-dir-relative path of a figure's stored display original."""
    name = PurePosixPath(_original_basename(figure)).with_suffix(".webp").name
    return f"figures/{figure['pmcid']}/{name}"


def write_original(original_bytes: bytes, figure: dict, data_dir: Path) -> str:
    """Save the display copy of a figure original (see ``_encode_original``)."""
    with Image.open(io.BytesIO(original_bytes)) as im:
        rel, data = _encode_original(im.convert("RGB"), figure)
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


def _row_get(row, key):
    """``row[key]`` for dicts and sqlite3.Rows; None when absent."""
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return None


def _bundle_for_row(article_row) -> pmc.ArticleBundle:
    """``pmc.get_article_bundle``, C3-hinted from the row's persisted
    ``s3_prefix``/``media_files_json`` when present (same resolver output as
    the unhinted path)."""
    prefix = _row_get(article_row, "s3_prefix")
    if not prefix:
        return pmc.get_article_bundle(article_row["pmcid"])
    media = db.from_json(_row_get(article_row, "media_files_json"), None)
    if not isinstance(media, list):
        media = None
    return pmc.get_article_bundle(
        article_row["pmcid"], prefix=prefix, media_files=media
    )


def _attribution_source(article_row) -> tuple[list[str], int, str | None]:
    """(authors, author_count, journal_name) for ``attribution_text``.

    Contract C6: persisted ``authors_json``/``author_count``/``journal_name``
    are used first; a hinted JATS refetch happens only when they are missing
    or empty (rows created before C6, or never parsed).
    """
    try:
        authors = db.from_json(_row_get(article_row, "authors_json"), None)
    except ValueError:
        authors = None
    author_count = _row_get(article_row, "author_count")
    journal_name = _row_get(article_row, "journal_name")
    have_authors = (
        isinstance(authors, list)
        and author_count is not None
        and (int(author_count) <= 0 or bool(authors))
    )
    have_journal = bool(journal_name) or bool(_row_get(article_row, "journal"))
    if have_authors and have_journal:
        return [str(a) for a in authors], int(author_count or 0), journal_name
    bundle = _bundle_for_row(article_row)
    parsed = jats.parse_article(bundle.xml_text)
    return parsed.authors, parsed.author_count, parsed.journal_name


def _original_bytes(figure: dict) -> bytes:
    """Judge-handoff bytes (C5) or a refetch, sha256-verified against the row.

    ``originals.take`` consumes the entry either way, so a cache mismatch
    falls back to ``pmc.fetch_image_bytes``; when ``figures.sha256`` is set
    the resolved bytes must match it — a mismatch raises so the figure stays
    ``vision_accepted`` rather than silently storing different pixels.
    """
    expected = figure.get("sha256") or ""
    if expected:
        data = originals.take(figure["figure_id"], expected)
        if data is not None:
            timing.count("store_originals_hit")
            return data
        timing.count("store_originals_miss")
    ref = pmc.ImageRef(url=figure["image_url"], format=figure["image_format"])
    data = pmc.fetch_image_bytes(ref)
    if expected and hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(
            f"sha256 mismatch for {figure['figure_id']}: fetched bytes do not "
            "match figures.sha256; refusing to store different pixels"
        )
    return data


def _prepare_figure(figure: dict, article_row, attrib=None) -> dict:
    """Worker-side: resolve + verify original bytes, decode, crop and encode.

    No DB access and no file writes — returns everything the main thread
    needs to apply the figure (``_apply_prepared``). ``attrib`` may carry a
    pre-resolved (authors, author_count, journal_name) tuple; otherwise it is
    resolved here from the article row (C6 fields, hinted JATS fallback).
    """
    started = time.monotonic()
    try:
        with timing.inflight("store_pool"):
            original_bytes = _original_bytes(figure)
            img = Image.open(io.BytesIO(original_bytes)).convert("RGB")
            original_file = _encode_original(img, figure)

            vision = db.from_json(figure["vision_json"], {}) or {}
            all_panels = vision.get("panels") or []
            fig_xml_id = figure["figure_id"].split(":", 1)[1]
            license_mode = pmc.license_allows(figure["effective_license"])

            width, height = img.size
            panels: list[dict] = []
            curation_exclusions: list[tuple[int, str]] = []
            for panel_index, (panel, mode) in enumerate(zip(
                all_panels, decide_crop_modes(all_panels, license_mode)
            )):
                if not panel.get("include"):
                    continue
                # Recheck the shared eligibility policy at the write boundary.
                # This blocks legacy or imported accepted judgments from
                # becoming newly visible when their caption/source is clearly
                # a collage, chart, normal image, or poor crop.
                reason = curation.exclusion_reason(
                    panel, figure, article_row, image_size=(width, height)
                )
                if reason:
                    curation_exclusions.append((panel_index, reason))
                    continue
                crop = img if mode == "whole_figure" else img.crop(
                    bbox_to_pixels(panel["bbox"], width, height)
                )
                crop = _cap_edge(crop, config.VP_PANEL_MAX_EDGE)
                img_bytes, img_ext = _encode_panel(crop)
                thumb_io = io.BytesIO()
                thumb = crop.copy()
                thumb.thumbnail((THUMB_EDGE, THUMB_EDGE))
                thumb.save(thumb_io, format="WEBP", quality=80, method=6)
                pid = panel_id_for(
                    figure["pmcid"], fig_xml_id, panel.get("panel_label") or "A"
                )
                modality = panel.get("modality") or "other"
                panels.append(
                    {
                        "panel": panel,
                        "mode": mode,
                        "img_bytes": img_bytes,
                        "thumb": thumb_io.getvalue(),
                        "sha": hashlib.sha256(img_bytes).hexdigest(),
                        "width": crop.width,
                        "height": crop.height,
                        "pid": pid,
                        "modality": modality,
                        "rel_img": f"panels/{panel['disease_key']}/{modality}/{pid}{img_ext}",
                        "rel_thumb": f"thumbs/{pid}.webp",
                    }
                )
            plate = vision.get("plate") or {}
            if plate.get("include"):
                # Same write-boundary recheck as panels, then store the whole
                # figure once — capped, encoded and thumbed like a panel.
                reason = curation.exclusion_reason(
                    plate, figure, article_row, image_size=(width, height)
                )
                if reason:
                    curation_exclusions.append(("plate", reason))
                else:
                    crop = _cap_edge(img, config.VP_PANEL_MAX_EDGE)
                    img_bytes, img_ext = _encode_panel(crop)
                    thumb_io = io.BytesIO()
                    thumb = crop.copy()
                    thumb.thumbnail((THUMB_EDGE, THUMB_EDGE))
                    thumb.save(thumb_io, format="WEBP", quality=80, method=6)
                    pid = panel_id_for(
                        figure["pmcid"], fig_xml_id, curation.PLATE_LABEL
                    )
                    modality = plate.get("modality") or "other"
                    panels.append(
                        {
                            "panel": plate,
                            "mode": "whole_figure",
                            "img_bytes": img_bytes,
                            "thumb": thumb_io.getvalue(),
                            "sha": hashlib.sha256(img_bytes).hexdigest(),
                            "width": crop.width,
                            "height": crop.height,
                            "pid": pid,
                            "modality": modality,
                            "rel_img": f"panels/{plate['disease_key']}/{modality}/{pid}{img_ext}",
                            "rel_thumb": f"thumbs/{pid}.webp",
                        }
                    )
            if attrib is None:
                attrib = _attribution_source(article_row)
            return {
                "original_file": original_file,
                "panels": panels,
                "curation_exclusions": curation_exclusions,
                "attrib": attrib,
            }
    finally:
        timing.record("store_prepare", time.monotonic() - started)


def _write_file(data_dir: Path, rel: str, data: bytes) -> None:
    path = data_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _apply_prepared(conn, figure: dict, article, prepared: dict, data_dir: Path) -> dict:
    """Main thread: write files, dedup-check, insert panel rows for one figure.

    Must run under ``with conn:`` so file writes, upserts and the caller's
    status flip commit or roll back together. Dedup sees every panel row
    inserted earlier in the run because callers apply in figure order.
    """
    vision = db.from_json(figure["vision_json"], {}) or {}
    for panel_index, reason in prepared.get("curation_exclusions", []):
        # The report is tied to the exact stored vision panel ordering and is
        # retained with the judgment for diagnosis. The DB audit remains the
        # source of reversible publication exclusions for existing rows.
        if panel_index == "plate":
            plate = vision.get("plate") or {}
            plate["include"] = False
            plate["exclusion_reason"] = reason
            conn.execute(
                "UPDATE figures SET vision_json=? WHERE figure_id=?",
                (db.to_json(vision), figure["figure_id"]),
            )
            continue
        panels = vision.get("panels") or []
        if panel_index < len(panels):
            panel = panels[panel_index]
            panel["include"] = False
            panel["exclusion_reason"] = (
                "collage" if "collage" in reason else
                "diagram" if "diagram" in reason or "chart" in reason else
                "normal_control" if "control" in reason or "normal" in reason else
                "poor_quality" if "crop" in reason or "bounds" in reason else
                "not_patient_image"
            )
            panel["curation_reason"] = reason
        conn.execute(
            "UPDATE figures SET vision_json=? WHERE figure_id=?",
            (db.to_json(vision), figure["figure_id"]),
        )
    fig_xml_id = figure["figure_id"].split(":", 1)[1]
    effective_license = figure["effective_license"]
    license_url = (
        article["license_url"]
        if effective_license == article["license_code"]
        else pmc.license_url_for(effective_license, figure["fig_permissions_text"])
    )
    authors, author_count, journal_name = prepared["attrib"]

    stats = {"panels": 0, "whole_figure": 0, "dedup": 0, "proposed": 0,
             "excluded": len(prepared.get("curation_exclusions", []))}
    seen_sha: set[str] = set()

    rel_orig, orig_data = prepared["original_file"]
    if prepared["panels"]:
        _write_file(data_dir, rel_orig, orig_data)

    for item in prepared["panels"]:
        panel = item["panel"]
        mode = item["mode"]
        if mode == "whole_figure":
            stats["whole_figure"] += 1
        sha = item["sha"]
        rel_img, rel_thumb = item["rel_img"], item["rel_thumb"]

        existing = conn.execute(
            "SELECT image_path, thumb_path FROM panels WHERE sha256 = ?", (sha,)
        ).fetchone()
        if existing is not None or sha in seen_sha:
            if existing is not None:
                rel_img, rel_thumb = existing["image_path"], existing["thumb_path"]
            stats["dedup"] += 1
        else:
            _write_file(data_dir, rel_img, item["img_bytes"])
            _write_file(data_dir, rel_thumb, item["thumb"])
            seen_sha.add(sha)

        findings = [
            {"finding_key": f["finding_key"], "evidence": f.get("evidence", "")}
            for f in (panel.get("findings") or [])
            if f.get("finding_key")
        ]
        proposed = set(panel.get("proposed_findings") or [])
        for term in proposed:
            stats["proposed"] += upsert_proposed(
                conn, term, panel.get("disease_key") or "", item["modality"]
            )

        attrib = attribution_text(
            authors,
            author_count,
            article["title"] or "",
            article["journal"] or journal_name or "",
            article["year"],
            article["doi"],
            effective_license,
            license_url,
            figure["label"] or fig_xml_id,
            panel_suffix(vision, panel.get("panel_label")),
        )
        plate_kind = panel.get("plate_kind")
        conn.execute(
            "INSERT OR REPLACE INTO panels (panel_id, figure_id, pmcid, panel_label, "
            "disease_key, subtype, modality, body_site, findings_json, typicality, "
            "stage, age_group, skin_tone, stated_ethnicity, stated_ethnicity_quote, "
            "study_region, annotations_present, bbox_json, crop_mode, confidence, "
            "rationale, image_path, thumb_path, width, height, sha256, "
            "attribution_text, license_code, license_url, source_url, "
            "plate_kind, plate_findings_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item["pid"],
                figure["figure_id"],
                figure["pmcid"],
                panel.get("panel_label"),
                panel.get("disease_key"),
                panel.get("subtype"),
                item["modality"],
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
                item["width"],
                item["height"],
                sha,
                attrib,
                effective_license,
                license_url,
                f"https://pmc.ncbi.nlm.nih.gov/articles/{figure['pmcid']}/",
                plate_kind,
                db.to_json(panel.get("plate_findings")) if plate_kind else None,
            ),
        )
        stats["panels"] += 1
    return stats


def store_figure(conn, figure: dict, article: dict, parsed: jats.ParsedArticle, data_dir: Path) -> dict:
    """Write files + panel rows for one vision_accepted figure.

    Sequential single-figure path (prepare + apply on the calling thread);
    ``run()`` uses the same internals across a worker pool.
    """
    prepared = _prepare_figure(
        figure,
        article,
        attrib=(parsed.authors, parsed.author_count, parsed.journal_name),
    )
    return _apply_prepared(conn, figure, article, prepared, data_dir)


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
            parsed = jats.parse_article(_bundle_for_row(article).xml_text)
        except Exception as exc:  # noqa: BLE001 - per-figure isolation
            print(f"{figure['pmcid']}: refresh bundle error {exc}")
            stats["errors"] += 1
            continue

        vision = db.from_json(figure["vision_json"], {}) or {}
        vision_by_label = {p.get("panel_label"): p for p in vision.get("panels") or []}
        plate = vision.get("plate")
        if isinstance(plate, dict):
            vision_by_label[curation.PLATE_LABEL] = plate
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

    article_rows: dict[str, dict | None] = {}
    for figure in figures:
        pmcid = figure["pmcid"]
        if pmcid not in article_rows:
            row = conn.execute(
                "SELECT * FROM articles WHERE pmcid = ?", (pmcid,)
            ).fetchone()
            article_rows[pmcid] = dict(row) if row is not None else None

    totals = {"stored": 0, "panels": 0, "whole_figure": 0, "dedup": 0, "proposed": 0, "errors": 0}

    def _store_error(fig: dict, exc: Exception) -> None:
        # Keep vision_accepted so the figure is retried on the next run; any
        # cached original for it is dropped (take() already consumed it).
        originals.discard(fig["figure_id"])
        db.set_status(
            conn, "figures", fig["figure_id"], "vision_accepted",
            error=f"store: {exc}"[:500],
        )
        totals["errors"] += 1
        print(f"{fig['figure_id']}: store error {exc}")
        conn.commit()

    # W7: fetch + decode + crop + WebP encode run on a VP_FETCH_CONCURRENCY pool
    # with bounded lookahead (2x workers); results are applied on this thread
    # strictly in figure order — one transaction per figure — so the sha256
    # dedup check sees panels inserted earlier in the same run and the whole
    # stage is byte-identical to the sequential implementation.
    workers = max(1, int(getattr(config, "VP_FETCH_CONCURRENCY", 8)))
    ahead = workers * 2
    pending: dict[int, Future | Exception] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:

        def submit(index: int) -> None:
            figure = figures[index]
            article = article_rows[figure["pmcid"]]
            if article is None:
                pending[index] = ValueError(
                    f"no articles row for {figure['pmcid']}"
                )
            else:
                pending[index] = pool.submit(_prepare_figure, figure, article)

        for index in range(min(ahead, len(figures))):
            submit(index)
        for index, figure in enumerate(figures):
            entry = pending.pop(index)
            if isinstance(entry, Future):
                try:
                    outcome = entry.result()
                except Exception as exc:  # noqa: BLE001 - per-figure isolation
                    outcome = exc
            else:
                outcome = entry
            follow = index + ahead
            if follow < len(figures):
                submit(follow)

            if isinstance(outcome, Exception):
                _store_error(figure, outcome)
                continue
            try:
                # One transaction per figure: file writes, panel rows,
                # proposed-finding upserts and the status flip commit together
                # or roll back together, so a rerun never double-counts
                # proposals.
                with conn:
                    stats = _apply_prepared(
                        conn, figure, article_rows[figure["pmcid"]], outcome, data_dir
                    )
                    final_status = (
                        "vision_rejected"
                        if stats["panels"] == 0 and stats["excluded"]
                        else "stored"
                    )
                    db.set_status(conn, "figures", figure["figure_id"], final_status, error=None)
            except Exception as exc:  # noqa: BLE001 - per-figure isolation
                _store_error(figure, exc)
                continue
            for key in ("panels", "whole_figure", "dedup", "proposed"):
                totals[key] += stats[key]
            totals["stored"] += 1

    print(
        f"store: {totals['stored']} figures stored, {totals['panels']} panels "
        f"({totals['whole_figure']} whole-figure, {totals['dedup']} deduped), "
        f"{totals['proposed']} proposed-finding upserts, {totals['errors']} errors"
    )
    if totals["panels"]:
        # Newly stored panels can change each pair's gallery lead; refresh
        # the representative table so mappings never wait for report/startup.
        # Storage itself is never capped — all eligible panels stay stored.
        from . import representatives

        changes = representatives.rebuild(conn)
        print(f"store: representatives refreshed ({changes})")
    conn.close()
    return 0
