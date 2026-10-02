"""Scratch fixtures for balanced-coverage tests.

Offline only: real Pillow image files under a tmp_path data directory and a
scratch SQLite DB; no network, providers, or the main database.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.visual_pilot import db

MALAR_CAPTION = (
    "Malar rash in a 23-year-old patient with systemic lupus erythematosus"
)


def make_db(tmp_path) -> "object":
    """Scratch DB; image paths resolve against its parent directory."""
    conn = db.connect(tmp_path / "visual_pilot.sqlite")
    db.init_db(conn)
    return conn


def write_image(root: Path, rel: str, size=(400, 300), color=(120, 80, 40)) -> str:
    path = Path(root) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", tuple(size), tuple(color)).save(path, format="PNG")
    return rel


def add_disease(conn, key: str, name: str | None = None) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO diseases(disease_key,name) VALUES(?,?)",
        (key, name or key),
    )


def add_finding(
    conn,
    finding_key: str,
    disease_keys,
    *,
    label: str | None = None,
    category: str = "skin",
    approved: int = 1,
    synonyms=(),
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO findings_vocab"
        "(finding_key,disease_keys_json,label,synonyms_json,category,approved) "
        "VALUES(?,?,?,?,?,?)",
        (
            finding_key,
            db.to_json(list(disease_keys)),
            label or finding_key.replace("_", " "),
            db.to_json(list(synonyms)),
            category,
            approved,
        ),
    )


def add_article(
    conn,
    pmcid: str,
    *,
    license_code: str | None = "CC-BY",
    title: str = "Cutaneous manifestations of lupus: a review",
    publication_types=("Review",),
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO articles"
        "(pmcid,title,license_code,publication_types_json,status) "
        "VALUES(?,?,?,?,'relevant')",
        (pmcid, title, license_code, db.to_json(list(publication_types))),
    )


def add_figure(
    conn,
    figure_id: str,
    pmcid: str,
    *,
    caption: str = MALAR_CAPTION,
    license_code: str | None = "CC-BY",
    triage: dict | None = None,
    vision: dict | None = None,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO figures"
        "(figure_id,pmcid,caption,effective_license,triage_json,vision_json,status) "
        "VALUES(?,?,?,?,?,?,'stored')",
        (
            figure_id,
            pmcid,
            caption,
            license_code,
            db.to_json(triage or {}),
            db.to_json(vision) if vision is not None else None,
        ),
    )


def add_panel(
    conn,
    root: Path,
    panel_id: str,
    figure_id: str,
    pmcid: str,
    disease_key: str,
    *,
    findings=("malar_rash",),
    sha256: str | None = None,
    license_code: str | None = "CC-BY",
    crop_mode: str = "whole_figure",
    plate_kind: str | None = None,
    plate_findings=(),
    width: int = 400,
    height: int = 300,
    confidence: float = 0.9,
    typicality: str = "classic",
    modality: str = "clinical_photo",
    body_site: str = "face",
    image_size=None,
    image: bool = True,
    thumb: bool = True,
) -> None:
    """Insert a panel row; write real PNG files under ``root`` unless disabled.

    ``image_size`` controls the actual file dimensions independently of the
    row's stored width/height so tests can exercise dimension mismatches.
    """
    rel_img = f"panels/{panel_id}.png"
    rel_thumb = f"thumbs/{panel_id}.png"
    file_size = image_size or (width, height)
    if image:
        write_image(root, rel_img, file_size)
    if thumb:
        write_image(root, rel_thumb, (100, 75))
    conn.execute(
        "INSERT OR REPLACE INTO panels"
        "(panel_id,figure_id,pmcid,disease_key,modality,body_site,"
        "findings_json,plate_findings_json,typicality,confidence,crop_mode,"
        "plate_kind,annotations_present,width,height,sha256,license_code,"
        "image_path,thumb_path) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            panel_id,
            figure_id,
            pmcid,
            disease_key,
            modality,
            body_site,
            db.to_json(
                [{"finding_key": k, "evidence": k.replace("_", " ")} for k in findings]
            ),
            db.to_json(
                [{"finding_key": k, "evidence": k.replace("_", " ")} for k in plate_findings]
            ),
            typicality,
            confidence,
            crop_mode,
            plate_kind,
            0,
            width,
            height,
            sha256,
            license_code,
            rel_img,
            rel_thumb,
        ),
    )


def add_identity_review(
    conn,
    panel_id: str,
    *,
    sha256: str,
    patient_group_key: str | None = None,
    reuse_group_key: str | None = None,
    source_quote: str = "Figure 1. Patient photograph.",
    review_provenance: str = "manual-audit",
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO panel_identity_reviews"
        "(panel_id,patient_group_key,reuse_group_key,source_quote,"
        "review_provenance,reviewed_image_sha256) VALUES(?,?,?,?,?,?)",
        (
            panel_id,
            patient_group_key,
            reuse_group_key,
            source_quote,
            review_provenance,
            sha256,
        ),
    )


def gallery_panel(
    panel_id: str,
    *,
    pmcid: str = "PMC_A",
    figure_id: str | None = None,
    sha256: str | None = None,
    disease_key: str = "sle",
    supported=("malar_rash",),
    eligible: bool = True,
    identity_review: dict | None = None,
    confidence: float = 0.9,
    typicality: str = "classic",
    width: int = 400,
    height: int = 300,
    crop_mode: str = "whole_figure",
    annotations_present: int = 0,
    **extra,
) -> dict:
    """Pure-selector input dict shaped like ``publication.eligible_panels``."""
    return {**extra,
        "panel_id": panel_id,
        "pmcid": pmcid,
        "figure_id": figure_id or f"{pmcid}:fig1",
        "disease_key": disease_key,
        "sha256": sha256,
        "eligible": eligible,
        "supported_finding_keys": list(supported),
        "identity_review": identity_review,
        "confidence": confidence,
        "typicality": typicality,
        "width": width,
        "height": height,
        "crop_mode": crop_mode,
        "annotations_present": annotations_present,
    }


def identity_review(
    sha256: str,
    *,
    patient_group_key: str | None = None,
    reuse_group_key: str | None = None,
    source_quote: str = "Figure 1. Patient photograph.",
    review_provenance: str = "manual-audit",
) -> dict:
    return {
        "patient_group_key": patient_group_key,
        "reuse_group_key": reuse_group_key,
        "source_quote": source_quote,
        "review_provenance": review_provenance,
        "reviewed_image_sha256": sha256,
    }
