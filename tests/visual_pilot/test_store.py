"""Store records crop size and pixel hash; it writes no image files."""

import io
from types import SimpleNamespace

from PIL import Image

from balanced_fixtures import MALAR_CAPTION, add_article, add_disease, add_figure, make_db
from src.visual_pilot import db, store


def _png(size=(1600, 1200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 80, 40)).save(buf, format="PNG")
    return buf.getvalue()


def test_store_writes_no_files_and_records_crop_size(tmp_path, monkeypatch):
    monkeypatch.setenv("VP_DATA_DIR", str(tmp_path))
    conn = make_db(tmp_path)
    add_disease(conn, "sle")
    add_article(conn, "PMC1")
    vision = {
        "figure_is_compound": False,
        "panels": [
            {"panel_label": "A", "include": True, "disease_key": "sle",
             "modality": "clinical_photo", "bbox": [0.25, 0.25, 0.75, 0.75],
             "findings": [{"finding_key": "malar_rash", "evidence": "malar rash"}]},
        ],
    }
    add_figure(conn, "PMC1:fig1", "PMC1", caption=MALAR_CAPTION, vision=vision)
    conn.execute("UPDATE figures SET status='vision_accepted', label='Figure 1' WHERE figure_id='PMC1:fig1'")
    conn.commit()
    conn.close()
    monkeypatch.setattr(store, "_original_bytes", lambda figure: _png())
    monkeypatch.setattr(store, "_attribution_source", lambda row: (["Doe J"], 1, "J Test"))

    args = SimpleNamespace(refresh=False, disease="all", pmcids=None, limit=None, dry_run=False)
    assert store.run(args) == 0

    conn = make_db(tmp_path)
    rows = [dict(r) for r in conn.execute("SELECT * FROM panels ORDER BY panel_label")]
    status = conn.execute("SELECT status FROM figures").fetchone()["status"]
    conn.close()
    assert status == "stored"
    assert len(rows) == 1
    x0, y0, x1, y1 = store.bbox_to_pixels([0.25, 0.25, 0.75, 0.75], 1600, 1200)
    assert (rows[0]["width"], rows[0]["height"]) == (x1 - x0, y1 - y0)
    assert rows[0]["image_path"] is None and rows[0]["thumb_path"] is None
    assert len(rows[0]["sha256"]) == 64
    assert store.crop_box(db.from_json(rows[0]["bbox_json"], None), rows[0]["crop_mode"]) == [
        0.23, 0.23, 0.77, 0.77,
    ]
    # Nothing but the database is written to the data directory.
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        p.name for p in tmp_path.iterdir() if p.name.startswith("visual_pilot.sqlite")
    )
