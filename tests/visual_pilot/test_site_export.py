"""Static site export: panels point at public figure URLs with crop boxes."""

from balanced_fixtures import add_article, add_disease, add_figure, add_panel, make_db
from src.visual_pilot import db, site_export, store

URL = "https://pmc-oa-opendata.s3.amazonaws.com/PMC1.1/f1.jpg"


def test_panel_sources_pad_crops_and_keep_whole_figures(tmp_path):
    conn = make_db(tmp_path)
    add_disease(conn, "sle")
    add_article(conn, "PMC1")
    add_figure(conn, "PMC1_f1", "PMC1")
    conn.execute("UPDATE figures SET image_url=? WHERE figure_id='PMC1_f1'", (URL,))
    add_panel(conn, tmp_path, "crop", "PMC1_f1", "PMC1", "sle", crop_mode="crop")
    add_panel(conn, tmp_path, "whole", "PMC1_f1", "PMC1", "sle")
    conn.execute("UPDATE panels SET bbox_json=? WHERE panel_id='crop'", (db.to_json([0.5, 0, 1, 0.5]),))
    conn.execute("UPDATE panels SET bbox_json=? WHERE panel_id='whole'", (db.to_json([0, 0, 1, 1]),))
    sources = site_export._panel_sources(conn)
    pad = store.PAD_FRAC
    assert sources["crop"] == {"url": URL, "crop": [0.5 - pad, 0.0, 1.0, 0.5 + pad]}
    assert sources["whole"] == {"url": URL, "crop": None}
