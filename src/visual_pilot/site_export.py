"""Static site export for GitHub Pages.

Calls the viewer API in-process and writes each response as JSON next to a
copy of the viewer pages, so the library runs with no server. Images are not
copied: each panel points at its figure's public URL in the PMC open-data
bucket, and cropped panels carry the normalized crop box (``crop``) that the
page applies in the browser. Hosting the stored crops on a CDN later only
changes ``_image_url``.

Usage: ``python -m src.visual_pilot.cli export-site [--out site]``
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from . import config, db, store
from .viewer.app import STATIC_DIR, create_app

DEFAULT_OUT = config.REPO_ROOT / "site"
PAGES = ("index.html", "disease.html", "compare.html")
ASSETS = ("app.js", "api.js", "style.css")
STATIC_FLAG = "<script>window.VP_STATIC = true;</script>"


def _panel_sources(conn) -> dict[str, dict]:
    """panel_id -> public figure URL and padded crop box (None = whole figure)."""
    out = {}
    for r in conn.execute(
        "SELECT p.panel_id, p.bbox_json, p.crop_mode, f.image_url "
        "FROM panels p JOIN figures f ON f.figure_id = p.figure_id"
    ):
        bbox = db.from_json(r["bbox_json"], None)
        crop = None
        if r["crop_mode"] != "whole_figure" and bbox and len(bbox) == 4 and bbox != [0, 0, 1, 1]:
            pad = store.PAD_FRAC
            crop = [
                round(max(0.0, bbox[0] - pad), 4), round(max(0.0, bbox[1] - pad), 4),
                round(min(1.0, bbox[2] + pad), 4), round(min(1.0, bbox[3] + pad), 4),
            ]
        out[r["panel_id"]] = {"url": r["image_url"], "crop": crop}
    return out


def _image_url(panel: dict, sources: dict[str, dict]) -> None:
    """Point a panel (and its source variants) at the public figure image."""
    src = sources.get(panel.get("panel_id")) or {}
    if not src.get("url"):
        panel["image"] = panel["thumb"] = None
        return
    panel["image"] = panel["thumb"] = src["url"]
    panel["crop"] = src["crop"]


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _get(client: TestClient, url: str):
    resp = client.get(url)
    resp.raise_for_status()
    return resp.json()


def export(out: Path = DEFAULT_OUT, data_dir: str | None = None, log=print) -> dict:
    data_root = Path(data_dir) if data_dir else config.data_dir()
    client = TestClient(create_app(str(data_root)))
    conn = db.connect(data_root / config.DB_FILENAME)
    try:
        sources = _panel_sources(conn)
    finally:
        conn.close()

    if out.exists():
        shutil.rmtree(out)
    data = out / "data"
    diseases = _get(client, "/api/diseases")
    _write(data / "diseases.json", diseases)
    _write(data / "articles.json", _get(client, "/api/articles"))
    totals = {}
    for d in diseases:
        key = d["key"]
        _write(data / key / "tabs.json", _get(client, f"/api/diseases/{key}/tabs"))
        _write(data / key / "vocab.json", _get(client, f"/api/vocab?disease={key}"))
        _write(data / key / "eye-evidence.json", _get(client, f"/api/diseases/{key}/eye-evidence"))
        panels = _get(client, f"/api/diseases/{key}/panels")
        for p in panels["panels"]:
            _image_url(p, sources)
        panels["panels"] = [p for p in panels["panels"] if p.get("image")]
        _write(data / key / "panels.json", panels)
        totals[key] = len(panels["panels"])
    compare = _get(client, "/api/compare/sle-dm-skin")
    for pair in compare:
        for side in ("left", "right"):
            for p in pair[side]["panels"]:
                _image_url(p, sources)
    _write(data / "compare-sle-dm-skin.json", compare)

    for name in PAGES:
        html = (STATIC_DIR / name).read_text(encoding="utf-8")
        html = html.replace('"/static/', '"').replace("<head>", f"<head>\n  {STATIC_FLAG}", 1)
        (out / name).write_text(html, encoding="utf-8")
    for name in ASSETS:
        shutil.copy2(STATIC_DIR / name, out / name)
    (out / ".nojekyll").write_text("", encoding="utf-8")
    log(f"export-site: {out} — {sum(totals.values())} image(s): "
        + ", ".join(f"{k}={n}" for k, n in totals.items()))
    return totals


def run(args) -> int:
    export(Path(args.out).resolve() if getattr(args, "out", None) else DEFAULT_OUT)
    return 0
