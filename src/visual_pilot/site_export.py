"""Static site export for GitHub Pages.

Calls the viewer API in-process and writes each response as JSON next to a
copy of the viewer pages, so the library runs with no server. Images are not
copied: as in the viewer, each panel points at its figure's public URL in the
PMC open-data bucket, and cropped panels carry the normalized crop box
(``crop``) that the page applies in the browser.

Usage: ``python -m src.visual_pilot.cli export-site [--out site]``
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from . import config
from .viewer.app import STATIC_DIR, create_app

DEFAULT_OUT = config.REPO_ROOT / "site"
PAGES = ("index.html", "disease.html", "compare.html")
ASSETS = ("app.js", "api.js", "style.css")
STATIC_FLAG = "<script>window.VP_STATIC = true;</script>"


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

    if out.exists():
        shutil.rmtree(out)
    data = out / "data"
    diseases = _get(client, "/api/diseases")
    _write(data / "diseases.json", diseases)
    # Only articles that contribute a published image; the rest were screened out.
    articles = [a for a in _get(client, "/api/articles") if a["published_total"]]
    _write(data / "articles.json", articles)
    totals = {}
    for d in diseases:
        key = d["key"]
        _write(data / key / "tabs.json", _get(client, f"/api/diseases/{key}/tabs"))
        _write(data / key / "vocab.json", _get(client, f"/api/vocab?disease={key}"))
        _write(data / key / "eye-evidence.json", _get(client, f"/api/diseases/{key}/eye-evidence"))
        panels = _get(client, f"/api/diseases/{key}/panels")
        panels["panels"] = [p for p in panels["panels"] if p.get("image")]
        _write(data / key / "panels.json", panels)
        totals[key] = len(panels["panels"])
    compare = _get(client, "/api/compare/sle-dm-skin")
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
