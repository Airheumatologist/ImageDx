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
from collections import Counter
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
    totals, shown = {}, {}
    for d in diseases:
        key = d["key"]
        _write(data / key / "tabs.json", _get(client, f"/api/diseases/{key}/tabs"))
        _write(data / key / "vocab.json", _get(client, f"/api/vocab?disease={key}"))
        _write(data / key / "eye-evidence.json", _get(client, f"/api/diseases/{key}/eye-evidence"))
        panels = _get(client, f"/api/diseases/{key}/panels")
        panels["panels"] = [p for p in panels["panels"] if p.get("image")]
        _write(data / key / "panels.json", panels)
        totals[key] = len(panels["panels"])
        shown[key] = panels["panels"]
    # Only articles that contribute a published image; the rest were screened out.
    _write(data / "articles.json", _article_rows(_get(client, "/api/articles"), shown))
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


def _article_rows(rows: list[dict], shown: dict[str, list[dict]]) -> list[dict]:
    """``rows`` (article metadata, first row per pmcid wins) for the articles
    whose images the topics in ``shown`` publish, counted from those panels so
    the article list always matches the topic pages, unlabeled fill included."""
    counts: dict[str, Counter] = {}
    for key, panels in shown.items():
        for panel in panels:
            counts.setdefault(panel.get("pmcid"), Counter())[key] += 1
    out, done = [], set()
    for row in rows:
        pmcid = row.get("pmcid")
        if pmcid in counts and pmcid not in done:
            done.add(pmcid)
            images = dict(sorted(counts[pmcid].items()))
            out.append({**row, "published_images": images, "published_total": sum(images.values())})
    return out


def _read(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _is_new(panel: dict, seen: set) -> bool:
    """Not already shown: no shared image hash, panel or figure (a crop of a
    figure the topic already shows would repeat it)."""
    keys = {panel.get("sha256"), panel.get("panel_id"), panel.get("figure_id")} - {None, ""}
    return not keys & seen


def _merge_topic(old_dir: Path, new_dir: Path, cap: int) -> tuple[str, list[dict]]:
    """Fold one topic's fresh export into its existing files, in ``new_dir``.

    Returns the topic's source (``fresh``, ``kept`` or ``merged``) and the
    panels it gained from the database.
    """
    old = _read(old_dir / "panels.json")
    fresh = _read(new_dir / "panels.json", {"panels": []})
    if old is None:
        return "fresh", fresh.get("panels") or []
    panels = list(old.get("panels") or [])
    seen = {v for p in panels for v in (p.get("sha256"), p.get("panel_id"), p.get("figure_id")) if v}
    added = []
    for panel in fresh.get("panels") or []:
        if cap and len(panels) + len(added) >= cap:
            break
        if _is_new(panel, seen):
            # The database's gallery picks do not carry over to the site's.
            added.append({**panel, "is_representative": False, "representative_findings": []})
            seen.update(v for v in (panel.get("sha256"), panel.get("panel_id"), panel.get("figure_id")) if v)
    fresh_tabs = _read(new_dir / "tabs.json", [])
    shutil.rmtree(new_dir, ignore_errors=True)
    shutil.copytree(old_dir, new_dir)
    if not added:
        return "kept", []
    _write(new_dir / "panels.json", {**old, "panels": panels + added, "count": len(panels) + len(added)})
    tabs = _read(new_dir / "tabs.json", [])
    known = {t.get("key") for t in tabs}
    used = {p.get("tab") for p in added}
    _write(new_dir / "tabs.json", tabs + [t for t in fresh_tabs if t.get("key") in used - known])
    return "merged", added


def merge(existing: Path, fresh: Path, log=print) -> dict:
    """Fold a fresh export into an existing site, topic by topic.

    A topic keeps every image it already shows and gains the database's
    images it does not show yet (no shared image hash, panel or figure), up
    to ``VP_TOPIC_IMAGE_CAP`` when set, so a database that holds only some
    topics, or fewer images than the site, never empties or shrinks one.
    A topic gaining images keeps its existing vocabulary and eye evidence and
    gains any tab its new images need; a topic the site lacks comes from the
    database whole. ``articles.json`` counts each article's images per topic
    from the merged topic pages. Pages, assets and ``diseases.json`` come from
    the fresh export. Returns ``{topic: "fresh" | "kept" | "merged"}``.
    """
    old_data, new_data = existing / "data", fresh / "data"
    topics = {d["key"] for d in json.loads((new_data / "diseases.json").read_text(encoding="utf-8"))}
    cap = int(config.VP_TOPIC_IMAGE_CAP or 0)
    source, added = {}, {}
    for key in sorted(topics):
        source[key], added[key] = _merge_topic(old_data / key, new_data / key, cap)
    if source.get("sle") != "fresh" and (old_data / "compare-sle-dm-skin.json").exists():
        shutil.copy2(old_data / "compare-sle-dm-skin.json", new_data / "compare-sle-dm-skin.json")

    # Fresh rows carry the database's metadata, including the articles behind new images.
    rows = _read(new_data / "articles.json", []) + _read(old_data / "articles.json", [])
    shown = {key: (_read(new_data / key / "panels.json", {}) or {}).get("panels") or [] for key in topics}
    _write(new_data / "articles.json", _article_rows(rows, shown))

    shutil.rmtree(existing)
    shutil.move(str(fresh), str(existing))
    counts = Counter(source.values())
    log(f"export-site --merge: {counts['merged']} topic(s) gained "
        f"{sum(len(v) for k, v in added.items() if source[k] == 'merged')} image(s), "
        f"{counts['fresh']} came whole from the database, {counts['kept']} unchanged")
    return source


def run(args) -> int:
    out = Path(args.out).resolve() if getattr(args, "out", None) else DEFAULT_OUT
    if getattr(args, "merge", False) and out.exists():
        fresh = out.with_name(out.name + ".fresh")
        export(fresh)
        merge(out, fresh)
    else:
        export(out)
    return 0
