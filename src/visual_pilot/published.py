"""Topic-specific live-site duplicate checks before spending LLM calls.

Use the same figure/hash identity as export-site --merge. An image is skipped
only when every topic attributed to its article already shows that identity.
An unrelated topic or a partial multi-topic match cannot suppress analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import io
import json
from pathlib import Path

from PIL import Image

from . import config, db, store


@dataclass
class PublishedIndex:
    figures: dict[str, set[str]] = field(default_factory=dict)
    urls: dict[str, set[str]] = field(default_factory=dict)
    hashes: dict[str, set[str]] = field(default_factory=dict)
    panel_ids: dict[str, set[str]] = field(default_factory=dict)
    gallery_identities: dict[str, set[str]] = field(default_factory=dict)

    def match(self, figure: dict, topics, pixel_hash: str | None = None) -> dict | None:
        keys = set(topics or [])
        if not keys:
            return None
        matches = {}
        for topic in sorted(keys):
            for kind, value, lookup in (
                ("figure_id", figure.get("figure_id"), self.figures),
                ("image_url", figure.get("image_url"), self.urls),
                ("pixel_sha256", pixel_hash, self.hashes),
            ):
                if value and value in lookup.get(topic, set()):
                    matches[topic] = {"kind": kind, "value": value}
                    break
            else:
                return None
        return matches

    def match_bytes(self, figure: dict, topics, original: bytes) -> dict | None:
        keys = set(topics or [])
        if not keys or not all(self.hashes.get(topic) for topic in keys):
            return None
        with Image.open(io.BytesIO(original)) as image:
            # Exactly the RGB decoded-pixel hash stored in panels/site JSON;
            # a hash of encoded JPEG/PNG bytes would not be comparable.
            pixel_hash = store._pixel_sha(image.convert("RGB"))
        return self.match(figure, keys, pixel_hash)


@lru_cache(maxsize=4)
def _load(root: str, catalog_mtime: int) -> PublishedIndex:
    index = PublishedIndex()
    for path in sorted((Path(root) / "data").glob("*/panels.json")):
        topic = path.parent.name
        try:
            panels = json.loads(path.read_text())["panels"]
        except (ValueError, KeyError) as exc:
            raise ValueError(f"Cannot read published-image index: {path}") from exc
        for panel in panels:
            identity = panel.get("sha256") or panel.get("panel_id")
            if identity:
                index.gallery_identities.setdefault(topic, set()).add(identity)
            for value, lookup in (
                (panel.get("figure_id"), index.figures),
                (panel.get("image"), index.urls),
                (panel.get("sha256"), index.hashes),
                (panel.get("panel_id"), index.panel_ids),
            ):
                if value:
                    lookup.setdefault(topic, set()).add(value)
    return index


def site_index() -> PublishedIndex:
    root = config.VP_PUBLISHED_SITE_DIR.resolve()
    catalog = root / "data/diseases.json"
    return _load(str(root), catalog.stat().st_mtime_ns if catalog.exists() else 0)


def merged_topic_counts(conn, topics) -> dict[str, int]:
    """Count the site plus its nonduplicate, currently gallery-ready additions.

    Read only. This measures the result of a merge, so a topic with eight
    live images and two new database images has reached a target of ten.
    Stored but ineligible panels and already-live source figures add nothing.
    """
    from . import coverage, pair_reporting

    keys = set(topics)
    index = site_index()
    images = {key: set(index.gallery_identities.get(key, ())) for key in keys}
    seen = {key: set().union(index.figures.get(key, ()), index.hashes.get(key, ()),
                            index.panel_ids.get(key, ())) for key in keys}
    with coverage.consistent_read(conn):
        ready = pair_reporting.library_rows(conn)
        with_urls = {r[0] for r in conn.execute(
            "SELECT figure_id FROM figures WHERE image_url IS NOT NULL AND image_url<>''")}
    for panel in ready:
        key = panel.get("disease_key")
        if key not in keys or panel.get("figure_id") not in with_urls:
            continue
        if config.VP_TOPIC_IMAGE_CAP and len(images[key]) >= config.VP_TOPIC_IMAGE_CAP:
            continue
        identities = {panel.get("sha256"), panel.get("panel_id"), panel.get("figure_id")} - {None, ""}
        if identities & seen[key]:
            continue
        images[key].add(panel.get("sha256") or panel["panel_id"])
        seen[key].update(identities)
    return {key: len(values) for key, values in images.items()}


def mark_skipped(conn, figure: dict, matches: dict, stage: str) -> None:
    triage = db.from_json(figure.get("triage_json"), {}) or {}
    triage["published_duplicate"] = {
        "policy_version": "site-duplicate.v1", "stage": stage,
        "reason": "already live for every attributed topic", "matches": matches,
    }
    db.set_status(conn, "figures", figure["figure_id"], "already_published",
                  triage_json=db.to_json(triage), error=None)


def filter_queue(conn, rows: list[dict], topics_for, stage: str, *, dry_run=False) -> list[dict]:
    index = site_index()
    retained = []
    skipped = 0
    for figure in rows:
        matches = index.match(figure, topics_for(figure))
        if matches:
            skipped += 1
            if not dry_run:
                mark_skipped(conn, figure, matches, stage)
        else:
            retained.append(figure)
    if skipped:
        if not dry_run:
            conn.commit()
        print(f"{stage}: {'would skip' if dry_run else 'skipped'} {skipped} already-published figure(s) before LLM/image fetch")
    return retained
