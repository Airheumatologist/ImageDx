"""Panels point at public figure URLs with padded crop boxes."""

from src.visual_pilot import store


def test_crop_box_pads_crops_and_keeps_whole_figures():
    pad = store.PAD_FRAC
    assert store.crop_box([0.5, 0, 1, 0.5], "panel") == [0.5 - pad, 0.0, 1.0, 0.5 + pad]
    assert store.crop_box([0, 0, 1, 1], "panel") is None
    assert store.crop_box([0.5, 0, 1, 0.5], "whole_figure") is None
    assert store.crop_box(None, "panel") is None


def _site(root, topics, articles):
    import json
    data = root / "data"
    data.mkdir(parents=True)
    (data / "diseases.json").write_text(json.dumps([{"key": k} for k in topics]))
    (data / "articles.json").write_text(json.dumps(articles))
    for key, n in topics.items():
        (data / key).mkdir()
        (data / key / "panels.json").write_text(json.dumps({"panels": [{"panel_id": i} for i in range(n)]}))
    return root


def test_merge_keeps_the_richer_side_per_topic(tmp_path):
    import json
    from src.visual_pilot import site_export

    old = _site(tmp_path / "site", {"a": 5, "b": 2}, [
        {"pmcid": "P1", "disease_keys": ["a"], "published_images": {"a": 5}, "published_total": 5},
        {"pmcid": "P2", "disease_keys": ["b"], "published_images": {"b": 2}, "published_total": 2},
    ])
    fresh = _site(tmp_path / "fresh", {"a": 0, "b": 4, "c": 1}, [
        {"pmcid": "P3", "disease_keys": ["b", "c"], "published_images": {"b": 4, "c": 1},
         "published_total": 5},
    ])
    source = site_export.merge(old, fresh, log=lambda *_: None)

    assert source == {"a": "kept", "b": "fresh", "c": "fresh"}
    data = old / "data"
    assert len(json.loads((data / "a" / "panels.json").read_text())["panels"]) == 5
    assert len(json.loads((data / "b" / "panels.json").read_text())["panels"]) == 4
    articles = {a["pmcid"]: a for a in json.loads((data / "articles.json").read_text())}
    assert set(articles) == {"P1", "P3"}
    assert articles["P3"]["published_total"] == 5
    assert not fresh.exists()
