"""Panels point at public figure URLs with padded crop boxes."""

from src.visual_pilot import store


def test_crop_box_pads_crops_and_keeps_whole_figures():
    pad = store.PAD_FRAC
    assert store.crop_box([0.5, 0, 1, 0.5], "panel") == [0.5 - pad, 0.0, 1.0, 0.5 + pad]
    assert store.crop_box([0, 0, 1, 1], "panel") is None
    assert store.crop_box([0.5, 0, 1, 0.5], "whole_figure") is None
    assert store.crop_box(None, "panel") is None


def _site(root, topics, articles):
    """``topics`` maps a topic to its panels as (panel_id, pmcid, sha256, figure_id, tab)."""
    import json
    data = root / "data"
    data.mkdir(parents=True)
    (data / "diseases.json").write_text(json.dumps([{"key": k} for k in topics]))
    (data / "articles.json").write_text(json.dumps(articles))
    for key, panels in topics.items():
        (data / key).mkdir()
        rows = [{"panel_id": pid, "pmcid": pmcid, "sha256": sha, "figure_id": fig, "tab": tab,
                 "is_representative": True, "representative_findings": ["f"]}
                for pid, pmcid, sha, fig, tab in panels]
        (data / key / "panels.json").write_text(json.dumps(
            {"panels": rows, "count": len(rows), "representatives": {"f": {"panel_id": "x"}}}))
        tabs = sorted({tab for *_, tab in panels})
        (data / key / "tabs.json").write_text(json.dumps([{"key": t} for t in tabs]))
        (data / key / "vocab.json").write_text(json.dumps([{"finding_key": f"{root.name}-{key}"}]))
    return root


def _panels(n, pmcid, prefix="p", tab="skin"):
    return [(f"{prefix}{i}", pmcid, f"{prefix}-sha{i}", f"{pmcid}:{prefix}fig{i}", tab) for i in range(n)]


def test_merge_adds_new_images_and_never_shrinks_a_topic(tmp_path):
    import json
    from src.visual_pilot import site_export

    old = _site(tmp_path / "site", {"a": _panels(5, "P1"), "b": _panels(2, "P2")}, [
        {"pmcid": "P1", "disease_keys": ["a"], "published_images": {"a": 5}, "published_total": 5},
        {"pmcid": "P2", "disease_keys": ["b"], "published_images": {"b": 2}, "published_total": 2},
    ])
    # The database shows fewer images for b than the site, but two are new.
    fresh = _site(tmp_path / "fresh", {"a": [], "b": _panels(1, "P2") + _panels(2, "P3", "q", "imaging"),
                                        "c": _panels(1, "P3", "r")}, [
        {"pmcid": "P2", "disease_keys": ["b"], "published_images": {"b": 1}, "published_total": 1},
        {"pmcid": "P3", "disease_keys": ["b", "c"], "published_images": {"b": 2, "c": 1},
         "published_total": 3},
    ])
    source = site_export.merge(old, fresh, log=lambda *_: None)

    assert source == {"a": "kept", "b": "merged", "c": "fresh"}
    data = old / "data"
    b = json.loads((data / "b" / "panels.json").read_text())
    assert [p["panel_id"] for p in b["panels"]] == ["p0", "p1", "q0", "q1"] and b["count"] == 4
    assert b["panels"][0]["is_representative"] and not b["panels"][2]["is_representative"]
    assert b["representatives"] == {"f": {"panel_id": "x"}}
    assert [t["key"] for t in json.loads((data / "b" / "tabs.json").read_text())] == ["skin", "imaging"]
    assert json.loads((data / "b" / "vocab.json").read_text()) == [{"finding_key": "site-b"}]
    assert len(json.loads((data / "a" / "panels.json").read_text())["panels"]) == 5
    articles = {a["pmcid"]: a for a in json.loads((data / "articles.json").read_text())}
    assert articles["P2"]["published_images"] == {"b": 2}
    assert articles["P3"]["published_images"] == {"b": 2, "c": 1} and articles["P3"]["published_total"] == 3
    assert not fresh.exists()


def test_merge_skips_known_images_and_figures_and_stops_at_the_cap(tmp_path, monkeypatch):
    import json
    from src.visual_pilot import site_export

    monkeypatch.setattr(site_export.config, "VP_TOPIC_IMAGE_CAP", 4)
    old = _site(tmp_path / "site", {"a": _panels(2, "P1")}, [
        {"pmcid": "P1", "disease_keys": ["a"], "published_images": {"a": 2}, "published_total": 2}])
    same_sha = ("n0", "P9", "p-sha0", "P9:other", "skin")      # same pixels as p0
    same_fig = ("n1", "P1", "n-sha1", "P1:pfig1", "skin")      # another crop of p1's figure
    fresh = _site(tmp_path / "fresh", {"a": [same_sha, same_fig] + _panels(5, "P2", "m")}, [
        {"pmcid": "P2", "disease_keys": ["a"], "published_images": {"a": 5}, "published_total": 5}])
    site_export.merge(old, fresh, log=lambda *_: None)

    panels = json.loads((old / "data" / "a" / "panels.json").read_text())["panels"]
    assert [p["panel_id"] for p in panels] == ["p0", "p1", "m0", "m1"]
    articles = {a["pmcid"]: a for a in json.loads((old / "data" / "articles.json").read_text())}
    assert articles["P2"]["published_images"] == {"a": 2} and articles["P1"]["published_images"] == {"a": 2}


def test_article_counts_follow_the_published_panels():
    from src.visual_pilot import site_export

    rows = [  # the API counts finding picks only, so fill images read as 0 there
        {"pmcid": "P1", "title": "fresh", "published_images": {}, "published_total": 0},
        {"pmcid": "P2", "title": "unshown", "published_images": {"a": 1}, "published_total": 1},
        {"pmcid": "P1", "title": "stale", "published_images": {"a": 9}, "published_total": 9},
    ]
    shown = {"a": [{"pmcid": "P1"}, {"pmcid": "P1"}], "b": [{"pmcid": "P1"}]}
    assert site_export._article_rows(rows, shown) == [
        {"pmcid": "P1", "title": "fresh", "published_images": {"a": 2, "b": 1}, "published_total": 3}]
