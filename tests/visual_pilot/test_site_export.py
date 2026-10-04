"""Panels point at public figure URLs with padded crop boxes."""

from src.visual_pilot import store


def test_crop_box_pads_crops_and_keeps_whole_figures():
    pad = store.PAD_FRAC
    assert store.crop_box([0.5, 0, 1, 0.5], "panel") == [0.5 - pad, 0.0, 1.0, 0.5 + pad]
    assert store.crop_box([0, 0, 1, 1], "panel") is None
    assert store.crop_box([0.5, 0, 1, 0.5], "whole_figure") is None
    assert store.crop_box(None, "panel") is None
