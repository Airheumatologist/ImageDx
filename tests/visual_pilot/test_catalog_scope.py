"""Configured disease scope is shared by CLI, prompts, seed data and judge."""

from __future__ import annotations

from src.visual_pilot import cli, diseases, judge
from src.visual_pilot.prompts import P1, P2, P3, P4


def test_configured_diseases_are_the_cli_and_prompt_allowlist():
    keys = list(diseases.load_diseases())
    parser = cli.build_parser()
    run_all = next(a for a in parser._actions if a.dest == "command").choices["run-all"]
    disease_action = next(a for a in run_all._actions if a.dest == "disease")

    assert set(disease_action.choices) == {*keys, "all"}
    assert P1.schema["properties"]["primary_disease_keys"]["items"]["enum"] == keys
    assert P2.schema["properties"]["results"]["items"]["properties"][
        "diseases_mentioned"
    ]["items"]["enum"] == [*keys, "other"]
    assert P3.schema["properties"]["panels"]["items"]["properties"][
        "disease_key"
    ]["enum"] == [*keys, None]
    assert P4.schema["properties"]["assertions"]["items"]["properties"][
        "disease_key"
    ]["enum"] == keys
    for key, disease in diseases.load_diseases().items():
        assert key in P1.system and disease["name"] in P1.system
        assert key in P3.system and disease["name"] in P3.system


def test_judge_subtypes_follow_configured_disease_metadata():
    expected = {
        key: {entry["key"] for entry in disease.get("subtypes", [])}
        for key, disease in diseases.load_diseases().items()
    }
    assert judge.SUBTYPES == expected
