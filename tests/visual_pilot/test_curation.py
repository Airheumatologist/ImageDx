"""Deterministic gates for clinically useful, whole-image candidates."""

from src.visual_pilot import curation, judge


def _panel(**overrides):
    return {
        "include": True,
        "disease_key": "sle",
        "subtype": "acle",
        "bbox": [0, 0, 1, 1],
        "modality": "clinical_photo",
        "body_site": "face",
        "findings": [{"finding_key": "malar_rash", "evidence": "malar rash"}],
        "proposed_findings": [],
        "rationale": "A clinical photo shows malar rash, not a diagram.",
        **overrides,
    }


def test_mechanism_review_can_contain_valid_photo_and_not_diagram_reasoning():
    assert curation.exclusion_reason(
        _panel(),
        {"caption": "Clinical photograph of malar rash in a patient with SLE."},
        {"title": "Pathogenesis and mechanisms in systemic lupus erythematosus"},
    ) is None


def test_disease_focused_title_supports_terse_caption_but_generic_title_does_not():
    panel = _panel()
    figure = {"caption": "Malar rash."}
    assert curation.exclusion_reason(
        panel, figure, {"title": "Cutaneous manifestations of lupus erythematosus"}
    ) is None
    assert curation.exclusion_reason(
        panel, figure, {"title": "Pericardial Effusion: Overview of Aetiology and Management"}
    ) == "disease association not stated in figure caption"


def test_collage_small_panel_chart_vet_and_controls_are_excluded():
    panel = _panel()
    assert "collage" in curation.exclusion_reason(
        panel,
        {"caption": "Clinical images", "figure_is_compound": True},
        {"title": "Lupus"},
    )
    assert "too small" in curation.exclusion_reason(
        _panel(bbox=[0, 0, 0.24, 0.25]), {"caption": "Clinical image"}, {"title": "Lupus"}
    )
    assert "chart" in curation.exclusion_reason(
        panel, {"caption": "Lupus nephritis classification chart."}, {"title": "Lupus"}
    )
    assert "veterinary" in curation.exclusion_reason(
        panel, {"caption": "Clinical images of dogs with lupus."}, {"title": "Lupus"}
    )
    assert "control" in curation.exclusion_reason(
        panel, {"caption": "Normal control MRI."}, {"title": "Lupus"}
    )


def test_missing_manifestation_is_rejected_but_rationale_negation_is_safe():
    assert curation.exclusion_reason(
        _panel(findings=[], proposed_findings=[]),
        {"caption": "Clinical photo of a face."},
        {"title": "SLE"},
    ) == "no specific clinical manifestation identified"
    assert curation.exclusion_reason(
        _panel(rationale="This is not a chart or diagram."),
        {"caption": "Clinical photograph of lupus malar rash."},
        {"title": "SLE"},
    ) is None


def test_post_validate_enforces_source_gate_and_records_reason():
    result = {
        "figure_is_compound": True,
        "panels": [_panel()],
    }
    validated = judge.post_validate(
        result, {"malar_rash"},
        figure={"caption": "Clinical photographs"},
        article={"title": "Lupus"},
        image_size=(800, 800),
    )
    assert validated["panels"][0]["include"] is False
    assert validated["panels"][0]["curation_reason"] == "multi-panel collage; skip whole source"


def test_explicit_graphic_captions_override_incorrect_patient_claim():
    article = {'title': 'Lupus review'}
    assert curation.exclusion_reason(_panel(), {
        'caption': 'Pathogenesis of oral ulcer. The schematic depicts a cascade (created with https://biorender.com/).',
        'triage_json': {'is_real_patient_image': True},
    }, article) == 'diagram or schematic'
    assert curation.exclusion_reason(_panel(), {
        'caption': 'Histopathology classification of lupus nephritis according to ISN/RPS criteria.',
        'triage_json': {'is_real_patient_image': True},
    }, article) == 'chart or classification table'


def test_another_pilot_disease_does_not_inherit_review_topic():
    assert curation.exclusion_reason(_panel(), {
        'caption': 'Clinical photograph of dermatomyositis in this patient.',
    }, {'title': 'Lupus and its differential diagnoses'}) == 'caption identifies another disease'


def test_catalog_matcher_uses_full_phrase_boundaries_and_cross_disease_scope(monkeypatch):
    monkeypatch.setattr(curation.diseases, "load_diseases", lambda: {
        "ra": {"name": "Rheumatoid arthritis", "synonyms": ["RA"]},
        "ad": {"name": "Atopic dermatitis", "synonyms": ["AD"]},
    })
    curation._disease_matchers.cache_clear()
    try:
        assert curation._diseases_in_text("RA is a common abbreviation in notes.") == set()
        assert curation._diseases_in_text("Rheumatoid-arthritis lesions.") == {"ra"}
        assert curation.exclusion_reason(
            _panel(disease_key="ra"),
            {"caption": "Clinical photograph of atopic dermatitis."},
            {"title": "Rheumatoid arthritis overview"},
        ) == "caption identifies another disease"
        assert curation.exclusion_reason(
            _panel(disease_key="ra"), {"caption": "Pannus."},
            {"title": "Rheumatoid arthritis"},
        ) is None
    finally:
        curation._disease_matchers.cache_clear()
