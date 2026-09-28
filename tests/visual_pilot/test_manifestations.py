"""Regression coverage for clinically relevant extra-cutaneous vocabulary."""

from src.visual_pilot import diseases


def test_psoriasis_and_psoriatic_arthritis_include_eye_manifestations():
    """Ocular findings are explicitly searchable and routed to the eye category.

    Psoriasis-associated uveitis is documented in reviews (PMCID: PMC7381949,
    PMC3699904); associations for ocular surface findings are summarized in
    the 2025 systematic review/meta-analysis (PMCID: PMC11914409).
    """
    vocab = {item["finding_key"]: item for item in diseases.load_findings_vocab()}
    expected = {
        "anterior_uveitis",
        "psoriasis_dry_eye",
        "psoriasis_conjunctivitis",
        "psoriasis_meibomian_gland_dysfunction",
    }

    assert expected <= vocab.keys()
    for key in expected:
        item = vocab[key]
        assert item["category"] == "eye"
        assert {"psoriasis", "psa"} <= set(item["disease_keys"])
        assert item["synonyms"]
    assert "as" in vocab["anterior_uveitis"]["disease_keys"]
