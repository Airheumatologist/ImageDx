"""Pure visual yield ranking tests. No JATS/network or model calls."""

from src.visual_pilot.article_rank import caption_signal, rank_articles, score_article


def test_visual_passage_evidence_outweighs_generic_review_title():
    generic = {
        "pmcid": "PMC1",
        "title": "A review of dermatomyositis pathogenesis",
        "retrieval_score": 0.05,
    }
    visual = {
        "pmcid": "PMC2",
        "title": "A review of dermatomyositis",
        "retrieval_score": 0.04,
        "matched_passages": [
            {
                "text": "Clinical findings include Gottron papules and heliotrope rash.",
                "section": "Clinical manifestations",
                "section_type": "clinical",
                "finding": "gottron papules",
                "modality": "clinical photograph",
            }
        ],
    }
    assert rank_articles([generic, visual], "dm")[0][0]["pmcid"] == "PMC2"


def test_specific_caption_keeps_broad_review_eligible():
    result = score_article(
        {
            "pmcid": "PMC3",
            "title": "A comprehensive review of dermatomyositis mechanisms",
            "retrieval_score": 0.01,
        },
        "dm",
        captions=[
            {
                "caption": "Figure 4. Gottron papules on the dorsal fingers in a patient with dermatomyositis; clinical photographs.",
                "eligible": True,
            }
        ],
    )
    assert result["useful_caption_count"] == 1
    assert result["parts"]["caption_yield"] > 0


def test_caption_ranking_prefers_useful_image_and_rejects_diagram_signal():
    image = caption_signal(
        "MRI in axial spondyloarthritis showing sacroiliac erosions and bone marrow edema",
        "as",
    )
    diagram = caption_signal(
        "Schematic diagram of the inflammatory signaling pathway in axial spondyloarthritis",
        "as",
    )
    assert image["useful"] is True
    assert "mri" in image["modalities"]
    assert diagram["useful"] is False
    assert diagram["score"] < image["score"]


def test_ineligible_caption_does_not_raise_article_visual_score():
    article = {"title": "Dermatomyositis clinical findings", "retrieval_score": 0.01}
    eligible = score_article(article, "dm", captions=[{"caption": "Gottron papules clinical photograph", "eligible": True}])
    third_party = score_article(article, "dm", captions=[{"caption": "Gottron papules clinical photograph", "eligible": False}])
    assert eligible["score"] > third_party["score"]


def test_checked_article_without_figures_ranks_below_unchecked_candidate():
    article = {"title": "Lupus nephritis review", "retrieval_score": 0.01}
    unchecked = score_article(article, "sle")
    checked_empty = score_article(article, "sle", captions=[])
    assert checked_empty["score"] < unchecked["score"]
    assert checked_empty["parts"]["caption_uncertainty"] == -4.0


def test_conceptual_phenotype_and_histopathology_captions_stay_uncertain():
    phenotype = caption_signal(
        "The clinical phenotypes of anti-MDA5 dermatomyositis with varying degrees of pulmonary damage",
        "dm",
    )
    explanatory = caption_signal(
        "Histopathological features and biomarkers in JDM with capillary dropout and muscle fiber atrophy",
        "dm",
    )
    assert phenotype["useful"] is False
    assert phenotype["uncertain"] is True
    assert explanatory["useful"] is False
    assert explanatory["uncertain"] is True


def test_specific_patient_histology_caption_is_useful():
    pathology = caption_signal(
        "H&E stains in a patient with dermatomyositis show perivascular infiltrates and perifascicular atrophy",
        "dm",
    )
    assert pathology["useful"] is True
    assert pathology["explicit_image"] is True
