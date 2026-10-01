from datetime import date

import pytest

from src.visual_pilot import config, db, manifestation_queue, pair_rank, parse, pmc, source_quality


@pytest.fixture
def conn(tmp_path):
    connection = db.init_db(db.connect(tmp_path / "ranking.sqlite"))
    yield connection
    connection.close()


def seed(conn, key, finding="anterior_uveitis"):
    # Post-C5 contract: lane candidates carry explicit pair provenance, and
    # the approved finding vocabulary defines the lane itself.
    conn.execute(
        "INSERT OR IGNORE INTO diseases(disease_key,name) "
        "VALUES('as','Ankylosing spondylitis')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO findings_vocab"
        "(finding_key,disease_keys_json,label,category,approved) "
        "VALUES(?, '[\"as\"]', ?, 'eye', 1)",
        (finding, finding.replace("_", " ")),
    )
    conn.execute("INSERT OR IGNORE INTO articles(pmcid,status) VALUES(?,'relevant')", (key,))
    conn.execute(
        "INSERT OR IGNORE INTO manifestation_candidates"
        "(disease_key,finding_key,pmcid,provenance_status,provenance_disease_key) "
        "VALUES('as',?,?,'explicit','as')",
        (finding, key),
    )


def eye_article(key):
    return {"pmcid": key, "title": "Ocular manifestations of ankylosing spondylitis",
            "caption_candidates": [{"caption": "Slit-lamp photographs show anterior uveitis "
                                                   "in a patient with ankylosing spondylitis.",
                                    "eligible": True}]}


def test_explicit_pair_image_beats_highly_cited_generic_review(conn):
    image = eye_article("image")
    generic = {"pmcid": "generic", "title": "Ankylosing spondylitis and cardiovascular therapy",
               "source_metadata": {"cited_by_count": 1000000, "publication_year": "2025"}}
    for row in (image, generic):
        seed(conn, row["pmcid"])
    pair_rank.rank_pairs(conn, [generic, image], "as", {"anterior_uveitis"}, persist=True)
    selected = manifestation_queue.reserve_batch(conn, "as", [generic, image], 1,
                                                  {"anterior_uveitis"}, persist=False)
    assert selected[0]["pmcid"] == "image"
    details = db.from_json(conn.execute("SELECT scoring_json FROM article_pair_rankings "
                                       "WHERE pmcid='image'").fetchone()[0], {})
    assert details["evidence_tier"] == 2
    assert details["semantic_status"] == "lexical_fallback"


def test_mixed_panels_and_ineligible_figures_do_not_claim_pair_attribution():
    mixed = {"caption": "(A) Photograph of anterior uveitis in sarcoidosis. "
                         "(B) Photograph of arthritis in ankylosing spondylitis.", "eligible": True}
    assert not pair_rank.caption_pair_evidence(mixed, "as", "anterior_uveitis")
    assert not pair_rank.caption_pair_evidence({**eye_article("x")["caption_candidates"][0],
                                               "eligible": False}, "as", "anterior_uveitis")


def test_semantic_cosine_changes_order_and_positive_embeddings_are_cached(conn):
    first = {"pmcid": "first", "title": "Review of ankylosing spondylitis inflammation"}
    second = {"pmcid": "second", "title": "Review of ankylosing spondylitis iris inflammation"}
    for row in (first, second):
        seed(conn, row["pmcid"])
    calls = []

    def embeddings(texts):
        calls.append(texts)
        return [[1, 0] if text.startswith("Clinical images") or "iris inflammation" in text
                else [0, 1] for text in texts]

    pair_rank.rank_pairs(conn, [first, second], "as", {"anterior_uveitis"},
                         semantic_rows=[first, second], embed_many=embeddings)
    assert second["pair_rankings"]["anterior_uveitis"]["cosine_similarity"] == 1
    selected = manifestation_queue.reserve_batch(conn, "as", [first, second], 1,
                                                  {"anterior_uveitis"}, persist=False)
    assert selected[0]["pmcid"] == "second"
    pair_rank.rank_pairs(conn, [first, second], "as", {"anterior_uveitis"},
                         semantic_rows=[first, second], embed_many=embeddings)
    assert len(calls) == 1


def test_failed_semantics_keep_caption_priority_and_record_fallback(conn):
    image = eye_article("image")
    seed(conn, "image")

    def broken(texts):
        raise RuntimeError("provider unavailable")

    pair_rank.rank_pairs(conn, [image], "as", {"anterior_uveitis"},
                         semantic_rows=[image], embed_many=broken)
    rank = image["pair_rankings"]["anterior_uveitis"]
    assert rank["evidence_tier"] == 2
    assert rank["semantic_fallback_reason"] == "embedding_unavailable_or_invalid"
    assert conn.execute("SELECT COUNT(*) FROM ranking_embeddings").fetchone()[0] == 0


def test_query_strings_are_not_article_evidence():
    article = {"title": "Cardiovascular mechanisms", "matched_passages": [
        {"query": "ankylosing spondylitis anterior uveitis", "text": "Drug effects on cardiac tissue"}]}
    assert pair_rank.score_pair(article, "as", "anterior_uveitis")["evidence_tier"] == 0


def test_caption_peek_reserves_other_lanes_under_bound(conn):
    generic = {"pmcid": "generic", "title": "Cardiovascular mechanisms", "retrieval_score": 99}
    eye = eye_article("eye")
    skin = {"pmcid": "skin", "title": "Ankylosing spondylitis and psoriasis"}
    for row, finding in ((generic, "anterior_uveitis"), (eye, "anterior_uveitis"), (skin, "psoriasis")):
        seed(conn, row["pmcid"], finding)
    selected = pair_rank.caption_shortlist(conn, [generic, eye, skin], "as",
                                           {"anterior_uveitis", "psoriasis"}, 2)
    assert {row["pmcid"] for row in selected} == {"eye", "skin"}


def test_epmc_adapter_preserves_identity_types_dates_and_retraction_without_link_id():
    metadata = source_quality.normalize_record({
        "id": "123", "source": "MED", "pubYear": "2020", "firstPublicationDate": "2020-09-12",
        "journalInfo": {"journal": {"title": "Journal", "issn": "1111-2222", "essn": "3333-4444"}},
        "citedByCount": 44, "pubTypeList": {"pubType": ["Review"]},
        "commentCorrectionList": {"commentCorrection": [{"type": "Retraction in", "source": "MED"}]},
    })
    assert metadata["journal_issn"] == "1111-2222"
    assert metadata["publication_types"] == ["Review"]
    assert metadata["cited_by_count"] == 44
    assert metadata["retraction_status"] == "retracted"


def test_secondary_signals_use_age_and_explicit_issn_preference(monkeypatch):
    monkeypatch.setattr(config, "VP_JOURNAL_PREFERENCES", '{"1111-2222": 0.75}')
    metadata = {"cited_by_count": 100, "publication_year": "2020", "journal_issn": "1111-2222",
                "publication_types": ["Review"]}
    signal = source_quality.quality_signal(metadata, today=date(2025, 1, 1))
    older = source_quality.quality_signal({**metadata, "publication_year": "2000"}, today=date(2025, 1, 1))
    assert signal["score"] > older["score"]
    assert signal["parts"]["journal_preference"] == 0.75
    assert signal["parts"]["article_type"] == 0.25
    assert "not field-normalized" in signal["limitation"]
    assert source_quality.quality_signal({"cited_by_count": 100})["citations_per_publication_year"] is None


def test_cached_retraction_survives_failed_refresh(conn, monkeypatch):
    seed(conn, "retracted")
    metadata = {"source": "europe_pmc_core", "status": "ok", "retraction_status": "retracted"}
    conn.execute("INSERT INTO article_source_metadata VALUES(?,?,?,?)",
                 ("retracted", "europe_pmc_core", db.to_json(metadata), "2020-01-01T00:00:00+00:00"))
    monkeypatch.setattr(source_quality, "fetch_metadata", lambda ids: {
        key: {"status": "fetch_failed"} for key in ids})
    row = {"pmcid": "retracted"}
    source_quality.enrich_shortlist(conn, [row])
    assert row["source_metadata"]["refresh_status"] == "fetch_failed"
    assert source_quality.quality_signal(row["source_metadata"])["retracted"]
    assert manifestation_queue.reserve_batch(conn, "as", [row], 1,
                                              {"anterior_uveitis"}, persist=False) == []


def test_missing_records_retry_and_successful_metadata_is_cached(conn, monkeypatch):
    seed(conn, "present")
    seed(conn, "missing")
    calls = []

    def fetch(ids):
        calls.append(ids)
        return {key: source_quality.normalize_record({"citedByCount": 5}) if key == "present"
                else {"status": "missing_record"} for key in ids}

    monkeypatch.setattr(source_quality, "fetch_metadata", fetch)
    rows = [{"pmcid": "present"}, {"pmcid": "missing"}]
    source_quality.enrich_shortlist(conn, rows)
    source_quality.enrich_shortlist(conn, rows)
    assert calls == [["present", "missing"], ["missing"]]


def test_fetch_uses_core_and_reports_transport_failure(monkeypatch):
    def failing(url, params):
        assert params["resultType"] == "core"
        assert params["query"] == "PMCID:(PMC1)"
        raise pmc.PmcError("timeout")

    monkeypatch.setattr(pmc, "_post", failing)
    assert source_quality.fetch_metadata(["PMC1"])["PMC1"]["status"] == "fetch_failed"


def test_invalid_embedding_not_treated_as_semantic():
    assert pair_rank.cosine([float("nan"), 1], [1, 0]) is None
    assert pair_rank.cosine([0, 0], [1, 0]) is None
    assert pair_rank.cosine([1], [1, 0]) is None


def test_parser_keeps_cached_retraction_out_of_global_fallback_without_peek(conn, monkeypatch):
    seed(conn, "bad")
    seed(conn, "good")
    conn.execute("UPDATE articles SET primary_disease_keys_json='[\"as\"]'")
    conn.execute("INSERT INTO article_source_metadata VALUES(?,?,?,?)",
                 ("bad", "europe_pmc_core", db.to_json({"retraction_status": "retracted", "status": "ok"}),
                  "2020-01-01T00:00:00+00:00"))
    monkeypatch.setattr(source_quality, "fetch_metadata", lambda ids: pytest.fail("No fetch allowed"))
    rows = parse.ranked_pending_articles(conn, "as", gaps={"anterior_uveitis": 0},
                                         peek_limit=0, persist=False)
    assert [row["pmcid"] for row in rows] == ["good"]
