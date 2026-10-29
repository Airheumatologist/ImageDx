"""Figure-first discovery: caption terms, Europe PMC queries, type gate."""

import httpx

from balanced_fixtures import add_disease, add_finding, make_db
from src.visual_pilot import (
    db, demographics, discover, diseases, europepmc, jats, pair_reporting, pair_terms, parse,
    pmc, publication,
)


def _finding(key):
    return next(f for f in diseases.load_findings_vocab() if f["finding_key"] == key)


def test_caption_terms_strip_disease_and_modality_context():
    assert "digital ulcer" in pair_terms.caption_terms("ssc", _finding("ssc_digital_ulcers"))
    terms = pair_terms.caption_terms("gout", _finding("gout_double_contour"))
    assert "double contour" in terms and "double contour sign" in terms
    assert "telangiectasia" in pair_terms.caption_terms("ssc", _finding("ssc_telangiectasias"))


def test_caption_terms_keep_phrase_when_only_an_adjective_remains():
    terms = pair_terms.caption_terms("psoriasis", _finding("psoriasis_inverse"))
    assert "inverse psoriasis" in terms
    assert "inverse" not in terms


def test_caption_terms_overrides_replace_generic_derivations():
    assert pair_terms.caption_terms("gout", _finding("gout_tophi"))[0] == "tophi"
    assert "galaxy" not in pair_terms.caption_terms("sarcoidosis", _finding("sarcoid_galaxy_sign"))


def test_caption_matches_phrase_and_words_modes():
    caption = "Erosions of both sacroiliac joints on CT."
    assert not pair_terms.caption_matches(caption, ["sacroiliac erosions"])
    assert pair_terms.caption_matches(caption, ["sacroiliac erosions"], mode="words")
    assert pair_terms.caption_matches("Multiple digital ulcers.", ["digital ulcer"])
    assert not pair_terms.caption_matches("Schematic of signaling", ["sign"], mode="words")


def test_caption_matches_ignore_punctuation_and_possessives():
    name = "periodic fever, aphthous stomatitis, pharyngitis"
    assert pair_terms.caption_matches(
        "Periodic fever, aphthous stomatitis, pharyngitis, and cervical adenitis", [name])
    assert pair_terms.caption_matches("periodic fever aphthous stomatitis pharyngitis", [name])
    assert pair_terms.caption_matches("Zenker diverticulum on barium swallow", ["Zenker's diverticulum"])
    assert pair_terms.caption_matches("Zenker's diverticulum", ["zenker diverticulum"])
    assert not pair_terms.caption_matches("periodic fever and pharyngitis", [name])


def test_build_query_modes():
    phrase = europepmc.build_query(["heliotrope rash"], ["dermatomyositis"])
    assert 'FIG:"heliotrope rash"' in phrase
    assert 'TITLE:"dermatomyositis" OR ABSTRACT:"dermatomyositis"' in phrase
    assert "OPEN_ACCESS:y" in phrase and 'LICENSE:"cc by"' in phrase
    words = europepmc.build_query(["sacroiliac erosion", "tophi"], ["gout"], mode="words")
    assert "FIG:(sacroiliac* AND erosion*)" in words
    assert "tophi" not in words.split(" AND (")[0]  # single words only in phrase mode
    assert europepmc.build_query(["tophi"], ["gout"], mode="words") == ""


def test_search_pages_with_cursor_and_parses_core_records():
    pages = {
        "*": {"hitCount": 3, "nextCursorMark": "c2", "resultList": {"result": [
            {"pmcid": "PMC1", "title": "Case", "license": "cc by", "pubYear": "2024",
             "pubTypeList": {"pubType": ["Case Reports", "Journal Article"]}},
            {"pmcid": "PMC2", "license": "cc by-nc", "pubTypeList": {"pubType": ["Review"]}},
        ]}},
        "c2": {"hitCount": 3, "nextCursorMark": "c2", "resultList": {"result": [
            {"pmcid": "PMC3", "license": "cc0", "pubTypeList": {"pubType": ["Published Erratum"]}},
        ]}},
    }

    def handler(request):
        return httpx.Response(200, json=pages[request.url.params["cursorMark"]])

    pmc.set_http_client(httpx.Client(transport=httpx.MockTransport(handler)))
    try:
        total, hits = europepmc.search("q", limit=10)
    finally:
        pmc.set_http_client(None)
    assert total == 3
    assert [h.pmcid for h in hits] == ["PMC1", "PMC2", "PMC3"]
    assert hits[0].year == 2024 and hits[0].license_code == "cc-by"
    assert [europepmc.eligible(h) for h in hits] == [True, False, False]


def test_publication_type_gate_admits_case_reports_but_not_notices():
    assert publication.passes_type_filter(["Case Reports", "Journal Article"])
    assert publication.passes_type_filter(["Research Support, Non-U.S. Gov't"], "research-article")
    assert publication.passes_type_filter(["Review"])
    assert not publication.passes_type_filter(["Published Erratum"])
    assert not publication.passes_type_filter(["Journal Article", "Retracted Publication"])
    assert not publication.passes_type_filter([], "correction")


def _hit(pmcid, pub_types=("Case Reports", "Journal Article"), abstract=""):
    return europepmc.Hit(
        pmcid=pmcid, pmid=None, doi=None, title="Malar rash in SLE", journal="J",
        year=2024, pub_types=list(pub_types), license_raw="cc by", license_code="cc-by",
        abstract=abstract, raw={"pmcid": pmcid, "pubTypeList": {"pubType": list(pub_types)}},
    )


def _parsed(captions, sections=()):
    figs = [
        jats.FigureInfo(fig_id=f"f{i}", label=f"Figure {i}", caption=c,
                        graphic_href=f"f{i}.jpg", permissions_text=None,
                        fig_license_raw=None, third_party=False, third_party_reason=None)
        for i, c in enumerate(captions, 1)
    ]
    return jats.ParsedArticle(
        figures=figs, body_sections=list(sections), authors=["Doe"], author_count=1,
        journal_name="J",
        corresp_country=None, first_aff_country=None,
    )


class _Bundle:
    metadata = {}

    @staticmethod
    def resolver(href):
        return pmc.ImageRef(url=f"{pmc.S3_BASE}/PMC1.1/{href}", format="jpg")


def test_discover_queues_only_caption_matched_figures(tmp_path, monkeypatch):
    conn = make_db(tmp_path)
    add_disease(conn, "sle", "Systemic lupus erythematosus")
    add_finding(conn, "malar_rash", ("sle",), label="Malar rash", synonyms=("butterfly rash",))
    conn.commit()
    queries = []

    def fake_search(query, limit=100, page_size=100):
        queries.append(query)
        return 1, [_hit("PMC1", abstract="We report a 34-year-old woman with SLE.")]

    parsed = _parsed(
        ["Butterfly rashes over both cheeks.", "Timeline of treatment."],
        sections=[("Case presentation", "She had arthralgia.")],
    )
    monkeypatch.setattr(europepmc, "search", fake_search)
    monkeypatch.setattr(discover, "_fetch", lambda pmcid: (_Bundle(), parsed))

    stats = discover.discover(conn, ["sle"], per_pair=5, target=10, log=lambda *_: None)

    assert stats["pmcids"] == ["PMC1"] and stats["pending"] == 1
    rows = {r["figure_id"]: dict(r) for r in conn.execute("SELECT * FROM figures")}
    assert rows["PMC1:f1"]["status"] == "pending"
    assert rows["PMC1:f2"]["status"] == "caption_rejected"
    assert "no_finding_term" in rows["PMC1:f2"]["triage_json"]
    assert "34-year-old" in rows["PMC1:f1"]["case_age_text"]
    assert demographics.resolve_age(rows["PMC1:f1"])["age_group"] == "adult"
    article = conn.execute("SELECT * FROM articles WHERE pmcid='PMC1'").fetchone()
    assert article["status"] == "parsed"
    assert db.from_json(article["primary_disease_keys_json"]) == ["sle"]
    # Phrase tier found too few articles, so the words tier also ran.
    attempts = conn.execute("SELECT * FROM pair_search_attempts").fetchall()
    assert {db.from_json(a["filters_json"])["mode"] for a in attempts} == {"phrase", "words"}
    funnel = pair_reporting.pair_funnel(conn)["sle"]["malar_rash"]
    assert funnel["retrieved_unique_articles"] == 1 and funnel["pending_work"] == 1

    # A second round skips the already-parsed article.
    stats = discover.discover(conn, ["sle"], per_pair=5, target=10, log=lambda *_: None)
    assert stats["pmcids"] == []
    conn.close()


def test_discover_holds_no_write_transaction_while_fetching(tmp_path, monkeypatch):
    # Search attempts were inserted before the article fetch, so the write
    # lock was held across network calls and the judge thread timed out.
    conn = make_db(tmp_path)
    add_disease(conn, "sle", "Systemic lupus erythematosus")
    add_finding(conn, "malar_rash", ("sle",), label="Malar rash", synonyms=("butterfly rash",))
    conn.commit()
    in_transaction = []

    def fake_fetch(pmcid):
        in_transaction.append(conn.in_transaction)
        return _Bundle(), _parsed(["Butterfly rash."])

    monkeypatch.setattr(europepmc, "search", lambda query, limit=100, page_size=100: (
        1, [_hit("PMC1", abstract="We report a woman with SLE.")]))
    monkeypatch.setattr(discover, "_fetch", fake_fetch)

    stats = discover.discover(conn, ["sle"], per_pair=5, target=10, log=lambda *_: None)

    assert stats["pmcids"] == ["PMC1"]
    assert in_transaction == [False]
    assert conn.execute("SELECT COUNT(*) FROM pair_search_attempts").fetchone()[0] > 0
    assert not conn.in_transaction
    conn.close()


def test_case_age_text_only_for_case_reports():
    parsed = _parsed([], sections=[("Case presentation", "A 9-year-old boy presented."),
                                   ("Discussion", "Adults aged 40 years differ.")])
    text = discover.case_age_text(_hit("PMC1", abstract=""), parsed)
    assert "9-year-old" in text and "40 years" not in text
    cohort = _hit("PMC2", pub_types=("research-article",), abstract="Patients aged 18 years or older.")
    assert discover.case_age_text(cohort, parsed) is None
    described = _hit("PMC3", pub_types=("brief-report",), abstract="We describe a 70-year-old man.")
    assert "70-year-old" in discover.case_age_text(described, None)


def test_resolve_age_prefers_caption_over_case_text():
    figure = {"caption": "Rash in a 10-year-old child.", "case_age_text": "A 45-year-old woman."}
    assert demographics.resolve_age(figure)["age_group"] == "child"
    assert demographics.resolve_age({"caption": "Rash.", "case_age_text": "A 45-year-old woman."})["age_group"] == "adult"
    conflicting = {"caption": "Rash.", "case_age_text": "A 45-year-old woman and her 8-year-old son."}
    assert demographics.resolve_age(conflicting)["age_group"] == "unknown"


def test_run_all_drains_leftover_figures_before_discovery(tmp_path, monkeypatch):
    from src.visual_pilot import cli, config

    monkeypatch.setattr(config, "data_dir", lambda: tmp_path)
    monkeypatch.setenv("VP_DATA_DIR", str(tmp_path))
    conn = make_db(tmp_path)
    add_disease(conn, "sle", "Systemic lupus erythematosus")
    conn.execute(
        "INSERT INTO articles(pmcid,status,primary_disease_keys_json) VALUES('PMC9','parsed','[\"sle\"]')"
    )
    conn.execute("INSERT INTO figures(figure_id,pmcid,status) VALUES('PMC9:f1','PMC9','caption_kept')")
    conn.commit()
    conn.close()
    monkeypatch.setattr(db, "connect", lambda path=None, _c=db.connect: _c(tmp_path / "visual_pilot.sqlite"))
    calls = []
    for name in ("triage", "judge", "store", "extract", "report"):
        monkeypatch.setitem(cli.COMMANDS, name, lambda a, n=name: calls.append((n, a.pmcids)) or 0)
    monkeypatch.setattr(cli, "_cmd_init", lambda a: 0)
    monkeypatch.setattr(discover, "discover", lambda *a, **k: calls.append(("discover", None)) or {"pmcids": []})

    assert cli.main(["run-all", "--disease", "sle"]) == 0
    assert calls[:4] == [("triage", ["PMC9"]), ("judge", ["PMC9"]), ("store", ["PMC9"]), ("discover", None)]


def test_article_tier_ranks_overview_sources_above_case_reports():
    from src.visual_pilot.source_quality import article_tier

    case = ["Journal Article", "Case Reports", "case-report"]
    assert article_tier(["Review"], "Atopic dermatitis: an update") == 0
    assert article_tier(["research-article"], "Skin findings in a cohort") == 1
    assert article_tier(case, "Case series of 12 patients with dermatomyositis") == 1
    assert article_tier(case, "Lebrikizumab for atopic dermatitis in the elderly: A case series.") == 3
    assert article_tier(case, "Erythrodermic atopic dermatitis in a child") == 2
    assert article_tier(case, "Lebrikizumab-induced psoriasis in a patient with atopic dermatitis.") == 3
    assert article_tier(["Review"], "Paradoxical psoriasis under TNF inhibitors") == 3


def test_find_articles_takes_reviews_before_case_reports(monkeypatch):
    def hit(pmcid, types, title):
        return europepmc.Hit(pmcid, None, None, title, None, 2024, types, "cc by", "cc-by")

    hits = [
        hit("PMC1", ["Case Reports"], "Drug-induced rash in lupus"),
        hit("PMC2", ["Case Reports"], "Malar rash in a young woman"),
        hit("PMC3", ["Review"], "Cutaneous lupus: a review"),
    ]
    monkeypatch.setattr(europepmc, "search", lambda query, limit: (len(hits), hits))
    _, picked, _ = discover.find_articles("sle", _finding("malar_rash"), 2, set())
    assert [h.pmcid for h, _ in picked] == ["PMC3", "PMC2"]


def test_article_tier_case_report_with_literature_review_is_not_a_review():
    from src.visual_pilot.source_quality import article_tier

    review = ["review-article", "Review", "Journal Article"]
    assert article_tier(review, "Skin Manifestations and Coeliac Disease in Paediatric Population") == 0
    assert article_tier(["Review", "Case Reports"], "Vitiligo in a child: a case report and literature review") == 2
    assert article_tier(review, "Atopic Dermatitis Treated with Dupilumab. A Case Report and Review") == 3
    assert article_tier(["research-article"], "Dupilumab in moderate atopic dermatitis: a phase 3 trial") == 1
    assert article_tier(["Case Reports"], "A new concept: plaque morphology in a case of psoriasis") == 2
    assert article_tier(["Case Reports"], "Upadacitinib in the treatment of SAPHO syndrome: a case report.") == 3


def test_review_scoped_and_overview_queries():
    q = europepmc.build_query(["dactylitis"], ["psoriatic arthritis"], scope="reviews")
    assert 'FIG:"dactylitis"' in q
    assert 'TITLE:"dactylitis" OR ABSTRACT:"dactylitis"' in q
    assert 'PUB_TYPE:"review"' in q and 'NOT PUB_TYPE:"systematic review"' in q
    assert 'PUB_TYPE:"review"' not in europepmc.build_query(["dactylitis"], ["psoriatic arthritis"])
    words = europepmc.build_query(["achilles enthesitis"], ["psoriatic arthritis"], "words", "reviews")
    assert "TITLE:(achille* AND enthesitis*)" in words
    overview = europepmc.build_overview_query(["psoriatic arthritis"], ["dactylitis", "nail pitting"])
    assert 'TITLE:"psoriatic arthritis"' in overview and "TITLE:manifestation*" in overview
    assert 'FIG:"dactylitis" OR FIG:"nail pitting"' in overview
    assert "ABSTRACT:" not in overview.split(" AND ")[0]
    assert europepmc.build_overview_query(["psoriatic arthritis"], []) == ""


def test_overview_pass_ingests_broad_reviews_and_skips_atypical(tmp_path, monkeypatch):
    conn = make_db(tmp_path)
    add_disease(conn, "sle", "Systemic lupus erythematosus")
    add_finding(conn, "malar_rash", ("sle",), label="Malar rash", synonyms=("butterfly rash",))
    conn.commit()
    review = _hit("PMC1", ("Review",))
    review.title = "Cutaneous manifestations of lupus"
    drug = _hit("PMC2", ("Review",))
    drug.title = "Drug-induced lupus: clinical features"
    queries = []

    def fake_search(query, limit=100, page_size=100):
        queries.append(query)
        return 2, [drug, review]

    monkeypatch.setattr(europepmc, "search", fake_search)
    monkeypatch.setattr(discover, "_fetch", lambda pmcid: (_Bundle(), _parsed(["Butterfly rash."])))

    stats = discover.discover(conn, ["sle"], per_pair=5, target=10, pass_name="overview",
                              log=lambda *_: None)

    assert stats["pmcids"] == ["PMC1"] and len(queries) == 1
    assert "TITLE:manifestation*" in queries[0]
    attempt = conn.execute("SELECT * FROM pair_search_attempts").fetchone()
    assert attempt["finding_key"] == discover.OVERVIEW_KEY
    evidence = db.from_json(conn.execute(
        "SELECT retrieval_evidence_json FROM articles WHERE pmcid='PMC1'").fetchone()[0])
    assert evidence[0]["pass"] == "overview"

    # Backfill still takes the atypical source when the pair needs it.
    stats = discover.discover(conn, ["sle"], per_pair=5, target=10, log=lambda *_: None)
    assert stats["pmcids"] == ["PMC2"]
    conn.close()


def test_run_all_runs_review_passes_before_backfill(tmp_path, monkeypatch):
    from src.visual_pilot import cli, config

    monkeypatch.setattr(config, "data_dir", lambda: tmp_path)
    monkeypatch.setenv("VP_DATA_DIR", str(tmp_path))
    conn = make_db(tmp_path)
    add_disease(conn, "sle", "Systemic lupus erythematosus")
    conn.close()
    monkeypatch.setattr(db, "connect", lambda path=None, _c=db.connect: _c(tmp_path / "visual_pilot.sqlite"))
    calls = []
    for name in ("triage", "judge", "store", "extract", "report"):
        monkeypatch.setitem(cli.COMMANDS, name, lambda a, n=name: calls.append(n) or 0)
    monkeypatch.setattr(cli, "_cmd_init", lambda a: 0)
    found = {"overview": ["PMC1"], "manifestation": [], "disease": [], "backfill": []}

    def fake_discover(*a, pass_name, **k):
        calls.append(pass_name)
        return {"pmcids": found[pass_name]}

    monkeypatch.setattr(discover, "discover", fake_discover)

    assert cli.main(["run-all", "--disease", "sle"]) == 0
    # Overview articles are judged before the manifestation pass re-plans;
    # an empty review pass does not stop backfill.
    assert calls[:7] == ["overview", "triage", "judge", "store", "manifestation", "disease",
                         "backfill"]


def test_overview_terms_drop_abbreviations():
    terms = discover.overview_disease_terms(diseases.load_diseases()["psa"])
    assert "PsA" not in terms and "psoriatic arthritis" in [t.lower() for t in terms]
    assert discover.overview_disease_terms(diseases.load_diseases()["gout"])[0] == "Gout"


def test_article_disease_keys_match_title_and_abstract():
    hit = _hit("PMC1", abstract="Systemic lupus erythematosus overlapping psoriatic arthritis.")
    keys = discover.article_disease_keys(hit, list(diseases.DISEASE_KEYS))
    assert "sle" in keys and "psa" in keys
    assert "gout" not in keys
    assert discover.article_disease_keys(hit, ["gout"]) == []


def test_article_disease_keys_read_spelled_out_names():
    hit = _hit("PMC1", abstract="Periodic Fever, Aphthous Stomatitis, Pharyngitis, and "
                                "Cervical Adenitis in an adult.")
    assert discover.article_disease_keys(hit, ["pfapa_syndrome", "gout"]) == ["pfapa_syndrome"]


def test_acronym_named_topics_keep_a_spelled_out_search_name():
    # The overview and disease passes drop bare abbreviations but keep one
    # joined to a spelled-out word, so every topic has a title search name.
    catalog = diseases.load_diseases()
    missing = {key for key, d in catalog.items() if not discover.overview_disease_terms(d)}
    assert missing == set()
    assert discover.overview_disease_terms(catalog["pfapa_syndrome"]) == [
        "PFAPA Syndrome", "periodic fever, aphthous stomatitis, pharyngitis"]
    assert discover.overview_disease_terms(catalog["kbg_syndrome"]) == ["KBG Syndrome"]


def test_gene_named_topics_search_their_only_name_in_use():
    catalog = diseases.load_diseases()
    foxn1 = discover.overview_disease_terms(catalog["foxn1_deficiency_nude_scid"])
    assert foxn1[0] == "FOXN1 deficiency" and "nude SCID" in foxn1
    assert not any("(" in t for t in foxn1)  # the gloss form is never a title
    assert "CARD9 deficiency" in discover.overview_disease_terms(catalog["card9_deficiency"])
    assert "MHC class I deficiency" in discover.overview_disease_terms(
        catalog["mhc_class_i_deficiency_tap1_tap2"])
    # Bare abbreviations still go: "PsA" is also prostate-specific antigen.
    assert "PsA" not in discover.overview_disease_terms(catalog["psa"])


def test_mesh_hits_must_be_indexed_under_the_disease():
    def hit(headings, keywords=()):
        return europepmc.Hit(
            pmcid="PMC1", pmid=None, doi=None, title="t", journal=None, year=None,
            pub_types=[], license_raw=None, license_code="cc-by",
            raw={"meshHeadingList": {"meshHeading": [{"descriptorName": h} for h in headings]},
                 "keywordList": {"keyword": list(keywords)}},
        )

    terms = ["Laryngeal cleft", "laryngotracheoesophageal cleft"]
    # MESH:"Laryngeal cleft" is not a descriptor; Europe PMC answers with
    # cleft lip and palate articles, which are indexed under those instead.
    assert not europepmc.indexed_under(hit(["Cleft Palate", "Cleft Lip"]), terms)
    assert europepmc.indexed_under(hit(["Humans"], keywords=["laryngeal cleft"]), terms)
    assert europepmc.indexed_under(hit(["Larynx", "Laryngeal cleft"]), terms)
    # A multi-word heading that is part of the disease name counts.
    assert europepmc.indexed_under(
        hit(["Vitamin D-Dependent Rickets"]), ["vitamin D-dependent rickets type 2"])
    # A one-word parent heading does not: "Rickets" covers every rickets.
    assert not europepmc.indexed_under(hit(["Rickets"]), ["Hereditary vitamin D-resistant rickets"])
    assert not europepmc.indexed_under(hit([]), terms)
    # MeSH stores many descriptors inverted; word order does not matter.
    assert europepmc.indexed_under(hit(["Lupus Erythematosus, Systemic"]), ["systemic lupus erythematosus"])
    assert europepmc.indexed_under(hit(["Lymphoma, Large-Cell, Anaplastic"]), ["anaplastic large cell lymphoma"])
    # A one-word term is a whole heading, not a word in one ("Lupus Nephritis").
    assert not europepmc.indexed_under(hit(["Lupus Nephritis"]), ["lupus"])
    assert europepmc.indexed_under(hit(["Gout"]), ["gout"])


def test_name_variants_strip_parenthetical_gloss_and_fold_accents():
    assert diseases.name_variants("Loiasis (Loa Loa Filariasis)") == ["Loiasis"]
    assert diseases.name_variants("Ménétrier Disease (Hypertrophic Gastropathy)") == [
        "Ménétrier Disease", "Menetrier Disease",
    ]
    assert diseases.name_variants("Klinefelter Syndrome") == []
    assert pair_terms._caption_norm("Romaña sign") == "romana sign"


def test_disease_query_terms_include_bare_topic_name():
    disease = {"name": "Loiasis (Loa Loa Filariasis)",
               "synonyms": diseases.name_variants("Loiasis (Loa Loa Filariasis)")}
    assert "Loiasis" in discover.disease_query_terms(disease)
    query = europepmc.build_disease_query(["Loiasis"])
    assert query.startswith('(TITLE:"Loiasis") AND OPEN_ACCESS:y') and "FIG:" not in query


def test_disease_pass_searches_thin_topics_and_gates_on_citing_text(tmp_path, monkeypatch):
    conn = make_db(tmp_path)
    add_disease(conn, "sle", "Systemic lupus erythematosus")
    add_disease(conn, "gout", "Gout")
    add_finding(conn, "malar_rash", ("sle",), label="Malar rash", synonyms=("butterfly rash",))
    add_finding(conn, "gout_tophi", ("gout",), label="Tophi")
    conn.commit()
    monkeypatch.setattr(discover, "topic_image_counts", lambda c: {"gout": 25})
    queries = []

    def fake_search(query, limit=100, page_size=100):
        queries.append(query)
        return 1, [_hit("PMC1")]

    parsed = _parsed(["Figure 1. Clinical photograph.", "Timeline of treatment.",
                      "Chest CT on admission.", "Western blot of patient fibroblasts.",
                      "Erythematous rash on the trunk.", "Prevalence of rash by age group."])
    parsed.figures[0].in_text_mentions = ["Figure 1 shows a malar rash sparing the folds."]
    monkeypatch.setattr(europepmc, "search", fake_search)
    monkeypatch.setattr(discover, "_fetch", lambda pmcid: (_Bundle(), parsed))

    stats = discover.discover(conn, ["sle", "gout"], per_pair=5, target=10,
                              pass_name="disease", log=lambda *_: None)

    # Only the topic under the floor is searched, by name and never by
    # caption: case reports by title, then by abstract, then any article by
    # title; then the MeSH and (for a zero-image topic) manifestation tiers.
    assert len(queries) == 5 and all("systemic lupus" in q.lower() and "FIG:" not in q for q in queries)
    assert 'PUB_TYPE:"case reports"' in queries[0] and "ABSTRACT:" not in queries[0]
    assert 'PUB_TYPE:"case reports"' in queries[1] and 'ABSTRACT:"Systemic lupus' in queries[1]
    assert "PUB_TYPE" not in queries[2] and "ABSTRACT:" not in queries[2]
    assert queries[3].startswith("((MESH:")
    assert queries[4].startswith("(TITLE:") and "ABSTRACT:" in queries[4]
    assert all(q.startswith("(TITLE:") for q in queries[:3])
    status = dict(conn.execute("SELECT figure_id, status FROM figures").fetchall())
    # A caption naming a patient image type or a lesion passes without a
    # finding term; lab work and charts do not.
    assert status == {"PMC1:f1": "pending", "PMC1:f2": "caption_rejected",
                      "PMC1:f3": "pending", "PMC1:f4": "caption_rejected",
                      "PMC1:f5": "pending", "PMC1:f6": "caption_rejected"}
    attempt = conn.execute("SELECT finding_key FROM pair_search_attempts").fetchone()
    assert attempt[0] == discover.DISEASE_KEY and stats["pairs"] == 0
    assert stats["pmcids"] == ["PMC1"]
    conn.close()


def test_names_patient_image_accepts_lesions_sites_and_the_disease_name():
    assert discover.names_patient_image("Erythematous rash on the trunk")
    assert discover.names_patient_image("Polypoid nodule protruding from the umbilicus")
    assert discover.names_patient_image("The patient's genitalia, day 3.")
    assert not discover.names_patient_image("Follow-up on day 3.")
    assert discover.names_patient_image("Napkin rash, day 3.", ["napkin rash"])
    assert discover.names_patient_image("Umbilical granuloma at presentation", ["umbilical granuloma"])
    assert not discover.names_patient_image("Timeline of treatment", ["umbilical granuloma"])
    assert not discover.names_patient_image("Incidence of umbilical granuloma by year",
                                            ["umbilical granuloma"])


def test_disease_pass_ranks_case_reports_before_research_and_reviews():
    review, study, case = _hit("PMC1", ("Review",)), _hit("PMC2", ("Journal Article",)), \
        _hit("PMC3", ("Journal Article",))
    review.title, study.title = "Klinefelter syndrome: an update", "Bone density in 47,XXY men"
    case.title = "A rare case of Klinefelter syndrome with gynecomastia"
    ranked = sorted([review, study, case], key=discover.disease_pass_rank)
    assert [h.pmcid for h in ranked] == ["PMC3", "PMC2", "PMC1"]


def test_requeue_terms_and_errors(tmp_path, monkeypatch):
    conn = make_db(tmp_path)
    add_disease(conn, "sle", "Systemic lupus erythematosus")
    add_finding(conn, "malar_rash", ("sle",), label="Malar rash", synonyms=("butterfly rash",))
    conn.execute(
        "INSERT INTO articles (pmcid, primary_disease_keys_json, status) "
        "VALUES ('PMC1', '[\"sle\"]', 'parsed')"
    )
    conn.execute(
        "INSERT INTO articles (pmcid, primary_disease_keys_json, retrieval_evidence_json, status) "
        "VALUES ('PMC2', '[\"sle\"]', '[{\"pass\":\"disease\"}]', 'parsed')"
    )
    drop = '{"route":"drop","reason":"no_finding_term"}'
    for fid, pmcid, caption in (("PMC1:f5", "PMC1", "Brain MRI on admission."),
                                ("PMC2:f1", "PMC2", "Brain MRI on admission."),
                                ("PMC2:f2", "PMC2", "Survival curve of the cohort.")):
        conn.execute(
            "INSERT INTO figures (figure_id, pmcid, caption, status, triage_json, attempts) "
            "VALUES (?, ?, ?, 'caption_rejected', ?, 3)", (fid, pmcid, caption, drop),
        )
    rows = [
        ("PMC1:f1", "Butterfly rash on the cheeks.", "caption_rejected",
         '{"route":"drop","reason":"no_finding_term"}', None),
        ("PMC1:f2", "Timeline.", "caption_rejected", '{"route":"drop","reason":"no_finding_term"}', None),
        ("PMC1:f3", "x", "vision_error", None, "llm: Request timed out."),
        ("PMC1:f4", "x", "vision_error", None, "llm: Error code: 451 - refused"),
    ]
    for fid, caption, status, triage, error in rows:
        conn.execute(
            "INSERT INTO figures (figure_id, pmcid, caption, status, triage_json, error, attempts) "
            "VALUES (?, 'PMC1', ?, ?, ?, ?, 3)", (fid, caption, status, triage, error),
        )
    conn.commit()

    assert discover.requeue_terms(conn, ["sle"]) == 2
    assert discover.requeue_errors(conn, ["sle"]) == 1
    got = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT figure_id, status, attempts FROM figures")}
    assert got["PMC1:f1"] == ("pending", 0) and got["PMC1:f2"] == ("caption_rejected", 3)
    # The image-type gate applies to disease-pass articles only.
    assert got["PMC2:f1"] == ("pending", 0) and got["PMC1:f5"][0] == "caption_rejected"
    assert got["PMC2:f2"][0] == "caption_rejected"
    assert got["PMC1:f3"] == ("vision_error", 0) and got["PMC1:f4"] == ("vision_error", 3)
    # --all-passes extends the image-type gate to non-disease-pass articles.
    assert discover.requeue_terms(conn, ["sle"], all_passes=True) == 1
    assert conn.execute(
        "SELECT status FROM figures WHERE figure_id='PMC1:f5'"
    ).fetchone()[0] == "pending"
    conn.close()


def test_figure_row_uses_jats_license_only_when_epmc_is_missing():
    fig = jats.FigureInfo(
        fig_id="f1", label="Figure 1", caption="Clinical photograph of the rash",
        graphic_href="f1.jpg", permissions_text=None, fig_license_raw=None,
        third_party=False, third_party_reason=None,
    )
    row = parse._figure_row(
        "PMC1", fig, "none", _Bundle.resolver, article_license_raw="CC BY 4.0",
    )
    assert row["effective_license"] == "cc-by" and row["status"] == "pending"
    # A specific non-commercial Europe PMC code is never upgraded by JATS.
    row = parse._figure_row(
        "PMC1", fig, "cc-by-nc", _Bundle.resolver, article_license_raw="CC BY 4.0",
    )
    assert row["effective_license"] == "cc-by-nc" and row["status"] == "caption_rejected"


def test_parse_article_reads_the_article_level_license():
    xml = """
    <article xmlns:xlink="http://www.w3.org/1999/xlink">
      <front><article-meta>
        <permissions>
          <license xlink:href="https://creativecommons.org/licenses/by/4.0/">
            <license-p>CC BY 4.0</license-p>
          </license>
        </permissions>
      </article-meta></front>
      <body><fig id="f1"><label>Figure 1</label>
        <caption><p>Clinical photograph of the rash.</p></caption>
        <graphic xlink:href="f1.jpg"/></fig></body>
    </article>
    """
    parsed = jats.parse_article(xml)
    assert parsed.article_license_raw == "https://creativecommons.org/licenses/by/4.0/"
    assert pmc.normalize_license(parsed.article_license_raw) == "cc-by"


def test_mesh_and_relaxed_queries_drop_the_license_clause():
    mesh = europepmc.build_mesh_query(["Kawasaki disease"])
    assert mesh.startswith('((MESH:"Kawasaki disease"))') and 'LICENSE:"cc by"' in mesh
    relaxed = europepmc.build_disease_query(
        ["Kawasaki disease"], fields=("TITLE",), license_clause=False,
    )
    assert "OPEN_ACCESS:y" in relaxed and "LICENSE:" not in relaxed


def test_pmc_client_keeps_a_connection_per_fetch_worker(monkeypatch):
    from src.visual_pilot import pmc

    monkeypatch.setattr(pmc.config, "VP_FETCH_CONCURRENCY", 64)
    monkeypatch.setattr(pmc, "_client", None)
    try:
        pool = pmc.http_client()._transport._pool
        assert pool._max_keepalive_connections == 64
        assert pool._max_connections == 128
    finally:
        pmc.set_http_client(None)


def test_mesh_query_parenthesizes_each_term():
    # Europe PMC returns only the last term's hits for MESH:"a" OR MESH:"b".
    mesh = europepmc.build_mesh_query(["Seborrheic Dermatitis", "dandruff"])
    assert mesh.startswith('((MESH:"Seborrheic Dermatitis") OR (MESH:"dandruff"))')
