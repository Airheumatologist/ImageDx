"""Full topic catalog and run throughput: catalog, scoped prompts, discovery, run-all."""

import io
import re
import threading

from balanced_fixtures import add_disease, add_finding, make_db
from src.visual_pilot import (
    cli, config, curation, db, discover, diseases, europepmc, llm, pmc, prompts,
    topic_vocab, triage,
)


def _hit(pmcid):
    return europepmc.Hit(
        pmcid=pmcid, pmid=None, doi=None, title="Malar rash in SLE", journal="J",
        year=2024, pub_types=["Review"], license_raw="cc by", license_code="cc-by",
        abstract="", raw={"pmcid": pmcid, "pubTypeList": {"pubType": ["Review"]}},
    )


# --- catalog -----------------------------------------------------------------

def test_topics_catalog_keeps_pilot_keys_and_adds_every_topic(monkeypatch):
    monkeypatch.setenv("VP_CATALOG", "topics")
    catalog = diseases.load_diseases()
    topics = diseases.load_topics()
    assert len(topics) == len({t["topic_id"] for t in topics})
    for topic in topics:
        assert diseases.topic_disease_key(topic) in catalog
    # Pilot topics keep their pilot key and curated subtypes.
    assert "systemic_lupus_erythematosus" not in catalog
    assert catalog["sle"]["subtypes"] and catalog["sle"]["specialty"] == "Rheumatology"
    # Pilot diseases outside the index stay in the catalog.
    assert "ad" in catalog
    assert catalog["vitiligo"]["subtypes"] == []


def test_pilot_catalog_is_the_curated_file(monkeypatch):
    monkeypatch.setenv("VP_CATALOG", "pilot")
    assert set(diseases.load_diseases()) == {
        "sle", "dm", "as", "ra", "ssc", "psoriasis", "psa", "sarcoidosis", "gout", "ad",
    }
    assert all(f["disease_keys"][0] in diseases.load_diseases() for f in diseases.load_findings_vocab())


def test_topic_vocab_rows_validate_against_the_catalog(monkeypatch):
    monkeypatch.setenv("VP_CATALOG", "topics")
    catalog = set(diseases.load_diseases())
    vocab = diseases.load_findings_vocab()
    assert len({f["finding_key"] for f in vocab}) == len(vocab)
    for item in vocab:
        diseases._validate_vocab_item(item, catalog)


def test_generated_caption_terms_feed_discovery(monkeypatch):
    finding = {"finding_key": "x_intimal_flap", "label": "Intimal flap on CT angiography",
               "synonyms": [], "caption_terms": ["intimal flap", "double barrel aorta"]}
    assert diseases.is_acronym("ITP") and diseases.is_acronym("MEN2A")
    assert not diseases.is_acronym("Gout") and not diseases.is_acronym("anti-GBM")
    from src.visual_pilot import pair_terms
    assert pair_terms.caption_terms("gout", finding) == ["intimal flap", "double barrel aorta"]


def test_topic_vocab_post_validate_drops_generic_terms():
    topic = {"topic_id": "aortic_dissection", "name": "Acute Aortic Dissection",
             "synonyms": ["dissecting aortic aneurysm"]}
    parsed = {"findings": [
        {"key": "Intimal flap", "label": "Intimal flap on CT", "category": "ct",
         "synonyms": ["dissection flap"],
         "caption_terms": ["Intimal flap", "flap", "mass", "contrast-enhanced CT", "T2-weighted MRI"]},
        {"key": "rash", "label": "Rash", "category": "skin", "synonyms": [],
         "caption_terms": ["rash", "acute aortic dissection"]},
        {"key": "x", "label": "Bad category", "category": "photo", "synonyms": [],
         "caption_terms": ["something specific"]},
    ]}
    rows = topic_vocab.post_validate(topic, parsed)
    assert [r["finding_key"] for r in rows] == ["aortic_dissection_intimal_flap"]
    assert rows[0]["caption_terms"] == ["intimal flap"]
    assert rows[0]["disease_keys"] == ["aortic_dissection"]
    assert rows[0]["source"] == "llm_topic_vocab"



def test_expanded_terms_put_plain_names_first_and_skip_taken_or_generic():
    topic = {"topic_id": "klinefelter_syndrome", "name": "Klinefelter Syndrome",
             "synonyms": ["47,XXY syndrome"]}
    rows = [
        {"finding_key": "ks_gynecomastia", "label": "Bilateral gynecomastia",
         "caption_terms": ["bilateral gynecomastia"]},
        {"finding_key": "ks_small_testes", "label": "Small testes",
         "caption_terms": ["small atrophic testes"]},
    ]
    parsed = {"findings": [
        {"key": "ks_gynecomastia",
         "caption_terms": ["Gynecomastia", "gynaecomastia", "swelling", "klinefelter syndrome"]},
        {"key": "ks_small_testes", "caption_terms": ["gynecomastia", "small testes"]},
        {"key": "unknown", "caption_terms": ["anything"]},
    ]}
    assert topic_vocab.merge_expanded_terms(topic, rows, parsed) == 3
    assert rows[0]["caption_terms"] == ["gynecomastia", "gynaecomastia", "bilateral gynecomastia"]
    assert rows[1]["caption_terms"] == ["small testes", "small atrophic testes"]
    assert rows[0]["caption_terms_version"] == topic_vocab.P7.version

# --- scoped prompts ----------------------------------------------------------

def test_scoped_prompts_list_only_the_call_diseases():
    p3 = prompts.scoped("p3", ["gout", "sle"])
    assert "sle (Systemic lupus erythematosus" in p3.system and "gout (Gout" in p3.system
    assert "dm (Dermatomyositis" not in p3.system
    enum = p3.schema["properties"]["panels"]["items"]["properties"]["disease_key"]["enum"]
    assert enum == ["sle", "gout", None] or enum == ["gout", "sle", None]
    p2 = prompts.scoped("p2", ["gout"])
    items = p2.schema["properties"]["results"]["items"]["properties"]["diseases_mentioned"]["items"]
    assert items["enum"] == ["gout", "other"]
    assert p2.version == prompts.P2.version
    # Nothing known: the full prompt.
    assert prompts.scoped("p4", ["not_a_disease"]) is prompts.P4


def test_triage_batches_group_articles_by_disease():
    rows = [
        {"figure_id": f"f{i}", "pmcid": f"PMC{i}", "_article_disease_keys": [key],
         "label": "Figure 1", "caption": "Tophi", "in_text_mentions_json": "[]"}
        for i, key in enumerate(["sle", "gout", "sle", "gout", "sle"])
    ]
    batches = triage._batches(rows, 3)
    assert [[r["_article_disease_keys"][0] for r in b] for b in batches] == [
        ["gout", "gout", "sle"], ["sle", "sle"],
    ]
    request = triage._p2_request(batches[1])
    assert "gout" not in request["system"].split(" Prefer ")[0]


def test_reasoning_effort_joins_cache_key_only_when_set():
    client = llm.LLMClient(db_conn=None)
    base = client._input_hash("p2", "m", "v", "sys", "user", [])
    assert client._input_hash("p2", "m", "v", "sys", "user", [], "") == base
    assert client._input_hash("p2", "m", "v", "sys", "user", [], "low") != base


# --- disease matching ----------------------------------------------------------

def test_acronym_synonyms_match_case_sensitively():
    pattern = curation._term_pattern("ALL")
    assert pattern.search("Bone marrow in ALL")
    assert not pattern.search("all panels show the rash")
    assert curation._term_pattern("Gout").search("gouty tophus in gout")


def test_disease_index_matches_the_brute_force_search():
    texts = [
        "Malar rash in systemic lupus erythematosus and psoriatic arthritis.",
        "Adult-onset Still's disease with salmon-pink rash; ITP excluded.",
        "Vitiligo and lichen planus on the same patient, all panels.",
        "Tophaceous gout of the hand",
    ]
    for text in texts:
        brute = {
            key for key, patterns in curation._disease_matchers().items()
            if any(p.search(text) for p in patterns)
        }
        assert curation._diseases_in_text(text) == brute


# --- discovery ---------------------------------------------------------------

def _pair_db(tmp_path):
    conn = make_db(tmp_path)
    add_disease(conn, "sle", "Systemic lupus erythematosus")
    add_finding(conn, "malar_rash", ("sle",), label="Malar rash", synonyms=("butterfly rash",))
    add_finding(conn, "discoid_plaque", ("sle",), label="Discoid plaque")
    conn.commit()
    return conn


def test_discover_streams_articles_and_retries_failed_searches(tmp_path, monkeypatch):
    conn = _pair_db(tmp_path)
    monkeypatch.setattr(config, "VP_SEARCH_RETRY_COOLDOWN", 0)
    calls = {"n": 0}
    lock = threading.Lock()

    def flaky_search(query, limit=100, page_size=100):
        with lock:
            calls["n"] += 1
            first = calls["n"] == 1
        if first:
            raise pmc.PmcError("giving up on europepmc")
        pmcid = "PMC2" if "discoid" in query else "PMC1"
        return 1, [_hit(pmcid)]

    from test_discover import _Bundle, _parsed
    monkeypatch.setattr(europepmc, "search", flaky_search)
    monkeypatch.setattr(
        discover, "_fetch",
        lambda pmcid: (_Bundle(), _parsed(["Butterfly rash.", "Discoid plaque on the scalp."])),
    )
    streamed = []
    logs = []
    stats = discover.discover(conn, ["sle"], per_pair=5, target=10,
                              log=logs.append, on_articles=streamed.append)
    assert sorted(p for batch in streamed for p in batch) == sorted(stats["pmcids"]) == ["PMC1", "PMC2"]
    assert any("retrying 1 failed search" in line for line in logs)
    assert stats.get("search_failures") == 0
    conn.close()


def test_discover_stops_when_asked(tmp_path, monkeypatch):
    conn = _pair_db(tmp_path)
    from test_discover import _Bundle, _parsed
    monkeypatch.setattr(europepmc, "search", lambda q, limit=100, page_size=100: (1, [_hit("PMC1")]))
    monkeypatch.setattr(discover, "_fetch", lambda pmcid: (_Bundle(), _parsed(["Butterfly rash."])))
    stats = discover.discover(conn, ["sle"], per_pair=5, target=10, log=lambda *_: None,
                              should_stop=lambda: True)
    assert stats["pmcids"] == []
    conn.close()


# --- run-all -----------------------------------------------------------------

def test_timestamped_stream_prefixes_whole_lines():
    out = io.StringIO()
    stream = cli._TimestampedStream(out)
    stream.write("first part ")
    stream.write("of a line\nsecond\n")
    lines = out.getvalue().splitlines()
    assert len(lines) == 2
    assert re.match(r"\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\] first part of a line$", lines[0])


def _run_all_env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "data_dir", lambda: tmp_path)
    monkeypatch.setenv("VP_DATA_DIR", str(tmp_path))
    conn = make_db(tmp_path)
    add_disease(conn, "sle", "Systemic lupus erythematosus")
    conn.close()
    monkeypatch.setattr(db, "connect", lambda path=None, _c=db.connect: _c(tmp_path / "visual_pilot.sqlite"))
    monkeypatch.setattr(cli, "_cmd_init", lambda a: 0)
    monkeypatch.setattr(cli, "_FILL_WAIT_SECONDS", 0.05)


def test_run_all_triages_while_discovery_streams(tmp_path, monkeypatch):
    _run_all_env(tmp_path, monkeypatch)
    events = []
    lock = threading.Lock()

    def record(name):
        def fn(args):
            with lock:
                events.append((name, tuple(args.pmcids or ())))
            return 0
        return fn

    for name in ("triage", "judge", "store", "describe", "extract", "report"):
        monkeypatch.setitem(cli.COMMANDS, name, record(name))
    triaged_first = threading.Event()

    def fake_discover(conn, keys, *, pass_name, on_articles=None, should_stop=None, **kw):
        if pass_name != "overview":
            return {"pmcids": []}
        on_articles(["PMC1"])
        # The first batch is triaged before discovery finishes.
        assert triaged_first.wait(5)
        on_articles(["PMC2"])
        return {"pmcids": ["PMC1", "PMC2"]}

    original = cli.COMMANDS["triage"]

    def triage_cmd(args):
        rc = original(args)
        triaged_first.set()
        return rc

    monkeypatch.setitem(cli.COMMANDS, "triage", triage_cmd)
    monkeypatch.setattr(discover, "discover", fake_discover)
    assert cli.main(["run-all", "--disease", "sle", "--batch-size", "1"]) == 0
    triaged = [p for name, p in events if name == "triage"]
    judged = [p for name, p in events if name == "judge"]
    assert triaged == [("PMC1",), ("PMC2",)]
    assert judged == [("PMC1",), ("PMC2",)]
    assert [name for name, _ in events][-2:] == ["extract", "report"]


def test_run_all_stops_inside_a_round_at_the_runtime_limit(tmp_path, monkeypatch):
    _run_all_env(tmp_path, monkeypatch)
    judged = []
    for name in ("triage", "store", "describe", "extract", "report"):
        monkeypatch.setitem(cli.COMMANDS, name, lambda a: 0)

    def slow_judge(args):
        judged.extend(args.pmcids)
        clock["now"] += 100  # each batch uses up the runtime
        return 0

    monkeypatch.setitem(cli.COMMANDS, "judge", slow_judge)
    clock = {"now": 1000.0}
    monkeypatch.setattr(cli.time, "monotonic", lambda: clock["now"])

    def fake_discover(conn, keys, *, pass_name, on_articles=None, should_stop=None, **kw):
        for pmcid in ("PMC1", "PMC2", "PMC3"):
            if should_stop():
                break
            on_articles([pmcid])
        return {"pmcids": ["PMC1", "PMC2", "PMC3"]}

    monkeypatch.setattr(discover, "discover", fake_discover)
    cli.main(["run-all", "--disease", "sle", "--batch-size", "1", "--max-runtime-seconds", "50"])
    assert judged == ["PMC1"]
