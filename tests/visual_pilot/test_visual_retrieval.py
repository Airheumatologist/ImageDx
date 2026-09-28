"""Visual query selection and passage evidence tests without network calls."""

from types import SimpleNamespace

from src.visual_pilot import diseases, retrieval, select_articles


def test_visual_passage_gate_requires_image_or_clinical_context():
    assert not select_articles._is_visual_passage({
        "page_content": "Mitochondrial signaling contributes to lupus nephritis pathogenesis.",
        "section_title": "Molecular mechanisms",
    })
    assert select_articles._is_visual_passage({
        "page_content": "Biopsy demonstrates a wire-loop lesion in lupus nephritis.",
        "section_title": "Renal findings",
    })


def test_visual_query_generation_is_bounded_and_covers_modalities():
    data = diseases.load_diseases()
    queries = select_articles.visual_queries_for_disease("as", data, max_queries=8)

    assert len(queries) == 8
    assert len({item["query"] for item in queries}) == len(queries)
    assert any("MRI" in item["query"] for item in queries)
    assert any("radiograph" in item["query"] for item in queries)
    assert all("Ankylosing spondylitis" in item["query"] for item in queries)
    assert select_articles.visual_queries_for_disease("invalid", data) == []


def test_visual_queries_default_to_every_approved_disease_finding(monkeypatch):
    data = diseases.load_diseases()
    findings = [
        {
            "finding_key": f"finding_{i}",
            "label": f"Manifestation {i}",
            "disease_keys": ["psoriasis"],
            "category": "skin",
            "approved": True,
        }
        for i in range(25)
    ]
    monkeypatch.setattr(select_articles.config, "VP_VISUAL_QUERY_CAP", 0)

    queries = select_articles.visual_queries_for_disease(
        "psoriasis", data, findings=findings
    )

    assert len(queries) == len(findings)
    assert select_articles.visual_queries_for_disease(
        "psoriasis", data, findings=findings, max_queries=0
    ) == []


def test_visual_query_includes_manifestation_and_organ_system_terms():
    data = diseases.load_diseases()
    findings = [{
        "finding_key": "psoriasis_uveitis",
        "label": "Anterior uveitis",
        "synonyms": ["iritis", "intraocular inflammation"],
        "disease_keys": ["psoriasis"],
        "category": "eye",
        "approved": True,
    }]

    queries = select_articles.visual_queries_for_disease(
        "psoriasis", data, findings=findings
    )

    assert len(queries) == 1
    assert queries[0]["query"] == (
        "Psoriasis Anterior uveitis iritis intraocular inflammation ophthalmic image"
    )
    assert queries[0]["category"] == "eye"


def test_ocular_passages_are_retained_and_long_synonym_lists_are_compact():
    row = {
        "page_content": (
            "Slit-lamp photographs show anterior uveitis with conjunctival injection."
        ),
        "section_title": "Ophthalmic manifestations",
    }
    assert select_articles._is_visual_passage(row)

    queries = select_articles.visual_queries_for_disease(
        "psoriasis", diseases.load_diseases(), findings=[{
            "finding_key": "uveitis",
            "label": "Anterior uveitis",
            "synonyms": [
                "acute anterior uveitis",
                "iritis",
                "iridocyclitis",
                "psoriasis-associated anterior uveitis",
                "uveitis associated with psoriasis",
                "uveitis associated with psoriatic arthritis",
            ],
            "disease_keys": ["psoriasis"],
            "category": "eye",
            "approved": True,
        }]
    )
    query = queries[0]["query"]
    assert query == "Psoriasis Anterior uveitis iritis iridocyclitis ophthalmic image"
    assert len(query) < 100


class _Result:
    def __init__(self, rows):
        self.rows = rows


class _Namespace:
    def __init__(self):
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("rank_by", [None])[0] == "page_content":
            return _Result([
                {
                    "pmcid": "PMC1",
                    "id": "PMC1:1",
                    "title": "Review",
                    "abstract": "A review",
                    "publication_type": ["Review"],
                    "article_type": "review-article",
                    "page_content": "Gottron papules are seen on the knuckles.",
                    "section_title": "Cutaneous manifestations",
                    "section_type": "clinical",
                }
            ])
        return _Result([])


def test_retrieval_keeps_passage_and_section_evidence_for_visual_query():
    ns = _Namespace()
    query = {
        "query": "dermatomyositis Gottron papules clinical photograph",
        "finding": "Gottron papules",
        "modality": "clinical photograph",
    }

    result = select_articles.retrieve_for_disease(
        ns, None, [], visual_queries=[query]
    )

    evidence = result["PMC1"]["matched_passages"][0]
    assert evidence["text"].startswith("Gottron papules")
    assert evidence["section"] == "Cutaneous manifestations"
    assert evidence["section_type"] == "clinical"
    assert evidence["finding"] == "Gottron papules"
    assert any(
        call.get("rank_by") == ["page_content", "BM25", query["query"]]
        for call in ns.calls
    )


def test_evidence_selection_reserves_visual_hit_when_synonym_hits_rank_higher():
    evidence = [
        {
            "query_kind": "synonym",
            "text": f"synonym passage {i}",
            "section": "Introduction",
            "score": 0.5 - i / 100,
        }
        for i in range(10)
    ]
    evidence.append({
        "query_kind": "visual",
        "query": "dermatomyositis Gottron papules clinical photograph",
        "text": "Gottron papules clinical morphology",
        "section": "Clinical manifestations",
        "finding": "Gottron papules",
        "modality": "clinical photograph",
        "score": 0.001,
    })

    selected = select_articles._select_evidence(evidence)

    assert len(selected) == select_articles.MAX_EVIDENCE_PER_ARTICLE
    assert any(item.get("query_kind") == "visual" for item in selected)


def test_visual_query_budget_prioritizes_findings_with_few_panels():
    data = diseases.load_diseases()
    findings = [
        {
            "finding_key": "common_skin",
            "label": "Common skin finding",
            "disease_keys": ["sle"],
            "category": "skin",
            "approved": True,
        },
        {
            "finding_key": "rare_skin",
            "label": "Rare skin finding",
            "disease_keys": ["sle"],
            "category": "skin",
            "approved": True,
        },
    ]

    queries = select_articles.visual_queries_for_disease(
        "sle", data, findings=findings, max_queries=1,
        coverage_counts={"common_skin": 20, "rare_skin": 0},
    )

    assert len(queries) == 1
    assert "Rare skin finding" in queries[0]["query"]


def test_upsert_refreshes_parsed_article_evidence_without_changing_keys(conn):
    conn.execute(
        "INSERT INTO articles (pmcid, title, status, retrieval_score, "
        "primary_disease_keys_json, retrieval_evidence_json) "
        "VALUES ('PMC777', 'Existing review', 'parsed', 0.7, '[\"sle\"]', '[]')"
    )
    conn.commit()
    evidence = [{
        "query_kind": "visual",
        "query": "SLE malar rash clinical photograph",
        "text": "Malar rash with sparing of the nasolabial folds.",
        "section": "Cutaneous manifestations",
        "score": 0.02,
    }]

    status = select_articles.upsert_candidate(
        conn,
        "PMC777",
        {"title": "Changed title", "publication_type": ["Review"]},
        0.9,
        {"dm"},
        evidence,
    )

    row = conn.execute(
        "SELECT status, primary_disease_keys_json, retrieval_score, "
        "retrieval_evidence_json FROM articles WHERE pmcid='PMC777'"
    ).fetchone()
    assert status == "parsed"
    assert row["status"] == "parsed"
    assert row["primary_disease_keys_json"] == '["sle"]'
    assert row["retrieval_score"] == 0.9
    assert "malar rash" in row["retrieval_evidence_json"].lower()


def test_visual_retrieval_weights_image_passages_above_broad_mechanism_hits():
    class RankedNamespace:
        def query(self, **kwargs):
            rank_by = kwargs.get("rank_by", [])
            if rank_by[0] == "page_content" and rank_by[2] == "Gottron papules dermatomyositis clinical photograph":
                return _Result([
                    {
                        "pmcid": "PMC_IMAGE",
                        "id": "PMC_IMAGE:1",
                        "page_content": "Clinical photographs show Gottron papules over the knuckles.",
                        "section_title": "Clinical manifestations",
                        "section_type": "clinical",
                    }
                ])
            if rank_by[0] == "page_content":
                return _Result([
                    {
                        "pmcid": "PMC_MECHANISM",
                        "id": "PMC_MECHANISM:1",
                        "page_content": "Interferon signaling and molecular pathways in dermatomyositis.",
                        "section_title": "Pathogenesis",
                        "section_type": "mechanism",
                    }
                ])
            return _Result([])

    queries = [{
        "query": "Gottron papules dermatomyositis clinical photograph",
        "finding": "Gottron papules",
        "modality": "clinical photograph",
    }]
    articles = select_articles.retrieve_for_disease(
        RankedNamespace(), None, ["dermatomyositis", "DM", "myositis"],
        visual_queries=queries, disease_key="dm",
    )

    assert articles["PMC_IMAGE"]["score"] > articles["PMC_MECHANISM"]["score"]
    assert articles["PMC_IMAGE"]["matched_passages"]
    assert articles["PMC_MECHANISM"]["matched_passages"] == []


# ---------------------------------------------------------------------------
# W4a: batched query embeddings
# ---------------------------------------------------------------------------
class _FakeEmbeddings:
    """OpenAI-style embeddings endpoint; vectors keyed by input string."""

    def __init__(self, vectors, batch_error=None, reversed_data=False):
        self._vectors = vectors
        self._batch_error = batch_error
        self._reversed = reversed_data
        self.batch_inputs = []
        self.single_inputs = []

    def create(self, model=None, input=None):
        if isinstance(input, list):
            self.batch_inputs.append(list(input))
            if self._batch_error is not None:
                raise self._batch_error
            data = [
                SimpleNamespace(index=i, embedding=self._vectors[q])
                for i, q in enumerate(input)
            ]
            if self._reversed:
                data.reverse()
            return SimpleNamespace(data=data)
        self.single_inputs.append(input)
        return SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=self._vectors[input])]
        )


def _retriever(embeddings_api):
    """VisualRetriever with a stubbed embeddings client (no tpuf needed)."""
    r = retrieval.VisualRetriever.__new__(retrieval.VisualRetriever)
    r.embedding_model = "test-embed-model"
    r.openai_client = (
        SimpleNamespace(embeddings=embeddings_api)
        if embeddings_api is not None
        else None
    )
    return r


def test_embed_queries_batched_matches_per_item():
    vectors = {"lupus": [0.1, 0.2], "SLE": [0.3, 0.4]}
    api = _FakeEmbeddings(vectors)
    r = _retriever(api)

    batched = r.embed_queries(["lupus", "SLE"])
    per_item = [r._embed_query(q) for q in ["lupus", "SLE"]]

    assert batched == per_item == [vectors["lupus"], vectors["SLE"]]
    assert api.batch_inputs == [["lupus", "SLE"]]


def test_embed_queries_aligns_out_of_order_data_by_index():
    vectors = {"a": [1.0], "b": [2.0], "c": [3.0]}
    api = _FakeEmbeddings(vectors, reversed_data=True)
    r = _retriever(api)
    assert r.embed_queries(["a", "b", "c"]) == [[1.0], [2.0], [3.0]]


def test_embed_queries_falls_back_to_per_item_on_batch_failure():
    vectors = {"a": [1.0], "b": [2.0]}
    api = _FakeEmbeddings(vectors, batch_error=RuntimeError("batch boom"))
    r = _retriever(api)

    assert r.embed_queries(["a", "b"]) == [[1.0], [2.0]]
    assert api.single_inputs == ["a", "b"]


def test_embed_queries_falls_back_on_incomplete_response():
    class _ShortEmbeddings(_FakeEmbeddings):
        def create(self, model=None, input=None):
            if isinstance(input, list):
                self.batch_inputs.append(list(input))
                return SimpleNamespace(
                    data=[SimpleNamespace(index=0, embedding=self._vectors[input[0]])]
                )
            return super().create(model=model, input=input)

    vectors = {"a": [1.0], "b": [2.0]}
    api = _ShortEmbeddings(vectors)
    r = _retriever(api)

    assert r.embed_queries(["a", "b"]) == [[1.0], [2.0]]
    assert api.single_inputs == ["a", "b"]


def test_embed_queries_without_client_returns_nones():
    r = _retriever(None)
    assert r.embed_queries(["a", "b"]) == [None, None]
    assert r.embed_queries([]) == []
