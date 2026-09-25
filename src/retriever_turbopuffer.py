"""turbopuffer retriever adapter with pipeline-compatible output contract."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import httpx
try:
    import turbopuffer as tpuf
except ImportError:  # pragma: no cover - environment-dependent
    tpuf = None  # type: ignore

from .config import (
    DEEPINFRA_API_KEY,
    DEEPINFRA_BASE_URL,
    DEEPINFRA_EMBED_TIMEOUT_SECONDS,
    EMBEDDING_SUBCACHE_ENABLED,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    HF_INFERENCE_EMBED_TIMEOUT_SECONDS,
    HF_INFERENCE_ENDPOINT_API_KEY,
    HF_INFERENCE_ENDPOINT_URL,
    PMC_BM25_RETRIEVAL_LIMIT,
    PMC_DENSE_RETRIEVAL_LIMIT,
    PUBMED_BM25_RETRIEVAL_LIMIT,
    PUBMED_DENSE_RETRIEVAL_LIMIT,
    RETRIEVAL_DENSE_WEIGHT,
    RETRIEVAL_FUSED_LIMIT,
    RETRIEVAL_RRF_K,
    RETRIEVAL_SUBCACHE_ENABLED,
    RETRIEVAL_SPARSE_WEIGHT,
    SCORE_THRESHOLD,
    DAILYMED_MULTI_QUERY_MAX_CONCURRENCY,
    DAILYMED_MULTI_QUERY_MAX_SUBQUERIES,
    DAILYMED_MULTI_QUERY_TOP_K,
    DAILYMED_MAX_RETRIEVAL_RESULTS,
    TURBOPUFFER_API_KEY,
    TURBOPUFFER_REGION,
    TURBOPUFFER_TIMEOUT_SECONDS,
    TURBOPUFFER_NAMESPACE_DAILYMED,
    TURBOPUFFER_INCLUDE_ALL_ATTRIBUTES,
    TURBOPUFFER_NAMESPACE_PMC,
    TURBOPUFFER_NAMESPACE_PUBMED,
    UPSTREAM_HTTP_MAX_CONNECTIONS,
    UPSTREAM_HTTP_MAX_KEEPALIVE,
    UPSTREAM_HTTP_KEEPALIVE_EXPIRY,
)
from .query_cache import QueryCache

logger = logging.getLogger(__name__)


class TurbopufferRetriever:
    """turbopuffer retriever preserving the existing pipeline contract."""

    PMC_PUBMED_INCLUDE_ATTRIBUTES: List[str] = [
        "id",
        "doc_id",
        "pmcid",
        "pmid",
        "doi",
        "title",
        "page_content",
        "abstract",
        "section_title",
        "section_type",
        "chunk_id",
        "chunk_index",
        "journal",
        "nlm_unique_id",
        "year",
        "article_type",
        "publication_type",
        "evidence_grade",
        "evidence_level",
        "evidence_term",
        "evidence_source",
        "source_family",
        "country",
        "has_full_text",
    ]
    DAILYMED_INCLUDE_ATTRIBUTES: List[str] = [
        "id",
        "doc_id",
        "set_id",
        "title",
        "drug_name",
        "page_content",
        "section_title",
        "abstract",
    ]

    DAILYMED_SECTION_ALIASES: Dict[str, List[str]] = {
        "highlights": [
            "highlights",
            "highlights of prescribing information",
        ],
        "summary": [
            "highlights",
            "highlights of prescribing information",
        ],
        "indications": [
            "indications",
            "indications and usage",
            "uses",
        ],
        "contraindications": [
            "contraindications",
        ],
        "adverse_reactions": [
            "adverse reactions",
            "adverse events",
            "undesirable effects",
        ],
        "clinical_studies": [
            "clinical studies",
            "clinical studies and experience",
            "clinical study",
            "clinical experience",
        ],
        "dosage": [
            "dosage",
            "dosage and administration",
            "dose and administration",
            "dosage forms and strengths",
            "recommended dosage",
        ],
    }

    def __init__(
        self,
        n_retrieval: int = 150,
        n_keyword_search: int = 50,
        n_fused_limit: int = RETRIEVAL_FUSED_LIMIT,
        score_threshold: float = SCORE_THRESHOLD,
    ):
        if tpuf is None:
            raise ValueError("turbopuffer package is not installed")
        if not TURBOPUFFER_API_KEY:
            raise ValueError("TURBOPUFFER_API_KEY not set in config")
        self.n_retrieval = n_retrieval
        self.n_keyword_search = max(1, int(n_keyword_search))
        self.n_fused_limit = max(0, int(n_fused_limit))
        self.score_threshold = score_threshold
        self.embedding_model = EMBEDDING_MODEL
        self.embedding_provider = EMBEDDING_PROVIDER
        self.rrf_k = RETRIEVAL_RRF_K
        self.pmc_dense_limit = max(1, int(PMC_DENSE_RETRIEVAL_LIMIT))
        self.pmc_bm25_limit = max(1, int(PMC_BM25_RETRIEVAL_LIMIT))
        self.pubmed_dense_limit = max(1, int(PUBMED_DENSE_RETRIEVAL_LIMIT))
        self.pubmed_bm25_limit = max(1, int(PUBMED_BM25_RETRIEVAL_LIMIT))
        self.include_all_attributes = bool(TURBOPUFFER_INCLUDE_ALL_ATTRIBUTES)

        self.tpuf = tpuf.Turbopuffer(
            api_key=TURBOPUFFER_API_KEY,
            region=TURBOPUFFER_REGION,
            timeout=TURBOPUFFER_TIMEOUT_SECONDS,
        )
        self.ns_pmc = self.tpuf.namespace(TURBOPUFFER_NAMESPACE_PMC)
        self.ns_pubmed = self.tpuf.namespace(TURBOPUFFER_NAMESPACE_PUBMED)
        self.ns_dailymed = self.tpuf.namespace(TURBOPUFFER_NAMESPACE_DAILYMED)

        self._openai_http_client: Optional[httpx.Client] = None
        self.openai_client = None
        if self.embedding_provider == "hf_inference_endpoint":
            from openai import OpenAI

            self._openai_http_client = self._build_openai_http_client(HF_INFERENCE_EMBED_TIMEOUT_SECONDS)
            self.openai_client = OpenAI(
                api_key=HF_INFERENCE_ENDPOINT_API_KEY,
                base_url=f"{HF_INFERENCE_ENDPOINT_URL}/v1",
                timeout=HF_INFERENCE_EMBED_TIMEOUT_SECONDS,
                http_client=self._openai_http_client,
            )
        elif self.embedding_provider == "deepinfra":
            from openai import OpenAI

            self._openai_http_client = self._build_openai_http_client(DEEPINFRA_EMBED_TIMEOUT_SECONDS)
            self.openai_client = OpenAI(
                api_key=DEEPINFRA_API_KEY,
                base_url=DEEPINFRA_BASE_URL,
                timeout=DEEPINFRA_EMBED_TIMEOUT_SECONDS,
                http_client=self._openai_http_client,
            )
        self.subquery_cache = QueryCache(namespace="retriever")
        self.last_embedding_stats: Dict[str, Any] = {
            "texts_requested": 0,
            "cache_hits": 0,
            "api_misses": 0,
            "api_calls": 0,
        }
        self.last_fixed_bucket_search_stats: Dict[str, Any] = {
            "cache_hit": False,
            "query_count": 0,
            "results_per_query": [],
        }
        self.last_dailymed_search_stats: Dict[str, Any] = {
            "drug_query_detected": False,
            "queried_drug_names": [],
            "attempted_strategies": [],
            "match_strategy": "none",
            "raw_hits": 0,
            "identity_matched_rows": 0,
            "grouped_labels": 0,
            "results_returned": 0,
            "cache_hit": False,
        }

    def _build_openai_http_client(self, timeout_seconds: float) -> httpx.Client:
        return httpx.Client(
            limits=httpx.Limits(
                max_connections=UPSTREAM_HTTP_MAX_CONNECTIONS,
                max_keepalive_connections=UPSTREAM_HTTP_MAX_KEEPALIVE,
                keepalive_expiry=UPSTREAM_HTTP_KEEPALIVE_EXPIRY,
            ),
            timeout=timeout_seconds,
        )

    def _include_attributes_for_source(self, source_family: str) -> bool | List[str]:
        if self.include_all_attributes:
            return True
        if source_family == "dailymed":
            return list(self.DAILYMED_INCLUDE_ATTRIBUTES)
        return list(self.PMC_PUBMED_INCLUDE_ATTRIBUTES)

    def _embed_query(self, query: str) -> Optional[List[float]]:
        if not getattr(self, "openai_client", None):
            return None
        try:
            response = self.openai_client.embeddings.create(model=self.embedding_model, input=query)
            return response.data[0].embedding
        except Exception as exc:
            logger.warning("Embedding failed: %s", exc)
            return None

    @staticmethod
    def _normalize_embedding_text(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip())

    def _embedding_cache_kwargs(self) -> Dict[str, Any]:
        return {
            "provider": self.embedding_provider,
            "model": self.embedding_model,
        }

    def build_shared_embedding_inputs(
        self,
        *,
        retrieval_queries: Optional[List[str]] = None,
        dailymed_keywords: Optional[List[str]] = None,
        context_query: str = "",
        query_variants: Optional[List[str]] = None,
    ) -> List[str]:
        bundled_texts: List[str] = []
        seen: set[str] = set()

        for raw in retrieval_queries or []:
            normalized = self._normalize_embedding_text(raw)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            bundled_texts.append(normalized)

        normalized_query_variants = self._normalize_unique_texts(query_variants)
        normalized_context_query = re.sub(r"\s+", " ", str(context_query or "").strip().lower())
        if normalized_context_query and normalized_context_query not in normalized_query_variants:
            normalized_query_variants.insert(0, normalized_context_query)

        for raw in [*(normalized_query_variants or []), *(self._normalize_drug_queries(dailymed_keywords or []) or [])]:
            normalized = self._normalize_embedding_text(raw)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            bundled_texts.append(normalized)

        return bundled_texts

    def build_embedding_lookup(self, texts: List[str]) -> Dict[str, Optional[List[float]]]:
        normalized_texts: List[str] = []
        seen: set[str] = set()
        for text in texts or []:
            normalized = self._normalize_embedding_text(text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            normalized_texts.append(normalized)

        if not normalized_texts:
            self.last_embedding_stats = {
                "texts_requested": 0,
                "cache_hits": 0,
                "api_misses": 0,
                "api_calls": 0,
            }
            return {}

        lookup: Dict[str, Optional[List[float]]] = {}
        missing_texts: List[str] = []
        cache_hits = 0

        for text in normalized_texts:
            cached = None
            if EMBEDDING_SUBCACHE_ENABLED:
                cached = self.subquery_cache.get_entry(
                    text,
                    namespace="retriever_embeddings",
                    **self._embedding_cache_kwargs(),
                )
            if cached is not None:
                lookup[text] = cached
                cache_hits += 1
                continue
            missing_texts.append(text)

        api_calls = 0
        if missing_texts and getattr(self, "openai_client", None):
            try:
                api_calls = 1
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=missing_texts,
                )
                sorted_data = sorted(response.data, key=lambda item: item.index)
                missing_embeddings = [item.embedding for item in sorted_data]
            except Exception as exc:
                logger.warning("Batch embedding failed: %s. Falling back to sequential.", exc)
                missing_embeddings = []
                api_calls = len(missing_texts)
                for query in missing_texts:
                    missing_embeddings.append(self._embed_query(query))

            for text, embedding in zip(missing_texts, missing_embeddings):
                lookup[text] = embedding
                if EMBEDDING_SUBCACHE_ENABLED and embedding is not None:
                    self.subquery_cache.set_entry(
                        text,
                        embedding,
                        namespace="retriever_embeddings",
                        **self._embedding_cache_kwargs(),
                    )
        else:
            for text in missing_texts:
                lookup[text] = None

        self.last_embedding_stats = {
            "texts_requested": len(normalized_texts),
            "cache_hits": cache_hits,
            "api_misses": len(missing_texts),
            "api_calls": api_calls,
        }
        return lookup

    def _resolve_embeddings_for_queries(
        self,
        queries: List[str],
        embedding_lookup: Optional[Dict[str, Optional[List[float]]]] = None,
    ) -> List[Optional[List[float]]]:
        if not queries:
            return []

        resolved_lookup = dict(embedding_lookup or {})
        missing: List[str] = []
        for query in queries:
            normalized = self._normalize_embedding_text(query)
            if normalized not in resolved_lookup:
                missing.append(normalized)

        if missing:
            resolved_lookup.update(self.build_embedding_lookup(missing))
        elif embedding_lookup is not None:
            self.last_embedding_stats = {
                "texts_requested": len(queries),
                "cache_hits": len(queries),
                "api_misses": 0,
                "api_calls": 0,
            }

        return [resolved_lookup.get(self._normalize_embedding_text(query)) for query in queries]

    def _embed_queries(self, queries: List[str]) -> List[Optional[List[float]]]:
        normalized_queries = [self._normalize_embedding_text(query) for query in queries or []]
        if not normalized_queries:
            return []
        if not getattr(self, "openai_client", None):
            self.last_embedding_stats = {
                "texts_requested": len(normalized_queries),
                "cache_hits": 0,
                "api_misses": len(normalized_queries),
                "api_calls": 0,
            }
            return [None] * len(normalized_queries)
        lookup = self.build_embedding_lookup(normalized_queries)
        return [lookup.get(query) for query in normalized_queries]

    @staticmethod
    def _rank_key_from_row(row: Dict[str, Any]) -> str:
        return str(row.get("chunk_id") or row.get("id") or row.get("pmcid") or row.get("doc_id") or "")

    @staticmethod
    def _doc_id_from_row(row: Dict[str, Any]) -> str:
        return str(row.get("pmcid") or row.get("doc_id") or row.get("pmid") or "")

    @staticmethod
    def _infer_source_family(row: Dict[str, Any]) -> str:
        source_family = str(row.get("source_family") or "").strip().lower()
        if source_family in {"pmc", "pubmed", "dailymed"}:
            return source_family
        pmcid_or_doc = str(row.get("pmcid") or row.get("doc_id") or "").strip().upper()
        if pmcid_or_doc.startswith("PMC"):
            return "pmc"
        return "pubmed"

    @staticmethod
    def _tag_rows_source_family(rows: List[Dict[str, Any]], source_family: str) -> List[Dict[str, Any]]:
        tagged_rows: List[Dict[str, Any]] = []
        normalized = source_family.strip().lower()
        for row in rows:
            tagged = dict(row)
            if not tagged.get("source_family"):
                tagged["source_family"] = normalized
            tagged_rows.append(tagged)
        return tagged_rows

    @staticmethod
    def _normalize_publication_type_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(v) for v in value]
        if value:
            return [str(value)]
        return []

    @staticmethod
    def _normalized_text(value: Any) -> str:
        import math

        if value is None:
            return ""
        if isinstance(value, float) and math.isnan(value):
            return ""
        return str(value).strip().lower()

    @staticmethod
    def _normalize_whitespace(value: Any) -> str:
        import math

        if value is None:
            return ""
        if isinstance(value, float) and math.isnan(value):
            return ""
        text = str(value)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        return text.strip()

    @staticmethod
    def _safe_text(value: Any) -> str:
        import math

        if value is None:
            return ""
        if isinstance(value, float) and math.isnan(value):
            return ""
        return str(value).strip()

    @staticmethod
    def _normalize_section_key(value: Any) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())).strip()

    @classmethod
    def _section_key_matches_alias(cls, normalized_key: str, normalized_alias: str) -> bool:
        if not normalized_key or not normalized_alias:
            return False
        if normalized_key == normalized_alias:
            return True
        key_tokens = normalized_key.split()
        alias_tokens = normalized_alias.split()
        if not key_tokens or not alias_tokens:
            return False
        if len(alias_tokens) == 1:
            return alias_tokens[0] in key_tokens
        window = len(alias_tokens)
        for i in range(len(key_tokens) - window + 1):
            if key_tokens[i:i + window] == alias_tokens:
                return True
        return False

    @classmethod
    def _extract_dailymed_sections_for_ui(cls, payload: Dict[str, Any]) -> Dict[str, str]:
        section_map = payload.get("dailymed_sections", {})
        if not isinstance(section_map, dict):
            section_map = {}

        normalized_section_items: List[tuple[str, str]] = []
        for raw_key, raw_value in section_map.items():
            normalized_key = cls._normalize_section_key(raw_key)
            text_value = cls._safe_text(raw_value)
            if normalized_key and text_value:
                normalized_section_items.append((normalized_key, text_value))

        def find_section(aliases: List[str], direct_value: str = "") -> str:
            direct = cls._safe_text(direct_value)
            if direct:
                return direct

            normalized_aliases = [cls._normalize_section_key(alias) for alias in aliases if alias]
            # Exact match first.
            for alias in normalized_aliases:
                for key, text in normalized_section_items:
                    if key == alias:
                        return text
            # Token-aware fuzzy match second.
            for alias in normalized_aliases:
                for key, text in normalized_section_items:
                    if cls._section_key_matches_alias(key, alias):
                        return text
            return ""

        normalized_sections: Dict[str, str] = {}
        for canonical_key, aliases in cls.DAILYMED_SECTION_ALIASES.items():
            normalized_sections[canonical_key] = find_section(
                aliases=aliases,
                direct_value=payload.get(canonical_key, ""),
            )

        # Keep summary/highlights synchronized for frontend + context consumers.
        if not normalized_sections.get("summary") and normalized_sections.get("highlights"):
            normalized_sections["summary"] = normalized_sections["highlights"]
        if not normalized_sections.get("highlights") and normalized_sections.get("summary"):
            normalized_sections["highlights"] = normalized_sections["summary"]

        return normalized_sections

    @classmethod
    def _is_dailymed_row(cls, row: Dict[str, Any]) -> bool:
        source = cls._normalized_text(row.get("source"))
        source_family = cls._normalized_text(row.get("source_family"))
        article_type = cls._normalized_text(row.get("article_type"))
        content_type = cls._normalized_text(row.get("content_type"))
        pmcid = cls._normalized_text(row.get("pmcid") or row.get("corpus_id"))
        doc_id = cls._normalized_text(row.get("doc_id"))
        set_id = cls._normalized_text(row.get("set_id"))

        if source == "dailymed" or source_family == "dailymed":
            return True
        if article_type == "drug_label" or content_type == "drug_label":
            return True
        if pmcid.startswith("dailymed_") or doc_id.startswith("dailymed_"):
            return True
        if set_id:
            return True
        return False

    def _rrf_scores(self, rows: List[Dict[str, Any]], weight: float) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        rrf_k = float(getattr(self, "rrf_k", RETRIEVAL_RRF_K))
        for rank, row in enumerate(rows, 1):
            key = str(row.get("_rank_key", ""))
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + ((1.0 / (rrf_k + rank)) * weight)
        return scores

    @staticmethod
    def _normalize_drug_queries(drug_names: List[str]) -> List[str]:
        return TurbopufferRetriever._normalize_unique_texts(drug_names)

    @staticmethod
    def _normalize_unique_texts(values: Optional[List[str]], *, min_len: int = 1) -> List[str]:
        normalized: List[str] = []
        seen: set[str] = set()
        for raw in values or []:
            text = re.sub(r"\s+", " ", str(raw or "").strip().lower())
            if len(text) < min_len or text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized

    @staticmethod
    def _normalized_drug_concept_key(row: Dict[str, Any]) -> str:
        source_name = str(row.get("drug_name") or row.get("title") or "").strip().lower()
        normalized = re.sub(r"[^a-z0-9\s\-]", " ", source_name)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = re.sub(
            r"\s+(?:tablets?|capsules?|injection|solution|oral|intravenous|iv|im|powder|suspension|syrup|cream|ointment|gel|patch|spray)\b.*$",
            "",
            normalized,
            flags=re.IGNORECASE,
        ).strip()
        return normalized

    def _query_namespace_title_fts(
        self,
        ns: Any,
        query_text: str,
        limit: int,
        include_attributes: bool | List[str],
    ) -> List[Dict[str, Any]]:
        try:
            result = ns.query(
                rank_by=["title", "BM25", query_text],
                top_k=limit,
                include_attributes=include_attributes,
            )
            rows = getattr(result, "rows", None) or getattr(result, "results", None) or []
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("title BM25 query failed: %s", exc)
            return []

    def _row_to_passage(self, row: Dict[str, Any], score: float, stype: str) -> Dict[str, Any]:
        if self._is_dailymed_row(row):
            return self._transform_dailymed_payload(row, score)
        doc_id = self._doc_id_from_row(row)
        source_family = self._infer_source_family(row)
        return {
            "corpus_id": doc_id,
            "doc_id": row.get("doc_id") or doc_id,
            "pmcid": row.get("pmcid", doc_id),
            "pmid": row.get("pmid"),
            "doi": row.get("doi"),
            "title": self._normalize_whitespace(row.get("title", "")),
            "text": self._normalize_whitespace(row.get("page_content") or row.get("abstract", "")),
            "abstract": self._normalize_whitespace(row.get("abstract", "")),
            "full_text": "",
            "has_full_text": row.get("has_full_text", False),
            "section_title": row.get("section_title", "abstract"),
            "section_type": row.get("section_type", "body"),
            "chunk_id": row.get("chunk_id"),
            "chunk_index": row.get("chunk_index"),
            "journal": row.get("journal", ""),
            "venue": row.get("journal", ""),
            "nlm_unique_id": row.get("nlm_unique_id"),
            "year": row.get("year"),
            "authors": [],
            "article_type": row.get("article_type", ""),
            "publication_type": self._normalize_publication_type_list(row.get("publication_type")),
            "score": score,
            "stype": stype,
            "source_family": source_family,
            "evidence_grade": row.get("evidence_grade"),
            "evidence_level": row.get("evidence_level"),
            "evidence_term": row.get("evidence_term"),
            "evidence_source": row.get("evidence_source"),
            "country": row.get("country"),
        }

    def _merge_ranked_passage_rows(
        self,
        bucket_rows: List[tuple[str, str, List[Dict[str, Any]]]],
    ) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        fused_scores: Dict[str, float] = {}
        normalized_bucket_rows: List[tuple[str, str, List[Dict[str, Any]]]] = []

        for source_family, stype, rows in bucket_rows:
            keyed_rows: List[Dict[str, Any]] = []
            for row in rows:
                tagged_row = dict(row)
                tagged_row["source_family"] = source_family
                key = self._rank_key_from_row(tagged_row)
                if not key:
                    continue
                tagged_row["_rank_key"] = key
                keyed_rows.append(tagged_row)
            weight = RETRIEVAL_DENSE_WEIGHT if "dense" in stype else RETRIEVAL_SPARSE_WEIGHT
            for key, score in self._rrf_scores(keyed_rows, weight).items():
                fused_scores[key] = fused_scores.get(key, 0.0) + score
            normalized_bucket_rows.append((source_family, stype, keyed_rows))

        for source_family, stype, rows in normalized_bucket_rows:
            for rank, tagged_row in enumerate(rows, start=1):
                key = str(tagged_row.get("_rank_key", ""))
                if not key:
                    continue
                fused_score = fused_scores.get(key, 1.0 / float(rank))
                passage = self._row_to_passage(tagged_row, fused_score, stype)
                passage["fused_retrieval_score"] = fused_score
                existing = merged.get(key)
                if existing is None:
                    passage["retrieval_bucket"] = stype
                    passage["retrieval_bucket_matches"] = [stype]
                    merged[key] = passage
                    continue

                existing["score"] = fused_score
                existing["fused_retrieval_score"] = fused_score
                bucket_matches = list(existing.get("retrieval_bucket_matches") or [])
                if stype not in bucket_matches:
                    bucket_matches.append(stype)
                existing["retrieval_bucket_matches"] = bucket_matches

                if (
                    not str(existing.get("source_family", "")).strip()
                    and str(passage.get("source_family", "")).strip()
                ):
                    existing["source_family"] = passage["source_family"]
                if (
                    not str(existing.get("section_title", "")).strip()
                    and str(passage.get("section_title", "")).strip()
                ):
                    existing["section_title"] = passage["section_title"]
                if (
                    not str(existing.get("text", "")).strip()
                    and str(passage.get("text", "")).strip()
                ):
                    existing["text"] = passage["text"]

        merged_rows = list(merged.values())
        merged_rows.sort(
            key=lambda row: (
                float(row.get("fused_retrieval_score", row.get("score", 0.0))),
                len(row.get("retrieval_bucket_matches") or []),
            ),
            reverse=True,
        )
        return merged_rows

    def retrieve_fixed_source_buckets(
        self,
        query: str,
        *,
        embedding_lookup: Optional[Dict[str, Optional[List[float]]]] = None,
    ) -> List[Dict[str, Any]]:
        if not query:
            return []
        normalized_query = re.sub(r"\s+", " ", str(query).strip())
        if not normalized_query:
            return []
        return self._retrieve_fixed_source_buckets_multi_query_batched(
            [normalized_query],
            embedding_lookup=embedding_lookup,
        )[0]

    def retrieve_dense_topk_per_query(
        self,
        queries: List[str],
        *,
        per_query_limit: int = 5,
    ) -> List[List[Dict[str, Any]]]:
        """Dense-only retrieval for each query (PMC + PubMed), capped per query."""
        normalized_queries: List[str] = []
        seen_queries: set[str] = set()
        for query in queries or []:
            normalized = re.sub(r"\s+", " ", str(query or "").strip())
            if not normalized or normalized in seen_queries:
                continue
            seen_queries.add(normalized)
            normalized_queries.append(normalized)
        if not normalized_queries:
            return []

        query_embeddings = self._embed_queries(normalized_queries)
        top_k = max(1, int(per_query_limit))
        with ThreadPoolExecutor(max_workers=2) as executor:
            pmc_future = executor.submit(
                self._query_namespace_dense_multi_query,
                namespace=self.ns_pmc,
                source_family="pmc",
                queries=normalized_queries,
                embeddings=query_embeddings,
                dense_limit=top_k,
            )
            pubmed_future = executor.submit(
                self._query_namespace_dense_multi_query,
                namespace=self.ns_pubmed,
                source_family="pubmed",
                queries=normalized_queries,
                embeddings=query_embeddings,
                dense_limit=top_k,
            )
            pmc_rows_by_query = pmc_future.result()
            pubmed_rows_by_query = pubmed_future.result()

        query_results: List[List[Dict[str, Any]]] = []
        for query_index in range(len(normalized_queries)):
            pmc_rows = pmc_rows_by_query[query_index] if query_index < len(pmc_rows_by_query) else []
            pubmed_rows = pubmed_rows_by_query[query_index] if query_index < len(pubmed_rows_by_query) else []

            scored_passages: List[Dict[str, Any]] = []
            for rank, row in enumerate(self._tag_rows_source_family(pmc_rows, "pmc"), start=1):
                try:
                    score = float(row.get("score", 1.0 / (self.rrf_k + rank)))
                except (TypeError, ValueError):
                    score = 1.0 / (self.rrf_k + rank)
                scored_passages.append(self._row_to_passage(row, score, "pmc_dense"))
            for rank, row in enumerate(self._tag_rows_source_family(pubmed_rows, "pubmed"), start=1):
                try:
                    score = float(row.get("score", 1.0 / (self.rrf_k + rank)))
                except (TypeError, ValueError):
                    score = 1.0 / (self.rrf_k + rank)
                scored_passages.append(self._row_to_passage(row, score, "pubmed_dense"))

            scored_passages.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
            query_results.append(scored_passages[:top_k])

        logger.info(
            "   Dense-only per-query retrieval: variants=%d top_k=%d",
            len(normalized_queries),
            top_k,
        )
        return query_results

    @staticmethod
    def _rows_from_namespace_multi_query_result(result: Any) -> List[Dict[str, Any]]:
        rows = getattr(result, "rows", None) or []
        return [dict(row) for row in rows]

    def _query_namespace_dense_multi_query(
        self,
        *,
        namespace: Any,
        source_family: str,
        queries: List[str],
        embeddings: List[Optional[List[float]]],
        dense_limit: int,
    ) -> List[List[Dict[str, Any]]]:
        """Run dense-only ANN retrieval for multiple queries in one API call."""
        dense_rows_by_query: List[List[Dict[str, Any]]] = [[] for _ in queries]
        include_attributes = self._include_attributes_for_source(source_family)
        batched_subqueries: List[Dict[str, Any]] = []
        index_map: List[int] = []

        for query_index, _query in enumerate(queries):
            query_embedding = embeddings[query_index] if query_index < len(embeddings) else None
            if query_embedding is None:
                continue
            batched_subqueries.append(
                {
                    "rank_by": ["vector", "ANN", query_embedding],
                    "top_k": dense_limit,
                    "include_attributes": include_attributes,
                }
            )
            index_map.append(query_index)

        if not batched_subqueries:
            return dense_rows_by_query

        response = namespace.multi_query(queries=batched_subqueries)
        results = list(getattr(response, "results", []) or [])
        if len(results) != len(index_map):
            raise RuntimeError(
                f"namespace.multi_query returned {len(results)} results for {len(index_map)} dense subqueries"
            )

        for query_index, result in zip(index_map, results):
            dense_rows_by_query[query_index] = self._rows_from_namespace_multi_query_result(result)

        return dense_rows_by_query

    def _query_namespace_hybrid_multi_query(
        self,
        *,
        namespace: Any,
        source_family: str,
        queries: List[str],
        embeddings: List[Optional[List[float]]],
        dense_limit: int,
        bm25_limit: int,
    ) -> tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
        dense_rows_by_query: List[List[Dict[str, Any]]] = [[] for _ in queries]
        bm25_rows_by_query: List[List[Dict[str, Any]]] = [[] for _ in queries]

        batched_subqueries: List[Dict[str, Any]] = []
        index_map: List[tuple[int, str]] = []
        include_attributes = self._include_attributes_for_source(source_family)
        for query_index, query in enumerate(queries):
            query_embedding = embeddings[query_index] if query_index < len(embeddings) else None
            if query_embedding is not None:
                batched_subqueries.append(
                    {
                        "rank_by": ["vector", "ANN", query_embedding],
                        "top_k": dense_limit,
                        "include_attributes": include_attributes,
                    }
                )
                index_map.append((query_index, "dense"))
            batched_subqueries.append(
                {
                    "rank_by": ["page_content", "BM25", query],
                    "top_k": bm25_limit,
                    "include_attributes": include_attributes,
                }
            )
            index_map.append((query_index, "bm25"))

        response = namespace.multi_query(queries=batched_subqueries)
        results = list(getattr(response, "results", []) or [])
        if len(results) != len(index_map):
            raise RuntimeError(
                f"namespace.multi_query returned {len(results)} results for {len(index_map)} subqueries"
            )
        for (query_index, bucket_type), result in zip(index_map, results):
            rows = self._rows_from_namespace_multi_query_result(result)
            if bucket_type == "dense":
                dense_rows_by_query[query_index] = rows
            else:
                bm25_rows_by_query[query_index] = rows

        # Keep title BM25 fallback when page_content BM25 has no hits.
        for query_index, query in enumerate(queries):
            if bm25_rows_by_query[query_index]:
                continue
            bm25_rows_by_query[query_index] = self._query_namespace_title_fts(
                namespace,
                query,
                bm25_limit,
                include_attributes=include_attributes,
            )

        return dense_rows_by_query, bm25_rows_by_query

    def _retrieve_fixed_source_buckets_multi_query_batched(
        self,
        normalized_queries: List[str],
        *,
        embedding_lookup: Optional[Dict[str, Optional[List[float]]]] = None,
    ) -> List[List[Dict[str, Any]]]:
        self.last_fixed_bucket_search_stats = {
            "cache_hit": False,
            "query_count": len(normalized_queries),
            "results_per_query": [],
        }
        cache_key_text = " || ".join(normalized_queries)
        cache_kwargs = {
            "queries": normalized_queries,
            "pmc_dense_limit": self.pmc_dense_limit,
            "pmc_bm25_limit": self.pmc_bm25_limit,
            "pubmed_dense_limit": self.pubmed_dense_limit,
            "pubmed_bm25_limit": self.pubmed_bm25_limit,
            "dense_weight": RETRIEVAL_DENSE_WEIGHT,
            "sparse_weight": RETRIEVAL_SPARSE_WEIGHT,
            "pmc_namespace": TURBOPUFFER_NAMESPACE_PMC,
            "pubmed_namespace": TURBOPUFFER_NAMESPACE_PUBMED,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
        }
        if RETRIEVAL_SUBCACHE_ENABLED:
            cached = self.subquery_cache.get_entry(
                cache_key_text,
                namespace="retriever_fixed_source_buckets",
                **cache_kwargs,
            )
            if cached is not None:
                self.last_fixed_bucket_search_stats = {
                    "cache_hit": True,
                    "query_count": len(normalized_queries),
                    "results_per_query": [len(rows) for rows in cached],
                }
                return cached

        embeddings = self._resolve_embeddings_for_queries(normalized_queries, embedding_lookup=embedding_lookup)
        query_count = len(normalized_queries)
        query_bucket_rows: List[List[tuple[str, str, List[Dict[str, Any]]]]] = [[] for _ in range(query_count)]

        def run_namespace(
            namespace: Any,
            source_label: str,
            dense_limit: int,
            bm25_limit: int,
        ) -> tuple[str, List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
            dense_rows, bm25_rows = self._query_namespace_hybrid_multi_query(
                namespace=namespace,
                source_family=source_label,
                queries=normalized_queries,
                embeddings=embeddings,
                dense_limit=dense_limit,
                bm25_limit=bm25_limit,
            )
            return source_label, dense_rows, bm25_rows

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    run_namespace,
                    self.ns_pmc,
                    "pmc",
                    self.pmc_dense_limit,
                    self.pmc_bm25_limit,
                ),
                executor.submit(
                    run_namespace,
                    self.ns_pubmed,
                    "pubmed",
                    self.pubmed_dense_limit,
                    self.pubmed_bm25_limit,
                ),
            ]
            namespace_results = [future.result() for future in futures]

        for source_label, dense_rows_by_query, bm25_rows_by_query in namespace_results:
            dense_bucket = f"{source_label}_dense"
            bm25_bucket = f"{source_label}_bm25"
            for query_index in range(query_count):
                query_bucket_rows[query_index].append(
                    (source_label, dense_bucket, dense_rows_by_query[query_index])
                )
                query_bucket_rows[query_index].append(
                    (source_label, bm25_bucket, bm25_rows_by_query[query_index])
                )

        merged_rows = [self._merge_ranked_passage_rows(bucket_rows) for bucket_rows in query_bucket_rows]
        self.last_fixed_bucket_search_stats = {
            "cache_hit": False,
            "query_count": len(normalized_queries),
            "results_per_query": [len(rows) for rows in merged_rows],
        }
        if RETRIEVAL_SUBCACHE_ENABLED:
            self.subquery_cache.set_entry(
                cache_key_text,
                merged_rows,
                namespace="retriever_fixed_source_buckets",
                **cache_kwargs,
            )
        return merged_rows

    def _aggregate_dailymed_payloads(self, payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not payloads:
            return {}
        base = dict(payloads[0])
        base["dailymed_sections"] = {}
        for row in payloads:
            section_title = str(row.get("section_title") or "")
            content = row.get("page_content") or row.get("text") or ""
            if section_title and content:
                base["dailymed_sections"][section_title] = content
        return base

    @classmethod
    def _dailymed_payload_matches_terms(
        cls,
        payload: Dict[str, Any],
        terms: List[str],
        phrases: Optional[List[str]] = None,
    ) -> bool:
        normalized_terms = [
            re.sub(r"\s+", " ", str(term or "").strip().lower())
            for term in terms or []
            if str(term or "").strip()
        ]
        normalized_phrases = [
            re.sub(r"\s+", " ", str(phrase or "").strip().lower())
            for phrase in phrases or []
            if str(phrase or "").strip()
        ]
        if not normalized_terms and not normalized_phrases:
            return True

        haystacks = [
            cls._normalized_text(payload.get("title")),
            cls._normalized_text(payload.get("drug_name")),
            cls._normalized_text(payload.get("abstract")),
            cls._normalized_text(payload.get("page_content") or payload.get("text")),
        ]
        section_map = payload.get("dailymed_sections", {})
        if isinstance(section_map, dict):
            for section_title, section_text in section_map.items():
                haystacks.append(cls._normalized_text(section_title))
                haystacks.append(cls._normalized_text(section_text))

        if normalized_phrases and any(
            phrase in haystack for phrase in normalized_phrases for haystack in haystacks if haystack
        ):
            return True

        matched_terms = {
            term for term in normalized_terms for haystack in haystacks if haystack and term in haystack
        }
        required_matches = 1 if len(normalized_terms) <= 1 else 2
        return len(matched_terms) >= required_matches

    def _transform_dailymed_payload(self, payload: Dict[str, Any], score: float) -> Dict[str, Any]:
        set_id = self._normalized_text(payload.get("set_id")) or self._normalized_text(payload.get("doc_id"))
        corpus_id = f"dailymed_{set_id}" if set_id else self._safe_text(payload.get("corpus_id") or payload.get("pmcid"))
        title = self._safe_text(payload.get("title") or payload.get("drug_name")) or "Untitled"
        venue = self._safe_text(payload.get("venue") or payload.get("journal")) or "DailyMed"
        normalized_sections = self._extract_dailymed_sections_for_ui(payload)
        drug_name = self._safe_text(payload.get("drug_name")) or title
        return {
            "corpus_id": corpus_id or f"dailymed_{set_id}",
            "pmcid": corpus_id or f"dailymed_{set_id}",
            "set_id": set_id,
            "title": title,
            "drug_name": drug_name,
            "text": payload.get("page_content") or payload.get("text", ""),
            "abstract": payload.get("abstract", ""),
            "source": "dailymed",
            "source_family": "dailymed",
            "content_type": "drug_label",
            "article_type": "drug_label",
            "journal": venue,
            "venue": venue,
            "doc_id": payload.get("doc_id") or set_id,
            "publication_type": self._normalize_publication_type_list(payload.get("publication_type")) or ["drug_label"],
            "dailymed_sections": payload.get("dailymed_sections", {}),
            "dosage": normalized_sections.get("dosage", ""),
            "score": score,
            "stype": "dailymed_lookup",
        }

    def search_dailymed_by_drug(
        self,
        drug_names: List[str],
        limit: int = DAILYMED_MAX_RETRIEVAL_RESULTS,
        context_query: str = "",
        query_variants: Optional[List[str]] = None,
        filter_phrases: Optional[List[str]] = None,
        embedding_lookup: Optional[Dict[str, Optional[List[float]]]] = None,
    ) -> List[Dict[str, Any]]:
        max_results = max(1, int(limit or DAILYMED_MAX_RETRIEVAL_RESULTS))
        normalized_drugs = self._normalize_drug_queries(drug_names)[:max_results]
        normalized_context_query = re.sub(r"\s+", " ", str(context_query or "").strip().lower())
        normalized_query_variants = self._normalize_unique_texts(query_variants)
        if normalized_context_query and normalized_context_query not in normalized_query_variants:
            normalized_query_variants.insert(0, normalized_context_query)
        normalized_filter_phrases = self._normalize_unique_texts(filter_phrases, min_len=4)
        filter_terms = self._extract_dailymed_context_terms(normalized_query_variants, normalized_drugs)
        self.last_dailymed_search_stats = {
            "drug_query_detected": bool(normalized_drugs),
            "queried_drug_names": list(normalized_drugs),
            "context_query": normalized_context_query,
            "query_variants": list(normalized_query_variants),
            "filter_phrases": list(normalized_filter_phrases),
            "context_terms": list(filter_terms),
            "attempted_strategies": [],
            "match_strategy": "none",
            "raw_hits": 0,
            "identity_matched_rows": 0,
            "grouped_labels": 0,
            "results_returned": 0,
            "raw_hits_by_strategy": {},
            "identity_matches_by_strategy": {},
            "cache_hit": False,
        }
        if not normalized_drugs and not normalized_query_variants:
            return []
        strategy_name = "hybrid_multi_query"
        self.last_dailymed_search_stats["attempted_strategies"].append(strategy_name)
        cache_key_text = normalized_context_query or " || ".join(normalized_query_variants) or " ".join(normalized_drugs)
        cache_kwargs = {
            "drug_names": normalized_drugs,
            "query_variants": normalized_query_variants,
            "filter_phrases": normalized_filter_phrases,
            "filter_terms": filter_terms,
            "limit": max_results,
            "strategy_name": strategy_name,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "dailymed_namespace": TURBOPUFFER_NAMESPACE_DAILYMED,
        }
        if RETRIEVAL_SUBCACHE_ENABLED:
            cached_payload = self.subquery_cache.get_entry(
                cache_key_text,
                namespace="retriever_dailymed",
                **cache_kwargs,
            )
            if cached_payload is not None:
                cached_stats = dict(cached_payload.get("stats", {}))
                cached_stats["cache_hit"] = True
                self.last_dailymed_search_stats = cached_stats
                return list(cached_payload.get("results", []))

        per_keyword_top_k = max(1, int(DAILYMED_MULTI_QUERY_TOP_K))
        dense_query_texts: List[str] = []
        dense_query_texts.extend(normalized_query_variants)
        for keyword in normalized_drugs:
            dense_query_texts.append(keyword)
        unique_dense_queries: List[str] = []
        seen_dense_queries: set[str] = set()
        for query_text in dense_query_texts:
            normalized = re.sub(r"\s+", " ", str(query_text or "").strip().lower())
            if not normalized or normalized in seen_dense_queries:
                continue
            seen_dense_queries.add(normalized)
            unique_dense_queries.append(normalized)
        dense_embeddings = (
            self._resolve_embeddings_for_queries(unique_dense_queries, embedding_lookup=embedding_lookup)
            if unique_dense_queries
            else []
        )
        dense_embedding_by_query = {
            query_text: embedding
            for query_text, embedding in zip(unique_dense_queries, dense_embeddings)
            if embedding is not None
        }

        strategy_specs = [
            ("drug_name", "drug_name", lambda keyword: keyword),
            ("title", "title", lambda keyword: keyword),
            ("label_text", "page_content", lambda keyword: keyword),
        ]
        subqueries: List[Dict[str, Any]] = []
        subquery_map: List[str] = []

        for keyword in normalized_drugs:
            for strategy_label, field_name, query_builder in strategy_specs:
                rank_query = query_builder(keyword).strip()
                if not rank_query:
                    continue
                subqueries.append(
                    {
                        "rank_by": [field_name, "BM25", rank_query],
                        "top_k": per_keyword_top_k,
                        "include_attributes": self._include_attributes_for_source("dailymed"),
                    }
                )
                subquery_map.append(strategy_label)

            keyword_dense_key = re.sub(r"\s+", " ", keyword.lower()).strip()
            keyword_dense_embedding = dense_embedding_by_query.get(keyword_dense_key)
            if keyword_dense_embedding is not None:
                subqueries.append(
                    {
                        "rank_by": ["vector", "ANN", keyword_dense_embedding],
                        "top_k": per_keyword_top_k,
                        "include_attributes": self._include_attributes_for_source("dailymed"),
                    }
                )
                subquery_map.append("dense_keyword")

        for query_text in normalized_query_variants:
            subqueries.append(
                {
                    "rank_by": ["page_content", "BM25", query_text],
                    "top_k": per_keyword_top_k,
                    "include_attributes": self._include_attributes_for_source("dailymed"),
                }
            )
            subquery_map.append("full_query_text")
            full_dense_key = re.sub(r"\s+", " ", query_text).strip()
            full_dense_embedding = dense_embedding_by_query.get(full_dense_key)
            if full_dense_embedding is None:
                continue
            subqueries.append(
                {
                    "rank_by": ["vector", "ANN", full_dense_embedding],
                    "top_k": per_keyword_top_k,
                    "include_attributes": self._include_attributes_for_source("dailymed"),
                }
            )
            subquery_map.append("dense_full_query")

        if not subqueries:
            logger.info("   DailyMed multi-query skipped: no subqueries generated")
            return []

        batch_size = max(1, int(DAILYMED_MULTI_QUERY_MAX_SUBQUERIES))
        batch_count = (len(subqueries) + batch_size - 1) // batch_size
        batch_concurrency = max(1, min(int(DAILYMED_MULTI_QUERY_MAX_CONCURRENCY), batch_count))
        batch_payloads: List[tuple[int, List[Dict[str, Any]]]] = []
        batch_sizes: List[int] = []
        for start in range(0, len(subqueries), batch_size):
            batch_index = len(batch_payloads)
            query_batch = subqueries[start : start + batch_size]
            batch_payloads.append((batch_index, query_batch))
            batch_sizes.append(len(query_batch))

        logger.info(
            "   DailyMed multi-query batching: total_subqueries=%d batches=%d batch_sizes=%s concurrency=%d top_k=%d",
            len(subqueries),
            batch_count,
            batch_sizes,
            batch_concurrency,
            per_keyword_top_k,
        )

        batched_results_by_batch: List[List[Any]] = [[] for _ in batch_payloads]

        def run_dailymed_batch(batch_index: int, query_batch: List[Dict[str, Any]]) -> tuple[int, List[Any]]:
            response = self.ns_dailymed.multi_query(queries=query_batch)
            results = list(getattr(response, "results", []) or [])
            if len(results) != len(query_batch):
                logger.warning(
                    "DailyMed batch %d returned %d result sets for %d subqueries",
                    batch_index + 1,
                    len(results),
                    len(query_batch),
                )
            return batch_index, results

        with ThreadPoolExecutor(max_workers=batch_concurrency) as executor:
            futures = [
                executor.submit(run_dailymed_batch, batch_index, query_batch)
                for batch_index, query_batch in batch_payloads
            ]
            for future in futures:
                try:
                    batch_index, results = future.result()
                    batched_results_by_batch[batch_index] = results
                except Exception as exc:
                    logger.warning("DailyMed multi-query batch failed: %s", exc)

        batched_results: List[Any] = []
        for (batch_index, query_batch), results in zip(batch_payloads, batched_results_by_batch):
            if not results:
                logger.warning(
                    "DailyMed batch %d produced no results due to failure or empty response",
                    batch_index + 1,
                )
                continue
            batched_results.extend(results[: len(query_batch)])

        if len(batched_results) != len(subquery_map):
            logger.warning(
                "DailyMed multi-query returned %d result sets for %d subqueries",
                len(batched_results),
                len(subquery_map),
            )

        group_rows: Dict[str, List[Dict[str, Any]]] = {}
        ordered_group_ids: List[str] = []
        group_context_hits: Dict[str, int] = {}
        total_rows = 0

        for sub_strategy, result in zip(subquery_map, batched_results):
            rows = self._rows_from_namespace_multi_query_result(result)
            self.last_dailymed_search_stats["raw_hits_by_strategy"][sub_strategy] = (
                self.last_dailymed_search_stats["raw_hits_by_strategy"].get(sub_strategy, 0) + len(rows)
            )
            total_rows += len(rows)
            for row in rows:
                group_id = str(row.get("set_id") or row.get("doc_id") or "").strip()
                if not group_id:
                    continue
                if group_id not in group_rows:
                    group_rows[group_id] = []
                    ordered_group_ids.append(group_id)
                    group_context_hits[group_id] = 0
                group_rows[group_id].append(dict(row))
                if filter_terms:
                    section_title = str(row.get("section_title") or "").lower()
                    text_content = str(row.get("page_content") or row.get("text") or "").lower()
                    phrase_hit = any(phrase in text_content for phrase in normalized_filter_phrases)
                    term_hit_count = sum(1 for term in filter_terms if term in text_content)
                    if phrase_hit or term_hit_count > 0:
                        # Prefer symptom/context matches, especially in safety-relevant sections.
                        base_weight = 3 if phrase_hit else min(2, term_hit_count)
                        hit_weight = base_weight + 1 if ("warning" in section_title or "precaution" in section_title or "contra" in section_title) else base_weight
                        group_context_hits[group_id] += hit_weight

        self.last_dailymed_search_stats["raw_hits_by_strategy"][strategy_name] = total_rows
        self.last_dailymed_search_stats["identity_matches_by_strategy"][strategy_name] = total_rows
        self.last_dailymed_search_stats["raw_hits"] = total_rows
        self.last_dailymed_search_stats["identity_matched_rows"] = total_rows
        self.last_dailymed_search_stats["match_strategy"] = strategy_name
        self.last_dailymed_search_stats["grouped_labels"] = len(ordered_group_ids)

        initial_order = {group_id: idx for idx, group_id in enumerate(ordered_group_ids)}
        ordered_group_ids = sorted(
            ordered_group_ids,
            key=lambda group_id: (
                int(group_context_hits.get(group_id, 0) > 0),
                group_context_hits.get(group_id, 0),
                len(group_rows.get(group_id, [])),
                -initial_order.get(group_id, 0),
            ),
            reverse=True,
        )

        results: List[Dict[str, Any]] = []
        seen_concepts: set[str] = set()
        for group_id in ordered_group_ids:
            aggregate = self._aggregate_dailymed_payloads(group_rows[group_id])
            if not normalized_drugs and not self._dailymed_payload_matches_terms(
                aggregate,
                filter_terms,
                normalized_filter_phrases,
            ):
                continue
            aggregate["set_id"] = group_id
            transformed = self._transform_dailymed_payload(aggregate, score=1.0)
            concept_key = self._normalized_drug_concept_key(transformed) or str(group_id).lower()
            if concept_key in seen_concepts:
                continue
            seen_concepts.add(concept_key)
            results.append(transformed)
            if len(results) >= max_results:
                break

        self.last_dailymed_search_stats["results_returned"] = len(results)
        logger.info(
            "   DailyMed %s search: keywords=%d context_terms=%d raw=%d grouped=%d returned=%d",
            strategy_name,
            len(normalized_drugs),
            len(filter_terms),
            total_rows,
            len(ordered_group_ids),
            len(results),
        )
        if RETRIEVAL_SUBCACHE_ENABLED:
            self.subquery_cache.set_entry(
                cache_key_text,
                {
                    "results": results,
                    "stats": self.last_dailymed_search_stats,
                },
                namespace="retriever_dailymed",
                **cache_kwargs,
            )
        return results

    @staticmethod
    def _extract_dailymed_context_terms(query_texts: List[str], normalized_drugs: List[str]) -> List[str]:
        if not query_texts:
            return []
        stopwords = {
            "the", "and", "with", "without", "from", "into", "for", "about",
            "management", "managed", "managing", "treatment", "treatments",
            "therapy", "therapies", "monitoring", "organ", "specific",
            "disease", "disorder", "syndrome",
            "drug", "drugs", "label", "labels", "dose", "dosing", "dosage",
            "administration", "warning", "warnings", "precaution", "precautions",
            "contraindication", "contraindications", "patient", "patients",
            "medicine", "medication",
        }
        blocked: set[str] = set(stopwords)
        for drug in normalized_drugs:
            blocked.update(re.findall(r"[a-z0-9]+", str(drug).lower()))
        terms: List[str] = []
        seen: set[str] = set()
        for query_text in query_texts:
            for token in re.findall(r"[a-z0-9]+", str(query_text or "").lower()):
                if len(token) < 4 or token in blocked or token in seen:
                    continue
                seen.add(token)
                terms.append(token)
                if len(terms) >= 6:
                    return terms
        return terms
