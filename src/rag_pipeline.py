"""
Elixir Medical RAG Pipeline.

Optimized pipeline for fast, comprehensive medical responses:
1. Query preprocessing with decomposition
2. Hybrid retrieval (dense + sparse lexical search)
3. DeepInfra Qwen3-Reranker with paper aggregation and evidence hierarchy
4. Direct LLM synthesis with ELIXIR system prompt via DeepInfra

Designed for speed and quality clinical decision support.
"""

import logging
import re
import hashlib
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Generator, Optional, Callable, Union
import httpx
from openai import OpenAI

from .config import (
    DEEPINFRA_API_KEY,
    DEEPINFRA_BASE_URL,
    OPENCODE_API_KEY,
    OPENCODE_BASE_URL,
    LLM_PROVIDER,
    LLM_CHAT_TIMEOUT_SECONDS,
    LLM_RETRY_COUNT,
    LLM_RETRY_DELAY,
    LLM_MAX_COMPLETION_TOKENS,
    LLM_REASONING_EFFORT,
    LLM_MODEL,
    QUERY_PREPROCESSOR_LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    MAX_DAILYMED_PER_DRUG,
    DAILYMED_MAX_FINAL_REFERENCES,
    PMC_PUBMED_ARTICLE_RELEVANCE_THRESHOLD,
    PMC_PUBMED_MIN_FINAL_ARTICLES,
    RETRIEVAL_CHUNK_LIMIT,
    RERANK_EVAL_LIMIT,
    RERANK_TOP_CHUNKS,
    FINAL_TOP_ARTICLES,
    FINAL_RECENCY_WINDOW_YEARS,
    ENTITY_FILTER_ENABLED,
    UPSTREAM_HTTP_MAX_CONNECTIONS,
    UPSTREAM_HTTP_MAX_KEEPALIVE,
    UPSTREAM_HTTP_KEEPALIVE_EXPIRY,
)
from .query_preprocessor import QueryPreprocessor, LLMProcessedQuery
from .retriever_factory import create_retriever
from .reranker import PaperFinderWithReranker, compute_metadata_multipliers, compute_query_focus_score
from .retry_utils import retry_with_exponential_backoff
from .prompts import (
    ELIXIR_FALLBACK_SYSTEM_PROMPT,
    ELIXIR_SYSTEM_PROMPT,
    ELIXIR_USMLE_FALLBACK_SYSTEM_PROMPT,
    ELIXIR_USMLE_SYSTEM_PROMPT,
)
from .query_cache import QueryCache
try:
    from scripts.ingestion_utils import get_evidence_hierarchy_levels
except Exception:
    def get_evidence_hierarchy_levels() -> Dict[str, Any]:
        return {
            "levels": [
                {"grade": "A", "level": 1, "label": "Highest evidence", "terms": []},
                {"grade": "B", "level": 2, "label": "High evidence", "terms": []},
                {"grade": "C", "level": 3, "label": "Moderate evidence", "terms": []},
                {"grade": "D", "level": 4, "label": "Lower evidence", "terms": []},
            ]
        }

logger = logging.getLogger(__name__)

USMLE_OPTION_RETRIEVAL_LIMIT = 5

CITATION_GROUP_PATTERN = re.compile(r"\[([0-9,\-\s]+)\]")
SOURCE_STYLE_CITATION_PATTERN = re.compile(
    r"\bsource\s*(\d+(?:\s*-\s*\d+)?(?:\s*,\s*\d+(?:\s*-\s*\d+)?)*)\b",
    re.IGNORECASE,
)
FOLLOW_UP_MARKER = "<FOLLOW_UP_QUESTIONS_JSON>"
CITATION_BRACKET_TRANSLATION = str.maketrans(
    {
        "【": "[",
        "】": "]",
        "［": "[",
        "］": "]",
        "〖": "[",
        "〗": "]",
    }
)
FOLLOW_UP_CONTEXT_DEPENDENT_PATTERN = re.compile(
    r"\b(this|that|these|those|above|below|previous|prior|earlier|answer|response|discussion)\b",
    re.IGNORECASE,
)

class MedicalRAGPipeline:
    """
    Elixir Medical RAG Pipeline.
    
    Optimized flow:
    1. Preprocess query (decompose, extract filters)
    2. Retrieve passages from retrieval backend
    3. Rerank passages with DeepInfra Qwen3-Reranker
    4. Aggregate to paper level
    5. Direct LLM synthesis with ELIXIR prompt (DeepInfra)
    """

    def __init__(
        self,
        model: str = LLM_MODEL,
        query_preprocessor_model: str = QUERY_PREPROCESSOR_LLM_MODEL,
        n_retrieval: int = RETRIEVAL_CHUNK_LIMIT,
        n_rerank: int = RERANK_TOP_CHUNKS,
        context_threshold: float = 0.0
    ):

        """Initialize pipeline components."""
        logger.info("🚀 Initializing Elixir Medical RAG Pipeline...")

        if LLM_PROVIDER == "deepinfra" and not DEEPINFRA_API_KEY:
            raise ValueError("DEEPINFRA_API_KEY not set")
        if LLM_PROVIDER == "opencode" and not OPENCODE_API_KEY:
            raise ValueError("OPENCODE_API_KEY not set")

        self.model = model
        self.query_preprocessor_model = query_preprocessor_model
        self.final_top_articles = FINAL_TOP_ARTICLES
        self.min_final_articles = max(1, PMC_PUBMED_MIN_FINAL_ARTICLES)
        self.article_relevance_threshold = max(0.0, PMC_PUBMED_ARTICLE_RELEVANCE_THRESHOLD)
        self.final_recency_window_years = max(1, FINAL_RECENCY_WINDOW_YEARS)
        self.dailymed_max_final_references = DAILYMED_MAX_FINAL_REFERENCES
        self._last_recency_stats = {
            "recent_kept_non_dailymed": 0,
            "dailymed_kept": 0,
            "older_backfilled": 0,
            "unknown_non_dailymed_excluded": 0,
            "recent_cutoff_year": None,
        }
        self._last_context_stats = {
            "pmc_recent_fulltext_used": 0,
        }
        self._last_timing_stats = {
            "preprocess_ms": 0,
            "embedding_ms": 0,
            "literature_retrieval_ms": 0,
            "dailymed_retrieval_ms": 0,
            "rerank_ms": 0,
            "aggregation_ms": 0,
            "pdf_check_ms": 0,
        }
        self._last_retrieval_stats = {
            "rerank_passages_retrieved": 0,
            "rerank_passages_scored": 0,
            "rerank_passages_kept": 0,
            "usmle_option_query_count": 0,
            "literature_query_count": 0,
            "retrieved_per_query_limit": 0,
            "retrieval_mode": "standard_hybrid_rerank",
            "embedding_texts_requested": 0,
            "embedding_cache_hits": 0,
            "embedding_api_calls": 0,
            "literature_retrieval_cache_hit": False,
            "literature_bucket_limits": {},
            "dailymed_retrieval_cache_hit": False,
            "dailymed_drug_query_detected": False,
            "dailymed_raw_hits": 0,
            "dailymed_identity_matched_rows": 0,
            "dailymed_grouped_labels": 0,
            "dailymed_appended_labels": 0,
            "dailymed_final_sources_returned": 0,
            "dailymed_match_strategy": "none",
        }
        self.llm_provider = LLM_PROVIDER
        self._openai_http_client = self._build_openai_http_client(timeout_seconds=LLM_CHAT_TIMEOUT_SECONDS)
        self.llm_client = self._create_llm_client(self.llm_provider)
        
        # Components
        self.preprocessor = QueryPreprocessor(model=query_preprocessor_model)
        self.retriever = create_retriever(n_retrieval=n_retrieval)

        # Initialize reranker (DeepInfra Qwen only)
        self.paper_finder = PaperFinderWithReranker(
            n_rerank=n_rerank,
            context_threshold=context_threshold
        )
        logger.info("✅ Reranker initialized (DeepInfra %s)", getattr(self.paper_finder.reranker_engine, "model", "Qwen reranker"))
        self.evidence_hierarchy = get_evidence_hierarchy_levels()
        
        # Initialize Query Cache
        self.cache = QueryCache()
        prompt_material = "\n||\n".join(
            [
                ELIXIR_SYSTEM_PROMPT,
                ELIXIR_USMLE_SYSTEM_PROMPT,
                ELIXIR_FALLBACK_SYSTEM_PROMPT,
                ELIXIR_USMLE_FALLBACK_SYSTEM_PROMPT,
            ]
        )
        prompt_hash = hashlib.sha256(prompt_material.encode("utf-8")).hexdigest()[:16]
        reranker_model = getattr(getattr(self.paper_finder, "reranker_engine", None), "model", "unknown")
        self._cache_key_context = {
            "pipeline": "elixir",
            "pipeline_cache_version": 6,
            "llm_model": self.model,
            "reranker_model": reranker_model,
            "embedding_model": getattr(self.retriever, "embedding_model", "unknown"),
            "collection_name": getattr(self.retriever, "collection_name", "unknown"),
                "entity_filter_enabled": ENTITY_FILTER_ENABLED,
                "prompt_hash": prompt_hash,
                "n_retrieval": n_retrieval,
                "article_relevance_threshold": self.article_relevance_threshold,
                "min_final_articles": self.min_final_articles,
                "rerank_eval_limit": RERANK_EVAL_LIMIT,
                "n_rerank": n_rerank,
                "final_top_articles": self.final_top_articles,
            }
        
        logger.info("✅ Pipeline initialized (Elixir direct synthesis, provider=%s)", self.llm_provider)

    def _build_openai_http_client(self, timeout_seconds: float) -> httpx.Client:
        return httpx.Client(
            limits=httpx.Limits(
                max_connections=UPSTREAM_HTTP_MAX_CONNECTIONS,
                max_keepalive_connections=UPSTREAM_HTTP_MAX_KEEPALIVE,
                keepalive_expiry=UPSTREAM_HTTP_KEEPALIVE_EXPIRY,
            ),
            timeout=timeout_seconds,
        )

    def _create_llm_client(self, provider: str):
        """Build an LLM client for a supported provider."""
        if provider == "deepinfra":
            return OpenAI(
                api_key=DEEPINFRA_API_KEY,
                base_url=DEEPINFRA_BASE_URL,
                timeout=LLM_CHAT_TIMEOUT_SECONDS,
                http_client=self._openai_http_client,
            )
        if provider == "opencode":
            return OpenAI(
                api_key=OPENCODE_API_KEY,
                base_url=OPENCODE_BASE_URL,
                timeout=LLM_CHAT_TIMEOUT_SECONDS,
                http_client=self._openai_http_client,
            )
        raise ValueError(f"Unsupported LLM provider: {provider}")

    def _build_chat_completion_kwargs(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        provider: str,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Build provider-specific request kwargs for chat completions."""
        request_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
        }
        if stream:
            request_kwargs["stream"] = True
        if provider == "opencode":
            request_kwargs["max_tokens"] = LLM_MAX_COMPLETION_TOKENS
        else:
            request_kwargs["max_completion_tokens"] = LLM_MAX_COMPLETION_TOKENS
        if LLM_REASONING_EFFORT:
            request_kwargs["reasoning_effort"] = LLM_REASONING_EFFORT
        return request_kwargs

    @staticmethod
    def _thread_context_fingerprint(thread_context: Optional[Dict[str, Any]]) -> str:
        if not isinstance(thread_context, dict):
            return ""
        normalized = {
            "initial_query": str(thread_context.get("initial_query", "")).strip().lower(),
            "latest_user_query": str(thread_context.get("latest_user_query", "")).strip().lower(),
            "latest_assistant_answer": str(thread_context.get("latest_assistant_answer", "")).strip().lower(),
            "follow_up_count": int(thread_context.get("follow_up_count", 0) or 0),
            "is_follow_up": bool(thread_context.get("is_follow_up", False)),
        }
        payload = json.dumps(normalized, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _cache_get(
        self,
        query: str,
        thread_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Read query cache using pipeline context so stale entries are isolated."""
        if not self.cache.enabled:
            return None
        cache_context = dict(self._cache_key_context)
        cache_context["thread_context_fp"] = self._thread_context_fingerprint(thread_context)
        cached = self.cache.get(query, **cache_context)
        return self._normalize_cached_response(cached)

    def _cache_set(
        self,
        query: str,
        response: Dict[str, Any],
        thread_context: Optional[Dict[str, Any]] = None,
    ):
        """Write query cache using pipeline context."""
        if not self.cache.enabled:
            return
        cache_context = dict(self._cache_key_context)
        cache_context["thread_context_fp"] = self._thread_context_fingerprint(thread_context)
        self.cache.set(query, response, **cache_context)

    @staticmethod
    def _parse_citation_group(
        group_text: str,
        *,
        max_index: Optional[int] = None,
        valid_indices: Optional[set[int]] = None,
    ) -> List[int]:
        """Parse a citation group like '1, 3-5' into ordered unique indices."""
        if not group_text:
            return []

        parsed: List[int] = []
        seen: set[int] = set()

        for raw_part in group_text.split(","):
            token = raw_part.strip().replace("–", "-")
            if not token:
                continue

            values: List[int] = []
            if "-" in token:
                bounds = [x.strip() for x in token.split("-", 1)]
                if len(bounds) != 2 or not bounds[0].isdigit() or not bounds[1].isdigit():
                    continue
                start, end = int(bounds[0]), int(bounds[1])
                if start > end:
                    continue
                values = list(range(start, end + 1))
            else:
                if not token.isdigit():
                    continue
                values = [int(token)]

            for idx in values:
                if idx < 1:
                    continue
                if max_index is not None and idx > max_index:
                    continue
                if valid_indices is not None and idx not in valid_indices:
                    continue
                if idx in seen:
                    continue
                parsed.append(idx)
                seen.add(idx)

        return parsed

    @staticmethod
    def _format_citation_group(indices: List[int]) -> str:
        """Format citation indices into compact ascending display text."""
        if not indices:
            return "[]"

        ordered = sorted(dict.fromkeys(indices))
        ranges: List[str] = []
        start = ordered[0]
        end = ordered[0]

        for idx in ordered[1:]:
            if idx == end + 1:
                end = idx
                continue
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = idx
            end = idx

        ranges.append(f"{start}-{end}" if start != end else str(start))
        return "[" + ", ".join(ranges) + "]"

    @staticmethod
    def _normalize_citation_markers(text: str) -> str:
        """Normalize alternate Unicode citation brackets to ASCII brackets."""
        if not text:
            return text
        return text.translate(CITATION_BRACKET_TRANSLATION)

    @staticmethod
    def _normalize_table_source_columns(text: str) -> str:
        """Wrap bare numeric citations in markdown Source/Citation table columns."""
        if not text:
            return text

        lines = text.splitlines()
        source_column_index: Optional[int] = None
        normalized_lines: List[str] = []

        def _is_separator_row(cells: List[str]) -> bool:
            return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)

        def _wrap_bare_citation_cell(value: str) -> str:
            trimmed = value.strip()
            if not re.fullmatch(r"\d+(?:\s*-\s*\d+)?(?:\s*,\s*\d+(?:\s*-\s*\d+)?)*", trimmed):
                return value
            return value.replace(trimmed, f"[{trimmed}]")

        for line in lines:
            if "|" not in line:
                normalized_lines.append(line)
                continue

            raw_cells = line.split("|")
            if len(raw_cells) < 3:
                normalized_lines.append(line)
                continue

            cells = raw_cells[1:-1]
            if source_column_index is None:
                header_index = next(
                    (
                        idx for idx, cell in enumerate(cells)
                        if cell.strip().lower() in {"source", "sources", "citation", "citations"}
                    ),
                    None,
                )
                if header_index is not None:
                    source_column_index = header_index
                    normalized_lines.append(line)
                    continue

            if _is_separator_row(cells):
                normalized_lines.append(line)
                continue

            if source_column_index is not None and source_column_index < len(cells):
                cells[source_column_index] = _wrap_bare_citation_cell(cells[source_column_index])
                normalized_lines.append("|" + "|".join(cells) + "|")
                continue

            normalized_lines.append(line)

        return "\n".join(normalized_lines)

    @classmethod
    def _force_bracketed_citations(cls, text: str) -> str:
        """Normalize supported citation variants into bracketed citations."""
        if not text:
            return text
        normalized = cls._normalize_citation_markers(text)
        normalized = SOURCE_STYLE_CITATION_PATTERN.sub(r"[\1]", normalized)
        normalized = cls._normalize_table_source_columns(normalized)
        return normalized

    def _normalize_answer_and_sources(
        self,
        answer_text: str,
        sources: List[Dict[str, Any]],
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Renumber citations by first appearance in the final answer and align sources.
        Uncited DailyMed entries remain available without a display citation.
        """
        if not answer_text or not sources:
            return answer_text, list(sources or [])

        answer_text = self._force_bracketed_citations(answer_text)

        cloned_sources = [dict(source) for source in sources]
        source_by_original_idx: Dict[int, Dict[str, Any]] = {}
        for source in cloned_sources:
            try:
                source_by_original_idx[int(source.get("citation_index"))] = source
            except (TypeError, ValueError):
                continue

        valid_indices = set(source_by_original_idx)
        if not valid_indices:
            return answer_text, cloned_sources

        max_index = max(valid_indices)
        def parse_group(group_text: str) -> list[int]:
            return self._parse_citation_group(
                group_text,
                max_index=max_index,
                valid_indices=valid_indices,
            )

        first_appearance_map: Dict[int, int] = {}
        for group_text in CITATION_GROUP_PATTERN.findall(answer_text):
            for original_idx in parse_group(group_text):
                first_appearance_map.setdefault(original_idx, len(first_appearance_map) + 1)

        if not first_appearance_map:
            return answer_text, cloned_sources

        def _rewrite_group(match: re.Match[str]) -> str:
            cited_indices = parse_group(match.group(1))
            if not cited_indices:
                return match.group(0)
            remapped = [first_appearance_map[original_idx] for original_idx in cited_indices]
            if not remapped:
                return match.group(0)
            return self._format_citation_group(remapped)

        normalized_answer = CITATION_GROUP_PATTERN.sub(_rewrite_group, answer_text)

        cited_sources: List[Dict[str, Any]] = []
        for original_idx, display_idx in sorted(first_appearance_map.items(), key=lambda item: item[1]):
            source = dict(source_by_original_idx[original_idx])
            source["citation_index"] = display_idx
            cited_sources.append(source)

        uncited_sources: List[Dict[str, Any]] = []
        for source in cloned_sources:
            try:
                idx = int(source.get("citation_index"))
            except (TypeError, ValueError):
                idx = None
            if idx in first_appearance_map:
                continue
            if self._is_dailymed_row(source):
                source["citation_index"] = None
            uncited_sources.append(source)

        return normalized_answer, cited_sources + uncited_sources

    def _normalize_cached_response(self, response: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Upgrade cached responses lazily to the current citation display contract."""
        if not response or not isinstance(response, dict):
            return response
        answer = response.get("answer")
        sources = response.get("sources")
        if not isinstance(answer, str) or not isinstance(sources, list):
            return response

        filtered_sources = self._filter_sources_to_citations(answer, sources)
        normalized_answer, normalized_sources = self._normalize_answer_and_sources(answer, filtered_sources)
        upgraded = dict(response)
        upgraded["answer"] = normalized_answer
        upgraded["sources"] = normalized_sources
        follow_ups = upgraded.get("follow_up_questions")
        if not isinstance(follow_ups, list):
            upgraded["follow_up_questions"] = []
        else:
            upgraded["follow_up_questions"] = [str(item).strip() for item in follow_ups if isinstance(item, str) and item.strip()]
        return upgraded

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 2:
                return "\n".join(lines[1:-1]).strip()
        return stripped

    @staticmethod
    def _tokenize_text(text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9]+", (text or "").lower()) if token}

    def _is_knowledge_expanding_followup(self, query: str, followup: str) -> bool:
        query_norm = " ".join((query or "").strip().lower().split())
        followup_norm = " ".join((followup or "").strip().lower().split())
        if not followup_norm or followup_norm == query_norm:
            return False

        query_tokens = self._tokenize_text(query_norm)
        followup_tokens = self._tokenize_text(followup_norm)
        if not query_tokens or not followup_tokens:
            return False

        overlap = len(query_tokens & followup_tokens)
        union = len(query_tokens | followup_tokens)
        jaccard = (overlap / union) if union else 1.0
        return jaccard < 0.85

    def _validate_followup_question(self, query: str, candidate: Any) -> Optional[str]:
        if not isinstance(candidate, str):
            return None
        normalized = " ".join(candidate.strip().split())
        if not normalized:
            return None
        if not normalized.endswith("?"):
            return None
        if FOLLOW_UP_CONTEXT_DEPENDENT_PATTERN.search(normalized):
            return None
        if not self._is_knowledge_expanding_followup(query, normalized):
            return None
        return normalized

    def _parse_followups_json_payload(self, payload_text: str) -> List[str]:
        payload = self._strip_code_fence(payload_text)
        candidates: Any = None
        try:
            candidates = json.loads(payload)
        except Exception:
            candidates = None

        if candidates is None:
            for opening, closing in (("{", "}"), ("[", "]")):
                start = payload.find(opening)
                end = payload.rfind(closing)
                if start == -1 or end == -1 or end < start:
                    continue
                snippet = payload[start : end + 1]
                try:
                    candidates = json.loads(snippet)
                    break
                except Exception:
                    continue

        if isinstance(candidates, dict):
            raw_followups = candidates.get("follow_up_questions")
            if isinstance(raw_followups, list):
                return raw_followups
            return []
        if isinstance(candidates, list):
            return candidates
        return []

    def _extract_answer_and_followups(self, query: str, llm_output: str) -> tuple[str, List[str]]:
        output = (llm_output or "").strip()
        if not output:
            return "", []

        if FOLLOW_UP_MARKER not in output:
            return output, []

        answer_part, payload = output.rsplit(FOLLOW_UP_MARKER, 1)
        answer = answer_part.rstrip()
        raw_followups = self._parse_followups_json_payload(payload)
        validated: List[str] = []
        seen: set[str] = set()
        for candidate in raw_followups:
            parsed = self._validate_followup_question(query, candidate)
            if not parsed:
                continue
            key = parsed.lower()
            if key in seen:
                continue
            seen.add(key)
            validated.append(parsed)
            if len(validated) >= 3:
                break
        return answer, validated

    def _build_retrieval_stats(
        self,
        passages_retrieved: int,
        papers_after_aggregation: int,
        abstracts_used: int,
    ) -> Dict[str, Any]:
        recency_stats = self._last_recency_stats or {}
        context_stats = self._last_context_stats or {}
        return {
            "passages_retrieved": passages_retrieved,
            "papers_after_aggregation": papers_after_aggregation,
            "abstracts_used": abstracts_used,
            "preprocess_ms": self._last_timing_stats.get("preprocess_ms", 0),
            "embedding_ms": self._last_timing_stats.get("embedding_ms", 0),
            "literature_retrieval_ms": self._last_timing_stats.get("literature_retrieval_ms", 0),
            "dailymed_retrieval_ms": self._last_timing_stats.get("dailymed_retrieval_ms", 0),
            "rerank_ms": self._last_timing_stats.get("rerank_ms", 0),
            "aggregation_ms": self._last_timing_stats.get("aggregation_ms", 0),
            "pdf_check_ms": self._last_timing_stats.get("pdf_check_ms", 0),
            "rerank_passages_retrieved": self._last_retrieval_stats.get("rerank_passages_retrieved", 0),
            "rerank_passages_scored": self._last_retrieval_stats.get("rerank_passages_scored", 0),
            "rerank_passages_kept": self._last_retrieval_stats.get("rerank_passages_kept", 0),
            "usmle_option_query_count": self._last_retrieval_stats.get("usmle_option_query_count", 0),
            "literature_query_count": self._last_retrieval_stats.get("literature_query_count", 0),
            "retrieved_per_query_limit": self._last_retrieval_stats.get("retrieved_per_query_limit", 0),
            "retrieval_mode": self._last_retrieval_stats.get("retrieval_mode", "standard_hybrid_rerank"),
            "embedding_texts_requested": self._last_retrieval_stats.get("embedding_texts_requested", 0),
            "embedding_cache_hits": self._last_retrieval_stats.get("embedding_cache_hits", 0),
            "embedding_api_calls": self._last_retrieval_stats.get("embedding_api_calls", 0),
            "literature_retrieval_cache_hit": self._last_retrieval_stats.get("literature_retrieval_cache_hit", False),
            "literature_bucket_limits": self._last_retrieval_stats.get("literature_bucket_limits", {}),
            "dailymed_retrieval_cache_hit": self._last_retrieval_stats.get("dailymed_retrieval_cache_hit", False),
            "recent_articles_kept": recency_stats.get("recent_kept_non_dailymed", 0),
            "older_high_evidence_backfilled": recency_stats.get("older_backfilled", 0),
            "unknown_year_non_dailymed_excluded": recency_stats.get("unknown_non_dailymed_excluded", 0),
            "pmc_recent_fulltext_used": context_stats.get("pmc_recent_fulltext_used", 0),
            "dailymed_drug_query_detected": self._last_retrieval_stats.get("dailymed_drug_query_detected", False),
            "dailymed_raw_hits": self._last_retrieval_stats.get("dailymed_raw_hits", 0),
            "dailymed_identity_matched_rows": self._last_retrieval_stats.get("dailymed_identity_matched_rows", 0),
            "dailymed_grouped_labels": self._last_retrieval_stats.get("dailymed_grouped_labels", 0),
            "dailymed_appended_labels": self._last_retrieval_stats.get("dailymed_appended_labels", 0),
            "dailymed_final_sources_returned": self._last_retrieval_stats.get("dailymed_final_sources_returned", 0),
            "dailymed_match_strategy": self._last_retrieval_stats.get("dailymed_match_strategy", "none"),
        }

    def _reset_run_stats(self) -> None:
        self._last_recency_stats = {
            "recent_kept_non_dailymed": 0,
            "dailymed_kept": 0,
            "older_backfilled": 0,
            "unknown_non_dailymed_excluded": 0,
            "recent_cutoff_year": None,
        }
        self._last_context_stats = {
            "pmc_recent_fulltext_used": 0,
            "recent_cutoff_year": None,
        }
        self._last_timing_stats = {
            "preprocess_ms": 0,
            "embedding_ms": 0,
            "literature_retrieval_ms": 0,
            "dailymed_retrieval_ms": 0,
            "rerank_ms": 0,
            "aggregation_ms": 0,
            "pdf_check_ms": 0,
        }
        self._last_retrieval_stats = {
            "rerank_passages_retrieved": 0,
            "rerank_passages_scored": 0,
            "rerank_passages_kept": 0,
            "usmle_option_query_count": 0,
            "literature_query_count": 0,
            "retrieved_per_query_limit": 0,
            "retrieval_mode": "standard_hybrid_rerank",
            "embedding_texts_requested": 0,
            "embedding_cache_hits": 0,
            "embedding_api_calls": 0,
            "literature_retrieval_cache_hit": False,
            "literature_bucket_limits": {},
            "dailymed_retrieval_cache_hit": False,
            "dailymed_drug_query_detected": False,
            "dailymed_raw_hits": 0,
            "dailymed_identity_matched_rows": 0,
            "dailymed_grouped_labels": 0,
            "dailymed_appended_labels": 0,
            "dailymed_final_sources_returned": 0,
            "dailymed_match_strategy": "none",
        }
    
    # =========================================================================
    # Step 1: Query Preprocessing
    # =========================================================================
    
    def preprocess_query(
        self,
        query: str,
        thread_context: Optional[Dict[str, Any]] = None,
    ) -> LLMProcessedQuery:
        """Decompose query into compact retrieval queries and routing hints."""
        logger.info("📝 Step 1: Query Preprocessing")
        start_time = time.perf_counter()
        result = self.preprocessor.decompose_query(query, thread_context=thread_context)
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        self._last_timing_stats["preprocess_ms"] = elapsed_ms
        logger.info(f"   Preprocessed: {result.primary_query}")
        logger.info("   Query preprocessing completed in %d ms", elapsed_ms)
        return result

    @staticmethod
    def _is_usmle_option_path(processed_query: LLMProcessedQuery) -> bool:
        if not processed_query or not processed_query.decomposed:
            return False
        return bool(
            processed_query.decomposed.is_usmle_query
            and processed_query.option_parse_success
            and processed_query.option_queries
        )

    @staticmethod
    def _stable_passage_key(passage: Dict[str, Any]) -> str:
        return str(
            passage.get("chunk_id")
            or passage.get("id")
            or passage.get("pmcid")
            or passage.get("doc_id")
            or passage.get("corpus_id")
            or ""
        ).strip()

    def _dedupe_passages(self, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: Dict[str, Dict[str, Any]] = {}
        for passage in passages:
            key = self._stable_passage_key(passage)
            if not key:
                continue
            if key not in deduped:
                deduped[key] = passage
                continue
            existing_score = self._safe_float(deduped[key].get("score", 0.0))
            current_score = self._safe_float(passage.get("score", 0.0))
            if current_score > existing_score:
                deduped[key] = passage
        merged = list(deduped.values())
        merged.sort(key=lambda row: self._safe_float(row.get("score", 0.0)), reverse=True)
        return merged
    
    # =========================================================================
    # Step 2: Retrieval
    # =========================================================================
    
    def retrieve_passages(
        self,
        processed_query: LLMProcessedQuery
    ) -> tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        Optional[Callable[[], List[Dict[str, Any]]]],
    ]:
        """
        Retrieve PMC/PubMed passages with fixed raw source quotas.

        The non-DailyMed path is intentionally simple:
        - PMC dense top 75 + PMC BM25 top 25
        - PubMed dense top 75 + PubMed BM25 top 25
        - merge/dedupe raw buckets
        - send all unique passages to reranking

        DailyMed continues to run on its existing parallel side path.
        """
        from concurrent.futures import ThreadPoolExecutor
        
        logger.info("🔍 Step 2: Passage Retrieval (Fixed Source Buckets)")
        
        dailymed_results = []
        self._last_timing_stats["embedding_ms"] = 0
        self._last_timing_stats["literature_retrieval_ms"] = 0
        self._last_timing_stats["dailymed_retrieval_ms"] = 0
        is_usmle_option_path = self._is_usmle_option_path(processed_query)

        if is_usmle_option_path and hasattr(self.retriever, "retrieve_dense_topk_per_query"):
            retrieval_queries = (
                list(processed_query.option_queries)
                if processed_query.option_parse_success and processed_query.option_queries
                else list(processed_query.retrieval_queries or [processed_query.primary_query])
            )
            retrieval_queries = [q for q in retrieval_queries if str(q or "").strip()]
            per_query_limit = USMLE_OPTION_RETRIEVAL_LIMIT
            retrieval_start = time.perf_counter()
            dense_runs = self.retriever.retrieve_dense_topk_per_query(
                retrieval_queries,
                per_query_limit=per_query_limit,
            )
            self._last_timing_stats["literature_retrieval_ms"] = int((time.perf_counter() - retrieval_start) * 1000)
            merged_passages: List[Dict[str, Any]] = []
            for rows in dense_runs:
                merged_passages.extend(rows)
            deduped_passages = self._dedupe_passages(merged_passages)
            self._last_retrieval_stats.update(
                {
                    "usmle_option_query_count": len(retrieval_queries),
                    "retrieved_per_query_limit": per_query_limit,
                    "retrieval_mode": "usmle_option_dense_no_rerank",
                    "embedding_texts_requested": getattr(self.retriever, "last_embedding_stats", {}).get("texts_requested", 0),
                    "embedding_cache_hits": getattr(self.retriever, "last_embedding_stats", {}).get("cache_hits", 0),
                    "embedding_api_calls": getattr(self.retriever, "last_embedding_stats", {}).get("api_calls", 0),
                    "literature_retrieval_cache_hit": False,
                    "dailymed_retrieval_cache_hit": False,
                    "dailymed_drug_query_detected": False,
                    "dailymed_raw_hits": 0,
                    "dailymed_identity_matched_rows": 0,
                    "dailymed_grouped_labels": 0,
                    "dailymed_match_strategy": "none",
                }
            )
            logger.info(
                "   USMLE option retrieval complete: queries=%d, dense_limit=%d, merged=%d, deduped=%d",
                len(retrieval_queries),
                per_query_limit,
                len(merged_passages),
                len(deduped_passages),
            )
            logger.info(
                "   Retrieval timings: embed=%d ms | literature=%d ms | dailymed=%d ms",
                self._last_timing_stats["embedding_ms"],
                self._last_timing_stats["literature_retrieval_ms"],
                self._last_timing_stats["dailymed_retrieval_ms"],
            )
            return deduped_passages, [], None
        
        retrieval_query = str(processed_query.primary_query or "").strip()
        if not retrieval_query:
            retrieval_query = str(processed_query.keyword_query or "").strip()
        if not retrieval_query:
            retrieval_query = str(processed_query.original_query or "").strip()

        is_drug_query = False
        drug_names: List[str] = []
        dailymed_keywords: List[str] = []
        if processed_query.decomposed is not None:
            # Trust decomposition output, including empty [] when no drugs are present.
            is_drug_query = bool(processed_query.decomposed.is_drug_query)
            drug_names = processed_query.decomposed.drug_names or []
            dailymed_keywords = processed_query.decomposed.dailymed_keywords or drug_names
        else:
            # Fallback only: infer drugs from query text when decomposition is unavailable.
            drug_names = self._extract_drug_names(processed_query.original_query, processed_query.primary_query)
            dailymed_keywords = list(drug_names)
            is_drug_query = bool(drug_names)

        dailymed_enabled = True
        # Always run the unified DailyMed retrieval path using query variants
        # and any extracted keywords; keep the flag only for telemetry.
        if dailymed_keywords and not is_drug_query:
            is_drug_query = True

        logger.info(
            "   Retrieval query: %s | DailyMed Enabled: %s | DailyMed Drugs: %s | DailyMed Keywords: %s",
            retrieval_query,
            dailymed_enabled,
            len(drug_names),
            len(dailymed_keywords),
        )
        self._last_retrieval_stats.update(
            {
                "dailymed_drug_query_detected": is_drug_query,
                "usmle_option_query_count": 0,
                "literature_query_count": 1 if retrieval_query else 0,
                "retrieved_per_query_limit": 0,
                "retrieval_mode": "standard_single_query_hybrid_rerank",
                "literature_bucket_limits": {
                    "pmc_dense": self.retriever.pmc_dense_limit,
                    "pmc_bm25": self.retriever.pmc_bm25_limit,
                    "pubmed_dense": self.retriever.pubmed_dense_limit,
                    "pubmed_bm25": self.retriever.pubmed_bm25_limit,
                },
            }
        )
        dailymed_filter_phrases: List[str] = []
        if processed_query.decomposed is not None:
            dailymed_filter_phrases.extend(processed_query.decomposed.corrected_entities or [])
            dailymed_filter_phrases.extend(processed_query.decomposed.key_entities or [])

        shared_embedding_lookup: Dict[str, Optional[List[float]]] = {}
        if hasattr(self.retriever, "build_shared_embedding_inputs") and hasattr(self.retriever, "build_embedding_lookup"):
            embedding_start = time.perf_counter()
            shared_embedding_inputs = self.retriever.build_shared_embedding_inputs(
                retrieval_queries=[retrieval_query] if retrieval_query else [],
                dailymed_keywords=dailymed_keywords,
                context_query=processed_query.original_query or processed_query.primary_query,
                query_variants=[retrieval_query] if retrieval_query else [],
            )
            if shared_embedding_inputs:
                shared_embedding_lookup = self.retriever.build_embedding_lookup(shared_embedding_inputs)
            self._last_timing_stats["embedding_ms"] = int((time.perf_counter() - embedding_start) * 1000)
            embedding_stats = getattr(self.retriever, "last_embedding_stats", {}) or {}
            self._last_retrieval_stats.update(
                {
                    "embedding_texts_requested": embedding_stats.get("texts_requested", 0),
                    "embedding_cache_hits": embedding_stats.get("cache_hits", 0),
                    "embedding_api_calls": embedding_stats.get("api_calls", 0),
                }
            )
            logger.info(
                "   Shared embedding prep: texts=%d cache_hits=%d api_calls=%d in %d ms",
                self._last_retrieval_stats["embedding_texts_requested"],
                self._last_retrieval_stats["embedding_cache_hits"],
                self._last_retrieval_stats["embedding_api_calls"],
                self._last_timing_stats["embedding_ms"],
            )

        def run_dailymed_search(keywords: List[str]) -> tuple[List[Dict[str, Any]], int]:
            start_time = time.perf_counter()
            try:
                results = self.retriever.search_dailymed_by_drug(
                    keywords,
                    context_query=processed_query.original_query or processed_query.primary_query,
                    query_variants=[retrieval_query] if retrieval_query else [],
                    filter_phrases=dailymed_filter_phrases,
                    embedding_lookup=shared_embedding_lookup,
                )
                return results, int((time.perf_counter() - start_time) * 1000)
            except Exception as e:
                logger.warning(f"DailyMed search failed: {e}")
                return [], int((time.perf_counter() - start_time) * 1000)
        
        dm_executor = ThreadPoolExecutor(max_workers=1)
        dm_future = dm_executor.submit(run_dailymed_search, dailymed_keywords)
        retrieval_start = time.perf_counter()
        all_passages = self.retriever.retrieve_fixed_source_buckets(
            retrieval_query,
            embedding_lookup=shared_embedding_lookup,
        )
        self._last_timing_stats["literature_retrieval_ms"] = int((time.perf_counter() - retrieval_start) * 1000)
        fixed_bucket_stats = getattr(self.retriever, "last_fixed_bucket_search_stats", {}) or {}
        self._last_retrieval_stats["literature_retrieval_cache_hit"] = fixed_bucket_stats.get("cache_hit", False)
        
        source_counts = {"pmc": 0, "pubmed": 0}
        for passage in all_passages:
            source_family = self._normalized_text(passage.get("source_family"))
            if source_family in source_counts:
                source_counts[source_family] += 1

        logger.info(
            "   Retrieved %s unique passages after raw bucket merge (PMC=%s, PubMed=%s)",
            len(all_passages),
            source_counts["pmc"],
            source_counts["pubmed"],
        )
        dailymed_resolved = False

        def resolve_dailymed_results() -> List[Dict[str, Any]]:
            nonlocal dailymed_results, dailymed_resolved
            if dailymed_resolved:
                return dailymed_results
            try:
                dailymed_results, dailymed_elapsed_ms = dm_future.result()
                self._last_timing_stats["dailymed_retrieval_ms"] = dailymed_elapsed_ms
                dailymed_stats = getattr(self.retriever, "last_dailymed_search_stats", {}) or {}
                self._last_retrieval_stats.update(
                    {
                        "literature_retrieval_cache_hit": fixed_bucket_stats.get("cache_hit", False),
                        "dailymed_retrieval_cache_hit": dailymed_stats.get("cache_hit", False),
                        "dailymed_drug_query_detected": dailymed_stats.get("drug_query_detected", is_drug_query),
                        "dailymed_raw_hits": dailymed_stats.get("raw_hits", 0),
                        "dailymed_identity_matched_rows": dailymed_stats.get("identity_matched_rows", 0),
                        "dailymed_grouped_labels": dailymed_stats.get("grouped_labels", 0),
                        "dailymed_match_strategy": dailymed_stats.get("match_strategy", "none"),
                    }
                )
                if dailymed_results:
                    logger.info(f"   ✅ DailyMed search found {len(dailymed_results)} results")
                else:
                    logger.info(
                        "   DailyMed search found 0 results (raw_hits=%s, identity_matched=%s, grouped=%s, strategy=%s)",
                        self._last_retrieval_stats["dailymed_raw_hits"],
                        self._last_retrieval_stats["dailymed_identity_matched_rows"],
                        self._last_retrieval_stats["dailymed_grouped_labels"],
                        self._last_retrieval_stats["dailymed_match_strategy"],
                    )
            except Exception as e:
                logger.warning(f"DailyMed task failed: {e}")
                dailymed_results = []
            finally:
                dm_executor.shutdown(wait=False)
                dailymed_resolved = True
                logger.info(
                    "   Retrieval timings: embed=%d ms | literature=%d ms | dailymed=%d ms | cache_hits(embedding=%d literature=%s dailymed=%s)",
                    self._last_timing_stats["embedding_ms"],
                    self._last_timing_stats["literature_retrieval_ms"],
                    self._last_timing_stats["dailymed_retrieval_ms"],
                    self._last_retrieval_stats["embedding_cache_hits"],
                    self._last_retrieval_stats["literature_retrieval_cache_hit"],
                    self._last_retrieval_stats.get("dailymed_retrieval_cache_hit", False),
                )
            return dailymed_results

        return all_passages, [], resolve_dailymed_results

    
    # Words that should NOT be extracted as drug names
    NON_DRUG_WORDS = {
        "guideline", "guidelines", "treatment", "treatments", "management",
        "screening", "diagnosis", "therapy", "therapies", "recommendation",
        "recommendations", "update", "review", "reviews", "criteria",
        "college", "rheumatology", "american", "european", "eular", "acr",
        "lupus", "nephritis", "arthritis", "disease", "syndrome", "disorder"
    }
    
    def _extract_drug_names(self, original_query: str, primary_query: str) -> List[str]:
        """
        Extract drug names from query using simple pattern matching.
        
        Looks for capitalized words that look like drug names (brand names)
        and known drug name patterns.
        """
        import re
        
        # Combine queries for better coverage
        text = f"{original_query} {primary_query}".lower()
        
        drug_names = set()
        
        # Common brand name drugs that are often asked about
        common_drugs = [
            "xeljanz", "tofacitinib", "humira", "adalimumab", "enbrel", "etanercept",
            "remicade", "infliximab", "rituxan", "rituximab", "orencia", "abatacept",
            "actemra", "tocilizumab", "simponi", "golimumab", "cimzia", "certolizumab",
            "rinvoq", "upadacitinib", "olumiant", "baricitinib", "kevzara", "sarilumab",
            "methotrexate", "mtx", "plaquenil", "hydroxychloroquine", "sulfasalazine",
            "leflunomide", "arava", "azathioprine", "imuran", "cyclosporine",
            "prednisone", "prednisolone", "methylprednisolone", "dexamethasone",
            "celebrex", "celecoxib", "meloxicam", "naproxen", "ibuprofen",
            "taltz", "ixekizumab", "cosentyx", "secukinumab", "stelara", "ustekinumab",
            "dupixent", "dupilumab", "otezla", "apremilast", "tremfya", "guselkumab",
        ]
        
        for drug in common_drugs:
            if re.search(r'\b' + re.escape(drug) + r'\b', text):
                drug_names.add(drug)
        
        # Extract capitalized words that might be brand names
        # Filter out NON_DRUG_WORDS to prevent false positives
        capitalized_words = re.findall(r'\b[A-Z][a-z]{3,}\b', original_query)
        for word in capitalized_words:
            word_lower = word.lower()
            if (len(word) >= 4 and 
                word_lower not in self.NON_DRUG_WORDS and
                word_lower not in ["what", "when", "where", "which", "this", "that", "with", "from", "have"]):
                drug_names.add(word_lower)
        
        return list(drug_names)[: max(1, int(self.dailymed_max_final_references or 15))]


    
    # =========================================================================
    # Step 3: Reranking & Aggregation
    # =========================================================================

    def _recent_cutoff_year(self, current_year: Optional[int] = None) -> int:
        year_now = current_year if current_year is not None else datetime.now().year
        return year_now - self.final_recency_window_years + 1

    @staticmethod
    def _normalized_text(value: Any) -> str:
        import math

        if value is None:
            return ""
        if isinstance(value, float) and math.isnan(value):
            return ""
        return str(value).strip().lower()

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _is_dailymed_row(cls, row: Any) -> bool:
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

    @staticmethod
    def _clean_text_value(value: Any) -> str:
        import math

        if value is None:
            return ""
        if isinstance(value, float) and math.isnan(value):
            return ""
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return ""
        return text

    @classmethod
    def _normalize_dailymed_section_key(cls, value: Any) -> str:
        return re.sub(
            r"\s+",
            " ",
            re.sub(r"[^a-z0-9]+", " ", cls._clean_text_value(value).lower()),
        ).strip()

    @classmethod
    def _normalized_dailymed_sections(cls, section_map: Any) -> Dict[str, str]:
        normalized_sections: Dict[str, str] = {}
        if not isinstance(section_map, dict):
            return normalized_sections

        for key, value in section_map.items():
            text_value = cls._clean_text_value(value)
            if not text_value:
                continue
            key_text = cls._clean_text_value(key)
            if not key_text:
                continue
            normalized_key = cls._normalize_dailymed_section_key(key_text)
            if normalized_key and normalized_key not in normalized_sections:
                normalized_sections[normalized_key] = text_value

        return normalized_sections

    @classmethod
    def _extract_dailymed_dosage(cls, row: Dict[str, Any], section_map: Any) -> str:
        return cls._extract_dailymed_section(
            row,
            cls._normalized_dailymed_sections(section_map),
            canonical_key="dosage",
            aliases=[
                "dosage",
                "dosage and administration",
                "dosage administration",
            ],
            fallback_fields=["dosage", "dosage_and_administration"],
        )

    @classmethod
    def _extract_dailymed_warnings_and_precautions(cls, row: Dict[str, Any], section_map: Any) -> str:
        return cls._extract_dailymed_section(
            row,
            cls._normalized_dailymed_sections(section_map),
            canonical_key="warnings_and_precautions",
            aliases=[
                "warnings and precautions",
                "warnings",
                "precautions",
                "warnings precautions",
                "boxed warning and precautions",
                "boxed warnings and precautions",
            ],
            fallback_fields=[
                "warnings_and_precautions",
                "warnings",
                "precautions",
                "boxed_warning",
                "boxed_warnings",
            ],
        )

    @classmethod
    def _extract_dailymed_contraindications(cls, row: Dict[str, Any], section_map: Any) -> str:
        return cls._extract_dailymed_section(
            row,
            cls._normalized_dailymed_sections(section_map),
            canonical_key="contraindications",
            aliases=[
                "contraindications",
                "contra indication",
                "contra indications",
                "contraindication",
            ],
            fallback_fields=["contraindications", "contraindication"],
        )

    @classmethod
    def _extract_dailymed_section(
        cls,
        row: Dict[str, Any],
        normalized_sections: Dict[str, str],
        canonical_key: str,
        aliases: List[str],
        fallback_fields: Optional[List[str]] = None,
    ) -> str:
        direct_value = cls._clean_text_value(row.get(canonical_key, ""))
        if direct_value:
            return direct_value

        for fallback_key in fallback_fields or []:
            fallback_value = cls._clean_text_value(row.get(fallback_key, ""))
            if fallback_value:
                return fallback_value

        for alias in aliases:
            alias_key = cls._normalize_dailymed_section_key(alias)
            if alias_key and alias_key in normalized_sections:
                return normalized_sections[alias_key]

        for alias in aliases:
            alias_key = cls._normalize_dailymed_section_key(alias)
            if not alias_key:
                continue
            for section_key, section_text in normalized_sections.items():
                if cls._section_key_matches_alias(section_key, alias_key):
                    return section_text

        return ""

    @staticmethod
    def _section_key_matches_alias(normalized_key: str, normalized_alias: str) -> bool:
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

    @staticmethod
    def _attach_sort_score(papers_df):
        if papers_df.empty:
            return papers_df

        if "boosted_relevance_score" in papers_df.columns:
            papers_df["_sort_score"] = papers_df["boosted_relevance_score"].fillna(0.0)
        elif "relevance_judgement" in papers_df.columns and "relevance_score" in papers_df.columns:
            papers_df["_sort_score"] = papers_df["relevance_judgement"].fillna(papers_df["relevance_score"]).fillna(0.0)
        elif "relevance_judgement" in papers_df.columns:
            papers_df["_sort_score"] = papers_df["relevance_judgement"].fillna(0.0)
        elif "relevance_score" in papers_df.columns:
            papers_df["_sort_score"] = papers_df["relevance_score"].fillna(0.0)
        else:
            papers_df["_sort_score"] = 0.0
        return papers_df

    @staticmethod
    def _article_focus_boost_ceiling(focus_score: float) -> float:
        """Limit metadata boosts when an article only weakly matches the query focus."""
        if focus_score >= 0.8:
            return 3.0
        if focus_score >= 0.5:
            return 2.0
        if focus_score >= 0.25:
            return 1.15
        return 1.0

    def _select_literature_articles(self, papers_df, focus_texts: Optional[List[str]] = None):
        if papers_df.empty:
            return papers_df

        boosted_df = self._apply_article_metadata_boosts(papers_df, focus_texts=focus_texts)
        if "relevance_judgement" not in boosted_df.columns:
            return boosted_df.head(self.final_top_articles).reset_index(drop=True)

        threshold = self.article_relevance_threshold
        threshold_mask = boosted_df["relevance_judgement"].fillna(0.0) >= threshold
        threshold_df = boosted_df[threshold_mask].reset_index(drop=True)
        logger.info(
            "   Article relevance threshold (>= %.2f): %s → %s",
            threshold,
            len(boosted_df),
            len(threshold_df),
        )

        if len(threshold_df) >= self.min_final_articles:
            return threshold_df.head(self.final_top_articles).reset_index(drop=True)

        remaining_needed = max(0, self.min_final_articles - len(threshold_df))
        backfill_df = boosted_df[~threshold_mask].head(remaining_needed).reset_index(drop=True)
        logger.info(
            "   Literature backfill to minimum %d articles: threshold-kept=%d backfilled=%d",
            self.min_final_articles,
            len(threshold_df),
            len(backfill_df),
        )
        combined_df = threshold_df
        if not backfill_df.empty:
            import pandas as pd

            combined_df = pd.concat([threshold_df, backfill_df], ignore_index=True)
        return combined_df.head(self.final_top_articles).reset_index(drop=True)

    def _apply_article_metadata_boosts(self, papers_df, focus_texts: Optional[List[str]] = None):
        if papers_df.empty:
            return papers_df

        boosted_rows: List[Dict[str, Any]] = []
        current_year = datetime.now().year

        for _, row in papers_df.iterrows():
            row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
            base_score = self._safe_float(row_dict.get("relevance_judgement", row_dict.get("relevance_score", 0.0)))
            multipliers = compute_metadata_multipliers(
                row_dict,
                current_year=current_year,
                include_recency=True,
            )
            article_focus_score = compute_query_focus_score(row_dict, focus_texts)
            boost_ceiling = self._article_focus_boost_ceiling(article_focus_score)
            focus_limited_boost_multiplier = min(multipliers["boost_multiplier"], boost_ceiling)
            row_dict["relevance_score"] = base_score
            row_dict["article_focus_score"] = article_focus_score
            row_dict["boost_ceiling"] = boost_ceiling
            row_dict["focus_limited_boost_multiplier"] = focus_limited_boost_multiplier
            row_dict["boosted_relevance_score"] = base_score * focus_limited_boost_multiplier
            row_dict["evidence_tier"] = multipliers["evidence_tier"]
            row_dict["boost_multiplier"] = multipliers["boost_multiplier"]
            row_dict["recency_multiplier"] = multipliers["recency_multiplier"]
            row_dict["journal_tier"] = multipliers["journal_tier"]
            row_dict["journal_boost"] = multipliers["journal_boost"]
            row_dict["country_multiplier"] = multipliers["country_multiplier"]
            row_dict["country_boost_applied"] = multipliers["country_boost_applied"]
            row_dict["country_boost_region"] = multipliers["country_boost_region"]
            row_dict["is_case_report"] = multipliers["is_case_report"]
            row_dict["guideline_society_match"] = multipliers["guideline_society_match"]
            row_dict["matched_guideline_societies"] = multipliers["matched_guideline_societies"]
            boosted_rows.append(row_dict)

        import pandas as pd

        boosted_df = pd.DataFrame(boosted_rows)
        boosted_df = self._attach_sort_score(boosted_df)
        boosted_df = boosted_df.sort_values(by="_sort_score", ascending=False).reset_index(drop=True)
        logger.info("   Applied metadata boosts to %s literature articles", len(boosted_df))
        return boosted_df

    def _mark_top_pmc_fulltext_eligible(self, papers_df):
        if papers_df.empty:
            return papers_df

        def is_pmc(row):
            source_family = self._normalized_text(row.get("source_family"))
            if source_family in {"pmc", "pubmed", "dailymed"}:
                return source_family == "pmc"
            return str(row.get("pmcid") or row.get("corpus_id") or "").upper().startswith("PMC")

        top_pmc_ids: List[str] = []
        for _, row in papers_df.iterrows():
            if self._is_dailymed_row(row):
                continue
            if not is_pmc(row):
                continue
            doc_id = str(row.get("doc_id") or row.get("corpus_id") or row.get("pmcid") or "").strip()
            if not doc_id or doc_id in top_pmc_ids:
                continue
            top_pmc_ids.append(doc_id)
            if len(top_pmc_ids) >= 2:
                break

        top_pmc_id_set = set(top_pmc_ids)
        papers_df["_pmc_fulltext_eligible"] = papers_df.apply(
            lambda row: str(row.get("doc_id") or row.get("corpus_id") or row.get("pmcid") or "").strip() in top_pmc_id_set,
            axis=1,
        )
        return papers_df

    @staticmethod
    def _build_passthrough_papers_df(passages: List[Dict[str, Any]]):
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "pmcid": passage.get("pmcid", ""),
                    "title": passage.get("title", ""),
                    "authors": passage.get("authors", []),
                    "venue": passage.get("venue", ""),
                    "journal": passage.get("journal", ""),
                    "year": passage.get("year"),
                    "doi": passage.get("doi", ""),
                    "article_type": passage.get("article_type", ""),
                    "evidence_grade": passage.get("evidence_grade"),
                    "evidence_level": passage.get("evidence_level"),
                    "evidence_term": passage.get("evidence_term"),
                    "evidence_source": passage.get("evidence_source"),
                    "relevance_score": passage.get("relevance_judgement", 0),
                    "abstract": passage.get("abstract", ""),
                    "corpus_id": passage.get("corpus_id", ""),
                    "doc_id": passage.get("doc_id", passage.get("corpus_id", "")),
                    "source_family": passage.get("source_family", ""),
                    "country": passage.get("country"),
                }
                for passage in passages
            ]
        )

    def _build_dailymed_rows(
        self,
        dailymed_results: List[Dict[str, Any]],
        existing_pmcids: set[str],
    ) -> List[Dict[str, Any]]:
        dailymed_rows: List[Dict[str, Any]] = []
        for dm in dailymed_results[: self.dailymed_max_final_references]:
            pmcid = str(dm.get("pmcid", "")).strip()
            if not pmcid or pmcid in existing_pmcids:
                continue

            existing_pmcids.add(pmcid)
            score = dm.get("score", dm.get("relevance_score", 0.0))
            dailymed_rows.append(
                {
                    "pmcid": pmcid,
                    "title": dm.get("title", ""),
                    "drug_name": dm.get("drug_name", "") or dm.get("title", ""),
                    "authors": dm.get("authors", []),
                    "venue": dm.get("venue", "") or dm.get("journal", "DailyMed"),
                    "journal": dm.get("journal", "DailyMed"),
                    "year": dm.get("year"),
                    "doi": dm.get("doi", ""),
                    "article_type": dm.get("article_type", "drug_label"),
                    "evidence_grade": dm.get("evidence_grade"),
                    "evidence_level": dm.get("evidence_level"),
                    "evidence_term": dm.get("evidence_term"),
                    "evidence_source": dm.get("evidence_source"),
                    "relevance_score": score,
                    "boosted_relevance_score": score,
                    "abstract": dm.get("abstract", ""),
                    "corpus_id": pmcid,
                    "doc_id": dm.get("doc_id", pmcid),
                    "source": "dailymed",
                    "source_family": "dailymed",
                    "set_id": dm.get("set_id", ""),
                    "dosage": self._extract_dailymed_dosage(dm, dm.get("dailymed_sections", {})),
                    "dailymed_sections": dm.get("dailymed_sections", {}),
                }
            )
        return dailymed_rows

    def _selection_counts(self, papers_df) -> tuple[int, int, int]:
        def is_pmc(row):
            source_family = self._normalized_text(row.get("source_family"))
            if source_family in {"pmc", "pubmed", "dailymed"}:
                return source_family == "pmc"
            return str(row.get("pmcid") or row.get("corpus_id") or "").upper().startswith("PMC")

        dailymed_count = int(papers_df.apply(self._is_dailymed_row, axis=1).sum())
        pmc_count = sum(1 for _, row in papers_df.iterrows() if not self._is_dailymed_row(row) and is_pmc(row))
        pubmed_count = max(0, len(papers_df) - dailymed_count - pmc_count)
        return pmc_count, pubmed_count, dailymed_count

    def rerank_and_aggregate(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        dailymed_results: Optional[Union[List[Dict[str, Any]], Callable[[], List[Dict[str, Any]]]]] = None,
        medical_conditions: List[str] = None,
        corrected_conditions: List[str] = None,
        focus_query: Optional[str] = None,
    ):
        """Rerank passages and aggregate to paper level.
        
        Args:
            query: User query
            passages: Main passages (will be reranked)
            dailymed_results: DailyMed drug labels (bypass reranking, merged directly)
            medical_conditions: LLM-extracted conditions for strict filtering
            corrected_conditions: Typo-corrected conditions for matching
        """
        import pandas as pd
        
        logger.info("📊 Step 3: Reranking & Aggregation")
        self._last_timing_stats["rerank_ms"] = 0
        self._last_timing_stats["aggregation_ms"] = 0
        self._last_recency_stats = {
            "recent_kept_non_dailymed": 0,
            "dailymed_kept": 0,
            "older_backfilled": 0,
            "unknown_non_dailymed_excluded": 0,
            "recent_cutoff_year": None,
        }

        if self.paper_finder is None:
            logger.info("   Skipping reranking (reranker not available)")
            reranked = list(passages)
            aggregation_start = time.perf_counter()
            papers_df = self._build_passthrough_papers_df(reranked)
            self._last_timing_stats["aggregation_ms"] = int((time.perf_counter() - aggregation_start) * 1000)
            self._last_retrieval_stats.update(
                {
                    "rerank_passages_retrieved": len(passages),
                    "rerank_passages_scored": len(passages),
                    "rerank_passages_kept": len(reranked),
                }
            )
        else:
            logger.info(f"   Reranking {len(passages)} retrieved passages")
            rerank_start = time.perf_counter()
            reranked = self.paper_finder.rerank(
                query,
                list(passages),
                medical_conditions=medical_conditions,
            )
            self._last_timing_stats["rerank_ms"] = int((time.perf_counter() - rerank_start) * 1000)
            aggregation_start = time.perf_counter()
            papers_df = self.paper_finder.aggregate_into_dataframe(reranked)
            self._last_timing_stats["aggregation_ms"] = int((time.perf_counter() - aggregation_start) * 1000)
            self._last_retrieval_stats.update(getattr(self.paper_finder, "last_rerank_stats", {}) or {})

        logger.info(f"   Aggregated to {len(papers_df)} literature papers")
        logger.info(
            "   Rerank timings: rerank=%d ms | aggregation=%d ms",
            self._last_timing_stats["rerank_ms"],
            self._last_timing_stats["aggregation_ms"],
        )

        if ENTITY_FILTER_ENABLED:
            papers_df = self._filter_by_entities(query, papers_df, medical_conditions, corrected_conditions)

        focus_texts: List[str] = []
        if focus_query:
            focus_texts.append(focus_query)
        if query:
            focus_texts.append(query)
        for collection in (corrected_conditions or [], medical_conditions or []):
            focus_texts.extend(collection)

        literature_df = self._select_literature_articles(
            papers_df,
            focus_texts=focus_texts,
        ) if not papers_df.empty else papers_df

        resolved_dailymed_results = dailymed_results() if callable(dailymed_results) else (dailymed_results or [])

        dailymed_df = pd.DataFrame()
        if resolved_dailymed_results:
            logger.info(f"   Appending {len(resolved_dailymed_results)} DailyMed results after literature ranking")
            existing_pmcids = set(literature_df['pmcid'].tolist()) if not literature_df.empty and 'pmcid' in literature_df.columns else set()
            dailymed_rows = self._build_dailymed_rows(resolved_dailymed_results, existing_pmcids)
            if dailymed_rows:
                dailymed_df = pd.DataFrame(dailymed_rows)
            self._last_retrieval_stats["dailymed_appended_labels"] = len(dailymed_rows)

        if not dailymed_df.empty:
            final_df = pd.concat([literature_df, dailymed_df], ignore_index=True)
            final_df = self._deduplicate_dailymed(final_df)
        else:
            final_df = literature_df

        if not final_df.empty:
            final_df = self._mark_top_pmc_fulltext_eligible(final_df)
            pmc_count, pubmed_count, dailymed_count = self._selection_counts(final_df)
            final_df = final_df.drop(columns=["_sort_score"], errors="ignore")
            logger.info(
                "   Final selection: literature=%s (top %s after threshold+boost) | appended DailyMed=%s | PMC=%s | PubMed=%s",
                len(literature_df),
                self.final_top_articles,
                dailymed_count,
                pmc_count,
                pubmed_count,
            )

        return final_df, reranked

    
    def _deduplicate_dailymed(self, papers_df) -> Any:
        """
        Deduplicate DailyMed entries by normalized drug concept.

        DailyMed is pass-through and not reranked. Keep deterministic first-seen
        entries up to MAX_DAILYMED_PER_DRUG per normalized concept key.
        
        Args:
            papers_df: DataFrame of papers
            
        Returns:
            Deduplicated DataFrame
        """
        if papers_df.empty:
            return papers_df
        
        import re
        
        logger.info("🔄 Deduplicating DailyMed entries...")
        
        # Track DailyMed entries by normalized drug concept.
        dailymed_by_drug: Dict[str, List[int]] = {}
        non_dailymed_indices = []
        
        for idx, row in papers_df.iterrows():
            is_dailymed = self._is_dailymed_row(row)
            pmcid = str(row.get('pmcid', '') or row.get('corpus_id', '') or row.get('doc_id', '') or f"row_{idx}")
            
            if is_dailymed:
                source_name = str(row.get("drug_name") or row.get("title") or "").strip().lower()
                normalized_name = re.sub(r"[^a-z0-9\s\-]", " ", source_name)
                normalized_name = re.sub(r"\s+", " ", normalized_name).strip()
                normalized_name = re.sub(
                    r'\s+(?:tablets?|capsules?|injection|solution|oral|intravenous|iv|im|powder|suspension|syrup|cream|ointment|gel|patch|spray)\b.*$',
                    '',
                    normalized_name,
                    flags=re.IGNORECASE
                ).strip().lower()
                
                normalized_name = re.sub(r'\s*\([^)]*\)\s*$', '', normalized_name).strip()
                
                if not normalized_name:
                    normalized_name = f"__unique_{pmcid}"
                dailymed_by_drug.setdefault(normalized_name, []).append(idx)
            else:
                non_dailymed_indices.append(idx)
        
        kept_dailymed_indices = []
        removed_count = 0
        
        for drug_name, entries in dailymed_by_drug.items():
            kept = entries[:MAX_DAILYMED_PER_DRUG]
            removed = entries[MAX_DAILYMED_PER_DRUG:]
            kept_dailymed_indices.extend(kept)
            removed_count += len(removed)
            
            if len(entries) > MAX_DAILYMED_PER_DRUG:
                logger.info(f"   Drug '{drug_name}': kept {len(kept)}/{len(entries)} DailyMed entries")
        
        if removed_count > 0:
            logger.info(f"   Removed {removed_count} duplicate DailyMed entries (keeping max {MAX_DAILYMED_PER_DRUG} per drug)")
            
            # Combine non-DailyMed + kept DailyMed indices, then sort to preserve original ranking
            all_kept_indices = sorted(non_dailymed_indices + kept_dailymed_indices)
            papers_df = papers_df.loc[all_kept_indices].reset_index(drop=True)
        else:
            logger.info("   No duplicate DailyMed entries found")
        
        return papers_df
    
    def _filter_by_entities(self, query: str, papers_df, medical_conditions: List[str] = None, corrected_conditions: List[str] = None) -> Any:
        """
        Filter papers based on medical entity matching.
        
        When LLM-extracted medical_conditions are provided, uses flexible matching:
        - Splits compound conditions into individual terms
        - Normalizes case variations (e.g., IgG4 vs IGG4)
        - Matches if paper contains ANY key term from the condition
        - Uses BOTH original and typo-corrected conditions for matching
        
        Falls back to regex-based entity extraction if no conditions provided.
        
        Args:
            query: Original query
            papers_df: DataFrame of papers
            medical_conditions: LLM-extracted conditions for matching (original spelling)
            corrected_conditions: Typo-corrected conditions for matching
            
        Returns:
            Filtered DataFrame
        """
        if papers_df.empty:
            return papers_df
        
        import re
        
        logger.info("🔍 Post-retrieval filtering: Checking entity matches...")
        
        # Combine original and corrected conditions (corrected takes priority for matching)
        raw_entities = []
        if medical_conditions:
            raw_entities.extend(medical_conditions)
            logger.info(f"   Using LLM-extracted conditions: {medical_conditions}")
        if corrected_conditions:
            raw_entities.extend(corrected_conditions)
            logger.info(f"   Also using typo-corrected conditions: {corrected_conditions}")
        
        # Fall back to regex extraction if no LLM conditions
        if not raw_entities and self.entity_expander:
            raw_entities = self._extract_query_entities(query)
            logger.info(f"   Using regex-extracted entities: {raw_entities}")
        
        if not raw_entities:
            logger.info("   No medical entities found in query, skipping filter")
            return papers_df
        
        # ========================================================================
        # Extract KEY TERMS from conditions for flexible matching
        # e.g., "Immunoglobulin G IGG4 disease" -> ["igg4", "immunoglobulin"]
        # ========================================================================
        key_terms = set()
        for entity in raw_entities:
            # Add the full entity (normalized)
            entity_lower = entity.lower()
            
            # Split by common separators and extract meaningful terms
            words = re.split(r'[\s\-_,]+', entity_lower)
            for word in words:
                # Skip common filler words
                if word in {'of', 'the', 'and', 'or', 'in', 'a', 'an', 'disease', 'syndrome', 'disorder', 'related'}:
                    continue
                # Keep meaningful medical terms (3+ chars)
                if len(word) >= 3:
                    key_terms.add(word)
            
            # Also add the full entity for exact phrase matching
            key_terms.add(entity_lower)
        
        # Add common variations (e.g., igg4 matches IgG4, IGG4)
        # These will match case-insensitively in the paper text
        logger.info(f"   Key terms for matching: {sorted(key_terms)[:10]}...")
        
        # Check each paper for entity matches
        filtered_indices = []
        removed_count = 0
        
        for idx, row in papers_df.iterrows():
            title = str(row.get('title', '')).lower()
            abstract = str(row.get('abstract', '')).lower()
            
            # Combine title and abstract for searching
            paper_text = f"{title} {abstract}"
            
            # Check if paper contains at least one key term
            has_entity = False
            for term in key_terms:
                if term in paper_text:
                    has_entity = True
                    break
            
            if has_entity:
                filtered_indices.append(idx)
            else:
                removed_count += 1
                logger.debug(f"   Removed: {row.get('title', 'Untitled')[:60]}... (no entity match)")
        
        if removed_count > 0:
            logger.info(f"   Filtered out {removed_count} papers without entity matches")
            papers_df = papers_df.loc[filtered_indices].reset_index(drop=True)
        else:
            logger.info("   All papers contain query entities")
        
        return papers_df
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """
        Extract medical entities (disease names, acronyms) from query.
        
        Args:
            query: Original query string
            
        Returns:
            List of medical entity terms
        """
        entities = []
        
        if self.entity_expander is None:
            return entities
        
        # Extract acronyms and their expansions
        words = query.split()
        for word in words:
            # Remove punctuation
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Check if it's a known acronym
            if self.entity_expander._is_likely_acronym(clean_word):
                expansions = self.entity_expander.expand_acronym(clean_word)
                if expansions:
                    # Add both acronym and full term
                    entities.append(clean_word.upper())
                    entities.append(expansions[0])
        
        # Also look for common medical condition patterns
        # (e.g., "Antiphospholipid Syndrome", "Classification Criteria")
        medical_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+Syndrome\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+Disease\b',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+Disorder\b',
        ]
        
        for pattern in medical_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower not in seen:
                seen.add(entity_lower)
                unique_entities.append(entity)
        
        return unique_entities
    
    # =========================================================================
    # Step 4: Direct LLM Synthesis
    # =========================================================================
    
    def _get_papers_for_context(self, papers_df, query: str = ""):
        """Build context and get used papers from reranked articles.
        
        For DailyMed articles, include dosage, warnings/precautions, and contraindications when available.
        """
        import math

        MAX_ABSTRACT_CHARS = 1200
        DAILYMED_CONTEXT_CHAR_LIMIT = 12000
        DAILYMED_SNIPPET_BLOCK_LIMIT = 1800
        DAILYMED_QUERY_TERMS_MAX = 3
        DAILYMED_SNIPPET_WINDOW_BEFORE = 150
        DAILYMED_SNIPPET_WINDOW_AFTER = 220

        def _safe_str(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, float) and math.isnan(value):
                return ""
            return str(value).strip()

        query_terms: List[str] = []
        if query:
            blocked = {
                "drug", "drugs", "label", "labels", "dose", "dosing", "dosage",
                "administration", "warning", "warnings", "precaution", "precautions",
                "contraindication", "contraindications", "patient", "patients",
                "information", "section", "sections",
            }
            seen_terms: set[str] = set()
            for token in re.findall(r"[a-z0-9]+", query.lower()):
                if len(token) < 4 or token in blocked or token in seen_terms:
                    continue
                seen_terms.add(token)
                query_terms.append(token)
                if len(query_terms) >= DAILYMED_QUERY_TERMS_MAX:
                    break
        used_papers = []
        context_parts = []
        dailymed_in_context = 0
        dailymed_structured_context_count = 0
        dailymed_dosage_count = 0
        dailymed_warnings_count = 0
        dailymed_contraindications_count = 0
        dailymed_query_snippet_count = 0
        dailymed_fallback_count = 0
        recent_cutoff_year = self._recent_cutoff_year()
        self._last_context_stats = {
            "pmc_recent_fulltext_used": 0,
            "recent_cutoff_year": recent_cutoff_year,
        }

        for idx, row in papers_df.iterrows():
            is_dailymed = self._is_dailymed_row(row)
            if is_dailymed:
                dailymed_in_context += 1
            article_type = _safe_str(row.get('article_type')) or ('drug_label' if is_dailymed else 'other')
            source_num = len(context_parts) + 1
            
            if hasattr(row, 'to_dict'):
                paper_dict = row.to_dict()
            else:
                paper_dict = dict(row)
            # Preserve canonical citation index so UI/reference ordering is stable.
            paper_dict["citation_index"] = source_num
            if is_dailymed:
                if not _safe_str(paper_dict.get("source")):
                    paper_dict["source"] = "dailymed"
                if not _safe_str(paper_dict.get("source_family")):
                    paper_dict["source_family"] = "dailymed"
                if not _safe_str(paper_dict.get("article_type")):
                    paper_dict["article_type"] = article_type
                if not _safe_str(paper_dict.get("content_type")):
                    paper_dict["content_type"] = "drug_label"
                if not _safe_str(paper_dict.get("journal")) and not _safe_str(paper_dict.get("venue")):
                    paper_dict["journal"] = "DailyMed"
                    paper_dict["venue"] = "DailyMed"
            used_papers.append(paper_dict)

            paper_info = f"**[{source_num}]** {row.get('title', 'Untitled')}"
            if row.get('venue') or row.get('journal'):
                paper_info += f" | *{row.get('venue') or row.get('journal')}*"
            if row.get('year'):
                paper_info += f" ({row.get('year')})"
            paper_info += f" [{article_type}]"

            # DailyMed: Intelligent section selection
            if is_dailymed:
                sections = []
                dailymed_section_map = row.get("dailymed_sections", {})
                dosage = self._extract_dailymed_dosage(row, dailymed_section_map)
                warnings_and_precautions = self._extract_dailymed_warnings_and_precautions(row, dailymed_section_map)
                contraindications = self._extract_dailymed_contraindications(row, dailymed_section_map)

                snippet_lines: List[str] = []
                seen_snippet_lines: set[str] = set()

                for section_name, section_text in [
                    ("Warnings and Precautions", warnings_and_precautions),
                    ("Contraindications", contraindications),
                    ("Dosage and Administration", dosage),
                ]:
                    normalized_text = self._clean_text_value(section_text)
                    if not normalized_text:
                        continue
                    lowered_text = normalized_text.lower()
                    for term in query_terms:
                        pos = lowered_text.find(term)
                        if pos < 0:
                            continue
                        start = max(0, pos - DAILYMED_SNIPPET_WINDOW_BEFORE)
                        end = min(len(normalized_text), pos + DAILYMED_SNIPPET_WINDOW_AFTER)
                        snippet_text = re.sub(r"\s+", " ", normalized_text[start:end]).strip()
                        if not snippet_text:
                            continue
                        line = f"- [{section_name}] {snippet_text}"
                        dedupe_key = line.lower()
                        if dedupe_key in seen_snippet_lines:
                            continue
                        seen_snippet_lines.add(dedupe_key)
                        snippet_lines.append(line)
                        if len(snippet_lines) >= 3:
                            break
                    if len(snippet_lines) >= 3:
                        break

                if snippet_lines:
                    snippet_block = "### Query-Relevant Label Snippets\n" + "\n".join(snippet_lines)
                    snippet_block = snippet_block[:DAILYMED_SNIPPET_BLOCK_LIMIT]
                    sections.append(snippet_block)
                    dailymed_query_snippet_count += 1

                if dosage:
                    sections.append(f"### Dosage and Administration\n{dosage[:8000]}")
                    dailymed_dosage_count += 1
                if warnings_and_precautions:
                    sections.append(f"### Warnings and Precautions\n{warnings_and_precautions[:8000]}")
                    dailymed_warnings_count += 1
                if contraindications:
                    sections.append(f"### Contraindications\n{contraindications[:8000]}")
                    dailymed_contraindications_count += 1

                if sections:
                    text_content = "\n\n".join(sections)[:DAILYMED_CONTEXT_CHAR_LIMIT]
                    dailymed_structured_context_count += 1
                else:
                    # Fallback to abstract if no sections available
                    text_content = (row.get('abstract', '') or row.get('text', ''))[:6000]
                    dailymed_fallback_count += 1

                text_content = self._clean_source_text(text_content)
                paper_info += f"\n{text_content}"
            else:
                # Provide sentence-level evidence retrieved and reranked for context.
                sections = []
                abst = _safe_str(row.get('abstract'))
                if abst:
                    sections.append(f"Abstract:\n{abst}")

                sentences = row.get("sentences", [])
                if isinstance(sentences, list) and sentences:
                    section_groups = {}
                    for sent in sorted(sentences, key=lambda x: x.get("char_start_offset", 0)):
                        sec_title = str(sent.get("section_title", "")).strip().lower()
                        if sec_title in ("abstract", "title"):
                            continue
                        if sec_title not in section_groups:
                            section_groups[sec_title] = []
                        section_groups[sec_title].append(str(sent.get("text", "")).strip())

                    for stitle, stexts in section_groups.items():
                        stext_joined = "\n...\n".join(stexts)
                        sections.append(f"### {stitle.title()}\n{stext_joined}")

                if sections:
                    text_content = "\n\n".join(sections)[:8000]
                else:
                    text_content = (row.get('abstract', '') or row.get('text', ''))[:MAX_ABSTRACT_CHARS]

                text_content = self._clean_source_text(text_content)
                paper_info += f"\n{text_content}"
            
            context_parts.append(paper_info)
            
        logger.info(
            "Context built: %d papers (DailyMed=%d, DailyMed structured=%d, snippet_blocks=%d, dosage=%d, warnings=%d, contraindications=%d, DailyMed fallback=%d, sentence-only context).",
            len(context_parts),
            dailymed_in_context,
            dailymed_structured_context_count,
            dailymed_query_snippet_count,
            dailymed_dosage_count,
            dailymed_warnings_count,
            dailymed_contraindications_count,
            dailymed_fallback_count,
        )
        self._last_context_stats = {
            "dailymed_in_context": dailymed_in_context,
            "dailymed_with_section_selection": dailymed_structured_context_count,
            "dailymed_with_dosage": dailymed_dosage_count,
            "dailymed_with_warnings_and_precautions": dailymed_warnings_count,
            "dailymed_with_contraindications": dailymed_contraindications_count,
            "dailymed_with_query_snippets": dailymed_query_snippet_count,
            "dailymed_fallback_context": dailymed_fallback_count,
            "pmc_recent_fulltext_used": 0,
            "recent_cutoff_year": recent_cutoff_year,
        }
        return context_parts, used_papers

    def _clean_source_text(self, text: str) -> str:
        """
        Strip internal citations from source text to prevent LLM confusion.
        Target patterns: [1], [1, 2], [1-5], [1,2,3], etc.
        """
        if not text:
            return ""
        
        # Pattern for [1], [1, 2], [1-5], [1,2,3] etc.
        # Targets brackets containing numbers, commas, spaces, and hyphens/dashes.
        # Includes leading space to avoid "statement [1]." -> "statement ."
        cleaned = re.sub(r'\s*\[[\d\s,\-\–\.]+\]', '', text)
        
        # Also handle potential superscript numbers if they were converted to text like ^1
        cleaned = re.sub(r'\^[\d,]+', '', cleaned)
        
        # Structural whitespace normalizer to replace the destructive cleanup
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        cleaned = re.sub(r'\n\s*\n+', '\n\n', cleaned)
        
        return cleaned.strip()

    def _is_usmle_query(self, processed_query: Optional[LLMProcessedQuery]) -> bool:
        """Check whether preprocessing classified this as a USMLE-style board question."""
        if not processed_query or not processed_query.decomposed:
            return False
        return bool(processed_query.decomposed.is_usmle_query)

    def _select_prompt_profile(
        self,
        processed_query: Optional[LLMProcessedQuery],
        has_sources: bool,
    ) -> Dict[str, Any]:
        """Pick the prompt profile once so generation modes stay aligned."""
        is_usmle = self._is_usmle_query(processed_query)
        if has_sources:
            if is_usmle:
                return {
                    "is_usmle": True,
                    "mode_label": "USMLE",
                    "system_prompt": ELIXIR_USMLE_SYSTEM_PROMPT,
                }
            return {
                "is_usmle": False,
                "mode_label": "Deep Research",
                "system_prompt": ELIXIR_SYSTEM_PROMPT,
            }

        if is_usmle:
            return {
                "is_usmle": True,
                "mode_label": "USMLE Fallback",
                "system_prompt": ELIXIR_USMLE_FALLBACK_SYSTEM_PROMPT,
            }
        return {
            "is_usmle": False,
            "mode_label": "Fallback",
            "system_prompt": ELIXIR_FALLBACK_SYSTEM_PROMPT,
        }

    def _build_grounded_user_prompt(
        self,
        query: str,
        context: str,
        abstract_count: int,
        is_usmle: bool,
        conversation_summary: str = "",
    ) -> str:
        """Render the grounded user prompt for the selected response mode."""
        if is_usmle:
            instructions = """
Answer the query as a USMLE-style question using only the provided literature context.
1. Determine the question lead-in and the exact answer type being asked.
2. Identify the decisive clinical clues from the vignette.
3. Apply acuity-first reasoning when the question asks for the next step or management.
4. Present a focused differential diagnosis when it helps distinguish the correct answer.
5. State the best answer clearly, then explain why it is correct.
6. Briefly explain why the most plausible alternatives are less appropriate.
7. Keep the response high-yield and exam-oriented while using inline citations **[1]**, **[2]**, etc. strictly.
   - DailyMed drug labels use the same citation numbering namespace as journal articles.
8. After the complete answer, append this exact marker on a new line: <FOLLOW_UP_QUESTIONS_JSON>
9. After the marker, output valid JSON only with this exact schema:
   {"follow_up_questions":["...","...","..."]}
10. Generate exactly 3 standalone, knowledge-expanding follow-up questions that do not rely on prior chat context.
"""
        else:
            instructions = """
Analyze and synthesize the medical literature above to create a detailed, clinically-focused clinical review or practice guideline section.
1. **Extract specific clinical details**: Classification systems, staging criteria, detailed medication protocols (dosing, administration, duration), trial results (outcomes, p-values), and guideline recommendations.
2. **Provide comparative analyses**: Efficacy comparisons with data points and safety profiles.
3. **Structure**: Use clear hierarchical headings, markdown tables for comparisons/staging, and evidence-based recommendations.
4. **Citation**: Use inline citations **[1]**, **[2]**, etc. strictly.
   - DailyMed drug labels use the same citation numbering namespace as journal articles.
5. **Depth**: High technical detail for physician decision-making. No word count limit, but maintain density.
6. After the complete answer, append this exact marker on a new line: <FOLLOW_UP_QUESTIONS_JSON>
7. After the marker, output valid JSON only with this exact schema:
   {"follow_up_questions":["...","...","..."]}
8. Generate exactly 3 standalone, knowledge-expanding follow-up questions that do not rely on prior chat context.
"""

        conversation_section = ""
        if conversation_summary.strip():
            conversation_section = f"""
# [CONVERSATION CONTEXT]
{conversation_summary.strip()}
"""

        return f"""
# [QUERY]
{query}
{conversation_section}

# [CONTEXT]
(Source Literature: {abstract_count} articles)
{context}

# [INSTRUCTIONS]
{instructions}
"""

    def _build_fallback_user_prompt(
        self,
        query: str,
        is_usmle: bool,
        conversation_summary: str = "",
    ) -> str:
        """Render the fallback user prompt when no evidence sources are available."""
        conversation_line = ""
        if conversation_summary.strip():
            conversation_line = f"\nConversation context summary: {conversation_summary.strip()}\n"
        if is_usmle:
            return f"""USMLE-Style Medical Query: {query}
{conversation_line}

Please answer this as a USMLE-style question based on general medical knowledge. Identify the question intent, choose the best answer or next step, explain the key reasoning, briefly note why major alternatives are less appropriate, and finish with high-yield takeaways.
After the answer, append this exact marker on a new line: <FOLLOW_UP_QUESTIONS_JSON>
Then output valid JSON only with this exact schema:
{{"follow_up_questions":["...","...","..."]}}
Generate exactly 3 standalone, knowledge-expanding follow-up questions that do not rely on prior chat context."""

        return f"""Medical Query: {query}
{conversation_line}

Please provide a comprehensive clinical response based on your medical knowledge.
After the answer, append this exact marker on a new line: <FOLLOW_UP_QUESTIONS_JSON>
Then output valid JSON only with this exact schema:
{{"follow_up_questions":["...","...","..."]}}
Generate exactly 3 standalone, knowledge-expanding follow-up questions that do not rely on prior chat context."""

    def run_generation(
        self,
        query: str,
        papers_df,
        processed_query: Optional[LLMProcessedQuery] = None,
        stream: bool = False,
    ):
        """Generate answer directly using LLM with ELIXIR system prompt."""
        logger.info("🧠 Step 4: Direct LLM Synthesis")

        if papers_df.empty:
            return ("No relevant evidence found for your query.", [], [])

        context_parts, used_papers = self._get_papers_for_context(papers_df, query)
        abstract_count = len(context_parts)

        logger.info(f"   Context: {abstract_count} abstracts = {len(context_parts)} total articles")
        context = "\n\n---\n\n".join(context_parts)
        prompt_profile = self._select_prompt_profile(processed_query, has_sources=True)
        system_prompt = prompt_profile["system_prompt"]
        user_prompt = self._build_grounded_user_prompt(
            query=query,
            context=context,
            abstract_count=abstract_count,
            is_usmle=bool(prompt_profile["is_usmle"]),
            conversation_summary=(processed_query.conversation_summary if processed_query else ""),
        )
        logger.info(f"   Mode: {prompt_profile['mode_label']} | Prompt: {len(system_prompt)} chars")

        try:
            if stream:
                # Return a generator for tokens
                return self._stream_generation(
                    query,
                    system_prompt,
                    user_prompt,
                    used_papers,
                )
            
            response = self._create_chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            raw_output = response.choices[0].message.content.strip()
            answer, follow_up_questions = self._extract_answer_and_followups(query, raw_output)
            logger.info(f"   Generated response ({len(answer)} chars)")
            return answer, used_papers, follow_up_questions

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return (f"Error generating response: {str(e)}", [], [])

    def _stream_generation(
        self,
        query: str,
        system_prompt: str,
        user_prompt: str,
        used_papers: list,
    ) -> Generator[Dict[str, Any], None, None]:
        """Generator for token-by-token streaming."""
        try:
            response = self._create_chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
            )
            
            full_answer = ""
            raw_output = ""
            buffer = ""
            marker_started = False
            hold_chars = max(1, len(FOLLOW_UP_MARKER) - 1)
            last_reasoning_emit = 0.0
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    raw_output += token

                    if marker_started:
                        continue

                    combined = buffer + token
                    marker_idx = combined.find(FOLLOW_UP_MARKER)
                    if marker_idx != -1:
                        visible = combined[:marker_idx]
                        if visible:
                            full_answer += visible
                            yield {"step": "generation", "status": "running", "token": visible}
                        marker_started = True
                        buffer = ""
                        continue

                    if len(combined) > hold_chars:
                        visible = combined[:-hold_chars]
                        if visible:
                            full_answer += visible
                            yield {"step": "generation", "status": "running", "token": visible}
                        buffer = combined[-hold_chars:]
                    else:
                        buffer = combined
                elif chunk.choices and getattr(chunk.choices[0].delta, "reasoning_content", None):
                    now = time.monotonic()
                    if now - last_reasoning_emit >= 2.0:
                        last_reasoning_emit = now
                        yield {"step": "generation", "status": "running", "message": "Reasoning before answering..."}

            if buffer and not marker_started:
                full_answer += buffer
                yield {"step": "generation", "status": "running", "token": buffer}

            parsed_answer, follow_up_questions = self._extract_answer_and_followups(query, raw_output)
            if parsed_answer:
                full_answer = parsed_answer
            else:
                full_answer = full_answer.strip()
            
            yield {
                "step": "generation",
                "status": "complete",
                "answer": full_answer,
                "used_papers": used_papers,
                "follow_up_questions": follow_up_questions,
            }
            
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield {"step": "error", "message": str(e)}
    
    # =========================================================================
    # Fallback Generation (when no sources found)
    # =========================================================================
    
    def _run_fallback_generation(
        self,
        query: str,
        processed_query: Optional[LLMProcessedQuery] = None,
        stream: bool = False,
    ):
        """
        Generate response when no sources are found using the same LLM model.
        
        Args:
            query: Original user query
            stream: Whether to stream the response
            
        Returns:
            If stream=False: tuple of (answer, [])
            If stream=True: Generator yielding response events
        """
        prompt_profile = self._select_prompt_profile(processed_query, has_sources=False)
        logger.info("📭 No sources found, using %s for response", prompt_profile["mode_label"])
        fallback_system_prompt = prompt_profile["system_prompt"]
        user_prompt = self._build_fallback_user_prompt(
            query,
            is_usmle=bool(prompt_profile["is_usmle"]),
            conversation_summary=(processed_query.conversation_summary if processed_query else ""),
        )

        try:
            if stream:
                return self._stream_fallback_generation(
                    query,
                    fallback_system_prompt,
                    user_prompt,
                )
            
            response = self._create_chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": fallback_system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            
            raw_output = response.choices[0].message.content.strip()
            answer, follow_up_questions = self._extract_answer_and_followups(query, raw_output)
            logger.info(f"   Fallback response generated ({len(answer)} chars)")
            return answer, [], follow_up_questions
            
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            return (f"Unable to generate response: {str(e)}", [], [])
    
    def _stream_fallback_generation(
        self,
        query: str,
        system_prompt: str,
        user_prompt: str,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream fallback generation tokens."""
        try:
            response = self._create_chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
            )
            
            full_answer = ""
            raw_output = ""
            buffer = ""
            marker_started = False
            hold_chars = max(1, len(FOLLOW_UP_MARKER) - 1)
            last_reasoning_emit = 0.0
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    raw_output += token

                    if marker_started:
                        continue

                    combined = buffer + token
                    marker_idx = combined.find(FOLLOW_UP_MARKER)
                    if marker_idx != -1:
                        visible = combined[:marker_idx]
                        if visible:
                            full_answer += visible
                            yield {"step": "generation", "status": "running", "token": visible}
                        marker_started = True
                        buffer = ""
                        continue

                    if len(combined) > hold_chars:
                        visible = combined[:-hold_chars]
                        if visible:
                            full_answer += visible
                            yield {"step": "generation", "status": "running", "token": visible}
                        buffer = combined[-hold_chars:]
                    else:
                        buffer = combined
                elif chunk.choices and getattr(chunk.choices[0].delta, "reasoning_content", None):
                    now = time.monotonic()
                    if now - last_reasoning_emit >= 2.0:
                        last_reasoning_emit = now
                        yield {"step": "generation", "status": "running", "message": "Reasoning before answering..."}

            if buffer and not marker_started:
                full_answer += buffer
                yield {"step": "generation", "status": "running", "token": buffer}

            parsed_answer, follow_up_questions = self._extract_answer_and_followups(query, raw_output)
            if parsed_answer:
                full_answer = parsed_answer
            else:
                full_answer = full_answer.strip()
            
            yield {
                "step": "generation",
                "status": "complete",
                "answer": full_answer,
                "used_papers": [],
                "follow_up_questions": follow_up_questions,
            }
            
        except Exception as e:
            logger.error(f"Fallback streaming failed: {e}")
            yield {"step": "error", "message": str(e)}

    def _create_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
    ):
        request_kwargs = self._build_chat_completion_kwargs(
            model=model,
            messages=messages,
            provider=self.llm_provider,
            stream=stream,
        )
        return retry_with_exponential_backoff(
            lambda: self.llm_client.chat.completions.create(**request_kwargs),
            max_attempts=LLM_RETRY_COUNT + 1,
            base_delay=float(LLM_RETRY_DELAY),
            operation_name=f"{self.llm_provider} chat completion",
            logger=logger,
        )

    @staticmethod
    def _parse_citation_indices(answer_text: str, max_index: Optional[int] = None) -> set[int]:
        """Parse inline citations like [1], [2,3], [4-6]."""
        if not answer_text:
            return set()
        answer_text = MedicalRAGPipeline._normalize_citation_markers(answer_text)
        cited: set[int] = set()
        for group in CITATION_GROUP_PATTERN.findall(answer_text):
            cited.update(
                MedicalRAGPipeline._parse_citation_group(
                    group,
                    max_index=max_index,
                )
            )
        return cited

    def _filter_sources_to_citations(self, answer_text: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Keep cited sources numbered while preserving uncited sources as supplemental entries.
        """
        if not sources:
            return []

        before_count = len(sources)
        before_dailymed = sum(1 for source in sources if self._is_dailymed_row(source))

        def _clone_source(source: Dict[str, Any]) -> Dict[str, Any]:
            return dict(source)

        def _mark_source_uncited(source: Dict[str, Any]) -> Dict[str, Any]:
            source["citation_index"] = None
            return source

        def _source_identity_key(source: Dict[str, Any]) -> str:
            source_family = str(source.get("source_family") or "").strip().lower()
            pmcid = str(source.get("pmcid") or "").strip().lower()
            pmid = str(source.get("pmid") or "").strip().lower()
            doi = str(source.get("doi") or "").strip().lower()
            set_id = str(source.get("set_id") or "").strip().lower()
            title = str(source.get("title") or "").strip().lower()
            return "|".join([source_family, pmcid, pmid, doi, set_id, title])

        citation_indices = []
        for source in sources:
            try:
                citation_indices.append(int(source.get("citation_index")))
            except (TypeError, ValueError):
                continue
        max_index = max(citation_indices) if citation_indices else None

        cited = self._parse_citation_indices(answer_text, max_index=max_index)

        if not cited:
            final_sources = [
                _mark_source_uncited(_clone_source(source))
                for source in sources
                if not self._is_dailymed_row(source)
            ]
            after_count = len(final_sources)
            after_dailymed = sum(1 for source in final_sources if self._is_dailymed_row(source))
            logger.info(
                "Citation filter skipped: no parseable citations found; keeping %d uncited non-DailyMed sources (DailyMed=%d).",
                after_count,
                after_dailymed,
            )
            return final_sources

        referenced_sources: List[Dict[str, Any]] = []
        referenced_non_dailymed_keys: set[str] = set()
        for source in sources:
            try:
                idx = int(source.get("citation_index"))
            except (TypeError, ValueError):
                continue
            if idx in cited:
                cloned = _clone_source(source)
                referenced_sources.append(cloned)
                if not self._is_dailymed_row(cloned):
                    referenced_non_dailymed_keys.add(_source_identity_key(cloned))

        uncited_non_dailymed_sources: List[Dict[str, Any]] = []
        seen_uncited_non_dailymed_keys: set[str] = set()
        for source in sources:
            if self._is_dailymed_row(source):
                continue
            try:
                idx = int(source.get("citation_index"))
            except (TypeError, ValueError):
                idx = None
            if idx in cited:
                continue
            key = _source_identity_key(source)
            if key in referenced_non_dailymed_keys or key in seen_uncited_non_dailymed_keys:
                continue
            uncited_non_dailymed_sources.append(_mark_source_uncited(_clone_source(source)))
            seen_uncited_non_dailymed_keys.add(key)

        final_sources = referenced_sources + uncited_non_dailymed_sources
        if not final_sources:
            final_sources = [
                _mark_source_uncited(_clone_source(source))
                for source in sources
                if not self._is_dailymed_row(source)
            ]

        after_count = len(final_sources)
        after_dailymed = sum(1 for source in final_sources if self._is_dailymed_row(source))
        logger.info(
            "Citation filter applied: sources %d→%d | DailyMed %d→%d | supplemental non-DailyMed=%d.",
            before_count,
            after_count,
            before_dailymed,
            after_dailymed,
            len(uncited_non_dailymed_sources),
        )
        return final_sources
    
    # =========================================================================
    # PDF Availability Check via Europe PMC
    # =========================================================================
    
    def _check_pdf_availability(self, papers: list) -> list:
        """
        Check PDF availability for papers via Europe PMC API.
        Uses one batched Europe PMC query built from DOI/PMCID identifiers.
        
        Europe PMC API returns:
        - isOpenAccess: "Y" or "N" (string)
        - inEPMC: "Y" or "N" (string) 
        - hasPDF: "Y" or "N" (string)
        - fullTextUrlList.fullTextUrl: Array with documentStyle, availabilityCode, url
        
        PDF URL format: https://europepmc.org/articles/{PMCID}?pdf=render
        
        Args:
            papers: List of paper dicts with doi, pmcid, pmid fields
            
        Returns:
            List of source dicts with pdf_url field added where available
        """
        import requests

        def _normalize_doi(doi: Any) -> str:
            raw = str(doi or "").strip()
            if not raw:
                return ""
            raw = re.sub(r"^https?://(dx\.)?doi\.org/", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"^doi:\s*", "", raw, flags=re.IGNORECASE)
            return raw.strip().lower()

        def _normalize_pmcid(pmcid: Any) -> str:
            raw = str(pmcid or "").strip().upper()
            if not raw:
                return ""
            raw = raw.replace("PMCID:", "").strip()
            raw = re.sub(r"\s+", "", raw)
            if raw.startswith("PMC"):
                return raw
            if raw.isdigit():
                return f"PMC{raw}"
            return raw

        def _extract_pdf_url(article: Dict[str, Any]) -> str:
            url_list = article.get("fullTextUrlList", {}).get("fullTextUrl", [])
            if isinstance(url_list, dict):
                url_list = [url_list]

            for url_info in url_list:
                if not isinstance(url_info, dict):
                    continue
                doc_style = str(url_info.get("documentStyle", "")).lower()
                avail_code = str(url_info.get("availabilityCode", "")).upper()
                availability = str(url_info.get("availability", "")).lower()

                if doc_style == "pdf" and (avail_code == "OA" or "open access" in availability):
                    pdf_url = url_info.get("url")
                    if pdf_url:
                        return pdf_url

            def _flag_is_yes(*names: str) -> bool:
                for name in names:
                    value = article.get(name)
                    if isinstance(value, str):
                        if value.strip().upper() == "Y":
                            return True
                    elif value:
                        return True
                return False

            is_open_access = _flag_is_yes("isOpenAccess", "is_open_access")
            in_epmc = _flag_is_yes("inEPMC", "in_epmc")
            has_pdf = _flag_is_yes("hasPDF", "has_pdf")
            if has_pdf and (is_open_access or in_epmc):
                article_pmcid = _normalize_pmcid(article.get("pmcid") or article.get("id"))
                if article_pmcid:
                    return f"https://europepmc.org/articles/{article_pmcid}?pdf=render"

            return ""

        logger.info("📄 Step 5: Checking PDF availability via Europe PMC")
        start_time = time.perf_counter()
        
        if not papers:
            self._last_timing_stats["pdf_check_ms"] = 0
            return []
        
        normalized_papers = []
        for i, paper in enumerate(papers):
            if hasattr(paper, 'to_dict'):
                p = paper.to_dict()
            else:
                p = dict(paper)
            if p.get("citation_index") in (None, ""):
                p["citation_index"] = i + 1
            normalized_papers.append(p)

        sources: List[Dict[str, Any]] = []
        doi_to_indices: Dict[str, List[int]] = {}
        pmcid_to_indices: Dict[str, List[int]] = {}
        query_terms: List[str] = []
        seen_dois = set()
        seen_pmcids = set()

        for i, paper in enumerate(normalized_papers):
            source = self._map_paper_to_source(paper)
            source["_order_index"] = i
            sources.append(source)

            doi_key = _normalize_doi(source.get("doi"))
            if doi_key:
                doi_to_indices.setdefault(doi_key, []).append(i)
                if doi_key not in seen_dois:
                    query_terms.append(f'DOI:"{doi_key}"')
                    seen_dois.add(doi_key)

            pmcid_key = _normalize_pmcid(source.get("pmcid"))
            if pmcid_key:
                pmcid_to_indices.setdefault(pmcid_key, []).append(i)
                if pmcid_key not in seen_pmcids:
                    query_terms.append(f"PMCID:{pmcid_key}")
                    seen_pmcids.add(pmcid_key)

        if query_terms:
            batch_query = "(" + " OR ".join(query_terms) + ")"
            api_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            params = {
                "query": batch_query,
                "format": "json",
                "resultType": "core",
                "pageSize": 1000,
            }

            try:
                response = requests.get(api_url, params=params, timeout=20)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("resultList", {}).get("result", [])

                    for article in results:
                        pdf_url = _extract_pdf_url(article)
                        if not pdf_url:
                            continue

                        matched_indices = set()

                        doi_key = _normalize_doi(article.get("doi"))
                        if doi_key:
                            matched_indices.update(doi_to_indices.get(doi_key, []))

                        pmcid_key = _normalize_pmcid(article.get("pmcid"))
                        if pmcid_key:
                            matched_indices.update(pmcid_to_indices.get(pmcid_key, []))

                        # Some records expose PMCID in the generic "id" field.
                        pmcid_id_key = _normalize_pmcid(article.get("id"))
                        if pmcid_id_key:
                            matched_indices.update(pmcid_to_indices.get(pmcid_id_key, []))

                        for idx in matched_indices:
                            if not sources[idx].get("pdf_url"):
                                sources[idx]["pdf_url"] = pdf_url
                else:
                    logger.debug(f"Batch PDF check failed with status {response.status_code}")
            except requests.exceptions.Timeout:
                logger.debug("Batch PDF check timed out")
            except Exception as e:
                logger.debug(f"Batch PDF check failed: {e}")

        # Preserve stable citation order for UI/inline citation mapping.
        def _citation_rank(source: Dict[str, Any]) -> int:
            val = source.get("citation_index")
            try:
                if val is None or val == "":
                    return 10**9
                return int(val)
            except (TypeError, ValueError):
                return 10**9

        sources.sort(key=lambda s: (_citation_rank(s), s.get("_order_index", 10**9)))
        for source in sources:
            source.pop("_order_index", None)
        
        # Count and log PDFs found
        pdf_count = sum(1 for s in sources if s.get("pdf_url"))
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        self._last_timing_stats["pdf_check_ms"] = elapsed_ms
        logger.info(f"   Found {pdf_count}/{len(sources)} articles with open access PDFs")
        logger.info("   PDF availability check completed in %d ms", elapsed_ms)
        
        return sources

    def _map_paper_to_source(self, paper: Any) -> Dict[str, Any]:
        """
        Helper to map a paper record (Dict or Series) to a standardized source object.
        Used both in initial streaming and final PDF verification.
        """
        # Convert Series/Namespace to dict if needed
        if hasattr(paper, 'to_dict'):
            p = paper.to_dict()
        else:
            p = dict(paper)

        # Helper to sanitize NaN values (pandas can return float('nan') which breaks JSON)
        def sanitize(val, default=""):
            import math
            if val is None:
                return default
            if isinstance(val, float) and math.isnan(val):
                return default
            return val

        pmcid = sanitize(p.get("pmcid"), "") or sanitize(p.get("corpus_id"), "")
        source_type = sanitize(p.get("source"), "")
        source_family = sanitize(p.get("source_family"), "")
        
        # Detect DailyMed articles
        is_dailymed = self._is_dailymed_row(p)
        
        # Extract set_id for DailyMed articles
        set_id = sanitize(p.get("set_id"), "")
        if is_dailymed and not set_id:
            if str(pmcid).startswith("dailymed_"):
                set_id = str(pmcid).replace("dailymed_", "", 1)
            else:
                set_id = sanitize(p.get("doc_id"), "") or sanitize(p.get("corpus_id"), "")
        if is_dailymed and not pmcid and set_id:
            pmcid = f"dailymed_{set_id}"

        if not source_family:
            if is_dailymed:
                source_family = "dailymed"
            elif str(pmcid).upper().startswith("PMC"):
                source_family = "pmc"
            else:
                source_family = "pubmed"

        raw_section_map = p.get("dailymed_sections", {})
        if not isinstance(raw_section_map, dict):
            raw_section_map = {}

        dosage = ""
        drug_name = ""
        if is_dailymed:
            drug_name = (
                self._clean_text_value(p.get("drug_name"))
                or self._clean_text_value(p.get("title"))
            )
            dosage = self._extract_dailymed_dosage(p, raw_section_map)

        source = {
            "pmcid": pmcid,
            "pmid": sanitize(p.get("pmid"), ""),
            "title": sanitize(p.get("title"), "Untitled"),
            "authors": p.get("authors", []),
            "journal": sanitize(p.get("venue"), "") or sanitize(p.get("journal"), "") or ("DailyMed" if is_dailymed else ""),
            "year": sanitize(p.get("year")),
            "doi": sanitize(p.get("doi"), ""),
            "article_type": sanitize(p.get("article_type"), "") or ("drug_label" if is_dailymed else ""),
            "evidence_grade": sanitize(p.get("evidence_grade"), None),
            "evidence_level": sanitize(p.get("evidence_level"), None),
            "evidence_term": sanitize(p.get("evidence_term"), None),
            "evidence_source": sanitize(p.get("evidence_source"), None),
            "relevance_score": sanitize(p.get("relevance_judgement", p.get("relevance_score", 0)), 0),
            "pdf_url": sanitize(p.get("pdf_url")), # Preserve if already present
            "source": source_type or ("dailymed" if is_dailymed else ""),
            "source_family": source_family,
            "citation_index": sanitize(p.get("citation_index"), None),
        }
        if is_dailymed:
            source.update({
                "drug_name": drug_name,
                "set_id": set_id,
                "dailymed_url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}" if set_id else None,
                "dosage": dosage,
            })
        else:
            source["drug_name"] = ""
            source["set_id"] = set_id
        return source
    
    # =========================================================================
    # Full Pipeline
    # =========================================================================
    
    def answer(
        self,
        query: str,
        thread_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run complete Elixir RAG pipeline.
        
        Returns structured response with answer and sources.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🔬 Elixir Pipeline: {query}")
        logger.info(f"{'='*60}")
        
        # Step 0: Check Cache
        cached_result = self._cache_get(query, thread_context=thread_context)
        if cached_result:
            # Add a marker that this was a cache hit
            cached_result["status"] = "cache_hit"
            return cached_result

        self._reset_run_stats()

        # Step 1: Preprocess
        processed_query = self.preprocess_query(query, thread_context=thread_context)
        is_usmle_option_path = self._is_usmle_option_path(processed_query)
        
        # Step 2: Retrieve
        passages, dailymed_results, dailymed_resolver = self.retrieve_passages(processed_query)

        if not passages and dailymed_resolver is not None and not dailymed_results:
            dailymed_results = dailymed_resolver()

        if not passages and not dailymed_results:
            # No passages retrieved - use fallback LLM generation
            logger.info("📭 No passages retrieved, using fallback generation")
            answer, _, follow_up_questions = self._run_fallback_generation(
                query,
                processed_query=processed_query,
                stream=False,
            )
            result = {
                "query": query,
                "report_title": "Clinical Response (No Sources)",
                "answer": answer,
                "follow_up_questions": follow_up_questions,
                "sections": [],
                "sources": [],
                "evidence_hierarchy": self.evidence_hierarchy,
                "status": "fallback"
            }
            # Cache the fallback result too
            self._cache_set(query, result, thread_context=thread_context)
            return result
        
        # Step 3: Rerank & Aggregate (or passthrough for USMLE option path)
        if is_usmle_option_path:
            logger.info("📊 Step 3: USMLE option path uses direct passthrough (reranker disabled)")
            papers_df = self._build_passthrough_papers_df(passages)
        else:
            # Pass decomposed entities (original + corrected) for strict filtering.
            medical_conditions = []
            corrected_conditions = []
            if processed_query.decomposed:
                if processed_query.decomposed.key_entities:
                    medical_conditions = processed_query.decomposed.key_entities
                if processed_query.decomposed.corrected_entities:
                    corrected_conditions = processed_query.decomposed.corrected_entities
            papers_df, _ = self.rerank_and_aggregate(
                query,
                passages,
                dailymed_resolver or dailymed_results,
                medical_conditions,
                corrected_conditions,
                focus_query=processed_query.primary_query,
            )
        
        if papers_df.empty:
            # No papers after filtering - use fallback LLM generation
            logger.info("📭 No papers after filtering, using fallback generation")
            answer, _, follow_up_questions = self._run_fallback_generation(
                query,
                processed_query=processed_query,
                stream=False,
            )
            result = {
                "query": query,
                "report_title": "Clinical Response (No Sources)",
                "answer": answer,
                "follow_up_questions": follow_up_questions,
                "sections": [],
                "sources": [],
                "evidence_hierarchy": self.evidence_hierarchy,
                "status": "fallback"
            }
            # Cache fallback
            self._cache_set(query, result, thread_context=thread_context)
            return result
        
        # Step 4: Direct LLM Synthesis
        synth_result = self.run_generation(
            query,
            papers_df,
            processed_query=processed_query,
        )
        
        # Handle tuple return (answer, used_papers, follow_up_questions) or error string
        if isinstance(synth_result, tuple):
            if len(synth_result) == 3:
                answer, used_papers, follow_up_questions = synth_result
            elif len(synth_result) == 2:
                answer, used_papers = synth_result
                follow_up_questions = []
            else:
                answer = synth_result[0]
                used_papers = []
                follow_up_questions = []
        else:
            # Error case
            answer = synth_result
            used_papers = []
            follow_up_questions = []
        
        # Step 5: Check PDF availability via Europe PMC for all used papers
        sources_with_pdf = self._check_pdf_availability(used_papers)
        sources_with_pdf = self._filter_sources_to_citations(answer, sources_with_pdf)
        answer, sources_with_pdf = self._normalize_answer_and_sources(answer, sources_with_pdf)
        self._last_retrieval_stats["dailymed_final_sources_returned"] = sum(
            1 for source in sources_with_pdf if self._is_dailymed_row(source)
        )
        
        final_result = {
            "query": query,
            "report_title": "Clinical Response",
            "answer": answer,
            "follow_up_questions": follow_up_questions,
            "sections": [],  # Direct synthesis - no sections
            "sources": sources_with_pdf,
            "evidence_hierarchy": self.evidence_hierarchy,
            "retrieval_stats": self._build_retrieval_stats(
                passages_retrieved=len(passages),
                papers_after_aggregation=len(papers_df),
                abstracts_used=len(used_papers),
            ),
            "status": "success"
        }
        
        # Step 6: Store in Cache
        self._cache_set(query, final_result, thread_context=thread_context)
        
        return final_result
    
    def answer_streaming(
        self,
        query: str,
        thread_context: Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Streaming version for real-time UI updates with parallel PDF checks."""
        from concurrent.futures import ThreadPoolExecutor

        cached_result = self._cache_get(query, thread_context=thread_context)
        if cached_result:
            cached_result["status"] = "cache_hit"
            yield {
                "step": "complete",
                "status": "cache_hit",
                "report_title": cached_result.get("report_title", "Clinical Response"),
                "answer": cached_result.get("answer", ""),
                "follow_up_questions": cached_result.get("follow_up_questions", []),
                "sources": cached_result.get("sources", []),
                "evidence_hierarchy": cached_result.get("evidence_hierarchy", self.evidence_hierarchy),
                "abstracts_used": cached_result.get("retrieval_stats", {}).get("abstracts_used", 0),
                "cache_hit": True,
            }
            return

        self._reset_run_stats()

        # Step 1: Preprocess
        yield {"step": "query_expansion", "status": "running", "message": "Analyzing query..."}
        processed_query = self.preprocess_query(query, thread_context=thread_context)
        is_usmle_option_path = self._is_usmle_option_path(processed_query)
        yield {
            "step": "query_expansion",
            "status": "complete",
            "data": {
                "primary_query": processed_query.primary_query,
                "keyword_query": processed_query.keyword_query,
            }
        }

        # Step 2: Retrieval
        yield {"step": "retrieval", "status": "running", "message": "Searching literature..."}
        passages, dailymed_results, dailymed_resolver = self.retrieve_passages(processed_query)
        if not passages and dailymed_resolver is not None and not dailymed_results:
            dailymed_results = dailymed_resolver()
        yield {
            "step": "retrieval",
            "status": "complete",
            "data": {"count": len(passages) + len(dailymed_results)}
        }

        if not passages and not dailymed_results:
            # No passages retrieved - use fallback LLM generation with streaming
            logger.info("📭 No passages retrieved, using fallback streaming generation")
            yield {"step": "generation", "status": "running", "message": "Generating response from medical knowledge..."}
            fallback_gen = self._run_fallback_generation(
                query,
                processed_query=processed_query,
                stream=True,
            )
            final_answer = ""
            follow_up_questions: List[str] = []
            for event in fallback_gen:
                if event["step"] == "generation" and event["status"] == "running":
                    yield event
                elif event["step"] == "generation" and event["status"] == "complete":
                    final_answer = event.get("answer", "")
                    follow_up_questions = event.get("follow_up_questions", []) or []
            yield {
                "step": "complete",
                "status": "fallback",
                "report_title": "Clinical Response (No Sources)",
                "answer": final_answer,
                "follow_up_questions": follow_up_questions,
                "sources": [],
                "evidence_hierarchy": self.evidence_hierarchy,
                "abstracts_used": 0
            }
            self._cache_set(query, {
                "query": query,
                "report_title": "Clinical Response (No Sources)",
                "answer": final_answer,
                "follow_up_questions": follow_up_questions,
                "sections": [],
                "sources": [],
                "evidence_hierarchy": self.evidence_hierarchy,
                "status": "fallback",
                "retrieval_stats": self._build_retrieval_stats(
                    passages_retrieved=0,
                    papers_after_aggregation=0,
                    abstracts_used=0,
                ),
            }, thread_context=thread_context)
            return

        # Step 3: Reranking / Passthrough
        yield {"step": "reranking", "status": "running", "message": "Ranking papers..."}
        if is_usmle_option_path:
            papers_df = self._build_passthrough_papers_df(passages)
        else:
            # Pass decomposed entities (original + corrected) for strict filtering.
            medical_conditions = []
            corrected_conditions = []
            if processed_query.decomposed:
                if processed_query.decomposed.key_entities:
                    medical_conditions = processed_query.decomposed.key_entities
                if processed_query.decomposed.corrected_entities:
                    corrected_conditions = processed_query.decomposed.corrected_entities
            papers_df, _ = self.rerank_and_aggregate(
                query,
                passages,
                dailymed_resolver or dailymed_results,
                medical_conditions,
                corrected_conditions,
                focus_query=processed_query.primary_query,
            )
        
        # [NEW] Reorder papers so reference ranking matches final citation order from the start
        # This prevents the UI from "jumping" and ensures references are correctly numbered
        _, used_papers = self._get_papers_for_context(papers_df, query)

        # Prepare initial sources (no PDF URLs yet)
        initial_sources = [self._map_paper_to_source(p) for p in used_papers]

        yield {
            "step": "reranking",
            "status": "complete",
            "data": {"papers": len(papers_df)},
            "sources": initial_sources,  # Send sources early and in corrected order!
            "evidence_hierarchy": self.evidence_hierarchy,
        }

        if papers_df.empty:
            # No papers after filtering - use fallback LLM generation with streaming
            logger.info("📭 No papers after filtering, using fallback streaming generation")
            yield {"step": "generation", "status": "running", "message": "Generating response from medical knowledge..."}
            fallback_gen = self._run_fallback_generation(
                query,
                processed_query=processed_query,
                stream=True,
            )
            final_answer = ""
            follow_up_questions = []
            for event in fallback_gen:
                if event["step"] == "generation" and event["status"] == "running":
                    yield event
                elif event["step"] == "generation" and event["status"] == "complete":
                    final_answer = event.get("answer", "")
                    follow_up_questions = event.get("follow_up_questions", []) or []
            yield {
                "step": "complete",
                "status": "fallback",
                "report_title": "Clinical Response (No Sources)",
                "answer": final_answer,
                "follow_up_questions": follow_up_questions,
                "sources": initial_sources if initial_sources else [],
                "evidence_hierarchy": self.evidence_hierarchy,
                "abstracts_used": 0
            }
            self._cache_set(query, {
                "query": query,
                "report_title": "Clinical Response (No Sources)",
                "answer": final_answer,
                "follow_up_questions": follow_up_questions,
                "sections": [],
                "sources": initial_sources if initial_sources else [],
                "evidence_hierarchy": self.evidence_hierarchy,
                "status": "fallback",
                "retrieval_stats": self._build_retrieval_stats(
                    passages_retrieved=len(passages),
                    papers_after_aggregation=0,
                    abstracts_used=0,
                ),
            }, thread_context=thread_context)
            return

        # used_papers already computed above; reuse for generation

        # START PARALLEL TASKS: Generation + PDF Check
        yield {"step": "pdf_check", "status": "running", "message": "Checking PDF availability..."}
        yield {"step": "generation", "status": "running", "message": "Synthesizing response..."}
        
        pdf_results = []
        pdf_yielded = False
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Start PDF check in background
            pdf_future = executor.submit(self._check_pdf_availability, used_papers)
            
            # Start streaming generation
            answer = ""
            follow_up_questions = []
            generation_gen = self.run_generation(
                query,
                papers_df,
                processed_query=processed_query,
                stream=True,
            )
            for event in generation_gen:
                if event["step"] == "generation" and event["status"] == "running":
                    if "token" in event:
                        yield event
                    
                    # [NEW] Check if PDF future is done during token synthesis
                    # Yield PDF links AS SOON AS AVAILABLE instead of waiting for end
                    if not pdf_yielded and pdf_future.done():
                        try:
                            pdf_results = pdf_future.result()
                            pdf_count = sum(1 for s in pdf_results if s.get("pdf_url"))
                            yield {
                                "step": "pdf_check", 
                                "status": "complete", 
                                "data": {"pdf_count": pdf_count},
                                "sources": pdf_results, # UPDATE SOURCES WITH PDFs NOW
                                "evidence_hierarchy": self.evidence_hierarchy,
                            }
                            pdf_yielded = True
                            logger.info(f"   ✅ PDF check yielded early ({pdf_count} PDFs found)")
                        except Exception as e:
                            logger.error(f"Error yielding PDF results early: {e}")

                elif event["step"] == "generation" and event["status"] == "complete":
                    answer = event["answer"]
                    follow_up_questions = event.get("follow_up_questions", []) or []
            
            # Final safety check if not already yielded
            if not pdf_results:
                pdf_results = pdf_future.result()

        if not pdf_yielded:
            yield {
                "step": "pdf_check",
                "status": "complete",
                "data": {"pdf_count": sum(1 for s in pdf_results if s.get("pdf_url"))},
                "sources": pdf_results,
                "evidence_hierarchy": self.evidence_hierarchy,
            }
        yield {"step": "generation", "status": "complete"}
        
        # Filter final sources to cited references; fallback logic keeps full list when no citations parsed.
        final_sources_base = pdf_results if pdf_results else initial_sources
        final_sources = self._filter_sources_to_citations(answer, final_sources_base)
        answer, final_sources = self._normalize_answer_and_sources(answer, final_sources)
        self._last_retrieval_stats["dailymed_final_sources_returned"] = sum(
            1 for source in final_sources if self._is_dailymed_row(source)
        )
        
        logger.info(f"🔍 Streaming Complete: Returning {len(final_sources)} sources")

        # Final event
        final_result = {
            "step": "complete",
            "status": "success",
            "report_title": "Clinical Response",
            "answer": answer,
            "follow_up_questions": follow_up_questions,
            "sources": final_sources,
            "evidence_hierarchy": self.evidence_hierarchy,
            "abstracts_used": len(used_papers),
            "original_sources_count": len(pdf_results),
            "retrieval_stats": self._build_retrieval_stats(
                passages_retrieved=len(passages),
                papers_after_aggregation=len(papers_df),
                abstracts_used=len(used_papers),
            ),
        }
        yield final_result
        self._cache_set(query, {
            "query": query,
            "report_title": final_result["report_title"],
            "answer": final_result["answer"],
            "follow_up_questions": final_result["follow_up_questions"],
            "sections": [],
            "sources": final_result["sources"],
            "evidence_hierarchy": final_result["evidence_hierarchy"],
            "status": "success",
            "retrieval_stats": self._build_retrieval_stats(
                passages_retrieved=len(passages),
                papers_after_aggregation=len(papers_df),
                abstracts_used=len(used_papers),
            ),
        }, thread_context=thread_context)
    
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 Testing Elixir Medical RAG Pipeline")
    print("=" * 70)
    
    pipeline = MedicalRAGPipeline()
    result = pipeline.answer("management of copd")
    
    print(f"\n📋 Report: {result['report_title']}")
    print(f"📚 Sources: {len(result['sources'])}")
    print(f"\n{result['answer'][:2000]}...")
