"""
Query preprocessor for Medical RAG retrieval.

Responsibilities:
- Correct likely medical typos
- Distill verbose prompts (including long clinical vignettes) into compact retrieval queries
- Extract key entities for downstream entity filtering
- Detect explicit drug-information intent for DailyMed routing
"""

import json
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, NamedTuple, Tuple

import httpx
from openai import OpenAI
from pydantic import BaseModel, Field

from .config import (
    DEEPINFRA_API_KEY,
    DEEPINFRA_BASE_URL,
    XAI_API_KEY,
    XAI_BASE_URL,
    LLM_PROVIDER,
    LLM_RETRY_COUNT,
    LLM_RETRY_DELAY,
    LLM_CHAT_TIMEOUT_SECONDS,
    LLM_MAX_COMPLETION_TOKENS,
    LLM_REASONING_EFFORT,
    QUERY_PREPROCESSOR_LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    RETRIEVAL_QUERY_VARIANT_LIMIT,
    UPSTREAM_HTTP_MAX_CONNECTIONS,
    UPSTREAM_HTTP_MAX_KEEPALIVE,
    UPSTREAM_HTTP_KEEPALIVE_EXPIRY,
)
from .retry_utils import retry_with_exponential_backoff

logger = logging.getLogger(__name__)


class DecomposedQuery(BaseModel):
    """Minimal query decomposition output from LLM."""

    corrected_query: str = Field(
        default="", description="Typo-corrected query text (empty if no correction needed)."
    )
    primary_query: str = Field(
        description="Primary semantic retrieval query. Keep compact and clinically specific."
    )
    keyword_query: str = Field(
        description="Compact keyword retrieval query for sparse/BM25 retrieval."
    )
    key_entities: List[str] = Field(
        default=[],
        description="Key diseases/conditions/findings from the user query (original spelling).",
    )
    corrected_entities: List[str] = Field(
        default=[],
        description="Typo-corrected key entities for matching and recall.",
    )
    is_drug_query: bool = Field(
        default=False,
        description="True only when user intent is medication information (dose/adverse effects/interactions/contraindications/MOA).",
    )
    is_usmle_query: bool = Field(
        default=False,
        description="True only for board-style clinical vignette questions asking for the best answer, next step, diagnosis, or definitive test.",
    )
    drug_names: List[str] = Field(
        default=[], description="Pharmaceutical drug names only (generic and applicable brand names)."
    )
    dailymed_keywords: List[str] = Field(
        default=[],
        description="Drug-focused normalized keywords used for DailyMed lookup routing.",
    )
    conversation_summary: str = Field(
        default="",
        description="One short sentence summarizing initial query and latest thread context for final answer synthesis.",
    )


class LLMProcessedQuery(NamedTuple):
    """Result of LLM query processing for retrieval."""

    primary_query: str
    keyword_query: str
    original_query: str
    decomposed: Optional[DecomposedQuery] = None
    retrieval_queries: List[str] = []
    option_queries: List[str] = []
    option_parse_success: bool = False
    conversation_summary: str = ""


QUERY_DECOMPOSER_PROMPT = """
<task>
You are a medical retrieval query preprocessor.
Analyze the user query and output compact retrieval-focused JSON.
If the input is a long clinical vignette (USMLE style), distill it into a concise 15-25 word primary_query that keeps only high-yield findings and question intent; never copy the full vignette.
Set is_drug_query=true ONLY when the user explicitly asks medication information (dose, adverse effects, interactions, contraindications, mechanism, administration).
Set is_usmle_query=true ONLY when the query is a board-style exam question (for example: clinical vignette + MCQ options, or lead-ins like next best step, most appropriate management, most likely diagnosis, definitive test).
Short topical requests are NOT USMLE queries. For example, "diagnosis of myxedema coma", "management of atrial fibrillation", and "workup for hyponatremia" must set is_usmle_query=false unless the user explicitly frames them as a board-style vignette or multiple-choice exam question.
If a drug is only background history in a case vignette, set is_drug_query=false and drug_names=[].
If a medication name appears misspelled, normalize it to the official generic drug name in corrected_query and drug_names.
Current year: __CURRENT_YEAR__
</task>

<fields>
corrected_query: Full typo-corrected text only when you are confident there is a misspelling; otherwise "".
primary_query: Main dense retrieval query. Compact and specific. Remove answer option blocks like (A)...(E).
keyword_query: Compact sparse retrieval query. May include condensed option terms for differential diagnosis.
key_entities: Key conditions/findings from user query (original spelling).
corrected_entities: Corrected spellings of key_entities when applicable.
is_drug_query: boolean with strict intent rule from <task>.
is_usmle_query: boolean with strict board-style exam question rule from <task>.
drug_names: Pharmaceutical names only (generic + applicable brand names). Never output conditions, anatomy, procedures, symptoms, labs, or demographics here.
dailymed_keywords: Drug-focused lookup keywords for DailyMed. Prefer normalized drug names; include concise aliases only when helpful.
conversation_summary: One short sentence (max ~25 words) summarizing the conversation context needed for final answer synthesis.
</fields>

<examples>
<example input>
What are the contraindications and adverse effects of lisinopril?
</example input>
<example output>
{
  "corrected_query": "",
  "primary_query": "lisinopril contraindications adverse effects",
  "keyword_query": "lisinopril contraindication adverse effects safety",
  "key_entities": ["hypertension"],
  "corrected_entities": [],
  "is_drug_query": true,
  "is_usmle_query": false,
  "drug_names": ["lisinopril"],
  "dailymed_keywords": ["lisinopril"],
  "conversation_summary": "Question asks for lisinopril safety profile, focused on contraindications and adverse effects."
}
</example output>

<example input>
transaminitis secondary to avocopan therapy
</example input>
<example output>
{
  "corrected_query": "transaminitis secondary to avacopan therapy",
  "primary_query": "avacopan-induced transaminitis hepatotoxicity",
  "keyword_query": "avacopan transaminitis hepatotoxicity liver injury adverse effect",
  "key_entities": ["transaminitis", "avacopan therapy"],
  "corrected_entities": ["transaminitis", "avacopan therapy"],
  "is_drug_query": true,
  "is_usmle_query": false,
  "drug_names": ["avacopan"],
  "dailymed_keywords": ["avacopan"],
  "conversation_summary": "Query focuses on suspected avacopan-associated liver injury and transaminitis adverse effects."
}
</example output>

<example input>
A 68-year-old female presents with 5 days of fever, chills, and painful swelling in the right groin with leukocytosis and neutrophilia. She has cat exposure. CT shows enlarged inflamed inguinal lymph node without abscess. What is the most likely diagnosis? (A) Lymphogranuloma venereum (B) Cat scratch disease (C) Pyogenic lymphadenitis (D) Inguinal hernia (E) Necrotizing fasciitis
</example input>
<example output>
{
  "corrected_query": "",
  "primary_query": "inguinal lymphadenopathy tender groin mass fever cat exposure leukocytosis neutrophilia differential diagnosis",
  "keyword_query": "inguinal lymphadenopathy fever cat scratch disease pyogenic lymphadenitis neutrophilia diagnosis",
  "key_entities": ["inguinal lymphadenopathy", "fever"],
  "corrected_entities": [],
  "is_drug_query": false,
  "is_usmle_query": true,
  "drug_names": [],
  "dailymed_keywords": [],
  "conversation_summary": "USMLE-style vignette on painful inguinal lymphadenopathy with cat exposure, asking for most likely diagnosis."
}
</example output>
</examples>

Output valid JSON only. No markdown.
"""


USMLE_OPTION_QUERY_PROMPT = """
You are Elixir AI, a medical assistant trained extensively on USMLE questionnaires.
Your task is to analyze a USMLE-style multiple-choice question and produce concise retrieval queries.

Return valid JSON only with this schema:
{
  "queries": [
    {"label": "A", "query": "..."},
    {"label": "B", "query": "..."}
  ]
}

Rules:
- Create one query for each provided option label.
- Each query must be at most 2 sentences.
- Keep each query concise and retrieval-focused (high-yield findings + option-specific differential cue).
- Do not include answer letters in query text.
- Do not include markdown.
"""


class QueryPreprocessor:
    """LLM-powered preprocessing for retrieval-focused medical queries."""

    _DRUG_INTENT_TERMS = (
        "dose",
        "dosing",
        "dosage",
        "contraindication",
        "contraindications",
        "adverse",
        "side effect",
        "side effects",
        "interaction",
        "interactions",
        "mechanism",
        "moa",
        "pharmacokinet",
        "administration",
    )
    _USMLE_LEAD_IN_TERMS = (
        "next best step",
        "most appropriate next step",
        "most appropriate management",
        "most appropriate treatment",
        "most likely diagnosis",
        "most likely cause",
        "most likely explanation",
        "most likely mechanism",
        "best initial test",
        "best diagnostic test",
        "definitive test",
        "which of the following",
        "what is the diagnosis",
    )
    _USMLE_EXPLICIT_EXAM_TERMS = (
        "usmle",
        "step 1",
        "step 2",
        "step 3",
        "nbme",
        "shelf exam",
        "board-style",
        "board question",
        "multiple choice",
        "multiple-choice",
        "mcq",
    )
    _USMLE_VIGNETTE_TERMS = (
        "presents with",
        "history of",
        "physical examination",
        "laboratory",
        "vital signs",
        "blood pressure",
        "heart rate",
        "temperature",
        "found unresponsive",
        "emergency department",
    )
    _USMLE_OPTION_MARKER_RE = re.compile(r"(?:\([A-E]\)|\b[A-E]\))\s*", flags=re.IGNORECASE)
    @staticmethod
    def _conversation_id_from_thread_context(thread_context: Optional[Dict[str, Any]]) -> str:
        if not isinstance(thread_context, dict):
            return ""
        for field in ("conversation_id", "thread_id", "session_id", "chat_id"):
            value = str(thread_context.get(field, "")).strip()
            if value:
                return value[:120]
        return ""

    _USMLE_PATIENT_CUE_RE = re.compile(
        r"\b(?:\d{1,3}-year-old|patient|man|woman|male|female|infant|child|boy|girl|newborn|neonate)\b"
    )

    _FALLBACK_STOPWORDS = {
        "the",
        "and",
        "with",
        "from",
        "that",
        "this",
        "have",
        "been",
        "into",
        "what",
        "which",
        "when",
        "where",
        "does",
        "about",
        "without",
        "normal",
        "reports",
        "patient",
        "female",
        "male",
        "year",
        "years",
        "old",
    }
    _DAILYMED_KEYWORD_MAX = 15

    def __init__(self, model: str = QUERY_PREPROCESSOR_LLM_MODEL):
        """Initialize query decomposition LLM client."""
        self.llm_provider = LLM_PROVIDER
        self._openai_http_client = self._build_openai_http_client(timeout_seconds=LLM_CHAT_TIMEOUT_SECONDS)
        self.llm_client = self._create_llm_client(self.llm_provider)
        self.model = model
        self.retry_count = LLM_RETRY_COUNT
        self.retry_delay = LLM_RETRY_DELAY
        logger.info("LLM query preprocessor provider initialized: %s (%s)", self.llm_provider, self.model)

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
            if not DEEPINFRA_API_KEY:
                raise ValueError("DEEPINFRA_API_KEY not set")
            return OpenAI(
                api_key=DEEPINFRA_API_KEY,
                base_url=DEEPINFRA_BASE_URL,
                timeout=LLM_CHAT_TIMEOUT_SECONDS,
                http_client=self._openai_http_client,
            )
        if provider == "xai":
            if not XAI_API_KEY:
                raise ValueError("XAI_API_KEY not set")
            return OpenAI(
                api_key=XAI_API_KEY,
                base_url=XAI_BASE_URL,
                timeout=LLM_CHAT_TIMEOUT_SECONDS,
                http_client=self._openai_http_client,
            )
        raise ValueError(f"Unsupported LLM provider: {provider}")

    def _build_request_kwargs(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        provider: str,
        conversation_id: str = "",
    ) -> Dict[str, Any]:
        """Build provider-specific chat completion kwargs."""
        request_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
        }
        request_kwargs["max_completion_tokens"] = LLM_MAX_COMPLETION_TOKENS
        if LLM_REASONING_EFFORT and provider != "xai":
            request_kwargs["reasoning_effort"] = LLM_REASONING_EFFORT
        if provider == "xai" and conversation_id:
            request_kwargs["extra_headers"] = {"x-grok-conv-id": conversation_id}
        return request_kwargs

    def _normalize_response_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Map legacy keys to the minimal schema for robustness."""
        normalized = dict(payload)

        if "primary_query" not in normalized and "rewritten_query" in normalized:
            normalized["primary_query"] = normalized.get("rewritten_query", "")
        if "keyword_query" not in normalized and "rewritten_query_for_keyword_search" in normalized:
            normalized["keyword_query"] = normalized.get("rewritten_query_for_keyword_search", "")
        if "key_entities" not in normalized and "medical_conditions" in normalized:
            normalized["key_entities"] = normalized.get("medical_conditions") or []
        if "corrected_entities" not in normalized and "corrected_medical_conditions" in normalized:
            normalized["corrected_entities"] = normalized.get("corrected_medical_conditions") or []
        if "dailymed_keywords" not in normalized:
            for legacy_key in ("dailymed_terms", "drug_lookup_keywords", "dailymed_query_terms"):
                if legacy_key in normalized:
                    normalized["dailymed_keywords"] = normalized.get(legacy_key) or []
                    break

        normalized.setdefault("corrected_query", "")
        normalized.setdefault("key_entities", [])
        normalized.setdefault("corrected_entities", [])
        normalized.setdefault("drug_names", [])
        normalized.setdefault("dailymed_keywords", [])
        normalized.setdefault("conversation_summary", "")
        normalized.setdefault("is_drug_query", False)
        normalized.setdefault("is_usmle_query", False)

        for boolean_field in ("is_drug_query", "is_usmle_query"):
            if not isinstance(normalized.get(boolean_field), bool):
                normalized[boolean_field] = str(normalized.get(boolean_field, "")).strip().lower() in {
                    "true",
                    "1",
                    "yes",
                    "on",
                }
        for list_field in ("key_entities", "corrected_entities", "drug_names", "dailymed_keywords"):
            value = normalized.get(list_field)
            if value is None:
                normalized[list_field] = []
            elif isinstance(value, str):
                normalized[list_field] = [value]
            elif not isinstance(value, list):
                normalized[list_field] = list(value)
        if not isinstance(normalized.get("conversation_summary"), str):
            normalized["conversation_summary"] = str(normalized.get("conversation_summary", "")).strip()
        else:
            normalized["conversation_summary"] = normalized["conversation_summary"].strip()

        return normalized

    @staticmethod
    def _build_conversation_summary_fallback(
        query: str,
        thread_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not isinstance(thread_context, dict):
            return re.sub(r"\s+", " ", query).strip()[:220]
        initial_query = re.sub(r"\s+", " ", str(thread_context.get("initial_query", "")).strip())
        latest_user = re.sub(r"\s+", " ", str(thread_context.get("latest_user_query", "")).strip())
        latest_answer = re.sub(r"\s+", " ", str(thread_context.get("latest_assistant_answer", "")).strip())
        current = re.sub(r"\s+", " ", query).strip()
        if not initial_query:
            return current[:220]
        parts = [f"Initial query: {initial_query}."]
        if latest_user:
            parts.append(f"Prior user follow-up: {latest_user}.")
        if latest_answer:
            parts.append(f"Prior answer focus: {latest_answer[:140]}.")
        parts.append(f"Current follow-up: {current}.")
        return " ".join(parts)[:220]

    @staticmethod
    def _build_decomposition_user_input(query: str, thread_context: Optional[Dict[str, Any]] = None) -> str:
        if not isinstance(thread_context, dict) or not thread_context.get("is_follow_up"):
            return query

        initial_query = str(thread_context.get("initial_query", "")).strip()
        latest_user_query = str(thread_context.get("latest_user_query", "")).strip()
        latest_assistant_answer = str(thread_context.get("latest_assistant_answer", "")).strip()
        follow_up_count = thread_context.get("follow_up_count", 0)
        answer_snippet = re.sub(r"\s+", " ", latest_assistant_answer).strip()
        answer_tokens = answer_snippet.split()
        answer_snippet = " ".join(answer_tokens[:90])

        return (
            "Follow-up conversation context:\n"
            f"- Initial query: {initial_query}\n"
            f"- Latest user query: {latest_user_query}\n"
            f"- Latest assistant answer (summary snippet): {answer_snippet}\n"
            f"- Follow-up turn count so far: {follow_up_count}\n"
            f"- Current follow-up query: {query}\n\n"
            "Use this context to generate retrieval-focused fields for the current follow-up query and "
            "provide a concise one-sentence conversation_summary for final synthesis."
        )

    @staticmethod
    def _normalize_keyword_phrase(value: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\-\s]", " ", str(value or "").strip().lower())).strip()

    def _derive_dailymed_keywords(
        self,
        query: str,
        primary_query: str,
        drug_names: List[str],
        llm_keywords: List[str],
        *,
        is_drug_query: bool,
    ) -> List[str]:
        candidates: List[str] = []

        for source_term in [*(llm_keywords or []), *(drug_names or [])]:
            normalized = self._normalize_keyword_phrase(source_term)
            if normalized:
                candidates.append(normalized)

        deduped: List[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            deduped.append(candidate)
            if len(deduped) >= self._DAILYMED_KEYWORD_MAX:
                break

        if deduped or not is_drug_query:
            return deduped

        fallback = self._compact_text(primary_query or query, max_tokens=6)
        fallback_norm = self._normalize_keyword_phrase(fallback)
        return [fallback_norm] if fallback_norm else []

    def _parse_llm_response(self, content: str) -> DecomposedQuery:
        """Parse LLM response, handling markdown wrappers and legacy keys."""
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
        payload = self._normalize_response_payload(json.loads(content))
        return DecomposedQuery(**payload)

    def _get_query_decomposer_prompt(self) -> str:
        """Render query decomposer prompt with current year."""
        return QUERY_DECOMPOSER_PROMPT.replace("__CURRENT_YEAR__", str(datetime.now().year))

    def _strip_mcq_options(self, text: str) -> str:
        """Strip multiple-choice answer option blocks such as (A) ... (E)."""
        option_marker = re.compile(r"(?:\([A-E]\)|\b[A-E]\))\s*", flags=re.IGNORECASE)
        if not option_marker.search(text):
            return re.sub(r"\s+", " ", text).strip()
        stem = option_marker.split(text, maxsplit=1)[0]
        return re.sub(r"\s+", " ", stem).strip()

    def _parse_usmle_options(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """Extract stem and option blocks from common MCQ formats."""
        option_pattern = re.compile(
            r"(?:(?<=\s)|^)(?:\(([A-E])\)|([A-E])[\.\)])\s*",
            flags=re.IGNORECASE,
        )
        matches = list(option_pattern.finditer(text or ""))
        if not matches:
            return re.sub(r"\s+", " ", str(text or "")).strip(), []

        stem = re.sub(r"\s+", " ", str(text[: matches[0].start()] or "")).strip()
        options: List[Dict[str, str]] = []
        for idx, match in enumerate(matches):
            label = (match.group(1) or match.group(2) or "").upper()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            option_text = re.sub(r"\s+", " ", str(text[start:end] or "")).strip()
            option_text = re.sub(r"^[\-\:\)]\s*", "", option_text).strip()
            if not label or not option_text:
                continue
            options.append({"label": label, "text": option_text})

        return stem, options

    @staticmethod
    def _truncate_to_two_sentences(text: str) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if not normalized:
            return ""
        sentences = re.split(r"(?<=[\.\!\?])\s+", normalized)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) <= 2:
            return normalized
        return " ".join(sentences[:2]).strip()

    def _build_option_query_fallback(
        self,
        stem: str,
        option_label: str,
        option_text: str,
    ) -> str:
        stem_compact = self._compact_text(stem, max_tokens=28)
        option_compact = self._compact_text(option_text, max_tokens=16)
        fallback_query = f"{stem_compact} differential includes {option_compact}"
        fallback_query = self._truncate_to_two_sentences(fallback_query)
        fallback_query = re.sub(r"\s+", " ", fallback_query).strip()
        if not fallback_query:
            return f"USMLE clinical vignette option {option_label} {option_compact}".strip()
        return fallback_query

    def _build_stem_fallback_query(self, stem: str, original_query: str) -> str:
        base = stem or self._strip_mcq_options(original_query) or original_query
        return self._compact_text(base, max_tokens=32)

    def _generate_usmle_option_queries(
        self,
        stem: str,
        options: List[Dict[str, str]],
        thread_context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        if not options:
            return []

        option_lines = "\n".join(
            [f"{item['label']}: {item['text']}" for item in options if item.get("label") and item.get("text")]
        )
        user_prompt = (
            f"Question stem:\n{stem}\n\n"
            f"Answer options:\n{option_lines}\n\n"
            "Generate option-specific retrieval queries."
        )

        label_to_query: Dict[str, str] = {}
        try:
            response = self._chat_completion_with_retry(
                messages=[
                    {"role": "system", "content": USMLE_OPTION_QUERY_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                operation_name=f"{self.llm_provider} usmle option query generation",
                thread_context=thread_context,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n?", "", content)
                content = re.sub(r"\n?```$", "", content)
            payload = json.loads(content)
            query_rows = payload.get("queries", []) if isinstance(payload, dict) else []
            if isinstance(query_rows, list):
                for row in query_rows:
                    if not isinstance(row, dict):
                        continue
                    label = str(row.get("label", "")).strip().upper()
                    query_text = self._truncate_to_two_sentences(str(row.get("query", "")).strip())
                    query_text = re.sub(r"\s+", " ", query_text).strip()
                    if label and query_text:
                        label_to_query[label] = self._compact_text(query_text, max_tokens=40)
        except Exception as exc:
            logger.warning("USMLE option query generation failed: %s; using deterministic fallback", exc)

        ordered_queries: List[str] = []
        for option in options:
            label = str(option.get("label", "")).upper()
            option_text = str(option.get("text", "")).strip()
            query = label_to_query.get(label, "")
            if not query:
                query = self._build_option_query_fallback(stem, label, option_text)
            if query:
                ordered_queries.append(query)
        return ordered_queries

    def _extract_option_terms(self, text: str, limit: int = 10) -> List[str]:
        """Extract compact diagnostic hints from MCQ options for sparse query variant."""
        option_marker = re.compile(r"(?:\([A-E]\)|\b[A-E]\))\s*", flags=re.IGNORECASE)
        chunks = option_marker.split(text)
        if len(chunks) <= 1:
            return []

        terms: List[str] = []
        for chunk in chunks[1:]:
            cleaned = re.sub(r"[^a-zA-Z0-9\-\s]", " ", chunk).lower()
            tokens = [tok for tok in cleaned.split() if len(tok) >= 3 and tok not in self._FALLBACK_STOPWORDS]
            if not tokens:
                continue
            terms.append(" ".join(tokens[:4]))
            if len(terms) >= limit:
                break
        return terms[:limit]

    def _compact_text(self, text: str, max_tokens: int) -> str:
        """Normalize whitespace and truncate to max token budget."""
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return ""
        tokens = normalized.split()
        return " ".join(tokens[:max_tokens])

    def _is_long_vignette(self, query: str) -> bool:
        """Heuristic detection for long case-vignette prompts."""
        lowered = query.lower()
        indicators = (
            "presents",
            "physical examination",
            "laboratory",
            "blood pressure",
            "ct scan",
            "history",
        )
        return len(query.split()) >= 120 or sum(1 for term in indicators if term in lowered) >= 3

    def _guess_usmle_query(self, text: str) -> bool:
        """Deterministic routing classifier for board-style queries."""
        lowered = text.lower()
        has_options = bool(self._USMLE_OPTION_MARKER_RE.search(text))
        has_lead_in = any(term in lowered for term in self._USMLE_LEAD_IN_TERMS)
        has_explicit_exam_intent = any(term in lowered for term in self._USMLE_EXPLICIT_EXAM_TERMS)
        if has_explicit_exam_intent:
            return True
        if has_options:
            return True
        vignette_signal_count = sum(1 for term in self._USMLE_VIGNETTE_TERMS if term in lowered)
        has_patient_cue = bool(self._USMLE_PATIENT_CUE_RE.search(lowered))
        has_vignette_structure = (
            (has_patient_cue and (vignette_signal_count >= 1 or len(text.split()) >= 12))
            or vignette_signal_count >= 3
            or self._is_long_vignette(text)
        )
        if has_lead_in and has_vignette_structure:
            return True
        return False

    def _extract_key_entities_fallback(self, text: str, limit: int = 6) -> List[str]:
        """Best-effort extraction for key entities when LLM fails."""
        cleaned = re.sub(r"[^a-zA-Z0-9\-\s]", " ", text.lower())
        candidates = [
            tok
            for tok in cleaned.split()
            if len(tok) >= 5 and tok not in self._FALLBACK_STOPWORDS
        ]
        deduped = list(dict.fromkeys(candidates))
        return deduped[:limit]

    def _guess_drug_intent(self, text: str) -> bool:
        lowered = text.lower()
        return any(term in lowered for term in self._DRUG_INTENT_TERMS)

    @staticmethod
    def _append_unique_terms(base_query: str, candidate_query: str, max_new_terms: int = 8) -> str:
        """Append up to max_new_terms not already present in base_query."""
        base_tokens = re.findall(r"[a-z0-9]+", (base_query or "").lower())
        candidate_tokens = re.findall(r"[a-z0-9]+", (candidate_query or "").lower())
        if not candidate_tokens:
            return base_query

        existing = set(base_tokens)
        additions: List[str] = []
        for token in candidate_tokens:
            if token in existing:
                continue
            existing.add(token)
            additions.append(token)
            if len(additions) >= max_new_terms:
                break

        if not additions:
            return base_query
        return f"{base_query} {' '.join(additions)}".strip()

    def _build_unified_retrieval_query(
        self,
        primary_query: str,
        keyword_query: str,
        corrected_entities: List[str],
        option_terms: List[str],
        is_long_vignette: bool,
    ) -> str:
        """
        Build one hybrid-ready retrieval query for both dense and sparse retrieval.

        Start from semantic intent (primary_query), then add compact lexical hints.
        """
        unified = re.sub(r"\s+", " ", (primary_query or "").strip())
        if not unified:
            unified = re.sub(r"\s+", " ", (keyword_query or "").strip())

        # Borrow a small number of extra lexical tokens from keyword rewrite.
        unified = self._append_unique_terms(unified, keyword_query, max_new_terms=8)

        if corrected_entities:
            unified = self._append_unique_terms(unified, " ".join(corrected_entities), max_new_terms=6)

        if option_terms:
            unified = self._append_unique_terms(unified, " ".join(option_terms), max_new_terms=6)

        max_tokens = 24 if is_long_vignette else 32
        return self._compact_text(unified, max_tokens=max_tokens)

    def _build_retrieval_queries(
        self,
        *,
        original_query: str,
        primary_query: str,
        keyword_query: str,
        corrected_query: str,
        is_drug_query: bool,
        is_long_vignette: bool,
    ) -> List[str]:
        """Build a small retrieval query fanout for better typo tolerance and recall."""
        max_tokens = 24 if is_long_vignette else 32
        candidates = [self._compact_text(primary_query, max_tokens=max_tokens)]
        corrected_compact = self._compact_text(corrected_query, max_tokens=max_tokens)
        keyword_compact = self._compact_text(keyword_query, max_tokens=max_tokens)
        original_compact = self._compact_text(
            self._strip_mcq_options(original_query),
            max_tokens=max_tokens,
        )

        if corrected_compact:
            candidates.append(corrected_compact)
        if keyword_compact:
            candidates.append(keyword_compact)
        if is_drug_query or (corrected_compact and corrected_compact != original_compact):
            candidates.append(original_compact)

        deduped: List[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = re.sub(r"\s+", " ", str(candidate or "").strip())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
            if len(deduped) >= RETRIEVAL_QUERY_VARIANT_LIMIT:
                break
        return deduped or [primary_query or keyword_query or original_query]

    def _fallback_decompose(self, query: str, expanded_query: str) -> DecomposedQuery:
        """Deterministic fallback decomposition when LLM parsing/call fails."""
        stripped = self._strip_mcq_options(expanded_query)
        base = stripped or expanded_query or query

        primary_limit = 25 if self._is_long_vignette(query) else 45
        primary_query = self._compact_text(base, max_tokens=primary_limit)

        option_terms = self._extract_option_terms(query)
        key_entities = self._extract_key_entities_fallback(stripped or query)

        keyword_parts = key_entities + option_terms
        if not keyword_parts:
            keyword_parts = primary_query.split()[:24]
        keyword_query = self._compact_text(" ".join(keyword_parts), max_tokens=24)

        return DecomposedQuery(
            corrected_query="",
            primary_query=primary_query,
            keyword_query=keyword_query or primary_query,
            key_entities=key_entities,
            corrected_entities=[],
            is_drug_query=self._guess_drug_intent(query),
            is_usmle_query=self._guess_usmle_query(query),
            drug_names=[],
            dailymed_keywords=[],
        )

    def _chat_completion_with_retry(
        self,
        messages: List[Dict[str, str]],
        operation_name: str,
        thread_context: Optional[Dict[str, Any]] = None,
    ):
        """Execute chat completion with exponential-backoff retry."""
        conversation_id = ""
        if self.llm_provider == "xai":
            conversation_id = self._conversation_id_from_thread_context(thread_context)
        request_kwargs = self._build_request_kwargs(
            model=self.model,
            messages=messages,
            provider=self.llm_provider,
            conversation_id=conversation_id,
        )
        return retry_with_exponential_backoff(
            lambda: self.llm_client.chat.completions.create(**request_kwargs),
            max_attempts=self.retry_count + 1,
            base_delay=float(self.retry_delay),
            operation_name=operation_name,
            logger=logger,
        )

    def decompose_query(
        self,
        query: str,
        thread_context: Optional[Dict[str, Any]] = None,
    ) -> LLMProcessedQuery:
        """Decompose query into compact retrieval inputs using LLM."""
        try:
            user_input = self._build_decomposition_user_input(query, thread_context)
            response = self._chat_completion_with_retry(
                messages=[
                    {"role": "system", "content": self._get_query_decomposer_prompt()},
                    {"role": "user", "content": user_input},
                ],
                operation_name=f"{self.llm_provider} query decomposition",
                thread_context=thread_context,
            )
            decomposed = self._parse_llm_response(response.choices[0].message.content.strip())
        except Exception as exc:
            logger.warning("Query decomposition failed: %s; using deterministic fallback", exc)
            decomposed = self._fallback_decompose(query, query)

        corrected = decomposed.corrected_query.strip()
        primary_query = corrected or decomposed.primary_query or query

        is_long_vignette = self._is_long_vignette(query)
        if is_long_vignette:
            primary_query = self._compact_text(self._strip_mcq_options(primary_query), max_tokens=25)
            keyword_seed = self._compact_text((decomposed.keyword_query or primary_query), max_tokens=24)
        else:
            primary_query = self._compact_text(primary_query, max_tokens=45)
            keyword_seed = self._compact_text((decomposed.keyword_query or primary_query), max_tokens=28)

        option_terms = self._extract_option_terms(query)
        unified_query = self._build_unified_retrieval_query(
            primary_query=primary_query,
            keyword_query=keyword_seed,
            corrected_entities=decomposed.corrected_entities,
            option_terms=option_terms,
            is_long_vignette=is_long_vignette,
        )

        # Keep compatibility fields while using one retrieval query for both dense+sparse.
        primary_query = unified_query
        keyword_query = unified_query

        # If LLM returns no explicit drug flag, infer conservatively from intent + extracted drugs.
        if not decomposed.is_drug_query and decomposed.drug_names and self._guess_drug_intent(query):
            decomposed = decomposed.model_copy(update={"is_drug_query": True})

        resolved_usmle_query = self._guess_usmle_query(query)
        if decomposed.is_usmle_query != resolved_usmle_query:
            decomposed = decomposed.model_copy(update={"is_usmle_query": resolved_usmle_query})

        dailymed_keywords = self._derive_dailymed_keywords(
            query=query,
            primary_query=primary_query,
            drug_names=decomposed.drug_names or [],
            llm_keywords=decomposed.dailymed_keywords or [],
            is_drug_query=bool(decomposed.is_drug_query),
        )
        if dailymed_keywords != (decomposed.dailymed_keywords or []):
            decomposed = decomposed.model_copy(update={"dailymed_keywords": dailymed_keywords})

        conversation_summary = decomposed.conversation_summary.strip()
        if not conversation_summary:
            conversation_summary = self._build_conversation_summary_fallback(query, thread_context)
            decomposed = decomposed.model_copy(update={"conversation_summary": conversation_summary})

        option_queries: List[str] = []
        option_parse_success = False
        if decomposed.is_usmle_query:
            stem, parsed_options = self._parse_usmle_options(query)
            option_parse_success = len(parsed_options) >= 2
            if option_parse_success:
                option_queries = self._generate_usmle_option_queries(
                    stem or primary_query,
                    parsed_options,
                    thread_context=thread_context,
                )
            if option_parse_success and option_queries:
                retrieval_queries = option_queries
            else:
                retrieval_queries = [self._build_stem_fallback_query(stem or primary_query, query)]
        else:
            retrieval_queries = self._build_retrieval_queries(
                original_query=query,
                primary_query=unified_query,
                keyword_query=keyword_query,
                corrected_query=corrected or primary_query,
                is_drug_query=bool(decomposed.is_drug_query),
                is_long_vignette=is_long_vignette,
            )

        logger.info(
            "Decomposed metadata: entities=%s corrected_entities=%s is_drug_query=%s is_usmle_query=%s drug_names=%s dailymed_keywords=%s",
            decomposed.key_entities,
            decomposed.corrected_entities,
            decomposed.is_drug_query,
            decomposed.is_usmle_query,
            decomposed.drug_names,
            decomposed.dailymed_keywords,
        )
        logger.info("Preprocessed query: %s", primary_query)

        return LLMProcessedQuery(
            primary_query=primary_query,
            keyword_query=keyword_query,
            original_query=query,
            decomposed=decomposed,
            retrieval_queries=retrieval_queries,
            option_queries=option_queries,
            option_parse_success=option_parse_success,
            conversation_summary=conversation_summary,
        )

if __name__ == "__main__":
    print("Testing Query Preprocessor")
    print("=" * 60)

    preprocessor = QueryPreprocessor()

    test_queries = [
        "What are the latest treatments for heart failure?",
        "Systematic reviews on metformin for type 2 diabetes from 2020",
        "Management of NUEROBROCELLOSIS",
        (
            "A 68-year-old female presents with 5 days of fever, chills, and painful swelling in the right groin. "
            "CT shows enlarged inflamed inguinal node. What is the most likely diagnosis? "
            "(A) Lymphogranuloma venereum (B) Cat scratch disease (C) Pyogenic lymphadenitis"
        ),
    ]

    for query in test_queries:
        print(f"\nOriginal: {query}")
        print("-" * 40)

        result = preprocessor.decompose_query(query)

        print(f"Preprocessed: {result.primary_query}")
        print(f"Retrieval queries: {result.retrieval_queries}")
        if result.decomposed:
            print(f"Drug intent: {result.decomposed.is_drug_query}")
            print(f"Drugs: {result.decomposed.drug_names}")
