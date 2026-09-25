"""
ScholarQA-Style Reranker with Paper Aggregation.

Implements ai2-scholarqa-lib's exact reranking methodology:
- Passage-level reranking using DeepInfra Qwen3-Reranker
- Paper aggregation using max rerank_score
- DataFrame output with ScholarQA reference_string format
"""

import logging
import re

from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from anyascii import anyascii
import httpx
from .config import (
    RERANKER_MODEL,
    DEEPINFRA_API_KEY,
    DEEPINFRA_RETRY_COUNT, DEEPINFRA_RETRY_DELAY,
    DEEPINFRA_RERANK_TIMEOUT_SECONDS,
    RERANK_EVAL_LIMIT,
    RERANK_KEEP_LIMIT,
    RERANK_TEXT_MAX_CHARS,
    RERANK_TABLE_ABSTRACT_MAX_CHARS,
    RERANK_EXACT_TEXT_DEDUPE_ENABLED,
    RERANKER_V2_ENABLED,
    TIER_1_BOOST as CFG_TIER_1_BOOST,
    TIER_2_BOOST as CFG_TIER_2_BOOST,
    TIER_3_BOOST as CFG_TIER_3_BOOST,
    TIER_4_PENALTY as CFG_TIER_4_PENALTY,
    RERANK_COUNTRY_BOOST_ENABLED,
    RERANK_COUNTRY_BOOST_MULTIPLIER,
    RERANK_COUNTRY_BOOST_POLICY,
    UPSTREAM_HTTP_MAX_CONNECTIONS,
    UPSTREAM_HTTP_MAX_KEEPALIVE,
    UPSTREAM_HTTP_KEEPALIVE_EXPIRY,
)
from .retry_utils import retry_with_exponential_backoff
from .specialty_journals import detect_guideline_society_signal, get_journal_tier



logger = logging.getLogger(__name__)

US_COUNTRY_ALIASES = {
    "US",
    "USA",
    "UNITED STATES",
    "UNITED STATES OF AMERICA",
}

EU27_ISO_CODES = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU",
    "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
}

EU27_COUNTRY_ALIASES = {
    "AUSTRIA",
    "BELGIUM",
    "BULGARIA",
    "CROATIA",
    "CYPRUS",
    "CZECHIA",
    "CZECH REPUBLIC",
    "DENMARK",
    "ESTONIA",
    "FINLAND",
    "FRANCE",
    "GERMANY",
    "GREECE",
    "HELLAS",
    "HUNGARY",
    "IRELAND",
    "ITALY",
    "LATVIA",
    "LITHUANIA",
    "LUXEMBOURG",
    "MALTA",
    "NETHERLANDS",
    "HOLLAND",
    "POLAND",
    "PORTUGAL",
    "ROMANIA",
    "SLOVAKIA",
    "SLOVAK REPUBLIC",
    "SLOVENIA",
    "SPAIN",
    "SWEDEN",
}


def _normalize_country_token(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if re.fullmatch(r"(?:[A-Z]\s+)+[A-Z]", text):
        text = text.replace(" ", "")
    return text


def _country_tokens(country_value: Any) -> List[str]:
    if country_value is None:
        return []
    raw = str(country_value).strip()
    if not raw:
        return []
    tokens = [segment.strip() for segment in re.split(r"[,;/|]", raw) if segment and segment.strip()]
    normalized = [_normalize_country_token(token) for token in tokens]
    return [token for token in normalized if token]


def _country_multiplier_for_doc(doc: Dict[str, Any]) -> tuple[float, bool, str]:
    if not RERANK_COUNTRY_BOOST_ENABLED:
        return 1.0, False, "none"

    source_family = str(doc.get("source_family", "")).strip().lower()
    if source_family == "dailymed":
        return 1.0, False, "none"

    if RERANK_COUNTRY_BOOST_POLICY != "us_eu27":
        return 1.0, False, "none"

    tokens = _country_tokens(doc.get("country"))
    if not tokens:
        return 1.0, False, "none"

    if any(token in US_COUNTRY_ALIASES for token in tokens):
        return float(RERANK_COUNTRY_BOOST_MULTIPLIER), True, "us"
    if any(token in EU27_ISO_CODES or token in EU27_COUNTRY_ALIASES for token in tokens):
        return float(RERANK_COUNTRY_BOOST_MULTIPLIER), True, "eu"
    return 1.0, False, "none"


# =============================================================================
# Utility Functions (from ScholarQA)
# =============================================================================

def make_int(value) -> int:
    """Convert value to int, returning 0 if not possible."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0


# =============================================================================
# Abstract Reranker Interface (matches ScholarQA)
# =============================================================================

# =============================================================================
# EVIDENCE HIERARCHY SYSTEM
# =============================================================================

# Tier multipliers - configurable via config.py / env vars.
# Legacy (v1) values kept for rollback when RERANKER_V2_ENABLED=False:
#   TIER_1=3.00  TIER_2=1.50  TIER_3=1.00  TIER_4=0.20
# v2 defaults (env-overridable): 2.00 / 1.25 / 1.00 / 0.40
_TIER_1_LEGACY, _TIER_2_LEGACY, _TIER_3_LEGACY, _TIER_4_LEGACY = 3.00, 1.50, 1.00, 0.20
TIER_1_BOOST   = CFG_TIER_1_BOOST   if RERANKER_V2_ENABLED else _TIER_1_LEGACY
TIER_2_BOOST   = CFG_TIER_2_BOOST   if RERANKER_V2_ENABLED else _TIER_2_LEGACY
TIER_3_BOOST   = CFG_TIER_3_BOOST   if RERANKER_V2_ENABLED else _TIER_3_LEGACY
TIER_4_PENALTY = CFG_TIER_4_PENALTY if RERANKER_V2_ENABLED else _TIER_4_LEGACY

# =============================================================================
# ENTITY EXTRACTION CONSTANTS
# =============================================================================

# Generic medical terms that add noise to entity matching (not specific disease terms)
ENTITY_STOPWORDS = {
    'management', 'treatment', 'therapy', 'diagnosis', 'clinical',
    'features', 'symptoms', 'guidelines', 'recommendations',
    'patients', 'patient', 'outcomes', 'approach', 'advances',
    'review', 'overview', 'practice', 'evidence', 'recent',
    'current', 'update', 'updates', 'prevention', 'screening',
    'assessment', 'evaluation', 'classification', 'pathogenesis',
    'epidemiology', 'prognosis', 'mechanism', 'pathophysiology',
    'presentation', 'complications', 'associated', 'chronic',
    'acute', 'primary', 'secondary', 'emerging', 'novel', 'standard',
    'latest', 'options', 'strategies', 'interventions',
    'adverse', 'effect', 'effects', 'safety',
}

# Article types for each tier
TIER_1_TYPES = {
    "guideline", "practice_guideline", "practice guideline",
    "systematic_review", "systematic review", 
    "meta_analysis", "meta-analysis", "meta analysis",
    "consensus", "consensus_statement", "consensus statement"
}
TIER_2_TYPES = {
    "clinical_trial", "clinical trial",
    "clinical_trial_phase_i", "clinical trial, phase i",
    "clinical_trial_phase_ii", "clinical trial, phase ii", 
    "clinical_trial_phase_iii", "clinical trial, phase iii",
    "clinical_trial_phase_iv", "clinical trial, phase iv",
    "controlled_clinical_trial", "controlled clinical trial",
    "multicenter_study", "multicenter study",
    "randomized_controlled_trial", "rct", "randomized controlled trial",
    "review_article", "review article", "review",
    "drug_label", "drug label"
}
TIER_4_TYPES = {
    "case_report", "case report",
    "case_series", "case series",
    "letter", "editorial", "comment", "commentary",
    "news", "correspondence"
}

# Title keywords that indicate guidelines (catches misclassified articles)
GUIDELINE_TITLE_KEYWORDS = [
    # Standard terms
    "guideline", "guidelines", 
    "consensus", "consensus statement",
    "recommendation", "recommendations",
    "position statement",
    "scientific statement",
    "clinical practice",
    # Evidence-based patterns
    "evidence-based", "evidence based",
    "management guidelines", "treatment guidelines",
    "international consensus",
    "standards of care",
    # Specific patterns
    "acr/vasculitis", "aha/acc",
]

def _normalize_match_text(value: Any) -> str:
    """Normalize free text for token-boundary matching."""
    return re.sub(r"[^a-z0-9]+", " ", anyascii(str(value or "")).lower()).strip()


def _normalize_whitespace(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()


def _edit_distance_with_cutoff(a: str, b: str, max_distance: int = 1) -> int:
    """Compute Levenshtein distance with a small cutoff for fuzzy entity matches."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_distance:
        return max_distance + 1

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        row_min = current[0]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if char_a == char_b else 1)
            value = min(insert_cost, delete_cost, replace_cost)
            current.append(value)
            row_min = min(row_min, value)
        if row_min > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


def _extract_query_focus_terms(medical_conditions: Optional[List[str]]) -> List[str]:
    """Convert query entities into compact focus terms for post-rerank calibration."""
    if not medical_conditions:
        return []

    focus_terms: List[str] = []
    seen: set[str] = set()
    for raw_condition in medical_conditions:
        normalized_condition = _normalize_match_text(raw_condition)
        if normalized_condition and normalized_condition not in seen:
            seen.add(normalized_condition)
            focus_terms.append(normalized_condition)

        for token in normalized_condition.split():
            if len(token) < 5 and not any(char.isdigit() for char in token):
                continue
            if token in ENTITY_STOPWORDS:
                continue
            if token not in seen:
                seen.add(token)
                focus_terms.append(token)
    return focus_terms


def _compute_query_focus_score(doc: Dict[str, Any], medical_conditions: Optional[List[str]]) -> float:
    """Score how directly a document matches query-specific entities or drug names."""
    focus_terms = _extract_query_focus_terms(medical_conditions)
    if not focus_terms:
        return 0.0

    normalized_text = _normalize_match_text(
        " ".join(
            str(part or "")
            for part in (
                doc.get("title"),
                doc.get("section_title"),
                doc.get("abstract"),
                doc.get("text"),
                doc.get("page_content"),
                doc.get("full_section_text"),
            )
        )
    )
    if not normalized_text:
        return 0.0

    text_tokens = set(normalized_text.split())
    matched_terms = 0
    for term in focus_terms:
        if " " in term:
            if _contains_normalized_phrase(normalized_text, term):
                matched_terms += 1
                continue
            tokens = term.split()
        else:
            tokens = [term]

        token_matches = 0
        for token in tokens:
            if token in text_tokens:
                token_matches += 1
                continue
            if any(
                _edit_distance_with_cutoff(token, candidate, max_distance=1) <= 1
                for candidate in text_tokens
                if abs(len(candidate) - len(token)) <= 1 and candidate[:1] == token[:1]
            ):
                token_matches += 1
        if token_matches == len(tokens):
            matched_terms += 1

    return matched_terms / max(1, len(focus_terms))


def compute_query_focus_score(doc: Dict[str, Any], focus_texts: Optional[List[str]]) -> float:
    """Public wrapper for query-focus scoring across pipeline stages."""
    return _compute_query_focus_score(doc, focus_texts)


def _focus_multiplier_from_score(score: float) -> float:
    if score <= 0.0:
        return 0.72
    if score >= 0.8:
        return 1.35
    if score >= 0.5:
        return 1.22
    if score >= 0.25:
        return 0.82
    return 0.72


def _contains_normalized_phrase(text: str, phrase: str) -> bool:
    if not text or not phrase:
        return False
    text_tokens = text.split()
    phrase_tokens = phrase.split()
    if len(phrase_tokens) > len(text_tokens):
        return False
    # Exact sliding-window match (original behaviour, unchanged)
    for idx in range(len(text_tokens) - len(phrase_tokens) + 1):
        if text_tokens[idx:idx + len(phrase_tokens)] == phrase_tokens:
            return True
    # Fuzzy fallback (v2): all phrase tokens present within a gap-tolerant window
    # e.g., "non small cell lung" matches "non-small-cell lung" after normalization
    if RERANKER_V2_ENABLED:
        window = len(phrase_tokens) + 2     # allow up to 2 extra tokens between phrase words
        phrase_set = set(phrase_tokens)
        for idx in range(max(0, len(text_tokens) - window + 1)):
            if phrase_set.issubset(set(text_tokens[idx:idx + window])):
                return True
    return False


def _normalize_publication_types(publication_types: Any) -> list[str]:
    if publication_types is None:
        return []
    if isinstance(publication_types, str):
        normalized = _normalize_match_text(publication_types)
        return [normalized] if normalized else []
    if isinstance(publication_types, dict):
        candidate = (
            publication_types.get("type")
            or publication_types.get("name")
            or publication_types.get("value")
        )
        normalized = _normalize_match_text(candidate)
        return [normalized] if normalized else []
    if isinstance(publication_types, (list, tuple, set)):
        normalized_values = []
        seen = set()
        for item in publication_types:
            if isinstance(item, dict):
                candidate = item.get("type") or item.get("name") or item.get("value")
            else:
                candidate = item
            normalized = _normalize_match_text(candidate)
            if normalized and normalized not in seen:
                seen.add(normalized)
                normalized_values.append(normalized)
        return normalized_values
    normalized = _normalize_match_text(publication_types)
    return [normalized] if normalized else []


def get_evidence_multiplier(
    article_type: str,
    title: str,
    publication_types: list = None,
    journal: Optional[str] = None,
    evidence_term: Optional[str] = None,
    evidence_source: Optional[str] = None,
) -> float:
    """
    Get evidence tier multiplier based on article type, title, and publication types.
    
    Priority:
    1. Publication types list (highest confidence)
    2. Society-guideline detection (context gated)
    3. Title keywords (catches misclassified guidelines)
    4. Article type field
    """
    article_type_lower = (article_type or "").lower().replace("-", "_").replace(" ", "_")
    title_normalized = _normalize_match_text(title)
    pub_types_lower = _normalize_publication_types(publication_types)

    # 1. Publication types list (from PubMed - most reliable)
    TIER1_PUBTYPE_PATTERNS = ["systematic review", "meta analysis", "guideline", "practice guideline", "consensus"]
    TIER2_PUBTYPE_PATTERNS = ["randomized controlled trial", "clinical trial", "review"]
    TIER4_PUBTYPE_PATTERNS = ["case report", "letter", "editorial", "comment", "news"]

    for pt in pub_types_lower:
        if any(_contains_normalized_phrase(pt, p) for p in TIER1_PUBTYPE_PATTERNS):
            return TIER_1_BOOST

    # 2. Society signal with guideline context (precision-first)
    society_signal = detect_guideline_society_signal(
        title=title,
        journal=journal,
        evidence_term=evidence_term,
        evidence_source=evidence_source,
        publication_types=pub_types_lower,
    )
    if society_signal["is_match"]:
        return TIER_1_BOOST

    # 3. Title-based detection (generic terms only)
    if any(_contains_normalized_phrase(title_normalized, _normalize_match_text(kw)) for kw in GUIDELINE_TITLE_KEYWORDS):
        return TIER_1_BOOST

    for pt in pub_types_lower:
        if any(_contains_normalized_phrase(pt, p) for p in TIER4_PUBTYPE_PATTERNS):
            return TIER_4_PENALTY
    for pt in pub_types_lower:
        if any(_contains_normalized_phrase(pt, p) for p in TIER2_PUBTYPE_PATTERNS):
            return TIER_2_BOOST

    # 4. Article type-based tier assignment
    if article_type_lower in TIER_1_TYPES or any(t in article_type_lower for t in ["guideline", "systematic", "meta"]):
        return TIER_1_BOOST
    
    if article_type_lower in TIER_2_TYPES or any(t in article_type_lower for t in ["trial", "review"]):
        return TIER_2_BOOST
    
    if article_type_lower in TIER_4_TYPES or any(t in article_type_lower for t in ["case", "letter", "editorial"]):
        return TIER_4_PENALTY
    
    return TIER_3_BOOST  # Default for standard research


def compute_metadata_multipliers(
    doc: Dict[str, Any],
    current_year: Optional[int] = None,
    include_recency: bool = True,
) -> Dict[str, Any]:
    """Return evidence, recency, and journal multipliers for a document."""
    if current_year is None:
        current_year = datetime.now().year

    tier_mult = get_evidence_multiplier(
        doc.get("article_type", ""),
        doc.get("title", ""),
        doc.get("publication_type", []),
        journal=doc.get("journal") or doc.get("venue"),
        evidence_term=doc.get("evidence_term"),
        evidence_source=doc.get("evidence_source"),
    )

    society_signal = detect_guideline_society_signal(
        title=doc.get("title"),
        journal=doc.get("journal") or doc.get("venue"),
        evidence_term=doc.get("evidence_term"),
        evidence_source=doc.get("evidence_source"),
        publication_types=doc.get("publication_type", []),
    )

    recency_mult = 1.0
    is_case_report = tier_mult == TIER_4_PENALTY
    if include_recency and not is_case_report:
        year = doc.get("year")
        if year:
            try:
                if RERANKER_V2_ENABLED:
                    age = current_year - int(year)
                    if age <= 1:
                        recency_mult = 1.50
                    elif age <= 3:
                        recency_mult = 1.35
                    elif age <= 5:
                        recency_mult = 1.18
                    elif age <= 7:
                        recency_mult = 1.06
                else:
                    if int(year) >= current_year - 2:
                        recency_mult = 1.10
            except (ValueError, TypeError):
                pass

    journal_mult = 1.0
    journal_tier = get_journal_tier(
        journal_name=doc.get("journal") or doc.get("venue"),
        nlm_id=doc.get("nlm_unique_id"),
    )
    if journal_tier == "specialty":
        journal_mult = 1.20
    elif journal_tier == "general":
        journal_mult = 1.15
    country_mult, country_boost_applied, country_boost_region = _country_multiplier_for_doc(doc)

    return {
        "evidence_tier": tier_mult,
        "recency_multiplier": recency_mult,
        "journal_tier": journal_tier,
        "journal_boost": journal_mult,
        "country_multiplier": country_mult,
        "country_boost_applied": country_boost_applied,
        "country_boost_region": country_boost_region,
        "boost_multiplier": tier_mult * recency_mult * journal_mult * country_mult,
        "is_case_report": is_case_report,
        "guideline_society_match": society_signal["is_match"],
        "matched_guideline_societies": society_signal["matched_societies"],
    }



class AbstractReranker:
    """Abstract base class for rerankers."""
    
    def get_scores(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[float]:
        raise NotImplementedError


# =============================================================================
# Self-Hosted Cross-Encoder Reranker (default)
# =============================================================================




# =============================================================================
# DeepInfra Reranker (OpenAI-compatible / Custom API)
# =============================================================================

class DeepInfraReranker(AbstractReranker):
    """
    Reranker using DeepInfra's inference endpoint for Qwen/Qwen3-Reranker models.
    
    API Documentation: https://deepinfra.com/Qwen/Qwen3-Reranker-0.6B/api
    """
    
    def __init__(self, model: str = None):
        self.model = model or RERANKER_MODEL
        if not DEEPINFRA_API_KEY:
            raise ValueError("DEEPINFRA_API_KEY not set")
        self.api_key = DEEPINFRA_API_KEY
        # DeepInfra inference endpoint for Qwen3-Reranker models
        self.api_url = f"https://api.deepinfra.com/v1/inference/{self.model}"
        self.retry_count = DEEPINFRA_RETRY_COUNT
        self.retry_delay = DEEPINFRA_RETRY_DELAY
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self._http_limits = httpx.Limits(
            max_connections=UPSTREAM_HTTP_MAX_CONNECTIONS,
            max_keepalive_connections=UPSTREAM_HTTP_MAX_KEEPALIVE,
            keepalive_expiry=UPSTREAM_HTTP_KEEPALIVE_EXPIRY,
        )
        self._http_timeout = httpx.Timeout(DEEPINFRA_RERANK_TIMEOUT_SECONDS)
        # Shared clients preserve connection reuse under high concurrency.
        self._client = httpx.Client(limits=self._http_limits, timeout=self._http_timeout)
        logger.info(f"✅ DeepInfra Reranker initialized with {self.model}")

    def _post_rerank_sync(self, payload: Dict[str, Any]) -> httpx.Response:
        response = self._client.post(
            self.api_url,
            json=payload,
            headers=self._headers,
        )
        response.raise_for_status()
        return response

    def get_scores(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[float]:
        """
        Get relevance scores for documents using DeepInfra inference API.
        
        DeepInfra Qwen3-Reranker API expects:
        - Request: {"queries": [...], "documents": [...]}
        - Response: {"scores": [...], "input_tokens": N}
        """
        if not documents:
            return []

        docs_to_score = documents[:top_n] if top_n else documents
        scored_count = len(docs_to_score)
        
        try:
            # DeepInfra Qwen3-Reranker API format
            payload = {
                "queries": [query],
                "documents": docs_to_score,
            }
            
            def _post_rerank() -> httpx.Response:
                return self._post_rerank_sync(payload)

            response = retry_with_exponential_backoff(
                _post_rerank,
                max_attempts=self.retry_count + 1,
                base_delay=float(self.retry_delay),
                operation_name="DeepInfra Qwen3-Reranker API call",
                logger=logger,
            )
            data = response.json()
            
            # DeepInfra returns scores directly in "scores" array
            scores = data.get("scores", [])

            if not scores:
                logger.critical("RERANKER DEGRADED: Empty scores from API — all scored passages will receive uniform 0.5 scores")
                return [0.5] * scored_count

            # Health check: detect all-identical scores (indicates API malfunction)
            unique_scores = set(round(s, 6) for s in scores)
            if len(unique_scores) <= 1:
                logger.critical(
                    "RERANKER HEALTH CHECK FAILED: All %d scores identical (%.4f). "
                    "API may be returning dummy values — reranking is effectively disabled.",
                    len(scores), scores[0]
                )

            # Log token usage and score distribution for monitoring
            input_tokens = data.get("input_tokens", 0)
            score_min, score_max = min(scores), max(scores)
            score_mean = sum(scores) / len(scores)
            logger.info(
                "   Reranker API: %d input tokens, %d docs | scores: min=%.3f, max=%.3f, mean=%.3f",
                input_tokens, scored_count, score_min, score_max, score_mean
            )

            return scores

        except Exception as e:
            logger.critical(
                "RERANKER DEGRADED: DeepInfra API call failed: %s — "
                "all %d passages will receive uniform 0.5 scores, reranking is disabled for this query",
                e, scored_count
            )
            return [0.5] * scored_count

# =============================================================================
# Paper Finder with Reranker (matches ScholarQA's PaperFinderWithReranker)
# =============================================================================

class PaperFinderWithReranker:
    """
    ScholarQA-style paper finder with reranking.
    
    Flow:
    1. Rerank passages using DeepInfra Qwen3-Reranker
    2. Aggregate passages to paper level (max score)
    3. Format into DataFrame with reference strings
    """
    
    def __init__(
        self,
        reranker: AbstractReranker = None,
        n_rerank: Optional[int] = None,
        context_threshold: float = 0.0,
        rerank_eval_limit: int = RERANK_EVAL_LIMIT,
        rerank_keep_limit: Optional[int] = None,
        exact_text_dedupe_enabled: Optional[bool] = None,
    ):
        """
        Initialize paper finder.
        
        Args:
            reranker: Reranker engine (defaults based on RERANKER_PROVIDER config)
            n_rerank: Backward-compatible alias for rerank_keep_limit
            context_threshold: Min score threshold
        """
        if reranker is not None:
            self.reranker_engine = reranker
        else:
            self.reranker_engine = DeepInfraReranker(model=RERANKER_MODEL)
        
        self.rerank_eval_limit = int(rerank_eval_limit)
        if rerank_keep_limit is None:
            rerank_keep_limit = n_rerank if n_rerank is not None else RERANK_KEEP_LIMIT
        self.rerank_keep_limit = int(rerank_keep_limit)
        self.context_threshold = context_threshold
        self.exact_text_dedupe_enabled = (
            RERANK_EXACT_TEXT_DEDUPE_ENABLED
            if exact_text_dedupe_enabled is None
            else bool(exact_text_dedupe_enabled)
        )
        self.last_rerank_stats = {
            "rerank_passages_retrieved": 0,
            "rerank_passages_scored": 0,
            "rerank_passages_kept": 0,
        }
    
    def rerank(
        self,
        query: str,
        retrieved_ctxs: List[Dict[str, Any]],
        pre_filter_threshold: float = 0.0,  # Disabled by default: retrieval score scales vary by retrieval mode
        medical_conditions: Optional[List[str]] = None,  # LLM-extracted conditions for entity scoring
    ) -> List[Dict[str, Any]]:
        """
        Rerank passages using DeepInfra Qwen3-Reranker.
        
        Threshold Strategy:
        1. pre_filter_threshold (0.0 default): Optional filter by retrieval score BEFORE reranking
           - Can reduce API costs when retrieval score scale is stable
           - Disabled by default to avoid score-scale coupling across retrieval modes

        Pipeline:
        1. Pre-filter by retrieval score (reduces API cost)
        2. Run Qwen3 reranking on the preselected 100 candidates
        3. Sort by reranker score
        """
        if not retrieved_ctxs:
            self.last_rerank_stats = {
                "rerank_passages_retrieved": 0,
                "rerank_passages_scored": 0,
                "rerank_passages_kept": 0,
            }
            return []
        
        original_count = len(retrieved_ctxs)
        logger.info("Reranking %d passages...", original_count)
        
        # Stage 1: Pre-filter by retrieval score (before DeepInfra API call)
        # This reduces API cost while preserving candidates for reranker's judgment
        if pre_filter_threshold > 0:
            filtered_ctxs = [
                ctx for ctx in retrieved_ctxs 
                if ctx.get("score", 0.5) >= pre_filter_threshold
            ]
            if len(filtered_ctxs) < len(retrieved_ctxs):
                logger.info(f"   Pre-filter (retrieval score ≥{pre_filter_threshold}): {original_count} → {len(filtered_ctxs)} passages")
                retrieved_ctxs = filtered_ctxs
        
        if not retrieved_ctxs:
            logger.warning("All passages filtered out by pre-filter, returning empty")
            self.last_rerank_stats = {
                "rerank_passages_retrieved": original_count,
                "rerank_passages_scored": 0,
                "rerank_passages_kept": 0,
            }
            return []

        scored_ctxs = list(retrieved_ctxs)
        if self.rerank_eval_limit > 0:
            scored_ctxs = scored_ctxs[: self.rerank_eval_limit]

        if not scored_ctxs:
            logger.warning("No passages selected for reranker scoring, returning empty")
            self.last_rerank_stats = {
                "rerank_passages_retrieved": original_count,
                "rerank_passages_scored": 0,
                "rerank_passages_kept": 0,
            }
            return []

        # Format documents as plain text for optimal DeepInfra reranking
        # using the same chunk elements produced by ingestion.
        passages = [self._build_rerank_text(doc) for doc in scored_ctxs]
        unique_passages = list(passages)
        doc_to_score_index = list(range(len(passages)))
        if self.exact_text_dedupe_enabled and passages:
            unique_passages = []
            doc_to_score_index = []
            text_to_index: Dict[str, int] = {}
            for passage in passages:
                score_index = text_to_index.get(passage)
                if score_index is None:
                    score_index = len(unique_passages)
                    text_to_index[passage] = score_index
                    unique_passages.append(passage)
                doc_to_score_index.append(score_index)
            duplicate_count = len(passages) - len(unique_passages)
            if duplicate_count > 0:
                logger.info(
                    "   Exact rerank text dedupe: %d passages -> %d unique payloads",
                    len(passages),
                    len(unique_passages),
                )
        
        # Call reranker API directly (I/O-bound — no threading needed here).
        rerank_scores = self.reranker_engine.get_scores(
            query,
            unique_passages,
        )

        if len(rerank_scores) != len(unique_passages):
            logger.warning(
                "Reranker score count mismatch: scored=%d returned=%d; truncating to aligned length",
                len(unique_passages),
                len(rerank_scores),
            )
        aligned_unique_count = min(len(unique_passages), len(rerank_scores))

        # Attach rerank scores to the scored subset only.
        rescored_ctxs: List[Dict[str, Any]] = []
        for doc, score_index in zip(scored_ctxs, doc_to_score_index):
            if score_index >= aligned_unique_count:
                continue
            score = rerank_scores[score_index]
            focus_score = _compute_query_focus_score(doc, medical_conditions)
            focus_multiplier = _focus_multiplier_from_score(focus_score)
            doc["rerank_input_rank"] = score_index + 1
            doc["rerank_score"] = score
            doc["query_focus_score"] = focus_score
            doc["focus_multiplier"] = focus_multiplier
            doc["combined_score"] = score * focus_multiplier
            rescored_ctxs.append(doc)
        scored_ctxs = rescored_ctxs
        aligned_count = len(scored_ctxs)

        top_combined = sorted(
            (doc.get("combined_score", doc.get("rerank_score", 0.0)) for doc in scored_ctxs),
            reverse=True,
        )[:5]
        logger.info("Reranker top scores: %s", top_combined)

        sorted_ctxs = sorted(
            scored_ctxs,
            key=lambda x: x.get("combined_score", x.get("rerank_score", 0)),
            reverse=True
        )
        
        # Apply keep limit after all selected candidates have been scored.
        if self.rerank_keep_limit > 0:
            sorted_ctxs = sorted_ctxs[:self.rerank_keep_limit]
        
        top_scores = [round(d.get("combined_score", d.get("rerank_score", 0)), 3) for d in sorted_ctxs[:5]]
        logger.info(
            "Done reranking: retrieved=%d scored=%d kept=%d (top scores: %s)",
            original_count,
            aligned_count,
            len(sorted_ctxs),
            top_scores,
        )
        self.last_rerank_stats = {
            "rerank_passages_retrieved": original_count,
            "rerank_passages_scored": aligned_count,
            "rerank_passages_kept": len(sorted_ctxs),
        }
        return sorted_ctxs

    def _build_rerank_text(self, doc: Dict[str, Any]) -> str:
        """Build rerank input from ingestion-aligned text elements."""
        title = _normalize_whitespace(doc.get("title", ""))
        section_title = _normalize_whitespace(doc.get("section_title", ""))
        sentence_text = _normalize_whitespace(doc.get("sentence_text", ""))
        if sentence_text:
            header_parts: List[str] = []
            if title:
                header_parts.append(title)
            if section_title:
                header_parts.append(section_title)
            header = "\n".join(header_parts).strip()
            if header:
                return f"{header}\n\n{sentence_text}"[:4000]
            return sentence_text[:4000]

        content = self._get_document_content(doc)

        if not content:
            content = _normalize_whitespace(doc.get("abstract", "") or doc.get("full_text", ""))

        abstract = _normalize_whitespace(doc.get("abstract", ""))
        if self._is_table_section(section_title) and abstract:
            if title and abstract and abstract.lower().startswith(title.lower()):
                text = abstract
            elif title and abstract:
                text = f"{title}\n\n{abstract}"
            else:
                text = abstract or title
            return text[:RERANK_TABLE_ABSTRACT_MAX_CHARS]

        if title and content and content.lower().startswith(title.lower()):
            text = content
        elif title and content:
            text = f"{title}\n\n{content}"
        elif section_title and content and not content.lower().startswith(section_title.lower()):
            text = f"{section_title}\n\n{content}"
        else:
            text = content or title

        return text[:RERANK_TEXT_MAX_CHARS]

    @staticmethod
    def _is_table_section(section_title: str) -> bool:
        return "table" in section_title.lower()

    def _get_document_content(self, doc: Dict[str, Any]) -> str:
        """
        Return the best available chunk/document content using ingestion field order.
        """
        for key in ("sentence_text", "page_content", "text", "full_section_text", "abstract", "full_text"):
            value = doc.get(key)
            if value is not None:
                text = _normalize_whitespace(value)
                if text:
                    return text
        return ""

    @staticmethod
    def _snippet_to_sentence_entry(snippet: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "text": snippet.get("sentence_text") or snippet.get("text") or snippet.get("page_content", ""),
            "section_title": snippet.get("section_title", "abstract"),
            "char_start_offset": snippet.get("char_offset", 0),
            "sentence_index": snippet.get("sentence_index"),
            "rerank_score": snippet.get("rerank_score"),
            "boosted_score": snippet.get("boosted_score"),
            "combined_score": snippet.get("combined_score"),
            "source_unit": snippet.get("source_unit", "passage"),
        }

    @staticmethod
    def _snippet_score(snippet: Dict[str, Any]) -> float:
        return float(snippet.get("combined_score", snippet.get("rerank_score", snippet.get("score", 0))))

    @staticmethod
    def _promote_source_identity(paper: Dict[str, Any], snippet: Dict[str, Any]) -> None:
        new_source_family = str(snippet.get("source_family", "")).strip().lower()
        new_pmcid = str(snippet.get("pmcid") or snippet.get("corpus_id") or "").strip()
        if new_source_family == "pmc":
            paper["source_family"] = "pmc"
            paper["source"] = snippet.get("source", paper.get("source", ""))
        if not new_pmcid:
            return
        if new_source_family == "pmc" or new_pmcid.upper().startswith("PMC"):
            if not str(paper.get("pmcid", "")).upper().startswith("PMC"):
                paper["pmcid"] = new_pmcid
                paper["corpus_id"] = new_pmcid

    def aggregate_snippets_to_papers(
        self,
        snippets_list: List[Dict[str, Any]],
        paper_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Aggregate passages to paper level with multi-level deduplication.
        
        Deduplication priority:
        1. DOI (most reliable - same content has same DOI across journals)
        2. Normalized title (catches sister journal publications)
        3. PMCID (fallback)
        
        Uses max rerank_score for each paper.
        """
        logger.info(f"Aggregating {len(snippets_list)} passages at paper level")
        
        paper_snippets = {}
        seen_dois = {}  # doi -> corpus_id mapping
        seen_titles = {}  # normalized_title -> corpus_id mapping
        
        def normalize_title(title: str) -> str:
            """Normalize title for comparison (lowercase, remove punctuation)."""
            import re
            if not title:
                return ""
            # Lowercase, remove punctuation, collapse whitespace
            normalized = re.sub(r'[^\w\s]', '', title.lower())
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            return normalized

        def normalized_evidence_level(value: Any) -> Optional[int]:
            """Normalize evidence level to int in range 1..4, else None."""
            try:
                if value is None:
                    return None
                level = int(value)
                return level if 1 <= level <= 4 else None
            except (TypeError, ValueError):
                return None
        
        for snippet in snippets_list:
            # Get identifiers
            pmcid = snippet.get("pmcid") or snippet.get("corpus_id", "")
            doi = snippet.get("doi", "")
            title = snippet.get("title", "")
            normalized_title = normalize_title(title)
            
            if not pmcid and not doi and not normalized_title:
                continue
            
            # Determine unique key using multi-level deduplication
            # Priority: DOI > Title > PMCID
            corpus_id = pmcid
            is_duplicate = False
            duplicate_of = None
            
            # 1. Check if we've seen this DOI before
            if doi and doi in seen_dois:
                is_duplicate = True
                duplicate_of = seen_dois[doi]
                logger.debug(f"DOI duplicate: {doi} -> {duplicate_of}")
            
            # 2. Check if we've seen this title before (for sister journal publications)
            elif normalized_title and len(normalized_title) > 30 and normalized_title in seen_titles:
                is_duplicate = True
                duplicate_of = seen_titles[normalized_title]
                logger.debug(f"Title duplicate: '{title[:50]}...' -> {duplicate_of}")
            
            if is_duplicate and duplicate_of:
                # Merge with existing paper - use higher score
                if duplicate_of in paper_snippets:
                    current_score = paper_snippets[duplicate_of]["relevance_judgement"]
                    new_score = self._snippet_score(snippet)
                    paper_snippets[duplicate_of]["relevance_judgement"] = max(current_score, new_score)
                    # Add snippet to existing paper
                    paper_snippets[duplicate_of]["sentences"].append(self._snippet_to_sentence_entry(snippet))
                    self._promote_source_identity(paper_snippets[duplicate_of], snippet)

                    existing_level = normalized_evidence_level(paper_snippets[duplicate_of].get("evidence_level"))
                    new_level = normalized_evidence_level(snippet.get("evidence_level"))
                    if new_level is not None and (existing_level is None or new_level < existing_level):
                        paper_snippets[duplicate_of]["evidence_grade"] = snippet.get("evidence_grade")
                        paper_snippets[duplicate_of]["evidence_level"] = new_level
                        paper_snippets[duplicate_of]["evidence_term"] = snippet.get("evidence_term")
                        paper_snippets[duplicate_of]["evidence_source"] = snippet.get("evidence_source")
                    else:
                        if not paper_snippets[duplicate_of].get("evidence_term") and snippet.get("evidence_term"):
                            paper_snippets[duplicate_of]["evidence_term"] = snippet.get("evidence_term")
                        if not paper_snippets[duplicate_of].get("evidence_source") and snippet.get("evidence_source"):
                            paper_snippets[duplicate_of]["evidence_source"] = snippet.get("evidence_source")
                    if not str(paper_snippets[duplicate_of].get("country", "")).strip() and str(snippet.get("country", "")).strip():
                        paper_snippets[duplicate_of]["country"] = snippet.get("country")
                continue
            
            # Not a duplicate - create new entry
            if corpus_id not in paper_snippets:
                paper_snippets[corpus_id] = {
                    "corpus_id": corpus_id,
                    "pmcid": corpus_id,
                    "pmid": snippet.get("pmid"),
                    "doi": doi,
                    "title": title,
                    "abstract": snippet.get("abstract", ""),
                    "venue": snippet.get("journal") or snippet.get("venue", ""),
                    "year": snippet.get("year"),
                    "authors": snippet.get("authors", []),
                    "article_type": snippet.get("article_type", "other"),
                    "evidence_grade": snippet.get("evidence_grade"),
                    "evidence_level": normalized_evidence_level(snippet.get("evidence_level")),
                    "evidence_term": snippet.get("evidence_term"),
                    "evidence_source": snippet.get("evidence_source"),
                    "source_family": snippet.get("source_family", ""),
                    "source": snippet.get("source", ""),
                    "doc_id": snippet.get("doc_id", snippet.get("corpus_id", "")),
                    "nlm_unique_id": snippet.get("nlm_unique_id"),
                    "country": snippet.get("country"),
                    "sentences": [],
                    "relevance_judgement": -1,
                    "citation_count": snippet.get("citation_count", 0),
                }
                # Track this DOI and title for deduplication
                if doi:
                    seen_dois[doi] = corpus_id
                if normalized_title and len(normalized_title) > 30:
                    seen_titles[normalized_title] = corpus_id
            
            # Add sentence/snippet
            paper_snippets[corpus_id]["sentences"].append(self._snippet_to_sentence_entry(snippet))
            
            # Update relevance using max rerank score at paper level.
            current_score = paper_snippets[corpus_id]["relevance_judgement"]
            new_score = self._snippet_score(snippet)
            paper_snippets[corpus_id]["relevance_judgement"] = max(current_score, new_score)
            
            # Update abstract if from abstract section
            if snippet.get("section_title") == "abstract" and not paper_snippets[corpus_id]["abstract"]:
                paper_snippets[corpus_id]["abstract"] = snippet.get("text") or snippet.get("page_content", "")

        
        # Log deduplication stats
        total_snippets = len(snippets_list)
        unique_papers = len(paper_snippets)
        duplicates_removed = total_snippets - sum(len(p["sentences"]) for p in paper_snippets.values())
        if duplicates_removed > 0:
            logger.info(f"   Deduplication: {total_snippets} passages → {unique_papers} unique papers ({duplicates_removed} duplicates removed)")
        
        # Sort by relevance

        sorted_papers = sorted(
            paper_snippets.values(),
            key=lambda x: x["relevance_judgement"],
            reverse=True
        )
        
        logger.info(f"Aggregated to {len(sorted_papers)} papers")
        logger.info(f"Scores: {[p['relevance_judgement'] for p in sorted_papers[:10]]}")
        
        return sorted_papers
    
    def format_retrieval_response(
        self,
        agg_candidates: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Format into DataFrame with ScholarQA reference strings.
        
        Creates:
        - reference_string: "[corpus_id | author | year | Citations: N]"
        - relevance_judgment_input_expanded: Formatted paper content
        """
        if not agg_candidates:
            return pd.DataFrame()
        
        def format_sections_to_markdown(sentences: List[Dict]) -> str:
            """Format paper sections to markdown."""
            if not sentences:
                return ""
            
            df = pd.DataFrame(sentences)
            if df.empty:
                return ""
            
            # Sort by offset
            if "char_start_offset" in df.columns:
                df = df.sort_values(by="char_start_offset")
            
            # Group by section
            grouped = df.groupby("section_title", sort=False)["text"].apply("\n...\n".join)
            
            # Exclude abstract (already in prepend)
            grouped = grouped[~grouped.index.isin(["abstract", "title"])]
            
            return "\n\n".join(f"## {title}\n{text}" for title, text in grouped.items())
        
        df = pd.DataFrame(agg_candidates)
        
        if df.empty:
            return df
        
        # Format authors
        df["authors"] = df["authors"].fillna("").apply(
            lambda x: x if isinstance(x, list) else []
        )
        
        # Create prepend text (title, venue, authors, abstract)
        prepend_text = df.apply(
            lambda row: (
                f"# Title: {row['title']}\n"
                f"# Venue: {row['venue']}\n"
                f"# Authors: {', '.join([a.get('name', a) if isinstance(a, dict) else str(a) for a in row['authors']])}\n"
                f"## Abstract\n{row['abstract']}\n"
            ),
            axis=1
        )
        
        # Format sections
        section_text = df["sentences"].apply(format_sections_to_markdown)
        
        # Create expanded input
        df["relevance_judgment_input_expanded"] = prepend_text + section_text
        
        # Create reference string (ScholarQA format with PMCID)
        df["reference_string"] = df.apply(
            lambda row: anyascii(
                f"[{row['pmcid']} | Unknown | "
                f"{make_int(row['year'])} | Citations: {make_int(row.get('citation_count', 0))}]"
            ),
            axis=1
        )
        
        logger.info(f"Formatted {len(df)} papers into DataFrame")
        return df
    
    def aggregate_into_dataframe(
        self,
        snippets_list: List[Dict[str, Any]],
        paper_metadata: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """
        Full pipeline: aggregate then format (ScholarQA interface).
        """
        aggregated = self.aggregate_snippets_to_papers(snippets_list, paper_metadata)
        return self.format_retrieval_response(aggregated)


if __name__ == "__main__":
    print("🧪 Testing ScholarQA-style Reranker")
    print("=" * 60)
    
    sample_passages = [
        {"pmcid": "PMC001", "title": "COPD Treatment", "text": "COPD management...", "year": 2023, "journal": "Lancet", "authors": [{"name": "Smith J"}]},
        {"pmcid": "PMC002", "title": "Diabetes Care", "text": "Type 2 diabetes...", "year": 2024, "journal": "NEJM", "authors": [{"name": "Doe A"}, {"name": "Lee B"}]},
    ]
    
    finder = PaperFinderWithReranker()
    reranked = finder.rerank("COPD management", sample_passages)
    print(f"Reranked {len(reranked)} passages")
    
    df = finder.aggregate_into_dataframe(reranked)
    print(f"DataFrame: {len(df)} rows")
    print(f"Reference strings: {df['reference_string'].tolist()}")
