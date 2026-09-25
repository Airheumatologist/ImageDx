"""
Configuration module for Medical RAG Pipeline.
Loads environment variables and provides central settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: str = "") -> list[str]:
    value = os.getenv(name, default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# =============================================================================
# API / CORS Configuration
# =============================================================================
CORS_ALLOWED_ORIGINS = _env_csv(
    "CORS_ALLOWED_ORIGINS",
    default="http://localhost:3000,http://127.0.0.1:3000",
)

# Runtime API controls
API_SHUTDOWN_GRACE_SECONDS = _env_int("API_SHUTDOWN_GRACE_SECONDS", 85)
API_INFLIGHT_DRAIN_POLL_SECONDS = _env_float("API_INFLIGHT_DRAIN_POLL_SECONDS", 0.2)
API_MAX_INFLIGHT_REQUESTS = _env_int("API_MAX_INFLIGHT_REQUESTS", 0)
API_INFLIGHT_ACQUIRE_TIMEOUT_MS = _env_int("API_INFLIGHT_ACQUIRE_TIMEOUT_MS", 250)
API_EXECUTOR_MAX_WORKERS = max(
    1,
    _env_int(
        "API_EXECUTOR_MAX_WORKERS",
        min(32, (os.cpu_count() or 1) + 4),
    ),
)
UPSTREAM_HTTP_MAX_CONNECTIONS = max(1, _env_int("UPSTREAM_HTTP_MAX_CONNECTIONS", 200))
UPSTREAM_HTTP_MAX_KEEPALIVE = max(1, _env_int("UPSTREAM_HTTP_MAX_KEEPALIVE", 100))
UPSTREAM_HTTP_KEEPALIVE_EXPIRY = max(1.0, _env_float("UPSTREAM_HTTP_KEEPALIVE_EXPIRY", 30.0))

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_pipeline")
GRAPHRAG_COLLECTION_NAME = os.getenv("GRAPHRAG_COLLECTION_NAME", "pmc_medical_graphrag")

# =============================================================================
# API Provider Configuration
# =============================================================================
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
DEEPINFRA_BASE_URL = os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai")
DEEPINFRA_RETRY_COUNT = int(os.getenv("DEEPINFRA_RETRY_COUNT", "3"))  # Number of retries for transient DeepInfra failures
DEEPINFRA_RETRY_DELAY = float(os.getenv("DEEPINFRA_RETRY_DELAY", "1.0"))  # Base delay in seconds (exponential backoff)
DEEPINFRA_CHAT_TIMEOUT_SECONDS = _env_float("DEEPINFRA_CHAT_TIMEOUT_SECONDS", 300.0)
DEEPINFRA_EMBED_TIMEOUT_SECONDS = _env_float("DEEPINFRA_EMBED_TIMEOUT_SECONDS", 120.0)
DEEPINFRA_RERANK_TIMEOUT_SECONDS = _env_float("DEEPINFRA_RERANK_TIMEOUT_SECONDS", 60.0)
OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY")
OPENCODE_BASE_URL = os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/zen/v1")
OPENCODE_RETRY_COUNT = _env_int("OPENCODE_RETRY_COUNT", 3)
OPENCODE_RETRY_DELAY = _env_float("OPENCODE_RETRY_DELAY", 1.0)
OPENCODE_CHAT_TIMEOUT_SECONDS = _env_float("OPENCODE_CHAT_TIMEOUT_SECONDS", 300.0)

# LLM Generation Parameters
LLM_TEMPERATURE = 0.7  # Controls randomness (0=deterministic, 1=creative)
LLM_TOP_P = 0.9  # Nucleus sampling threshold

# LLM Configuration (Generation)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "opencode").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "space-bunny-free")
QUERY_PREPROCESSOR_LLM_MODEL = os.getenv(
    "QUERY_PREPROCESSOR_LLM_MODEL",
    "space-bunny-free",
)
LLM_MAX_COMPLETION_TOKENS = _env_int("LLM_MAX_COMPLETION_TOKENS", 8192)
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "").strip().lower()
if LLM_PROVIDER == "opencode":
    LLM_RETRY_COUNT = OPENCODE_RETRY_COUNT
    LLM_RETRY_DELAY = OPENCODE_RETRY_DELAY
    LLM_CHAT_TIMEOUT_SECONDS = OPENCODE_CHAT_TIMEOUT_SECONDS
else:
    LLM_RETRY_COUNT = DEEPINFRA_RETRY_COUNT
    LLM_RETRY_DELAY = DEEPINFRA_RETRY_DELAY
    LLM_CHAT_TIMEOUT_SECONDS = DEEPINFRA_CHAT_TIMEOUT_SECONDS

# =============================================================================
# Embedding Model Configuration
# =============================================================================
# Provider: "hf_inference_endpoint" | "deepinfra"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "deepinfra").strip().lower()
EMBEDDING_MODEL = os.getenv(
    "RUNTIME_EMBEDDING_MODEL",
    os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
)
EMBEDDING_DIMENSION = 1024  # Qwen/Qwen3-Embedding-0.6B output dimension
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

# HF Self-Hosted Inference Endpoint (TEI) — used when EMBEDDING_PROVIDER=hf_inference_endpoint
# Endpoint exposes an OpenAI-compatible /v1/embeddings route.
HF_INFERENCE_ENDPOINT_URL = os.getenv(
    "HF_INFERENCE_ENDPOINT_URL",
    "https://jp5vxer2scm0dmve.us-east-1.aws.endpoints.huggingface.cloud",
)
HF_INFERENCE_ENDPOINT_API_KEY = os.getenv("HF_INFERENCE_ENDPOINT_API_KEY")
HF_INFERENCE_EMBED_TIMEOUT_SECONDS = _env_float("HF_INFERENCE_EMBED_TIMEOUT_SECONDS", 60.0)

# =============================================================================
# Chunking Configuration (CRITICAL: Must match .env values)
# =============================================================================
# Optimized for Qwen3-Embedding-0.6B (32k context window)
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "2048"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "256"))

# Quantization Configuration
# =============================================================================
QUANTIZATION_TYPE = os.getenv("QUANTIZATION_TYPE", "scalar").strip().lower()
SCALAR_QUANTILE = float(os.getenv("SCALAR_QUANTILE", "0.99"))
QUANTIZATION_ALWAYS_RAM = _env_bool("QUANTIZATION_ALWAYS_RAM", default=True)
# Search rescore improves accuracy with quantized vectors
QUANTIZATION_RESCORE = _env_bool("QUANTIZATION_RESCORE", default=False)
QUANTIZATION_OVERSAMPLING = float(os.getenv("QUANTIZATION_OVERSAMPLING", "1.0"))

# Hybrid Search Configuration
USE_HYBRID_SEARCH = _env_bool("USE_HYBRID_SEARCH", default=True)
DENSE_WEIGHT = 0.7  # Weight for dense vector scores in hybrid search
SPARSE_WEIGHT = 0.3  # Weight for sparse vector scores in hybrid search
SPARSE_RETRIEVAL_MODE = os.getenv("SPARSE_RETRIEVAL_MODE", "bm25").strip().lower()
SPARSE_MAX_TERMS_QUERY = int(os.getenv("SPARSE_MAX_TERMS_QUERY", "64"))
SPARSE_MIN_TOKEN_LEN = int(os.getenv("SPARSE_MIN_TOKEN_LEN", "2"))
SPARSE_REMOVE_STOPWORDS = _env_bool("SPARSE_REMOVE_STOPWORDS", default=True)

# =============================================================================
# Search Configuration
# =============================================================================
TOP_K_RESULTS = 5  # Number of documents to retrieve
SCORE_THRESHOLD = 0.25  # Lowered from 0.3 to capture more relevant papers
# Lower threshold helps retrieve more candidates before aggressive filtering
ENTITY_FILTER_ENABLED = _env_bool("ENTITY_FILTER_ENABLED", default=False)

# Chunk retrieval/reranking profile for chunk-level indexing
# Fixed non-DailyMed retrieval profile: raw PMC/PubMed buckets flow directly into reranking.
PMC_DENSE_RETRIEVAL_LIMIT = _env_int("PMC_DENSE_RETRIEVAL_LIMIT", 35)
PMC_BM25_RETRIEVAL_LIMIT = _env_int("PMC_BM25_RETRIEVAL_LIMIT", 25)
PUBMED_DENSE_RETRIEVAL_LIMIT = _env_int("PUBMED_DENSE_RETRIEVAL_LIMIT", 35)
PUBMED_BM25_RETRIEVAL_LIMIT = _env_int("PUBMED_BM25_RETRIEVAL_LIMIT", 25)
RETRIEVAL_CHUNK_LIMIT = (
    PMC_DENSE_RETRIEVAL_LIMIT
    + PMC_BM25_RETRIEVAL_LIMIT
    + PUBMED_DENSE_RETRIEVAL_LIMIT
    + PUBMED_BM25_RETRIEVAL_LIMIT
)
RETRIEVAL_FUSED_LIMIT = RETRIEVAL_CHUNK_LIMIT
# Deprecated for PMC/PubMed: pre-rerank slicing was removed in favor of reranking all retrieved passages.
MAX_CHUNKS_PER_ARTICLE_PRE_RERANK = int(os.getenv("MAX_CHUNKS_PER_ARTICLE_PRE_RERANK", "2"))
RERANK_INPUT_CHUNK_LIMIT = int(os.getenv("RERANK_INPUT_CHUNK_LIMIT", "100"))
PRE_RERANK_BOOSTED_RATIO = _env_float("PRE_RERANK_BOOSTED_RATIO", 0.85)
RERANK_EVAL_LIMIT = _env_int("RERANK_EVAL_LIMIT", 100)
RERANK_KEEP_LIMIT = _env_int("RERANK_KEEP_LIMIT", _env_int("RERANK_TOP_CHUNKS", 100))
RERANK_TOP_CHUNKS = RERANK_KEEP_LIMIT
PMC_PUBMED_ARTICLE_RELEVANCE_THRESHOLD = _env_float("PMC_PUBMED_ARTICLE_RELEVANCE_THRESHOLD", 0.8)
PMC_PUBMED_FINAL_TOP_ARTICLES = _env_int("PMC_PUBMED_FINAL_TOP_ARTICLES", 30)
PMC_PUBMED_MIN_FINAL_ARTICLES = _env_int("PMC_PUBMED_MIN_FINAL_ARTICLES", 8)
FINAL_TOP_ARTICLES = PMC_PUBMED_FINAL_TOP_ARTICLES
FINAL_RECENCY_POLICY_MODE = os.getenv("FINAL_RECENCY_POLICY_MODE", "hybrid").strip().lower()
FINAL_RECENCY_WINDOW_YEARS = int(os.getenv("FINAL_RECENCY_WINDOW_YEARS", "5"))
FINAL_RECENCY_BACKFILL_MAX_EVIDENCE_LEVEL = int(os.getenv("FINAL_RECENCY_BACKFILL_MAX_EVIDENCE_LEVEL", "2"))
FINAL_RECENCY_EXCLUDE_UNKNOWN_NON_DAILYMED = _env_bool("FINAL_RECENCY_EXCLUDE_UNKNOWN_NON_DAILYMED", default=True)
PMC_FULLTEXT_RECENT_ONLY = _env_bool("PMC_FULLTEXT_RECENT_ONLY", default=True)
RETRIEVAL_RECENCY_BOOST_ENABLED = _env_bool("RETRIEVAL_RECENCY_BOOST_ENABLED", default=True)
RETRIEVAL_RECENCY_APPLY_WITH_YEAR_FILTER = _env_bool("RETRIEVAL_RECENCY_APPLY_WITH_YEAR_FILTER", default=True)
RETRIEVAL_RECENCY_Y1_MULT = float(os.getenv("RETRIEVAL_RECENCY_Y1_MULT", "1.20"))
RETRIEVAL_RECENCY_Y3_MULT = float(os.getenv("RETRIEVAL_RECENCY_Y3_MULT", "1.14"))
RETRIEVAL_RECENCY_Y5_MULT = float(os.getenv("RETRIEVAL_RECENCY_Y5_MULT", "1.08"))
RETRIEVAL_RECENCY_Y7_MULT = float(os.getenv("RETRIEVAL_RECENCY_Y7_MULT", "1.03"))
PRE_RERANK_RECENT_WINDOW_YEARS = int(os.getenv("PRE_RERANK_RECENT_WINDOW_YEARS", "7"))
# Deprecated: pre-rerank selection now uses metadata-priority slices instead of a recent quota.
PRE_RERANK_RECENT_QUOTA_RATIO = float(os.getenv("PRE_RERANK_RECENT_QUOTA_RATIO", "0.35"))
RETRIEVAL_SOURCE_FANOUT_ENABLED = _env_bool("RETRIEVAL_SOURCE_FANOUT_ENABLED", default=False)
RETRIEVAL_SOURCE_FANOUT_MODE = os.getenv("RETRIEVAL_SOURCE_FANOUT_MODE", "parallel").strip().lower()
RETRIEVAL_SOURCE_FANOUT_MIN_RESULTS = _env_int("RETRIEVAL_SOURCE_FANOUT_MIN_RESULTS", 60)
RETRIEVAL_SOURCE_FANOUT_FALLBACK_BROAD = _env_bool(
    "RETRIEVAL_SOURCE_FANOUT_FALLBACK_BROAD",
    default=False,
)
RETRIEVAL_BACKEND = os.getenv("RETRIEVAL_BACKEND", "turbopuffer").strip().lower()
RETRIEVAL_PREFILTER = _env_bool("RETRIEVAL_PREFILTER", default=True)
TURBOPUFFER_API_KEY = os.getenv("TURBOPUFFER_API_KEY", "")
TURBOPUFFER_REGION = os.getenv("TURBOPUFFER_REGION", "gcp-us-central1").strip()
TURBOPUFFER_NAMESPACE_PMC = os.getenv("TURBOPUFFER_NAMESPACE_PMC", "medical_database_pmc")
TURBOPUFFER_NAMESPACE_PUBMED = os.getenv("TURBOPUFFER_NAMESPACE_PUBMED", "medical_database_pubmed")
TURBOPUFFER_NAMESPACE_DAILYMED = os.getenv("TURBOPUFFER_NAMESPACE_DAILYMED", "medical_database_dailymed")
TURBOPUFFER_TIMEOUT_SECONDS = _env_int("TURBOPUFFER_TIMEOUT_SECONDS", 30)
RETRIEVAL_RRF_K = _env_int("RETRIEVAL_RRF_K", 60)
RETRIEVAL_DENSE_WEIGHT = _env_float("RETRIEVAL_DENSE_WEIGHT", 0.7)
RETRIEVAL_SPARSE_WEIGHT = _env_float("RETRIEVAL_SPARSE_WEIGHT", 0.3)
TURBOPUFFER_INCLUDE_ALL_ATTRIBUTES = _env_bool("TURBOPUFFER_INCLUDE_ALL_ATTRIBUTES", default=False)

# =============================================================================
# Reranker Configuration
# =============================================================================
# Provider: "deepinfra" (default) or "cross-encoder" (self-hosted)
RERANKER_PROVIDER = "deepinfra"  # Fixed: Only DeepInfra supported
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")
RERANK_TEXT_MAX_CHARS = _env_int("RERANK_TEXT_MAX_CHARS", 4000)
RERANK_TABLE_ABSTRACT_MAX_CHARS = _env_int("RERANK_TABLE_ABSTRACT_MAX_CHARS", 1200)

# =============================================================================
# Reranker v2 — Configurable Scoring Constants
# =============================================================================
# Set RERANKER_V2_ENABLED=0 in env to roll back to original scoring behaviour.
RERANKER_V2_ENABLED   = _env_bool("RERANKER_V2_ENABLED", default=True)
# Evidence tier multipliers (v2 defaults — less aggressive than original 3.0/1.5/1.0/0.2)
TIER_1_BOOST          = float(os.getenv("TIER_1_BOOST",   "2.00"))   # guidelines
TIER_2_BOOST          = float(os.getenv("TIER_2_BOOST",   "1.25"))   # RCTs/reviews
TIER_3_BOOST          = float(os.getenv("TIER_3_BOOST",   "1.00"))   # standard research
TIER_4_PENALTY        = float(os.getenv("TIER_4_PENALTY", "0.40"))   # case reports
# Combined score weights (v2: reranker 85%, entity 15%; legacy: 70%/30%)
RERANKER_SCORE_WEIGHT = float(os.getenv("RERANKER_SCORE_WEIGHT", "0.85"))
ENTITY_SCORE_WEIGHT   = float(os.getenv("ENTITY_SCORE_WEIGHT",   "0.15"))
RERANK_COUNTRY_BOOST_ENABLED = _env_bool("RERANK_COUNTRY_BOOST_ENABLED", default=True)
RERANK_COUNTRY_BOOST_MULTIPLIER = _env_float("RERANK_COUNTRY_BOOST_MULTIPLIER", 1.05)
RERANK_COUNTRY_BOOST_POLICY = os.getenv("RERANK_COUNTRY_BOOST_POLICY", "us_eu27").strip().lower()

# =============================================================================
# Query Preprocessing Configuration
# =============================================================================
QUERY_EXPANSION_COUNT = int(os.getenv("QUERY_EXPANSION_COUNT", "2"))  # Number of expanded query variations
RETRIEVAL_QUERY_VARIANT_LIMIT = _env_int("RETRIEVAL_QUERY_VARIANT_LIMIT", 4)
RETRIEVAL_QUERY_FANOUT_LIMIT = _env_int("RETRIEVAL_QUERY_FANOUT_LIMIT", 400)
# Bulk retrieval limits (adjusted for larger 2048-token chunks)
BULK_RETRIEVAL_LIMIT = 200  # Balanced: enough diversity without excessive payload overhead
BULK_RETRIEVAL_PER_QUERY = 100  # Reduced from 150 - more efficient with larger chunks
RERANK_TOP_K = FINAL_TOP_ARTICLES  # Final literature articles after paper-level aggregation
MAX_ABSTRACTS = FINAL_TOP_ARTICLES  # Literature article cap before DailyMed append
MAX_DAILYMED_PER_DRUG = _env_int("MAX_DAILYMED_PER_DRUG", 1)  # Max DailyMed entries per normalized drug concept
DAILYMED_MAX_RETRIEVAL_RESULTS = _env_int("DAILYMED_MAX_RETRIEVAL_RESULTS", 10)
DAILYMED_MAX_FINAL_REFERENCES = _env_int("DAILYMED_MAX_FINAL_REFERENCES", 10)
DAILYMED_MULTI_QUERY_MAX_SUBQUERIES = _env_int("DAILYMED_MULTI_QUERY_MAX_SUBQUERIES", 16)
DAILYMED_MULTI_QUERY_MAX_CONCURRENCY = _env_int("DAILYMED_MULTI_QUERY_MAX_CONCURRENCY", 4)
DAILYMED_MULTI_QUERY_TOP_K = _env_int("DAILYMED_MULTI_QUERY_TOP_K", 50)

# =============================================================================
# Weekly Update QoS Configuration
# =============================================================================
WEEKLY_UPDATE_THROTTLE_SECONDS = _env_float("WEEKLY_UPDATE_THROTTLE_SECONDS", 0.5)
WEEKLY_UPDATE_BATCH_SIZE = _env_int("WEEKLY_UPDATE_BATCH_SIZE", 0)

# =============================================================================
# Query Caching Configuration
# =============================================================================
QUERY_CACHE_ENABLED = _env_bool("QUERY_CACHE_ENABLED", default=True)
QUERY_CACHE_DIR = os.getenv("QUERY_CACHE_DIR", "data/cache")
QUERY_CACHE_EXPIRY_DAYS = int(os.getenv("QUERY_CACHE_EXPIRY_DAYS", "30"))
QUERY_CACHE_NAMESPACE = os.getenv("QUERY_CACHE_NAMESPACE", "default")
QUERY_CACHE_KEY_VERSION = int(os.getenv("QUERY_CACHE_KEY_VERSION", "2"))
EMBEDDING_SUBCACHE_ENABLED = _env_bool("EMBEDDING_SUBCACHE_ENABLED", default=True)
RETRIEVAL_SUBCACHE_ENABLED = _env_bool("RETRIEVAL_SUBCACHE_ENABLED", default=True)
RERANK_EXACT_TEXT_DEDUPE_ENABLED = _env_bool("RERANK_EXACT_TEXT_DEDUPE_ENABLED", default=True)

# =============================================================================
# Validation
# =============================================================================
def validate_config():
    """Validate that all required environment variables are set."""
    errors = []

    if RETRIEVAL_SOURCE_FANOUT_MODE not in {"parallel"}:
        errors.append("RETRIEVAL_SOURCE_FANOUT_MODE must be: parallel")
    if RETRIEVAL_BACKEND not in {"turbopuffer"}:
        errors.append("RETRIEVAL_BACKEND must be: turbopuffer")
    if not TURBOPUFFER_REGION:
        errors.append("TURBOPUFFER_REGION not set")
    # Only require DeepInfra key when it is actually used for embeddings or LLM
    if EMBEDDING_PROVIDER == "deepinfra" and not DEEPINFRA_API_KEY:
        errors.append("DEEPINFRA_API_KEY not set (required when EMBEDDING_PROVIDER=deepinfra)")
    if EMBEDDING_PROVIDER not in {"deepinfra", "hf_inference_endpoint"}:
        errors.append(f"EMBEDDING_PROVIDER '{EMBEDDING_PROVIDER}' is not supported")
    if EMBEDDING_PROVIDER == "hf_inference_endpoint" and not HF_INFERENCE_ENDPOINT_API_KEY:
        errors.append("HF_INFERENCE_ENDPOINT_API_KEY not set (required when EMBEDDING_PROVIDER=hf_inference_endpoint)")
    if LLM_PROVIDER not in {"deepinfra", "opencode"}:
        errors.append("LLM_PROVIDER must be one of: deepinfra, opencode")
    if LLM_PROVIDER == "deepinfra" and not DEEPINFRA_API_KEY:
        errors.append("DEEPINFRA_API_KEY not set (required when LLM_PROVIDER=deepinfra)")
    if LLM_PROVIDER == "opencode" and not OPENCODE_API_KEY:
        errors.append("OPENCODE_API_KEY not set (required when LLM_PROVIDER=opencode)")
    if not (1.0 <= float(RERANK_COUNTRY_BOOST_MULTIPLIER) <= 1.2):
        errors.append("RERANK_COUNTRY_BOOST_MULTIPLIER must be between 1.0 and 1.2")
    if RERANK_COUNTRY_BOOST_POLICY not in {"us_eu27"}:
        errors.append("RERANK_COUNTRY_BOOST_POLICY must be: us_eu27")

    if errors:
        raise ValueError(
            "Missing configuration:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return True


if __name__ == "__main__":
    # Test configuration loading
    print("🔧 Configuration Check")
    print("=" * 50)
    try:
        validate_config()
        llm_key = OPENCODE_API_KEY if LLM_PROVIDER == "opencode" else DEEPINFRA_API_KEY
        llm_base_url = OPENCODE_BASE_URL if LLM_PROVIDER == "opencode" else DEEPINFRA_BASE_URL
        llm_key_display = f"{llm_key[:20]}..." if llm_key else "<missing>"
        print(f"✅ LLM API key: {llm_key_display}")
        print(f"✅ Final LLM Model: {LLM_MODEL}")
        print(f"✅ Query Preprocessor LLM Model: {QUERY_PREPROCESSOR_LLM_MODEL}")
        print(f"✅ LLM Provider: {LLM_PROVIDER}")
        print(f"✅ LLM Base URL: {llm_base_url}")
        print(f"✅ Embedding Model: {EMBEDDING_MODEL}")
        print(f"✅ Collection Name: {COLLECTION_NAME}")
        print("\n✅ All configuration validated!")
        
        # Show search configuration
        print("\n🎯 SEARCH CONFIGURATION:")
        print(f"   - BULK_RETRIEVAL_LIMIT: {BULK_RETRIEVAL_LIMIT}")
        print(f"   - RERANK_TOP_K: {RERANK_TOP_K}")
        print(f"   - SCORE_THRESHOLD: {SCORE_THRESHOLD}")
        
    except ValueError as e:
        print(f"❌ Configuration error:\n{e}")
