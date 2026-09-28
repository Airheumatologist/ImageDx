"""Configuration for the Visual Findings Library pilot.

Provider settings (OpenRouter, DeepInfra embeddings, turbopuffer) are read
from the environment;
the repo-root ``.env`` is loaded at import. Pilot-specific ``VP_*`` settings
are read here with their spec defaults (docs/visual_pilot_plan.md §3).

Data paths are resolved lazily through functions (not module constants) so
tests can override ``VP_DATA_DIR`` with env vars / monkeypatch.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")


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


# -----------------------------------------------------------------------------
# Provider settings: turbopuffer PMC namespace for retrieval, DeepInfra for
# query embeddings only, and OpenRouter for all LLM stages (P1-P4).
# -----------------------------------------------------------------------------
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
DEEPINFRA_BASE_URL = os.getenv(
    "DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai"
)
OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY")
OPENCODE_BASE_URL = os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/zen/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "").strip()
TURBOPUFFER_API_KEY = os.getenv("TURBOPUFFER_API_KEY", "")
TURBOPUFFER_REGION = os.getenv("TURBOPUFFER_REGION", "gcp-us-central1").strip()
TURBOPUFFER_NAMESPACE_PMC = os.getenv(
    "TURBOPUFFER_NAMESPACE_PMC", "medical_database_pmc"
)
TURBOPUFFER_TIMEOUT_SECONDS = _env_int("TURBOPUFFER_TIMEOUT_SECONDS", 30)
EMBEDDING_MODEL = os.getenv(
    "RUNTIME_EMBEDDING_MODEL",
    os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
)
EMBEDDING_TIMEOUT_SECONDS = _env_int("DEEPINFRA_EMBED_TIMEOUT_SECONDS", 120)

# Credentials for each supported VP_LLM_PROVIDER value.
_LLM_PROVIDER_CREDENTIALS = {
    "opencode": lambda: (OPENCODE_API_KEY, OPENCODE_BASE_URL),
    # DeepInfra is primarily the embeddings/reranking provider, but the same
    # OpenAI-compatible endpoint serves chat models (used for parity runs while
    # space-bunny-free's upstream rejects union-type json_schema).
    "deepinfra": lambda: (DEEPINFRA_API_KEY, DEEPINFRA_BASE_URL),
    "openrouter": lambda: (OPENROUTER_API_KEY, OPENROUTER_BASE_URL),
}
LLM_PROVIDERS = frozenset(_LLM_PROVIDER_CREDENTIALS)


def llm_credentials(provider: str | None = None) -> tuple[str | None, str]:
    """(api_key, base_url) for the given provider (default VP_LLM_PROVIDER)."""
    name = (provider or VP_LLM_PROVIDER).strip().lower()
    if name not in _LLM_PROVIDER_CREDENTIALS:
        raise ValueError(f"VP_LLM_PROVIDER must be one of {sorted(LLM_PROVIDERS)}")
    return _LLM_PROVIDER_CREDENTIALS[name]()

# -----------------------------------------------------------------------------
# Visual pilot settings (VP_*), defaults per spec §3 / repo adjustments box.
# Every LLM stage runs on OpenRouter; stealth/space-bunny-alpha is multimodal, so
# the same model covers caption triage, extraction, and the vision judge.
# -----------------------------------------------------------------------------
VP_LLM_PROVIDER = os.getenv("VP_LLM_PROVIDER", "openrouter").strip().lower()
VP_TRIAGE_MODEL = os.getenv("VP_TRIAGE_MODEL", "stealth/space-bunny-alpha")
VP_EXTRACT_MODEL = os.getenv("VP_EXTRACT_MODEL", "stealth/space-bunny-alpha")
VP_JUDGE_MODEL = os.getenv("VP_JUDGE_MODEL", "stealth/space-bunny-alpha")
VP_IMAGE_MAX_EDGE = _env_int("VP_IMAGE_MAX_EDGE", 1568)
# Stage-5 storage encoding. Panels are written lossy (WebP q90 by default)
# capped at VP_PANEL_MAX_EDGE; the stored "original" is a display copy capped
# at VP_ORIGINAL_MAX_EDGE (raw bytes stay refetchable from the PMC S3 bundle).
# VP_PANEL_FORMAT=png keeps lossless archival crops at much larger file sizes.
VP_PANEL_FORMAT = os.getenv("VP_PANEL_FORMAT", "webp").strip().lower()
VP_PANEL_QUALITY = max(1, min(100, _env_int("VP_PANEL_QUALITY", 90)))
VP_PANEL_MAX_EDGE = max(0, _env_int("VP_PANEL_MAX_EDGE", 2048))
VP_ORIGINAL_MAX_EDGE = max(0, _env_int("VP_ORIGINAL_MAX_EDGE", 2048))
VP_CONCURRENCY = max(1, _env_int("VP_CONCURRENCY", 4))
VP_LLM_TIMEOUT_SECONDS = max(1, _env_int("VP_LLM_TIMEOUT_SECONDS", 300))
# Figures per P2 caption-triage batch.
VP_TRIAGE_BATCH = max(1, _env_int("VP_TRIAGE_BATCH", 40))
# Maximum per-disease visual finding/modality passage queries in stage 2.
VP_VISUAL_QUERY_CAP = max(0, _env_int("VP_VISUAL_QUERY_CAP", 12))

# -----------------------------------------------------------------------------
# Throughput-plan keys (docs/visual_pilot_plan.md §4 contract C1). Defaults
# preserve current behavior; later workstreams switch each of these on.
# -----------------------------------------------------------------------------
# Request rate for the public S3 bucket (pmc-oa-opendata); NCBI hosts unchanged.
VP_S3_RPS = max(0.1, _env_float("VP_S3_RPS", 20.0))
# Worker pool size for fetches/parse/store (W5/W6/W7).
VP_FETCH_CONCURRENCY = max(1, _env_int("VP_FETCH_CONCURRENCY", 8))
# Max in-flight P3 vision-judge calls (W6; raise only after the W11 probe).
VP_JUDGE_CONCURRENCY = max(1, _env_int("VP_JUDGE_CONCURRENCY", 4))
# Max in-flight P1 relevance calls (W4); defaults to VP_CONCURRENCY.
VP_P1_CONCURRENCY = max(1, _env_int("VP_P1_CONCURRENCY", VP_CONCURRENCY))
# Per-request LLM timeout for the P3 judge (W6).
VP_JUDGE_TIMEOUT_SECONDS = max(1, _env_int("VP_JUDGE_TIMEOUT_SECONDS", 120))
# Retries on HTTP 429 for provider calls (W3).
VP_RATE_LIMIT_RETRIES = max(0, _env_int("VP_RATE_LIMIT_RETRIES", 4))
# In-memory cap for the judge->store original-bytes handoff (W6/W7).
VP_ORIGINALS_CACHE_MB = max(0, _env_int("VP_ORIGINALS_CACHE_MB", 512))
# W0 timing instrumentation: 1 records timings and writes reports/timings_*.json.
VP_TIMINGS = _env_int("VP_TIMINGS", 1)
# W0 parity harness: 1 makes an llm_calls cache miss raise LLMError instead of
# calling the provider (used with seeded candidate runs; default off).
VP_LLM_CACHE_ONLY = _env_int("VP_LLM_CACHE_ONLY", 0)

# Per-model USD per 1M tokens (input/output). stealth/space-bunny-alpha is
# free on OpenRouter; override or extend at runtime with
# VP_MODEL_PRICES_JSON='{"model": {"in": x, "out": y}}'.
MODEL_PRICES = {
    "space-bunny-free": {"in": 0.0, "out": 0.0},
    "stealth/space-bunny-alpha": {"in": 0.0, "out": 0.0},
}


def model_price(model: str) -> tuple[float, float] | None:
    """(input, output) USD per 1M tokens, or None if the model is unpriced."""
    prices = dict(MODEL_PRICES)
    override = os.getenv("VP_MODEL_PRICES_JSON")
    if override:
        try:
            for key, value in json.loads(override).items():
                prices[key] = value
        except (ValueError, AttributeError):
            pass
    entry = prices.get(model)
    if entry is None:
        return None
    return float(entry["in"]), float(entry["out"])

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "visual_pilot"
DB_FILENAME = "visual_pilot.sqlite"


# -----------------------------------------------------------------------------
# Data paths — resolved lazily so tests can override VP_DATA_DIR
# -----------------------------------------------------------------------------
def data_dir() -> Path:
    return Path(os.getenv("VP_DATA_DIR", str(DEFAULT_DATA_DIR))).expanduser()


def db_path() -> Path:
    return data_dir() / DB_FILENAME


def reports_dir() -> Path:
    return data_dir() / "reports"
