"""Configuration for the Visual Findings Library pilot.

OpenCode provider settings are read from the environment;
the repo-root ``.env`` is loaded at import. Pilot-specific ``VP_*`` settings
are read here with their defaults.

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
# Provider settings: OpenCode runs all LLM stages (P2-P6).
# Article discovery uses the public Europe PMC REST API (no key).
# -----------------------------------------------------------------------------
OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY")
# OpenCode Go gateway (OpenAI-compatible chat completions). The Zen endpoint
# (https://opencode.ai/zen/v1) refuses free-tier models outside the OpenCode app.
OPENCODE_BASE_URL = os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/zen/go/v1")
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "").strip()

# Credentials for each supported VP_LLM_PROVIDER value.
_LLM_PROVIDER_CREDENTIALS = {
    "opencode": lambda: (OPENCODE_API_KEY, OPENCODE_BASE_URL),
}
LLM_PROVIDERS = frozenset(_LLM_PROVIDER_CREDENTIALS)


def llm_credentials(provider: str | None = None) -> tuple[str | None, str]:
    """(api_key, base_url) for the given provider (default VP_LLM_PROVIDER)."""
    name = (provider or VP_LLM_PROVIDER).strip().lower()
    if name not in _LLM_PROVIDER_CREDENTIALS:
        raise ValueError(f"VP_LLM_PROVIDER must be one of {sorted(LLM_PROVIDERS)}")
    return _LLM_PROVIDER_CREDENTIALS[name]()

# -----------------------------------------------------------------------------
# Visual pilot settings (VP_*).
# Every LLM stage runs on OpenCode Go's multimodal muse-spark-1.3-contributor
# (Meta may train on its prompts and completions in exchange for the discount).
# -----------------------------------------------------------------------------
DEFAULT_MODEL = "muse-spark-1.3-contributor"
# Models OpenCode serves only through the Responses API (/responses).
RESPONSES_API_MODELS = frozenset(
    m.strip() for m in os.getenv(
        "VP_RESPONSES_API_MODELS",
        "muse-spark-1.2-contributor,muse-spark-1.3-contributor",
    ).split(",") if m.strip()
)


def uses_responses_api(model: str) -> bool:
    return model in RESPONSES_API_MODELS


VP_LLM_PROVIDER = os.getenv("VP_LLM_PROVIDER", "opencode").strip().lower()
VP_TRIAGE_MODEL = os.getenv("VP_TRIAGE_MODEL", DEFAULT_MODEL)
VP_EXTRACT_MODEL = os.getenv("VP_EXTRACT_MODEL", DEFAULT_MODEL)
VP_JUDGE_MODEL = os.getenv("VP_JUDGE_MODEL", DEFAULT_MODEL)
VP_DESCRIBE_MODEL = os.getenv("VP_DESCRIBE_MODEL", VP_EXTRACT_MODEL)
# Topic findings vocabulary (build-vocab, P6); defaults to VP_EXTRACT_MODEL.
VP_VOCAB_MODEL = os.getenv("VP_VOCAB_MODEL", VP_EXTRACT_MODEL)
VP_IMAGE_MAX_EDGE = _env_int("VP_IMAGE_MAX_EDGE", 1568)
VP_CONCURRENCY = max(1, _env_int("VP_CONCURRENCY", 16))
VP_LLM_TIMEOUT_SECONDS = max(1, _env_int("VP_LLM_TIMEOUT_SECONDS", 300))
# Wall-clock cap on one LLM request. The timeouts above are per read, which
# provider keep-alive bytes reset forever on a stalled request.
VP_LLM_MAX_REQUEST_SECONDS = max(1, _env_int("VP_LLM_MAX_REQUEST_SECONDS", 600))
# Figures per P2 caption-triage batch. Smaller batches finish sooner (each
# figure adds reasoning output) and spread across the VP_CONCURRENCY slots.
VP_TRIAGE_BATCH = max(1, _env_int("VP_TRIAGE_BATCH", 20))
# Per-stage reasoning effort (low/medium/high): ``reasoning.effort`` on the
# Responses API, ``reasoning_effort`` on chat completions. Unset
# falls back to LLM_REASONING_EFFORT; empty means the model default. A set
# effort is part of the llm_calls cache key.
VP_TRIAGE_REASONING_EFFORT = os.getenv("VP_TRIAGE_REASONING_EFFORT", LLM_REASONING_EFFORT).strip()
VP_JUDGE_REASONING_EFFORT = os.getenv("VP_JUDGE_REASONING_EFFORT", LLM_REASONING_EFFORT).strip()
VP_DESCRIBE_REASONING_EFFORT = os.getenv("VP_DESCRIBE_REASONING_EFFORT", LLM_REASONING_EFFORT).strip()
VP_EXTRACT_REASONING_EFFORT = os.getenv("VP_EXTRACT_REASONING_EFFORT", LLM_REASONING_EFFORT).strip()
VP_VOCAB_REASONING_EFFORT = os.getenv("VP_VOCAB_REASONING_EFFORT", LLM_REASONING_EFFORT).strip()
# Per-pair gallery coverage band for the balanced pair search. FLOOR is the
# minimum distinct published images a (disease, finding) lane must retain,
# TARGET is the coverage goal at which a lane counts as covered, and
# GALLERY_CAP bounds only the published gallery; every eligible surplus image
# remains stored as an uncapped reserve. Unlike the clamped VP_* ints above,
# malformed values raise instead of silently falling back.
def _coverage_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer") from None


VP_FINDING_IMAGE_FLOOR = _coverage_int("VP_FINDING_IMAGE_FLOOR", 3)
VP_FINDING_IMAGE_TARGET = _coverage_int("VP_FINDING_IMAGE_TARGET", 10)
VP_FINDING_GALLERY_CAP = _coverage_int("VP_FINDING_GALLERY_CAP", 20)
# Topics with fewer distinct published images than this get the disease-wide
# discovery pass (any article whose title names the disease).
VP_TOPIC_IMAGE_FLOOR = _coverage_int("VP_TOPIC_IMAGE_FLOOR", 10)


def validate_coverage_settings() -> tuple[int, int, int]:
    if not (
        1
        <= VP_FINDING_IMAGE_FLOOR
        <= VP_FINDING_IMAGE_TARGET
        <= VP_FINDING_GALLERY_CAP
    ):
        raise ValueError(
            "Coverage settings must satisfy 1 <= VP_FINDING_IMAGE_FLOOR "
            "<= VP_FINDING_IMAGE_TARGET <= VP_FINDING_GALLERY_CAP"
        )
    return (
        VP_FINDING_IMAGE_FLOOR,
        VP_FINDING_IMAGE_TARGET,
        VP_FINDING_GALLERY_CAP,
    )


validate_coverage_settings()

# -----------------------------------------------------------------------------
# Throughput settings: request rates, worker pools and caches.
# -----------------------------------------------------------------------------
# Request rate for the public S3 bucket (pmc-oa-opendata); NCBI hosts unchanged.
VP_S3_RPS = max(0.1, _env_float("VP_S3_RPS", 20.0))
# Worker pool size for fetches/parse/store (W5/W6/W7). Host-level rate
# limiters (pmc.RATE_LIMITER) still cap per-host throughput, so this mainly
# controls how much of the S3/LLM budget is kept in flight.
VP_FETCH_CONCURRENCY = max(1, _env_int("VP_FETCH_CONCURRENCY", 16))
# Europe PMC searches run ahead of ingestion on this many workers, capped by
# VP_EPMC_RPS requests/second to www.ebi.ac.uk.
VP_SEARCH_CONCURRENCY = max(1, _env_int("VP_SEARCH_CONCURRENCY", 8))
VP_EPMC_RPS = max(0.1, _env_float("VP_EPMC_RPS", 8.0))
# Seconds to wait before retrying the searches a discovery pass lost to
# Europe PMC outages (each failed search is retried once at the pass end).
VP_SEARCH_RETRY_COOLDOWN = max(0, _env_int("VP_SEARCH_RETRY_COOLDOWN", 60))
# Max in-flight P3 vision-judge calls (W6).
VP_JUDGE_CONCURRENCY = max(1, _env_int("VP_JUDGE_CONCURRENCY", 16))
# Process-wide cap on in-flight LLM requests across every client and stage
# (run-all overlaps triage with judge/describe). 429s back off and retry
# (VP_RATE_LIMIT_RETRIES); lower this if the timings report shows many.
VP_LLM_MAX_IN_FLIGHT = max(1, _env_int("VP_LLM_MAX_IN_FLIGHT", 32))
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

# Per-model list prices, USD per 1M tokens: "in" is a prompt cache miss,
# "cached_in" a cache hit (defaults to "in"), "out" is output. Override or
# extend at runtime with
# VP_MODEL_PRICES_JSON='{"model": {"in": x, "cached_in": y, "out": z}}'.
MODEL_PRICES = {
    # OpenCode Go prices; Go usage counts against its monthly dollar limit.
    "muse-spark-1.3-contributor": {"in": 0.10, "cached_in": 0.002, "out": 0.20},
    # Free on OpenCode for a limited time.
    "step-5-preview-free": {"in": 0.0, "cached_in": 0.0, "out": 0.0},
}


def model_price(model: str) -> tuple[float, float, float] | None:
    """(input, cached input, output) USD per 1M tokens, or None if unpriced."""
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
    return (
        float(entry["in"]),
        float(entry.get("cached_in", entry["in"])),
        float(entry["out"]),
    )

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
