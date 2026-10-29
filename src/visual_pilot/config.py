"""Configuration for the Visual Findings Library pilot.

OpenRouter provider settings are read from the environment;
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
# Provider settings: OpenRouter runs all LLM stages (P2-P6) through its
# OpenAI-compatible chat completions endpoint.
# Article discovery uses the public Europe PMC REST API (no key).
# -----------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "").strip()

# -----------------------------------------------------------------------------
# Visual pilot settings (VP_*).
# Every LLM stage runs on qwen/qwen3.7-flash: multimodal (text + image), so it
# also serves the P3 vision judge, with JSON mode and reasoning effort.
# -----------------------------------------------------------------------------
DEFAULT_MODEL = "qwen/qwen3.7-flash"
VP_TRIAGE_MODEL = os.getenv("VP_TRIAGE_MODEL", DEFAULT_MODEL)
VP_EXTRACT_MODEL = os.getenv("VP_EXTRACT_MODEL", DEFAULT_MODEL)
VP_JUDGE_MODEL = os.getenv("VP_JUDGE_MODEL", DEFAULT_MODEL)
VP_DESCRIBE_MODEL = os.getenv("VP_DESCRIBE_MODEL", VP_EXTRACT_MODEL)
# Topic findings vocabulary (build-vocab, P6); defaults to VP_EXTRACT_MODEL.
VP_VOCAB_MODEL = os.getenv("VP_VOCAB_MODEL", VP_EXTRACT_MODEL)
VP_IMAGE_MAX_EDGE = _env_int("VP_IMAGE_MAX_EDGE", 1568)
VP_CONCURRENCY = max(1, _env_int("VP_CONCURRENCY", 32))
VP_LLM_TIMEOUT_SECONDS = max(1, _env_int("VP_LLM_TIMEOUT_SECONDS", 300))
# Wall-clock cap on one LLM request. The timeouts above are per read, which
# provider keep-alive bytes reset forever on a stalled request.
VP_LLM_MAX_REQUEST_SECONDS = max(1, _env_int("VP_LLM_MAX_REQUEST_SECONDS", 600))
# Figures per P2 caption-triage batch. Smaller batches finish sooner (each
# figure adds reasoning output) and spread across the VP_CONCURRENCY slots.
VP_TRIAGE_BATCH = max(1, _env_int("VP_TRIAGE_BATCH", 20))
# Per-stage reasoning effort (off/low/medium/high), sent as OpenRouter's
# ``reasoning`` request field; off disables thinking. Unset
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
# A topic's default gallery tops up with unlabeled images (eligible except
# that the source never names the model's finding label) until it shows this
# many distinct images. They earn no pair credit. 0 disables the fill.
VP_UNLABELED_FILL_TARGET = _coverage_int("VP_UNLABELED_FILL_TARGET", 20)
# Most distinct images a topic's default library shows; pair galleries are
# taken round-robin (each pair's best first), then combined plates, then the
# unlabeled fill. Pair coverage credit is unaffected. 0 disables the cap.
VP_TOPIC_IMAGE_CAP = _coverage_int("VP_TOPIC_IMAGE_CAP", 0)
# Articles fetched per thin topic per disease-pass round. A full sweep of the
# tail matters more than depth on one topic, so this stays small by default;
# later rounds deepen the topics still under target.
VP_TOPIC_ARTICLES_PER_ROUND = max(
    1, _coverage_int("VP_TOPIC_ARTICLES_PER_ROUND", 50)
)
# Whole-figure collages whose panels carry no approved finding are normally
# dropped ("unlabeled single-disease plate"). For thin topics they are useful
# unlabeled fill: set to 1 to publish them as whole figures, uncropped, with
# no pair credit.
VP_ALLOW_UNLABELED_PLATES = _env_int("VP_ALLOW_UNLABELED_PLATES", 0) == 1
# Second-tier disease search without the Europe PMC LICENSE clause, for thin
# topics whose strict search is exhausted. The local JATS/figure license gate
# still decides, so non-commercial articles are fetched but never published.
VP_RELAXED_LICENSE_SEARCH = _env_int("VP_RELAXED_LICENSE_SEARCH", 0) == 1


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
# The bucket is AWS Open Data and serves far more; this only keeps a run polite.
VP_S3_RPS = max(0.1, _env_float("VP_S3_RPS", 150.0))
# Worker pool size for fetches/parse/store (W5/W6/W7). Host-level rate
# limiters (pmc.RATE_LIMITER) still cap per-host throughput, so this mainly
# controls how much of the S3/LLM budget is kept in flight.
VP_FETCH_CONCURRENCY = max(1, _env_int("VP_FETCH_CONCURRENCY", 64))
# Europe PMC searches run ahead of ingestion on this many workers, capped by
# VP_EPMC_RPS requests/second to www.ebi.ac.uk.
VP_SEARCH_CONCURRENCY = max(1, _env_int("VP_SEARCH_CONCURRENCY", 8))
VP_EPMC_RPS = max(0.1, _env_float("VP_EPMC_RPS", 8.0))
# Seconds to wait before retrying the searches a discovery pass lost to
# Europe PMC outages (each failed search is retried once at the pass end).
VP_SEARCH_RETRY_COOLDOWN = max(0, _env_int("VP_SEARCH_RETRY_COOLDOWN", 60))
# Max in-flight P3 vision-judge calls (W6).
VP_JUDGE_CONCURRENCY = max(1, _env_int("VP_JUDGE_CONCURRENCY", 32))
# Process-wide cap on in-flight LLM requests across every client and stage
# (run-all overlaps triage with judge/describe). 429s back off and retry
# (VP_RATE_LIMIT_RETRIES); lower this if the timings report shows many.
VP_LLM_MAX_IN_FLIGHT = max(1, _env_int("VP_LLM_MAX_IN_FLIGHT", 64))
# Output-token cap per LLM request (reasoning included; 0 = provider default).
# Valid replies stay under ~8k; the cap turns a rare runaway generation into a
# quick repair retry instead of minutes of output.
VP_LLM_MAX_OUTPUT_TOKENS = max(0, _env_int("VP_LLM_MAX_OUTPUT_TOKENS", 16384))
# Process-wide LLM requests per minute, spaced evenly (0 = unpaced). Set it to
# a provider's per-minute limit so requests wait instead of failing with 429s.
VP_LLM_RPM = max(0.0, _env_float("VP_LLM_RPM", 0.0))
# Per-request LLM timeout for the P3 judge (W6).
VP_JUDGE_TIMEOUT_SECONDS = max(1, _env_int("VP_JUDGE_TIMEOUT_SECONDS", 120))
# Retries on HTTP 429 for provider calls (W3).
VP_RATE_LIMIT_RETRIES = max(0, _env_int("VP_RATE_LIMIT_RETRIES", 4))
# In-memory cap for the judge->store original-bytes handoff (W6/W7).
VP_ORIGINALS_CACHE_MB = max(0, _env_int("VP_ORIGINALS_CACHE_MB", 512))
# Seconds a SQLite write waits for another connection's write lock before
# "database is locked". Parallel run-all processes (e.g. one per topic group)
# share the database, and a batch write can hold the lock for a while.
VP_SQLITE_BUSY_TIMEOUT_SECONDS = max(1, _env_int("VP_SQLITE_BUSY_TIMEOUT_SECONDS", 300))
# W0 timing instrumentation: 1 records timings and writes reports/timings_*.json.
VP_TIMINGS = _env_int("VP_TIMINGS", 1)
# W0 parity harness: 1 makes an llm_calls cache miss raise LLMError instead of
# calling the provider (used with seeded candidate runs; default off).
VP_LLM_CACHE_ONLY = _env_int("VP_LLM_CACHE_ONLY", 0)

# Per-model list prices, USD per 1M tokens: "in" is a prompt cache miss,
# "cached_in" a cache hit (defaults to "in"), "out" is output. Override or
# extend at runtime with
# VP_MODEL_PRICES_JSON='{"model": {"in": x, "cached_in": y, "out": z}}'.
# Used only when a reply carries no ``usage.cost`` (OpenRouter reports one).
MODEL_PRICES = {
    # OpenRouter list price for prompts under 32k tokens (higher tiers above).
    "qwen/qwen3.7-flash": {"in": 0.03, "cached_in": 0.006, "out": 0.13},
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
