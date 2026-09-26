"""Configuration for the Visual Findings Library pilot.

Provider settings (xAI, DeepInfra, turbopuffer) are read from the environment;
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


# -----------------------------------------------------------------------------
# Provider settings for retrieval (turbopuffer PMC namespace + DeepInfra
# embeddings) and the LLM stages (P1-P4).
# -----------------------------------------------------------------------------
XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
DEEPINFRA_BASE_URL = os.getenv(
    "DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai"
)
OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY")
OPENCODE_BASE_URL = os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/zen/v1")
TURBOPUFFER_API_KEY = os.getenv("TURBOPUFFER_API_KEY", "")
TURBOPUFFER_REGION = os.getenv("TURBOPUFFER_REGION", "gcp-us-central1").strip()
TURBOPUFFER_NAMESPACE_PMC = os.getenv(
    "TURBOPUFFER_NAMESPACE_PMC", "medical_database_pmc"
)
TURBOPUFFER_TIMEOUT_SECONDS = _env_int("TURBOPUFFER_TIMEOUT_SECONDS", 30)
EMBEDDING_MODEL = os.getenv(
    "RUNTIME_EMBEDDING_MODEL",
    os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
)
EMBEDDING_TIMEOUT_SECONDS = _env_int("DEEPINFRA_EMBED_TIMEOUT_SECONDS", 120)

# Credentials for each supported VP_LLM_PROVIDER value.
_LLM_PROVIDER_CREDENTIALS = {
    "xai": lambda: (XAI_API_KEY, XAI_BASE_URL),
    "deepinfra": lambda: (DEEPINFRA_API_KEY, DEEPINFRA_BASE_URL),
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
# Visual pilot settings (VP_*), defaults per spec §3 / repo adjustments box.
# Provider settled in W2/W4: DeepInfra (the repo no longer provisions xAI).
# -----------------------------------------------------------------------------
VP_LLM_PROVIDER = os.getenv("VP_LLM_PROVIDER", "deepinfra").strip().lower()
# Cheap instruct model with reliable JSON output for relevance/triage/extract.
# P2 triage default is Llama-4-Scout: the 235B endpoint timed out / 429'd on
# every batched P2 call (2026-09-25); Scout completed identical batches.
VP_TRIAGE_MODEL = os.getenv("VP_TRIAGE_MODEL", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
VP_EXTRACT_MODEL = os.getenv("VP_EXTRACT_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")
# Strongest VL model available on DeepInfra for the vision judge.
VP_JUDGE_MODEL = os.getenv("VP_JUDGE_MODEL", "Qwen/Qwen3-VL-235B-A22B-Instruct")
VP_IMAGE_MAX_EDGE = _env_int("VP_IMAGE_MAX_EDGE", 1568)
VP_CONCURRENCY = max(1, _env_int("VP_CONCURRENCY", 4))
VP_LLM_TIMEOUT_SECONDS = max(1, _env_int("VP_LLM_TIMEOUT_SECONDS", 300))
# Figures per P2 caption-triage batch.
VP_TRIAGE_BATCH = max(1, _env_int("VP_TRIAGE_BATCH", 40))
# Maximum per-disease visual finding/modality passage queries in stage 2.
VP_VISUAL_QUERY_CAP = max(0, _env_int("VP_VISUAL_QUERY_CAP", 12))

# Per-model USD per 1M tokens (input/output). Source: DeepInfra
# `GET {DEEPINFRA_BASE_URL}/models` -> metadata.pricing, cross-checked against
# https://deepinfra.com/models (e.g. /Qwen/Qwen3-VL-235B-A22B-Instruct).
# Override or extend at runtime with VP_MODEL_PRICES_JSON='{"model": {"in": x, "out": y}}'.
MODEL_PRICES = {
    "Qwen/Qwen3-235B-A22B-Instruct-2507": {"in": 0.09, "out": 0.55},
    "Qwen/Qwen3-VL-235B-A22B-Instruct": {"in": 0.20, "out": 0.88},
    # Cheaper VL fallback.
    "Qwen/Qwen3-VL-30B-A3B-Instruct": {"in": 0.15, "out": 0.60},
    # Backup VLM.
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": {"in": 0.10, "out": 0.30},
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
