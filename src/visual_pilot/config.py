"""Configuration for the Visual Findings Library pilot.

Provider settings (OpenCode Zen, DeepInfra embeddings, turbopuffer) are read
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


# -----------------------------------------------------------------------------
# Provider settings: turbopuffer PMC namespace for retrieval, DeepInfra for
# query embeddings only, and OpenCode Zen for all LLM stages (P1-P4).
# -----------------------------------------------------------------------------
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
    os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
)
EMBEDDING_TIMEOUT_SECONDS = _env_int("DEEPINFRA_EMBED_TIMEOUT_SECONDS", 120)

# Credentials for each supported VP_LLM_PROVIDER value. OpenCode Zen is the
# only LLM provider; DeepInfra credentials above are for embeddings only.
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
# Visual pilot settings (VP_*), defaults per spec §3 / repo adjustments box.
# Every LLM stage runs on OpenCode Zen; space-bunny-free is multimodal, so the
# same model covers caption triage, extraction, and the vision judge.
# -----------------------------------------------------------------------------
VP_LLM_PROVIDER = os.getenv("VP_LLM_PROVIDER", "opencode").strip().lower()
VP_TRIAGE_MODEL = os.getenv("VP_TRIAGE_MODEL", "space-bunny-free")
VP_EXTRACT_MODEL = os.getenv("VP_EXTRACT_MODEL", "space-bunny-free")
VP_JUDGE_MODEL = os.getenv("VP_JUDGE_MODEL", "space-bunny-free")
VP_IMAGE_MAX_EDGE = _env_int("VP_IMAGE_MAX_EDGE", 1568)
VP_CONCURRENCY = max(1, _env_int("VP_CONCURRENCY", 4))
VP_LLM_TIMEOUT_SECONDS = max(1, _env_int("VP_LLM_TIMEOUT_SECONDS", 300))
# Figures per P2 caption-triage batch.
VP_TRIAGE_BATCH = max(1, _env_int("VP_TRIAGE_BATCH", 40))
# Maximum per-disease visual finding/modality passage queries in stage 2.
VP_VISUAL_QUERY_CAP = max(0, _env_int("VP_VISUAL_QUERY_CAP", 12))

# Per-model USD per 1M tokens (input/output). space-bunny-free is free for a
# limited time on OpenCode Zen; override or extend at runtime with
# VP_MODEL_PRICES_JSON='{"model": {"in": x, "out": y}}'.
MODEL_PRICES = {
    "space-bunny-free": {"in": 0.0, "out": 0.0},
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
