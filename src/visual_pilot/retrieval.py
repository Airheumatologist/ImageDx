"""turbopuffer PMC namespace access + DeepInfra query embeddings.

Stage 2 (``select_articles``) queries the PMC chunk index directly with BM25
and dense ANN rank-bys, so this module exposes just the two pieces it needs:
``ns_pmc`` (the namespace handle) and ``_embed_query`` (one OpenAI-compatible
embeddings call per query string).
"""

from __future__ import annotations

import logging
from typing import List, Optional

try:
    import turbopuffer as tpuf
except ImportError:  # pragma: no cover - environment-dependent
    tpuf = None  # type: ignore

from . import config

logger = logging.getLogger(__name__)


class VisualRetriever:
    """Minimal retriever surface used by stage 2 article selection."""

    def __init__(self) -> None:
        if tpuf is None:
            raise ValueError("turbopuffer package is not installed")
        if not config.TURBOPUFFER_API_KEY:
            raise ValueError("TURBOPUFFER_API_KEY not set")
        self.tpuf = tpuf.Turbopuffer(
            api_key=config.TURBOPUFFER_API_KEY,
            region=config.TURBOPUFFER_REGION,
            timeout=config.TURBOPUFFER_TIMEOUT_SECONDS,
        )
        self.ns_pmc = self.tpuf.namespace(config.TURBOPUFFER_NAMESPACE_PMC)

        self.embedding_model = config.EMBEDDING_MODEL
        self.openai_client = None
        if config.DEEPINFRA_API_KEY:
            from openai import OpenAI

            self.openai_client = OpenAI(
                api_key=config.DEEPINFRA_API_KEY,
                base_url=config.DEEPINFRA_BASE_URL,
                timeout=config.EMBEDDING_TIMEOUT_SECONDS,
            )

    def _embed_query(self, query: str) -> Optional[List[float]]:
        if not self.openai_client:
            return None
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model, input=query
            )
            return response.data[0].embedding
        except Exception as exc:
            logger.warning("Embedding failed: %s", exc)
            return None
