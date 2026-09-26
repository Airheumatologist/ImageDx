"""turbopuffer PMC namespace access + DeepInfra query embeddings.

Stage 2 (``select_articles``) queries the PMC chunk index directly with BM25
and dense ANN rank-bys, so this module exposes just the pieces it needs:
``ns_pmc`` (the namespace handle), ``_embed_query`` (one OpenAI-compatible
embeddings call per query string) and ``embed_queries`` (one batched request
for a whole list, with per-item fallback).
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

    def embed_queries(self, queries: List[str]) -> List[Optional[List[float]]]:
        """Embed all queries in ONE batched embeddings request.

        DeepInfra accepts a list for ``input`` and returns one embedding
        object per item (aligned by ``index``), so the vectors are identical
        to per-item calls. On any batch failure (or an incomplete response)
        this falls back to ``_embed_query`` per item; entries that still fail
        come back as ``None`` so dense ANN is skipped for that query.
        """
        queries = list(queries)
        if not self.openai_client:
            return [None] * len(queries)
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model, input=queries
            )
            vectors: List[Optional[List[float]]] = [None] * len(queries)
            for item in response.data:
                if 0 <= item.index < len(vectors):
                    vectors[item.index] = item.embedding
            if any(vector is None for vector in vectors):
                raise ValueError("incomplete batched embeddings response")
            return vectors
        except Exception as exc:
            logger.warning(
                "Batched embedding failed (%s); falling back to per-item calls",
                exc,
            )
            return [self._embed_query(query) for query in queries]
