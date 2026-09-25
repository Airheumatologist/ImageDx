"""Retriever factory for turbopuffer-only runtime retrieval."""

from __future__ import annotations

import logging

from .config import RETRIEVAL_BACKEND

logger = logging.getLogger(__name__)


def create_retriever(n_retrieval: int):
    backend = RETRIEVAL_BACKEND.strip().lower()
    if backend != "turbopuffer":
        raise ValueError("RETRIEVAL_BACKEND must be 'turbopuffer'")

    from .retriever_turbopuffer import TurbopufferRetriever

    logger.info("Using turbopuffer retriever backend")
    return TurbopufferRetriever(n_retrieval=n_retrieval)
