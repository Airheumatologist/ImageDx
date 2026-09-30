"""Disease–finding semantic reranking with image evidence as the first tier.

Cosine similarity uses real BGE-M3 embeddings through the existing provider.
The deterministic lexical fallback is reported, never called semantic. Europe
PMC source signals are bounded secondary terms. Ranking does not relax any
figure license, age or publication gate.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections.abc import Callable
from functools import lru_cache

from . import article_rank, config, db, diseases, source_quality

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1024)
def finding_terms(finding_key: str) -> list[str]:
    item = next((item for item in diseases.load_findings_vocab()
                 if item.get("finding_key") == finding_key), {})
    return list(dict.fromkeys([str(item.get("label") or finding_key.replace("_", " ")),
                              *map(str, item.get("synonyms", []))]))


def _hits(text: str, terms: list[str]) -> bool:
    return any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.I)
               for term in terms if len(term) > 2)


def caption_pair_evidence(caption: dict, disease_key: str, finding_key: str) -> bool:
    """Conservative explicit pair evidence within one sentence/panel segment.

    Co-occurrence in separate labeled panels of a mixed-disease figure is not
    explicit attribution. This is prioritization evidence, not final curation.
    """
    text = str(caption.get("caption") or "")
    if not caption.get("eligible", True):
        return False
    disease_terms = article_rank._disease_terms(disease_key)
    terms = finding_terms(finding_key)
    # Separate captions by labeled panels when available, then sentences.
    segments = re.split(r"\([A-Za-z0-9]\)|(?<=[.!?])\s+", text)
    for segment in segments:
        signal = article_rank.caption_signal(segment, disease_key)
        if (_hits(segment, disease_terms) and _hits(segment, terms)
                and signal["useful"]):
            return True
    return False


def document_text(article: dict) -> str:
    captions = " ".join(str(item.get("caption") or "") for item in
                        article.get("caption_candidates", []) if item.get("eligible", True))
    passages = " ".join(str(item.get("text") or item.get("page_content") or item.get("passage") or "")
                        for item in article.get("matched_passages", []) if isinstance(item, dict))
    # Actual text only: retrieval query terms are not article evidence.
    return (str(article.get("title") or "")[:1000] + "\nFigure captions: " + captions[:5000]
            + "\nAbstract: " + str(article.get("abstract") or
                                   article.get("source_metadata", {}).get("abstract") or "")[:4000]
            + "\nPassages: " + passages[:3000])


@lru_cache(maxsize=4096)
def pair_query(disease_key: str, finding_key: str) -> str:
    info = diseases.load_diseases().get(disease_key, {})
    terms = finding_terms(finding_key)
    return (f"Clinical images showing {terms[0]} in a patient with "
            f"{info.get('name') or disease_key}. "
            f"Disease synonyms: {', '.join(info.get('synonyms', []))}. "
            f"Manifestation synonyms: {', '.join(terms[1:])}. "
            "Figure caption explicitly attributes this manifestation to this disease.")


def _provider_embeddings(texts: list[str]) -> list[list[float] | None]:
    if not config.DEEPINFRA_API_KEY or not config.VP_PAIR_SEMANTIC_RERANK:
        return [None] * len(texts)
    from openai import OpenAI

    client = OpenAI(api_key=config.DEEPINFRA_API_KEY, base_url=config.DEEPINFRA_BASE_URL,
                    timeout=min(60, config.EMBEDDING_TIMEOUT_SECONDS), max_retries=1)
    try:
        result = client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
        vectors = [None] * len(texts)
        for item in result.data:
            if 0 <= item.index < len(texts):
                vectors[item.index] = item.embedding
        return vectors
    finally:
        client.close()


def _valid(vector) -> bool:
    return (isinstance(vector, (list, tuple)) and bool(vector)
            and all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector)
            and any(vector))


def cosine(left, right) -> float | None:
    if not _valid(left) or not _valid(right) or len(left) != len(right):
        return None
    norm = math.sqrt(sum(v*v for v in left) * sum(v*v for v in right))
    return max(-1.0, min(1.0, sum(a*b for a, b in zip(left, right)) / norm))


def _embeddings(conn, texts: list[str], embed_many: Callable | None, persist: bool) -> dict:
    unique = list(dict.fromkeys(texts))
    vectors = {}
    keys = {text: hashlib.sha256((config.EMBEDDING_MODEL + "\0" + text).encode()).hexdigest()
            for text in unique}
    missing = []
    for text in unique:
        row = conn.execute("SELECT vector_json FROM ranking_embeddings WHERE cache_key=?", (keys[text],)).fetchone()
        vector = db.from_json(row["vector_json"], None) if row else None
        if _valid(vector):
            vectors[text] = vector
        else:
            missing.append(text)
    for offset in range(0, len(missing), 32):
        batch = missing[offset:offset + 32]
        try:
            result = list((embed_many or _provider_embeddings)(batch))
            if len(result) != len(batch):
                raise ValueError("incomplete embedding response")
        except Exception as exc:
            logger.warning("Pair semantic reranking unavailable; lexical fallback: %s", exc)
            result = [None] * len(batch)
        for text, vector in zip(batch, result):
            vectors[text] = vector if _valid(vector) else None
            if persist and _valid(vector):
                conn.execute("INSERT OR REPLACE INTO ranking_embeddings(cache_key,model,vector_json) VALUES(?,?,?)",
                             (keys[text], config.EMBEDDING_MODEL, db.to_json(vector)))
    return vectors


def score_pair(article: dict, disease_key: str, finding_key: str, *, similarity=None) -> dict:
    text = document_text(article)
    disease_hit = _hits(text, article_rank._disease_terms(disease_key))
    finding_hit = _hits(text, finding_terms(finding_key))
    explicit = sum(caption_pair_evidence(item, disease_key, finding_key)
                   for item in article.get("caption_candidates", []))
    quality = source_quality.quality_signal(article.get("source_metadata"))
    # Tier 2 is reserved for explicit, eligible pair images. Pair mention alone
    # is tier 1; broad disease papers tier 0. Citations cannot change a tier.
    tier = 2 if explicit else 1 if disease_hit and finding_hit else 0
    semantic = 10.0 * max(0.0, similarity) if similarity is not None else 0.0
    lexical = (2.0 if disease_hit else 0.0) + (3.0 if finding_hit else 0.0)
    parts = {"semantic_similarity": semantic, "pair_text": lexical,
             "explicit_pair_caption": min(explicit, 3) * 2.0, **quality["parts"]}
    return {"score": sum(parts.values()), "parts": parts, "evidence_tier": tier,
            "explicit_pair_caption_count": explicit, "cosine_similarity": similarity,
            "semantic_status": "ok" if similarity is not None else "lexical_fallback",
            "semantic_model": config.EMBEDDING_MODEL if similarity is not None else None,
            "query": pair_query(disease_key, finding_key),
            "retracted": quality["retracted"], "source_quality": quality,
            "source_metadata": article.get("source_metadata", {})}


def sort_key(row: dict, finding_key: str) -> tuple:
    result = row.get("pair_rankings", {}).get(finding_key, {})
    return (result.get("evidence_tier", 0), result.get("score", 0.0),
            float(row.get("retrieval_score") or 0), str(row.get("pmcid") or ""))


def caption_shortlist(conn, ranked: list[dict], disease_key: str, finding_keys, limit: int) -> list[dict]:
    """Round-robin pair-aware caption peeks, then global spare capacity."""
    if limit <= 0:
        return []
    lane_ids = {}
    for record in conn.execute("SELECT pmcid,finding_key FROM manifestation_candidates WHERE disease_key=?",
                               (disease_key,)):
        lane_ids.setdefault(record["finding_key"], set()).add(record["pmcid"])
    ordered = {}
    for finding in sorted(finding_keys):
        candidates = [row for row in ranked if row["pmcid"] in lane_ids.get(finding, set())]
        scores = {row["pmcid"]: score_pair(row, disease_key, finding) for row in candidates}
        ordered[finding] = sorted(candidates, key=lambda row: (
            scores[row["pmcid"]]["evidence_tier"],
            scores[row["pmcid"]]["score"],
            float(row.get("retrieval_score") or 0), row["pmcid"]), reverse=True)
    selected = []
    seen = set()
    while len(selected) < limit:
        progressed = False
        for candidates in ordered.values():
            while candidates and candidates[0]["pmcid"] in seen:
                candidates.pop(0)
            if candidates and len(selected) < limit:
                row = candidates.pop(0)
                selected.append(row)
                seen.add(row["pmcid"])
                progressed = True
        if not progressed:
            break
    for row in ranked:
        if len(selected) >= limit:
            break
        if row["pmcid"] not in seen:
            selected.append(row)
            seen.add(row["pmcid"])
    return selected


def rank_pairs(conn, rows: list[dict], disease_key: str, finding_keys, *,
               semantic_rows: list[dict] = (), embed_many=None, persist=True) -> None:
    """Score actual candidate-lane pairs; semantically enrich only shortlist.

    Rows beyond the bounded shortlist receive a named lexical fallback. Pair
    details persist separately, retaining model, query, metadata and features.
    """
    by_id = {row["pmcid"]: row for row in rows}
    keys = set(finding_keys)
    pairs = [(record["pmcid"], record["finding_key"]) for record in conn.execute(
        "SELECT pmcid,finding_key FROM manifestation_candidates WHERE disease_key=?", (disease_key,))
        if record["pmcid"] in by_id and record["finding_key"] in keys]
    semantic_ids = {row["pmcid"] for row in semantic_rows} & by_id.keys()
    if not config.VP_PAIR_SEMANTIC_RERANK and embed_many is None:
        semantic_ids = set()
    active_pairs = [(key, finding) for key, finding in pairs if key in semantic_ids]
    queries = {finding: pair_query(disease_key, finding) for _, finding in active_pairs}
    documents = {key: document_text(by_id[key]) for key, _ in active_pairs}
    vectors = _embeddings(conn, list(queries.values()) + list(documents.values()), embed_many, persist)
    fallback_count = 0
    for key, finding in pairs:
        similarity = cosine(vectors.get(queries.get(finding)), vectors.get(documents.get(key)))
        result = score_pair(by_id[key], disease_key, finding, similarity=similarity)
        if similarity is None:
            fallback_count += 1
            result["semantic_fallback_reason"] = (
                "disabled" if not config.VP_PAIR_SEMANTIC_RERANK else
                "outside_shortlist" if key not in semantic_ids else
                "missing_credentials" if not config.DEEPINFRA_API_KEY and embed_many is None else
                "embedding_unavailable_or_invalid")
        by_id[key].setdefault("pair_rankings", {})[finding] = result
        if persist:
            conn.execute("INSERT INTO article_pair_rankings(disease_key,finding_key,pmcid,scoring_json) "
                         "VALUES(?,?,?,?) ON CONFLICT(disease_key,finding_key,pmcid) DO UPDATE SET "
                         "scoring_json=excluded.scoring_json,updated_at=datetime('now')",
                         (disease_key, finding, key, db.to_json(result)))
    if persist:
        conn.commit()
    logger.info("Pair ranking %s: %d candidate pairs, %d semantic, %d lexical fallback",
                disease_key, len(pairs), len(pairs) - fallback_count, fallback_count)
