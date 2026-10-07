"""Stage 1 seed data: the disease catalog and findings vocabulary.

Two catalogs exist, selected by ``VP_CATALOG``:

- ``topics`` (default): every topic in ``data/main_database_topics.json``
  (its ``specialties`` index). Topics flagged with a ``pilot_key`` keep that
  key and the curated pilot entry from ``diseases.json`` (subtypes, search
  synonyms), and pilot diseases missing from the index stay in the catalog,
  so existing library rows keep their disease.
- ``pilot``: only the curated pilot diseases in ``diseases.json``.

The vocabulary is the curated ``findings_vocab.json`` plus, in ``topics``
mode, ``topic_findings_vocab.json`` written by ``build-vocab`` for every
topic without curated findings.

Seed files live in ``src/visual_pilot/data/``. ``seed()`` is idempotent: it
upserts descriptive fields but never resets ``approved``, ``proposed_by_llm``
or ``proposal_count`` on ``findings_vocab`` rows (LLM-proposed findings must
survive re-seeding).
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import unicodedata
from functools import lru_cache
from pathlib import Path

from .db import to_json

DATA_DIR = Path(__file__).resolve().parent / "data"
REPO_ROOT = Path(__file__).resolve().parents[2]
TOPICS_PATH = REPO_ROOT / "data" / "main_database_topics.json"
TOPIC_VOCAB_PATH = DATA_DIR / "topic_findings_vocab.json"
CATALOGS = ("topics", "pilot")

# findings_vocab.category enum.
FINDING_CATEGORIES = frozenset(
    {
        "skin",
        "mucosa",
        "nail",
        "capillaroscopy",
        "histology",
        "radiology_xray",
        "ct",
        "mri",
        "us",
        "echo",
        "eye",
        "clinical_msk",
        # Added for the full topic index: endoscopic views, gross pathology,
        # nuclear medicine, and whole-body appearance (facial dysmorphism,
        # body habitus) that is neither skin nor musculoskeletal.
        "endoscopy",
        "gross",
        "nuclear",
        "clinical_general",
    }
)

_ACRONYM = re.compile(r"[A-Z0-9][A-Z0-9\-]{1,7}")


def is_acronym(term: str) -> bool:
    """Upper-case short forms ("ITP", "UC", "MEN2A") that collide with words.

    Case-insensitive matching turns them into common words ("all", "as"), so
    text matchers either skip them or match them case-sensitively.
    """
    return bool(_ACRONYM.fullmatch(str(term).strip()))


def catalog_mode() -> str:
    mode = os.getenv("VP_CATALOG", "topics").strip().lower()
    if mode not in CATALOGS:
        raise ValueError(f"VP_CATALOG must be one of {list(CATALOGS)}")
    return mode


def _load_pilot() -> dict[str, dict]:
    data = json.loads((DATA_DIR / "diseases.json").read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("diseases.json must contain a non-empty disease object")
    for key, disease in data.items():
        if not isinstance(key, str) or not key or not isinstance(disease, dict):
            raise ValueError("diseases.json entries must map non-empty keys to objects")
    return data


def load_topics(path: Path | None = None) -> list[dict]:
    """Every topic of the main database index, in specialty order."""
    data = json.loads((path or TOPICS_PATH).read_text(encoding="utf-8"))
    topics, seen = [], set()
    for specialty_topics in (data.get("specialties") or {}).values():
        for topic in specialty_topics:
            topic_id = topic.get("topic_id")
            if not topic_id or topic_id in seen:
                continue
            seen.add(topic_id)
            topics.append(topic)
    if not topics:
        raise ValueError(f"{path or TOPICS_PATH} lists no topics")
    return topics


def topic_disease_key(topic: dict) -> str:
    """Catalog key of a topic: its pilot key when it has one, else its id."""
    return topic.get("pilot_key") or topic["topic_id"]


def fold_accents(value: str) -> str:
    """"Romaña" -> "Romana", "Ménétrier" -> "Menetrier"; dropping the letter
    instead would split the word ("roma a sign")."""
    decomposed = unicodedata.normalize("NFKD", str(value))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


_PARENTHETICAL = re.compile(r"\s*\([^)]*\)")


def name_variants(name: str) -> list[str]:
    """Searchable forms of a display name, most useful first.

    Topic names carry a gloss in parentheses ("Loiasis (Loa Loa
    Filariasis)", "Cardiac Fibroma (Gorlin-Goltz Syndrome)") that no article
    title contains verbatim, so the name without it is searched; the gloss
    itself is often an associated condition rather than a synonym and is not.
    Accented names also get an unaccented form ("Menetrier Disease").
    """
    bare = " ".join(_PARENTHETICAL.sub(" ", name).split())
    out = [bare] if bare and bare != name else []
    out += [fold_accents(v) for v in (out or [name]) if fold_accents(v) != v]
    return list(dict.fromkeys(out))


def _topics_catalog() -> dict[str, dict]:
    pilot = _load_pilot()
    catalog: dict[str, dict] = {}
    for topic in load_topics():
        key = topic_disease_key(topic)
        context = {
            "topic_id": topic["topic_id"],
            "specialty": topic.get("specialty"),
            "subspecialty": topic.get("subspecialty"),
        }
        if key in pilot:
            catalog[key] = {**pilot[key], **context}
            continue
        name = str(topic.get("name") or key).strip()
        synonyms = [*name_variants(name), *(topic.get("synonyms") or [])]
        catalog[key] = {
            "name": name,
            "mondo_id": topic.get("mondo_id"),
            "mesh_id": topic.get("mesh_id"),
            "synonyms": [
                s for s in dict.fromkeys(str(s).strip() for s in synonyms)
                if s and s.casefold() != name.casefold()
            ],
            "subtypes": [],
            **context,
        }
    for key, disease in pilot.items():
        catalog.setdefault(key, disease)
    return catalog


@lru_cache(maxsize=None)
def _catalog(mode: str) -> dict[str, dict]:
    return _load_pilot() if mode == "pilot" else _topics_catalog()


def load_diseases() -> dict[str, dict]:
    """The configured catalog, parsed once per process. Treat as read-only."""
    return _catalog(catalog_mode())


def disease_keys_from_catalog() -> tuple[str, ...]:
    """Return the configured catalog keys in stable order."""
    return tuple(load_diseases())


def selected_keys(args) -> list[str]:
    """Catalog keys a command targets: ``--diseases`` list, else ``--disease``."""
    many = getattr(args, "diseases", None)
    if many:
        unknown = sorted(set(many) - set(load_diseases()))
        if unknown:
            raise SystemExit(f"unknown disease key(s): {', '.join(unknown)}")
        return list(dict.fromkeys(many))
    disease = getattr(args, "disease", "all")
    return list(disease_keys_from_catalog()) if disease == "all" else [disease]


# Kept as a module-level compatibility surface for stage code; each process
# loads the authoritative configured catalog once at import time.
DISEASE_KEYS = disease_keys_from_catalog()


@lru_cache(maxsize=None)
def _vocab(mode: str) -> tuple[dict, ...]:
    items = json.loads((DATA_DIR / "findings_vocab.json").read_text(encoding="utf-8"))
    if mode == "topics" and TOPIC_VOCAB_PATH.exists():
        known = {item["finding_key"] for item in items}
        catalog = _catalog(mode)
        for item in json.loads(TOPIC_VOCAB_PATH.read_text(encoding="utf-8")):
            if item["finding_key"] in known:
                continue
            if not set(item.get("disease_keys") or []) <= set(catalog):
                continue  # topic dropped from the index since the build
            known.add(item["finding_key"])
            items.append(item)
    return tuple(items)


def load_findings_vocab() -> list[dict]:
    """Curated pilot vocabulary plus the generated topic vocabulary."""
    return [dict(item) for item in _vocab(catalog_mode())]


@lru_cache(maxsize=None)
def _caption_terms(mode: str) -> dict[str, tuple[str, ...]]:
    return {
        item["finding_key"]: tuple(item["caption_terms"])
        for item in _vocab(mode)
        if item.get("caption_terms")
    }


def vocab_caption_terms(finding_key: str) -> list[str]:
    """Caption phrasings stored on a vocabulary row (generated topic vocab)."""
    return list(_caption_terms(catalog_mode()).get(finding_key, ()))


def disease_keys(conn: sqlite3.Connection) -> set[str]:
    return {row["disease_key"] for row in conn.execute("SELECT disease_key FROM diseases")}


def seed(conn: sqlite3.Connection) -> dict[str, int]:
    """Idempotently upsert diseases and the approved findings vocabulary.

    Returns row counts seeded (for CLI reporting).
    """
    diseases = load_diseases()
    for key, disease in diseases.items():
        conn.execute(
            """
            INSERT INTO diseases
                (disease_key, name, mondo_id, mesh_id, synonyms_json, subtypes_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(disease_key) DO UPDATE SET
                name          = excluded.name,
                mondo_id      = excluded.mondo_id,
                mesh_id       = excluded.mesh_id,
                synonyms_json = excluded.synonyms_json,
                subtypes_json = excluded.subtypes_json
            """,
            (
                key,
                disease["name"],
                disease.get("mondo_id"),
                disease.get("mesh_id"),
                to_json(disease.get("synonyms") or []),
                to_json(disease.get("subtypes") or []),
            ),
        )

    vocab = load_findings_vocab()
    valid_diseases = set(diseases)
    for item in vocab:
        _validate_vocab_item(item, valid_diseases)
        conn.execute(
            """
            INSERT INTO findings_vocab
                (finding_key, disease_keys_json, label, synonyms_json,
                 category, approved, proposed_by_llm, proposal_count)
            VALUES (?, ?, ?, ?, ?, 1, 0, 0)
            ON CONFLICT(finding_key) DO UPDATE SET
                disease_keys_json = excluded.disease_keys_json,
                label             = excluded.label,
                synonyms_json     = excluded.synonyms_json,
                category          = excluded.category
            """,
            (
                item["finding_key"],
                to_json(item["disease_keys"]),
                item["label"],
                to_json(item.get("synonyms", [])),
                item["category"],
            ),
        )
    conn.commit()
    return {"diseases": len(diseases), "findings_vocab": len(vocab)}


def _validate_vocab_item(item: dict, valid_diseases: set[str]) -> None:
    key = item["finding_key"]
    bad_diseases = set(item["disease_keys"]) - valid_diseases
    if bad_diseases:
        raise ValueError(f"{key}: unknown disease_keys {sorted(bad_diseases)}")
    if item["category"] not in FINDING_CATEGORIES:
        raise ValueError(f"{key}: unknown category {item['category']!r}")
