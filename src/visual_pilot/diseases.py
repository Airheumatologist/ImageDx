"""Stage 1 seed data: the three pilot diseases and the findings vocabulary.

Seed files live in ``src/visual_pilot/data/``. ``seed()`` is idempotent: it
upserts descriptive fields but never resets ``approved``, ``proposed_by_llm``
or ``proposal_count`` on ``findings_vocab`` rows (LLM-proposed findings must
survive re-seeding).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .db import to_json

DATA_DIR = Path(__file__).resolve().parent / "data"

DISEASE_KEYS = ("sle", "dm", "as")

# findings_vocab.category enum per spec §4.
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
    }
)


def load_diseases() -> dict[str, dict]:
    return json.loads((DATA_DIR / "diseases.json").read_text(encoding="utf-8"))


def load_findings_vocab() -> list[dict]:
    return json.loads((DATA_DIR / "findings_vocab.json").read_text(encoding="utf-8"))


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
                disease["mondo_id"],
                disease["mesh_id"],
                to_json(disease["synonyms"]),
                to_json(disease["subtypes"]),
            ),
        )

    vocab = load_findings_vocab()
    valid_diseases = set(DISEASE_KEYS)
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
