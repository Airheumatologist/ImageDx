"""SQLite schema and data access for the Visual Findings Library pilot.

Schema follows docs/visual_pilot_plan.md §4 exactly. ``llm_calls`` doubles as
the response cache (unique ``input_hash`` covers stage + model + payload) and
the cost ledger.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS diseases (
    disease_key     TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    mondo_id        TEXT,
    mesh_id         TEXT,
    synonyms_json   TEXT NOT NULL DEFAULT '[]',
    subtypes_json   TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS findings_vocab (
    finding_key      TEXT PRIMARY KEY,
    disease_keys_json TEXT NOT NULL DEFAULT '[]',
    label            TEXT NOT NULL,
    synonyms_json    TEXT NOT NULL DEFAULT '[]',
    category         TEXT NOT NULL,
    approved         INTEGER NOT NULL DEFAULT 0,
    proposed_by_llm  INTEGER NOT NULL DEFAULT 0,
    proposal_count   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS articles (
    pmcid                   TEXT PRIMARY KEY,
    pmid                    TEXT,
    doi                     TEXT,
    title                   TEXT,
    journal                 TEXT,
    year                    INTEGER,
    country                 TEXT,
    publication_types_json  TEXT NOT NULL DEFAULT '[]',
    license_code            TEXT,
    license_url             TEXT,
    oa_subset               TEXT,
    retrieval_score         REAL,
    retrieval_evidence_json TEXT NOT NULL DEFAULT '[]',
    primary_disease_keys_json TEXT NOT NULL DEFAULT '[]',
    relevance_decision      TEXT,
    relevance_reason        TEXT,
    study_region            TEXT,
    error                   TEXT,
    s3_prefix               TEXT,
    media_files_json        TEXT,
    authors_json            TEXT,
    author_count            INTEGER,
    journal_name            TEXT,
    status                  TEXT NOT NULL DEFAULT 'candidate',
    created_at              TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at              TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS figures (
    figure_id            TEXT PRIMARY KEY,
    pmcid                TEXT NOT NULL REFERENCES articles(pmcid),
    label                TEXT,
    caption              TEXT,
    in_text_mentions_json TEXT NOT NULL DEFAULT '[]',
    fig_permissions_text TEXT,
    effective_license    TEXT,
    image_url            TEXT,
    image_format         TEXT,
    sha256               TEXT,
    status               TEXT NOT NULL DEFAULT 'pending',
    triage_json          TEXT,
    vision_json          TEXT,
    error                TEXT,
    attempts             INTEGER NOT NULL DEFAULT 0,
    created_at           TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at           TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS panels (
    panel_id               TEXT PRIMARY KEY,
    figure_id              TEXT NOT NULL REFERENCES figures(figure_id),
    pmcid                  TEXT NOT NULL REFERENCES articles(pmcid),
    panel_label            TEXT,
    disease_key            TEXT REFERENCES diseases(disease_key),
    subtype                TEXT,
    modality               TEXT,
    body_site              TEXT,
    findings_json          TEXT NOT NULL DEFAULT '[]',
    typicality             TEXT,
    stage                  TEXT,
    age_group              TEXT,
    skin_tone              TEXT,
    stated_ethnicity       TEXT,
    stated_ethnicity_quote TEXT,
    study_region           TEXT,
    annotations_present    INTEGER,
    bbox_json              TEXT,
    crop_mode              TEXT,
    confidence             REAL,
    rationale              TEXT,
    image_path             TEXT,
    thumb_path             TEXT,
    width                  INTEGER,
    height                 INTEGER,
    sha256                 TEXT,
    attribution_text       TEXT,
    license_code           TEXT,
    license_url            TEXT,
    source_url             TEXT,
    created_at             TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at             TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS disease_findings (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    disease_key        TEXT NOT NULL REFERENCES diseases(disease_key),
    finding_key        TEXT,
    subtype            TEXT,
    frequency_text     TEXT,
    frequency_pct_low  REAL,
    frequency_pct_high REAL,
    source             TEXT,
    pmcid              TEXT,
    quote              TEXT,
    created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Reversible publication exclusions. Original judgments, rows and files are
-- retained; a review applies only to the exact image that was audited.
CREATE TABLE IF NOT EXISTS panel_curation (
    panel_id        TEXT PRIMARY KEY REFERENCES panels(panel_id),
    image_sha256    TEXT NOT NULL,
    decision        TEXT NOT NULL CHECK (decision = 'exclude'),
    reason          TEXT NOT NULL,
    policy_version  TEXT NOT NULL,
    reviewed_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE VIEW IF NOT EXISTS published_panels AS
SELECT p.* FROM panels p WHERE NOT EXISTS (
    SELECT 1 FROM panel_curation pc
    WHERE pc.panel_id = p.panel_id AND pc.decision = 'exclude'
      AND pc.image_sha256 = COALESCE(p.sha256, '')
);

CREATE TABLE IF NOT EXISTS llm_calls (
    call_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    stage             TEXT NOT NULL,
    model             TEXT NOT NULL,
    input_hash        TEXT NOT NULL,
    request_meta_json TEXT,
    response_json     TEXT,
    input_tokens      INTEGER,
    output_tokens     INTEGER,
    cost_usd          REAL,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_llm_calls_input_hash
    ON llm_calls(input_hash);
CREATE INDEX IF NOT EXISTS idx_articles_status ON articles(status);
CREATE INDEX IF NOT EXISTS idx_figures_pmcid ON figures(pmcid);
CREATE INDEX IF NOT EXISTS idx_figures_status ON figures(status);
CREATE INDEX IF NOT EXISTS idx_panels_figure ON panels(figure_id);
CREATE INDEX IF NOT EXISTS idx_panels_disease ON panels(disease_key);
CREATE INDEX IF NOT EXISTS idx_panels_sha256 ON panels(sha256);
"""

_PK_COLUMNS = {
    "diseases": "disease_key",
    "findings_vocab": "finding_key",
    "articles": "pmcid",
    "figures": "figure_id",
    "panels": "panel_id",
    "disease_findings": "id",
    "llm_calls": "call_id",
    "panel_curation": "panel_id",
}

def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a connection to the pilot DB (creating parent dirs as needed)."""
    path = Path(db_path) if db_path is not None else config.db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False lets LLMClient.call_many share one connection
    # across its worker threads; callers must serialize writes (llm.py does
    # this with a lock).
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection | None = None) -> sqlite3.Connection:
    """Create all tables/indexes (idempotent). Returns the connection used."""
    if conn is None:
        conn = connect()
    conn.executescript(SCHEMA)
    _migrate_articles(conn)
    conn.commit()
    return conn


# Columns added after the first schema version. CREATE TABLE above already
# includes them; the ALTERs only fire on databases created before they
# existed (init stays idempotent via PRAGMA table_info).
_ARTICLE_MIGRATIONS = (
    "ALTER TABLE articles ADD COLUMN study_region TEXT",
    "ALTER TABLE articles ADD COLUMN error TEXT",
    "ALTER TABLE articles ADD COLUMN retrieval_evidence_json TEXT NOT NULL DEFAULT '[]'",
    # C2 additions (contract §4): S3 bundle + attribution metadata.
    "ALTER TABLE articles ADD COLUMN s3_prefix TEXT",
    "ALTER TABLE articles ADD COLUMN media_files_json TEXT",
    "ALTER TABLE articles ADD COLUMN authors_json TEXT",
    "ALTER TABLE articles ADD COLUMN author_count INTEGER",
    "ALTER TABLE articles ADD COLUMN journal_name TEXT",
)


def _migrate_articles(conn: sqlite3.Connection) -> None:
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(articles)")}
    if not existing:
        return
    wanted = {
        "study_region": _ARTICLE_MIGRATIONS[0],
        "error": _ARTICLE_MIGRATIONS[1],
        "retrieval_evidence_json": _ARTICLE_MIGRATIONS[2],
        "s3_prefix": _ARTICLE_MIGRATIONS[3],
        "media_files_json": _ARTICLE_MIGRATIONS[4],
        "authors_json": _ARTICLE_MIGRATIONS[5],
        "author_count": _ARTICLE_MIGRATIONS[6],
        "journal_name": _ARTICLE_MIGRATIONS[7],
    }
    for column, statement in wanted.items():
        if column not in existing:
            conn.execute(statement)


# -----------------------------------------------------------------------------
# JSON column helpers
# -----------------------------------------------------------------------------
def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def from_json(text: str | None, default: Any = None) -> Any:
    if text is None:
        return default
    return json.loads(text)


def llm_input_hash(stage: str, model: str, payload: Any) -> str:
    """Stable cache key for llm_calls.input_hash (covers stage + model)."""
    canonical = json.dumps(
        {"stage": stage, "model": model, "payload": payload},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# Status helpers
# -----------------------------------------------------------------------------
def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    _require_table(table)
    return [row["name"] for row in conn.execute(f"PRAGMA table_info({table})")]


def set_status(
    conn: sqlite3.Connection,
    table: str,
    pk: Any,
    status: str,
    **fields: Any,
) -> None:
    """Set ``status`` (and optional extra columns) on one row by primary key."""
    pk_col = _pk_column(table)
    columns = set(table_columns(conn, table))
    updates = {"status": status, **fields}
    unknown = set(updates) - columns
    if unknown:
        raise ValueError(f"{table}: unknown column(s) {sorted(unknown)}")
    sql_sets = ", ".join(f"{name} = :{name}" for name in updates)
    if "updated_at" in columns and "updated_at" not in updates:
        sql_sets += ", updated_at = datetime('now')"
    cur = conn.execute(
        f"UPDATE {table} SET {sql_sets} WHERE {pk_col} = :__pk",
        {**updates, "__pk": pk},
    )
    if cur.rowcount == 0:
        raise KeyError(f"{table}: no row with {pk_col} = {pk!r}")


def rows_with_status(
    conn: sqlite3.Connection,
    table: str,
    statuses: str | Iterable[str],
    disease: str | None = None,
    limit: int | None = None,
) -> list[sqlite3.Row]:
    """Rows whose status is in ``statuses``, optionally scoped to a disease.

    ``articles`` and ``figures`` carry their own status column; ``panels``
    inherit the pipeline status of their parent figure (§4 gives panels no
    status column). Disease scoping uses ``articles.primary_disease_keys_json``
    — directly for articles, via the article join for figures and panels.
    """
    pk_col = _pk_column(table)
    status_list = [statuses] if isinstance(statuses, str) else list(statuses)
    if not status_list:
        return []

    join = ""
    if table == "panels":
        join += " JOIN figures sf ON sf.figure_id = t.figure_id"
        status_expr = "sf.status"
    elif table in {"articles", "figures"}:
        status_expr = "t.status"
    else:
        raise ValueError(f"table {table!r} has no status column")

    conditions = [
        status_expr + " IN (" + ", ".join(f":s{i}" for i in range(len(status_list))) + ")"
    ]
    params: dict[str, Any] = {f"s{i}": s for i, s in enumerate(status_list)}

    if disease is not None:
        if table == "articles":
            conditions.append(
                "EXISTS (SELECT 1 FROM json_each(t.primary_disease_keys_json) je"
                " WHERE je.value = :disease)"
            )
        elif table in {"figures", "panels"}:
            join += " JOIN articles a ON a.pmcid = t.pmcid"
            conditions.append(
                "EXISTS (SELECT 1 FROM json_each(a.primary_disease_keys_json) je"
                " WHERE je.value = :disease)"
            )
        else:
            raise ValueError(f"disease filter not supported for table {table!r}")
        params["disease"] = disease

    sql = f"SELECT t.* FROM {table} t{join} WHERE {' AND '.join(conditions)}"
    sql += f" ORDER BY t.{pk_col}"
    if limit is not None:
        sql += " LIMIT :limit"
        params["limit"] = limit
    return list(conn.execute(sql, params))


def _require_table(table: str) -> None:
    if table not in _PK_COLUMNS:
        raise ValueError(f"unknown table {table!r}; expected one of {sorted(_PK_COLUMNS)}")


def _pk_column(table: str) -> str:
    _require_table(table)
    return _PK_COLUMNS[table]
