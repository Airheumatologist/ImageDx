"""Balanced pair-search schema migrations and coverage-config validation.

Everything runs against scratch SQLite files under tmp_path; config checks
spawn a clean interpreter subprocess so import-time validation sees a
controlled environment.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.visual_pilot import config, db

REPO_ROOT = Path(__file__).resolve().parents[2]

NEW_TABLES = {"panel_identity_reviews", "pair_search_attempts"}
CANDIDATE_PROVENANCE_COLUMNS = ("provenance_status", "provenance_disease_key")
LANE_BALANCED_COLUMNS = (
    "tier",
    "last_served_sequence",
    "blocked_reason",
    "search_policy_version",
    "last_deficit_reduction_at",
    "last_published_distinct",
    "last_search_at",
    "created_at",
)
COVERAGE_ENV_KEYS = (
    "VP_FINDING_IMAGE_FLOOR",
    "VP_FINDING_IMAGE_TARGET",
    "VP_FINDING_GALLERY_CAP",
)


def _columns(conn, table):
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def _make_old_db(path):
    """Build a scratch DB as it looked before the balanced-schema workstream."""
    conn = db.connect(path)
    conn.executescript(db.SCHEMA)
    conn.execute("DROP TABLE panel_identity_reviews")
    conn.execute("DROP TABLE pair_search_attempts")
    for column in CANDIDATE_PROVENANCE_COLUMNS:
        conn.execute(f"ALTER TABLE manifestation_candidates DROP COLUMN {column}")
    for column in LANE_BALANCED_COLUMNS:
        conn.execute(f"ALTER TABLE manifestation_lanes DROP COLUMN {column}")
    # Seed a terminal candidate, a covered lane, a locked representative, a
    # curation exclusion and the panel identity hash it pins.
    conn.execute("INSERT INTO diseases(disease_key,name) VALUES('d1','D1')")
    conn.execute(
        "INSERT INTO findings_vocab(finding_key,label,category,approved) "
        "VALUES('f1','F1','skin',1)"
    )
    conn.execute("INSERT INTO articles(pmcid,status) VALUES('PMC1','irrelevant')")
    conn.execute(
        "INSERT INTO manifestation_candidates"
        "(disease_key,finding_key,pmcid,query,best_rank,status,last_outcome) "
        "VALUES('d1','f1','PMC1','rash image',3,'exhausted','article_irrelevant')"
    )
    conn.execute(
        "INSERT INTO manifestation_lanes"
        "(disease_key,finding_key,status,last_outcome,selected_count) "
        "VALUES('d1','f1','covered','target_reached',10)"
    )
    conn.execute(
        "INSERT INTO figures(figure_id,pmcid,status) VALUES('PMC1:f1','PMC1','stored')"
    )
    conn.execute(
        "INSERT INTO panels(panel_id,figure_id,pmcid,disease_key,sha256) "
        "VALUES('p1','PMC1:f1','PMC1','d1','sha-old')"
    )
    conn.execute(
        "INSERT INTO manifestation_representatives"
        "(disease_key,finding_key,panel_id,score,locked) VALUES('d1','f1','p1',1.5,1)"
    )
    conn.execute(
        "INSERT INTO panel_curation(panel_id,image_sha256,decision,reason,policy_version) "
        "VALUES('p1','sha-old','exclude','off-topic','v1')"
    )
    conn.commit()
    return conn


def _assert_balanced_schema(conn):
    tables = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert NEW_TABLES <= tables
    indexes = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
    }
    assert "idx_pair_search_attempts_pair" in indexes
    assert set(CANDIDATE_PROVENANCE_COLUMNS) <= _columns(
        conn, "manifestation_candidates"
    )
    assert set(LANE_BALANCED_COLUMNS) <= _columns(conn, "manifestation_lanes")


def test_old_schema_migrates_idempotently_and_preserves_rows(tmp_path):
    conn = _make_old_db(tmp_path / "old.sqlite")
    db.init_db(conn)
    db.init_db(conn)  # second init must be a no-op

    _assert_balanced_schema(conn)

    candidate = conn.execute(
        "SELECT status,last_outcome,provenance_status,provenance_disease_key "
        "FROM manifestation_candidates WHERE pmcid='PMC1'"
    ).fetchone()
    # Pre-existing outcome preserved; provenance stays unresolved until an
    # explicit refresh activates it.
    assert tuple(candidate) == ("exhausted", "article_irrelevant", "unresolved", None)

    lane = conn.execute("SELECT * FROM manifestation_lanes").fetchone()
    assert (lane["status"], lane["last_outcome"], lane["selected_count"]) == (
        "covered",
        "target_reached",
        10,
    )
    assert lane["tier"] == "empty"
    assert lane["last_served_sequence"] == 0
    assert lane["last_published_distinct"] == 0
    assert lane["blocked_reason"] is None
    assert lane["search_policy_version"] is None
    assert lane["last_deficit_reduction_at"] is None
    assert lane["last_search_at"] is None
    assert lane["created_at"]  # backfilled by the migration

    rep = conn.execute(
        "SELECT panel_id,score,locked FROM manifestation_representatives"
    ).fetchone()
    assert tuple(rep) == ("p1", 1.5, 1)
    curation = conn.execute(
        "SELECT image_sha256,decision FROM panel_curation"
    ).fetchone()
    assert tuple(curation) == ("sha-old", "exclude")
    assert (
        conn.execute("SELECT sha256 FROM panels WHERE panel_id='p1'").fetchone()[
            "sha256"
        ]
        == "sha-old"
    )
    conn.close()


def test_fresh_schema_has_balanced_columns_and_defaults(tmp_path):
    conn = db.connect(tmp_path / "fresh.sqlite")
    db.init_db(conn)

    _assert_balanced_schema(conn)

    conn.execute("INSERT INTO diseases(disease_key,name) VALUES('d1','D1')")
    conn.execute("INSERT INTO articles(pmcid,status) VALUES('PMC1','candidate')")
    conn.execute(
        "INSERT INTO manifestation_candidates(disease_key,finding_key,pmcid) "
        "VALUES('d1','f1','PMC1')"
    )
    conn.execute(
        "INSERT INTO manifestation_lanes(disease_key,finding_key) VALUES('d1','f1')"
    )
    candidate = conn.execute(
        "SELECT provenance_status,provenance_disease_key "
        "FROM manifestation_candidates"
    ).fetchone()
    assert tuple(candidate) == ("unresolved", None)
    lane = conn.execute("SELECT * FROM manifestation_lanes").fetchone()
    assert lane["tier"] == "empty"
    assert lane["last_served_sequence"] == 0
    assert lane["last_published_distinct"] == 0
    assert lane["created_at"]
    conn.close()


def _coverage_env(overrides):
    env = {k: v for k, v in os.environ.items() if k not in COVERAGE_ENV_KEYS}
    env.update(overrides)
    return env


def _coverage_import(overrides):
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.visual_pilot import config;"
            "print(config.VP_FINDING_IMAGE_FLOOR,"
            " config.VP_FINDING_IMAGE_TARGET, config.VP_FINDING_GALLERY_CAP)",
        ],
        cwd=REPO_ROOT,
        env=_coverage_env(overrides),
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_coverage_defaults_are_3_10_20():
    proc = _coverage_import({})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "3 10 20"


def test_coverage_valid_override_2_5_12():
    proc = _coverage_import(
        {
            "VP_FINDING_IMAGE_FLOOR": "2",
            "VP_FINDING_IMAGE_TARGET": "5",
            "VP_FINDING_GALLERY_CAP": "12",
        }
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "2 5 12"


@pytest.mark.parametrize(
    "overrides",
    [
        {"VP_FINDING_IMAGE_FLOOR": "0"},  # 0/10/20: floor below 1
        {"VP_FINDING_IMAGE_TARGET": "2"},  # 3/2/20: floor above target
        {"VP_FINDING_GALLERY_CAP": "9"},  # 3/10/9: target above cap
    ],
)
def test_coverage_invalid_band_raises(overrides):
    proc = _coverage_import(overrides)
    assert proc.returncode != 0
    assert "Coverage settings must satisfy" in proc.stderr


def test_coverage_non_integer_raises_clear_error():
    proc = _coverage_import({"VP_FINDING_IMAGE_TARGET": "ten"})
    assert proc.returncode != 0
    assert "VP_FINDING_IMAGE_TARGET must be an integer" in proc.stderr


def test_pair_search_policy_constants_and_validator():
    assert config.PAIR_SEARCH_POLICY_VERSION == "balanced-pair-search.v1"
    assert config.PAIR_SEARCH_DEPTHS == (300, 600, 1200)
    assert config.PAIR_SEARCH_VARIANTS_PER_ROUND == 6
    assert config.validate_coverage_settings() == (
        config.VP_FINDING_IMAGE_FLOOR,
        config.VP_FINDING_IMAGE_TARGET,
        config.VP_FINDING_GALLERY_CAP,
    )
