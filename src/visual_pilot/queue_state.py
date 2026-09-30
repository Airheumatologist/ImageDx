"""Durable per-article queue outcomes; article stage statuses stay resumable."""

from . import db


def record(conn, pmcid, stage, status, reason=None, **details):
    conn.execute(
        "INSERT INTO article_queue_state(pmcid,stage,status,reason,details_json) "
        "VALUES(?,?,?,?,?) ON CONFLICT(pmcid,stage) DO UPDATE SET "
        "status=excluded.status,reason=excluded.reason,details_json=excluded.details_json,"
        "updated_at=datetime('now')",
        (pmcid, stage, status, reason, db.to_json(details)),
    )
