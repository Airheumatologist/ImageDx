"""Stage 4: caption triage (workstream W5b).

``pending`` figures are sent to the cheap text model in batches of 40 using
prompt P2. The per-figure result is stored verbatim in ``figures.triage_json``
and routed: ``drop`` or ``third_party`` -> ``caption_rejected``, ``keep`` ->
``caption_kept``, ``uncertain`` -> ``caption_uncertain``.

Figures the model omits stay ``pending`` with ``attempts`` bumped; a failed
batch bumps attempts and records the error. ``llm.BudgetExceeded`` leaves
figures pending so a rerun resumes cleanly (the response cache also makes a
rerun cost nothing for inputs already answered).
"""

from __future__ import annotations

import json
import logging

from . import config, db, diseases, llm
from .prompts import P2

logger = logging.getLogger(__name__)

BATCH_SIZE = config.VP_TRIAGE_BATCH
CAPTION_MAX_CHARS = 1500

_ROUTE_STATUS = {
    "keep": "caption_kept",
    "uncertain": "caption_uncertain",
}


def _batch_payload(rows) -> str:
    figures = [
        {
            "figure_id": row["figure_id"],
            "label": row["label"],
            "caption": (row["caption"] or "")[:CAPTION_MAX_CHARS],
            "in_text_mentions": db.from_json(row["in_text_mentions_json"], []),
        }
        for row in rows
    ]
    return json.dumps({"figures": figures}, ensure_ascii=False)


def _p2_request(rows) -> dict:
    return {
        "stage": "p2",
        "model": config.VP_TRIAGE_MODEL,
        "system": P2.system,
        "schema": P2.schema,
        "prompt_version": P2.version,
        "user_content": _batch_payload(rows),
    }


def _bump_attempts(conn, figure_id: str, error: str | None = None) -> None:
    if error is not None:
        conn.execute(
            "UPDATE figures SET attempts = attempts + 1, error = ?, "
            "updated_at = datetime('now') WHERE figure_id = ?",
            (error, figure_id),
        )
    else:
        conn.execute(
            "UPDATE figures SET attempts = attempts + 1, "
            "updated_at = datetime('now') WHERE figure_id = ?",
            (figure_id,),
        )


def _apply_batch(conn, rows, result) -> str | None:
    """Write one batch's results. Returns 'budget' on BudgetExceeded."""
    batch_ids = {row["figure_id"] for row in rows}
    if result.error is not None:
        if isinstance(result.error, llm.BudgetExceeded):
            return "budget"
        for row in rows:
            _bump_attempts(conn, row["figure_id"], str(result.error))
        return "error"
    parsed = result.parsed or {}
    seen: set[str] = set()
    for item in parsed.get("results") or []:
        figure_id = item.get("figure_id")
        if figure_id not in batch_ids or figure_id in seen:
            continue
        route = item.get("route")
        if route == "drop" or item.get("third_party"):
            status = "caption_rejected"
        else:
            status = _ROUTE_STATUS.get(route)
        if status is None:
            continue  # unknown route -> counts as missing
        seen.add(figure_id)
        db.set_status(
            conn, "figures", figure_id, status, triage_json=db.to_json(item)
        )
    for row in rows:
        if row["figure_id"] not in seen:
            _bump_attempts(conn, row["figure_id"])
    return None


def _print_summary(conn) -> None:
    status_counts = {
        r["status"]: r["n"]
        for r in conn.execute(
            "SELECT status, COUNT(*) AS n FROM figures GROUP BY status"
        )
    }
    print(f"figures by status: {status_counts}")
    reasons: dict[str, int] = {}
    for row in conn.execute(
        "SELECT triage_json FROM figures WHERE status = 'caption_rejected'"
    ):
        triage = db.from_json(row["triage_json"], {}) or {}
        reason = triage.get("reason") or "unknown"
        reasons[reason] = reasons.get(reason, 0) + 1
    print(f"caption rejections by reason: {reasons}")


def run(args) -> int:
    conn = db.init_db()
    disease = None if args.disease == "all" else args.disease
    if disease is not None and disease not in diseases.DISEASE_KEYS:
        print(f"unknown disease {disease!r}")
        return 2
    rows = db.rows_with_status(
        conn, "figures", "pending", disease=disease, limit=getattr(args, "limit", None)
    )
    pmcid_filter = getattr(args, "pmcids", None)
    if pmcid_filter:
        allowed = set(pmcid_filter)
        rows = [r for r in rows if r["pmcid"] in allowed]
    batches = [rows[i : i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]

    if args.dry_run:
        print(
            f"dry-run: {len(rows)} pending figures in {len(batches)} batch(es) "
            f"of <= {BATCH_SIZE}"
        )
        conn.close()
        return 0

    if not rows:
        print("no pending figures")
        _print_summary(conn)
        conn.close()
        return 0

    client = llm.LLMClient(db_conn=conn, budget_usd=args.budget_usd)
    print(f"triaging {len(rows)} figures in {len(batches)} batch(es)")
    results = client.call_many(_p2_request(b) for b in batches)

    budget_hit = False
    for batch_rows, result in zip(batches, results):
        outcome = _apply_batch(conn, batch_rows, result)
        if outcome == "budget":
            budget_hit = True
        conn.commit()

    if budget_hit:
        print(
            f"LLM budget exhausted (${client.spent_usd:.4f} spent); "
            "remaining figures left pending. Rerun to resume."
        )
    _print_summary(conn)
    print(f"live LLM spend this run: ${client.spent_usd:.4f}")
    conn.close()
    return 0
