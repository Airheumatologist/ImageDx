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
import re

from . import config, db, diseases, llm, pmc
from .prompts import P2

logger = logging.getLogger(__name__)

BATCH_SIZE = config.VP_TRIAGE_BATCH
CAPTION_MAX_CHARS = 1500

_ROUTE_STATUS = {
    "keep": "caption_kept",
    "uncertain": "caption_uncertain",
}
_IMAGE_CUES = (
    "photo", "photograph", "clinical image", "radiograph", "x-ray", "mri",
    "magnetic resonance", "ct scan", "computed tomography", "ultrasound",
    "sonogram", "histology", "histological", "biopsy", "h&e", "stain",
)
_PATIENT_IMAGE_CATEGORIES = {
    "clinical_photo", "dermoscopy", "capillaroscopy", "histology",
    "immunofluorescence", "radiology", "ultrasound", "echo",
    "endoscopy", "ophthalmic", "gross", "mixed",
}
_OTHER_DISEASE_CUES = (
    "systemic sclerosis", "rheumatoid arthritis", "psoriatic arthritis",
    "polymyositis", "inclusion body myositis", "healthy control",
)


def _contains_any(text: str, terms) -> bool:
    low = (text or "").casefold()
    return any(
        term and re.search(
            r"(?<![a-z0-9])" + re.escape(str(term).casefold()) + r"(?![a-z0-9])",
            low,
        )
        for term in terms
    )


def _explicit_target_visual(conn, row, image_category=None) -> bool:
    """True when the figure text names a parent disease/finding and image type."""
    article = conn.execute(
        "SELECT primary_disease_keys_json FROM articles WHERE pmcid=?",
        (row["pmcid"],),
    ).fetchone()
    keys = db.from_json(article["primary_disease_keys_json"], []) if article else []
    # Use the figure caption for disease attribution. Broad in-text mentions
    # can discuss another condition even when the parent article covers SLE/DM.
    text = row.get("caption") or ""
    image = _contains_any(text, _IMAGE_CUES) or image_category in _PATIENT_IMAGE_CATEGORIES
    if not image:
        return False
    disease_data = diseases.load_diseases()
    disease_match = any(
        _contains_any(text, [disease_data.get(key, {}).get("name", ""), *disease_data.get(key, {}).get("synonyms", [])])
        for key in keys
    )
    if disease_match:
        return True
    if _contains_any(text, _OTHER_DISEASE_CUES):
        return False
    for vocab in conn.execute(
        "SELECT disease_keys_json,label,synonyms_json FROM findings_vocab WHERE approved=1"
    ):
        if not keys or not (set(db.from_json(vocab["disease_keys_json"], [])) & set(keys)):
            continue
        terms = [vocab["label"], *db.from_json(vocab["synonyms_json"], [])]
        if _contains_any(text, terms):
            return True
    return False


def _contradictory_drop(conn, row, item) -> bool:
    if item.get("route") != "drop":
        return False
    # Some historical P2 batches set reason='keep' on every drop, including
    # flowcharts and mechanisms. Require independent patient-image evidence.
    return bool(
        item.get("is_real_patient_image") is True
        and _explicit_target_visual(conn, row, item.get("category"))
    )


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
    rows_by_id = {row["figure_id"]: dict(row) for row in rows}
    for item in parsed.get("results") or []:
        figure_id = item.get("figure_id")
        if figure_id not in batch_ids or figure_id in seen:
            continue
        route = item.get("route")
        # Third-party material is a hard exclusion even when other P2 fields
        # conflict. Ambiguous internal classifications are retained for P3.
        if item.get("third_party"):
            status = "caption_rejected"
        elif route == "drop" and _contradictory_drop(conn, rows_by_id[figure_id], item):
            item = dict(item)
            item["route_adjustment"] = "uncertain_due_to_conflicting_patient_or_target_evidence"
            item["route"] = "uncertain"
            status = "caption_uncertain"
        elif route == "drop":
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


def revisit_conflicting_rejections(conn, disease=None, pmcids=None, dry_run=False) -> int:
    """Re-queue historical P2 contradictions for vision review, once only."""
    rows = db.rows_with_status(conn, "figures", "caption_rejected", disease=disease)
    wanted = set(pmcids) if pmcids else None
    count = 0
    for source in rows:
        row = dict(source)
        if wanted is not None and row["pmcid"] not in wanted:
            continue
        item = db.from_json(row.get("triage_json"), {}) or {}
        if item.get("source") == "parse" or item.get("third_party") or not row.get("image_url"):
            continue
        article = conn.execute(
            "SELECT license_code FROM articles WHERE pmcid=?", (row["pmcid"],)
        ).fetchone()
        license_code = row.get("effective_license") or (article["license_code"] if article else None)
        if pmc.license_allows(license_code) is None or not _contradictory_drop(conn, row, item):
            continue
        count += 1
        if dry_run:
            continue
        item = dict(item)
        item["route"] = "uncertain"
        item["route_adjustment"] = "uncertain_due_to_conflicting_patient_or_target_evidence"
        db.set_status(
            conn, "figures", row["figure_id"], "caption_uncertain",
            triage_json=db.to_json(item),
        )
    if count and not dry_run:
        conn.commit()
    return count


def run(args) -> int:
    conn = db.init_db()
    disease = None if args.disease == "all" else args.disease
    if disease is not None and disease not in diseases.DISEASE_KEYS:
        print(f"unknown disease {disease!r}")
        return 2
    revisited = revisit_conflicting_rejections(
        conn, disease=disease, pmcids=getattr(args, "pmcids", None),
        dry_run=bool(args.dry_run),
    )
    if revisited:
        verb = "would revisit" if args.dry_run else "re-queued"
        print(f"triage: {verb} {revisited} contradictory historical rejection(s)")
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

    # Failed caption batches remain pending for a later, smaller retry. Avoid
    # repeated full network timeouts inside a single batch attempt.
    client = llm.LLMClient(db_conn=conn, budget_usd=args.budget_usd, max_retries=0)
    print(f"triaging {len(rows)} figures in {len(batches)} batch(es)")
    results = client.call_many(_p2_request(b) for b in batches)

    budget_hit = False
    for batch_rows, result in zip(batches, results):
        if result.error is not None and not isinstance(result.error, llm.BudgetExceeded) and len(batch_rows) > 1:
            middle = len(batch_rows) // 2
            smaller_batches = (batch_rows[:middle], batch_rows[middle:])
            for smaller_rows, smaller_result in zip(
                smaller_batches,
                client.call_many(_p2_request(part) for part in smaller_batches),
            ):
                if _apply_batch(conn, smaller_rows, smaller_result) == "budget":
                    budget_hit = True
                conn.commit()
            continue
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
