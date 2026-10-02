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

from . import config, curation, db, diseases, llm, pmc
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
# W2 montage retriage: caption drops that mention a multi-panel/montage
# layout AND a patient-image modality return to pending under P2.v5, which
# judges a single-disease collage as one whole figure.
MONTAGE = re.compile(
    r"multi[- ]?panel|montage|collage|composite|compound|multiple panels|several panels",
    re.I,
)
_MONTAGE_MODALITY = re.compile(
    r"clinical photo|photograph|dermoscop|capillaroscop|histolog|histopath|"
    r"biopsy|immunohisto|immunofluoresc|cytolog|radiograph|x-ray|\bct\b|"
    r"computed tomograph|\bmri\b|magnetic resonance|ultrasound|sonograph|"
    r"echocardiog|\bpet\b|endoscop|ophthalm|fundus|stain|imaging",
    re.I,
)
_MONTAGE_CATEGORIES = {"diagram", "chart", "flowchart", "table", "illustration", "other"}
_KNOWN_OTHER_DISEASE_CUES = (
    "systemic sclerosis", "rheumatoid arthritis", "psoriatic arthritis",
    "polymyositis", "inclusion body myositis", "healthy control",
)


def _other_disease_cues() -> tuple[str, ...]:
    """Keep deterministic alternate-disease guards out of the target scope."""
    configured = " ".join(
        " ".join([data.get("name", ""), *data.get("synonyms", [])]).casefold()
        for data in diseases.load_diseases().values()
    )
    return tuple(
        cue for cue in _KNOWN_OTHER_DISEASE_CUES
        if cue not in configured
    )


_OTHER_DISEASE_CUES = _other_disease_cues()


def _cached_pattern(patterns: dict, term) -> re.Pattern:
    """Compile the word-boundary regex for ``term`` once per ``patterns`` dict."""
    key = str(term).casefold()
    pattern = patterns.get(key)
    if pattern is None:
        pattern = re.compile(
            r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])"
        )
        patterns[key] = pattern
    return pattern


def _contains_any(text: str, terms, patterns: dict | None = None) -> bool:
    low = (text or "").casefold()
    cache = patterns if patterns is not None else {}
    return any(term and _cached_pattern(cache, term).search(low) for term in terms)


class _RunContext:
    """Per-run routing caches for ``_explicit_target_visual``.

    Disease names/synonyms, approved ``findings_vocab`` rows, article disease
    keys and compiled term regexes are stable within a single ``run``/revisit
    pass, so they are loaded or compiled once here instead of once per figure
    row. Each property loads lazily at the same point the old code queried.
    """

    def __init__(self, conn) -> None:
        self._conn = conn
        self.patterns: dict[str, re.Pattern] = {}
        self._disease_terms: dict[str, list] | None = None
        self._vocab: list[tuple[frozenset, list]] | None = None
        self._article_keys: dict[str, list] = {}

    def contains_any(self, text: str, terms) -> bool:
        return _contains_any(text, terms, self.patterns)

    @property
    def disease_terms(self) -> dict[str, list]:
        """``diseases.json`` name+synonyms term lists, keyed by disease key."""
        if self._disease_terms is None:
            self._disease_terms = {
                key: [data.get("name", ""), *data.get("synonyms", [])]
                for key, data in diseases.load_diseases().items()
            }
        return self._disease_terms

    @property
    def vocab(self) -> list[tuple[frozenset, list]]:
        """Approved vocab rows as ``(disease keys, [label, *synonyms])``."""
        if self._vocab is None:
            self._vocab = [
                (
                    frozenset(db.from_json(row["disease_keys_json"], [])),
                    [row["label"], *db.from_json(row["synonyms_json"], [])],
                )
                for row in self._conn.execute(
                    "SELECT disease_keys_json,label,synonyms_json "
                    "FROM findings_vocab WHERE approved=1"
                )
            ]
        return self._vocab

    def article_keys(self, pmcid: str) -> list:
        """``articles.primary_disease_keys_json`` for ``pmcid``, memoized."""
        if pmcid not in self._article_keys:
            article = self._conn.execute(
                "SELECT primary_disease_keys_json FROM articles WHERE pmcid=?",
                (pmcid,),
            ).fetchone()
            self._article_keys[pmcid] = (
                db.from_json(article["primary_disease_keys_json"], [])
                if article
                else []
            )
        return self._article_keys[pmcid]


def _explicit_target_visual(conn, row, image_category=None, ctx=None) -> bool:
    """True when the figure text names a parent disease/finding and image type."""
    ctx = ctx or _RunContext(conn)
    keys = ctx.article_keys(row["pmcid"])
    # Use the figure caption for disease attribution. Broad in-text mentions
    # can discuss another condition even when the parent article covers SLE/DM.
    text = row.get("caption") or ""
    image = ctx.contains_any(text, _IMAGE_CUES) or image_category in _PATIENT_IMAGE_CATEGORIES
    if not image:
        return False
    disease_match = any(
        ctx.contains_any(text, ctx.disease_terms.get(key, [""]))
        for key in keys
    )
    if disease_match:
        return True
    if ctx.contains_any(text, _OTHER_DISEASE_CUES):
        return False
    for vocab_keys, terms in ctx.vocab:
        if not keys or not (vocab_keys & set(keys)):
            continue
        if ctx.contains_any(text, terms):
            return True
    return False


def _contradictory_drop(conn, row, item, ctx=None) -> bool:
    if item.get("route") != "drop":
        return False
    # Some historical P2 batches set reason='keep' on every drop, including
    # flowcharts and mechanisms. Require independent patient-image evidence.
    return bool(
        item.get("is_real_patient_image") is True
        and _explicit_target_visual(conn, row, item.get("category"), ctx=ctx)
    )


def _batch_payload(rows) -> str:
    figures = [
        {
            "figure_id": row["figure_id"],
            "label": row["label"],
            "caption": (row["caption"] or "")[:CAPTION_MAX_CHARS],
            "in_text_mentions": db.from_json(row["in_text_mentions_json"], []),
            "article_title": row.get("_article_title") or "",
            "article_disease_keys": row.get("_article_disease_keys") or [],
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


def _apply_batch(conn, rows, result, ctx=None) -> str | None:
    """Write one batch's results. Returns 'budget' on BudgetExceeded."""
    batch_ids = {row["figure_id"] for row in rows}
    if result.error is not None:
        if isinstance(result.error, llm.BudgetExceeded):
            return "budget"
        for row in rows:
            _bump_attempts(conn, row["figure_id"], str(result.error))
        return "error"
    ctx = ctx or _RunContext(conn)
    parsed = result.parsed or {}
    seen: set[str] = set()
    rows_by_id = {row["figure_id"]: dict(row) for row in rows}
    for item in parsed.get("results") or []:
        figure_id = item.get("figure_id")
        if figure_id not in batch_ids or figure_id in seen:
            continue
        route = item.get("route")
        row = rows_by_id[figure_id]
        deterministic_reason = curation.source_exclusion_reason(
            {**row, "triage_json": None}, {"title": row.get("_article_title")}
        )
        # Third-party material is a hard exclusion even when other P2 fields
        # conflict. Ambiguous internal classifications are retained for P3.
        if item.get("third_party"):
            status = "caption_rejected"
        elif deterministic_reason:
            item = dict(item)
            item["route_adjustment"] = f"deterministic_source_exclusion: {deterministic_reason}"
            item["route"] = "drop"
            item["reason"] = deterministic_reason
            status = "caption_rejected"
        elif route == "drop" and _contradictory_drop(conn, rows_by_id[figure_id], item, ctx):
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
        prior = db.from_json(rows_by_id[figure_id].get("triage_json"), {}) or {}
        if prior.get("montage_retriage") and not item.get("montage_retriage"):
            # Keep the W2 marker so a re-dropped montage is not requeued again.
            item = dict(item)
            item["montage_retriage"] = prior["montage_retriage"]
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


def requeue_montage_rejections(conn, disease=None, pmcids=None, dry_run=False) -> int:
    """Requeue caption_rejected montage/collage drops for the v5 whole-plate
    policy. A drop qualifies only when its reason names a multi-panel layout
    AND a patient-image modality (or the model still assigned a patient-image
    category). Rows already requeued for this P2 version are skipped via the
    ``montage_retriage`` marker kept in triage_json.
    """
    rows = db.rows_with_status(conn, "figures", "caption_rejected", disease=disease)
    wanted = set(pmcids) if pmcids else None
    count = 0
    for source in rows:
        row = dict(source)
        if wanted is not None and row["pmcid"] not in wanted:
            continue
        item = db.from_json(row.get("triage_json"), {}) or {}
        if (
            item.get("source") == "parse"
            or item.get("third_party")
            or not row.get("image_url")
            or item.get("is_real_patient_image") is False
            or str(item.get("route_adjustment") or "").startswith("deterministic_source_exclusion")
            or item.get("category") in _MONTAGE_CATEGORIES
            or item.get("montage_retriage") == P2.version
        ):
            continue
        reason = str(item.get("reason") or "")
        if not MONTAGE.search(reason):
            continue
        if not (
            _MONTAGE_MODALITY.search(reason)
            or item.get("category") in _PATIENT_IMAGE_CATEGORIES
        ):
            continue
        article = conn.execute(
            "SELECT license_code FROM articles WHERE pmcid=?", (row["pmcid"],)
        ).fetchone()
        license_code = row.get("effective_license") or (
            article["license_code"] if article else None
        )
        if pmc.license_allows(license_code) is None:
            continue
        count += 1
        if dry_run:
            continue
        db.set_status(
            conn, "figures", row["figure_id"], "pending",
            attempts=0,
            triage_json=db.to_json(
                {"montage_retriage": P2.version, "previous": item}
            ),
        )
    if count and not dry_run:
        conn.commit()
    return count


def revisit_conflicting_rejections(conn, disease=None, pmcids=None, dry_run=False, ctx=None) -> int:
    """Re-queue historical P2 contradictions for vision review, once only."""
    ctx = ctx or _RunContext(conn)
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
        if pmc.license_allows(license_code) is None or not _contradictory_drop(conn, row, item, ctx):
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
    # One routing context per run: vocab rows, disease terms, article keys and
    # compiled regexes are stable for the whole pass.
    ctx = _RunContext(conn)
    disease = None if args.disease == "all" else args.disease
    if disease is not None and disease not in diseases.DISEASE_KEYS:
        print(f"unknown disease {disease!r}")
        return 2
    if getattr(args, "retriage_montages", False):
        montages = requeue_montage_rejections(
            conn, disease=disease, pmcids=getattr(args, "pmcids", None),
            dry_run=bool(args.dry_run),
        )
        verb = "would requeue" if args.dry_run else "requeued"
        print(f"triage: {verb} {montages} montage caption rejection(s)")
    revisited = revisit_conflicting_rejections(
        conn, disease=disease, pmcids=getattr(args, "pmcids", None),
        dry_run=bool(args.dry_run), ctx=ctx,
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
    # Give P2 the parent article context too: captions alone can be terse, but
    # a broad article topic remains context only and cannot make a figure pass.
    contextual_rows = []
    for source in rows:
        row = dict(source)
        article = conn.execute(
            "SELECT title,primary_disease_keys_json FROM articles WHERE pmcid=?",
            (row["pmcid"],),
        ).fetchone()
        if article:
            row["_article_title"] = article["title"] or ""
            row["_article_disease_keys"] = db.from_json(
                article["primary_disease_keys_json"], []
            )
        contextual_rows.append(row)
    rows = contextual_rows
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
                if _apply_batch(conn, smaller_rows, smaller_result, ctx) == "budget":
                    budget_hit = True
                conn.commit()
            continue
        outcome = _apply_batch(conn, batch_rows, result, ctx)
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
