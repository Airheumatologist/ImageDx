"""Stage 1: figure-first discovery from Europe PMC, in three passes.

Broad sources come first so the library shows how a disease usually looks;
single case reports only fill what they leave short:

1. ``overview``: per disease, narrative reviews and case series whose title
   surveys its presentation ("Psoriatic arthritis: clinical manifestations").
   One article can picture many findings.
2. ``manifestation``: per under-target pair, narrative reviews and case
   series about the finding ("dactylitis in psoriatic arthritis") with a
   figure caption naming it.
3. ``backfill``: per pair still under target, any article type with a
   caption naming the finding (case reports included); only notices such as
   errata and retractions are dropped.

The first two passes skip atypical sources (drug-induced, treatment stories,
rarities; ``source_quality.article_tier``), which stay available to
backfill. Within a pass, hits are taken reviews first, then case series and
original studies, then case reports, atypical last. ``run-all`` stores and
judges each pass before the next one plans, so later passes see the coverage
earlier ones earned.

Matching articles are fetched from the PMC open-data bucket and parsed
straight into ``figures``: a figure whose caption names an approved finding
of the disease becomes ``pending`` for triage, every other figure lands as
``caption_rejected`` without an LLM call. Articles land at ``parsed`` so triage → judge → store run unchanged.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import logging
import re
import time

from . import config, db, diseases, europepmc, gallery, jats, pair_reporting, pair_terms, pmc, source_quality
from . import parse as parse_stage

logger = logging.getLogger(__name__)

POLICY_VERSION = pair_reporting.DISCOVERY_POLICY
DEFAULT_PER_PAIR = 25
SEARCH_DEPTH = 300
PASSES = ("overview", "manifestation", "backfill")
# Search attempts of the per-disease overview pass are logged under this key.
OVERVIEW_KEY = "_overview"
OVERVIEW_PER_DISEASE = 60


def _vocab(conn) -> list[dict]:
    rows = conn.execute("SELECT * FROM findings_vocab WHERE approved=1").fetchall()
    return [
        {
            "finding_key": r["finding_key"], "label": r["label"], "category": r["category"],
            "synonyms": db.from_json(r["synonyms_json"], []) or [],
            "disease_keys": db.from_json(r["disease_keys_json"], []) or [],
        }
        for r in rows
    ]


def disease_query_terms(disease: dict) -> list[str]:
    terms = pair_terms.disease_terms(disease) + [
        s for s in disease.get("synonyms") or [] if len(s) > 4
    ]
    return list(dict.fromkeys(terms))


def _vocab_by_disease(conn) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for finding in _vocab(conn):
        for disease_key in finding["disease_keys"]:
            out.setdefault(disease_key, []).append(finding)
    return out


def plan_pairs(conn, disease_keys, target: int) -> list[tuple[str, dict, int]]:
    """(disease, finding, current count) under target, fewest images first."""
    vocab = _vocab_by_disease(conn)
    out = []
    for disease_key in disease_keys:
        findings = vocab.get(disease_key) or []
        if not findings:
            continue
        counts = gallery.published_coverage(conn, disease_key)
        for finding in findings:
            have = counts.get(finding["finding_key"], 0)
            if have < target:
                out.append((disease_key, finding, have))
    out.sort(key=lambda item: (item[2], item[0], item[1]["finding_key"]))
    return out


def _known_by_disease(conn) -> dict[str, set[str]]:
    """Parsed articles per disease, read once per pass (see ``_known_pmcids``)."""
    out: dict[str, set[str]] = {}
    for row in conn.execute(
        "SELECT pmcid, primary_disease_keys_json FROM articles WHERE status='parsed'"
    ):
        for key in db.from_json(row["primary_disease_keys_json"], []) or []:
            out.setdefault(key, set()).add(row["pmcid"])
    return out


def _known_pmcids(conn, disease_key: str) -> set[str]:
    """Articles already parsed for this disease (their figures are queued or done)."""
    out = set()
    for row in conn.execute(
        "SELECT pmcid, primary_disease_keys_json FROM articles WHERE status='parsed'"
    ):
        if disease_key in (db.from_json(row["primary_disease_keys_json"], []) or []):
            out.add(row["pmcid"])
    return out


def hit_tier(hit: europepmc.Hit) -> int:
    return source_quality.article_tier(hit.pub_types, hit.title, hit.abstract)


def _take(hits, seen: set[str], limit: int, picked: list, mode: str, *, broad_only: bool) -> list[str]:
    """Add eligible unseen hits to ``picked`` best tier first; return their PMCIDs.

    Europe PMC relevance order is kept within each tier. ``broad_only`` skips
    atypical sources, leaving them to the backfill pass.
    """
    new = []
    for hit in sorted(hits, key=hit_tier):
        if len(picked) >= limit:
            break
        if hit.pmcid in seen or not europepmc.eligible(hit):
            continue
        if broad_only and hit_tier(hit) == source_quality.TIER_ATYPICAL:
            continue
        seen.add(hit.pmcid)
        new.append(hit.pmcid)
        picked.append((hit, mode))
    return new


def find_articles(disease_key: str, finding: dict, per_pair: int, skip: set[str],
                  scope: str = "all"):
    """Phrase tier first, then same-caption words tier, until ``per_pair`` new hits.

    ``scope="reviews"`` is the manifestation pass: narrative reviews and case
    series about the finding, atypical sources skipped.
    """
    disease = diseases.load_diseases()[disease_key]
    terms = pair_terms.caption_terms(disease_key, finding, disease)
    dterms = disease_query_terms(disease)
    picked: list[tuple[europepmc.Hit, str]] = []
    seen = set(skip)
    attempts = []
    for mode in ("phrase", "words"):
        if len(picked) >= per_pair:
            break
        query = europepmc.build_query(terms, dterms, mode, scope=scope)
        if not query:
            continue
        total, hits = europepmc.search(query, limit=SEARCH_DEPTH)
        new = _take(hits, seen, per_pair, picked, mode, broad_only=scope == "reviews")
        attempts.append({"mode": mode, "scope": scope, "query": query, "total": total,
                         "returned": [h.pmcid for h in hits], "new": new})
    return terms, picked, attempts


_ABBREVIATION = re.compile(r"[A-Z]{2}|[a-z][A-Z]")


def overview_disease_terms(disease: dict) -> list[str]:
    """Full disease names only: in a title, "PsA" also means prostate-specific antigen."""
    terms = [t for t in disease_query_terms(disease) if not _ABBREVIATION.search(t)]
    return list({t.lower(): t for t in terms}.values())


def find_overview_articles(disease_key: str, caption_terms: list[str], limit: int, skip: set[str]):
    """Narrative reviews and case series surveying the disease's presentation."""
    query = europepmc.build_overview_query(
        overview_disease_terms(diseases.load_diseases()[disease_key]), caption_terms,
    )
    if not query:
        return [], []
    total, hits = europepmc.search(query, limit=SEARCH_DEPTH)
    picked: list[tuple[europepmc.Hit, str]] = []
    new = _take(hits, set(skip), limit, picked, "overview", broad_only=True)
    return picked, [{"mode": "overview", "scope": "overview", "query": query, "total": total,
                     "returned": [h.pmcid for h in hits], "new": new}]


def _record_attempts(conn, disease_key, finding_key, attempts) -> None:
    round_no = conn.execute(
        "SELECT COALESCE(MAX(round_no), 0) + 1 FROM pair_search_attempts "
        "WHERE disease_key=? AND finding_key=? AND policy_version=?",
        (disease_key, finding_key, POLICY_VERSION),
    ).fetchone()[0]
    for item in attempts:
        conn.execute(
            "INSERT OR IGNORE INTO pair_search_attempts (disease_key, finding_key, "
            "policy_version, round_no, query, query_filter_hash, filters_json, depth, "
            "status, returned_pmcids_json, new_pmcids_json, completed_at) "
            "VALUES (?,?,?,?,?,?,?,?, 'completed', ?, ?, datetime('now'))",
            (
                disease_key, finding_key, POLICY_VERSION, round_no, item["query"],
                hashlib.sha256(item["query"].encode()).hexdigest(),
                db.to_json({"mode": item["mode"], "scope": item.get("scope", "all"),
                            "hit_count": item["total"]}),
                SEARCH_DEPTH, db.to_json(item["returned"]), db.to_json(item["new"]),
            ),
        )


def _fetch(pmcid: str):
    try:
        bundle = pmc.get_article_bundle(pmcid)
        return bundle, jats.parse_article(bundle.xml_text)
    except Exception as exc:  # noqa: BLE001 - one bad article must not stop the run
        return exc


def _upsert_article(conn, hit: europepmc.Hit, disease_key: str, finding_key: str, mode: str,
                    pass_name: str = "backfill") -> dict:
    row = conn.execute("SELECT * FROM articles WHERE pmcid=?", (hit.pmcid,)).fetchone()
    evidence = {"source": "europepmc_fig", "disease": disease_key, "finding": finding_key,
                "mode": mode, "pass": pass_name}
    if row is None:
        conn.execute(
            "INSERT INTO articles (pmcid, pmid, doi, title, journal, year, "
            "publication_types_json, license_code, license_url, oa_subset, "
            "retrieval_evidence_json, primary_disease_keys_json, relevance_decision, "
            "relevance_reason, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'candidate')",
            (
                hit.pmcid, hit.pmid, hit.doi, hit.title, hit.journal, hit.year,
                db.to_json(hit.pub_types), hit.license_code,
                pmc.license_url_for(hit.license_code, hit.license_raw), "oa",
                db.to_json([evidence]), db.to_json([disease_key]), "relevant",
                "europepmc caption match",
            ),
        )
    else:
        keys = db.from_json(row["primary_disease_keys_json"], []) or []
        evidence_list = db.from_json(row["retrieval_evidence_json"], []) or []
        conn.execute(
            "UPDATE articles SET primary_disease_keys_json=?, retrieval_evidence_json=?, "
            "relevance_decision='relevant', relevance_reason=COALESCE(relevance_reason, ?), "
            "license_code=COALESCE(license_code, ?), publication_types_json=?, "
            "updated_at=datetime('now') WHERE pmcid=?",
            (
                db.to_json(sorted({*keys, disease_key})), db.to_json([*evidence_list, evidence]),
                "europepmc caption match", hit.license_code, db.to_json(hit.pub_types), hit.pmcid,
            ),
        )
    conn.execute(
        "INSERT INTO article_source_metadata(pmcid,source,metadata_json,fetched_at) "
        "VALUES(?,?,?,datetime('now')) ON CONFLICT(pmcid) DO UPDATE SET "
        "source=excluded.source,metadata_json=excluded.metadata_json,fetched_at=excluded.fetched_at",
        (hit.pmcid, "europe_pmc_core", db.to_json(source_quality.normalize_record(hit.raw))),
    )
    return dict(conn.execute("SELECT * FROM articles WHERE pmcid=?", (hit.pmcid,)).fetchone())


def disease_caption_terms(conn, disease_key: str, vocab: list[dict] | None = None) -> list[str]:
    """Caption terms of every approved finding of the disease.

    ``vocab`` is the disease's findings when the caller already grouped them
    (``_vocab_by_disease``); otherwise they are read here.
    """
    disease = diseases.load_diseases()[disease_key]
    if vocab is None:
        vocab = [f for f in _vocab(conn) if disease_key in f["disease_keys"]]
    terms: list[str] = []
    for finding in vocab:
        terms += pair_terms.caption_terms(disease_key, finding, disease)
    return list(dict.fromkeys(terms))


_CASE_TYPE = re.compile(r"case", re.I)
_CASE_ABSTRACT = re.compile(r"\b(?:we|here we|this report)\s+(?:report|present|describe)", re.I)
_CASE_SECTION = re.compile(
    r"^\s*case\b|case (?:report|presentation|description|history|summary)"
    r"|clinical (?:case|presentation)|patient presentation", re.I,
)
_CASE_SECTION_CHARS = 1500


def case_age_text(hit: europepmc.Hit, parsed: jats.ParsedArticle | None = None) -> str | None:
    """Patient-describing text for age evidence, from case reports only.

    Abstract plus the case-presentation sections, used only when the article
    is typed as a case (Case Reports, case-report, case-study) or its abstract
    says it reports a case. Cohort studies return None: a study-wide age
    range is not evidence for one pictured patient.
    """
    abstract = " ".join(re.sub(r"<[^>]+>", " ", hit.abstract or "").split())
    if not (any(_CASE_TYPE.search(t) for t in hit.pub_types) or _CASE_ABSTRACT.search(abstract)):
        return None
    parts = [abstract] if abstract else []
    for title, text in (parsed.body_sections if parsed else []):
        if _CASE_SECTION.search(title or ""):
            parts.append(" ".join(str(text).split())[:_CASE_SECTION_CHARS])
    return " ".join(parts) or None


def _store_article(conn, article_row, bundle, parsed, match_terms, stats,
                   age_text: str | None = None) -> list[str]:
    pmcid = article_row["pmcid"]
    fig_rows = []
    pending = []
    for fig in parsed.figures:
        row = parse_stage._figure_row(pmcid, fig, article_row["license_code"], bundle.resolver)
        if row["status"] == "pending" and not pair_terms.caption_matches(
            fig.caption or "", match_terms, mode="words"
        ):
            row["status"] = "caption_rejected"
            row["triage_json"] = db.to_json(
                {"route": "drop", "reason": "no_finding_term", "source": "discover"}
            )
        parse_stage._insert_figure(conn, row)
        if age_text:
            conn.execute(
                "UPDATE figures SET case_age_text=? WHERE figure_id=? AND case_age_text IS NULL",
                (age_text, row["figure_id"]),
            )
        fig_rows.append(row)
        stats[row["status"]] = stats.get(row["status"], 0) + 1
        if row["status"] == "pending":
            pending.append(row["figure_id"])
    db.set_status(
        conn, "articles", pmcid, "parsed",
        study_region=parse_stage.study_region_for(article_row, parsed),
        authors_json=db.to_json(parsed.authors),
        author_count=parsed.author_count,
        journal_name=parsed.journal_name,
        **parse_stage._c6_article_fields(article_row, bundle, fig_rows),
    )
    parse_stage._sections_put(pmcid, parsed.body_sections)
    return pending


def _ingest(conn, pool, disease_key: str, finding_key: str, picked, pass_name: str,
            match_terms: list[str], stats: dict) -> int:
    """Fetch and parse picked hits; return the number of figures queued."""
    outcomes = list(pool.map(_fetch, [hit.pmcid for hit, _ in picked]))
    pending = 0
    for (hit, mode), outcome in zip(picked, outcomes):
        article = _upsert_article(conn, hit, disease_key, finding_key, mode, pass_name)
        if isinstance(outcome, Exception):
            db.set_status(conn, "articles", hit.pmcid, "parse_error", error=str(outcome))
            stats["fetch_errors"] += 1
            continue
        bundle, parsed = outcome
        pending += len(_store_article(conn, article, bundle, parsed, match_terms, stats,
                                      case_age_text(hit, parsed)))
        stats["pmcids"].append(hit.pmcid)
    conn.commit()
    stats["articles"] += len(picked)
    return pending


def _ahead(pool, jobs, fn, window: int, prepare=lambda job: job):
    """Yield ``(job, result)`` in job order with up to ``window`` calls in flight.

    ``fn(prepare(job))`` runs on ``pool``; an exception is yielded as the
    result. ``prepare`` runs on the consuming thread at submit time, so it
    can snapshot state the consumer keeps updating between results.
    """
    pending: deque = deque()
    jobs = iter(jobs)

    def fill() -> None:
        while len(pending) < window:
            job = next(jobs, None)
            if job is None:
                return
            pending.append((job, pool.submit(fn, prepare(job))))

    fill()
    try:
        while pending:
            job, future = pending.popleft()
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - handed to the consumer
                result = exc
            yield job, result
            fill()
    finally:
        for _, future in pending:
            future.cancel()


def discover(conn, disease_keys, *, per_pair: int, target: int, max_pairs: int | None = None,
             max_runtime: float | None = None, pass_name: str = "backfill", log=print,
             on_articles=None, should_stop=None) -> dict:
    """Search, fetch and parse one pass for under-target pairs; returns run stats.

    Europe PMC searches run ahead of ingestion on ``VP_SEARCH_CONCURRENCY``
    workers while fetching, parsing and every DB write stay on this thread in
    plan order. A search sees the articles known when it was submitted, so
    hits another pair took meanwhile are dropped at ingest. ``on_articles``
    receives each pair's newly parsed PMCIDs as soon as they are stored, so
    ``run-all`` can triage them while discovery continues; ``should_stop``
    ends the pass early. Searches lost to Europe PMC errors are retried once
    after ``VP_SEARCH_RETRY_COOLDOWN`` seconds at the end of the pass.
    """
    if pass_name not in PASSES:
        raise ValueError(f"unknown discovery pass {pass_name!r}")
    started = time.monotonic()
    stats: dict = {"pass": pass_name, "pairs": 0, "articles": 0, "fetch_errors": 0, "pmcids": []}
    vocab = _vocab_by_disease(conn)
    match_terms: dict[str, list[str]] = {}

    def terms_for(disease_key: str) -> list[str]:
        if disease_key not in match_terms:
            match_terms[disease_key] = disease_caption_terms(
                conn, disease_key, vocab.get(disease_key) or []
            )
        return match_terms[disease_key]

    plan = plan_pairs(conn, disease_keys, target)
    if max_pairs is not None:
        plan = plan[:max_pairs]
    log(f"discover[{pass_name}]: {len(plan)} pair(s) under target {target}")
    known = _known_by_disease(conn)

    def out_of_time() -> bool:
        if should_stop is not None and should_stop():
            log(f"discover[{pass_name}]: stopping early")
            return True
        if max_runtime is not None and time.monotonic() - started >= max_runtime:
            log(f"discover[{pass_name}]: runtime limit reached")
            return True
        return False

    overview = pass_name == "overview"
    scope = "reviews" if pass_name == "manifestation" else "all"
    if overview:
        jobs = [(d, None, 0) for d in dict.fromkeys(d for d, _, _ in plan)]
    else:
        jobs = plan

    def prepare(job):
        # Snapshot on this thread: ingest keeps adding to ``known``.
        disease_key = job[0]
        terms = terms_for(disease_key) if overview else None
        return job, set(known.get(disease_key, ())), terms

    def search(prepared):
        (disease_key, finding, _), skip, terms = prepared
        if overview:
            return find_overview_articles(disease_key, terms, OVERVIEW_PER_DISEASE, skip)
        _, picked, attempts = find_articles(disease_key, finding, per_pair, skip, scope=scope)
        return picked, attempts

    def label(job) -> str:
        disease_key, finding, _ = job
        return disease_key if overview else f"{disease_key}/{finding['finding_key']}"

    def ingest(job, result) -> None:
        disease_key, finding, have = job
        picked, attempts = result
        mine = known.setdefault(disease_key, set())
        # Another pair may have ingested a hit since this search was submitted.
        picked = [(hit, mode) for hit, mode in picked if hit.pmcid not in mine]
        key = OVERVIEW_KEY if overview else finding["finding_key"]
        _record_attempts(conn, disease_key, key, attempts)
        before = len(stats["pmcids"])
        pending = _ingest(conn, pool, disease_key, key, picked, pass_name,
                          terms_for(disease_key), stats)
        new = stats["pmcids"][before:]
        mine.update(new)
        if not overview:
            stats["pairs"] += 1
        totals = "/".join(str(a["total"]) for a in attempts)
        have_text = "" if overview else f" (have {have})"
        log(f"discover[{pass_name}]: {label(job)}{have_text}: hits {totals}, "
            f"{len(picked)} new article(s), {pending} figure(s) queued")
        if new and on_articles is not None:
            on_articles(new)

    def run_jobs(job_list) -> list:
        failed = []
        for job, result in _ahead(search_pool, job_list, search, window, prepare):
            if isinstance(result, pmc.PmcError):
                log(f"discover[{pass_name}]: {label(job)}: search failed: {result}")
                failed.append(job)
                continue
            if isinstance(result, Exception):
                raise result
            ingest(job, result)
            if out_of_time():
                return []
        return failed

    window = 4 * config.VP_SEARCH_CONCURRENCY
    with ThreadPoolExecutor(config.VP_FETCH_CONCURRENCY) as pool, \
            ThreadPoolExecutor(config.VP_SEARCH_CONCURRENCY) as search_pool:
        if out_of_time():
            return stats
        failed = run_jobs(jobs)
        if failed:
            log(f"discover[{pass_name}]: retrying {len(failed)} failed search(es) "
                f"in {config.VP_SEARCH_RETRY_COOLDOWN}s")
            deadline = time.monotonic() + config.VP_SEARCH_RETRY_COOLDOWN
            while time.monotonic() < deadline:
                if out_of_time():
                    return stats
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
            still = run_jobs(failed)
            stats["search_failures"] = len(still)
            if still:
                log(f"discover[{pass_name}]: {len(still)} search(es) failed twice: "
                    + ", ".join(label(job) for job in still[:20]))
    return stats


FIXED_KEY = "_fixed"


def article_disease_keys(hit: europepmc.Hit, disease_keys) -> list[str]:
    """Diseases among ``disease_keys`` named in the article's title or abstract."""
    text = f"{hit.title} {hit.abstract}".lower()
    catalog = diseases.load_diseases()
    return [
        key for key in disease_keys
        if any(term.lower() in text for term in disease_query_terms(catalog[key]))
    ]


def ingest_pmcids(conn, pmcids, disease_keys, log=print) -> dict:
    """Fetch and parse an explicit article list, skipping search; returns run stats.

    Each article is attributed to the diseases its title or abstract names
    (or to the one requested disease when it names none); ineligible and
    already-parsed articles are skipped.
    """
    stats: dict = {"pass": "fixed", "pairs": 0, "articles": 0, "fetch_errors": 0, "pmcids": []}
    match_terms = {d: disease_caption_terms(conn, d) for d in disease_keys}
    parsed = {r["pmcid"] for r in conn.execute("SELECT pmcid FROM articles WHERE status='parsed'")}
    with ThreadPoolExecutor(config.VP_FETCH_CONCURRENCY) as pool:
        for pmcid in dict.fromkeys(pmcids):
            if pmcid in parsed:
                log(f"discover[fixed]: {pmcid}: already parsed")
                continue
            _, hits = europepmc.search(f"PMCID:{pmcid}", limit=1)
            hit = hits[0] if hits else None
            if hit is None or not europepmc.eligible(hit):
                log(f"discover[fixed]: {pmcid}: not found or not eligible")
                continue
            # A single --disease is trusted for articles whose text names none.
            keys = article_disease_keys(hit, disease_keys) or (
                list(disease_keys) if len(disease_keys) == 1 else []
            )
            if not keys:
                log(f"discover[fixed]: {pmcid}: names no configured disease")
                continue
            for key in keys[1:]:
                _upsert_article(conn, hit, key, FIXED_KEY, "fixed", "fixed")
            terms = list(dict.fromkeys(t for k in keys for t in match_terms[k]))
            pending = _ingest(conn, pool, keys[0], FIXED_KEY, [(hit, "fixed")], "fixed",
                              terms, stats)
            log(f"discover[fixed]: {pmcid} ({'+'.join(keys)}): {pending} figure(s) queued")
    return stats


def run(args) -> int:
    conn = db.init_db()
    disease_keys = list(diseases.DISEASE_KEYS) if args.disease == "all" else [args.disease]
    if getattr(args, "pmcids", None):
        stats = ingest_pmcids(conn, args.pmcids, disease_keys)
        print(f"discover: {stats['articles']} article(s), "
              f"{stats.get('pending', 0)} figure(s) pending, "
              f"{stats.get('caption_rejected', 0)} rejected, {stats['fetch_errors']} fetch error(s)")
        conn.close()
        return 0
    target = int(config.VP_FINDING_IMAGE_TARGET)
    per_pair = getattr(args, "per_pair", None) or DEFAULT_PER_PAIR
    if args.dry_run:
        for disease_key, finding, have in plan_pairs(conn, disease_keys, target):
            print(f"{disease_key}/{finding['finding_key']}: have {have}")
        return 0
    stats = discover(conn, disease_keys, per_pair=per_pair, target=target,
                     max_pairs=getattr(args, "limit", None),
                     pass_name=getattr(args, "discovery_pass", None) or "backfill")
    print(f"discover: {stats['pairs']} pair(s), {stats['articles']} article(s), "
          f"{stats.get('pending', 0)} figure(s) pending, "
          f"{stats.get('caption_rejected', 0)} rejected, {stats['fetch_errors']} fetch error(s)")
    conn.close()
    return 0
