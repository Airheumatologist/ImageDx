"""Stage 7: P4 text extraction of disease->finding statements.

For each ``parsed`` article the JATS body sections come from the in-process
parse cache (``parse.sections_for``) when available, else a hinted in-memory
refetch (``articles.s3_prefix``/``media_files_json`` skip the S3 listing);
body sections whose headings match clinical keywords go to P4 (whole body as
fallback, ~120k char cap). Assertions are post-validated: non-pilot
diseases dropped, unknown finding keys moved to ``proposed_finding``, and
quotes that do not appear verbatim in the source text (after whitespace,
quote-char and dash normalization) or exceed 40 words are dropped and
counted as ``unverified_quote``.

Rows land in ``disease_findings`` (source='text') for both mapped keys and
proposed terms (stored under their snake_case key). Rows are
deleted+reinserted per article so reruns are idempotent; articles that
already have source='text' rows are skipped unless --force. Proposed
terms upsert findings_vocab with the same rules as store, and are not
re-counted when the P4 response was served from the llm_calls cache.
Panels' stored findings are then rebuilt as source='image' rows
(quote = panel evidence), also delete+reinsert.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from . import config, db, diseases, jats, llm, parse, pmc, timing
from .prompts import P4
from .store import slugify, upsert_proposed

PILOT_KEYS = set(diseases.DISEASE_KEYS)
MAX_SOURCE_CHARS = 120_000
MAX_QUOTE_WORDS = 40

SECTION_KEYWORDS = (
    "clinic",
    "feature",
    "manifest",
    "presentation",
    "diagnos",
    "imaging",
    "radiolog",
    "mri",
    "ultrasound",
    "histo",
    "patholog",
    "biops",
    "classif",
    "criteria",
    "cutaneous",
    "skin",
    "capillaroscop",
    "sign",
    "finding",
    "muscle",
    "ocular",
    "eye",
)


# ---------------------------------------------------------------------------
# Section picking + quote verification (unit-tested)
# ---------------------------------------------------------------------------
def pick_sections(sections: list[tuple[str, str]], max_chars: int = MAX_SOURCE_CHARS) -> list[tuple[str, str]]:
    """Body sections whose heading matches a clinical keyword (else all)."""
    matched = [
        (title, text)
        for title, text in sections
        if title and any(k in title.lower() for k in SECTION_KEYWORDS)
    ]
    chosen = matched or sections
    out: list[tuple[str, str]] = []
    total = 0
    for title, text in chosen:
        if not text.strip():
            continue
        if total >= max_chars:
            break
        room = max_chars - total
        out.append((title, text[:room]))
        total += len(text)
    return out


# Bracketed/parenthesized numeric citations: [15], [79, 82, 83], [3-5], (15).
_CITATION_RE = re.compile(r"(\[\s*[\d\s,\-–—]+\]|\(\s*[\d\s,\-–—]+\))")


def _normalize(text: str) -> str:
    """Aggressive normalization for quote verification: NFKC, lowercase,
    numeric citations removed, every quote char and punctuation mark stripped,
    whitespace collapsed — only word/digit tokens survive."""
    text = unicodedata.normalize("NFKC", text or "")
    text = _CITATION_RE.sub(" ", text)
    text = re.sub(r"[^\w\s]", " ", text)  # all punctuation + quote chars
    return re.sub(r"\s+", " ", text).strip().lower()


def _ngrams(tokens: list[str], n: int = 4) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def quote_verified(quote: str, source: str) -> bool:
    """True when the quote is a contiguous normalized substring of the source,
    or — for quotes of >=6 tokens — >=85% of its word 4-grams occur in the
    normalized source (tolerates minor model paraphrases like dropped citation
    brackets, pronoun swaps and quote-character changes)."""
    nq = _normalize(quote)
    if not nq:
        return False
    ns = _normalize(source)
    if nq in ns:
        return True
    tokens = nq.split()
    if len(tokens) < 6:
        return False
    grams = _ngrams(tokens)
    if not grams:
        return False
    hits = sum(1 for gram in grams if gram in ns)
    return hits / len(grams) >= 0.85


def post_validate(assertion: dict, valid_keys: set[str], source: str) -> dict | None:
    """Clean one P4 assertion; None means drop."""
    out = dict(assertion)
    if out.get("disease_key") not in PILOT_KEYS:
        return None
    key = out.get("finding_key")
    if key and key not in valid_keys:
        # Some models write the literal string "proposed_finding" (or another
        # placeholder) here — keep the real term if they supplied one.
        out["proposed_finding"] = out.get("proposed_finding") or key
        out["finding_key"] = None
    quote = out.get("quote") or ""
    if len(quote.split()) > MAX_QUOTE_WORDS or not quote_verified(quote, source):
        return {"_unverified_quote": True}
    out["quote"] = quote
    return out


def _vocabulary(conn, disease_keys: list[str]) -> list[dict]:
    """Approved vocab rows relevant to the article's diseases."""
    rows = conn.execute("SELECT * FROM findings_vocab WHERE approved = 1")
    return [
        {"finding_key": r["finding_key"], "label": r["label"]}
        for r in rows
        if not disease_keys
        or set(disease_keys) & set(db.from_json(r["disease_keys_json"], []))
    ]


def user_content(
    article_title: str,
    disease_keys: list[str],
    vocabulary: list[dict],
    sections: list[tuple[str, str]],
) -> str:
    return json.dumps(
        {
            "article_title": article_title or "",
            "primary_disease_keys": disease_keys,
            "vocabulary": vocabulary,
            "sections": [
                {"title": title, "text": text} for title, text in sections
            ],
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Per-article prepare / apply
# ---------------------------------------------------------------------------
def _body_sections(article: dict) -> list[tuple[str, str]]:
    """Raw JATS body sections for the article (memory only — never disk).

    W8/B5: reuse ``parse.sections_for`` when the article was parsed in this
    process; on a miss refetch the XML via the hinted bundle path —
    ``articles.s3_prefix``/``media_files_json`` skip the S3 listing round
    trip. Either way the same XML produces the same sections.
    """
    sections = parse.sections_for(article["pmcid"])
    if sections is not None:
        timing.count("extract_sections", source="memory")
        return sections
    timing.count("extract_sections", source="refetch")
    bundle = pmc.get_article_bundle(
        article["pmcid"],
        prefix=article.get("s3_prefix"),
        media_files=db.from_json(article.get("media_files_json")) or None,
    )
    return jats.parse_article(bundle.xml_text).body_sections


def _vocab_key(article: dict) -> tuple[str, ...]:
    """Sorted disease-key tuple: the vocab cache key (equal sets share it)."""
    return tuple(sorted(db.from_json(article["primary_disease_keys_json"], []) or []))


def prepare_article(article: dict, vocabulary: list[dict]) -> dict | None:
    """Pick sections and build the P4 request for one article.

    Returns None when the article yields no usable source text. ``vocabulary``
    is precomputed by the caller (cached per disease-key tuple on the main
    thread) so pooled preparation never touches the shared sqlite connection.
    """
    sections = pick_sections(_body_sections(article))
    source_text = "\n\n".join(text for _, text in sections)
    if not source_text.strip():
        return None
    disease_keys = db.from_json(article["primary_disease_keys_json"], [])
    return {
        "sections": sections,
        "source_text": source_text,
        "valid_keys": {v["finding_key"] for v in vocabulary},
        "request": {
            "stage": "p4",
            "model": config.VP_EXTRACT_MODEL,
            "system": P4.system,
            "user_content": user_content(
                article["title"], disease_keys, vocabulary, sections
            ),
            "schema": P4.schema,
            "prompt_version": P4.version,
        },
    }


def apply_response(
    conn, article: dict, ctx: dict, parsed: dict | None, cached: bool, stats: dict
) -> None:
    """Validate assertions and rewrite the article's source='text' rows."""
    pmcid = article["pmcid"]
    assertions = (parsed or {}).get("assertions") or []

    # A re-apply (--force or a direct call over already-applied rows) must not
    # re-count proposals; a first apply — including a cache-only replay onto a
    # fresh DB — upserts them so findings_vocab matches the live run.
    reapplied = _has_text_rows(conn, pmcid)
    conn.execute(
        "DELETE FROM disease_findings WHERE pmcid = ? AND source = 'text'", (pmcid,)
    )
    seen_proposed: set[str] = set()
    for raw in assertions:
        cleaned = post_validate(raw, ctx["valid_keys"], ctx["source_text"])
        if cleaned is None:
            stats["dropped_disease"] += 1
            continue
        if cleaned.pop("_unverified_quote", False):
            stats["unverified_quote"] += 1
            continue
        finding_key = cleaned.get("finding_key")
        proposed = cleaned.get("proposed_finding")
        if proposed:
            if not reapplied and proposed not in seen_proposed:
                seen_proposed.add(proposed)
                stats["proposed"] += upsert_proposed(
                    conn, proposed, cleaned["disease_key"], None
                )
            if not finding_key:
                finding_key = slugify(proposed)
        if not finding_key:
            continue
        conn.execute(
            "INSERT INTO disease_findings (disease_key, finding_key, subtype, "
            "frequency_text, frequency_pct_low, frequency_pct_high, source, "
            "pmcid, quote) VALUES (?, ?, ?, ?, ?, ?, 'text', ?, ?)",
            (
                cleaned["disease_key"],
                finding_key,
                cleaned.get("subtype"),
                cleaned.get("frequency_text"),
                cleaned.get("pct_low"),
                cleaned.get("pct_high"),
                pmcid,
                cleaned["quote"],
            ),
        )
        stats["inserted"] += 1


def rebuild_image_rows(conn) -> int:
    """Rebuild source='image' disease_findings from stored panels."""
    conn.execute("DELETE FROM disease_findings WHERE source = 'image'")
    n = 0
    for row in conn.execute(
        "SELECT pmcid, disease_key, subtype, findings_json FROM published_panels "
        "WHERE disease_key IS NOT NULL"
    ):
        for finding in db.from_json(row["findings_json"], []):
            key = finding.get("finding_key") if isinstance(finding, dict) else finding
            quote = finding.get("evidence", "") if isinstance(finding, dict) else ""
            if not key:
                continue
            conn.execute(
                "INSERT INTO disease_findings (disease_key, finding_key, subtype, "
                "source, pmcid, quote) VALUES (?, ?, ?, 'image', ?, ?)",
                (row["disease_key"], key, row["subtype"], row["pmcid"], quote),
            )
            n += 1
    return n


def _has_text_rows(conn, pmcid: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM disease_findings WHERE pmcid = ? AND source = 'text' LIMIT 1",
            (pmcid,),
        ).fetchone()
        is not None
    )


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(args) -> int:
    conn = db.init_db()
    if getattr(args, "image_only", False):
        with conn:
            image_rows = rebuild_image_rows(conn)
        print(f"extract: {image_rows} image rows rebuilt")
        conn.close()
        return 0
    disease = None if args.disease == "all" else args.disease
    rows = db.rows_with_status(conn, "articles", "parsed", disease=disease)
    if args.pmcids:
        wanted = set(args.pmcids)
        rows = [r for r in rows if r["pmcid"] in wanted]
    articles = [dict(r) for r in rows]
    articles.sort(key=lambda a: -(a["retrieval_score"] or 0.0))
    if args.limit:
        articles = articles[: args.limit]

    if not getattr(args, "force", False):
        before = len(articles)
        articles = [a for a in articles if not _has_text_rows(conn, a["pmcid"])]
        skipped = before - len(articles)
    else:
        skipped = 0

    print(f"extract: {len(articles)} parsed articles ({skipped} already extracted)")
    if args.dry_run or not articles:
        conn.close()
        return 0

    client = llm.LLMClient(db_conn=conn, budget_usd=args.budget_usd)
    stats = {
        "inserted": 0,
        "proposed": 0,
        "unverified_quote": 0,
        "dropped_disease": 0,
        "no_sections": 0,
        "errors": 0,
        "image_rows": 0,
    }

    # W8/B10: vocabulary once per distinct disease-key tuple, here on the
    # main thread before any pooled work starts — prepare workers then never
    # touch the shared sqlite connection.
    vocab_cache: dict[tuple[str, ...], list[dict]] = {}
    for article in articles:
        key = _vocab_key(article)
        if key not in vocab_cache:
            vocab_cache[key] = _vocabulary(
                conn, db.from_json(article["primary_disease_keys_json"], []) or []
            )

    # W8: stream P4 through client.iter_many. A fetch pool prepares articles
    # just ahead of the LLM calls — iter_many pulls a new request only when
    # a slot frees, so P4 starts before all articles finish preparing and
    # prepared ctxs stay bounded to ~2x the in-flight cap. Results are
    # applied and committed on this thread in completion order; per-article
    # error isolation and stats accounting are unchanged.
    window = max(1, 2 * client.concurrency)
    submitted: list[tuple[dict, dict]] = []  # consumed index -> (article, ctx)

    def _requests():
        """Yield P4 call_json kwargs in article order, preparing just ahead."""
        pending: deque = deque()  # (article, prepare future), input order
        art_iter = iter(articles)
        pool = ThreadPoolExecutor(max_workers=config.VP_FETCH_CONCURRENCY)
        try:
            while True:
                while len(pending) < window:
                    article = next(art_iter, None)
                    if article is None:
                        break
                    pending.append(
                        (
                            article,
                            pool.submit(
                                prepare_article,
                                article,
                                vocab_cache[_vocab_key(article)],
                            ),
                        )
                    )
                if not pending:
                    return
                article, fut = pending.popleft()
                try:
                    ctx = fut.result()
                except Exception as exc:  # noqa: BLE001 - per-article isolation
                    print(f"{article['pmcid']}: fetch error {exc}")
                    stats["errors"] += 1
                    continue
                if ctx is None:
                    stats["no_sections"] += 1
                    continue
                submitted.append((article, ctx))
                yield ctx["request"]
        finally:
            # Early stop: cancel not-yet-started prepares; running ones
            # finish in the background and are discarded.
            pool.shutdown(wait=False, cancel_futures=True)

    budget_hit = False
    requests = _requests()
    results = client.iter_many(requests)
    try:
        for res in results:
            article, ctx = submitted[res.index]
            if res.error is not None:
                if isinstance(res.error, llm.BudgetExceeded):
                    budget_hit = True  # leave the article parsed; stop cleanly
                else:
                    stats["errors"] += 1
                continue
            with conn:  # delete + reinsert per pmcid in one transaction
                apply_response(
                    conn,
                    article,
                    ctx,
                    res.parsed,
                    bool((res.meta or {}).get("cached")),
                    stats,
                )
    finally:
        requests.close()
        results.close()
    if budget_hit:
        print(f"LLM budget exhausted (${client.spent_usd:.4f}); stopping cleanly. Rerun to resume.")

    stats["image_rows"] = rebuild_image_rows(conn)
    conn.commit()
    print(
        f"extract: {stats['inserted']} text rows ({stats['unverified_quote']} "
        f"unverified_quote dropped, {stats['dropped_disease']} disease dropped), "
        f"{stats['proposed']} proposed upserts, {stats['image_rows']} image rows, "
        f"spend=${client.spent_usd:.4f}"
    )
    conn.close()
    return 0
