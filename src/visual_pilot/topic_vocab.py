"""Build the findings vocabulary for the main database topics (prompt P6).

``data/main_database_topics.json`` describes each topic's clinical,
radiological and pathology manifestations as prose ("CT angiography:
pathognomonic intimal flap dividing the aorta..."). Discovery needs short,
searchable visual findings instead: a label, a category, synonyms and the
phrases a figure caption would use. P6 turns one topic's manifestations into
3-12 such findings; the post-validation below keeps only findings with usable
caption terms and writes ``src/visual_pilot/data/topic_findings_vocab.json``.

Topics that already have curated findings (the pilot diseases) are skipped.
Generated findings are seeded as approved vocabulary, like the curated file,
and carry ``source``/``model``/``prompt_version`` for review. Reruns keep
existing topics unless ``--force`` or ``--topics`` names them; the
``llm_calls`` cache makes a forced rerun of an unchanged prompt free.

Usage::

    python -m src.visual_pilot.cli build-vocab            # missing topics
    python -m src.visual_pilot.cli build-vocab --topics gout_x vitiligo --force
"""

from __future__ import annotations

import json
import re

from . import config, db, diseases, llm, pair_terms
from .prompts import Prompt

P6_VERSION = "p6.v1"
_CATEGORIES = sorted(diseases.FINDING_CATEGORIES)
P6_SYSTEM = f"""You build the findings vocabulary of a clinical visual-diagnosis image library. The library collects real patient images (clinical photographs, dermoscopy, endoscopy, fundus and slit-lamp photographs, radiographs, CT, MRI, ultrasound, echocardiography, nuclear imaging, histopathology, gross specimens) from figures in open-access articles, found by searching figure captions.

You receive one disease topic with its synonyms and its clinical, radiological and pathology manifestations. Return the 3-12 most characteristic findings of this disease that a published figure can show and its caption would name. Order them from most to least characteristic.

For each finding:
- `key`: short snake_case name of the finding alone, without the disease name (e.g. "intimal_flap", "heliotrope_rash").
- `label`: short clinical name, at most 60 characters, sentence case (e.g. "Intimal flap on CT angiography").
- `category`: one of {", ".join(_CATEGORIES)}. skin covers skin lesions and rashes; clinical_general covers whole-body or facial appearance (dysmorphism, habitus, swelling) that is not skin; clinical_msk covers joints, limbs and deformities; eye covers any ophthalmic image; radiology_xray is plain radiography; us is ultrasound other than echocardiography; nuclear is PET or scintigraphy; gross is a gross pathology specimen.
- `synonyms`: up to 5 other names clinicians use for the finding.
- `caption_terms`: 2-6 lower-case phrases that a figure caption showing this finding would contain verbatim, e.g. "intimal flap", "double barrel aorta". Each must be specific to the finding: never a bare generic word such as "rash", "lesion", "mass", "nodule", "ulcer", "opacity", "erythema", "biopsy" or "inflammation", and never only the disease name.

Rules:
- Only visible findings. Skip symptoms (pain, fever, fatigue), laboratory values, genetic tests, physiology, and signs found only by palpation, auscultation or history.
- One finding per entry; do not merge different findings or modalities into one entry.
- Prefer findings that are distinctive for this disease over findings shared by many diseases.

Return only JSON."""
P6_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "label": {"type": "string"},
                    "category": {"type": "string", "enum": _CATEGORIES},
                    "synonyms": {"type": "array", "items": {"type": "string"}},
                    "caption_terms": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["key", "label", "category", "synonyms", "caption_terms"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}
P6 = Prompt(name="p6_topic_vocab", version=P6_VERSION, system=P6_SYSTEM, schema=P6_SCHEMA)

MAX_FINDINGS = 12
MAX_SYNONYMS = 5
MAX_CAPTION_TERMS = 6
MIN_SINGLE_WORD = 8
# Bare words P6 is told not to use, on top of pair_terms' generic list.
_TOO_GENERIC = frozenset({
    "mass", "masses", "opacity", "opacities", "swelling", "lump", "tumor",
    "tumour", "deformity", "image", "finding", "lesion", "lesions", "rash",
})


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def user_content(topic: dict) -> str:
    return json.dumps(
        {
            "topic": topic["name"],
            "synonyms": topic.get("synonyms") or [],
            "specialty": topic.get("specialty"),
            "clinical_manifestations": topic.get("clinical_manifestations") or [],
            "radiological_manifestations": topic.get("radiological_manifestations") or [],
            "pathology_manifestations": topic.get("pathology_manifestations") or [],
        },
        ensure_ascii=False,
    )


def _disease_words(topic: dict) -> set[str]:
    names = [topic["name"], *(topic.get("synonyms") or [])]
    return {pair_terms._caption_norm(n) for n in names if n}


def post_validate(topic: dict, parsed: dict, meta: dict | None = None) -> list[dict]:
    """Vocabulary rows for one topic; findings without usable caption terms drop."""
    disease_key = diseases.topic_disease_key(topic)
    disease_words = _disease_words(topic)
    rows: list[dict] = []
    seen_keys: set[str] = set()
    seen_terms: set[str] = set()
    for item in (parsed or {}).get("findings") or []:
        slug = _slug(item.get("key") or item.get("label") or "")
        label = " ".join(str(item.get("label") or "").split())[:80]
        category = item.get("category")
        if not slug or not label or category not in diseases.FINDING_CATEGORIES:
            continue
        terms = []
        for term in item.get("caption_terms") or []:
            term = pair_terms._caption_norm(term)
            if (
                len(term) < 4
                # A lone short word ("flap", "cyst") matches far too many
                # captions; long single terms ("onycholysis") are specific.
                or (" " not in term and len(term) < MIN_SINGLE_WORD)
                or term in pair_terms._GENERIC_TERMS
                or term in _TOO_GENERIC
                or term in disease_words
                or term in seen_terms
                or term in terms
            ):
                continue
            terms.append(term)
        if not terms:
            continue
        key = f"{topic['topic_id']}_{slug}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        seen_terms.update(terms)
        synonyms = [
            s for s in dict.fromkeys(" ".join(str(s).split()) for s in item.get("synonyms") or [])
            if s and s.casefold() != label.casefold()
        ][:MAX_SYNONYMS]
        rows.append({
            "finding_key": key,
            "label": label,
            "disease_keys": [disease_key],
            "category": category,
            "synonyms": synonyms,
            "caption_terms": terms[:MAX_CAPTION_TERMS],
            "source": "llm_topic_vocab",
            "model": (meta or {}).get("model") or config.VP_VOCAB_MODEL,
            "prompt_version": P6.version,
        })
        if len(rows) >= MAX_FINDINGS:
            break
    return rows


def _request(topic: dict) -> dict:
    return {
        "stage": "p6",
        "model": config.VP_VOCAB_MODEL,
        "system": P6.system,
        "user_content": user_content(topic),
        "schema": P6.schema,
        "prompt_version": P6.version,
        "reasoning_effort": config.VP_VOCAB_REASONING_EFFORT,
    }


def _read_existing() -> list[dict]:
    if not diseases.TOPIC_VOCAB_PATH.exists():
        return []
    return json.loads(diseases.TOPIC_VOCAB_PATH.read_text(encoding="utf-8"))


def _write(rows: list[dict], topic_order: list[str]) -> None:
    rank = {key: i for i, key in enumerate(topic_order)}
    rows = sorted(rows, key=lambda r: (rank.get(r["disease_keys"][0], len(rank)),))
    tmp = diseases.TOPIC_VOCAB_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(diseases.TOPIC_VOCAB_PATH)


def run(args) -> int:
    topics = diseases.load_topics()
    curated = {
        key
        for item in json.loads((diseases.DATA_DIR / "findings_vocab.json").read_text(encoding="utf-8"))
        for key in item["disease_keys"]
    }
    existing = _read_existing()
    have = {r["disease_keys"][0] for r in existing}
    wanted = set(getattr(args, "topics", None) or [])
    todo = [
        t for t in topics
        if diseases.topic_disease_key(t) not in curated
        and (t["topic_id"] in wanted if wanted else (args.force or diseases.topic_disease_key(t) not in have))
    ]
    if args.limit:
        todo = todo[: args.limit]
    order = [diseases.topic_disease_key(t) for t in topics]
    print(f"build-vocab: {len(todo)} topic(s) to build, {len(have)} already built, "
          f"{len(curated)} curated")
    if args.dry_run or not todo:
        return 0

    conn = db.init_db()
    client = llm.LLMClient(db_conn=conn, budget_usd=args.budget_usd, max_retries=2)
    rebuilt = {diseases.topic_disease_key(t) for t in todo}
    kept = [r for r in existing if r["disease_keys"][0] not in rebuilt]
    built: list[dict] = []
    empty, errors = [], []
    done = 0
    try:
        for res in client.iter_many(_request(t) for t in todo):
            topic = todo[res.index]
            done += 1
            if res.error is not None:
                if isinstance(res.error, llm.BudgetExceeded):
                    print("build-vocab: budget exhausted; stopping")
                    break
                errors.append(topic["topic_id"])
                print(f"build-vocab: {topic['topic_id']}: error: {res.error}")
                continue
            rows = post_validate(topic, res.parsed, res.meta)
            if not rows:
                empty.append(topic["topic_id"])
            built.extend(rows)
            print(f"build-vocab: [{done}/{len(todo)}] {topic['topic_id']}: {len(rows)} finding(s)")
            if done % 25 == 0:
                _write(kept + built, order)  # checkpoint: a crash keeps progress
    finally:
        _write(kept + built, order)
        conn.close()
    print(f"build-vocab: wrote {len(kept) + len(built)} finding(s) to {diseases.TOPIC_VOCAB_PATH} "
          f"({len(built)} new); {len(empty)} topic(s) without usable findings, "
          f"{len(errors)} error(s)")
    if empty:
        print("build-vocab: no findings: " + ", ".join(empty))
    if errors:
        print("build-vocab: errors (rerun to retry): " + ", ".join(errors))
    return 0
