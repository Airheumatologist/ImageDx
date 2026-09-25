"""Stage 8: FastAPI viewer for the Visual Findings Library pilot.

Pages are plain HTML/JS (no build step) served from ``viewer/static``; JSON
comes from ``/api/...``; images come from ``/media/...`` which only serves
paths under panels/, thumbs/ and figures/ inside the pilot data dir.

Tab membership lives in ``TABS`` below: an ordered per-disease mapping from
modality / body_site / finding keys / finding categories to a tab, evaluated
first-match-wins. Sorting everywhere is classic < variant < atypical, then
confidence descending. Unapproved (LLM-proposed) findings are filtered out of
every response.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .. import config, db

STATIC_DIR = Path(__file__).resolve().parent / "static"
MEDIA_PREFIXES = {"panels", "thumbs", "figures"}

TYPICALITY_ORDER = {"classic": 0, "variant": 1, "atypical": 2}

# ---------------------------------------------------------------------------
# Tab membership (§5 stage 8). Each tab's match is a list of alternatives
# ("any_of"): within one alternative every declared key must hit (AND);
# alternatives are OR'd. ``not_*`` exclusions apply to all alternatives.
# Tabs are evaluated in display order, except tabs carrying ``priority`` are
# evaluated before non-priority ones; first match wins, unmatched panels land
# in a trailing "other" tab.
# ---------------------------------------------------------------------------
TABS: dict[str, list[dict]] = {
    "sle": [
        {
            "key": "skin",
            "label": "Skin",
            "skin_tone_filter": True,
            "group_by": "subtype",
            "group_order": ["acle", "scle", "dle"],
            "match": {
                "any_of": [
                    {
                        "modalities": {"clinical_photo", "dermoscopy"},
                        "categories": {"skin", "nail"},
                    }
                ],
                "not_categories": {"mucosa"},
                "not_body_sites": {"oral mucosa", "mouth", "lips", "tongue"},
            },
        },
        {
            "key": "mucosa",
            "label": "Mucosa",
            "match": {
                "any_of": [
                    {"categories": {"mucosa"}},
                    {"body_sites": {"oral mucosa", "mouth", "lips", "tongue", "nasal"}},
                ]
            },
        },
        {
            "key": "musculoskeletal",
            "label": "Musculoskeletal",
            "match": {
                "any_of": [
                    {"modalities": {"clinical_photo"}, "categories": {"clinical_msk"}}
                ]
            },
        },
        {
            "key": "renal_histology",
            "label": "Renal histology",
            "match": {
                "any_of": [
                    {
                        "modalities": {"histology_he", "histology_ihc", "immunofluorescence"},
                        "findings": {
                            "lupus_nephritis_class",
                            "wire_loop_lesion",
                            "full_house_immunofluorescence",
                        },
                    },
                    {
                        "modalities": {"histology_he", "histology_ihc", "immunofluorescence"},
                        "body_sites": {"kidney", "renal"},
                    },
                ]
            },
        },
        {
            "key": "skin_histology",
            "label": "Skin histology / DIF",
            "match": {
                "any_of": [
                    {"modalities": {"histology_he", "histology_ihc", "immunofluorescence"}}
                ]
            },
        },
        {
            "key": "imaging",
            "label": "Imaging",
            "match": {
                "any_of": [
                    {"modalities": {"radiograph", "ct", "mri", "ultrasound", "echo", "pet"}}
                ]
            },
        },
        {
            "key": "capillaroscopy",
            "label": "Capillaroscopy",
            "match": {
                "any_of": [
                    {"modalities": {"capillaroscopy"}},
                    {"categories": {"capillaroscopy"}},
                ]
            },
        },
    ],
    "dm": [
        {
            "key": "skin",
            "label": "Skin",
            "skin_tone_filter": True,
            "group_by": "finding",
            "match": {"any_of": [{"modalities": {"clinical_photo", "dermoscopy"}}]},
        },
        {
            "key": "nailfold",
            "label": "Nailfold / Capillaroscopy",
            "priority": 0,
            "match": {
                "any_of": [
                    {"modalities": {"capillaroscopy"}},
                    {"categories": {"capillaroscopy", "nail"}},
                    {"body_sites": {"nailfold", "nail", "periungual"}},
                ]
            },
        },
        {
            "key": "muscle_histology",
            "label": "Muscle histology",
            "match": {
                "any_of": [
                    {"modalities": {"histology_he", "histology_ihc", "immunofluorescence"}},
                    {"categories": {"histology"}},
                ]
            },
        },
        {
            "key": "mri",
            "label": "MRI",
            "match": {"any_of": [{"modalities": {"mri"}}]},
        },
        {
            "key": "lung_ct",
            "label": "Lung CT",
            "match": {
                "any_of": [
                    {"modalities": {"ct"}},
                    {"body_sites": {"lung", "chest", "thorax"}},
                ]
            },
        },
        {
            "key": "calcinosis",
            "label": "Calcinosis",
            "priority": 0,
            "match": {
                "any_of": [{"findings": {"calcinosis_cutis", "calcinosis_radiograph"}}]
            },
        },
    ],
    "as": [
        {
            "key": "si_radiograph",
            "label": "SI radiograph",
            "group_by": "stage",
            "group_order": ["nr_axspa", "nr-axSpA", "early", "advanced"],
            "match": {
                "any_of": [
                    {
                        "modalities": {"radiograph"},
                        "body_sites": {"sacroiliac", "si joint", "sij", "pelvis"},
                    },
                    {
                        "modalities": {"radiograph"},
                        "findings": {"sacroiliitis", "si_erosions", "si_sclerosis", "si_ankylosis"},
                    },
                ]
            },
        },
        {
            "key": "si_mri",
            "label": "SI MRI",
            "group_by": "stage",
            "group_order": ["nr_axspa", "nr-axSpA", "early", "advanced"],
            "match": {
                "any_of": [
                    {
                        "modalities": {"mri"},
                        "body_sites": {"sacroiliac", "si joint", "sij", "pelvis"},
                    },
                    {
                        "modalities": {"mri"},
                        "findings": {"si_bone_marrow_edema", "fat_metaplasia", "backfill"},
                    },
                ]
            },
        },
        {
            "key": "spine",
            "label": "Spine radiograph / CT",
            "group_by": "stage",
            "group_order": ["nr_axspa", "nr-axSpA", "early", "advanced"],
            "match": {
                "any_of": [
                    {"modalities": {"radiograph", "ct"}},
                    {
                        "modalities": {"mri"},
                        "findings": {"corner_inflammatory_lesion", "corner_fat_lesion", "romanus_lesion"},
                    },
                ]
            },
        },
        {
            "key": "clinical",
            "label": "Clinical",
            "match": {
                "any_of": [
                    {"modalities": {"clinical_photo"}},
                    {"categories": {"clinical_msk"}},
                ]
            },
        },
        {
            "key": "eye",
            "label": "Eye",
            "match": {
                "any_of": [
                    {"modalities": {"ophthalmic"}},
                    {"categories": {"eye"}},
                ]
            },
        },
    ],
}

# Side-by-side comparisons for /compare/sle-dm-skin (§5 stage 8).
COMPARISONS = [
    {
        "title": "Gottron papules vs SLE hand/knuckle-sparing rash",
        "left": {"disease": "dm", "finding": "gottron_papules", "label": "DM: Gottron papules"},
        "right": {
            "disease": "sle",
            "body_sites": {"hand", "hands", "knuckle", "periungual", "fingers", "dorsal hands"},
            "modalities": {"clinical_photo"},
            "label": "SLE: hand/periungual rash",
        },
    },
    {
        "title": "Heliotrope rash vs malar rash",
        "left": {"disease": "dm", "finding": "heliotrope_rash", "label": "DM: heliotrope rash"},
        "right": {"disease": "sle", "finding": "malar_rash", "label": "SLE: malar rash"},
    },
    {
        "title": "V/shawl sign vs SLE photosensitive rash",
        "left": {
            "disease": "dm",
            "findings_any": {"v_sign", "shawl_sign"},
            "label": "DM: V/shawl sign",
        },
        "right": {
            "disease": "sle",
            "subtype_in": {"acle", "scle"},
            "modalities": {"clinical_photo"},
            "label": "SLE: ACLE/SCLE photosensitive rash",
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _norm(value) -> str:
    return str(value or "").strip().lower()


def _approved_keys(conn) -> set[str]:
    return {r["finding_key"] for r in conn.execute("SELECT finding_key FROM findings_vocab WHERE approved=1")}


def _categories(conn) -> dict[str, str]:
    return {r["finding_key"]: r["category"] for r in conn.execute("SELECT finding_key, category FROM findings_vocab")}


def _sort_panels(panels: list[dict]) -> list[dict]:
    return sorted(
        panels,
        key=lambda p: (
            TYPICALITY_ORDER.get(_norm(p.get("typicality")), 3),
            -(p.get("confidence") or 0.0),
        ),
    )


def _finding_keys(panel: dict) -> set[str]:
    """Finding keys from either findings_json shape (str or dict)."""
    keys: set[str] = set()
    for f in panel.get("findings") or []:
        keys.add(f["key"] if isinstance(f, dict) else str(f))
    return keys


def _panel_json(row: sqlite3.Row, approved: set[str], labels: dict[str, str]) -> dict:
    raw = db.from_json(row["findings_json"], [])
    findings = []
    for f in raw:
        key = f.get("finding_key") if isinstance(f, dict) else f
        if key and key in approved:
            findings.append(
                {
                    "key": key,
                    "label": labels.get(key, key),
                    "evidence": f.get("evidence", "") if isinstance(f, dict) else "",
                }
            )
    thumb = row["thumb_path"] or row["image_path"]
    return {
        "panel_id": row["panel_id"],
        "figure_id": row["figure_id"],
        "pmcid": row["pmcid"],
        "panel_label": row["panel_label"],
        "disease_key": row["disease_key"],
        "subtype": row["subtype"],
        "modality": row["modality"],
        "body_site": row["body_site"],
        "findings": findings,
        "sha256": row["sha256"],
        "typicality": row["typicality"],
        "stage": row["stage"],
        "age_group": row["age_group"],
        "skin_tone": row["skin_tone"],
        "stated_ethnicity": row["stated_ethnicity"],
        "stated_ethnicity_quote": row["stated_ethnicity_quote"],
        "study_region": row["study_region"],
        "confidence": row["confidence"],
        "rationale": row["rationale"],
        "attribution_text": row["attribution_text"],
        "figure_label": row["figure_label"],
        "figure_caption": row["figure_caption"],
        "in_text_mentions": db.from_json(row["in_text_mentions_json"], []),
        "license_code": row["license_code"],
        "doi_url": f"https://doi.org/{row['doi']}" if row["doi"] else row["source_url"],
        "image": f"/media/{row['image_path']}" if row["image_path"] else None,
        "thumb": f"/media/{thumb}" if thumb else None,
    }


def _labels(conn) -> dict[str, str]:
    return {r["finding_key"]: r["label"] for r in conn.execute("SELECT finding_key, label FROM findings_vocab")}


def _collapse_duplicates(panels: list[dict]) -> list[dict]:
    """Panels sharing an image sha256 collapse into one card that lists
    every row's attribution (dedup by sha256 done at store time)."""
    grouped: dict[str, dict] = {}
    for p in panels:
        key = p.get("sha256") or p["panel_id"]
        existing = grouped.get(key)
        if existing is None:
            p["attribution_variants"] = [p["attribution_text"]]
            p["caption_variants"] = [p.get("figure_caption")]
            p["panel_ids"] = [p["panel_id"]]
            grouped[key] = p
            continue
        existing["attribution_variants"].append(p["attribution_text"])
        existing["panel_ids"].append(p["panel_id"])
        if p.get("figure_caption") not in existing["caption_variants"]:
            existing["caption_variants"].append(p.get("figure_caption"))
        known = {f["key"] for f in existing["findings"]}
        for f in p["findings"]:
            if f["key"] not in known:
                existing["findings"].append(f)
                known.add(f["key"])
        # Merge tag union fields when the duplicate adds information.
        for field in ("disease_key", "subtype", "modality", "body_site", "stage"):
            values = {existing.get(field), p.get(field)} - {None}
            if len(values) > 1:
                existing[field] = "/".join(sorted(values))
    return list(grouped.values())


def _match_alt(alt: dict, panel: dict, cats: set[str], body: str) -> bool:
    """One alternative: every declared key must hit (AND within it)."""
    findings = _finding_keys(panel)
    if "modalities" in alt and _norm(panel.get("modality")) not in {
        _norm(m) for m in alt["modalities"]
    }:
        return False
    if "body_sites" in alt and body not in {_norm(b) for b in alt["body_sites"]}:
        return False
    if "findings" in alt and not (findings & {_norm(f) for f in alt["findings"]}):
        return False
    if "categories" in alt and not (cats & {_norm(c) for c in alt["categories"]}):
        return False
    return True


def _match(tab_match: dict, panel: dict, categories: dict[str, str]) -> bool:
    findings = _finding_keys(panel)
    cats = {categories.get(k) for k in findings} - {None}
    body = _norm(panel.get("body_site"))
    if cats & {_norm(x) for x in tab_match.get("not_categories", ())}:
        return False
    if body and body in {_norm(x) for x in tab_match.get("not_body_sites", ())}:
        return False
    if findings & {_norm(x) for x in tab_match.get("not_findings", ())}:
        return False
    return any(
        _match_alt(alt, panel, cats, body) for alt in tab_match.get("any_of", [])
    )


def assign_tab(panel: dict, tabs: list[dict], categories: dict[str, str]) -> str:
    ordered = sorted(tabs, key=lambda t: 0 if t.get("priority") == 0 else 1)
    for tab in ordered:
        if _match(tab["match"], panel, categories):
            return tab["key"]
    return "other"


def _query_panels(
    conn,
    disease: str,
    *,
    modality=None,
    subtype=None,
    skin_tone=None,
    finding=None,
    typicality=None,
) -> list[sqlite3.Row]:
    where = ["p.disease_key = :disease"]
    params: dict = {"disease": disease}
    if modality:
        where.append("p.modality = :modality")
        params["modality"] = modality
    if subtype:
        where.append("p.subtype = :subtype")
        params["subtype"] = subtype
    if skin_tone:
        where.append("p.skin_tone = :skin_tone")
        params["skin_tone"] = skin_tone
    if finding:
        where.append(
            "EXISTS (SELECT 1 FROM json_each(p.findings_json) je "
            "WHERE COALESCE(json_extract(je.value, '$.finding_key'), je.value) = :finding)"
        )
        params["finding"] = finding
    if typicality:
        where.append("p.typicality = :typicality")
        params["typicality"] = typicality
    sql = (
        "SELECT p.*, a.doi, f.label AS figure_label, f.caption AS figure_caption, "
        "f.in_text_mentions_json FROM panels p "
        "LEFT JOIN articles a ON a.pmcid = p.pmcid "
        "LEFT JOIN figures f ON f.figure_id = p.figure_id "
        f"WHERE {' AND '.join(where)}"
    )
    return list(conn.execute(sql, params))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def create_app(data_dir: str | None = None) -> FastAPI:
    data_root = (Path(data_dir) if data_dir is not None else config.data_dir()).resolve()
    app = FastAPI(title="Visual Findings Library")
    app.state.data_dir = data_root

    def conn() -> sqlite3.Connection:
        return db.connect(data_root / config.DB_FILENAME)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.middleware("http")
    async def _no_cache(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache"
        return response

    def _page(name: str) -> HTMLResponse:
        return HTMLResponse((STATIC_DIR / name).read_text(encoding="utf-8"))

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return _page("index.html")

    @app.get("/disease/{key}", response_class=HTMLResponse)
    def disease_page(key: str) -> HTMLResponse:
        if key not in TABS:
            raise HTTPException(404, "unknown disease")
        return _page("disease.html")

    @app.get("/compare/sle-dm-skin", response_class=HTMLResponse)
    def compare_page() -> HTMLResponse:
        return _page("compare.html")

    @app.get("/api/diseases")
    def api_diseases() -> list[dict]:
        c = conn()
        try:
            return [
                {
                    "key": r["disease_key"],
                    "name": r["name"],
                    "subtypes": db.from_json(r["subtypes_json"], []),
                }
                for r in c.execute("SELECT * FROM diseases ORDER BY disease_key")
            ]
        finally:
            c.close()

    @app.get("/api/vocab")
    def api_vocab(disease: str | None = Query(default=None)) -> list[dict]:
        c = conn()
        try:
            rows = list(c.execute("SELECT * FROM findings_vocab WHERE approved=1"))
            if disease:
                rows = [r for r in rows if disease in db.from_json(r["disease_keys_json"], [])]
            return [
                {
                    "finding_key": r["finding_key"],
                    "label": r["label"],
                    "category": r["category"],
                    "disease_keys": db.from_json(r["disease_keys_json"], []),
                }
                for r in rows
            ]
        finally:
            c.close()

    @app.get("/api/diseases/{key}/tabs")
    def api_tabs(key: str) -> list[dict]:
        if key not in TABS:
            raise HTTPException(404, "unknown disease")
        return [{"key": t["key"], "label": t["label"], "group_by": t.get("group_by"), "group_order": t.get("group_order"), "skin_tone_filter": bool(t.get("skin_tone_filter"))} for t in TABS[key]]

    @app.get("/api/diseases/{key}/panels")
    def api_panels(
        key: str,
        modality: str | None = None,
        subtype: str | None = None,
        skin_tone: str | None = None,
        finding: str | None = None,
        typicality: str | None = None,
    ) -> dict:
        if key not in TABS:
            raise HTTPException(404, "unknown disease")
        c = conn()
        try:
            approved = _approved_keys(c)
            categories = _categories(c)
            labels = _labels(c)
            if finding and finding not in approved:
                # unapproved/proposed findings are never shown
                return {"panels": [], "count": 0}
            rows = _query_panels(
                c,
                key,
                modality=modality,
                subtype=subtype,
                skin_tone=skin_tone,
                finding=finding,
                typicality=typicality,
            )
            panels = [_panel_json(r, approved, labels) for r in rows]
            panels = _collapse_duplicates(panels)
            for p in panels:
                p["tab"] = assign_tab(p, TABS[key], categories)
            return {"panels": _sort_panels(panels), "count": len(panels)}
        finally:
            c.close()

    @app.get("/api/diseases/{key}/findings")
    def api_findings(key: str) -> list[dict]:
        if key not in TABS:
            raise HTTPException(404, "unknown disease")
        c = conn()
        try:
            # Approved vocab keys only — proposed findings are never shown.
            rows = list(
                c.execute(
                    "SELECT df.*, fv.label AS vocab_label FROM disease_findings df "
                    "JOIN findings_vocab fv "
                    "  ON fv.finding_key = df.finding_key AND fv.approved = 1 "
                    "WHERE df.disease_key = ? "
                    "ORDER BY CASE df.source WHEN 'text' THEN 0 ELSE 1 END, "
                    "COALESCE(df.frequency_pct_high, -1) DESC, df.id",
                    (key,),
                )
            )
            seen: set[tuple] = set()
            out = []
            for r in rows:
                pair = (r["finding_key"], r["pmcid"])
                if pair in seen:
                    continue
                seen.add(pair)
                out.append(
                    {
                        "finding_key": r["finding_key"],
                        "label": r["vocab_label"],
                        "subtype": r["subtype"],
                        "frequency_text": r["frequency_text"],
                        "pct_low": r["frequency_pct_low"],
                        "pct_high": r["frequency_pct_high"],
                        "quote": r["quote"],
                        "pmcid": r["pmcid"],
                        "source": r["source"],
                    }
                )
                if len(out) >= 12:
                    break
            return out
        finally:
            c.close()

    @app.get("/api/compare/sle-dm-skin")
    def api_compare() -> list[dict]:
        c = conn()
        try:
            approved = _approved_keys(c)
            labels = _labels(c)
            out = []
            for pair in COMPARISONS:
                entry = {"title": pair["title"], "left": {"label": pair["left"]["label"]}, "right": {"label": pair["right"]["label"]}}
                for side in ("left", "right"):
                    spec = pair[side]
                    rows = _query_panels(
                        c,
                        spec["disease"],
                        modality=None,
                        finding=spec.get("finding"),
                    )
                    panels = [_panel_json(r, approved, labels) for r in rows]
                    panels = _collapse_duplicates(panels)
                    if spec.get("findings_any"):
                        wanted = {_norm(f) for f in spec["findings_any"]}
                        panels = [p for p in panels if _finding_keys(p) & wanted]
                    if spec.get("body_sites"):
                        wanted = {_norm(b) for b in spec["body_sites"]}
                        panels = [p for p in panels if _norm(p.get("body_site")) in wanted]
                    if spec.get("subtype_in"):
                        wanted = {_norm(s) for s in spec["subtype_in"]}
                        panels = [p for p in panels if _norm(p.get("subtype")) in wanted]
                    if spec.get("modalities"):
                        wanted = {_norm(m) for m in spec["modalities"]}
                        panels = [p for p in panels if _norm(p.get("modality")) in wanted]
                    entry[side]["panels"] = _sort_panels(panels)[:6]
                    entry[side]["disease"] = spec["disease"]
                out.append(entry)
            return out
        finally:
            c.close()

    @app.get("/media/{path:path}")
    def media(path: str):
        root = data_root
        target = (root / path).resolve()
        # Path traversal guard: must resolve inside data_dir AND under an
        # allowed image subdirectory.
        if not str(target).startswith(str(root) + "/"):
            raise HTTPException(403, "forbidden")
        try:
            rel = target.relative_to(root)
        except ValueError:
            raise HTTPException(403, "forbidden") from None
        if rel.parts[0] not in MEDIA_PREFIXES:
            raise HTTPException(403, "forbidden")
        if not target.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(target)

    return app


def run(args) -> int:
    import uvicorn

    print(f"viewer: http://127.0.0.1:{args.port} (data: {config.data_dir()})")
    uvicorn.run(
        "src.visual_pilot.viewer.app:create_app",
        factory=True,
        host="127.0.0.1",
        port=args.port,
        log_level="warning",
    )
    return 0
