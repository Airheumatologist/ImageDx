"""Live smoke for the W4 LLM layer: one real call per prompt P1-P4 plus the
stage-0 URL-fetch check (~5 real figure URLs sent to the vision model).

Usage (from repo root):  python3 scripts/vp_smoke_llm.py [--budget-usd 0.5]

Writes llm_calls rows into the pilot DB (the cost ledger is the point).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visual_pilot import config, db, llm, pmc
from src.visual_pilot.prompts import P1, P2, P3, P4

EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
BUDGET = 0.50


def _short(obj, limit=280) -> str:
    text = json.dumps(obj, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def real_article() -> dict:
    resp = pmc.http_client().get(
        EPMC_SEARCH,
        params={
            "query": 'TITLE:"dermatomyositis" AND OPEN_ACCESS:y AND PUB_TYPE:"review"',
            "format": "json",
            "pageSize": 3,
            "resultType": "core",
        },
    )
    resp.raise_for_status()
    return resp.json()["resultList"]["result"][0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-usd", type=float, default=BUDGET)
    args = ap.parse_args()

    conn = db.init_db()
    client = llm.LLMClient(db_conn=conn, budget_usd=args.budget_usd)
    print(f"provider={client.provider} base_url={client.base_url}")
    print(f"triage/extract model={config.VP_TRIAGE_MODEL}  judge={config.VP_JUDGE_MODEL}")

    article = real_article()
    pmcid = article["pmcid"]
    print(f"\nusing real article {pmcid}: {article.get('title', '')[:80]}")

    # --- P1: relevance on a real title+abstract -------------------------
    user = f"Title: {article.get('title')}\nAbstract: {(article.get('abstractText') or '')[:1500]}"
    parsed, meta = client.call_prompt("p1", P1, config.VP_TRIAGE_MODEL, user)
    print("\n[P1]", _short(parsed), "| cost", meta["cost_usd"], "| fmt", meta.get("response_format"), "| cached", meta["cached"])

    # --- P2: caption triage on 2-3 real captions ------------------------
    bundle = pmc.get_article_bundle(pmcid)
    captions = _captions(bundle.xml_text)[:3]
    figures = [
        {"figure_id": f"{pmcid}:fig{i+1}", "label": c[0], "caption": c[1][:800], "mentions": []}
        for i, c in enumerate(captions)
    ]
    parsed, meta = client.call_prompt(
        "p2", P2, config.VP_TRIAGE_MODEL, json.dumps({"figures": figures})
    )
    print("[P2]", _short(parsed), "| cost", meta["cost_usd"])

    # --- P3: vision judge on one real figure (base64 mode) --------------
    ref = _first_image_ref(bundle)
    mime, blob = pmc.prepare_for_llm(pmc.fetch_image_bytes(ref))
    vocab = ["heliotrope_rash", "gottron_papules", "malar_rash", "sacroiliitis"]
    p3_user = json.dumps(
        {
            "figure_id": f"{pmcid}:fig1",
            "article_title": article.get("title"),
            "primary_disease_keys": ["dm"],
            "label": captions[0][0] if captions else "Figure 1",
            "caption": (captions[0][1] if captions else "")[:800],
            "in_text_mentions": [],
            "vocabulary": vocab,
        }
    )
    parsed, meta = client.call_prompt(
        "p3",
        P3,
        config.VP_JUDGE_MODEL,
        p3_user,
        images=[llm.ImageInput(data_url=pmc.to_data_url(mime, blob))],
    )
    print("[P3]", _short(parsed, 400), "| cost", meta["cost_usd"], "| fmt", meta.get("response_format"), "| cached", meta["cached"])

    # --- P4: findings from a real paragraph -----------------------------
    section = _first_section_paragraph(bundle.xml_text)
    parsed, meta = client.call_prompt(
        "p4",
        P4,
        config.VP_EXTRACT_MODEL,
        json.dumps({"vocabulary": vocab, "sections": [{"title": "text", "text": section[:3000]}]}),
    )
    print("[P4]", _short(parsed), "| cost", meta["cost_usd"])

    # --- Stage 0 check 3: can the provider fetch ~5 image URLs? ---------
    print("\n[URL FETCH CHECK]")
    refs = _image_refs(bundle, limit=5)
    url_ok = 0
    for i, r in enumerate(refs):
        try:
            parsed, meta = client.call_json(
                stage="url_probe",
                model=config.VP_JUDGE_MODEL,
                system="You describe medical images. Return only JSON.",
                user_content="In one word, what kind of image is this? "
                "(e.g. clinical photo, histology, radiograph, diagram)",
                schema={
                    "type": "object",
                    "properties": {
                        "image_type": {"type": "string"},
                        "looks_real": {"type": "boolean"},
                    },
                    "required": ["image_type", "looks_real"],
                    "additionalProperties": False,
                },
                images=[llm.ImageInput(url=r.url)],
                prompt_version="urlprobe.v1",
            )
            print(f"  url[{i}] {r.url[-60:]} -> {_short(parsed, 120)}")
            url_ok += 1
        except Exception as exc:
            print(f"  url[{i}] {r.url[-60:]} -> FAILED: {str(exc)[:160]}")

    if url_ok < len(refs):
        # Confirm base64 fallback works on one image.
        try:
            data = pmc.fetch_image_bytes(refs[0])
            mime, blob = pmc.prepare_for_llm(data)
            parsed, meta = client.call_json(
                stage="url_probe",
                model=config.VP_JUDGE_MODEL,
                system="You describe medical images. Return only JSON.",
                user_content="In one word, what kind of image is this?",
                schema={
                    "type": "object",
                    "properties": {
                        "image_type": {"type": "string"},
                        "looks_real": {"type": "boolean"},
                    },
                    "required": ["image_type", "looks_real"],
                    "additionalProperties": False,
                },
                images=[llm.ImageInput(data_url=pmc.to_data_url(mime, blob))],
                prompt_version="urlprobe.v1",
            )
            print(f"  base64 fallback -> {_short(parsed, 120)}")
        except Exception as exc:
            print(f"  base64 fallback FAILED: {str(exc)[:160]}")

    print(f"\nURL mode succeeded for {url_ok}/{len(refs)} image URLs")
    print(f"total spent this run: ${client.spent_usd:.4f} (budget ${args.budget_usd})")
    total = conn.execute("SELECT SUM(cost_usd) AS c FROM llm_calls").fetchone()["c"]
    print(f"llm_calls ledger total: ${total or 0:.4f}")
    conn.close()
    return 0


def _captions(xml_text: str) -> list[tuple[str, str]]:
    from lxml import etree

    root = etree.fromstring(xml_text.encode("utf-8"))
    out = []
    for fig in root.iter("fig"):
        label = fig.find("label")
        caption = fig.find("caption")
        text = " ".join((caption.itertext() if caption is not None else []))
        text = " ".join(text.split())
        out.append((label.text if label is not None and label.text else "Fig", text))
    return out


def _image_refs(bundle: pmc.ArticleBundle, limit: int) -> list[pmc.ImageRef]:
    from lxml import etree

    root = etree.fromstring(bundle.xml_text.encode("utf-8"))
    refs, seen = [], set()
    for graphic in root.iter("graphic"):
        href = graphic.get("{http://www.w3.org/1999/xlink}href")
        if not href or href in seen:
            continue
        seen.add(href)
        ref = bundle.resolver(href)
        if ref.url:
            refs.append(ref)
        if len(refs) >= limit:
            break
    return refs


def _first_image_ref(bundle: pmc.ArticleBundle) -> pmc.ImageRef:
    refs = _image_refs(bundle, 1)
    if not refs:
        raise RuntimeError("no resolvable figure in bundle")
    return refs[0]


def _first_section_paragraph(xml_text: str) -> str:
    from lxml import etree

    root = etree.fromstring(xml_text.encode("utf-8"))
    for sec in root.iter("sec"):
        for p in sec.iter("p"):
            text = " ".join("".join(p.itertext()).split())
            if len(text) > 200:
                return text
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
