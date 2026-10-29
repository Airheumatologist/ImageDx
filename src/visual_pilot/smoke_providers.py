"""Live synthetic checks of every LLM prompt and the restricted vision route.

Run ``python -m src.visual_pilot.smoke_providers``. No production DB writes.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time

import httpx
from PIL import Image, ImageDraw

from . import config, judge, llm, pmc, prompts, topic_vocab
from .smoke_iceman import CASES, MODEL

VISION_MODEL = "z-ai/glm-5.3-flash"


def _image() -> llm.ImageInput:
    image = Image.new("RGB", (320, 200), "white")
    draw = ImageDraw.Draw(image)
    # A clearly elongated rectangle avoids square/rectangle ambiguity.
    draw.rectangle((10, 65, 150, 125), fill="red")
    draw.ellipse((185, 40, 305, 160), fill="blue")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return llm.ImageInput(data_url="data:image/png;base64," + base64.b64encode(output.getvalue()).decode())


def _request(prompt, payload, model=MODEL, images=None) -> dict:
    return dict(stage=prompt.name, model=model, system=prompt.system,
                user_content=json.dumps(payload), schema=prompt.schema,
                prompt_version=prompt.version, reasoning_effort="low" if images else "",
                images=images or [])


def cases(include_clinical: bool = False) -> list[tuple[str, dict, object]]:
    source = "Gottron papules were present in 75% of patients with dermatomyositis."
    vocabulary = [{"finding_key": "gottron_papules", "label": "Gottron papules",
                   "synonyms": [], "category": "clinical"}]
    triage = {"figures": [dict(figure_id=name, label=name, caption=caption,
                              in_text_mentions=[], article_title="Dermatomyositis",
                              article_disease_keys=["dm"])
                          for name, caption, _, _ in CASES]}

    def triage_ok(reply):
        rows = reply["results"]
        by_id = {r["figure_id"]: r for r in rows}
        return len(rows) == len(CASES) and len(by_id) == len(CASES) and all(
            name in by_id and (route is None or by_id[name]["route"] == route)
            and by_id[name]["third_party"] is third
            for name, _, route, third in CASES)

    shapes_schema = {"type": "object", "properties": {
        k: {"type": "string"} for k in ("left_color", "left_shape", "right_color", "right_shape")},
        "required": ["left_color", "left_shape", "right_color", "right_shape"]}
    shapes = dict(stage="vision_shapes", model=VISION_MODEL,
                  system="Inspect the image. Return only JSON describing the two shapes from left to right.",
                  user_content="Identify each shape and its color.", schema=shapes_schema,
                  images=[_image()], reasoning_effort="low")
    fig = dict(figure_id="synthetic:diagram", label="Figure 1",
               caption="Schematic diagnostic pathway for dermatomyositis; colored geometric nodes, not patient images.",
               in_text_mentions_json="[]")
    article = dict(title="Dermatomyositis diagnostic pathway", primary_disease_keys_json='["dm"]')
    p3 = _request(prompts.scoped("p3", ["dm"]),
                  json.loads(judge.user_content(fig, article, vocabulary)), VISION_MODEL, [_image()])
    checks = [
        ("p2_triage", _request(prompts.scoped("p2", ["dm"]), triage), triage_ok),
        ("p4_extract", _request(prompts.scoped("p4", ["dm"]), {
            "sections": [{"heading": "Clinical findings", "text": source}], "vocabulary": vocabulary}),
         lambda r: any(a["finding_key"] == "gottron_papules" and a["pct_low"] == 75
                       and a["pct_high"] == 75 and a["quote"] in source for a in r["assertions"])),
        ("p5_describe", _request(prompts.P5, {
            "image_is": "the whole figure", "figure_caption": "Untreated Gottron papules over the knuckles in dermatomyositis.",
            "in_text_mentions": [], "disease": "Dermatomyositis", "modality": "clinical_photo",
            "body_site": "hands", "findings": ["Gottron papules"], "sections": []}),
         lambda r: bool(r["title"].strip()) and 0 < len(r["description"]) <= 300 and not r["treatment_related"]),
        ("p6_vocab", _request(topic_vocab.P6, {
            "topic": "Dermatomyositis", "clinical_manifestations": ["Gottron papules", "Heliotrope rash", "Shawl sign"],
            "radiological_manifestations": [], "pathology_manifestations": []}),
         lambda r: len(topic_vocab.post_validate(
             {"name": "Dermatomyositis", "topic_id": "dm"}, r)) >= 3),
        ("p8_mimic", _request(prompts.P8, {
            "topic": "Dermatomyositis", "topic_synonyms": [], "topic_subtypes": [],
            "article_title": "Psoriasis mimicking dermatomyositis",
            "abstract": "The patient had biopsy-confirmed psoriasis. Dermatomyositis was excluded.",
            "figure_caption": "Psoriatic plaques over the knuckles.", "in_text_mentions": []}),
         lambda r: r["verdict"] == "other_disease" and "psoriasis" in r["actual_diagnosis"].lower()),
        ("vision_shapes", shapes,
         lambda r: r["left_color"].lower() == "red" and r["left_shape"].lower() == "rectangle"
         and r["right_color"].lower() == "blue" and r["right_shape"].lower() in {"circle", "ellipse", "oval"}),
        ("p3_diagram", p3,
         lambda r: r["figure_id"] == "synthetic:diagram" and bool(r["panels"])
         and all(not p["include"] for p in r["panels"])),
    ]
    if include_clinical:
        # An already published CC-BY image from the repository's exported
        # library. Read only; this never modifies the database or export.
        exported = json.loads((config.REPO_ROOT / "site/data/dm/panels.json").read_text())
        panel = next(p for p in exported["panels"]
                     if p["figure_id"] == "PMC7139382:ijms-21-02178-f006")
        response = httpx.get(panel["image"], timeout=30)
        response.raise_for_status()
        mime, image_bytes = pmc.prepare_for_llm(response.content)
        figure = dict(figure_id=panel["figure_id"], label=panel["figure_label"],
                      caption=panel["figure_caption"],
                      in_text_mentions_json=json.dumps(panel["in_text_mentions"]))
        article = dict(title=panel["article_title"], primary_disease_keys_json='["dm"]')
        vocabulary = [{"finding_key": "v_sign", "label": "V-sign", "synonyms": [], "category": "skin"}]
        request = _request(prompts.scoped("p3", ["dm"]),
                           json.loads(judge.user_content(figure, article, vocabulary)),
                           VISION_MODEL, [llm.ImageInput(data_url=pmc.to_data_url(mime, image_bytes))])
        checks.append(("p3_public_clinical_photo", request,
                       lambda r: r["figure_id"] == panel["figure_id"] and any(
                           p["include"] and p["disease_key"] == "dm"
                           and p["modality"] == "clinical_photo"
                           and any(f["finding_key"] == "v_sign" for f in p["findings"])
                           for p in r["panels"])))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical", action="store_true", help="Also test an exported public CC-BY patient photograph")
    parser.add_argument("--vision-only", action="store_true", help="Run only the image checks")
    args = parser.parse_args()
    client = llm.LLMClient(max_retries=0, concurrency=3, timeout_seconds=60, budget_usd=0.25)
    checks = cases(args.clinical)
    if args.vision_only:
        checks = [check for check in checks if check[1].get("images")]
    reports = []
    started = time.monotonic()
    for result in client.iter_many(req for _, req, _ in checks):
        name, req, verify = checks[result.index]
        if result.error:
            report = {"case": name, "model": req["model"], "passed": False,
                      "error": str(result.error), "error_type": type(result.error).__name__}
        else:
            report = {"case": name, "model": req["model"], "passed": bool(verify(result.parsed)),
                      "meta": result.meta, "reply": result.parsed}
            if req["model"] == VISION_MODEL:
                report["passed"] &= result.meta.get("upstream_provider") == "OpenInference"
        reports.append(report)
        print(json.dumps(report), flush=True)
    passed = len(reports) == len(checks) and all(r["passed"] for r in reports)
    print(json.dumps({"passed": passed, "checks": len(reports),
                      "seconds": round(time.monotonic() - started, 2),
                      "cost_usd": client.spent_usd,
                      "openrouter_only": list(config.OPENROUTER_PROVIDERS)}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
