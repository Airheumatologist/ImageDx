"""Live, synthetic P2 evaluation; never writes to the production database.

Run: python -m src.visual_pilot.smoke_iceman
"""

from __future__ import annotations

import json
import time

from . import llm, prompts

MODEL = "opencode-go/iceman-contributor-free"
CASES = [
    ("untreated", "Clinical photograph of untreated Gottron papules in a patient with dermatomyositis.", "keep", False),
    ("chart", "Flowchart of the diagnostic algorithm for dermatomyositis.", "drop", False),
    ("animal", "Photograph of skin lesions in a mouse model of dermatomyositis.", "drop", False),
    ("treatment", "Clinical photographs before and after prednisone treatment, showing resolution of dermatomyositis rash.", "drop", False),
    ("vague", "Representative case.", "uncertain", False),
    ("copyright", "Clinical photograph of Gottron papules in dermatomyositis. Reproduced with permission from another source.", None, True),
]


def main() -> int:
    prompt = prompts.scoped("p2", ["dm"])
    payload = {"figures": [
        {"figure_id": name, "label": f"Figure {i + 1}", "caption": caption,
         "in_text_mentions": [], "article_title": "Dermatomyositis",
         "article_disease_keys": ["dm"]}
        for i, (name, caption, _, _) in enumerate(CASES)
    ]}
    client = llm.LLMClient(max_retries=0, timeout_seconds=45)
    start = time.monotonic()
    try:
        parsed, meta = client.call_json(
            "p2", MODEL, prompt.system, json.dumps(payload), prompt.schema,
            prompt_version=prompt.version, reasoning_effort="",
        )
    except Exception as exc:
        print(json.dumps({"model": MODEL, "passed": False,
                          "error_type": type(exc).__name__, "error": str(exc),
                          "seconds": round(time.monotonic() - start, 2)}, indent=2))
        return 1
    rows = parsed["results"]
    results = {row["figure_id"]: row for row in rows}
    checks = []
    for name, _, route, third_party in CASES:
        row = results.get(name, {})
        checks.append({"case": name, "passed": bool(row)
                       and (route is None or row["route"] == route)
                       and row["third_party"] is third_party,
                       "route": row.get("route"), "third_party": row.get("third_party")})
    passed = (all(check["passed"] for check in checks)
              and len(rows) == len(CASES) and len(results) == len(CASES))
    print(json.dumps({"model": MODEL, "passed": passed, "checks": checks,
                      "meta": meta, "seconds": round(time.monotonic() - start, 2)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
