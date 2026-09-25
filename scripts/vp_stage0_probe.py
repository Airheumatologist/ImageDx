"""Stage 0 probe for the visual pilot (spec §5 Stage 0).

Runs the source checks live on real PMC open-access *review* articles about
SLE / dermatomyositis / ankylosing spondylitis and prints a summary:

1. License source: S3 per-article metadata JSON (license_code) vs JATS XML
   <permissions> fallback (oa.fcgi is dead in the 2025 layout).
2. Figure access: direct public HTTPS URLs into s3://pmc-oa-opendata
   (per-article {pmcid}.{version}/ dirs) with needs_bytes fallback.
3. TIFF share across resolved figure files.
4. Reported separately by scripts/vp_smoke_llm.py: can the LLM fetch URLs?

Usage (from repo root):  python3 scripts/vp_stage0_probe.py [--articles 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lxml import etree

from src.visual_pilot import pmc

EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
DISEASE_QUERIES = {
    "sle": 'TITLE:"systemic lupus erythematosus" AND OPEN_ACCESS:y AND PUB_TYPE:"review"',
    "dm": 'TITLE:"dermatomyositis" AND OPEN_ACCESS:y AND PUB_TYPE:"review"',
    "as": 'TITLE:"ankylosing spondylitis" AND OPEN_ACCESS:y AND PUB_TYPE:"review"',
}
XLINK = "{http://www.w3.org/1999/xlink}href"


def find_candidates(query: str, page_size: int = 15) -> list[str]:
    resp = pmc.http_client().get(
        EPMC_SEARCH,
        params={"query": query, "format": "json", "pageSize": page_size, "resultType": "core"},
    )
    resp.raise_for_status()
    out = []
    for r in resp.json()["resultList"]["result"]:
        if r.get("pmcid") and r.get("pmcid") not in out:
            out.append(r["pmcid"])
    return out


def graphic_hrefs(xml_text: str) -> list[str]:
    try:
        root = etree.fromstring(xml_text.encode("utf-8"))
    except etree.XMLSyntaxError:
        return []
    hrefs = []
    for fig in root.iter("fig"):
        for graphic in fig.iter("graphic"):
            href = graphic.get(XLINK) or graphic.get("href")
            if href:
                hrefs.append(href)
    return hrefs


def probe_article(pmcid: str) -> dict:
    row = {"pmcid": pmcid}
    lic = pmc.get_license(pmcid)
    row["license_code"] = lic.code
    row["license_source"] = lic.source
    row["license_allows"] = pmc.license_allows(lic.code)
    if row["license_allows"] is None:
        return row  # excluded license: don't fetch figures for the probe
    bundle = pmc.get_article_bundle(pmcid)
    row["xml_chars"] = len(bundle.xml_text)
    row["has_s3_metadata"] = bool(bundle.metadata)
    hrefs = graphic_hrefs(bundle.xml_text)
    row["n_graphic_hrefs"] = len(hrefs)
    resolved, needs_bytes, fmts = 0, 0, []
    for href in hrefs:
        ref = bundle.resolver(href)
        if ref.url:
            resolved += 1
            if ref.format:
                fmts.append(ref.format.lower())
        else:
            needs_bytes += 1
    row["resolved_urls"] = resolved
    row["needs_bytes"] = needs_bytes
    row["formats"] = fmts
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles", type=int, default=10)
    parser.add_argument("--per-disease-candidates", type=int, default=12)
    args = parser.parse_args()

    picked: dict[str, list[str]] = {}
    for key, query in DISEASE_QUERIES.items():
        try:
            picked[key] = find_candidates(query, args.per_disease_candidates)
        except Exception as exc:
            print(f"! search failed for {key}: {exc}")
            picked[key] = []

    rows, seen = [], set()
    target = args.articles
    # Round-robin across diseases until we have `target` usable articles.
    idx = 0
    while len(rows) < target and idx < args.per_disease_candidates:
        progressed = False
        for key, cands in picked.items():
            if idx >= len(cands):
                continue
            progressed = True
            pmcid = cands[idx]
            if pmcid in seen:
                continue
            seen.add(pmcid)
            try:
                row = probe_article(pmcid)
            except Exception as exc:
                print(f"! probe failed for {pmcid}: {exc}")
                continue
            row["disease"] = key
            rows.append(row)
            print(f"  {pmcid} [{key}] lic={row['license_code']} ({row['license_source']}) "
                  f"allow={row['license_allows']} figs={row.get('n_graphic_hrefs', '-')} "
                  f"resolved={row.get('resolved_urls', '-')} needs={row.get('needs_bytes', '-')}")
            if len(rows) >= target:
                break
        idx += 1
        if not progressed:
            break

    print("\n===== STAGE 0 SUMMARY =====")
    print(json.dumps({"articles": rows}, indent=1))
    licenses = [r for r in rows if r.get("license_allows")]
    all_fmts = [f for r in licenses for f in r.get("formats", [])]
    tiffs = sum(1 for f in all_fmts if f in {"tif", "tiff"})
    total_resolved = sum(r.get("resolved_urls", 0) for r in licenses)
    total_hrefs = sum(r.get("n_graphic_hrefs", 0) for r in licenses)
    print("\narticles probed:", len(rows))
    print("license codes:", {c: sum(1 for r in rows if r["license_code"] == c) for c in {r["license_code"] for r in rows}})
    print("license sources:", {s: sum(1 for r in rows if r["license_source"] == s) for s in {r["license_source"] for r in rows}})
    print(f"graphic hrefs: {total_hrefs}, resolved to S3 URL: {total_resolved}, "
          f"needs_bytes: {total_hrefs - total_resolved}")
    print(f"formats seen: {sorted(set(all_fmts))}")
    print(f"TIFF share: {tiffs}/{len(all_fmts)} = {tiffs / len(all_fmts) if all_fmts else 0:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
