"""Frozen disease/finding term policy for balanced search, without vocabulary edits."""

from __future__ import annotations

import re

from . import diseases


def normalize_query(value: str) -> str:
    return " ".join(str(value).casefold().split())


def finding_synonyms(disease_key: str, finding: dict) -> list[str]:
    terms = list(finding.get("synonyms") or [])
    if finding.get("finding_key") != "anterior_uveitis":
        return terms
    generic = {"acute anterior uveitis", "iritis", "iridocyclitis"}
    psoriasis = {
        "psoriasis-associated anterior uveitis", "psoriatic uveitis",
        "uveitis associated with psoriasis",
    }
    psa = {"psoriatic uveitis", "uveitis associated with psoriatic arthritis"}
    allowed = generic | (psoriasis if disease_key == "psoriasis" else psa if disease_key == "psa" else set())
    return [term for term in terms if normalize_query(term) in allowed]


def disease_terms(disease: dict) -> list[str]:
    canonical = str(disease["name"]).strip()
    terms = [canonical]
    for synonym in disease.get("synonyms") or []:
        synonym = str(synonym).strip()
        letters = re.sub(r"[^A-Za-z]", "", synonym)
        if not synonym or normalize_query(synonym) == normalize_query(canonical):
            continue
        if len(letters) <= 6 and " " not in synonym and letters.upper() == letters:
            continue
        if normalize_query(synonym) in {"as", "axspa", "hla-b27", "hla b27"}:
            continue
        terms.append(synonym)
        break
    return terms


def search_queries(disease_key: str, finding: dict, disease: dict | None = None) -> list[str]:
    """Enumerate canonical pairs first; callers bound unattempted variants."""
    disease = disease or diseases.load_diseases()[disease_key]
    category = finding.get("category")
    modality = {
        "skin": "clinical photograph", "mucosa": "clinical photograph",
        "nail": "clinical photograph", "clinical_msk": "clinical photograph",
        "capillaroscopy": "capillaroscopy image", "histology": "histology micrograph",
        "radiology_xray": "radiograph", "ct": "CT scan", "mri": "MRI",
        "us": "ultrasound image", "echo": "echocardiogram", "eye": "ophthalmic image",
    }.get(category, "patient image")
    if disease_key == "as" and category == "eye":
        modality = "slit lamp ophthalmic image"
    labels = [str(finding.get("label") or finding["finding_key"]), *finding_synonyms(disease_key, finding)]
    out, seen = [], set()
    for label in labels:
        for name in disease_terms(disease):
            for suffix in (modality, "patient photograph figure caption"):
                query = " ".join(f"{name} {label} {suffix}".split())
                normalized = normalize_query(query)
                if normalized not in seen:
                    seen.add(normalized)
                    out.append(query)
    return out
