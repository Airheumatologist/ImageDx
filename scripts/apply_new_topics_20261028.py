"""Apply the reviewed October topic expansion once, preserving all existing IDs."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data/main_database_topics.json"
REVIEW = ROOT / "outputs/final_new_topics"
AUDIT = ROOT / "outputs/catalog_update_20261028"
MANIFEST = ROOT / "data/new_topics_20261028.json"

RENAMES = {
    "hypertrophic_obstructive_cardiomyopathy": "Hypertrophic Cardiomyopathy",
    "duchenne_muscular_dystrophy": "Dystrophinopathies (Duchenne and Becker Muscular Dystrophy)",
    "hemoglobin_h_disease": "Alpha-Thalassemia (Hemoglobin H Disease and Hemoglobin Bart’s Hydrops Fetalis)",
    "isthmic_spondylolisthesis": "Spondylolisthesis (Isthmic and Degenerative)",
    "conjunctival_lymphoma": "MALT Lymphoma and Other Conjunctival Lymphomas",
    "thoracic_empyema_necessitans": "Pleural Empyema (Including Empyema Necessitans)",
    "tracheobronchial_amyloidosis": "Localized Pulmonary Amyloidosis (Tracheobronchial and Nodular Forms)",
    "urachal_carcinoma": "Urachal Anomalies and Urachal Carcinoma",
    "urticaria": "Urticaria (Acute, Chronic Spontaneous and Chronic Inducible)",
    "subacute_combined_degeneration": "Subacute Combined Degeneration and Other Metabolic Dorsal Column Myelopathies",
    "tympanic_membrane_perforation": "Tympanic Membrane Perforation and Chronic Suppurative Otitis Media",
}

CONTEXT = {
    "duchenne_muscular_dystrophy": "Duchenne and Becker are distinct dystrophinopathy phenotypes. Existing early loss-of-ambulation and absent-dystrophin descriptions concern Duchenne; Becker generally has a milder, variable course and reduced or abnormal dystrophin. Label images by phenotype.",
    "conjunctival_lymphoma": "Organize MALT lymphoma into ocular and gastric sections. Other conjunctival lymphoma histologies remain a separate ocular section and must not be labeled MALT merely because of their site. Gastric appearances and the conjunctival salmon-patch appearance require diagnostic confirmation.",
    "subacute_combined_degeneration": "Vitamin B12 deficiency and nitrous-oxide-related functional B12 deficiency can cause subacute combined degeneration. Copper-deficiency myeloneuropathy is a separate etiologic differential with overlapping dorsal-column MRI findings, not a B12-deficiency subtype.",
    "urachal_carcinoma": "Keep congenital urachal cysts, sinuses and other remnants separate from urachal carcinoma in captions and subsections; an uncomplicated urachal cyst is not malignant.",
    "tracheobronchial_amyloidosis": "Distinguish the tracheobronchial and nodular localized pulmonary forms. Do not infer systemic amyloidosis from the lung appearance alone.",
    "isthmic_spondylolisthesis": "Distinguish isthmic disease with a pars defect from degenerative spondylolisthesis with an intact pars.",
    "tympanic_membrane_perforation": "Distinguish a dry or traumatic perforation from chronic suppurative otitis media with persistent middle-ear infection and drainage.",
    "sickle_cell_disease": "Renal papillary necrosis is a possible renal complication, not a synonym for sickle-cell disease; it also occurs with other causes.",
    "focal_segmental_glomerulosclerosis": "HIV-associated nephropathy belongs in an explicitly labeled etiologic section; enlarged echogenic kidneys alone are nonspecific and do not establish collapsing glomerulopathy.",
    "lumbar_disc_herniation": "Cauda equina syndrome is an emergency presentation with several possible causes, not a synonym for disc herniation. MRI shows the compressive cause; clinical findings establish the syndrome.",
    "lumbar_spinal_stenosis": "Cauda equina syndrome is an emergency presentation with several possible causes, not a synonym for spinal stenosis. MRI shows the compressive cause; clinical findings establish the syndrome.",
    "congenital_adrenal_hyperplasia": "Label steroidogenic defects separately: existing 21-hydroxylase-related descriptions do not apply to every form. Include lipoid and 17-alpha-hydroxylase-deficiency phenotypes without assigning all forms the same genital or adrenal appearance.",
    "hypertrophic_obstructive_cardiomyopathy": "Include obstructive, nonobstructive and apical phenotypes. Apical morphology does not imply left ventricular outflow obstruction.",
    "hemoglobin_h_disease": "Keep hemoglobin H disease and hemoglobin Bart’s hydrops fetalis as distinct alpha-thalassemia phenotypes; prenatal hydrops findings should not be attributed to all alpha-thalassemia.",
}

# These are manifestations/complications or differential diagnoses, not aliases
# for the entire parent disease. Their names still appear in labeled sections.
NOT_ALIASES = {
    "sickle_cell_disease", "focal_segmental_glomerulosclerosis",
    "vesicoureteral_reflux", "lumbar_disc_herniation", "lumbar_spinal_stenosis",
    "subacute_combined_degeneration", "aortic_dissection", "tetralogy_of_fallot",
}

# Searchable expanded names retain the short display titles from the review.
SEARCH_ALIASES = {
    "pomc_deficiency": ["Proopiomelanocortin deficiency"],
    "short_syndrome": ["Short stature, hyperextensibility, hernia, ocular depression, Rieger anomaly and teething delay"],
    "vipoma": ["Vasoactive intestinal peptide secreting tumor"],
    "ebv_positive_mucocutaneous_ulcer": ["Epstein-Barr virus positive mucocutaneous ulcer"],
    "systemic_alk_positive_anaplastic_large_cell_lymphoma": ["Anaplastic lymphoma kinase positive anaplastic large cell lymphoma"],
    "systemic_chronic_active_ebv_disease": ["Systemic chronic active Epstein-Barr virus disease"],
    "tempi_syndrome": ["Telangiectasias, elevated erythropoietin and erythrocytosis, monoclonal gammopathy, perinephric fluid collections and intrapulmonary shunting syndrome"],
    "hnf1b_related_kidney_disease": ["Hepatocyte nuclear factor 1 beta related kidney disease", "Renal cysts and diabetes syndrome"],
    "autoimmune_gfap_astrocytopathy": ["Autoimmune glial fibrillary acidic protein astrocytopathy"],
    "clippers": ["Chronic lymphocytic inflammation with pontine perivascular enhancement responsive to steroids"],
    "col4a1_related_disorders": ["Collagen type 4 alpha 1 related disorders"],
    "csf1r_related_leukoencephalopathy": ["Colony stimulating factor 1 receptor related leukoencephalopathy"],
    "htra1_related_cerebral_small_vessel_disease": ["High temperature requirement A1 related cerebral small vessel disease"],
    "lama2_related_muscular_dystrophy": ["Laminin alpha 2 related muscular dystrophy"],
    "spg11_related_hereditary_spastic_paraplegia": ["Spastic paraplegia type 11"],
    "atr_x_syndrome": ["Alpha-thalassemia X-linked intellectual disability syndrome"],
    "child_syndrome": ["Congenital hemidysplasia with ichthyosiform erythroderma and limb defects"],
    "lumbar_syndrome": ["Lower body hemangioma, urogenital anomalies and ulceration, spinal cord malformations, bony defects, anorectal, arterial and renal anomalies syndrome"],
    "microcephalic_osteodysplastic_primordial_dwarfism_type_ii": ["Microcephalic osteodysplastic primordial dwarfism type 2"],
    "card9_deficiency": ["Caspase recruitment domain containing protein 9 deficiency"],
    "copa_syndrome": ["Coatomer subunit alpha syndrome", "Autoimmune interstitial lung disease arthritis syndrome"],
    "foxn1_deficiency_nude_scid": ["Forkhead box N1 deficiency", "T-cell immunodeficiency with congenital alopecia and nail dystrophy"],
    "il_10_and_il_10_receptor_deficiency": ["Interleukin 10 and interleukin 10 receptor deficiency"],
    "mhc_class_i_deficiency_tap1_tap2": ["Major histocompatibility complex class 1 deficiency", "Bare lymphocyte syndrome type 1"],
    "plcg2_associated_disease_plaid_aplaid": ["Phospholipase C gamma 2 associated antibody deficiency and immune dysregulation", "Autoinflammation with phospholipase C gamma 2 associated antibody deficiency and immune dysregulation"],
    "whim_syndrome": ["Warts, hypogammaglobulinemia, infections and myelokathexis syndrome"],
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def append_unique(items, value):
    if value and value.casefold() not in {s.casefold() for s in items}:
        items.append(value)


def anchor_field(anchor):
    return "radiological_manifestations" if re.search(
        r"\b(MRI|CT|US|DMSA|PET|ultrasound|radiograph\w*|urography|films|imaging|echo\w*)\b",
        anchor, re.I,
    ) else "clinical_manifestations"


def main():
    if MANIFEST.exists():
        raise SystemExit("Expansion already recorded; do not apply the migration twice.")
    original = CATALOG.read_bytes()
    catalog = json.loads(original)
    before = {t["topic_id"]: copy.deepcopy(t) for rows in catalog["specialties"].values() for t in rows}
    additions = json.loads((REVIEW / "final_new_topics.json").read_text())["topics"]
    decisions = json.loads((REVIEW / "topic_decisions.json").read_text())["decisions"]
    assert len(before) == 1643 and len(additions) == 493
    ids = [t["topic_id"] for t in additions]
    assert len(set(ids)) == 493 and not set(ids) & before.keys()
    AUDIT.mkdir(parents=True, exist_ok=True)
    backup = AUDIT / "main_database_topics.before.json"
    if backup.exists():
        assert backup.read_bytes() == original, "Existing backup differs; inspect before applying."
    else:
        backup.write_bytes(original)
    existing = {t["topic_id"]: t for rows in catalog["specialties"].values() for t in rows}
    for key, title in RENAMES.items():
        topic = existing[key]
        old_name = topic["name"]
        append_unique(topic.setdefault("synonyms", []), old_name)
        topic["name"] = title
        # A former subtype's ontology identifier must not label a broader group.
        mappings = {k: topic.get(k) for k in ("mondo_id", "mesh_id") if topic.get(k)}
        if mappings:
            topic["prior_scope_ontology_mappings"] = [{"name": old_name, **mappings}]
            topic["mondo_id"] = topic["mesh_id"] = None
    existing["adamantinoma"]["synonyms"] = [
        s for s in existing["adamantinoma"]["synonyms"] if s.casefold() != "angiosarcoma of bone"
    ]
    for decision in decisions:
        if decision["action"] != "MERGE_EXISTING":
            continue
        label = decision["canonical_name"]
        anchor = re.sub(r"^Merge with [^;]+;\s*", "", decision["visual_anchor"])
        for key in decision["existing_parent_ids"]:
            topic = existing[key]
            section = {"name": label, "visual_anchor": anchor,
                       "relationship_note": decision["rationale"],
                       "source_decision_id": decision["decision_id"]}
            topic.setdefault("reviewed_sections", []).append(section)
            append_unique(topic.setdefault(anchor_field(anchor), []), f"{label}: {anchor}")
            if key not in NOT_ALIASES:
                append_unique(topic.setdefault("synonyms", []), label)
    for key, context in CONTEXT.items():
        existing[key]["scope_note"] = context
    # Retain old Duchenne material as a clearly labeled phenotype, not as a
    # blanket statement about Becker after broadening the heading.
    for field in ("clinical_manifestations", "radiological_manifestations", "pathology_manifestations"):
        old_values = set(before["duchenne_muscular_dystrophy"].get(field, []))
        existing["duchenne_muscular_dystrophy"][field] = [
            f"Duchenne phenotype: {v}" if v in old_values else v
            for v in existing["duchenne_muscular_dystrophy"].get(field, [])
        ]
    new_rows = []
    for reviewed in additions:
        name = reviewed["name"].replace(" (types I–III as one topic)", "")
        aliases = []
        for mention in reviewed["source_mentions"]:
            alias = mention["name"]
            if alias.casefold() != name.casefold() and "as one topic" not in alias:
                append_unique(aliases, alias)
        topic = {
            "topic_id": reviewed["topic_id"], "name": name,
            "specialty": reviewed["specialty"], "mondo_id": None, "mesh_id": None,
            "synonyms": aliases, "summary": None,
            "clinical_manifestations": [], "radiological_manifestations": [], "pathology_manifestations": [],
            "inclusion_rationale": reviewed["inclusion_rationale"],
            "visual_anchor": reviewed["visual_anchor"],
            "priority": reviewed["priority"], "curation_status": "pending_images",
            "added_on": "2026-10-28", "source_review": "Newtopics.docx editorial review 2026-10-28",
            "source_decision_ids": [m["decision_id"] for m in reviewed["source_mentions"]],
            "cross_specialties": reviewed["cross_specialties"],
            "supporting_reference_urls": reviewed["source_urls"],
            "reference_status": reviewed["reference_status"],
        }
        topic[anchor_field(reviewed["visual_anchor"])].append(reviewed["visual_anchor"])
        for alias in SEARCH_ALIASES.get(topic["topic_id"], []):
            append_unique(topic["synonyms"], alias)
        catalog["specialties"][topic["specialty"]].append(topic)
        new_rows.append(topic)
    meta = catalog["metadata"]
    meta.update(version="4.2.0", total_topics=2136, total_specialties=20, audited_on="2026-10-28")
    meta["description"] = "Master clinical visual-diagnosis topic catalog for physicians and medical students, including image-backed topics and reviewed additions awaiting image curation. Publication eligibility is determined separately from catalog membership."
    meta["curation_policy"]["inclusion_criteria"] = "Distinct diseases, syndromes, congenital disorders and selected clinically important injuries or complications with useful visible clinical or radiologic manifestations. Pathology may support these topics; pathology alone is insufficient for a new addition."
    meta["curation_policy"]["topic_grain"] = "Prefer one disease or syndrome per topic. Closely related phenotypes may share a clearly labeled parent; avoid duplicate site, subtype or manifestation pages. Retain distinct etiologies and diagnoses in labeled sections rather than equating them as synonyms."
    meta["curation_policy"]["exclusion_criteria"] = "Exclude nonvisual conditions without a useful clinical-image or radiologic teaching pattern, isolated symptoms or laboratory abnormalities, duplicate subtype/site pages and broad catch-all headings. Merge manifestations and complications into existing parents when a separate visual topic adds little. Retain the specifically reviewed additions and labeled parent expansions."
    meta["curation_policy"]["pending_images"] = "Reviewed additions remain in the master list while images are acquired. Catalog inclusion does not establish image availability, diagnostic verification or reuse permission."
    meta["latest_expansion"] = {"date": "2026-10-28", "baseline_topics": 1643,
        "added_topics": 493, "priority_counts": dict(Counter(t["priority"] for t in additions)),
        "review_artifact": "outputs/final_new_topics/final_new_topics.json",
        "new_topic_manifest": "data/new_topics_20261028.json"}
    changes = []
    for key, old in before.items():
        current = existing[key]
        if current != old:
            changes.append({"topic_id": key, "old_name": old["name"], "name": current["name"],
                            "changed_fields": [k for k in dict.fromkeys([*old, *current]) if old.get(k) != current.get(k)]})
    all_topics = [t for rows in catalog["specialties"].values() for t in rows]
    assert len(all_topics) == len({t["topic_id"] for t in all_topics}) == 2136
    assert set(before) <= {t["topic_id"] for t in all_topics}
    assert len(changes) == 32
    checkpoint = json.loads((ROOT / "data/visual_pilot/resume_20261027_checkpoint.json").read_text())
    from src.visual_pilot import config
    safe_names = [k for k in dir(config) if k.startswith("VP_") and not any(
        word in k for word in ("KEY", "PATH", "DIR", "URL"))
        and isinstance(getattr(config, k), (str, int, float, bool))]
    environment = {k: str(int(getattr(config, k))) if isinstance(getattr(config, k), bool)
                   else str(getattr(config, k)) for k in safe_names}
    environment.update(checkpoint["runtime_overrides"])
    for stage in ("TRIAGE", "EXTRACT", "DESCRIBE", "VOCAB"):
        environment[f"VP_{stage}_MODEL"] = checkpoint["models"]["text"]
    environment["VP_JUDGE_MODEL"] = checkpoint["models"]["vision"]
    manifest = {"name": "Approved new topics — 2026-10-28", "created_on": "2026-10-28",
        "total_topics": len(ids), "topics": ids,
        "baseline_topic_count": len(before), "catalog_topic_count": len(all_topics),
        "priority_counts": dict(Counter(t["priority"] for t in additions)),
        "specialty_counts": dict(Counter(t["specialty"] for t in additions)),
        "catalog_source": "data/main_database_topics.json", "review_source": "outputs/final_new_topics/final_new_topics.json",
        "reference_run": "data/visual_pilot/resume_20261027_checkpoint.json",
        "batch_size": checkpoint["batch_size"], "topic_image_target": checkpoint["topic_image_target"],
        "max_runtime_seconds": checkpoint["max_runtime_seconds"], "per_pair": 25, "max_rounds": 3,
        "skip_review_passes": True, "budget_usd": None,
        "models": checkpoint["models"], "openrouter_only": checkpoint["openrouter_only"],
        "environment": environment,
        "notes": "Only the 493 approved additions; corrected existing parents are excluded. Safe runtime settings captured from the recorded run and current configuration; no credentials. Vocabulary preparation precedes acquisition."}
    write_json(AUDIT / "change_report.json", {"before_sha256": hashlib.sha256(original).hexdigest(),
        "before_count": len(before), "after_count": len(all_topics), "added_ids": ids,
        "existing_ids_preserved": True, "existing_topics_corrected": changes,
        "renamed_existing_topics": len(RENAMES), "merged_source_mentions": 36})
    write_json(CATALOG, catalog)
    write_json(MANIFEST, manifest)
    (ROOT / "data/new_topic_ids_20261028.txt").write_text("\n".join(ids) + "\n")
    print(f"Catalog: {len(before)} → {len(all_topics)}; corrected {len(changes)} existing topics; added {len(ids)}.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
