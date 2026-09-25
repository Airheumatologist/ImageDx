"""
High-impact and specialty journal configuration for prioritization.
These journals should be prioritized when selecting full-text articles.

This module provides multiple journal identification methods:
1. NLM Unique IDs (most stable, preferred) - from NLM Catalog
2. Normalized journal name matching (fallback)
"""

import re
from typing import Any, Literal, Optional

from anyascii import anyascii


def _normalize_journal_name(value: Any) -> str:
    """Normalize journal names for exact token-boundary set membership checks."""
    return re.sub(r"[^a-z0-9]+", " ", anyascii(str(value or "")).lower()).strip()

# =============================================================================
# NLM UNIQUE IDs (PREFERRED IDENTIFIER)
# =============================================================================
# These are the official NLM Unique IDs from the NLM Catalog
# https://www.ncbi.nlm.nih.gov/nlmcatalog/
# NLM Unique IDs are the most stable journal identifiers available

_ALL_PRIORITY_JOURNALS_NLM = {
    # General/Multi-specialty High-Impact
    '7501160',  # JAMA
    '0255562',  # New England Journal of Medicine
    '2985213',  # The Lancet (fixed: removed erroneous 'R')
    '8900488',  # BMJ
    '101589534',  # JAMA Internal Medicine
    '0372351',  # Annals of Internal Medicine
    '101231360',  # PLOS Medicine
    '9502015',  # Nature Medicine
    '101729235',  # JAMA Network Open
    '101613665',  # The Lancet Global Health
    '0267200',  # The American Journal of Medicine
    '0405543',  # Mayo Clinic Proceedings
    '8904841',  # Journal of Internal Medicine
    '9711805',  # CMAJ
    '101190723',  # BMC Medicine
    '8605834',  # Journal of General Internal Medicine
    '100959882',  # Journal of Medical Internet Research (JMIR)
    '7802877',  # The Journal of Clinical Investigation
    '8303128',  # Health Affairs
    '101769500',  # JAMA Health Forum
    '101672103',  # Nature Reviews. Disease Primers
    '100909747',  # Cochrane Database of Systematic Reviews
    '101505086',  # Science Translational Medicine
    '101751302',  # The Lancet Digital Health
    '101731738',  # npj Digital Medicine

    # Family Medicine / Primary Care
    '101167762',  # Annals of Family Medicine
    '1272646',  # American Family Physician
    '101256526',  # Journal of the American Board of Family Medicine
    '9005323',  # British Journal of General Practice

    # Public Health / Epidemiology
    '1254074',  # American Journal of Public Health
    '8704773',  # American Journal of Preventive Medicine
    '101699003',  # The Lancet Public Health
    '7910653',  # American Journal of Epidemiology
    '7802429',  # MMWR. Morbidity and Mortality Weekly Report
    '0330411',  # Environmental Health Perspectives

    # Geriatrics
    '7503062',  # Journal of the American Geriatrics Society

    # Palliative Care
    '9808462',  # Journal of Palliative Medicine
    '9008229',  # American Journal of Hospice and Palliative Care

    # Medical Education
    '8904605',  # Academic Medicine
    '7909593',  # Medical Teacher

    # Hospital Medicine
    '101271025',  # Journal of Hospital Medicine

    # Cardiology
    '0147763',  # Circulation
    '8301365',  # Journal of the American College of Cardiology (JACC)
    '8006263',  # European Heart Journal
    '101598241',  # JACC: Heart Failure
    '0047103',  # Circulation Research
    '101676033',  # JAMA Cardiology
    '101580524',  # Journal of the American Heart Association (JAHA)
    '101200317',  # Heart Rhythm
    '101500075',  # Nature Reviews Cardiology
    '0207277',  # American Journal of Cardiology
    '101467004',  # JACC: Cardiovascular Interventions

    # Emergency Medicine
    '8007457',  # Emergency Medicine Clinics of North America
    '101244046',  # Emergency Medicine International  # Note: lower impact but listed
    '8002646',  # Annals of Emergency Medicine
    '9205910',  # Academic Emergency Medicine

    # Rheumatology
    '101623795',  # Arthritis & Rheumatology
    '0372355',  # Annals of the Rheumatic Diseases
    '101132864',  # Arthritis Research & Therapy
    '7605992',  # The Journal of Rheumatology
    '101215956',  # Clinical Rheumatology
    '100897526',  # Rheumatology (Oxford)
    '101500080',  # Nature Reviews. Rheumatology

    # Pulmonology/Respiratory
    '101088569',  # Respiratory Research
    '9421642',  # American Journal of Respiratory and Critical Care Medicine
    '8803460',  # European Respiratory Journal
    '15035010',  # Chest
    '0413533',  # Thorax
    '101605555',  # The Lancet Respiratory Medicine

    # Nephrology
    '9013836',  # Journal of the American Society of Nephrology
    '0077425',  # Kidney International
    '8612470',  # American Journal of Kidney Diseases
    '101271570',  # Clinical Journal of the American Society of Nephrology
    '8706402',  # Nephrology Dialysis Transplantation
    '101500081',  # Nature Reviews. Nephrology

    # Hematology
    '7603509',  # Blood
    '0372542',  # British Journal of Haematology
    '0406014',  # Haematologica
    '101690679',  # Blood Advances
    '101643584',  # The Lancet Haematology

    # Neurology
    '0401060',  # Neurology
    '101589536',  # JAMA Neurology
    '7707447',  # Annals of Neurology
    '101139309',  # The Lancet Neurology
    '0235266',  # Stroke
    '2985191',  # Journal of Neurology, Neurosurgery & Psychiatry (fixed: removed 'R')
    '0372537',  # Brain
    '101500072',  # Nature Reviews. Neurology
    '8610688',  # Movement Disorders

    # Neurosurgery
    '0253357',  # Journal of Neurosurgery
    '7802914',  # Neurosurgery

    # Gastroenterology
    '0374630',  # Gastroenterology
    '2985108',  # Gut (fixed: removed 'R')
    '101241154',  # Clinical Gastroenterology and Hepatology
    '8912027',  # Alimentary Pharmacology & Therapeutics
    '8302946',  # Hepatology
    '8503886',  # Journal of Hepatology
    '0421030',  # American Journal of Gastroenterology
    '101690683',  # The Lancet Gastroenterology & Hepatology
    '101500079',  # Nature Reviews. Gastroenterology & Hepatology

    # Endocrinology
    '7805975',  # Diabetes Care
    '0375362',  # The Journal of Clinical Endocrinology & Metabolism
    '0370670',  # Diabetologia
    '0372763',  # Diabetes
    '101618821',  # The Lancet Diabetes & Endocrinology
    '101500078',  # Nature Reviews. Endocrinology

    # Infectious Disease
    '9203213',  # Clinical Infectious Diseases
    '7708673',  # The Journal of Infectious Diseases
    '8407191',  # Antimicrobial Agents and Chemotherapy
    '101130150',  # The Lancet Infectious Diseases
    '9508155',  # Emerging Infectious Diseases

    # Oncology
    '8309333',  # Journal of Clinical Oncology
    '0374236',  # Cancer
    '101652861',  # JAMA Oncology
    '100957246',  # The Lancet Oncology
    '0370647',  # CA: A Cancer Journal for Clinicians
    '9007735',  # Annals of Oncology
    '7503089',  # Journal of the National Cancer Institute (JNCI)
    '101124168',  # Nature Reviews. Cancer
    '0370635',  # British Journal of Cancer

    # Critical Care
    '0355501',  # Critical Care Medicine
    '7704851',  # Intensive Care Medicine
    '9801906',  # Critical Care

    # Dermatology
    '7708717',  # Journal of the American Academy of Dermatology
    '101589530',  # JAMA Dermatology
    '0004041',  # British Journal of Dermatology
    '0375417',  # Journal of Investigative Dermatology
    '0241006',  # Dermatologic Surgery

    # Ophthalmology
    '7708242',  # Ophthalmology
    '101589539',  # JAMA Ophthalmology
    '0370507',  # American Journal of Ophthalmology
    '0370657',  # British Journal of Ophthalmology
    '8309910',  # Retina

    # Otolaryngology (ENT)
    '0376432',  # Laryngoscope
    '7807629',  # Otolaryngology-Head and Neck Surgery
    '101589542',  # JAMA Otolaryngology-Head & Neck Surgery
    '101558145',  # International Forum of Allergy & Rhinology

    # Orthopedics
    '0375355',  # Journal of Bone and Joint Surgery
    '0240663',  # Clinical Orthopaedics and Related Research
    '8309867',  # Journal of Orthopaedic Research
    '0432520',  # British Journal of Sports Medicine
    '7609541',  # The American Journal of Sports Medicine
    '101599229',  # The Bone & Joint Journal

    # Psychiatry
    '0370522',  # American Journal of Psychiatry
    '7806866',  # Journal of Clinical Psychiatry
    '101589550',  # JAMA Psychiatry
    '101638123',  # The Lancet Psychiatry
    '101189643',  # World Psychiatry
    '0342367',  # British Journal of Psychiatry
    '0213264',  # Biological Psychiatry

    # Radiology
    '0401260',  # Radiology
    '7708173',  # American Journal of Roentgenology
    '100883023',  # European Radiology
    '9502789',  # Journal of Vascular and Interventional Radiology
    '8302501',  # Radiographics

    # Pathology
    '7707903',  # American Journal of Surgical Pathology
    '101468356',  # Modern Pathology

    # Anesthesiology
    '0063126',  # Anesthesiology
    '0148047',  # Anesthesia & Analgesia
    '0372541',  # British Journal of Anaesthesia

    # General Surgery
    '0372654',  # Annals of Surgery
    '0372552',  # British Journal of Surgery
    '101589553',  # JAMA Surgery
    '8407742',  # Journal of Vascular Surgery
    '9431305',  # Journal of the American College of Surgeons (JACS)

    # Allergy & Immunology
    '1275002',  # The Journal of Allergy and Clinical Immunology

    # Pediatrics
    '0376422',  # Pediatrics
    '0375416',  # Journal of Pediatrics
    '101589544',  # JAMA Pediatrics
    '101712925',  # The Lancet Child & Adolescent Health
    '9102136',  # Journal of Adolescent Health
    '0372434',  # Archives of Disease in Childhood

    # Obstetrics & Gynecology
    '0372343',  # Obstetrics & Gynecology
    '0370476',  # American Journal of Obstetrics & Gynecology
    '8701199',  # Human Reproduction
    '9108340',  # Ultrasound in Obstetrics & Gynecology
    '100935741',  # BJOG
    '0372772',  # Fertility and Sterility

    # Urology
    '0376372',  # Journal of Urology
    '7512719',  # European Urology
    '100886721',  # BJU International

    # Plastic Surgery
    '0370656',  # Plastic and Reconstructive Surgery

    # Radiation Oncology
    '7706812',  # International Journal of Radiation Oncology

    # Nuclear Medicine
    '7611109',  # Journal of Nuclear Medicine

    # Medical Genetics / Genomics
    '0370475',  # The American Journal of Human Genetics
    '9815831',  # Genetics in Medicine
    '101500077',  # Nature Reviews. Genetics

    # Clinical Pharmacology
    '0372741',  # Clinical Pharmacology & Therapeutics

    # Transplantation Medicine
    '100968638',  # American Journal of Transplantation

    # Physical Medicine & Rehabilitation
    '2985158',  # Archives of Physical Medicine and Rehabilitation (fixed: removed 'R')

    # Pain Medicine
    '7504682',  # Pain

    # Sleep Medicine
    '7809084',  # Sleep

    # Quality Improvement, Patient Safety & Implementation Science
    '101258411',  # Implementation Science
    '101085951',  # BMJ Quality & Safety

    # Occupational & Environmental Medicine
    '9800481',  # Occupational and Environmental Medicine
}

# Explicit general/high-impact tier used for score boosts.
GENERAL_HIGH_IMPACT_NLM = {
    '7501160',    # JAMA
    '0255562',    # New England Journal of Medicine
    '2985213',    # The Lancet
    '8900488',    # BMJ
    '101589534',  # JAMA Internal Medicine
    '0372351',    # Annals of Internal Medicine
    '101231360',  # PLOS Medicine
    '9502015',    # Nature Medicine
    '101729235',  # JAMA Network Open
    '101613665',  # The Lancet Global Health
    '0267200',    # The American Journal of Medicine
    '0405543',    # Mayo Clinic Proceedings
    '8904841',    # Journal of Internal Medicine
    '9711805',    # CMAJ
    '101190723',  # BMC Medicine
    '8605834',    # Journal of General Internal Medicine
    '100959882',  # Journal of Medical Internet Research (JMIR)
    '7802877',    # The Journal of Clinical Investigation
    '8303128',    # Health Affairs
    '101769500',  # JAMA Health Forum
    '101672103',  # Nature Reviews. Disease Primers
    '100909747',  # Cochrane Database of Systematic Reviews
    '101505086',  # Science Translational Medicine
    '101751302',  # The Lancet Digital Health
    '101731738',  # npj Digital Medicine
}

SPECIALTY_JOURNALS_NLM = _ALL_PRIORITY_JOURNALS_NLM - GENERAL_HIGH_IMPACT_NLM

if not GENERAL_HIGH_IMPACT_NLM.issubset(_ALL_PRIORITY_JOURNALS_NLM):
    raise ValueError("GENERAL_HIGH_IMPACT_NLM must be a subset of _ALL_PRIORITY_JOURNALS_NLM")

# =============================================================================
# JOURNAL NAME MATCHING (FALLBACK)
# =============================================================================
# Normalized journal-name aliases for runtime matching by journal title

_PRIORITY_JOURNAL_NAME_TEXT = """
JAMA
New England Journal of Medicine
The Lancet
BMJ
JAMA Internal Medicine
Annals of Internal Medicine
PLOS Medicine
Nature Medicine
JAMA Network Open
The Lancet Global Health
The American Journal of Medicine
Mayo Clinic Proceedings
Journal of Internal Medicine
CMAJ
BMC Medicine
Journal of General Internal Medicine
Journal of Medical Internet Research (JMIR)
The Journal of Clinical Investigation
Health Affairs
JAMA Health Forum
Nature Reviews. Disease Primers
Cochrane Database of Systematic Reviews
Science Translational Medicine
The Lancet Digital Health
npj Digital Medicine
Annals of Family Medicine
American Family Physician
Journal of the American Board of Family Medicine
British Journal of General Practice
American Journal of Public Health
American Journal of Preventive Medicine
The Lancet Public Health
American Journal of Epidemiology
MMWR. Morbidity and Mortality Weekly Report
Environmental Health Perspectives
Journal of the American Geriatrics Society
Journal of Palliative Medicine
American Journal of Hospice and Palliative Care
Academic Medicine
Medical Teacher
Journal of Hospital Medicine
Circulation
Journal of the American College of Cardiology (JACC)
European Heart Journal
JACC: Heart Failure
Circulation Research
JAMA Cardiology
Journal of the American Heart Association (JAHA)
Heart Rhythm
Nature Reviews Cardiology
American Journal of Cardiology
JACC: Cardiovascular Interventions
Emergency Medicine Clinics of North America
Emergency Medicine International
Annals of Emergency Medicine
Academic Emergency Medicine
Arthritis & Rheumatology
Annals of the Rheumatic Diseases
Arthritis Research & Therapy
The Journal of Rheumatology
Clinical Rheumatology
Rheumatology (Oxford)
Nature Reviews. Rheumatology
Respiratory Research
American Journal of Respiratory and Critical Care Medicine
European Respiratory Journal
Chest
Thorax
The Lancet Respiratory Medicine
Journal of the American Society of Nephrology
Kidney International
American Journal of Kidney Diseases
Clinical Journal of the American Society of Nephrology
Nephrology Dialysis Transplantation
Nature Reviews. Nephrology
Blood
British Journal of Haematology
Haematologica
Blood Advances
The Lancet Haematology
Neurology
JAMA Neurology
Annals of Neurology
The Lancet Neurology
Stroke
Journal of Neurology, Neurosurgery & Psychiatry
Brain
Nature Reviews. Neurology
Movement Disorders
Journal of Neurosurgery
Neurosurgery
Gastroenterology
Gut
Clinical Gastroenterology and Hepatology
Alimentary Pharmacology & Therapeutics
Hepatology
Journal of Hepatology
American Journal of Gastroenterology
The Lancet Gastroenterology & Hepatology
Nature Reviews. Gastroenterology & Hepatology
Diabetes Care
The Journal of Clinical Endocrinology & Metabolism
Diabetologia
Diabetes
The Lancet Diabetes & Endocrinology
Nature Reviews. Endocrinology
Clinical Infectious Diseases
The Journal of Infectious Diseases
Antimicrobial Agents and Chemotherapy
The Lancet Infectious Diseases
Emerging Infectious Diseases
Journal of Clinical Oncology
Cancer
JAMA Oncology
The Lancet Oncology
CA: A Cancer Journal for Clinicians
Annals of Oncology
Journal of the National Cancer Institute (JNCI)
Nature Reviews. Cancer
British Journal of Cancer
Critical Care Medicine
Intensive Care Medicine
Critical Care
Journal of the American Academy of Dermatology
JAMA Dermatology
British Journal of Dermatology
Journal of Investigative Dermatology
Dermatologic Surgery
Ophthalmology
JAMA Ophthalmology
American Journal of Ophthalmology
British Journal of Ophthalmology
Retina
Laryngoscope
Otolaryngology-Head and Neck Surgery
JAMA Otolaryngology-Head & Neck Surgery
International Forum of Allergy & Rhinology
Journal of Bone and Joint Surgery
Clinical Orthopaedics and Related Research
Journal of Orthopaedic Research
British Journal of Sports Medicine
The American Journal of Sports Medicine
The Bone & Joint Journal
American Journal of Psychiatry
Journal of Clinical Psychiatry
JAMA Psychiatry
The Lancet Psychiatry
World Psychiatry
British Journal of Psychiatry
Biological Psychiatry
Radiology
American Journal of Roentgenology
European Radiology
Journal of Vascular and Interventional Radiology
Radiographics
American Journal of Surgical Pathology
Modern Pathology
Anesthesiology
Anesthesia & Analgesia
British Journal of Anaesthesia
Annals of Surgery
British Journal of Surgery
JAMA Surgery
Journal of Vascular Surgery
Journal of the American College of Surgeons (JACS)
The Journal of Allergy and Clinical Immunology
Pediatrics
Journal of Pediatrics
JAMA Pediatrics
The Lancet Child & Adolescent Health
Journal of Adolescent Health
Archives of Disease in Childhood
Obstetrics & Gynecology
American Journal of Obstetrics & Gynecology
Human Reproduction
Ultrasound in Obstetrics & Gynecology
BJOG
Fertility and Sterility
Journal of Urology
European Urology
BJU International
Plastic and Reconstructive Surgery
International Journal of Radiation Oncology
Journal of Nuclear Medicine
The American Journal of Human Genetics
Genetics in Medicine
Nature Reviews. Genetics
Clinical Pharmacology & Therapeutics
American Journal of Transplantation
Archives of Physical Medicine and Rehabilitation
Pain
Sleep
Implementation Science
BMJ Quality & Safety
Occupational and Environmental Medicine
"""

PRIORITY_JOURNAL_NAMES = {
    _normalize_journal_name(name)
    for name in _PRIORITY_JOURNAL_NAME_TEXT.splitlines()
    if name.strip()
}

_GENERAL_HIGH_IMPACT_NAME_TEXT = """
JAMA
New England Journal of Medicine
The Lancet
BMJ
JAMA Internal Medicine
Annals of Internal Medicine
PLOS Medicine
Nature Medicine
JAMA Network Open
The Lancet Global Health
The American Journal of Medicine
Mayo Clinic Proceedings
Journal of Internal Medicine
CMAJ
BMC Medicine
Journal of General Internal Medicine
Journal of Medical Internet Research (JMIR)
The Journal of Clinical Investigation
Health Affairs
JAMA Health Forum
Nature Reviews. Disease Primers
Cochrane Database of Systematic Reviews
Science Translational Medicine
The Lancet Digital Health
npj Digital Medicine
"""

GENERAL_HIGH_IMPACT_NAMES = {
    _normalize_journal_name(name)
    for name in _GENERAL_HIGH_IMPACT_NAME_TEXT.splitlines()
    if name.strip()
}
SPECIALTY_JOURNALS_NAMES = PRIORITY_JOURNAL_NAMES - GENERAL_HIGH_IMPACT_NAMES

if not GENERAL_HIGH_IMPACT_NAMES.issubset(PRIORITY_JOURNAL_NAMES):
    raise ValueError("GENERAL_HIGH_IMPACT_NAMES must be a subset of PRIORITY_JOURNAL_NAMES")

# =============================================================================
# GUIDELINE SOCIETY DETECTION
# =============================================================================

_NORMALIZE_NON_ALNUM = re.compile(r"[^a-z0-9]+")

_GUIDELINE_SOCIETY_REGISTRY = {
    "ACC": {
        "aliases": ["ACC", "American College of Cardiology"],
        "ambiguous_acronym": False,
    },
    "AHA": {
        "aliases": ["AHA", "American Heart Association"],
        "ambiguous_acronym": False,
    },
    "ESC": {
        "aliases": ["ESC", "European Society of Cardiology"],
        "ambiguous_acronym": False,
    },
    "HRS": {
        "aliases": ["HRS", "Heart Rhythm Society"],
        "ambiguous_acronym": False,
    },
    "ACCP": {
        "aliases": ["ACCP", "American College of Chest Physicians"],
        "ambiguous_acronym": False,
    },
    "ACP": {
        "aliases": ["ACP", "American College of Physicians"],
        "ambiguous_acronym": False,
    },
    "ACR": {
        "aliases": ["ACR", "American College of Rheumatology"],
        "ambiguous_acronym": False,
    },
    "EULAR": {
        "aliases": ["EULAR", "European Alliance of Associations for Rheumatology"],
        "ambiguous_acronym": False,
    },
    "IDSA": {
        "aliases": ["IDSA", "Infectious Diseases Society of America"],
        "ambiguous_acronym": False,
    },
    "ATS": {
        "aliases": ["ATS", "American Thoracic Society"],
        "ambiguous_acronym": False,
    },
    "ERS": {
        "aliases": ["ERS", "European Respiratory Society"],
        "ambiguous_acronym": False,
    },
    "ASCO": {
        "aliases": ["ASCO", "American Society of Clinical Oncology"],
        "ambiguous_acronym": False,
    },
    "NCCN": {
        "aliases": ["NCCN", "National Comprehensive Cancer Network"],
        "ambiguous_acronym": False,
    },
    "ESMO": {
        "aliases": ["ESMO", "European Society for Medical Oncology"],
        "ambiguous_acronym": False,
    },
    "AAD": {
        "aliases": ["AAD", "American Academy of Dermatology"],
        "ambiguous_acronym": False,
    },
    "AAO": {
        "aliases": ["AAO", "American Academy of Ophthalmology"],
        "ambiguous_acronym": False,
    },
    "AAO-HNS": {
        "aliases": ["AAO-HNS", "American Academy of Otolaryngology Head and Neck Surgery"],
        "ambiguous_acronym": False,
    },
    "AAOS": {
        "aliases": ["AAOS", "American Academy of Orthopaedic Surgeons"],
        "ambiguous_acronym": False,
    },
    "APA": {
        "aliases": ["APA", "American Psychiatric Association"],
        "ambiguous_acronym": False,
    },
    "RSNA": {
        "aliases": ["RSNA", "Radiological Society of North America"],
        "ambiguous_acronym": False,
    },
    "CAP": {
        "aliases": ["CAP", "College of American Pathologists"],
        "ambiguous_acronym": True,
    },
    "ASA": {
        "aliases": ["ASA", "American Society of Anesthesiologists"],
        "ambiguous_acronym": True,
    },
    "ACS": {
        "aliases": ["ACS", "American College of Surgeons"],
        "ambiguous_acronym": True,
    },
    "AAP": {
        "aliases": ["AAP", "American Academy of Pediatrics"],
        "ambiguous_acronym": False,
    },
    "ACOG": {
        "aliases": ["ACOG", "American College of Obstetricians and Gynecologists"],
        "ambiguous_acronym": False,
    },
    "AAFP": {
        "aliases": ["AAFP", "American Academy of Family Physicians"],
        "ambiguous_acronym": False,
    },
    "AAPMR": {
        "aliases": ["AAPMR", "American Academy of Physical Medicine and Rehabilitation"],
        "ambiguous_acronym": False,
    },
    "AUA": {
        "aliases": ["AUA", "American Urological Association"],
        "ambiguous_acronym": False,
    },
    "ASPS": {
        "aliases": ["ASPS", "American Society of Plastic Surgeons"],
        "ambiguous_acronym": False,
    },
    "ASTRO": {
        "aliases": ["ASTRO", "American Society for Radiation Oncology"],
        "ambiguous_acronym": False,
    },
    "SNMMI": {
        "aliases": ["SNMMI", "Society of Nuclear Medicine and Molecular Imaging"],
        "ambiguous_acronym": False,
    },
    "ACG": {
        "aliases": ["ACG", "American College of Gastroenterology"],
        "ambiguous_acronym": False,
    },
    "AGA": {
        "aliases": ["AGA", "American Gastroenterological Association"],
        "ambiguous_acronym": False,
    },
    "ADA": {
        "aliases": ["ADA", "American Diabetes Association"],
        "ambiguous_acronym": True,
    },
    "AACE": {
        "aliases": ["AACE", "American Association of Clinical Endocrinology"],
        "ambiguous_acronym": False,
    },
    "AAN": {
        "aliases": ["AAN", "American Academy of Neurology"],
        "ambiguous_acronym": False,
    },
    "ASN": {
        "aliases": ["ASN", "American Society of Nephrology"],
        "ambiguous_acronym": False,
    },
    "KDIGO": {
        "aliases": ["KDIGO", "Kidney Disease Improving Global Outcomes"],
        "ambiguous_acronym": False,
    },
    "ASH": {
        "aliases": ["ASH", "American Society of Hematology"],
        "ambiguous_acronym": False,
    },
    "ACEP": {
        "aliases": ["ACEP", "American College of Emergency Physicians"],
        "ambiguous_acronym": False,
    },
    "SCCM": {
        "aliases": ["SCCM", "Society of Critical Care Medicine"],
        "ambiguous_acronym": False,
    },
    "USPSTF": {
        "aliases": ["USPSTF", "United States Preventive Services Task Force"],
        "ambiguous_acronym": False,
    },
    "ACE": {
        "aliases": ["ACE", "American College of Endocrinology"],
        "ambiguous_acronym": True,
    },
}

GUIDELINE_CONTEXT_TERMS = {
    "guideline",
    "guidelines",
    "practice guideline",
    "practice guidelines",
    "clinical practice guideline",
    "clinical practice guidelines",
    "recommendation",
    "recommendations",
    "position statement",
    "scientific statement",
    "consensus",
    "consensus statement",
    "expert consensus",
    "standards of care",
    "management guideline",
    "management guidelines",
    "treatment guideline",
    "treatment guidelines",
}

_SOCIETY_CUE_TERMS = {
    "association",
    "academy",
    "college",
    "society",
    "task force",
    "working group",
}


def _normalize_text(value: Any) -> str:
    text = anyascii(str(value or "")).lower()
    text = _NORMALIZE_NON_ALNUM.sub(" ", text)
    return text.strip()


def _normalize_publication_types(publication_types: Any) -> list[str]:
    if publication_types is None:
        return []

    if isinstance(publication_types, str):
        normalized = _normalize_text(publication_types)
        return [normalized] if normalized else []

    if isinstance(publication_types, dict):
        candidate = (
            publication_types.get("type")
            or publication_types.get("name")
            or publication_types.get("value")
        )
        normalized = _normalize_text(candidate)
        return [normalized] if normalized else []

    if isinstance(publication_types, (list, tuple, set)):
        normalized_items: list[str] = []
        seen = set()
        for item in publication_types:
            if isinstance(item, dict):
                candidate = item.get("type") or item.get("name") or item.get("value")
            else:
                candidate = item
            normalized = _normalize_text(candidate)
            if normalized and normalized not in seen:
                seen.add(normalized)
                normalized_items.append(normalized)
        return normalized_items

    normalized = _normalize_text(publication_types)
    return [normalized] if normalized else []


def _is_acronym_alias(alias: str) -> bool:
    alias = alias.strip()
    return bool(alias) and bool(re.fullmatch(r"[A-Z0-9][A-Z0-9\-/.&]*", alias))


def _contains_phrase(tokens: list[str], phrase_tokens: tuple[str, ...]) -> bool:
    if not tokens or not phrase_tokens:
        return False
    phrase_len = len(phrase_tokens)
    if phrase_len > len(tokens):
        return False
    for idx in range(len(tokens) - phrase_len + 1):
        if tuple(tokens[idx:idx + phrase_len]) == phrase_tokens:
            return True
    return False


def _match_phrases(text: str, phrase_index: list[tuple[str, ...]]) -> bool:
    if not text:
        return False
    tokens = text.split()
    return any(_contains_phrase(tokens, phrase_tokens) for phrase_tokens in phrase_index)


_GUIDELINE_CONTEXT_INDEX = [
    tuple(_normalize_text(term).split())
    for term in GUIDELINE_CONTEXT_TERMS
    if _normalize_text(term)
]
_SOCIETY_CUE_INDEX = [
    tuple(_normalize_text(term).split())
    for term in _SOCIETY_CUE_TERMS
    if _normalize_text(term)
]

_GUIDELINE_SOCIETY_INDEX = {}
for society_code, config in _GUIDELINE_SOCIETY_REGISTRY.items():
    alias_entries = []
    seen_aliases = set()
    for alias in [society_code, *config.get("aliases", [])]:
        normalized = _normalize_text(alias)
        if not normalized or normalized in seen_aliases:
            continue
        seen_aliases.add(normalized)
        alias_entries.append({
            "tokens": tuple(normalized.split()),
            "is_acronym": _is_acronym_alias(alias),
        })
    _GUIDELINE_SOCIETY_INDEX[society_code] = {
        "ambiguous_acronym": bool(config.get("ambiguous_acronym", False)),
        "aliases": alias_entries,
    }


def _match_societies_in_field(text: str) -> set[str]:
    tokens = text.split()
    matched = set()
    if not tokens:
        return matched

    has_society_cue = _match_phrases(text, _SOCIETY_CUE_INDEX)

    for society_code, config in _GUIDELINE_SOCIETY_INDEX.items():
        matched_acronym = False
        matched_full_name = False
        for alias in config["aliases"]:
            if _contains_phrase(tokens, alias["tokens"]):
                if alias["is_acronym"]:
                    matched_acronym = True
                else:
                    matched_full_name = True

        if not (matched_acronym or matched_full_name):
            continue

        if (
            config["ambiguous_acronym"]
            and matched_acronym
            and not matched_full_name
            and not has_society_cue
        ):
            continue

        matched.add(society_code)

    return matched


def detect_guideline_society_signal(
    title: Optional[str],
    journal: Optional[str] = None,
    evidence_term: Optional[str] = None,
    evidence_source: Optional[str] = None,
    publication_types: Any = None,
) -> dict[str, Any]:
    """
    Detect society-published guideline signals with strict context gating.

    Matching behavior:
    - Society aliases are matched on normalized token boundaries.
    - A match requires a society signal in strong fields (title/evidence metadata).
    - Journal-only society mentions never trigger a positive match.
    - Ambiguous acronyms (e.g., ADA, ASA) require either full-name match or
      explicit society cues in the same field.
    """
    normalized_fields = {
        "title": _normalize_text(title),
        "journal": _normalize_text(journal),
        "evidence_term": _normalize_text(evidence_term),
        "evidence_source": _normalize_text(evidence_source),
    }
    normalized_publication_types = _normalize_publication_types(publication_types)

    context_candidates = [
        normalized_fields["title"],
        normalized_fields["evidence_term"],
        normalized_fields["evidence_source"],
        *normalized_publication_types,
    ]
    has_guideline_context = any(
        _match_phrases(text, _GUIDELINE_CONTEXT_INDEX)
        for text in context_candidates
        if text
    )

    matched_societies = set()
    matched_fields = set()
    strong_fields = ("title", "evidence_term", "evidence_source")
    for field in strong_fields:
        field_text = normalized_fields[field]
        if not field_text:
            continue
        field_matches = _match_societies_in_field(field_text)
        if field_matches:
            matched_societies.update(field_matches)
            matched_fields.add(field)

    # Journal is intentionally weak and only used as supporting evidence.
    journal_text = normalized_fields["journal"]
    if journal_text and matched_societies:
        journal_matches = _match_societies_in_field(journal_text)
        if journal_matches.intersection(matched_societies):
            matched_fields.add("journal")

    return {
        "is_match": has_guideline_context and bool(matched_societies),
        "matched_societies": sorted(matched_societies),
        "matched_fields": sorted(matched_fields),
        "has_guideline_context": has_guideline_context,
    }


def get_journal_tier(
    journal_name: Optional[str] = None,
    nlm_id: Optional[str] = None,
) -> Literal["specialty", "general", "none"]:
    """
    Classify a journal into specialty/general/none using NLM ID first, then name.
    """
    normalized_nlm_id = str(nlm_id or "").strip()
    if normalized_nlm_id:
        if normalized_nlm_id in SPECIALTY_JOURNALS_NLM:
            return "specialty"
        if normalized_nlm_id in GENERAL_HIGH_IMPACT_NLM:
            return "general"

    normalized_name = _normalize_journal_name(journal_name)
    if normalized_name:
        if normalized_name in SPECIALTY_JOURNALS_NAMES:
            return "specialty"
        if normalized_name in GENERAL_HIGH_IMPACT_NAMES:
            return "general"

    return "none"
