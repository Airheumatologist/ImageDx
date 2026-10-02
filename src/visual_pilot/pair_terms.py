"""Disease and finding search terms for Europe PMC caption discovery.

The vocabulary file stays frozen; derived phrasings and overrides live here.
"""

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


# Words that describe the modality or disease context rather than what a
# caption calls the finding ("Ultrasound double-contour sign").
_CAPTION_PREFIXES = (
    "ultrasound", "us", "radiographic", "radiograph", "mri", "ct", "hrct",
    "histologic", "histological", "capillaroscopic", "nailfold", "pulmonary",
    "cutaneous", "clinical",
)
# Too generic to search alone once the context words are gone.
_GENERIC_TERMS = frozenset({
    "rash", "papules", "papule", "nodules", "nodule", "ulcers", "ulcer",
    "erosions", "erosion", "edema", "plaques", "plaque", "lesions", "lesion",
    "sign", "erythema", "fibrosis", "inflammation", "uveitis", "arthritis",
    "hair", "band", "scalp", "finger", "fingers", "scar", "galaxy", "shawl",
    "holster", "dagger", "nephritis", "vasculitis", "ulceration", "hemorrhage",
    "hemorrhages", "histopathology", "biopsy", "eyelid", "periorbital",
    "inverse", "joint space", "stage ii", "tophus", "osteolysis",
})
_ADJECTIVE_ENDINGS = ("al", "ar", "ic", "ate", "ous", "ive", "ed", "ent", "ant")

# Hand-written caption phrasings where label/synonym derivation is too
# generic or loses the finding; the vocabulary file itself stays frozen.
CAPTION_TERM_OVERRIDES: dict[str, list[str]] = {
    "non_scarring_alopecia": ["non-scarring alopecia", "nonscarring alopecia", "diffuse alopecia", "lupus hair"],
    "cutaneous_vasculitis": ["cutaneous vasculitis", "leukocytoclastic vasculitis", "palpable purpura"],
    "lupus_band": ["lupus band", "lupus band test", "dermoepidermal junction immunofluorescence"],
    "lupus_nephritis_class": ["lupus nephritis", "glomerulonephritis", "renal biopsy"],
    "scalp_dermatomyositis": ["scalp dermatomyositis", "scalp involvement", "scalp erythema", "psoriasiform scalp dermatitis"],
    "capillary_hemorrhages": ["capillary hemorrhage", "capillary haemorrhage", "microhemorrhage", "microhaemorrhage"],
    "shawl_sign": ["shawl sign", "shawl distribution"],
    "holster_sign": ["holster sign", "holster distribution"],
    "dagger_sign": ["dagger sign", "single dagger sign", "supraspinous ossification"],
    "sarcoid_galaxy_sign": ["galaxy sign", "sarcoid galaxy", "coalescent nodules with satellite nodules"],
    "ssc_sclerodactyly": ["sclerodactyly", "skin tightening of the fingers", "sclerotic fingers"],
    "psoriasis_plaque": ["plaque psoriasis", "psoriatic plaque", "psoriasis vulgaris", "silvery scale"],
    "psoriasis_munro_microabscesses": ["munro microabscess", "neutrophils in the stratum corneum", "psoriasiform hyperplasia"],
    "sarcoid_scar_infiltration": ["scar sarcoidosis", "sarcoidosis in scars", "sarcoidosis in a scar", "tattoo sarcoidosis", "scar infiltration"],
    "sarcoid_parenchymal_opacities": ["parenchymal opacities", "reticulonodular opacities", "stage ii sarcoidosis", "stage iii sarcoidosis"],
    "sarcoid_papules": ["papular sarcoidosis", "sarcoidal papule", "granulomatous papule", "cutaneous sarcoidosis"],
    "sarcoid_bilateral_hilar_adenopathy": ["bilateral hilar lymphadenopathy", "hilar lymphadenopathy", "hilar adenopathy", "stage i sarcoidosis"],
    "sarcoid_uveitis": ["granulomatous anterior uveitis", "granulomatous uveitis", "ocular sarcoidosis", "mutton-fat keratic precipitates"],
    "gout_tophi": ["tophi", "tophus", "tophaceous", "gouty tophus"],
    "gout_podagra": ["podagra", "first metatarsophalangeal joint", "acute gout attack"],
    "gout_joint_space_preservation": ["preserved joint space", "joint space preservation"],
    "gout_ultrasound_tophi": ["tophus on ultrasound", "tophi on ultrasound", "hyperechoic aggregates", "ultrasound tophus"],
    "gout_deposit_dect": ["dual-energy ct", "dual-energy computed tomography", "dect"],
    "gout_synovial_crystals": ["monosodium urate crystals", "msu crystals", "urate crystals", "negatively birefringent crystals"],
    "ad_flexural_eczema": ["flexural eczema", "flexural dermatitis", "antecubital fossa", "popliteal fossa"],
    "ad_hand_eczema": ["hand eczema", "hand dermatitis"],
    "ad_eyelid_eczema": ["eyelid eczema", "eyelid dermatitis", "periorbital eczema", "periorbital dermatitis"],
    "ad_head_neck_dermatitis": ["head and neck dermatitis", "head and neck eczema", "facial eczema", "facial dermatitis"],
    "ad_infantile_face_scalp": ["infantile eczema", "infantile atopic dermatitis", "infant with atopic dermatitis"],
    "ad_spongiosis_histology": ["spongiotic dermatitis", "spongiosis"],
}


def _caption_norm(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).split())


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 5:
        return word[:-3] + "y"
    if word.endswith(("sses", "xes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        return word[:-1]
    return word


def _disease_words(disease: dict) -> list[str]:
    names = [disease["name"], *(disease.get("synonyms") or [])]
    words = {_caption_norm(n) for n in names if len(_caption_norm(n)) > 3}
    # "sarcoid", "scleroderma", "psoriatic", "lupus" style stems.
    words |= {w[:-4] for w in words if w.endswith("osis") and len(w) > 8}
    return sorted(words, key=len, reverse=True)


def caption_terms(disease_key: str, finding: dict, disease: dict | None = None) -> list[str]:
    """Phrases a figure caption would use for this finding.

    Vocabulary labels carry disease and modality context ("Telangiectasias in
    systemic sclerosis"); captions name the finding alone. An explicit
    ``caption_terms`` list on the vocabulary row overrides the derivation.
    """
    override = finding.get("caption_terms") or CAPTION_TERM_OVERRIDES.get(finding.get("finding_key"))
    if override:
        return list(dict.fromkeys(_caption_norm(t) for t in override if t))
    disease = disease or diseases.load_diseases()[disease_key]
    disease_words = _disease_words(disease)
    raw = [re.sub(r"\(.*?\)", " ", str(finding.get("label") or "")),
           *finding_synonyms(disease_key, finding)]
    out: list[str] = []
    for value in raw:
        original = text = _caption_norm(value)
        for word in disease_words:
            text = re.sub(rf"\b(in|of|associated with)\s+{re.escape(word)}\b", " ", text)
            text = re.sub(rf"\b{re.escape(word)}\b", " ", text)
        words = text.split()
        while words and words[0] in _CAPTION_PREFIXES:
            words = words[1:]
        while words and words[-1] in {"in", "of", "with"}:
            words = words[:-1]
        if not words:
            continue
        if len(words) == 1 and text.split() != original.split() and (
            words[0] in _GENERIC_TERMS or words[0].endswith(_ADJECTIVE_ENDINGS)
        ):
            # "inverse psoriasis" must not become "inverse".
            words = original.split()
        variants = {" ".join(words), " ".join([*words[:-1], _singular(words[-1])])}
        if len(words) > 1 and words[-1] == "sign":
            variants.add(" ".join(words[:-1]))
        for term in sorted(variants):
            if term in _GENERIC_TERMS or len(term) < 4:
                continue
            out.append(term)
    return list(dict.fromkeys(out))


_STOPWORDS = frozenset({"of", "the", "in", "on", "with", "and", "a", "an", "at", "to", "for", "by"})


def content_words(term: str) -> list[str]:
    """Singular content words of a term, for same-caption word matching."""
    return [_singular(w) for w in _caption_norm(term).split() if w not in _STOPWORDS]


def caption_matches(caption: str, terms: list[str], mode: str = "phrase") -> bool:
    """Plural-tolerant caption match, mirroring the Europe PMC query mode.

    ``phrase``: a whole term appears as consecutive words. ``words``: every
    content word of a multi-word term appears somewhere in the caption
    (prefix match for words of four or more letters, like ``word*``).
    """
    tokens = [_singular(w) for w in _caption_norm(caption).split()]
    text = " " + " ".join(tokens) + " "
    for term in terms:
        if " " + " ".join(_singular(w) for w in term.split()) + " " in text:
            return True
        if mode != "words":
            continue
        words = content_words(term)
        if len(words) >= 2 and all(
            any(t.startswith(w) if len(w) >= 4 else t == w for t in tokens) for w in words
        ):
            return True
    return False
