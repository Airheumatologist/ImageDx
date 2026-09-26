# Visual Findings Library pilot baseline — 2026-09-25

> Historical pre-storage snapshot. Later on 2026-09-25, the 186
> vision-accepted figures in this snapshot were stored: the live database now
> has 219 stored figures, 599 panels, and 575 distinct panel image hashes.
> The counts below preserve the pre-storage baseline for comparison.

Snapshot of the live pilot database on 2026-09-25. The last recorded LLM call is
`2026-09-25 21:02:36` (SQLite timestamp; UTC). The database contains 2,407
article rows, 909 figure rows, and 71 stored panels. Disease membership is
multi-label: per-disease counts overlap, so they do not sum to the global total.

## Reproduce

From the repository root, refresh the existing no-LLM report:

```sh
python3 -m src.visual_pilot.cli report
```

This writes the current `data/visual_pilot/reports/pilot_report.md` and `.json`
plus review sheets. The data directory is git-ignored. For read-only checks,
use `sqlite3 -readonly data/visual_pilot/visual_pilot.sqlite` (or open Python's
SQLite URI `file:data/visual_pilot/visual_pilot.sqlite?mode=ro`). The figures
and panel counts below were cross-checked against SQL on that read-only
connection. The stored-image rate is
`50 * distinct panel SHA-256 values with an existing image_path / parsed
articles`; all 66 distinct hashes had an existing image file. Vision-call
rates use `llm_calls.stage='p3'`, joined by the response `figure_id`; per-disease
membership is assigned through the parent article's
`primary_disease_keys_json`. The audit sample was chosen from `status =
'vision_accepted'` figures without a matching panel row.

## Funnel and usable image yield

The report's `relevant` article count is cumulative and includes articles
already at `parsed`. Its `figures` count is per disease and includes figures
from the articles assigned to that disease.

| Disease | Relevant articles (incl. parsed) | Parsed articles | Figures | Caption kept | Uncertain | Caption rejected | Vision accepted (incl. stored) | Stored panels | File-present unique image hashes (proxy) | Images per 50 parsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SLE | 472 | 187 | 352 | 227 | 15 | 110 | 49 | 29 | 29 | 7.75 |
| DM | 162 | 155 | 381 | 275 | 5 | 101 | 116 | 7 | 7 | 2.26 |
| AS | 236 | 166 | 364 | 233 | 5 | 126 | 80 | 35 | 30 | 9.04 |
| All (unique articles/figures) | 768 | 417 | 909 | 599 | 23 | 287 | 219 | 71 | 66 | 7.91* |

`*` Global yield is 66 / 417 * 50. Global panel/image totals deduplicate across
diseases; per-disease totals can overlap. These 66 image hashes are called
“usable” only as a file-existence proxy: their stored path exists and SHA-256 is
present. This does not validate image decoding, clinical content, or usefulness.
Figure triage and vision outcomes are cumulative from their JSON payloads, not
just current statuses.

At the article table level, statuses are 1,211 irrelevant, 428 license-rejected,
351 relevant but not parsed, and 417 parsed. There are 362 relevant-but-unparsed
disease memberships (285 SLE, 7 DM, 70 AS; memberships overlap). Current figure
queues are 11 caption-kept and 23 caption-uncertain figures awaiting vision,
186 vision-accepted figures awaiting storage, and 2 vision errors. The 33
already-stored figures yielded 71 panels and 66 distinct stored images.

## Vision throughput, coverage, and cost

| Disease | Completed P3 ledger calls | Accepted figures | Accepted per 100 calls |
|---|---:|---:|---:|
| SLE | 226 | 49 | 21.68 |
| DM | 273 | 116 | 42.49 |
| AS | 235 | 80 | 34.04 |
| All (unique calls) | 602 | 219 | 36.38 |

Per-disease call totals overlap because some articles are assigned to multiple
diseases; the global row uses unique calls. Disease acceptance numerator is the
number of P3 outputs currently accepted or stored for that disease.

Stored panel coverage (panel occurrences, not unique images):

| Disease | Modalities | Finding keys |
|---|---|---|
| SLE (29 panels) | clinical_photo 15; histology_he 9; MRI 4; immunofluorescence 1 | cutaneous_vasculitis 3; dermal_mucin 3; discoid_plaque 4; interface_dermatitis 7; jaccoud_arthropathy 1; lupus_nephritis_class 1; malar_rash 2; npsle_white_matter_lesions 3; oral_ulcer 1; raynaud_phenomenon 1; scarring_alopecia 1; scle_annular 2; scle_papulosquamous 1 |
| DM (7 panels) | clinical_photo 7 | gottron_sign 2; heliotrope_rash 3; mda5_cutaneous_ulcers 1; mda5_palmar_papules 1 |
| AS (35 panels, 30 images) | MRI 32; CT 1; radiograph 1; ultrasound 1 | achilles_enthesitis 1; corner_fat_lesion 1; corner_inflammatory_lesion 1; fat_metaplasia 2; sacroiliitis 3; si_bone_marrow_edema 26; si_erosions 6; si_sclerosis 6 |

The ledger records $0.956927 across P1 ($0.182057), P2 ($0.064721), P3
($0.696632), P4 ($0.013355), and URL probe ($0.000162). This is recorded cost,
not a complete billing total: cache hits do not add ledger rows, and failed
requests may not be represented. There are 9 P4 ledger calls and 126 text
finding rows across 8 articles. Extraction completion/backlog cannot be
reconstructed reliably because articles have no extraction status and a
completed extraction with no accepted assertions leaves no text finding row.

## Metadata audit of accepted, unstored figures

Purposive sample of two figures per disease (6 of 186; stratified across the
three disease labels). No external requests were made. None of these candidate
figures had an exact matching local source image; consequently this is a review
of stored captions and P3 output, not visual confirmation.

| Disease / figure | Metadata review |
|---|---|
| SLE `PMC11318049:fig3` | Caption describes patient kidney ultrastructure and histology; four accepted pathology panels are mapped to `lupus_nephritis_class`. Caption evidence supports disease/finding mapping, but panels cannot be visually checked here. |
| SLE `PMC11826623:f1` | Caption describes lupus retinopathy; two accepted ophthalmic panels have no mapped finding key. Likely a vocabulary gap or missing finding annotation. |
| DM `PMC10017873:F1` | Caption explicitly describes a child's finger calcinosis and permission; one accepted clinical-photo panel maps to `calcinosis_cutis`. Strong caption-level support. |
| DM `PMC9051059:F2` | Caption contrasts a healthy control with DM PET uptake; panel A is excluded as control and panel B accepted. PET uptake is mapped to `muscle_edema_stir`, which appears semantically mismatched to the modality and warrants correction/review. |
| AS `PMC12283877:Fig2` | Caption contrasts normal, non-inflammatory, and inflammatory hip appearances; only panel C is accepted as AS and mapped to `enthesophyte`. Disease attribution from this caption/context is less direct and merits human review. |
| AS `PMC12350260:f2` | Caption describes one axSpA patient's MRI, radiograph, and CT findings; four accepted panels map to those modalities and captioned lesions. Strong caption-level support; visual segmentation remains unverified. |

## Data-quality limits

- Article and figure counts overlap by disease membership; do not add disease rows
  as if they represented unique records.
- The current `vision_accepted` status (186) is the unprocessed storage queue;
  the cumulative accepted outcome is 219 after including 33 stored figures.
- There are 186 accepted-but-unstored figures and no local candidate image files
  to audit visually. The existing thumbnails belong to already stored panels.
- The 66-image “usable” count is a file-existence proxy: the recorded
  `image_path` exists and has a SHA-256. It is not clinician validation and
  does not confirm image decoding or finding quality. The local visual sample
  below is a spot check, not validation of all images.
- Coverage reflects only 71 currently stored panels. DM has only 7 panels; its
  stored images are all clinical photos, so the present sample is sparse.
- Cache behavior limits cost completeness; the report is not an invoice.

## Visual spot check of already-stored panels

Inspected six local panel files with `view_image`, two per disease, sampling
different modalities where the stored collection permits it. These are visual
observations about the rendered files, not clinical diagnoses.

| Disease / panel | Stored label | Visual match and image usability |
|---|---|---|
| SLE `PMC8154312_fig2_A` | clinical photo; malar rash | The close facial photo visibly has confluent erythema across the cheeks and bridge of the nose, consistent with the stored label. Face is centered and adequately exposed; eyes are covered by a black bar. Useful as a recognizable example. |
| SLE `PMC11011942_diagnostics_14_00780_f002_A` | H&E; dermal mucin, interface dermatitis | It is visibly a histology slide, but this 384×228 crop includes a sliver of the neighboring panel at the right edge. The broad tissue pattern is visible; specific subtle findings cannot be confirmed from this small view. Modality matches; crop and resolution limit finding review. |
| DM `PMC4637993_Fig1_a` | clinical photo; heliotrope rash | The eyelids show marked violaceous erythema/swelling, consistent with the label. The 202×226 crop is small and includes a narrow part of the neighboring panel at right, but remains recognizable. |
| DM `PMC4637993_Fig2_d` | clinical photo; MDA5 cutaneous ulcers | The palms and arrow annotations are visible, but discrete ulcers are not obvious at this resolution. The hand image is usable for broad context; the specific stored finding is visually uncertain and should be reviewed. |
| AS `PMC10093281_diagnostics_13_01342_f004_A` | MRI; SI-joint bone marrow edema | Coronal MRI shows both SI joints, with conspicuous high-signal areas around the joints consistent with the stored edema label. The 378×271 crop includes the relevant anatomy and is usable, though fine detail is limited. |
| AS `PMC5913283_F1_A` | pelvic radiograph; sacroiliitis, erosions, sclerosis | The image is a well-framed AP pelvis radiograph with both SI joints visible. Specific erosions/sclerosis are subtle at this view size and are not independently clear; modality matches, while the finding label needs expert/contextual confirmation. |
