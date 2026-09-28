# Visual library curation review

The SLE browser feedback exposed errors in both selection and presentation.
The stored P3 judgments already received the figure caption, article title,
in-text mentions, and approved vocabulary. Several models nonetheless inferred
a disease from the review topic and described charts as patient images. The
old post-validation checked keys and coordinate shape, but did not enforce
image eligibility. Confidence was used to sort results and did not establish
that an image was clinically appropriate.

| Reported example | Evidence in the stored judgment | Failure |
| --- | --- | --- |
| `PMC10070984:fig1` | Compound MRI figure; panel A described as unremarkable and included as a negative control | Normal images were accepted and a collage was split into crops |
| `PMC6699445:f2` | Biopsy images mixed with a chart; returned boxes span headings, chart regions and adjacent tiles | Coordinate validation did not verify that crops isolated usable images |
| `PMC6699445:f3` | Caption describes a histopathology classification; six crops labeled as patient biopsies | The model interpreted text labels as visual pathology |
| `PMC12676321:fig4` | Caption explicitly says schematic and BioRender; rationale claims a real patient ulcer | Contradictory eligibility was not rejected after model judgment |
| `PMC5907183` | Title and captions explicitly describe canine cutaneous lupus | Human-patient scope was not enforced |
| Vascular skin images | Approved findings say cutaneous vasculitis or Raynaud, while subtype often says ACLE/SCLE/DLE | The viewer grouped using the model's subtype instead of the depicted finding |

The shared policy in `src/visual_pilot/curation.py` now governs judgment,
materialization, and existing-gallery eligibility. It favors complete clinical
images and excludes compound sources rather than trying to rescue small tiles.
It checks diagram/chart/species evidence, usable bounds and dimensions, normal
or control images, specific manifestations, and disease attribution. This is a
conservative publication policy: a hidden image can still be valid in another
context, and automated checks do not constitute an exhaustive clinical review.

`python3 -m src.visual_pilot.curation_audit` produces a local audit without
changing source rows. `--apply` backs up SQLite and records hash-bound exclusions
in `panel_curation`. Original image files, panel rows and model judgments remain
available. The `published_panels` view keeps exclusions out of coverage counts,
yield calculations, reports and derived image-finding rows. A changed image hash
does not inherit an old exclusion. Rerunning the audit updates the policy and
can restore publication when an exclusion no longer applies.

The SLE Skin view groups vascular manifestations separately and uses supported
finding evidence for cutaneous subtype sections. A Pediatric view gathers known
child/adolescent images across clinical sections while preserving their usual
section membership. Unknown ages remain unknown; a pediatric article title
does not establish a pictured patient's age.

Cards show the image and a readable depiction label. The full image and short
source context, age metadata, article title/link, attribution and license are
available on opening a card. Thumbnails preserve the frame instead of cropping
it with `object-fit: cover`. Duplicate images retain separate source credits.

This patch does not purchase model calls or rerun the article retrieval pipeline.
Audit results and the pre-change database backup are saved under the local data
directory's `reports/` folder.
