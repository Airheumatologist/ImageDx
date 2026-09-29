# Visual Findings Library — image coverage expansion plan

**Goal:** every approved disease/finding pair has ≥10 representative images,
and each disease's total image set grows 5–10× over the 2026-09-29 baseline.

Measured against the fresh `run-all` completed 2026-09-29
(`logs/run_all_fresh_20260928_2340.log`).

## 1. Targets vs. current state

169 approved finding–disease pairs (145 findings × their disease keys).
Target ≥10 images/pair → ≥1,690 panels; disease-level 5–10× → ~1,125–2,250
panels total. Both point at roughly **1,700–2,300 stored panels**.

Current published-image distribution per pair (from `published_panels`):

| images/pair | pairs |
|---|---|
| 0 | 89 |
| 1 | 35 |
| 2 | 18 |
| 3 | 15 |
| 4 | 6 |
| 5–6 | 2 |
| ≥10 | 4 |

80 of 169 pairs have any image; 89 pairs have zero. 150 of 225 published
panels carry an approved finding for their disease (195 pair credits).

Funnel of the 2026-09-29 run vs the previous regeneration:

| stage | previous run | fresh run |
|---|---|---|
| license-passing articles | ~6,031 | ~7,554 |
| P1 relevant | ~1,739 | ~1,742 |
| parsed | 1,646 | 1,712 |
| figures triaged | 4,711 | 4,816 |
| caption kept + uncertain | ~1,053 | 1,000 |
| vision accepted → stored | 265 | 225 |
| final library | 307 saved / 204 published* | 225 saved / 225 published |

*previous 307 included ~40 merged legacy panels that did not re-qualify.

## 2. Root causes, ranked by recoverable yield

### C1 — Single-disease collages are rejected whole (keep the plate, do not crop it)

A collage is worth keeping when every patient panel is the same disease.
Cropping it into per-panel tiles is the presentation that produced small
snippets of individual manifestations. Store the full figure once.

The judge already localizes panels, then marks every panel of a multi-panel
figure `include:false, exclusion_reason:"collage"`. That flag is also
enforced after the model returns: `curation.exclusion_reason` rejects any
figure with `figure_is_compound` or more than one panel, and the same
function gates `judge.post_validate`, `store`, the viewer, and
`curation_audit`. A prompt change alone stores nothing. `judge` does not
reread `vision_rejected` rows, and `store` only reads `vision_accepted`.

Measured on the 468 compound figures inside the 775 `vision_rejected` rows
(1,574 panels stamped `collage`):

| whole-figure class | figures | where it belongs |
|---|---|---|
| same manifestation, clinical/histology | 50 | that finding's existing section |
| same manifestation, radiology | 56 | that finding's group on the Imaging tab |
| several manifestations, one disease | 25 | Combined views |
| several manifestations, radiology | 41 | Combined views on Imaging |
| one disease, no approved finding yet | 93 | hold for a label; 70 of these are radiology |
| mixed with a chart, diagram, or `other` panel | 28 | leave rejected |
| fewer than two patient panels | 165 | leave rejected |
| patient panels from more than one disease | 10 | leave rejected |

Same-manifestation plates are 106 figures touching 45 pairs. Counted as one
image each, zeros move 89 → 74 and pairs at ≥10 move 4 → 6. Multi-manifestation
plates (66) are real library images and do not each fill every finding they
depict. This recovers useful figures. The per-pair target of 10 still depends
on Phases 2 and 3. A plate that mixes a photo with a chart stays unpublished.

### C2 — Caption triage drops montages before the judge sees them

1,033 of 3,813 `caption_rejected` figures are dropped for a multi-panel or
montage reason. About 717 of those captions also name a clinical or imaging
modality. The judge never classifies them. Same keep-whole rule as C1, one
stage earlier. Uncertain images already flow to the judge (kept+uncertain =
1,000 judged); montages do not.

### C3 — The relevant-article pool is exhausted; yield stops before coverage

- `--limit 800` is now a license-passing target (commit `a17ff04`), which
  raised the licensed pool ~25% but P1 still passed only ~1,742 articles
  (~33% of evaluated) — and parse consumed ~98% of them. The licensed pool
  grew; the bottleneck moved to P1 and to the corpus itself.
- 23,400 persisted `candidate` rows were never license-checked (selection
  stopped at the 800/disease target). Volume headroom exists: per-disease
  type-passed pools are ~4–9k.
- The run loop stops a disease after 2 consecutive zero-yield batches
  (`--zero-yield-batches`). It measures *any* new image or *any* newly
  covered finding — not per-finding image counts — so it halts while dozens
  of pairs are still at 0–3 images.
- Selection round-robins candidates across finding lanes
  (`VP_MANIFESTATION_QUOTA=20`) to cover findings, which trades raw image
  yield for coverage: `as` parsed 205 articles → 4 panels; `ad` 288 → 9.

### C4 — Case reports stay out of image collection

Review articles are the only corpus. Four gates already enforce that, and
all four stay:

- Retrieval is the review-only `Or` (`publication_type` contains "Review",
  or `article_type` is `review-article`).
- `passes_type_filter` keeps an article only when a label contains "review"
  or `article_type` is `review-article`, and `_EXCLUDED_LABEL_PARTS` includes
  the substring `case`.
- The P1 prompt tells the model to set `is_narrative_review` false for a
  case report.
- `_p1_relevant` requires `is_narrative_review`.

A case report, including an images-in-medicine article tagged as one, dies
at retrieval or at P1. There is no case-report lane. A pair that cannot
reach 10 distinct images from licensed reviews gets a written exclusion.

### C5 — License boundary (policy, not a bug)

1,306 articles were license-rejected: `license:none` 536, `cc-by-nc*` 698,
`other` 71. The spec requires commercial-use licenses; relaxing this is a
downstream-use decision, flagged here only because it is a real volume cap.

### C6 — Smaller frictions

- Accepted collages are stored as one `whole_figure`. ND licenses already
  require that mode, so they are compatible with this policy. Overlap-based
  promotion to `whole_figure` must not fire on a figure we rejected for a
  mixed chart panel.
- 3 figures remain `pending` (fetch/judge residue).

## 3. Work plan

### Phase 1 — keep single-disease collages whole (est. +106 same-finding images, +66 combined plates)

**W1: publish the whole plate for an already-judged single-disease collage.**
Re-route a `vision_rejected` compound figure only when it has at least two
patient-image panels, all of them share one `disease_key`, and none of the
panels is a diagram, chart, normal control, other disease, or `modality=other`.
Write one panel row with `crop_mode=whole_figure` and the full image. Keep
the per-panel judge labels so the viewer can place the card; do not crop
those boxes into stored images.

Placement:

- One approved finding across the panels (several angles, slices, or
  repeats): file the card in that finding's existing group. Radiology
  modalities (`radiograph`, `ct`, `mri`, `ultrasound`, `echo`, `pet`)
  already land on the Imaging tab, which is the right home for a CT/MRI
  plate of one pattern.
- Two or more approved findings of that same disease: file the card once
  in a **Combined views** group on the tab those modalities share (Imaging,
  when the plate is radiology). Leave it out of each finding's own group,
  so one plate does not stand in for several manifestations.
- One disease and no approved finding yet (99 figures, 73 radiology): leave
  them unpublished until a label pass assigns a finding, or show the
  radiology subset on Imaging only under an explicit unlabeled group.

Implementation, in order:

1. `curation.exclusion_reason` allows this whole-figure case and still
   rejects mixed-disease plates, mixed chart/photo plates, and per-panel
   snippet crops. Bump `POLICY_VERSION`. Update the P2/P3 prompts (and
   their versions, which are part of the LLM cache key) so a multi-panel
   patient figure is kept whole and classified, and a tile is not returned
   as its own image. Update `docs/visual_curation_review.md` and the
   clinical-policy note in `src/visual_pilot/README.md`.
2. Requeue the qualifying `vision_rejected` figures onto a status `judge`
   and `store` actually read. Deterministic re-route from the current JSON
   is enough for the 106 + 66 already labeled; a judge re-pass is only
   needed for the 93 with no approved finding.
3. Viewer: `groupName` gains Combined views. Same-finding plates use the
   finding group they already would.

`record_published_outcomes` credits the same-finding plate to that one
pair. A combined plate does not increment every finding it shows.

**W2: montage re-triage.** Re-run `triage` on the ~717 caption-rejected
montages whose reason also names a clinical or imaging modality. Allow
`kept`/`uncertain` for a single-disease patient montage, then judge and
store under the W1 whole-figure rule.

**W3: QA the plates.** Spot-check whole images in the viewer: same-finding
plates sit in the finding group, multi-finding plates sit only in Combined
views, and a photo-plus-chart plate stays unpublished. Confirm Imaging
holds radiology plates as one card.

### Phase 2 — coverage-directed volume (est. fills most pairs to ≥5–10)

**W4: per-finding image quota in the run loop.** Continue batches while any
in-scope pair has <10 distinct published images, bounded by
`--max-articles` / `--budget-usd`. `covered` is binary today in
`coverage_gaps`, `sync_candidates`, and `reserve_batch`, not only in the
zero-yield counter: one image closes the lane. Extend that trio with a
count. On this database the change has almost no backlog — 30 `relevant`
articles remain, all systemic-sclerosis therapy reviews, and 58 lanes are
`exhausted` (zero images and no remaining relevant candidate). Only
sarcoidosis, SLE, and RA actually hit the two-batch zero-yield stop.

**W5: raise the licensed ceiling.** `--limit` 800→2,000–3,000 license-passing
articles per disease, so selection continues into the 23,400 persisted
candidates. CLI defaults are already `--max-articles 6000`, `--batch-size
200`, and `--max-runtime-seconds 14400`; those are not the ceiling that
stopped the fresh run. Expect P1 to keep ~33% pass.

**W6: caption-rescue relevance lane.** `relevance_reason=
'visual_figure_caption_rescue'` already exists (147 articles parsed this
way). The peek today runs inside `select_batch` on a shortlist of articles
that already passed the review gates and the license check. It does not see
the 23,400 unlicensed candidates. Extending it is a pre-P1 caption pass over
that review pool: a license-passing review with enough patient-image
captions can skip full P1. The pass stays behind the four C4 gates, so a
case report never enters it.

### Phase 3 — more reviews for stubborn pairs

**W7: targeted retrieval for zero-coverage pairs.** For pairs still under
10 after W4–W6, emit finding-specific queries with synonyms expanded from
`findings_vocab`, still under the review filter. Europe PMC full-text
search, if used, applies the same review-only and case-report exclusions.
`as`, `gout`, and `ad` have the thinnest coverage relative to parsed volume.

**W8 (policy decision):** confirm the commercial-license boundary stands.
If downstream use permits, `cc-by-nc` adds ~700 candidate articles —
including several explicitly rejected image-rich reviews. Case reports
remain excluded either way.

### Verification

- `python3 -m src.visual_pilot.cli report` per-disease funnel + new
  per-pair coverage table (images/pair histogram as in §1).
- `store --refresh` + spot sheets for crop QA; viewer page/media check
  (`serve --port 8765`).
- Success criteria: all 169 pairs ≥10 distinct published images, or an
  audited written exclusion per pair (analogous to the existing
  psoriasis-ocular note in the README). A same-finding collage counts as
  one image. A combined plate counts in Combined views and not toward
  each finding's ten.

## 4. Sequencing and expected yield

| phase | work | est. new images | cumulative |
|---|---|---|---|
| — | baseline | 225 | 225 |
| 1 | W1 whole plates already labeled | +106 same-finding, +66 combined | ~400 |
| 1b | W2 judged montages, same rule | unlabeled until judged | — |
| 2–3 | W4–W8 more licensed reviews | the rest of the per-pair gap | target ≥1,690 distinct, or a written exclusion |

Phase 1 needs a policy change in `curation.py` and a requeue. It does not
need a judge call for the 172 figures that already carry approved findings.
The 93 unlabeled single-disease plates, and W2's montages, do. Phase 2 is
queue code plus a higher `--limit`. Phase 3 adds finding-specific review
queries. The type filter, the review-only retrieval filter, and P1 stay
as they are.

## 5. Risks / open questions

- **Snippet crops stay unpublished.** Panel boxes are classification
  metadata. W3 checks that a stored collage is the full figure and that a
  mixed chart/photo plate was left out.
- **Combined plates must not inflate a finding's count.** One image, one
  card, in Combined views.
- **`as`/`gout`/`ad` ceiling**: case reports are not a fallback. If licensed
  reviews still cannot supply 10 distinct images for a rare manifestation,
  that pair gets a documented exclusion.
- **Run duration**: Phase 2 roughly triples parsed volume; at the observed
  ~1,700 articles/4–5h pace expect a long run — all stages resume cleanly
  from `status`, so it can be split across invocations.
