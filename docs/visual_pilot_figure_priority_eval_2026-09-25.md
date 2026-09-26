# Figure priority metadata audit — 2026-09-25

The new deterministic figure ordering was compared with the old order
(`caption_kept` first, then `figure_id`) at the same budget of three figures
per disease. Both methods saw the same 19 manually labeled, locally available
figures: seven SLE, four DM, and eight AS. Candidates were drawn from the
current `caption_kept` and `caption_uncertain` queue. The audit script opened
the database read-only and made no image downloads or model calls.

“Useful” means the stored caption explicitly supports an eligible patient
photo, scan, ultrasound, or pathology figure with an indexed-disease link.
Negative examples include a flowchart, a scan captioned as a mimic, or an
unrelated finding in a patient who also has the disease. These are metadata
labels, not visual review or clinical verification.

| Disease | Labeled queue | Caption-positive | Old P@3 / R@3 | New P@3 / R@3 |
|---|---:|---:|---:|---:|
| SLE | 7 | 5 | 0.33 / 0.20 | 1.00 / 0.60 |
| DM | 4 | 3 | 0.67 / 0.67 | 1.00 / 1.00 |
| AS | 8 | 2 | 0.67 / 1.00 | 0.67 / 1.00 |
| Total | 19 | 10 | 0.56 / 0.50 | 0.89 / 0.80 |

At this small equal call budget the heuristic selected eight caption-positive
figures versus five for the old order. AS did not improve on this sample. The
per-figure IDs, metrics, and exact executable output are in
`visual_pilot_figure_priority_eval_results.json`; labels and rationale are in
`../scripts/vp_figure_eval_labels.json`.

The separate rejected-figure spot check found one caption-positive entry in
each of the SLE and DM samples. `PMC10173173:fig0005` says its caption is a
patient photo of acute cutaneous lupus, but P2 stored `route=drop` with
`reason=keep`; `PMC10222774:vaccines-11-00898-f002` describes patient
dermatomyositis calcinosis, while P2 assigned only SLE even though the parent
article is indexed for both DM and SLE. The narrow triage conflict guard
routes this kind of contradictory result to `caption_uncertain`; it keeps
third-party material rejected. These historical rows remain rejected in this
database snapshot and are re-queued on the next `triage` run for vision review.

An article-rank audit compares old retrieval-score ordering with the article
caption-yield reranker at two articles per disease. The sample includes 18
articles (8 caption-positive), including both currently parsed and irrelevant
articles. The equal-budget totals are P@2 0.33→1.00 and R@2 0.25→0.75; the
new order improves all three diseases in this sample. Earlier scoring ranked
`PMC8564476` too highly because its figure caption describes a phenotype
summary. The image-cue rule was tightened, and that article is no longer in
the top two. Exact rankings and workflow statuses are in
`visual_pilot_article_priority_eval_results.json`.

The database snapshot lacks persisted abstracts and matched retrieval passages.
This audit therefore uses the same old retrieval score and local JATS captions
with article titles; it does not validate the full title/abstract/passage
reranker used with live retrieval evidence. Article labels are metadata audits
of the local figure captions and statuses, not visual review, clinical
validation, or proof of attribution. The labels include `PMC13113392` and
`PMC9450188` as rejected disease-mapping controls and cases with only diagram
captions to check cross-disease and mechanism-only ranking behavior.

## Limits

- The sample is purposive, small, and only covers figures already admitted to
  the vision queue. It does not estimate recall among caption-rejected figures
  or validate article retrieval.
- “Correctly attributed” here means supported by caption text and article
  disease metadata. It does not establish that image pixels show the stated
  finding or that a clinician would accept the label.
- Multiple figures come from the same articles, so results are not independent.
- AS currently has fewer caption-positive examples and a tie at this budget;
  the result supports a pilot ordering change, not a broad performance claim.

Reproduce from the repository root with:

```sh
python3 scripts/vp_eval_figure_priority.py
python3 scripts/vp_eval_article_priority.py
```
