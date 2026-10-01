# Balanced Manifestation Image Coverage - Multi-Agent Plan

Date: 2026-09-30
Status: implementation plan only; no production run or policy relaxation authorized.

## 1. Goal And Decisions

Optimize for eligible images that actually appear in each approved
`(disease_key, finding_key)` gallery, not retrieved or relevant article counts.
An eye manifestation with 22 relevant candidates and zero images is still a gap.

| Setting | Default | Meaning |
|---|---|---|
| `VP_FINDING_IMAGE_FLOOR` | 3 | Initial coverage milestone |
| `VP_FINDING_IMAGE_TARGET` | 10 | Existing expansion milestone; retain its meaning |
| `VP_FINDING_GALLERY_CAP` | 20 | Maximum published gallery images per disease/finding pair |

The cap is **20, not 5**. Ten is not a stopping ceiling. Galleries may grow
through 10 to 20 when lower-coverage work has been served or is explicitly
blocked. All eligible surplus images remain stored as reserves. No truncation,
file deletion, or eligibility rejection merely because a gallery is full.
Validate `1 <= floor <= target <= cap`; reject inconsistent overrides clearly.

The cap is per disease/finding pair, across modalities, tabs, and age groups,
not 20 per tab. Distinct eligible findings may share an image when the existing
policy permits; a combined whole-figure plate remains in Combined views and
does not earn credit for each depicted finding. This plan does not change that
policy or introduce a new combined-views ceiling.

## 2. Grounded Starting Point

Code and the local database were inspected read-only on 2026-09-30:

- Distinct hashes in `published_panels` already drive coverage; article batches
  already reserve slots in count order. Do not rebuild those mechanisms blindly.
- `published_panels` excludes current-hash audit exclusions, but the viewer also
  applies current eligibility rules. Counting the SQL view alone is insufficient.
- `reserve_batch` fills remaining capacity globally. `run-all` retrieves once
  and drops a disease from rotation when no selectable article remains.
- Shared anterior-uveitis synonyms include psoriasis-specific terms; AS queries
  contain those terms. `sync_candidates` imports finding-keyed evidence without
  verifying the originating disease. Both need pair-specific provenance.
- AS anterior uveitis and hypopyon have zero published images. There are 22
  relevant, AS-associated anterior-uveitis candidates awaiting parsing.
- `PMC7488890` Figure 3 is a real slit-lamp plate; its retained verdict rejects
  disease attribution and its age is unknown. `PMC7982681` Figure 1 contains an
  explicitly attributed AS eye panel but mixes diseases and has unknown ages.
  Neither is automatically eligible under the current whole-figure policy.
- These retained eye judgments date from September 29. The inspected database
  lacks `article_pair_rankings`; the latest reranker has not been demonstrated
  against this database. This is a rollout check, not proof the reranker fails.

These are observations, not guaranteed recoverable yield. Capture a fresh,
consistent baseline at execution time; never reuse these numbers as live counts.
Keep `visual_pilot_image_coverage_plan.md` as historical context. This document
supersedes its scheduling and coverage-target recommendations, not clinical policy.

## 3. Non-Negotiable Invariants

- Preserve commercial-use and third-party licensing checks, review-only source
  restrictions, explicit source-supported age, disease attribution, approved
  finding support, image quality, and whole-figure/mixed-disease restrictions.
- HLA-B27-associated uveitis alone is not proof of AS in the depicted patient.
  Article relevance and retrieval query text are not image attribution.
- Never infer age from disease epidemiology or patient identity from age alone.
- Preserve original panels, files, judgments, audit exclusions, and manual locks.
  A lock cannot override current eligibility, a duplicate group, or the cap.
- No prompt/model/embedding changes, journal-score changes, new ranking weights,
  case-report fallback, or automatic perceptual-duplicate threshold in this wave.
- All tests are offline. Live retrieval, provider calls, and production backfills
  require separate approval. Use scratch databases for mutation tests.

## 4. Fixed Shared Contracts

The coordinator owns policy/configuration, measurement/query authorship, and
acceptance assertions. Agents implement the settled contracts; they do not
invent thresholds, source queries, metrics, or clinical evidence rules.

### C1. One Eligibility Surface

Add `publication.py` with:

```python
eligible_panels(conn, disease_key: str | None = None) -> list[dict]
panel_eligibility(panel: dict, figure: dict, article: dict,
                  approved_findings: set[str]) -> dict
```

`panel_eligibility` returns `eligible`, `reasons`, and `supported_finding_keys`.
Load current-hash exclusions and source metadata; use existing licensing,
`curation.exclusion_reason`, demographics, and source-supported finding logic
currently in the viewer. Move reusable support checks here without weakening
them; viewer serialization remains in the viewer. Missing image/thumb files or
invalid image dimensions make an image unavailable and must be reported.
Unapproved or unsupported finding labels earn no pair credit.

Do not modify the SQL view to approximate regex/source-based checks. All consumers
use this shared Python surface. Retain failed rows for diagnostics, not publication.

### C2. Gallery Selection And Coverage Snapshot

Add `gallery.py` with a pure selection function:

```python
select_gallery(panels: list[dict], disease_key: str, finding_key: str,
               *, cap: int, locked_panel_id: str | None = None) -> dict
coverage_snapshot(conn, disease_key: str | None = None) -> dict
```

Gallery output: `published_panel_ids`, `reserve_panel_ids`, `selection_reasons`,
`identity_unknown_count`. Snapshot maps disease keys to finding keys to records:
`eligible_distinct`, `published_distinct`, `reserve_distinct`, `floor_deficit`,
`target_deficit`, `cap_remaining`, `tier`, and `blocked_reason`.
Reserves include gallery-full and diversity-held images; count disjoint groups,
not duplicate panel rows. Scheduler completion uses `published_distinct`.

Build snapshots from one consistent SQLite read transaction. Refresh after store,
curation changes, lock changes, and configuration changes. Do not cache indefinitely.
Keep the existing single-primary representative table and API compatible; choose
the primary from the actual selected gallery, never from an invisible reserve.
No persisted multi-image gallery table is required in this wave.

### C3. Diversity Without Throwing Away Images

Selection order is fixed:

1. Filter through C1 and approved source-supported pair attribution.
2. Collapse identical stored hashes; empty hashes use `panel_id`, but mark the
   identity limitation. Collapse manually confirmed reused-image families.
3. Allow one representative per documented patient per pair, including images
   reused in different papers. Preserve evidence with every confirmed grouping.
4. Preserve a valid manual primary lock as the first representative of its group.
5. First pass: round-robin source articles, up to two images/article. Within each
   article retain `representatives.score_panel`, ties ascending by `panel_id`.
6. Second pass: fill remaining capacity to 20 from remaining distinct groups in
   score order. The two-per-article preference is soft, not a permanent ceiling.

The same original figure is one source family when identity is undocumented.
Different figures with unknown identities may be selected, but must not be
reported as verified distinct patients. Never merge two patients just because
they have the same age, sex, or article. Near-duplicate suggestions require review
before they affect grouping; no new image-similarity model is required.
Add optional evidence-backed identity metadata in an additive table owned by
the schema workstream: panel ID, patient-group key, reuse-group key, source quote,
review provenance, and reviewed image hash. Stale-hash groupings do not apply.

### C4. Deficit Scheduling

Use these tiers, based on the selected gallery rather than raw stored rows:

| Tier | Published count | Priority |
|---|---|---|
| `empty` | 0 | First |
| `below_floor` | 1-2 | Second |
| `below_target` | 3-9 | Third |
| `expanding` | 10-19 | Fourth |
| `full` | 20 | No new retrieval for this pair |

Only the highest-priority actionable tier receives new expansion slots. Within
it, sort by `(published_distinct, last_served_sequence, disease_key, finding_key)`
and allocate one article per lane per round. Persist last-served sequence so
small batches do not starve alphabetical latecomers. Shared articles use one
slot but mark every matching active lane served. Retain existing pair reranking
inside a lane and existing transient-error retries.

An empty lane needing replenishment stays actionable until bounded search is
attempted. Once explicitly blocked or paused by a safety limit, it must not spin
forever or prevent other actionable tiers from progressing. Record that it still
has a deficit. Never globally fill spare slots from unrelated covered lanes.
Drain already-started downstream work even when it supplies surplus reserves.

`covered` alone must no longer hide the difference between floor reached,
target reached, full, search exhausted, and paused. Use additive lane fields for
tier, last-served sequence, blocked reason, search-policy version, and timestamps;
retain existing candidate terminal outcomes and backwards-compatible statuses.

### C5. Pair Provenance And Bounded Replenishment

Attach explicit `disease_key` to every query context, matched passage, and
manifestation-candidate record before fusion. Merge evidence by disease as well
as finding. Queue sync accepts only matching explicit disease provenance; do not
copy evidence into every disease that shares a finding.

Do not delete legacy ambiguous candidates. Mark their provenance unresolved and
exclude them from active reservations until refreshed by a disease-scoped query.
Legacy query-text inference is not an authoritative replacement for provenance.
Use pair-scoped synonyms: generic `acute anterior uveitis`, `iritis`, and
`iridocyclitis` apply to AS; psoriasis-specific phrases apply only to their
documented psoriasis/PsA pairs. Preserve original vocabulary records.

Add `replenish_pair(conn, disease_key, finding_key, *, round_no, dry_run=False)`
in `select_articles.py`. Return `new_candidates`, `pending_candidates`,
`queries_attempted`, `status`, `reason`; it must not mark a pair covered.
First process unlicensed/license-ok/relevant pending candidates for the pair
through the existing selection pipeline. Only then issue new retrieval.

Coordinator-authored initial retrieval recipe:

- Disease terms: canonical disease name, then its first distinct approved
  non-acronym synonym. Finding terms: canonical label, then pair-valid synonyms.
- Query templates: `{disease} {finding} {modality}`, then
  `{disease} {finding} patient photograph figure caption`.
- For AS eye pairs, modality is `slit lamp ophthalmic image`; AS terms include
  `ankylosing spondylitis` and `axial spondyloarthritis`. Do not use HLA-B27 as a
  disease synonym or attach psoriatic-uveitis synonyms to AS.
- Deterministically enumerate canonical terms first, deduplicate normalized
  query strings, and try at most six unattempted query variants per pair/round.
- Retain current review-only filters. BM25 `page_content` retrieval uses bounded
  depths 300, 600, 1200 across at most three automatic rounds per search-policy
  version. This is deeper top-k, not assumed cursor pagination.
- Persist attempts, policy version, query/filter hash, depth, unique returned
  PMCIDs, newly discovered PMCIDs, pending outcomes, errors, and completion time.
  Resume unfinished attempts; do not restart completed searches blindly.
- If a provider rejects a depth, report the limitation; never silently remove
  source filters. Exhausted rounds mean `search_plan_exhausted`, not proof that
  no eligible image exists anywhere in the corpus.

After a completed batch with no deficit reduction, finish its pending outcomes
and progress to the next unattempted search strategy. Article quotas are buffers,
never success criteria. Existing time, article, and spend limits remain hard.
`--skip-select` and explicit `--pmcids` remain no-new-retrieval modes.

### C6. Viewer And Reporting

Default manifestation galleries show all selected images, up to 20, not five.
Select once per disease/finding, before tab, modality, and pediatric filtering,
so filters cannot produce separate 20-image allowances. Preserve existing design.
Expose reserve counts and a separate inspection path without publishing reserves
into the default gallery. A full gallery cannot truncate the underlying store.

Reports include every approved pair, even when it has no articles or images:
retrieved unique articles, licensed articles, pair-supported caption figures,
eligible distinct images, published distinct images, reserves, milestone/tier,
blocked reason, attempted strategies, last deficit reduction, and next action.
Article membership alone does not make all its figures pair-supported.
Use structured rejection categories: license/third-party, review type, no patient
image, age unclear, attribution unclear/other disease, unsupported finding,
mixed plate, quality, duplicate/diversity reserve, retrieval error, pending work.
Eligibility failures and selection reserves are separate concepts.

## 5. Agent Ownership And Dependencies

| Owner | Exclusive implementation files | Deliverable | Depends on |
|---|---|---|---|
| Coordinator | This plan; policy/config and query/measurement specifications | Settled contracts, reviews, integration approval | None |
| A: Schema/config | `db.py`, `config.py`, `env.example`; new migration tests | Additive lane/search/identity schema and 3/10/20 validation, exactly as coordinator specifies | Coordinator SQL/config artifacts |
| B: Eligibility | New `publication.py`; new eligibility tests | C1 and extracted source-support helpers | A |
| C: Gallery | New `gallery.py`, `representatives.py`, `store.py`; gallery/representative tests | C2/C3, reserves, primary consistency | A, B |
| D: Retrieval | `select_articles.py`, `retrieval.py`, pair-term helpers/data; retrieval/recovery tests | C5 provenance, pair terms, bounded replenishment | A, coordinator query artifacts |
| E: Scheduler | `manifestation_queue.py`, `parse.py`, `cli.py`; queue/scheduler tests | C4 and C5 orchestration, resume/safety semantics | B, C, D |
| F: Viewer | `viewer/app.py`, existing viewer static files; new viewer tests | Shared eligibility, global pair cap, reserve visibility | B, C |
| G: Reporting | `report.py`, new report tests, user-facing README updates | C6 pair funnel and explicit unresolved gaps | C, D, E |

Do not edit another owner's files. Request a contract change through the
coordinator. `prompts.py`, `judge.py`, `article_rank.py`, and `pair_rank.py` stay
unchanged unless a separately reviewed defect requires a new workstream.
Reuse `representatives.score_panel`; no independent ranking implementation.

Execution waves:

1. Coordinator freezes the fresh baseline, authors exact additive SQL/config,
   identity payloads, source-query requests, and acceptance fixtures; A implements.
2. B and D run in parallel after the shared schema is available.
3. C follows B; D may continue independently.
4. E follows C and D; F runs concurrently after C.
5. G integrates completed APIs; coordinator reviews every full diff before landing.

Each handoff lists owned files, settled contracts, exact narrow verification,
evidence paths, and runtime state. Reuse existing processes; no agent launches
live expansion independently. One coordinator owns integration and data writes.

## 6. Acceptance Cases And Gates

Coordinator-owned assertions; agents may implement ordinary unit tests from
these cases, but may not redesign their expected outcomes.

- Eligibility: an audited exclusion, disallowed license, unknown source age,
  unsupported label, or mixed-disease plate earns zero gallery/coverage credit;
  missing files earn zero available-image credit with an actionable reason.
- Equality: viewer, scheduler, report, and primary representative consume the
  same frozen eligible/gallery inputs and agree on pair counts and IDs.
- Cap: 25 distinct eligible images from distinct documented patients select 20,
  retain five reserves, and preserve all 25 DB rows and files.
- Milestones: counts 0/1/2/3/9/10/19/20 produce the defined tiers; 10 remains
  eligible for expansion. Invalid floor/target/cap combinations fail clearly.
- Diversity: identical bytes count once; two confirmed same-patient images earn
  one representative; unrelated patients with equal ages stay separate. Soft
  article preference does not prevent a diverse 20-image gallery from filling.
- Locks: valid lock leads the gallery; invalid or duplicate-conflicting locks
  cannot override eligibility. Rebuilding twice yields identical selected IDs.
- Scheduling: with counts `[0, 2, 10, 20]`, an actionable empty lane wins; after
  it is explicitly search-blocked the two-image lane wins. Batch size one across
  equally covered lanes rotates using persisted last-served sequence.
- Shared article: two active lanes share one parse slot and both advance their
  service sequence. Unrelated covered articles never fill spare deficit slots.
- Provenance: PsA-only uveitis evidence cannot create an AS reservation; explicit
  evidence for both pairs can. Legacy ambiguous rows remain retained/inactive.
- Replenishment: 22 rejected/unsuitable articles with zero images trigger the
  next bounded search round; duplicate-only search does not count as progress.
- Resume: completed query/depth attempts are not repeated; interrupted attempts
  remain resumable; transient errors do not falsely declare corpus exhaustion.
- Safety: dry-run makes no writes/calls; explicit PMCIDs and skip-select make no
  new retrieval; zero remaining runtime/spend prevents another expansion call.
- Viewer: modality/pediatric filters draw subsets of the same 20-image gallery;
  a combined plate earns no per-finding credit; reserves remain inspectable.
- AS evidence: retained mixed plates, HLA-B27-only attribution, and unknown-age
  photographs remain ineligible. No test expects an AS eye image to be invented.

Gate 0: coordinator checks baseline/source definitions and migration preservation.
Gate 1: each owner runs only its changed-module tests and hands back logs/diffs.
Gate 2: offline integration fixtures prove C1-C6, cap/reserve retention, and resume.
Gate 3: one final `python3 -m pytest tests/visual_pilot` pass; no repeated full-suite
gating for unchanged work. Coordinator checks measurement outputs against fixtures.
Gate 4: optional user-approved AS-only live pilot on a scratch copy, with fixed
budget/runtime/article limits; record queries, rankings, outcomes, and gallery IDs.
Do not move a scratch run into the main database without separate review.

## 7. Rollout And Definition Of Done

Initialize and verify the latest ranking/schema on the scratch database first.
Audit retained candidates and judgments; re-evaluate only cases affected by an
actual policy/version change, never blanket-reset every rejected figure.
Process the current AS eye backlog under the installed pair reranker, then use
bounded replenishment while published coverage remains deficient.

Done means the 3/10/20 policy works consistently end to end; surplus is preserved;
empty manifestations receive actionable priority; failed candidates cause new,
tracked work; every remaining gap has an honest reason and next action.
It does not mean every approved manifestation is guaranteed three images.
Unavailable eligible review images remain documented gaps, not relaxed criteria.
