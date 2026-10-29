# Approved topic expansion — 28 October 2026

The main catalog now has **2,136 topics across 20 specialties**: the existing
1,643 and **493 approved additions** (128 P1 and 365 P2). All existing topic IDs
are retained. The 36 source mentions assigned to existing parents were added
as labeled sections in 31 existing topics. Those expansions, plus the incorrect
adamantinoma synonym removal, correct 32 existing entries; 11 titles were broadened.
Old titles remain searchable. Former narrow ontology identifiers are retained
under `prior_scope_ontology_mappings` rather than labeling a broader group.
The local database's disease records were synchronized with the catalog. Existing
articles, figures, panels, findings, vocabulary and LLM-call history were preserved;
their complete table contents have matching before/after hashes in the sync audit.

New topics carry `curation_status: pending_images`, visual manifestations,
review provenance, priority and any checked reference URLs. They stay in the
master list while images are acquired. Catalog inclusion is separate from image
verification and publication. The static site has not been regenerated.

## Start only the new topics

From the repository root:

```sh
venv/bin/python scripts/run_new_topics.py --dry-run
venv/bin/python -u scripts/run_new_topics.py
```

When another acquisition job is active, use:

```sh
venv/bin/python -u scripts/run_new_topics.py --isolated
```

This stores the new job's database and generated vocabulary in
`data/visual_pilot/new_topics_20261028/`. It leaves the other job's database,
vocabulary, status and logs untouched. Results stay separate for consolidation
after the ongoing run completes; no automatic merge or site export occurs.

The launcher uses `data/new_topics_20261028.json` (exactly 493 IDs), also available
one per line in `data/new_topic_ids_20261028.txt`. Corrected existing parent topics
are excluded from this acquisition scope. The launcher has been dry-run checked;
the acquisition run has **not** been started.

The reference run is `data/visual_pilot/resume_20261027_checkpoint.json`. Its recorded
status is completed with exit code 0 at 2026-10-28 20:15:33 UTC (3:15 PM Chicago).
The new launcher matches its saved acquisition settings:

| Setting | Value |
| --- | --- |
| Topic image target | 100 |
| Article batch size | 300 |
| Acquisition runtime cap | 86,400 seconds |
| Articles per finding pair | 25 |
| Maximum backfill rounds | 3 |
| Overview and manifestation review passes | Skipped |
| Text stages, including vocabulary | `opencode-go/iceman-contributor-free` |
| Vision judge | `z-ai/glm-5.3-flash` |
| OpenRouter provider allowlist | `open-inference` |
| General / judge concurrency | 64 / 64 |
| Maximum LLM requests in flight | 128 |
| Vision reasoning effort | `low` |
| Explicit dollar budget | None, matching the reference run |

Additional safe runtime settings are frozen in the manifest. Credentials continue
to come from the existing environment / `.env`; no credentials are in the manifest.
The 24-hour cap applies to the acquisition stage, after vocabulary preparation.

Before acquisition, the launcher builds missing finding vocabularies only for
these additions, verifies that every addition has usable vocabulary, then expands
their caption terms. An incomplete vocabulary stops acquisition and records the
failure; rerunning retries missing topics. The acquisition pipeline resumes
unfinished articles belonging to these additions. Logs and status are separate:

- `data/visual_pilot/run_new_topics_20261028.log`
- `data/visual_pilot/new_topics_20261028_status.json`

The final extraction step now honors the selected topic list, including shared
articles, and preserves unrelated topics' text and image assertions. Reports can
still summarize the full library; they do not expand acquisition scope. The
launcher does not export or deploy the static site. Use the normal export workflow
after image curation.

## Review and audit files

- Approved additions: `outputs/final_new_topics/final_new_topics.json`
- Source decisions: `outputs/final_new_topics/topic_decisions.json`
- Original catalog backup: `outputs/catalog_update_20261028/main_database_topics.before.json`
- Catalog change report: `outputs/catalog_update_20261028/change_report.json`
- Database preservation audit: `outputs/catalog_update_20261028/database_sync_report.json`
- One-time migration: `scripts/apply_new_topics_20261028.py` (refuses a second application)

Expanded search names were checked against authoritative nomenclature where
needed, including [CLIPPERS](https://www.ncbi.nlm.nih.gov/medgen/1433964),
[SHORT syndrome](https://www.medlineplus.gov/genetics/condition/short-stature-hyperextensibility-hernia-ocular-depression-rieger-anomaly-and-teething-delay/),
[CHILD syndrome](https://medlineplus.gov/genetics/condition/child-syndrome/),
[WHIM syndrome](https://www.ncbi.nlm.nih.gov/gtr/conditions/C0472817/),
[PLCG2-associated immune dysregulation](https://pubmed.ncbi.nlm.nih.gov/37769878/),
and [spastic paraplegia type 11](https://www.medlineplus.gov/genetics/condition/spastic-paraplegia-type-11/).
These nomenclature checks do not replace topic-by-topic image verification.
