# Active provider configuration — October 27, 2026

The pipeline now uses **OpenCode Go Iceman for text** and
**OpenRouter GLM 5.3 Flash through OpenInference only for vision**. The
stage defaults, environment templates and active local `.env` have been
updated. Existing API keys were preserved.

| Stage | Input | Model | Service |
| --- | --- | --- | --- |
| P2 caption triage | Text | `opencode-go/iceman-contributor-free` | OpenCode Go |
| P4 finding extraction | Text | `opencode-go/iceman-contributor-free` | OpenCode Go |
| P5 display descriptions | Text | `opencode-go/iceman-contributor-free` | OpenCode Go |
| P6 vocabulary | Text | `opencode-go/iceman-contributor-free` | OpenCode Go |
| P8 mimic audit | Text | `opencode-go/iceman-contributor-free` | OpenCode Go |
| P3 image judge | Text + image | `z-ai/glm-5.3-flash` | OpenRouter / OpenInference |

OpenRouter requests include this hard restriction, including on retries and
schema-repair calls:

```json
{
  "provider": {
    "only": ["open-inference"],
    "order": ["open-inference"],
    "allow_fallbacks": false,
    "require_parameters": true
  }
}
```

Relace was removed at the user's request. A direct image probe restricted
to Relace had returned HTTP 404: no allowed image-capable endpoint. If
OpenInference is unavailable, requests fail without trying Relace or other
providers. The actual upstream provider and response ID are recorded in
the call ledger. The allowlist also enters the cache identity, so existing
unrestricted OpenRouter calls cannot satisfy these restricted requests.

## Live checks

All five text prompt checks passed against Iceman: six triage routes,
source-supported finding extraction with a stated percentage, display
description, usable vocabulary and rejection of a misattributed mimic.
Replies validated against the production schemas.

All three vision checks passed against GLM. They were re-run after the
allowlist was reduced to OpenInference alone, and every response identified
its actual upstream as `OpenInference`:

- Identify the colors and shapes in a synthetic image.
- Reject a schematic diagram using the actual P3 schema.
- Accept a published CC-BY clinical photograph from the exported library
  (`PMC7139382:ijms-21-02178-f006`), identifying dermatomyositis, clinical
  photograph modality and the source-supported `v_sign` finding.

The final OpenInference-only vision run took **12.95 seconds** with three
concurrent checks and cost **$0.0006572841**. Iceman reported zero cost.
The earlier all-stage run took 40.71 seconds and passed eight checks. Its
vision allowlist still contained both OpenInference and Relace, though all
three actual vision responses came from OpenInference. The final vision
revalidation confirms the current single-provider policy.

The initial shape fixture was close to square, and the model called it a
square instead of a rectangle. It was replaced with an elongated rectangle
before the final successful checks. No clinical acceptance criteria were
relaxed.

Machine-readable replies, costs and provider IDs are in
[`provider-smoke-2026-10-27.json`](provider-smoke-2026-10-27.json).
The live tests do not write to the production database or exported library.

```bash
python -m src.visual_pilot.smoke_providers --clinical
python -m src.visual_pilot.smoke_providers --clinical --vision-only
```

## Implementation and regression checks

Iceman now responds successfully, but rejects structured `response_format`.
Its requests omit that field and rely on the schema in the prompt, local
JSON-schema validation and one repair retry. GLM uses JSON mode and low
reasoning effort. Go requests carry the application's own user agent and a
stable session header. Credentials and HTTP clients are separate per service.

The regression tests cover mixed text/vision batches, credential isolation,
hard provider restrictions, retries, schema repair, actual provider metadata,
reported costs, deadlines, cache separation and Iceman's text-only guard.
All **178 tests and 11 subtests** passed. `git diff --check` passed.

These are integration checks on a small sample, not a broad clinical-quality
evaluation or a benchmark at production concurrency. Existing accepted
database results were not regenerated. Newly started pipeline processes read
the updated `.env`; processes already running retain their imported settings.

Sources: [OpenRouter model and routes](https://openrouter.ai/z-ai/glm-5.3-flash),
[provider restrictions](https://openrouter.ai/docs/guides/routing/provider-selection),
[OpenCode Go](https://opencode.ai/docs/go/).
