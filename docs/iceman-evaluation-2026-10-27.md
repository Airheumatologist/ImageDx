# Iceman Contributor Free evaluation — October 27, 2026

Historical initial attempt. The follow-up succeeded after removing the
unsupported `response_format` parameter; see
[the current provider evaluation](provider-evaluation-2026-10-27.md).

**Result: integration prepared; production replacement blocked.** The model
is listed by OpenCode Go, but the project's existing OpenCode key receives
HTTP 503, `Upstream request failed: Endpoint is unavailable.` No successful
completion was returned, so quality, latency under load and usable throughput
have not been established. The active `.env`, production database and existing
Xiaomi provider defaults were left intact.

## Catalog and API findings

- Model ID: `iceman-contributor-free`; pipeline ID:
  `opencode-go/iceman-contributor-free`.
- Endpoint: `https://opencode.ai/inference/go/openai/v1/chat/completions`.
- Catalog metadata advertises text input/output, reasoning, no attachments,
  a 262,144-token context and 131,072-token output limit. It does not advertise
  configurable reasoning options. It cannot replace the image judge.
- Currently free for a limited time. OpenCode documents use of prompts and
  completions to improve the model; the probes used synthetic input only.
- Go requires a client-specific user agent and a stable `x-opencode-session`.
  The integration sends `image-dx-visual-pilot/1.0` and a UUID per client.
- The saved local Go credential returned HTTP 401 (`Invalid credential`).
  The separate project OpenCode key returned HTTP 503 on the canonical
  endpoint and the legacy `/zen/go/v1` endpoint, for both minimal and JSON
  requests. The saved local credential was not copied or changed.

Sources: [OpenCode Go API, limits and privacy](https://opencode.ai/docs/go/),
[live Go model list](https://opencode.ai/inference/go/v1/models),
[model metadata](https://models.dev/api.json).

## Tests

`python -m src.visual_pilot.smoke_iceman` uses the pipeline's real scoped P2
prompt/schema and six synthetic captions: untreated patient image, flowchart,
animal image, treatment comparison, vague caption and third-party reproduction.
It makes no database writes. The October 27 run failed before any completion:

```json
{
  "model": "opencode-go/iceman-contributor-free",
  "passed": false,
  "error_type": "InternalServerError",
  "error": "Error code: 503 - {'error': {'type': 'server_error', 'message': 'Upstream request failed: Endpoint is unavailable.'}}",
  "seconds": 0.25
}
```

The local provider tests cover Go credentials and endpoint isolation, stable
session headers, streamed JSON, usage/cost recording, response caching,
separate MiMo/Iceman cache identities, missing credentials and rejection of
image requests before cache or network access. Existing Xiaomi behavior,
retry handling and deadline tests continue to run.

Validation: **170 tests and 11 subtests passed** in the full
`tests/visual_pilot` suite. `git diff --check` also passed.

## Candidate activation

`env.iceman.example` contains a candidate overlay for the four text stages,
retaining MiMo for P3 vision. Go uses `OPENCODE_GO_API_KEY`, falling back to
the existing `OPENCODE_API_KEY`; Xiaomi credentials are never sent to Go.
Xiaomi's `thinking` setting is not forwarded to Iceman. Failures never silently
switch providers or write fabricated successful results.

Re-run the live smoke test once the endpoint is available. If it passes,
evaluate description, extraction and vocabulary on a small isolated sample
before applying the overlay. A complete replacement also requires a separately
tested vision-capable model. A P2 smoke-test pass alone does not establish the
quality of those other stages or clinical correctness.
