"""LLM provider layer for the visual pilot (workstream W4).

OpenAI-compatible chat calls on OpenCode Zen (``VP_LLM_PROVIDER=opencode``)
with strict JSON-schema output, one repair retry on validation failure, an
``llm_calls``-backed response cache + cost ledger, a budget guard, dry-run
mode, and concurrent batching.

There is no provider batch API, so ``call_many`` runs requests on a thread
pool of ``VP_CONCURRENCY`` workers.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import jsonschema
import openai

from . import config, db, timing
from .prompts import Prompt


class LLMError(Exception):
    pass


class BudgetExceeded(LLMError):
    """Raised before a live call when the run budget is exhausted."""


class SchemaValidationError(LLMError):
    """Response failed JSON-schema validation even after the repair retry."""


@dataclass(frozen=True)
class ImageInput:
    """One image for the model: a public URL or a base64 data URL."""

    url: str | None = None
    data_url: str | None = None
    sha256: str | None = None  # identity of the underlying bytes (bytes mode)

    def identity(self) -> str:
        """Stable cache-hash identity for this image."""
        if self.url:
            return f"url:{self.url}"
        if self.sha256:
            return f"sha256:{self.sha256}"
        return f"sha256:{hashlib.sha256((self.data_url or '').encode()).hexdigest()}"

    def as_content_part(self) -> dict:
        value = self.url or self.data_url
        return {"type": "image_url", "image_url": {"url": value}}


@dataclass
class BatchResult:
    index: int
    parsed: Any = None
    meta: dict | None = None
    error: Exception | None = None


# Error classes worth a retry (rate limits, network blips, server 5xx).
_TRANSIENT_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class LLMClient:
    """Provider-agnostic JSON caller with cache, ledger and budget guard."""

    def __init__(
        self,
        provider: str | None = None,
        db_conn=None,
        budget_usd: float | None = None,
        dry_run: bool = False,
        max_retries: int = 3,
        concurrency: int | None = None,
    ) -> None:
        self.provider = (provider or config.VP_LLM_PROVIDER).strip().lower()
        self.api_key, self.base_url = config.llm_credentials(self.provider)
        self.conn = db_conn
        self.budget_usd = budget_usd
        self.dry_run = dry_run
        self.max_retries = max_retries
        self.concurrency = max(1, concurrency or config.VP_CONCURRENCY)
        self.spent_usd = 0.0  # live spend in this run only
        self._client: openai.OpenAI | None = None
        self._lock = threading.Lock()
        # Remember per model whether strict json_schema is accepted.
        self._response_mode: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def call_json(
        self,
        stage: str,
        model: str,
        system: str,
        user_content: str,
        schema: dict,
        images: list[ImageInput] | None = None,
        prompt_version: str = "",
    ) -> tuple[Any, dict]:
        """One JSON call. Returns (parsed_response, meta)."""
        images = images or []
        started = time.monotonic()  # W0/C7 timing hook (observation only)
        input_hash = self._input_hash(stage, model, prompt_version, system, user_content, images)

        cached = self._cache_lookup(input_hash)
        if cached is not None:
            timing.record("llm_latency", time.monotonic() - started, stage=stage, source="cache")
            return json.loads(cached["response_json"]), {
                "cached": True,
                "input_hash": input_hash,
                "stage": stage,
                "model": model,
                "cost_usd": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "attempts": 0,
            }

        if config.VP_LLM_CACHE_ONLY:
            # W0 parity harness: never hit the provider on a cache miss.
            raise LLMError(f"cache miss: {stage} {input_hash}")

        if self.dry_run:
            timing.record("llm_latency", time.monotonic() - started, stage=stage, source="dry_run")
            return None, {
                "cached": False,
                "dry_run": True,
                "input_hash": input_hash,
                "stage": stage,
                "model": model,
                "n_images": len(images),
            }

        self._check_budget()

        with timing.inflight("llm"):
            parsed, raw_content, usage, attempts, mode = self._call_with_validation(
                model, system, user_content, schema, images
            )
        timing.record("llm_latency", time.monotonic() - started, stage=stage, source="live")
        cost = self._cost(model, usage)
        with self._lock:
            self.spent_usd += cost or 0.0

        meta = {
            "cached": False,
            "input_hash": input_hash,
            "stage": stage,
            "model": model,
            "provider": self.provider,
            "response_format": mode,
            "cost_usd": cost,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "attempts": attempts,
        }
        self._record(input_hash, stage, model, prompt_version, raw_content, usage, cost, meta)
        return parsed, meta

    def call_prompt(
        self,
        stage: str,
        prompt: Prompt,
        model: str,
        user_content: str,
        images: list[ImageInput] | None = None,
    ) -> tuple[Any, dict]:
        """call_json with a Prompt object (brings system, schema, version)."""
        return self.call_json(
            stage=stage,
            model=model,
            system=prompt.system,
            user_content=user_content,
            schema=prompt.schema,
            images=images,
            prompt_version=prompt.version,
        )

    def call_many(self, requests: Iterable[dict]) -> list[BatchResult]:
        """Run call_json requests concurrently (no provider batch API).

        ``requests`` items are kwargs dicts for call_json. Results keep input
        order; per-item failures land in BatchResult.error.
        """
        requests = list(requests)
        results: list[BatchResult | None] = [None] * len(requests)
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {
                pool.submit(self.call_json, **req): i for i, req in enumerate(requests)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    parsed, meta = fut.result()
                    results[i] = BatchResult(index=i, parsed=parsed, meta=meta)
                except Exception as exc:  # noqa: BLE001 - per-item capture
                    results[i] = BatchResult(index=i, error=exc)
        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _openai(self) -> openai.OpenAI:
        if self._client is None:
            if not self.api_key:
                raise LLMError(
                    f"no API key configured for provider {self.provider!r}"
                )
            # max_retries=0: transient retries are handled here so that
            # budget/cache semantics stay under our control.
            self._client = openai.OpenAI(
                api_key=self.api_key, base_url=self.base_url,
                timeout=config.VP_LLM_TIMEOUT_SECONDS, max_retries=0,
            )
        return self._client

    def _input_hash(
        self,
        stage: str,
        model: str,
        prompt_version: str,
        system: str,
        user_content: str,
        images: list[ImageInput],
    ) -> str:
        return db.llm_input_hash(
            stage,
            model,
            {
                "prompt_version": prompt_version,
                "system": system,
                "user_content": user_content,
                "images": [im.identity() for im in images],
            },
        )

    def _cache_lookup(self, input_hash: str):
        if self.conn is None:
            return None
        with self._lock:
            return self.conn.execute(
                "SELECT response_json FROM llm_calls WHERE input_hash = ?", (input_hash,)
            ).fetchone()

    def _check_budget(self) -> None:
        with self._lock:
            if self.budget_usd is not None and self.spent_usd >= self.budget_usd:
                raise BudgetExceeded(
                    f"budget ${self.budget_usd:.2f} exhausted "
                    f"(spent ${self.spent_usd:.4f} this run)"
                )

    def _user_parts(self, user_content: str, images: list[ImageInput]) -> Any:
        if not images:
            return user_content
        parts = [{"type": "text", "text": user_content}]
        parts.extend(im.as_content_part() for im in images)
        return parts

    def _send_once(
        self,
        model: str,
        system: str,
        user_content: str,
        schema: dict,
        images: list[ImageInput],
    ):
        """Single chat completion with strict-schema -> json_object fallback."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": self._user_parts(user_content, images)},
        ]
        mode = self._response_mode.get(model, "json_schema")
        try:
            resp = self._with_retries(
                lambda: self._openai().chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    response_format=_response_format_for(mode, schema),
                )
            )
        except openai.BadRequestError as exc:
            if mode == "json_schema" and self._looks_like_schema_unsupported(exc):
                mode = "json_object"
                self._response_mode[model] = mode
                resp = self._with_retries(
                    lambda: self._openai().chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0,
                        response_format={"type": "json_object"},
                    )
                )
            else:
                raise
        else:
            self._response_mode[model] = mode
        return resp, mode

    def _call_with_validation(
        self,
        model: str,
        system: str,
        user_content: str,
        schema: dict,
        images: list[ImageInput],
    ) -> tuple[Any, str, dict, int, str]:
        usage = {"input_tokens": 0, "output_tokens": 0}
        attempts = 0
        last_error = "no response"
        for retry in range(2):
            request_text = user_content
            if retry == 1:
                request_text = (
                    f"{user_content}\n\nThe previous response failed validation: "
                    f"{last_error}\nReturn corrected JSON only."
                )
            resp, mode = self._send_once(model, system, request_text, schema, images)
            attempts += 1
            u = getattr(resp, "usage", None)
            if u is not None:
                usage["input_tokens"] += getattr(u, "prompt_tokens", 0) or 0
                usage["output_tokens"] += getattr(u, "completion_tokens", 0) or 0
            content = resp.choices[0].message.content or ""
            parsed, last_error = self._parse_validate(content, schema)
            if last_error is None:
                return parsed, content, usage, attempts, mode
        raise SchemaValidationError(f"response failed schema validation: {last_error}")

    @staticmethod
    def _parse_validate(content: str, schema: dict) -> tuple[Any, str | None]:
        try:
            parsed = json.loads(content)
        except ValueError as exc:
            return None, f"invalid JSON: {exc}"
        try:
            jsonschema.validate(parsed, schema)
        except jsonschema.ValidationError as exc:
            path = "/".join(str(p) for p in exc.absolute_path)
            return parsed, f"schema violation at /{path}: {exc.message}"
        return parsed, None

    def _with_retries(self, call):
        for attempt in range(self.max_retries + 1):
            try:
                return call()
            except _TRANSIENT_ERRORS as exc:
                # W0/C7: count transient failures (429s/timeouts) — observation only.
                if isinstance(exc, openai.RateLimitError):
                    timing.count("llm_429")
                elif isinstance(exc, openai.APITimeoutError):
                    timing.count("llm_timeout")
                else:
                    timing.count("llm_transient")
                if attempt >= self.max_retries:
                    raise
                time.sleep(min(8.0, 2.0**attempt))
        return None  # unreachable

    @staticmethod
    def _looks_like_schema_unsupported(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(
            marker in text
            for marker in ("response_format", "json_schema", "json_schema", "schema")
        )

    def _cost(self, model: str, usage: dict) -> float | None:
        price = config.model_price(model)
        if price is None:
            return None
        in_rate, out_rate = price
        return (
            usage.get("input_tokens") or 0
        ) * in_rate / 1e6 + (usage.get("output_tokens") or 0) * out_rate / 1e6

    def _record(
        self,
        input_hash: str,
        stage: str,
        model: str,
        prompt_version: str,
        raw_content: str,
        usage: dict,
        cost: float | None,
        meta: dict,
    ) -> None:
        if self.conn is None:
            return
        request_meta = {
            "prompt_version": prompt_version,
            "attempts": meta["attempts"],
            "response_format": meta["response_format"],
            "provider": self.provider,
        }
        with self._lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO llm_calls "
                "(stage, model, input_hash, request_meta_json, response_json, "
                " input_tokens, output_tokens, cost_usd) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    stage,
                    model,
                    input_hash,
                    db.to_json(request_meta),
                    raw_content,
                    usage.get("input_tokens"),
                    usage.get("output_tokens"),
                    cost,
                ),
            )
            self.conn.commit()


def _response_format_for(mode: str, schema: dict) -> dict:
    if mode == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema, "strict": True},
        }
    return {"type": "json_object"}
