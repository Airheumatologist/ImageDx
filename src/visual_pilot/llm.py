"""LLM provider layer for the visual pilot.

Streamed OpenRouter chat completions in JSON mode, with the schema in the
system prompt and the JSON reply validated locally. One repair retry on
validation failure, an ``llm_calls``-backed response cache + cost ledger, a
budget guard, dry-run mode, and concurrent batching.

There is no provider batch API, so ``call_many``/``iter_many`` run requests
on a thread pool of ``VP_CONCURRENCY`` workers, with a per-client timeout,
lazy completion-order
``iter_many``, process-wide request pacing (``VP_LLM_RPM``), and a 429 retry budget (``VP_RATE_LIMIT_RETRIES``) that is
independent of ``max_retries``. A 402 (OpenRouter credits spent) stops the
run like an exhausted budget.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from types import SimpleNamespace
from typing import Any

import jsonschema
import openai
import httpx

try:  # openai>=3 sends through its own httpx fork; the transport must match.
    import httpx2 as _openai_httpx
except ImportError:  # pragma: no cover - older openai uses httpx itself
    _openai_httpx = httpx

from . import config, db, timing


PROVIDER = "openrouter"
USER_AGENT = "image-dx-visual-pilot/1.0"
# OpenRouter's app attribution header.
APP_TITLE = "image-dx-visual-pilot"


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


# Shared by every LLMClient so overlapping stages stay under the provider's
# account-wide concurrency limit; held only while a request is open.
_IN_FLIGHT = threading.BoundedSemaphore(config.VP_LLM_MAX_IN_FLIGHT)

class _RequestPacer:
    """Spaces requests evenly to stay under a requests-per-minute cap."""

    def __init__(self, rpm: float) -> None:
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._next_allowed = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_allowed)
            self._next_allowed = slot + self.interval
        if slot > now:
            time.sleep(slot - now)
        timing.record("limiter_wait", max(slot - now, 0.0), host="llm")


# Shared by every LLMClient: the provider's rate limit is per account.
_PACER = _RequestPacer(config.VP_LLM_RPM)

# OpenRouter answers 402 once the account's credits are spent; retrying
# cannot help.
_NO_CREDITS = 402

# Error classes worth a retry (rate limits, network blips, server 5xx).
_TRANSIENT_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class _DeadlineStream(_openai_httpx.SyncByteStream):
    """Response body that raises ReadTimeout once a wall-clock deadline passes.

    httpx timeouts are per read, so provider keep-alive whitespace on a
    stalled non-streaming request resets them indefinitely.
    """

    def __init__(self, stream, request, deadline: float) -> None:
        self._stream = stream
        self._request = request
        self._deadline = deadline

    def __iter__(self):
        for chunk in self._stream:
            if time.monotonic() > self._deadline:
                raise _openai_httpx.ReadTimeout(
                    "request exceeded VP_LLM_MAX_REQUEST_SECONDS",
                    request=self._request,
                )
            yield chunk

    def close(self) -> None:
        self._stream.close()


def _env_proxy(url: str) -> str | None:
    """The environment's HTTPS proxy for ``url`` (None if NO_PROXY bypasses it).

    httpx applies HTTPS_PROXY/NO_PROXY only to transports it builds itself,
    so an explicit transport has to be given the proxy.
    """
    host = urllib.parse.urlsplit(url).hostname or ""
    if not host or urllib.request.proxy_bypass(host):
        return None
    return urllib.request.getproxies().get("https")


class _DeadlineTransport(_openai_httpx.HTTPTransport):
    """HTTP transport capping each request's total time at ``max_seconds``."""

    def __init__(self, max_seconds: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_seconds = max_seconds

    def handle_request(self, request):
        deadline = time.monotonic() + self.max_seconds
        response = super().handle_request(request)
        response.stream = _DeadlineStream(response.stream, request, deadline)
        return response


def _chat_extra_kwargs(effort: str) -> dict:
    """Request kwargs beyond the messages: JSON mode, reasoning, token cap."""
    extra_kwargs: dict = {"response_format": {"type": "json_object"}}
    if effort == "off":
        extra_kwargs["extra_body"] = {"reasoning": {"enabled": False}}
    elif effort:
        extra_kwargs["extra_body"] = {"reasoning": {"effort": effort}}
    if config.VP_LLM_MAX_OUTPUT_TOKENS:
        extra_kwargs["max_tokens"] = config.VP_LLM_MAX_OUTPUT_TOKENS
    return extra_kwargs


class LLMClient:
    """Provider-agnostic JSON caller with cache, ledger and budget guard."""

    def __init__(
        self,
        db_conn=None,
        budget_usd: float | None = None,
        dry_run: bool = False,
        max_retries: int = 3,
        concurrency: int | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.api_key = config.OPENROUTER_API_KEY
        self.base_url = config.OPENROUTER_BASE_URL
        self.conn = db_conn
        self.budget_usd = budget_usd
        self.dry_run = dry_run
        self.max_retries = max_retries
        self.concurrency = max(1, concurrency or config.VP_CONCURRENCY)
        # C4: per-client request timeout (judge uses VP_JUDGE_TIMEOUT_SECONDS).
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else config.VP_LLM_TIMEOUT_SECONDS
        )
        self.spent_usd = 0.0  # live spend in this run only
        self._client: openai.OpenAI | None = None
        self._lock = threading.Lock()
        # Worker threads write llm_calls through the caller's connection under
        # this lock; callers streaming iter_many must hold it for their own
        # writes on that connection (see db_lock).
        self._client_init_lock = threading.Lock()

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
        reasoning_effort: str | None = None,
        fill_missing: dict | None = None,
    ) -> tuple[Any, dict]:
        """One JSON call. Returns (parsed_response, meta).

        ``reasoning_effort`` overrides ``LLM_REASONING_EFFORT`` for this call
        (stages pass their ``VP_*_REASONING_EFFORT``); empty means the model
        default. JSON mode does not enforce the schema, so before validation
        top-level keys the schema forbids are dropped, and ``fill_missing``
        supplies top-level keys the reply omits: a value, or a callable
        taking the reply object (e.g. the judge's echoed ``figure_id``). It
        is not part of the cache key.
        """
        images = images or []
        effort = (
            config.LLM_REASONING_EFFORT if reasoning_effort is None else reasoning_effort
        ).strip()
        started = time.monotonic()  # W0/C7 timing hook (observation only)
        input_hash = self._input_hash(
            stage, model, prompt_version, system, user_content, images, effort
        )

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
            parsed, raw_content, usage, attempts = self._call_with_validation(
                model, system, user_content, schema, images, effort, fill_missing
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
            "provider": PROVIDER,
            "response_format": "json_object",
            "cost_usd": cost,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "attempts": attempts,
            "reasoning_effort": effort or None,
        }
        self._record(input_hash, stage, model, prompt_version, raw_content, usage, cost, meta)
        return parsed, meta

    def call_many(self, requests: Iterable[dict]) -> list[BatchResult]:
        """Run call_json requests concurrently (no provider batch API).

        ``requests`` items are kwargs dicts for call_json. Results keep input
        order; per-item failures land in BatchResult.error.
        """
        return sorted(self.iter_many(requests), key=lambda r: r.index)

    def iter_many(
        self,
        requests: Iterable[dict],
        max_in_flight: int | None = None,
    ) -> Iterator[BatchResult]:
        """Stream call_json results in completion order (contract C4).

        ``requests`` items are kwargs dicts for call_json. The iterable is
        consumed lazily: at most ``max_in_flight`` (default
        ``self.concurrency``) calls are in flight, and the next request is
        pulled only when a slot frees — so a generator feeding off a fetch
        pool is never drained ahead of the caller. ``BatchResult.index`` is
        the item's 0-based position in the consumed iterable; per-item
        failures land in ``BatchResult.error``.
        """
        cap = max(1, max_in_flight or self.concurrency)
        items = enumerate(iter(requests))
        with ThreadPoolExecutor(max_workers=cap) as pool:
            pending: dict[Future, int] = {}

            def _fill() -> None:
                while len(pending) < cap:
                    try:
                        index, req = next(items)
                    except StopIteration:
                        return
                    pending[pool.submit(self.call_json, **req)] = index

            _fill()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                results: list[BatchResult] = []
                for fut in done:
                    index = pending.pop(fut)
                    try:
                        parsed, meta = fut.result()
                    except Exception as exc:  # noqa: BLE001 - per-item capture
                        results.append(BatchResult(index=index, error=exc))
                    else:
                        results.append(
                            BatchResult(index=index, parsed=parsed, meta=meta)
                        )
                # Deliver completed results before advancing the request feed.
                # A failing/interrupted producer must not lose these verdicts.
                yield from results
                _fill()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _openai(self) -> openai.OpenAI:
        with self._client_init_lock:
            if self._client is None:
                if not self.api_key:
                    raise LLMError("OPENROUTER_API_KEY is not set")
                self._client = self._new_openai(self.api_key)
            return self._client

    def _new_openai(self, api_key: str) -> openai.OpenAI:
        # max_retries=0: transient retries are handled here so that
        # budget/cache semantics stay under our control.
        return openai.OpenAI(
            api_key=api_key, base_url=self.base_url,
            timeout=self.timeout_seconds, max_retries=0,
            default_headers={"User-Agent": USER_AGENT, "X-Title": APP_TITLE},
            http_client=openai.DefaultHttpxClient(
                transport=_DeadlineTransport(
                    config.VP_LLM_MAX_REQUEST_SECONDS,
                    proxy=_env_proxy(self.base_url),
                ),
            ),
        )

    def _input_hash(
        self,
        stage: str,
        model: str,
        prompt_version: str,
        system: str,
        user_content: str,
        images: list[ImageInput],
        effort: str = "",
    ) -> str:
        payload = {
            "prompt_version": prompt_version,
            "system": system,
            "user_content": user_content,
            "images": [im.identity() for im in images],
        }
        if effort:
            # Only a set effort joins the identity, so model-default calls
            # keep the cache keys they had before efforts were configurable.
            payload["reasoning_effort"] = effort
        return db.llm_input_hash(stage, model, payload)

    @property
    def db_lock(self) -> threading.Lock:
        """Lock guarding the shared ``db_conn``.

        ``iter_many`` workers insert and commit ``llm_calls`` rows on the
        caller's connection; a caller writing to the same connection while
        results stream must hold this lock, or a worker commit can end the
        caller's transaction between its writes and its own commit.
        """
        return self._lock

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
        effort: str = "",
    ):
        """Single streamed chat completion in JSON mode."""
        messages = [
            {
                "role": "system",
                "content": (
                    f"{system}\n\nRespond with a single JSON object (the data "
                    "itself, not a schema) that validates against this JSON "
                    f"Schema:\n{json.dumps(schema)}"
                ),
            },
            {"role": "user", "content": self._user_parts(user_content, images)},
        ]

        def _send():
            with _IN_FLIGHT:
                return self._stream_chat(model, messages, _chat_extra_kwargs(effort))

        return self._with_retries(_send)

    def _stream_chat(self, model: str, messages: list[dict], extra_kwargs: dict):
        """One streamed chat completion, returned in the non-streamed shape.

        Streamed so a slow generation never sits on an idle socket.
        _normalize unwraps fenced replies in case JSON mode is ignored.
        """
        stream = self._openai().chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **extra_kwargs,
        )
        content: list[str] = []
        usage = None
        got_choice = False
        try:
            for chunk in stream:
                error = getattr(chunk, "error", None)
                if error:
                    # OpenRouter reports an upstream failure after the 200 as
                    # a chunk carrying ``error``; retry it like a 5xx.
                    raise openai.InternalServerError(
                        f"provider error mid-stream: {error}",
                        response=self._error_response(), body=None,
                    )
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                if chunk.choices:
                    got_choice = True
                    content.append(chunk.choices[0].delta.content or "")
        except _openai_httpx.TimeoutException as exc:
            raise openai.APITimeoutError(request=stream.response.request) from exc
        except _openai_httpx.TransportError as exc:
            raise openai.APIConnectionError(
                message=f"stream interrupted: {exc}", request=stream.response.request
            ) from exc
        finally:
            stream.close()
        if not got_choice:
            raise openai.InternalServerError(
                "provider error: empty choices returned",
                response=self._error_response(), body=None,
            )
        message = SimpleNamespace(content="".join(content))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    def _error_response(self) -> httpx.Response:
        """A 502 response object for errors raised from a 200 stream."""
        return httpx.Response(502, request=httpx.Request(
            "POST", self.base_url.rstrip("/") + "/chat/completions"))

    def _call_with_validation(
        self,
        model: str,
        system: str,
        user_content: str,
        schema: dict,
        images: list[ImageInput],
        effort: str = "",
        fill_missing: dict | None = None,
    ) -> tuple[Any, str, dict, int]:
        usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
        attempts = 0
        last_error = "no response"
        for retry in range(2):
            request_text = user_content
            if retry == 1:
                request_text = (
                    f"{user_content}\n\nThe previous response failed validation: "
                    f"{last_error}\nReturn corrected JSON only."
                )
            resp = self._send_once(model, system, request_text, schema, images, effort)
            attempts += 1
            u = getattr(resp, "usage", None)
            if u is not None:
                usage["input_tokens"] += getattr(u, "prompt_tokens", 0) or 0
                usage["cached_input_tokens"] += _cached_tokens(u)
                usage["output_tokens"] += getattr(u, "completion_tokens", 0) or 0
                reported = getattr(u, "cost", None)
                if reported is not None:
                    usage["cost_usd"] = usage.get("cost_usd", 0.0) + float(reported)
            raw = resp.choices[0].message.content or ""
            content = _normalize(raw, schema, fill_missing or {})
            parsed, last_error = self._parse_validate(content, schema)
            if last_error is None:
                return parsed, content, usage, attempts
        raise SchemaValidationError(
            f"response failed schema validation: {last_error}; "
            f"raw reply began {raw[:300]!r}"
        )

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
        """``call()`` with retries.

        C4: two independent retry budgets. HTTP 429s consume
        VP_RATE_LIMIT_RETRIES and honor Retry-After; timeouts, connection
        errors and 5xx consume max_retries with the original backoff. A 402
        (credits spent) raises BudgetExceeded at once.
        """
        transient_attempt = 0
        rate_limit_attempt = 0
        while True:
            _PACER.wait()  # every attempt is one provider request
            try:
                return call()
            except openai.APIStatusError as exc:
                if exc.status_code == _NO_CREDITS:
                    timing.count("llm_402")
                    raise BudgetExceeded(f"OpenRouter credits exhausted: {exc}") from exc
                if not isinstance(exc, _TRANSIENT_ERRORS):
                    raise
                error = exc
            except _TRANSIENT_ERRORS as exc:
                error = exc
            # W0/C7: count transient failures (429s/timeouts) — observation only.
            if isinstance(error, openai.RateLimitError):
                timing.count("llm_429")
                if rate_limit_attempt >= config.VP_RATE_LIMIT_RETRIES:
                    raise error
                time.sleep(self._retry_delay_429(error, rate_limit_attempt))
                rate_limit_attempt += 1
            else:
                if isinstance(error, openai.APITimeoutError):
                    timing.count("llm_timeout")
                else:
                    timing.count("llm_transient")
                if transient_attempt >= self.max_retries:
                    raise error
                time.sleep(min(8.0, 2.0**transient_attempt))
                transient_attempt += 1

    @staticmethod
    def _retry_delay_429(exc: openai.RateLimitError, attempt: int) -> float:
        """Seconds to wait before retrying a 429.

        Honors a ``Retry-After`` response header (seconds or HTTP date);
        falls back to exponential backoff. Either way capped at 30 s.
        """
        delay: float | None = None
        response = getattr(exc, "response", None)
        header = None
        if response is not None:
            try:
                header = response.headers.get("Retry-After")
            except Exception:  # noqa: BLE001 - tolerate odd response objects
                header = None
        if header:
            try:
                delay = float(header)
            except ValueError:
                try:
                    when = parsedate_to_datetime(header)
                    delay = (when - datetime.now(timezone.utc)).total_seconds()
                except (TypeError, ValueError):
                    delay = None
        if delay is None:
            delay = 2.0**attempt
        return min(30.0, max(0.0, delay))

    def _cost(self, model: str, usage: dict) -> float | None:
        if usage.get("cost_usd") is not None:
            return usage["cost_usd"]  # OpenRouter's billed amount
        price = config.model_price(model)
        if price is None:
            return None
        in_rate, cached_rate, out_rate = price
        cached = usage.get("cached_input_tokens") or 0
        uncached = max(0, (usage.get("input_tokens") or 0) - cached)
        return (
            uncached * in_rate
            + cached * cached_rate
            + (usage.get("output_tokens") or 0) * out_rate
        ) / 1e6

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
            "provider": PROVIDER,
        }
        row = (
            stage,
            model,
            input_hash,
            db.to_json(request_meta),
            raw_content,
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            cost,
        )
        with self._lock:
            for attempt in range(2):
                try:
                    self._insert_call(row)
                    return
                except sqlite3.OperationalError:
                    # A write on a stale WAL snapshot fails at once and leaves
                    # the shared connection failing every write until
                    # rollback; retry once on a fresh snapshot, and never
                    # hand the caller a stuck connection.
                    self.conn.rollback()
                    if attempt:
                        raise

    def _insert_call(self, row: tuple) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO llm_calls "
            "(stage, model, input_hash, request_meta_json, response_json, "
            " input_tokens, output_tokens, cost_usd) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            row,
        )
        self.conn.commit()


def _unwrap_json(content: str) -> str:
    """The JSON object in a reply that may wrap it in fences or prose."""
    text = content.strip()
    try:
        json.loads(text)
        return text
    except ValueError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return content
    candidate = text[start : end + 1]
    try:
        json.loads(candidate)
    except ValueError:
        return content
    return candidate


def _normalize(content: str, schema: dict, defaults: dict) -> str:
    """Repair top-level drift in a JSON reply before schema validation.

    Unwraps a JSON object from markdown fences or surrounding prose, drops
    keys a closed schema (``additionalProperties: false``) forbids and sets
    absent keys from ``defaults`` (callables get the reply object). Anything
    else, including nested problems, is left to validation.
    """
    unwrapped = _unwrap_json(content)
    try:
        parsed = json.loads(unwrapped)
    except ValueError:
        return content
    if not isinstance(parsed, dict):
        return unwrapped
    changed = False
    allowed = schema.get("properties")
    if schema.get("additionalProperties") is False and allowed is not None:
        for key in [k for k in parsed if k not in allowed]:
            del parsed[key]
            changed = True
    for key, value in defaults.items():
        if key not in parsed:
            parsed[key] = value(parsed) if callable(value) else value
            changed = True
    return json.dumps(parsed, ensure_ascii=False) if changed else unwrapped


def _cached_tokens(usage) -> int:
    """Prompt tokens served from the provider cache.

    OpenRouter reports the OpenAI-style ``prompt_tokens_details.cached_tokens``;
    a flat ``usage.cached_tokens`` is accepted too.
    """
    cached = getattr(usage, "cached_tokens", None)
    if cached is None:
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details else None
    return int(cached or 0)
