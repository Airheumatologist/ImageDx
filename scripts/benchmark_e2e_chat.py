#!/usr/bin/env python3
"""Reproducible async load test harness for /api/v1/chat.

Example:
  python scripts/benchmark_e2e_chat.py \
    --base-url http://localhost:8000 \
    --concurrency-steps 1,2,4,8,16,32 \
    --duration-seconds 120 \
    --query "What is the treatment for myxedema coma?" \
    --timeout-seconds 90 \
    --baseline-req-min 28.9
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import signal
import string
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class StepResult:
    concurrency: int
    duration_seconds: float
    elapsed_seconds: float
    total_requests: int
    success_requests: int
    error_requests: int
    rate_limited_requests: int
    status_counts: dict[int, int] = field(default_factory=dict)
    exception_counts: dict[str, int] = field(default_factory=dict)
    latency_ms: list[float] = field(default_factory=list)

    @property
    def success_req_per_sec(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.success_requests / self.elapsed_seconds

    @property
    def success_req_per_min(self) -> float:
        return self.success_req_per_sec * 60.0

    @property
    def error_rate_pct(self) -> float:
        if self.total_requests <= 0:
            return 0.0
        return (self.error_requests / self.total_requests) * 100.0

    @property
    def p50_latency_ms(self) -> float:
        return percentile(self.latency_ms, 50)

    @property
    def p95_latency_ms(self) -> float:
        return percentile(self.latency_ms, 95)


class StepAccumulator:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.total_requests = 0
        self.success_requests = 0
        self.error_requests = 0
        self.rate_limited_requests = 0
        self.status_counts: dict[int, int] = {}
        self.exception_counts: dict[str, int] = {}
        self.latency_ms: list[float] = []

    async def record_response(self, status_code: int, latency_ms: float) -> None:
        async with self._lock:
            self.total_requests += 1
            self.status_counts[status_code] = self.status_counts.get(status_code, 0) + 1
            self.latency_ms.append(latency_ms)
            if 200 <= status_code < 300:
                self.success_requests += 1
            else:
                self.error_requests += 1
                if status_code == 429:
                    self.rate_limited_requests += 1

    async def record_exception(self, exception_name: str) -> None:
        async with self._lock:
            self.total_requests += 1
            self.error_requests += 1
            self.exception_counts[exception_name] = self.exception_counts.get(exception_name, 0) + 1

    def to_result(self, concurrency: int, duration_seconds: float, elapsed_seconds: float) -> StepResult:
        return StepResult(
            concurrency=concurrency,
            duration_seconds=duration_seconds,
            elapsed_seconds=elapsed_seconds,
            total_requests=self.total_requests,
            success_requests=self.success_requests,
            error_requests=self.error_requests,
            rate_limited_requests=self.rate_limited_requests,
            status_counts=dict(sorted(self.status_counts.items())),
            exception_counts=dict(sorted(self.exception_counts.items())),
            latency_ms=list(self.latency_ms),
        )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ordered[int(k)]
    d0 = ordered[f] * (c - k)
    d1 = ordered[c] * (k - f)
    return d0 + d1


def parse_steps(raw: str) -> list[int]:
    steps = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("Concurrency values must be positive integers")
        steps.append(value)
    if not steps:
        raise ValueError("At least one concurrency step is required")
    return steps


def build_payload(query: str) -> dict[str, Any]:
    return {
        "query": query,
        "stream": False,
    }


def _random_suffix(length: int = 10) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


async def wait_for_server_ready(base_url: str, health_endpoint: str, timeout_seconds: float) -> None:
    timeout = httpx.Timeout(5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        deadline = time.perf_counter() + timeout_seconds
        url = f"{base_url.rstrip('/')}{health_endpoint}"
        while time.perf_counter() < deadline:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    body = resp.json()
                    if body.get("pipeline_ready", True):
                        return
            except Exception:
                pass
            await asyncio.sleep(0.5)
    raise TimeoutError(
        f"Server did not become ready at {health_endpoint} within {timeout_seconds:.1f}s"
    )


async def worker(
    worker_id: int,
    client: httpx.AsyncClient,
    url: str,
    query: str,
    deadline: float,
    accumulator: StepAccumulator,
    stop_event: asyncio.Event,
    cache_bust: bool,
) -> None:
    while not stop_event.is_set():
        if time.perf_counter() >= deadline:
            return
        start = time.perf_counter()
        try:
            request_query = query
            if cache_bust:
                request_query = f"{query} [bench:{worker_id}-{_random_suffix()}]"
            payload = build_payload(request_query)
            resp = await client.post(url, json=payload)
            latency_ms = (time.perf_counter() - start) * 1000.0
            await accumulator.record_response(resp.status_code, latency_ms)
        except Exception as exc:  # noqa: BLE001 - benchmark tool should capture any failure class.
            await accumulator.record_exception(type(exc).__name__)


async def run_step(
    base_url: str,
    endpoint: str,
    concurrency: int,
    duration_seconds: int,
    query: str,
    timeout_seconds: float,
    cache_bust: bool,
) -> StepResult:
    url = f"{base_url.rstrip('/')}{endpoint}"
    accumulator = StepAccumulator()
    stop_event = asyncio.Event()

    timeout = httpx.Timeout(timeout_seconds)
    limits = httpx.Limits(max_connections=max(100, concurrency * 4), max_keepalive_connections=max(50, concurrency * 2))

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        start = time.perf_counter()
        deadline = start + duration_seconds
        tasks = [
            asyncio.create_task(
                worker(
                    worker_id=i,
                    client=client,
                    url=url,
                    query=query,
                    deadline=deadline,
                    accumulator=accumulator,
                    stop_event=stop_event,
                    cache_bust=cache_bust,
                )
            )
            for i in range(concurrency)
        ]

        try:
            await asyncio.gather(*tasks)
        finally:
            stop_event.set()
            await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.perf_counter() - start

    return accumulator.to_result(concurrency=concurrency, duration_seconds=duration_seconds, elapsed_seconds=elapsed)


def print_step_result(result: StepResult) -> None:
    print(
        " | ".join(
            [
                f"concurrency={result.concurrency}",
                f"elapsed={result.elapsed_seconds:.1f}s",
                f"success_rpm={result.success_req_per_min:.2f}",
                f"error_rate={result.error_rate_pct:.2f}%",
                f"p50={result.p50_latency_ms:.1f}ms",
                f"p95={result.p95_latency_ms:.1f}ms",
                f"rate_limit_429={result.rate_limited_requests}",
                f"total={result.total_requests}",
                f"success={result.success_requests}",
                f"errors={result.error_requests}",
            ]
        )
    )
    if result.status_counts:
        print(f"  status_counts={json.dumps(result.status_counts, sort_keys=True)}")
    if result.exception_counts:
        print(f"  exception_counts={json.dumps(result.exception_counts, sort_keys=True)}")


def build_summary(
    results: list[StepResult], baseline_req_min: float, target_multiplier: float
) -> dict[str, Any]:
    target_req_min = baseline_req_min * target_multiplier
    best = max(results, key=lambda r: r.success_req_per_min) if results else None

    summary = {
        "baseline_req_min": baseline_req_min,
        "target_multiplier": target_multiplier,
        "target_req_min": target_req_min,
        "pass": bool(best and best.success_req_per_min >= target_req_min),
        "steps": [
            {
                "concurrency": r.concurrency,
                "elapsed_seconds": round(r.elapsed_seconds, 3),
                "success_req_per_min": round(r.success_req_per_min, 3),
                "error_rate_pct": round(r.error_rate_pct, 3),
                "p50_latency_ms": round(r.p50_latency_ms, 3),
                "p95_latency_ms": round(r.p95_latency_ms, 3),
                "rate_limited_requests": r.rate_limited_requests,
                "total_requests": r.total_requests,
                "success_requests": r.success_requests,
                "error_requests": r.error_requests,
                "status_counts": r.status_counts,
                "exception_counts": r.exception_counts,
            }
            for r in results
        ],
        "best_step": None,
    }

    if best:
        summary["best_step"] = {
            "concurrency": best.concurrency,
            "success_req_per_min": round(best.success_req_per_min, 3),
            "error_rate_pct": round(best.error_rate_pct, 3),
            "p50_latency_ms": round(best.p50_latency_ms, 3),
            "p95_latency_ms": round(best.p95_latency_ms, 3),
            "rate_limited_requests": best.rate_limited_requests,
        }

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async stepped load test for /api/v1/chat")
    parser.add_argument("--base-url", required=True, help="Base URL, e.g. http://localhost:8000")
    parser.add_argument(
        "--endpoint",
        default="/api/v1/chat",
        help="API endpoint path (default: /api/v1/chat)",
    )
    parser.add_argument(
        "--health-endpoint",
        default="/api/v1/health",
        help="Health endpoint path used to wait for readiness before benchmark (default: /api/v1/health)",
    )
    parser.add_argument(
        "--concurrency-steps",
        default="1,2,4,8,16,32",
        help="Comma-separated concurrency levels (default: 1,2,4,8,16,32)",
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=120,
        help="Duration per concurrency step in seconds (default: 120)",
    )
    parser.add_argument(
        "--query",
        default="What is the treatment for myxedema coma?",
        help="Query text to send in chat payload",
    )
    parser.add_argument(
        "--cache-bust",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append random suffix to each query to avoid query-cache inflation (default: true)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=90.0,
        help="Request timeout in seconds (default: 90)",
    )
    parser.add_argument(
        "--baseline-req-min",
        type=float,
        default=28.9,
        help="Baseline successful requests/min for 2x target comparison",
    )
    parser.add_argument(
        "--target-multiplier",
        type=float,
        default=2.0,
        help="Multiplier against baseline req/min for pass/fail (default: 2.0)",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write machine-readable JSON summary",
    )
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    steps = parse_steps(args.concurrency_steps)

    stop_requested = asyncio.Event()

    def _request_stop() -> None:
        stop_requested.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            # Some environments may not support signal handlers.
            pass

    print("Benchmark configuration")
    print(f"  base_url={args.base_url}")
    print(f"  endpoint={args.endpoint}")
    print(f"  concurrency_steps={steps}")
    print(f"  duration_seconds={args.duration_seconds}")
    print(f"  timeout_seconds={args.timeout_seconds}")
    print(f"  cache_bust={args.cache_bust}")
    print(f"  baseline_req_min={args.baseline_req_min}")
    print(f"  target_multiplier={args.target_multiplier}")
    print("")

    await wait_for_server_ready(
        base_url=args.base_url,
        health_endpoint=args.health_endpoint,
        timeout_seconds=max(10.0, args.timeout_seconds),
    )
    print("Server readiness check passed.")
    print("")

    results: list[StepResult] = []

    for step in steps:
        if stop_requested.is_set():
            print("Stop requested; ending benchmark early.")
            break
        print(f"Running step: concurrency={step} ...")
        result = await run_step(
            base_url=args.base_url,
            endpoint=args.endpoint,
            concurrency=step,
            duration_seconds=args.duration_seconds,
            query=args.query,
            timeout_seconds=args.timeout_seconds,
            cache_bust=args.cache_bust,
        )
        print_step_result(result)
        print("")
        results.append(result)

    if not results:
        print("No completed benchmark steps.")
        return 1

    summary = build_summary(
        results=results,
        baseline_req_min=args.baseline_req_min,
        target_multiplier=args.target_multiplier,
    )

    print("Summary")
    print(f"  baseline_req_min={summary['baseline_req_min']:.3f}")
    print(f"  target_req_min={summary['target_req_min']:.3f}")
    if summary["best_step"]:
        best = summary["best_step"]
        print(
            "  best_step="
            f"concurrency={best['concurrency']}, "
            f"success_req_min={best['success_req_per_min']:.3f}, "
            f"error_rate={best['error_rate_pct']:.3f}%, "
            f"p50={best['p50_latency_ms']:.1f}ms, "
            f"p95={best['p95_latency_ms']:.1f}ms, "
            f"429s={best['rate_limited_requests']}"
        )
    print(f"  pass={summary['pass']} (target: >= 2x baseline unless --target-multiplier is changed)")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        print(f"  wrote_json={args.output_json}")

    return 0 if summary["pass"] else 2


def main() -> None:
    args = parse_args()
    try:
        exit_code = asyncio.run(main_async(args))
    except KeyboardInterrupt:
        exit_code = 130
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
