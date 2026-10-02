"""Process-local, thread-safe pipeline timing (contract C7, workstream W0).

``timing.stage(name)`` wraps a pipeline stage; ``timing.record`` adds a
labeled duration sample; ``timing.count`` increments an event counter;
``timing.inflight`` tracks peak concurrency; ``timing.write_report`` dumps
``reports/timings_<utc>.json`` (called at the end of ``run-all``).

Everything is in memory and observation-only: ``VP_TIMINGS=0`` turns all
recording into a cheap no-op, and none of these hooks alter pipeline
behavior.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from . import config

_lock = threading.Lock()
# stage name -> {"wall_seconds": float, "calls": int}
_stages: dict[str, dict] = {}
# (kind, sorted label pairs) -> duration samples in seconds
_samples: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = {}
# "kind|k=v,k=v" -> event count
_counts: dict[str, int] = {}
_inflight: dict[str, int] = {}
_inflight_peak: dict[str, int] = {}


def enabled() -> bool:
    """VP_TIMINGS=0 disables recording. Read per call so tests can flip it."""
    raw = os.getenv("VP_TIMINGS")
    if raw is None:
        return bool(config.VP_TIMINGS)
    return raw.strip().lower() not in {"0", "false", "no", "off"}


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Record the wall time of one invocation of a pipeline stage."""
    if not enabled():
        yield
        return
    start = time.monotonic()
    try:
        yield
    finally:
        wall = time.monotonic() - start
        with _lock:
            entry = _stages.setdefault(name, {"wall_seconds": 0.0, "calls": 0})
            entry["wall_seconds"] += wall
            entry["calls"] += 1


def record(kind: str, seconds: float, **labels: object) -> None:
    """Add one duration sample (seconds) under a kind + label tuple."""
    if not enabled():
        return
    key = (kind, tuple(sorted((str(k), str(v)) for k, v in labels.items())))
    with _lock:
        _samples.setdefault(key, []).append(float(seconds))


def count(kind: str, n: int = 1, **labels: object) -> None:
    """Increment an event counter (e.g. http_429, llm_timeout)."""
    if not enabled():
        return
    suffix = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    key = f"{kind}|{suffix}" if suffix else kind
    with _lock:
        _counts[key] = _counts.get(key, 0) + n


@contextmanager
def inflight(name: str = "llm") -> Iterator[None]:
    """Track peak concurrent calls of a kind (e.g. live LLM requests)."""
    if not enabled():
        yield
        return
    with _lock:
        _inflight[name] = _inflight.get(name, 0) + 1
        _inflight_peak[name] = max(_inflight_peak.get(name, 0), _inflight[name])
    try:
        yield
    finally:
        with _lock:
            _inflight[name] -= 1


def _percentile(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = q / 100.0 * (len(sorted_vals) - 1)
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (rank - lo)


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"count": 0, "total_seconds": 0.0}
    return {
        "count": len(vals),
        "total_seconds": round(sum(vals), 6),
        "mean_seconds": round(sum(vals) / len(vals), 6),
        "p50_seconds": _percentile(vals, 50),
        "p95_seconds": _percentile(vals, 95),
        "max_seconds": max(vals),
    }


def report() -> dict:
    """Assemble the timings report dict."""
    with _lock:
        samples = {k: sorted(v) for k, v in _samples.items()}
        stages = {k: dict(v) for k, v in _stages.items()}
        counts = dict(_counts)
        peaks = dict(_inflight_peak)

    limiter_wait: dict[str, dict] = {}
    http_fetch: dict[str, dict] = {}
    llm_latency: dict[str, dict] = {}
    other: dict[str, dict] = {}
    for (kind, labels), vals in samples.items():
        label_map = dict(labels)
        if kind == "limiter_wait":
            limiter_wait[label_map.get("host", "?")] = _stats(vals)
        elif kind == "http_fetch":
            http_fetch[label_map.get("host", "?")] = _stats(vals)
        elif kind == "llm_latency":
            stage_name = label_map.get("stage", "?")
            source = label_map.get("source", "?")
            llm_latency.setdefault(stage_name, {})[source] = _stats(vals)
        else:
            suffix = " ".join(f"{k}={v}" for k, v in labels)
            other[f"{kind} {suffix}" if suffix else kind] = _stats(vals)

    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "enabled": enabled(),
        "stages": stages,
        "limiter_wait_seconds_by_host": limiter_wait,
        "http_fetch_seconds_by_host": http_fetch,
        "llm_latency_seconds_by_stage": llm_latency,
        "peak_in_flight": peaks,
        "counts": counts,
        "other_samples": other,
    }


def write_report(path: str | Path) -> Path:
    """Write the timings report JSON. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report(), indent=1, sort_keys=True) + "\n")
    return path
