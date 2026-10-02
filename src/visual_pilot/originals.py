"""Contract C5: in-memory judge -> store handoff of original figure bytes.

The vision judge holds each fetched image in memory while it is judged
(``judge.fetch_and_prepare``). A figure that ends ``vision_accepted`` would
otherwise be refetched from S3 by the store stage; this module is a
process-local, thread-safe cache so the judge can hand the *original* bytes
(never the downscaled LLM copy) over instead.

Bounded by ``config.VP_ORIGINALS_CACHE_MB`` with LRU eviction on total byte
size. Nothing is ever written to disk. Entries are consumed
by ``take`` so memory does not outlive the store stage.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

from . import config

_lock = threading.Lock()
# figure_id -> (sha256 recorded at put time, original bytes)
_entries: OrderedDict[str, tuple[str, bytes]] = OrderedDict()
_total_bytes = 0


def _cap_bytes() -> int:
    return max(0, config.VP_ORIGINALS_CACHE_MB) * 1024 * 1024


def put(figure_id: str, sha256: str, data: bytes) -> None:
    """Cache ``data`` for ``figure_id``; evict least-recently-used on overflow.

    An item larger than the whole budget is dropped immediately (it could
    never be served). Re-``put``-ing an id replaces the entry and refreshes
    its recency.
    """
    global _total_bytes
    if not figure_id:
        return
    cap = _cap_bytes()
    if len(data) > cap:
        return
    with _lock:
        old = _entries.pop(figure_id, None)
        if old is not None:
            _total_bytes -= len(old[1])
        _entries[figure_id] = (sha256, data)
        _total_bytes += len(data)
        while _total_bytes > cap and _entries:
            _, (_, evicted) = _entries.popitem(last=False)
            _total_bytes -= len(evicted)


def take(figure_id: str, expected_sha256: str) -> bytes | None:
    """Remove the entry and return its bytes when ``sha256(data)`` matches.

    The entry is consumed whether or not the hash matches — a mismatch means
    the wrong bytes were held and retrying ``take`` cannot help. ``None`` on
    a miss or mismatch; callers fall back to ``pmc.fetch_image_bytes``.
    """
    global _total_bytes
    with _lock:
        entry = _entries.pop(figure_id, None)
        if entry is None:
            return None
        _total_bytes -= len(entry[1])
    if hashlib.sha256(entry[1]).hexdigest() != expected_sha256:
        return None
    return entry[1]


def discard(figure_id: str) -> None:
    """Drop the entry for ``figure_id`` if present (e.g. store errored out)."""
    global _total_bytes
    with _lock:
        entry = _entries.pop(figure_id, None)
        if entry is not None:
            _total_bytes -= len(entry[1])


def clear() -> None:
    """Drop every entry (tests, end of a pipeline run)."""
    global _total_bytes
    with _lock:
        _entries.clear()
        _total_bytes = 0


def _stats() -> tuple[int, int]:
    """(entry_count, total_bytes) — observability/tests."""
    with _lock:
        return len(_entries), _total_bytes
