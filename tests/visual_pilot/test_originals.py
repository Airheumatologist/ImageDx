"""W6 tests: contract C5 originals handoff (in-memory, thread-safe, LRU)."""

import hashlib
import threading

import pytest

from src.visual_pilot import config, originals


@pytest.fixture(autouse=True)
def _clean_originals():
    originals.clear()
    yield
    originals.clear()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_put_take_roundtrip_consumes_entry():
    data = b"image-bytes"
    originals.put("PMC1:F1", _sha(data), data)
    assert originals.take("PMC1:F1", _sha(data)) == data
    # take consumes the entry
    assert originals.take("PMC1:F1", _sha(data)) is None
    assert originals._stats() == (0, 0)


def test_take_miss_returns_none():
    assert originals.take("nope", "0" * 64) is None


def test_take_wrong_sha_returns_none_and_consumes():
    data = b"real-bytes"
    originals.put("PMC1:F1", _sha(data), data)
    assert originals.take("PMC1:F1", _sha(b"other")) is None
    # The entry is consumed even on mismatch: take can never serve it again.
    assert originals.take("PMC1:F1", _sha(data)) is None
    assert originals._stats() == (0, 0)


def test_put_replaces_same_id():
    originals.put("f", _sha(b"a"), b"a")
    originals.put("f", _sha(b"b"), b"b")
    assert originals.take("f", _sha(b"b")) == b"b"
    assert originals._stats() == (0, 0)


def test_lru_eviction_by_byte_budget(monkeypatch):
    monkeypatch.setattr(config, "VP_ORIGINALS_CACHE_MB", 1)
    third = 1024 * 1024 // 3
    blobs = {k: (k.encode() * third)[:third] for k in "abc"}
    for k, blob in blobs.items():
        originals.put(k, _sha(blob), blob)
    assert originals._stats()[0] == 3
    d = (b"d" * third)[:third]
    originals.put("d", _sha(d), d)  # pushes past the cap -> evicts 'a' (LRU)
    assert originals.take("a", _sha(blobs["a"])) is None
    for k in "bcd":
        blob = d if k == "d" else blobs[k]
        assert originals.take(k, _sha(blob)) == blob
    assert originals._stats() == (0, 0)


def test_lru_refresh_on_reput(monkeypatch):
    # Re-putting an id refreshes recency, so the *other* entry is evicted.
    monkeypatch.setattr(config, "VP_ORIGINALS_CACHE_MB", 1)
    half = 1024 * 1024 // 2 - 1
    a, b = b"a" * half, b"b" * half
    originals.put("a", _sha(a), a)
    originals.put("b", _sha(b), b)
    originals.put("a", _sha(a), a)  # refresh 'a'; 'b' is now the LRU entry
    c = b"c" * (1024 * 1024 // 3)
    originals.put("c", _sha(c), c)
    assert originals.take("b", _sha(b)) is None
    assert originals.take("a", _sha(a)) == a
    assert originals.take("c", _sha(c)) == c


def test_oversized_item_is_dropped(monkeypatch):
    monkeypatch.setattr(config, "VP_ORIGINALS_CACHE_MB", 0)
    data = b"x" * 100
    originals.put("f", _sha(data), data)
    assert originals.take("f", _sha(data)) is None
    assert originals._stats() == (0, 0)


def test_discard_and_clear():
    a = b"a"
    originals.put("a", _sha(a), a)
    originals.discard("a")
    assert originals.take("a", _sha(a)) is None
    originals.discard("missing")  # no-op
    originals.put("x", _sha(b"x"), b"x")
    originals.put("y", _sha(b"y"), b"y")
    originals.clear()
    assert originals._stats() == (0, 0)


def test_threadsafe_put_take_smoke():
    errors = []

    def worker(n):
        try:
            for i in range(50):
                fid = f"f{n}:{i}"
                blob = f"{n}-{i}".encode() * 10
                originals.put(fid, _sha(blob), blob)
                got = originals.take(fid, _sha(blob))
                if got is not None and got != blob:
                    errors.append(f"{fid}: wrong bytes")
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert originals._stats()[1] <= config.VP_ORIGINALS_CACHE_MB * 1024 * 1024
