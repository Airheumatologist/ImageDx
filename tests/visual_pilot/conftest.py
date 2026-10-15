import pytest

from src.visual_pilot import llm


@pytest.fixture(autouse=True)
def _unpaced_llm(monkeypatch):
    """Tests never wait on the provider's requests-per-minute pacing."""
    monkeypatch.setattr(llm, "_PACER", llm._RequestPacer(rpm=0))
