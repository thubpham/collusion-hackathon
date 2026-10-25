import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(autouse=True)
def project_cwd(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr("src.agents.player.LLMClient", lambda: object(), raising=False)
