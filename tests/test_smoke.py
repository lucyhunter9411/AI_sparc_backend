"""
CI smoke-tests — run with  `pytest -q`

We stub the env-vars **before** importing anything that might transitively
import `main.py`, so app.core.config.Settings() sees valid values.
"""

import os

# ----- 1)  FORCE dummy environment variables -------------------------------
os.environ["MONGO_USER"] = "test"
os.environ["MONGO_PASSWORD"] = "test"
os.environ["DB_NAME"] = "testdb"                  # <- key that was missing
os.environ["AZURE_OPENAI_KEY"] = "dummy"
os.environ["AZURE_OPENAI_BASE"] = "https://dummy.endpoint"
os.environ["TESTING"] = "1"  

# ----- 2)  Make project root importable ------------------------------------
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# ----- 3)  Now it’s safe to import the app ---------------------------------
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_ok() -> None:
    """Health check endpoint responds 200 with expected JSON."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.parametrize("path", ["/ws/before/dummyrobot"])
def test_websocket_handshake(path: str) -> None:
    """WebSocket endpoint accepts a basic handshake."""
    with client.websocket_connect(path) as ws:
        ws.send_json({"type": "ping"})     # minimal keep-alive
