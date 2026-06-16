"""End-to-end smoke tests for the FastAPI app.

This is the Phase 0 'definition of done': the app boots and the provider
abstraction returns a completion locally with no model and no API key.
"""

from fastapi.testclient import TestClient

from eaip.app import _provider, create_app
from eaip.providers import Completion
from eaip.providers.stub import StubProvider


def test_health_endpoint_reports_stub_provider():
    client = TestClient(create_app())
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["provider"] == "stub"


def test_hello_endpoint_round_trips_through_default_stub():
    client = TestClient(create_app())
    resp = client.post("/hello", json={"message": "ping"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"] == "[stub] received: ping"
    assert body["provider"] == "stub"
    assert body["total_tokens"] > 0


def test_hello_endpoint_honors_dependency_override():
    """A scripted stub injected via FastAPI's override drives a deterministic reply."""
    app = create_app()
    scripted = StubProvider([Completion(text="canned answer", model="stub")])
    app.dependency_overrides[_provider] = lambda: scripted

    client = TestClient(app)
    resp = client.post("/hello", json={"message": "anything"})
    assert resp.status_code == 200
    assert resp.json()["reply"] == "canned answer"
