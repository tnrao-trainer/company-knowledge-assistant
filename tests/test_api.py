"""
Unit & Integration tests for the Company Knowledge Assistant API (Ollama edition).

- Basic tests always run
- Full RAG test automatically skips if the knowledge base is not ready
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "docs" in data
    assert data.get("llm") == "Ollama (local)"


def test_health(client):
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert data.get("llm_provider") == "ollama"


def test_ask_empty_question(client):
    """Empty question should return 400 (input validation)."""
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_ask_missing_question(client):
    """Missing 'question' field should return 422 (validation error)."""
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_supported_question(client):
    """Debug version – shows the real reason the knowledge base is not ready."""
    
    # 1. Check health from the TestClient
    health_response = client.get("/health")
    health_data = health_response.json()
    print("\n" + "="*60)
    print("HEALTH CHECK FROM TEST CLIENT:")
    print(health_data)
    print("="*60)

    # 2. Always call /ask so we can see the real error message
    response = client.post(
        "/ask",
        json={"question": "How many days of annual leave do employees get?"}
    )
    
    print("\n/ask STATUS CODE:", response.status_code)
    print("/ask RESPONSE BODY:")
    print(response.text)
    print("="*60 + "\n")

    # 3. Fail the test with a clear message
    if response.status_code != 200:
        pytest.fail(
            f"Knowledge base not ready inside TestClient.\n"
            f"Status: {response.status_code}\n"
            f"Detail: {response.text}"
        )

    # If we reach here, it worked
    data = response.json()
    assert data["is_supported"] is True
    assert len(data["sources"]) > 0
    print("✅ Test passed successfully!")