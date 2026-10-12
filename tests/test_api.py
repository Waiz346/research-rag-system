"""Integration tests for FastAPI endpoints."""
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "indexed_documents_count" in data

def test_query_endpoint():
    response = client.post(
        "/query",
        json={"question": "What is self-attention?", "top_k": 2}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "latency_ms" in data
    assert "sources" in data

def test_evaluate_endpoint():
    response = client.post(
        "/evaluate",
        json={"question": "Explain neural retrieval models.", "top_k": 2}
    )
    assert response.status_code == 200
    data = response.json()
    assert "context_relevance" in data
    assert "faithfulness" in data
    assert "composite_score" in data
