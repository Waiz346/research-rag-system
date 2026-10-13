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

def test_dynamic_llm_config_switching():
    # Switch to Qwen
    res = client.post(
        "/config/llm",
        json={"provider": "qwen", "model_name": "qwen-plus", "api_key": "sk-test123456"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["provider"] == "qwen"
    assert data["model_name"] == "qwen-plus"
    assert data["has_api_key"] is True

    # Test connection on mock mode
    client.post("/config/llm", json={"provider": "mock"})
    test_res = client.post("/config/llm/test")
    assert test_res.status_code == 200
    test_data = test_res.json()
    assert test_data["success"] is True
    assert test_data["provider"] == "mock"

def test_query_hybrid_and_threshold_cutoffs():
    # 1. Query with hybrid mode and normal threshold
    res = client.post(
        "/query",
        json={
            "question": "What is transformer architecture?",
            "top_k": 2,
            "search_mode": "hybrid",
            "score_threshold": 0.05
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert data["search_mode"] == "hybrid"
    assert "filtered_out_count" in data

    # 2. Query with ultra-strict threshold cutoff to trigger safe rejection
    res_strict = client.post(
        "/query",
        json={
            "question": "Completely unrelated astrophysical query on dark energy",
            "top_k": 2,
            "search_mode": "hybrid",
            "score_threshold": 0.99
        }
    )
    assert res_strict.status_code == 200
    data_strict = res_strict.json()
    assert data_strict["total_retrieved"] == 0
    assert data_strict["is_low_confidence"] is True
    assert any(phrase in data_strict["answer"] for phrase in [
        "No retrieved paper passages met the required relevance threshold",
        "No relevant research papers found in the index"
    ])

def test_upload_and_document_catalog():
    # 1. Upload a text document
    file_content = (
        b"Title: Custom Deep Learning Notes\n"
        b"Abstract: This note discusses diffusion probabilistic models and continuous latent dynamics.\n\n"
        b"1. Introduction\n"
        b"Generative diffusion frameworks define forward Gaussian noise Markov chains.\n\n"
        b"3. Methodology\n"
        b"We compute score matching via U-Net denoisers with residual attention."
    )
    res = client.post(
        "/documents/upload",
        files={"file": ("custom_notes.txt", file_content, "text/plain")}
    )
    assert res.status_code == 200
    up_data = res.json()
    assert up_data["status"] == "success"
    doc_id = up_data["doc_id"]
    assert "Abstract" in up_data["sections"]

    # 2. List documents (lightweight manifest)
    list_res = client.get("/documents")
    assert list_res.status_code == 200
    docs = list_res.json()
    assert any(d["doc_id"] == doc_id for d in docs)

    # 3. Get full document content (lazy loaded)
    detail_res = client.get(f"/documents/{doc_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert "Generative diffusion frameworks" in detail["full_text"]

    # 4. Query specifically scoped to this uploaded document
    query_res = client.post(
        "/query",
        json={
            "question": "What is diffusion Markov chain?",
            "top_k": 2,
            "doc_ids": [doc_id],
            "score_threshold": 0.1
        }
    )
    assert query_res.status_code == 200
    q_data = query_res.json()
    assert q_data["total_retrieved"] > 0

    # 5. Cleanup
    del_res = client.delete(f"/documents/{doc_id}")
    assert del_res.status_code == 200
