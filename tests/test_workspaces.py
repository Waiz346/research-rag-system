"""Unit tests for Workspace, Folder, and Multi-Conversation Storage."""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.storage.workspace_store import WorkspaceStore

client = TestClient(app)


@pytest.fixture
def temp_store(tmp_path):
    """Create an isolated WorkspaceStore instance using SQLite in a temp directory."""
    db_file = tmp_path / "test_workspaces.db"
    store = WorkspaceStore(db_url=f"sqlite:///{db_file}")
    return store


def test_default_workspace_seeded(temp_store):
    workspaces = temp_store.list_workspaces()
    assert len(workspaces) >= 1
    default_ws = temp_store.get_workspace("default")
    assert default_ws is not None
    assert default_ws["name"] == "General Research"
    assert len(default_ws["conversations"]) >= 1


def test_workspace_crud_and_conversations(temp_store):
    # 1. Create workspace
    ws = temp_store.create_workspace("Biomedical Research", "Papers on clinical RAG")
    assert ws["name"] == "Biomedical Research"
    assert ws["description"] == "Papers on clinical RAG"
    assert ws["id"].startswith("ws_")
    assert len(ws["conversations"]) == 1

    # 2. Add sub-conversations
    c2 = temp_store.create_conversation(ws["id"], "Clinical Trials Q&A")
    assert c2 is not None
    assert c2["title"] == "Clinical Trials Q&A"
    assert c2["workspace_id"] == ws["id"]

    c3 = temp_store.create_conversation(ws["id"], "Drug Repurposing Discussion")
    assert c3 is not None
    assert c3["title"] == "Drug Repurposing Discussion"

    # List conversations
    convs = temp_store.list_conversations(ws["id"])
    assert len(convs) == 3

    # 3. Add messages to conversation
    m1 = temp_store.add_message(
        c2["id"],
        role="user",
        content="What are the adverse effects identified in trial X?"
    )
    assert m1["role"] == "user"
    assert m1["content"].startswith("What are the")

    m2 = temp_store.add_message(
        c2["id"],
        role="assistant",
        content="Based on Table 3, the most common effect was mild fatigue.",
        sources=[{"title": "Trial X Study", "section": "Results", "score": 0.88}],
        confidence_score=0.91,
        is_low_confidence=False
    )
    assert m2["role"] == "assistant"
    assert len(m2["sources"]) == 1
    assert m2["sources"][0]["section"] == "Results"

    # Verify conversation message history
    c2_detail = temp_store.get_conversation(c2["id"])
    assert len(c2_detail["messages"]) == 2

    # 4. Clear conversation messages
    temp_store.clear_conversation_messages(c2["id"])
    c2_cleared = temp_store.get_conversation(c2["id"])
    assert len(c2_cleared["messages"]) == 0

    # 5. Delete conversation
    temp_store.delete_conversation(c3["id"])
    assert len(temp_store.list_conversations(ws["id"])) == 2

    # 6. Delete workspace
    deleted = temp_store.delete_workspace(ws["id"])
    assert deleted is True
    assert temp_store.get_workspace(ws["id"]) is None


def test_workspace_api_endpoints():
    # Test GET /workspaces
    res = client.get("/workspaces")
    assert res.status_code == 200
    data = res.json()
    assert "workspaces" in data
    assert data["count"] >= 1

    # Test POST /workspaces
    res_post = client.post("/workspaces", json={"name": "Quantum Computing", "description": "NISQ and error correction"})
    assert res_post.status_code == 200
    created_ws = res_post.json()
    ws_id = created_ws["id"]
    assert created_ws["name"] == "Quantum Computing"
    assert len(created_ws["conversations"]) >= 1

    # Test GET /workspaces/{id}
    res_get = client.get(f"/workspaces/{ws_id}")
    assert res_get.status_code == 200
    assert res_get.json()["id"] == ws_id

    # Test POST /workspaces/{id}/conversations
    res_conv = client.post(f"/workspaces/{ws_id}/conversations", json={"title": "Surface Codes Discussion"})
    assert res_conv.status_code == 200
    conv_id = res_conv.json()["id"]
    assert res_conv.json()["title"] == "Surface Codes Discussion"

    # Test POST /conversations/{id}/messages
    res_msg_u = client.post(
        f"/conversations/{conv_id}/messages",
        json={"role": "user", "content": "How many physical qubits are needed?"}
    )
    assert res_msg_u.status_code == 200

    res_msg_a = client.post(
        f"/conversations/{conv_id}/messages",
        json={
            "role": "assistant",
            "content": "Typically 1,000 to 10,000 per logical qubit.",
            "sources": [{"title": "Surface Code Review", "score": 0.95}],
            "confidence_score": 0.95
        }
    )
    assert res_msg_a.status_code == 200

    # Test GET /conversations/{id}
    res_conv_detail = client.get(f"/conversations/{conv_id}")
    assert res_conv_detail.status_code == 200
    conv_detail = res_conv_detail.json()
    assert len(conv_detail["messages"]) == 2

    # Test DELETE /conversations/{id}/messages
    res_clear = client.delete(f"/conversations/{conv_id}/messages")
    assert res_clear.status_code == 200
    res_conv_after_clear = client.get(f"/conversations/{conv_id}")
    assert len(res_conv_after_clear.json()["messages"]) == 0

    # Test DELETE /workspaces/{id}
    res_del_ws = client.delete(f"/workspaces/{ws_id}")
    assert res_del_ws.status_code == 200
    res_404 = client.get(f"/workspaces/{ws_id}")
    assert res_404.status_code == 404


def test_health_includes_database_engine():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "database_engine" in data
    assert "SQLite" in data["database_engine"] or "PostgreSQL" in data["database_engine"]


def test_workspace_document_storage(temp_store):
    ws = temp_store.create_workspace("Robotics Lab", "Robotics and embodied AI")
    ws_id = ws["id"]

    # Register document under workspace
    doc = temp_store.register_document(
        doc_id="robotics_paper_1",
        workspace_id=ws_id,
        title="Visual Motor Policies",
        full_text="This paper investigates imitation learning for robotic manipulation.",
        source_type="arxiv",
        filename="vmp.pdf",
        authors="Dr. Robot",
        citation_count=42,
        chunk_count=5,
        sections=["Abstract", "Methodology"]
    )
    assert doc["doc_id"] == "robotics_paper_1"
    assert doc["workspace_id"] == ws_id
    assert doc["title"] == "Visual Motor Policies"

    # List documents scoped to workspace
    docs = temp_store.list_documents(workspace_id=ws_id)
    assert len(docs) == 1
    assert docs[0]["doc_id"] == "robotics_paper_1"

    # Get document with full text
    doc_detail = temp_store.get_document("robotics_paper_1")
    assert doc_detail is not None
    assert "imitation learning" in doc_detail["full_text"]

    # Delete document
    deleted = temp_store.delete_document("robotics_paper_1")
    assert deleted is True
    assert temp_store.get_document("robotics_paper_1") is None


def test_document_api_workspace_scoping():
    # 1. Create workspace via API
    ws_res = client.post("/workspaces", json={"name": "Vision Transformer Study"})
    assert ws_res.status_code == 200
    ws_id = ws_res.json()["id"]

    # 2. Upload file scoped to this workspace
    file_bytes = b"Vision Transformers (ViT) apply self-attention to sequences of image patches."
    upload_res = client.post(
        "/documents/upload",
        files={"file": ("vit_survey.txt", file_bytes, "text/plain")},
        data={"workspace_id": ws_id}
    )
    assert upload_res.status_code == 200
    doc_id = upload_res.json()["doc_id"]

    # 3. List documents for this workspace
    docs_res = client.get(f"/documents?workspace_id={ws_id}")
    assert docs_res.status_code == 200
    docs = docs_res.json()
    assert any(d["doc_id"] == doc_id for d in docs)

    # 4. Get document detail
    detail_res = client.get(f"/documents/{doc_id}")
    assert detail_res.status_code == 200
    assert "Vision Transformers" in detail_res.json()["full_text"]

    # 5. Clean up
    client.delete(f"/documents/{doc_id}")
    client.delete(f"/workspaces/{ws_id}")

