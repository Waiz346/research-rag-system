"""Unit and integration tests for Google Drive connector and API endpoints."""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.ingestion.gdrive_connector import get_gdrive_connector, GoogleDriveConnector

client = TestClient(app)

def test_gdrive_connector_demo_mode():
    conn = GoogleDriveConnector()
    res = conn.connect(auth_type="demo")
    assert res["status"] == "success"
    assert res["auth_type"] == "demo"

    status = conn.get_status()
    assert status["connected"] is True
    assert status["auth_type"] == "demo"

    # Browse root
    root_data = conn.list_folder_contents(folder_id="root")
    assert root_data["current_folder_id"] == "root"
    assert len(root_data["folders"]) >= 3
    folder_names = [f["name"] for f in root_data["folders"]]
    assert any("Deep Learning" in n for n in folder_names)
    assert any("RAG" in n for n in folder_names)

    # Browse subfolder
    dl_data = conn.list_folder_contents(folder_id="folder_dl")
    assert dl_data["current_folder_id"] == "folder_dl"
    assert len(dl_data["files"]) >= 2
    file_names = [f["name"] for f in dl_data["files"]]
    assert any("Attention Is All You Need" in n for n in file_names)

    # Download content
    name, content, mime = conn.download_file_content("demo_attention")
    assert "Attention Is All You Need" in name
    assert "Transformer" in content
    assert "Abstract" in content

def test_gdrive_api_endpoints_flow():
    # 1. Status endpoint
    s_res = client.get("/gdrive/status")
    assert s_res.status_code == 200
    s_data = s_res.json()
    assert "connected" in s_data
    assert "auth_type" in s_data

    # 2. Connect endpoint (Demo)
    c_res = client.post("/gdrive/connect", json={"auth_type": "demo"})
    assert c_res.status_code == 200
    assert c_res.json()["connected"] is True

    # 3. Browse root folder
    b_res = client.get("/gdrive/browse?folder_id=root")
    assert b_res.status_code == 200
    b_data = b_res.json()
    assert b_data["current_folder_id"] == "root"
    assert len(b_data["folders"]) >= 3

    # 4. Browse RAG folder
    rag_res = client.get("/gdrive/browse?folder_id=folder_rag")
    assert rag_res.status_code == 200
    rag_data = rag_res.json()
    assert len(rag_data["files"]) >= 3
    lewis_file = next(f for f in rag_data["files"] if f["id"] == "demo_rag_lewis")
    assert "Retrieval-Augmented Generation" in lewis_file["name"]

    # 5. Import and index a paper from Google Drive
    imp_res = client.post(
        "/gdrive/import",
        json={
            "file_id": "demo_rag_lewis",
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive Tasks",
            "section_aware": True
        }
    )
    assert imp_res.status_code == 200
    imp_data = imp_res.json()
    assert imp_data["status"] == "success"
    doc_id = imp_data["doc_id"]
    assert imp_data["chunks_indexed"] > 0
    assert "Abstract" in imp_data["sections"]

    # 6. Verify paper in Document Catalog
    doc_res = client.get(f"/documents/{doc_id}")
    assert doc_res.status_code == 200
    doc_detail = doc_res.json()
    assert doc_detail["summary"]["source_type"] == "gdrive"
    assert "parametric" in doc_detail["full_text"]

    # 7. Ask question scoped to this Google Drive paper
    q_res = client.post(
        "/query",
        json={
            "question": "What is the difference between parametric and non-parametric memory?",
            "top_k": 2,
            "doc_ids": [doc_id],
            "score_threshold": 0.1
        }
    )
    assert q_res.status_code == 200
    q_data = q_res.json()
    assert q_data["total_retrieved"] > 0
    assert any(doc_id in c["paper_id"] for c in q_data["sources"])

    # 8. Clean up test document
    del_res = client.delete(f"/documents/{doc_id}")
    assert del_res.status_code == 200

def test_gdrive_accounts_endpoints():
    # 1. Test listing accounts
    acc_res = client.get("/gdrive/accounts")
    assert acc_res.status_code == 200
    acc_data = acc_res.json()
    assert "accounts" in acc_data
    assert "count" in acc_data
    assert isinstance(acc_data["accounts"], list)

    # 2. Test login status
    st_res = client.get("/gdrive/accounts/login-status")
    assert st_res.status_code == 200
    st_data = st_res.json()
    assert "status" in st_data

    # 3. Test switching to nonexistent account
    sw_err = client.post("/gdrive/accounts/switch", json={"email": "nonexistent_user_xyz@gmail.com"})
    assert sw_err.status_code == 404

    # 4. If any accounts exist, test switching to first one
    if acc_data["accounts"]:
        target_email = acc_data["accounts"][0]["email"]
        sw_ok = client.post("/gdrive/accounts/switch", json={"email": target_email})
        assert sw_ok.status_code == 200
        assert sw_ok.json()["status"] == "success"

