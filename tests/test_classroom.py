"""Unit and integration tests for authentic Google Classroom connector and endpoints."""
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.ingestion.classroom_connector import GoogleClassroomConnector, get_classroom_connector

client = TestClient(app)


def test_classroom_status_endpoint():
    """Verify Classroom status endpoint returns consistent schema."""
    res = client.get("/classroom/status")
    assert res.status_code == 200
    data = res.json()
    assert "connected" in data
    assert "valid" in data
    assert "courses_count" in data
    assert "latency_ms" in data
    assert "auth_type" in data
    assert "message" in data


def test_classroom_oauth_config():
    """Verify custom OAuth client credentials can be configured."""
    payload = {
        "client_id": "mock_client_id_placeholder",
        "client_secret": "mock_client_secret_placeholder"
    }
    res = client.post("/classroom/oauth-config", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"


def test_classroom_connect_invalid_token():
    """Verify connecting with an invalid or expired access token fails gracefully."""
    payload = {
        "auth_type": "access_token",
        "access_token": "invalid_fake_token_12345"
    }
    res = client.post("/classroom/connect", json=payload)
    # Either 400 bad request or status with valid=False
    if res.status_code == 200:
        data = res.json()
        assert data["valid"] is False
    else:
        assert res.status_code in (400, 401, 403, 500)


def test_classroom_accounts_listing():
    """Verify saved accounts listing endpoint returns account info with scope flags."""
    res = client.get("/classroom/accounts")
    assert res.status_code == 200
    data = res.json()
    assert "accounts" in data
    assert "count" in data
    assert isinstance(data["accounts"], list)
    for acc in data["accounts"]:
        assert "email" in acc
        assert "has_classroom_scopes" in acc


def test_classroom_list_courses_mocked():
    """Verify listing courses when authenticated returns parsed course models."""
    mock_courses = [
        {
            "id": "course_101",
            "name": "Machine Learning Research",
            "section": "Section A",
            "description_heading": "Deep Learning & RAG",
            "course_state": "ACTIVE",
            "alternate_link": "https://classroom.google.com/c/course_101",
            "creation_time": "2026-01-15T00:00:00Z",
            "update_time": "2026-03-01T00:00:00Z"
        }
    ]

    with patch("src.api.main.get_classroom_connector") as mock_get_conn:
        mock_conn = MagicMock()
        mock_conn.list_courses.return_value = mock_courses
        mock_get_conn.return_value = mock_conn

        res = client.get("/classroom/courses")
        assert res.status_code == 200
        data = res.json()
        assert data["total_courses"] == 1
        assert len(data["courses"]) == 1
        assert data["courses"][0]["id"] == "course_101"
        assert data["courses"][0]["name"] == "Machine Learning Research"


def test_classroom_course_stream_mocked():
    """Verify retrieving course stream formats announcements, coursework materials, and coursework."""
    mock_stream = {
        "course_id": "course_101",
        "course_name": "Machine Learning Research",
        "announcements": [
            {
                "id": "ann_1",
                "course_id": "course_101",
                "text": "Assignment 1 paper references uploaded.",
                "created_at": "2026-02-01T10:00:00Z",
                "alternate_link": "https://classroom.google.com/c/course_101/p/ann_1",
                "materials": [
                    {
                        "type": "drive_file",
                        "id": "file_doc_1",
                        "title": "Retrieval Augmented Generation Survey.pdf",
                        "alternate_link": "https://drive.google.com/file/d/file_doc_1",
                        "mime_type": "application/pdf",
                        "thumbnail_url": None
                    }
                ]
            }
        ],
        "coursework_materials": [
            {
                "id": "mat_1",
                "course_id": "course_101",
                "title": "Week 1 Lecture Slides",
                "description": "Introduction to Vector Databases and Embeddings",
                "created_at": "2026-01-20T08:00:00Z",
                "alternate_link": "https://classroom.google.com/c/course_101/m/mat_1",
                "materials": []
            }
        ],
        "coursework": [
            {
                "id": "cw_1",
                "course_id": "course_101",
                "title": "Milestone Report 1",
                "description": "Submit initial literature review",
                "state": "PUBLISHED",
                "due_date": "2026-04-01",
                "materials": []
            }
        ],
        "total_items": 3
    }

    with patch("src.api.main.get_classroom_connector") as mock_get_conn:
        mock_conn = MagicMock()
        mock_conn.get_course_stream.return_value = mock_stream
        mock_get_conn.return_value = mock_conn

        res = client.get("/classroom/courses/course_101/stream")
        assert res.status_code == 200
        data = res.json()
        assert data["course_id"] == "course_101"
        assert data["total_items"] == 3
        assert len(data["announcements"]) == 1
        assert data["announcements"][0]["materials"][0]["title"] == "Retrieval Augmented Generation Survey.pdf"
        assert len(data["coursework_materials"]) == 1
        assert len(data["coursework"]) == 1


def test_classroom_import_announcement_text_into_rag():
    """Verify importing notice body text directly into RAG as course memo."""
    payload = {
        "course_id": "course_real_101",
        "course_name": "CS4000 Advanced AI",
        "item_type": "announcement_text",
        "title": "CS4000 Midterm Announcement",
        "raw_text": "Midterm test will cover dense passage retrieval, hybrid search with BM25, and reciprocal rank fusion.",
        "workspace_id": "default"
    }
    res = client.post("/classroom/import", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "doc_id" in data
    assert data["chunks_indexed"] >= 1
    client.delete(f"/documents/{data['doc_id']}")


def test_classroom_batch_import_mocked():
    """Verify batch importing multiple items."""
    with patch("src.api.main.get_classroom_connector") as mock_get_conn:
        mock_conn = MagicMock()
        mock_conn.download_document_content.return_value = (
            "lecture_01.pdf",
            "Lecture 1: Introduction to Dense Retrieval and Bi-Encoder architectures.",
            "application/pdf"
        )
        mock_get_conn.return_value = mock_conn

        payload = {
            "items": [
                {
                    "item_type": "drive_file",
                    "course_id": "course_101",
                    "course_name": "ML Research",
                    "drive_file_id": "drive_file_test_99",
                    "title": "Lecture 01 Notes.pdf",
                    "mime_type": "application/pdf"
                },
                {
                    "item_type": "announcement_text",
                    "course_id": "course_101",
                    "course_name": "ML Research",
                    "title": "Project Guideline Notice",
                    "raw_text": "Please cite all ArXiv papers properly using APA style format."
                }
            ],
            "workspace_id": "default"
        }
        res = client.post("/classroom/import/batch", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["total_requested"] == 2
        assert data["total_succeeded"] == 2
        assert len(data["imported_docs"]) == 2
        for d in data["imported_docs"]:
            client.delete(f"/documents/{d['doc_id']}")

