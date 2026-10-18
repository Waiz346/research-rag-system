"""Tests for Everyday RAG improvements: Universal Parser, Deduplication, Conversational Rewriting, Streaming, Studio Tools, and Obsidian/BibTeX Export."""
import io
import json
import zipfile
from fastapi.testclient import TestClient

from src.api.main import app
from src.ingestion.universal_parser import (
    compute_deterministic_doc_id,
    parse_any_file,
    parse_docx_bytes,
    parse_pptx_bytes,
    parse_xlsx_bytes,
    parse_csv_bytes,
    parse_ipynb_bytes,
)
from src.ingestion.chunker import DocumentChunker
from src.rag.generator import LLMGenerator

client = TestClient(app)


def _build_minimal_docx() -> bytes:
    """Create a valid in-memory .docx ZIP archive with heading, paragraph, and table."""
    buf = io.BytesIO()
    xml_content = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p>
          <w:pPr><w:pStyle w:val="Heading1"/></w:pPr>
          <w:r><w:t>Syllabus &amp; Grading Policy</w:t></w:r>
        </w:p>
        <w:p>
          <w:r><w:t>The midterm exam is worth 30% and the final project is due on December 12.</w:t></w:r>
        </w:p>
        <w:tbl>
          <w:tr>
            <w:tc><w:p><w:r><w:t>Component</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>Weight</w:t></w:r></w:p></w:tc>
          </w:tr>
          <w:tr>
            <w:tc><w:p><w:r><w:t>Midterm</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>30%</w:t></w:r></w:p></w:tc>
          </w:tr>
        </w:tbl>
      </w:body>
    </w:document>"""
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", xml_content)
    return buf.getvalue()


def _build_minimal_pptx() -> bytes:
    """Create a valid in-memory .pptx ZIP archive with 2 slides and speaker notes."""
    buf = io.BytesIO()
    slide1 = """<?xml version="1.0" encoding="UTF-8"?>
    <p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
           xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
      <p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Lecture 4: Transformer Attention</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld>
    </p:sld>"""
    note1 = """<?xml version="1.0" encoding="UTF-8"?>
    <p:notes xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
      <p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Emphasize scaled dot-product scaling factor 1/sqrt(d_k) for exam.</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld>
    </p:notes>"""
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("ppt/slides/slide1.xml", slide1)
        zf.writestr("ppt/notesSlides/notesSlide1.xml", note1)
    return buf.getvalue()


def _build_minimal_xlsx() -> bytes:
    """Create a valid in-memory .xlsx ZIP archive with sharedStrings and sheet1."""
    buf = io.BytesIO()
    shared = """<?xml version="1.0" encoding="UTF-8"?>
    <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
      <si><t>Model</t></si>
      <si><t>Accuracy</t></si>
      <si><t>HybridRAG</t></si>
    </sst>"""
    sheet1 = """<?xml version="1.0" encoding="UTF-8"?>
    <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
      <sheetData>
        <row r="1">
          <c r="A1" t="s"><v>0</v></c>
          <c r="B1" t="s"><v>1</v></c>
        </row>
        <row r="2">
          <c r="A2" t="s"><v>2</v></c>
          <c r="B2"><v>94.5</v></c>
        </row>
      </sheetData>
    </worksheet>"""
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("xl/sharedStrings.xml", shared)
        zf.writestr("xl/worksheets/sheet1.xml", sheet1)
    return buf.getvalue()


def test_universal_office_and_data_parsers():
    # 1. DOCX parsing
    docx_text = parse_docx_bytes(_build_minimal_docx())
    assert "Syllabus & Grading Policy" in docx_text
    assert "December 12" in docx_text
    assert "| Midterm | 30% |" in docx_text

    # 2. PPTX parsing + chunk location badge extraction
    pptx_text = parse_pptx_bytes(_build_minimal_pptx())
    assert "[Slide 1]" in pptx_text
    assert "Transformer Attention" in pptx_text
    assert "Speaker Notes:" in pptx_text

    chunker = DocumentChunker()
    chunks = chunker.split_text(pptx_text, base_metadata={"doc_id": "test_pptx", "title": "Lecture 4"})
    assert len(chunks) >= 1
    assert chunks[0].metadata.get("location_label") == "Slide 1"
    assert chunks[0].metadata.get("page_number") == 1

    # 3. XLSX parsing
    xlsx_text = parse_xlsx_bytes(_build_minimal_xlsx())
    assert "HybridRAG" in xlsx_text
    assert "94.5" in xlsx_text

    # 4. CSV parsing
    csv_bytes = b"Paper,Year,Score\nAttention Is All You Need,2017,0.98\nRAG Lewis,2020,0.95\n"
    csv_text = parse_csv_bytes(csv_bytes)
    assert "| Paper | Year | Score |" in csv_text
    assert "Attention Is All You Need" in csv_text

    # 5. Jupyter Notebook (.ipynb) parsing
    nb_dict = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Experiment 1: Retrieval Latency\n", "Testing HNSW index speed."]},
            {"cell_type": "code", "source": ["latency_ms = 18.4\n", "print(latency_ms)"], "outputs": [{"text": ["18.4\n"]}]}
        ]
    }
    ipynb_text = parse_ipynb_bytes(json.dumps(nb_dict).encode("utf-8"))
    assert "Experiment 1: Retrieval Latency" in ipynb_text
    assert "latency_ms = 18.4" in ipynb_text


def test_upload_docx_and_deterministic_deduplication():
    docx_bytes = _build_minimal_docx()
    res1 = client.post(
        "/documents/upload",
        files={"file": ("cs4000_syllabus.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        data={"workspace_id": "default"}
    )
    assert res1.status_code == 200
    d1 = res1.json()
    assert d1["status"] == "success"
    doc_id_1 = d1["doc_id"]

    # Re-uploading the exact same file should produce the exact same deterministic doc_id
    res2 = client.post(
        "/documents/upload",
        files={"file": ("cs4000_syllabus.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        data={"workspace_id": "default"}
    )
    assert res2.status_code == 200
    d2 = res2.json()
    assert d2["doc_id"] == doc_id_1

    # Cleanup
    client.delete(f"/documents/{doc_id_1}")


def test_quick_note_studio_tools_and_obsidian_export():
    # 1. Ingest a quick note
    note_res = client.post(
        "/documents/ingest-note",
        json={
            "title": "CS4000 Midterm Review Notes",
            "content": (
                "1. Introduction\n"
                "The CS4000 Midterm Exam is scheduled for October 28 in Room 302.\n\n"
                "2. Methodology\n"
                "Students must understand Reciprocal Rank Fusion (RRF), BM25 tokenization, and cosine similarity."
            ),
            "workspace_id": "default"
        }
    )
    assert note_res.status_code == 200
    note_data = note_res.json()
    doc_id = note_data["doc_id"]

    # 2. Test conversational query rewriting & intent classification via /query
    q_res = client.post(
        "/query",
        json={
            "question": "What topics are covered on it?",
            "top_k": 3,
            "doc_ids": [doc_id],
            "score_threshold": 0.05,
            "history": [
                {"role": "user", "content": "When is the CS4000 Midterm Exam?"},
                {"role": "assistant", "content": "The CS4000 Midterm Exam is scheduled for October 28 in Room 302."}
            ]
        }
    )
    assert q_res.status_code == 200
    q_data = q_res.json()
    assert q_data["rewritten_query"] is not None
    assert "CS4000" in q_data["rewritten_query"] or "Midterm" in q_data["rewritten_query"]
    assert q_data["query_intent"] in ["factual", "coursework_factual", "synthesis", "general"]

    # 3. Test SSE streaming endpoint (/query/stream)
    stream_res = client.post(
        "/query/stream",
        json={
            "question": "When is the CS4000 Midterm Exam?",
            "top_k": 2,
            "doc_ids": [doc_id],
            "score_threshold": 0.05
        }
    )
    assert stream_res.status_code == 200
    assert "event: metadata" in stream_res.text
    assert "event: done" in stream_res.text

    # 4. Test 1-Click Studio artifact generation (/studio/generate)
    for tool in ["study_guide", "quiz", "comparison_matrix", "action_items", "executive_briefing"]:
        st_res = client.post(
            "/studio/generate",
            json={
                "tool_type": tool,
                "workspace_id": "default",
                "doc_ids": [doc_id]
            }
        )
        assert st_res.status_code == 200
        st_data = st_res.json()
        assert st_data["tool_type"] == tool
        assert len(st_data["content"]) > 40

    # 5. Test Obsidian Markdown + BibTeX export (/export/markdown)
    exp_res = client.post(
        "/export/markdown",
        json={
            "title": "Midterm Study Synthesis",
            "content": "The exam covers **BM25** and **Reciprocal Rank Fusion**.",
            "workspace_name": "CS4000 Workspace",
            "include_bibtex": True,
            "sources": [
                {
                    "title": "CS4000 Midterm Review Notes",
                    "paper_id": doc_id,
                    "section": "Methodology",
                    "snippet": "Students must understand Reciprocal Rank Fusion (RRF).",
                    "metadata": {"authors": "Quick Note", "published": "2026-10-17"}
                }
            ]
        }
    )
    assert exp_res.status_code == 200
    exp_data = exp_res.json()
    assert "---" in exp_data["markdown"]
    assert "[[CS4000 Midterm Review Notes]]" in exp_data["markdown"]
    assert "@article{" in exp_data["bibtex"]

    # 6. Test Deduplicate endpoint
    dedup_res = client.post("/documents/deduplicate")
    assert dedup_res.status_code == 200
    assert "duplicates_removed" in dedup_res.json()

    # Cleanup
    client.delete(f"/documents/{doc_id}")
