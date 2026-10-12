"""Unit tests for the DocumentChunker module."""
import pytest
from src.ingestion.chunker import DocumentChunker

def test_chunker_basic_split():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    sample_text = (
        "Retrieval Augmented Generation enhances language models with external knowledge. "
        "It provides grounded answers and minimizes hallucinations in deep neural architectures. "
        "This makes it an ideal solution for production enterprise use cases."
    )
    meta = {"doc_id": "test_doc_01", "title": "Intro to RAG"}
    chunks = chunker.split_text(sample_text, base_metadata=meta)

    assert len(chunks) > 0
    assert chunks[0].chunk_id.startswith("test_doc_01_chunk_")
    assert chunks[0].metadata["title"] == "Intro to RAG"
    assert "chunk_index" in chunks[0].metadata

def test_chunker_invalid_overlap():
    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=100, chunk_overlap=120)

def test_chunker_empty_string():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.split_text("", base_metadata={})
    assert chunks == []
