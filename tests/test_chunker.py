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

def test_section_aware_paper_chunking():
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=30)
    paper_text = (
        "Title: AR-RAG: Autoregressive Retrieval Augmentation\n"
        "Authors: Jingyuan Qi, Lifu Huang\n"
        "Abstract: We introduce Autoregressive Retrieval Augmentation for visual patch retrieval.\n\n"
        "1. Introduction\n"
        "Standard RAG models retrieve single static references before generation.\n\n"
        "3. Methodology\n"
        "Our framework operates over discrete autoregressive visual tokens and optimizes k-NN patches.\n\n"
        "4. Experiments\n"
        "We evaluate on ImageNet-1K and report substantial FID improvements.\n\n"
        "References\n"
        "[1] Vaswani et al. Attention is all you need. NeurIPS 2017.\n"
    )
    meta = {"doc_id": "paper_2506_06962", "title": "AR-RAG"}
    chunks = chunker.split_text(paper_text, base_metadata=meta, section_aware=True, exclude_references=True)

    sections = [c.section for c in chunks]
    assert "Abstract" in sections
    assert "Introduction" in sections
    assert "Methodology" in sections
    assert "Experiments" in sections
    # Reference section must be excluded
    assert "References" not in sections

    abstract_chunk = next(c for c in chunks if c.section == "Abstract")
    assert "Title: AR-RAG" in abstract_chunk.text
    assert "Autoregressive Retrieval Augmentation" in abstract_chunk.text
