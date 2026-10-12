"""Unit tests for VectorStoreManager Hybrid Search and BM25 integration."""
import pytest
from src.indexing.vector_store import VectorStoreManager
from src.ingestion.chunker import TextChunk

@pytest.fixture
def temp_vector_store(tmp_path):
    store = VectorStoreManager(
        persist_directory=str(tmp_path / "chroma"),
        collection_name="test_hybrid_collection"
    )
    yield store
    store.reset()

def test_hybrid_and_bm25_search(temp_vector_store):
    chunks = [
        TextChunk(
            chunk_id="paper1_abs_0",
            text="[Abstract] Autoregressive Retrieval Augmentation (AR-RAG) for generative vision and image synthesis.",
            metadata={"doc_id": "2506.06962", "title": "AR-RAG", "section": "Abstract"},
            section="Abstract"
        ),
        TextChunk(
            chunk_id="paper2_abs_0",
            text="[Abstract] Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks by Lewis et al.",
            metadata={"doc_id": "2005.11401", "title": "RAG for NLP", "section": "Abstract"},
            section="Abstract"
        ),
        TextChunk(
            chunk_id="paper3_abs_0",
            text="[Abstract] Attention Is All You Need. The Transformer architecture for neural sequence modeling.",
            metadata={"doc_id": "1706.03762", "title": "Attention Is All You Need", "section": "Abstract"},
            section="Abstract"
        )
    ]
    temp_vector_store.add_chunks(chunks)
    assert temp_vector_store.count() == 3

    # 1. Test BM25 sparse search for exact keyword "Transformer"
    bm25_res = temp_vector_store.bm25_search("Transformer", top_k=2)
    assert len(bm25_res) > 0
    assert bm25_res[0].metadata["title"] == "Attention Is All You Need"
    assert bm25_res[0].bm25_score > 0.0

    # 2. Test Dense semantic search for "language model external knowledge"
    dense_res = temp_vector_store.dense_search("language model external knowledge", top_k=2)
    assert len(dense_res) > 0
    assert dense_res[0].dense_score > 0.0

    # 3. Test Hybrid search combining both
    hybrid_res = temp_vector_store.hybrid_search("Lewis retrieval augmented NLP", top_k=2)
    assert len(hybrid_res) > 0
    assert hybrid_res[0].metadata["title"] == "RAG for NLP"
    assert hybrid_res[0].score > 0.0

    # 4. Test Threshold Filtering (cutoff)
    strict_res = temp_vector_store.hybrid_search("Astrophysics and black holes", top_k=2, score_threshold=0.85)
    # Completely unrelated query should be dropped under high threshold
    assert len(strict_res) == 0
