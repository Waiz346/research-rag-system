"""Unit tests for the enhanced ArxivCollector."""
from src.ingestion.arxiv_collector import ArxivCollector

def test_build_query_basic():
    q = ArxivCollector.build_query("large language models")
    assert q == "large language models"

def test_build_query_with_title_and_abstract():
    q = ArxivCollector.build_query(
        query="retrieval",
        title_keyword="RAG",
        abstract_keyword="benchmark"
    )
    assert 'ti:RAG' in q
    assert 'abs:benchmark' in q
    assert 'retrieval' in q

def test_fetch_citation_count_known_paper():
    # Lewis et al. 2020 RAG paper
    count = ArxivCollector.fetch_citation_count("2005.11401")
    assert isinstance(count, int)
    assert count >= 0

def test_resolve_seed_citation_direct_id():
    res = ArxivCollector.resolve_seed_citation("2005.11401v2")
    assert res is not None
    assert res["arxiv_id"] == "2005.11401"
