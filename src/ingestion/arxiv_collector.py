"""ArXiv paper fetcher supporting seed citation tracking, Semantic Scholar forward citations, and keyword narrowing."""
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import arxiv
import httpx
from pypdf import PdfReader
from src.config import settings

logger = logging.getLogger(__name__)

@dataclass
class PaperMetadata:
    id: str
    title: str
    authors: List[str]
    summary: str
    published: str
    pdf_url: str
    text_content: str = ""
    categories: List[str] = field(default_factory=list)
    citation_count: int = 0
    cites_seed: Optional[str] = None

class ArxivCollector:
    """Collects research papers with citation graph expansion and targeted keyword narrowing."""

    def __init__(self, download_dir: Optional[Path] = None):
        self.download_dir = download_dir or settings.DOWNLOADS_DIR
        self.download_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def build_query(
        query: str,
        title_keyword: Optional[str] = None,
        abstract_keyword: Optional[str] = None
    ) -> str:
        """Construct structured ArXiv search query supporting title and abstract field filters."""
        query_parts = []
        clean_query = query.strip() if query else ""
        if clean_query:
            query_parts.append(clean_query)

        if title_keyword and title_keyword.strip():
            kw = title_keyword.strip()
            escaped = f'"{kw}"' if " " in kw else kw
            query_parts.append(f"ti:{escaped}")

        if abstract_keyword and abstract_keyword.strip():
            kw = abstract_keyword.strip()
            escaped = f'"{kw}"' if " " in kw else kw
            query_parts.append(f"abs:{escaped}")

        if not query_parts:
            return "all:retrieval augmented generation"

        return " AND ".join(query_parts)

    @staticmethod
    def fetch_citation_count(arxiv_id: str) -> int:
        """Query Semantic Scholar public Graph API to retrieve paper citation count."""
        clean_id = re.sub(r"v\d+$", "", arxiv_id.strip())
        url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{clean_id}?fields=citationCount"

        try:
            with httpx.Client(timeout=4.0) as client:
                res = client.get(url, headers={"User-Agent": "Research-RAG-System/1.0"})
                if res.status_code == 200:
                    data = res.json()
                    count = data.get("citationCount")
                    return int(count) if count is not None else 0
        except Exception as e:
            logger.debug(f"Could not fetch Semantic Scholar citation count for {clean_id}: {e}")
        return 0

    @staticmethod
    def resolve_seed_citation(seed_citation: str) -> Optional[Dict[str, str]]:
        """Resolve a user-prompted citation (ArXiv ID, DOI, or title) to an identifier and title."""
        cleaned = seed_citation.strip()
        # Direct ArXiv ID detection (e.g. 2005.11401, 2309.15217, arXiv:2005.11401)
        arxiv_match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", cleaned, re.IGNORECASE)
        if arxiv_match:
            raw_id = arxiv_match.group(1)
            clean_id = re.sub(r"v\d+$", "", raw_id)
            return {"arxiv_id": clean_id, "title": f"ArXiv Paper {clean_id}"}

        # Otherwise, search ArXiv by title/keywords to find the seed paper ID
        try:
            client = arxiv.Client(page_size=3, delay_seconds=0.5, num_retries=1)
            search = arxiv.Search(query=f'ti:"{cleaned}"', max_results=1)
            results = list(client.results(search))
            if not results:
                # Fallback to general search without quotes
                search = arxiv.Search(query=cleaned, max_results=1)
                results = list(client.results(search))

            if results:
                paper = results[0]
                clean_id = re.sub(r"v\d+$", "", paper.get_short_id())
                return {"arxiv_id": clean_id, "title": paper.title}
        except Exception as e:
            logger.warning(f"Could not resolve seed citation '{cleaned}' on ArXiv: {e}")

        return None

    def fetch_papers_citing_seed(self, seed_id: str, limit: int = 80) -> List[PaperMetadata]:
        """Fetch papers from Semantic Scholar that cite the specified seed paper."""
        clean_id = re.sub(r"v\d+$", "", seed_id.strip())
        url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{clean_id}/citations?fields=title,abstract,citationCount,externalIds,year,authors&limit={limit}"

        papers: List[PaperMetadata] = []
        try:
            with httpx.Client(timeout=8.0) as client:
                res = client.get(url, headers={"User-Agent": "Research-RAG-System/1.0"})
                if res.status_code == 200:
                    data = res.json().get("data", [])
                    for item in data:
                        cp = item.get("citingPaper")
                        if not cp or not cp.get("title"):
                            continue

                        title = cp.get("title")
                        abstract = cp.get("abstract") or "No abstract available in citation record."
                        external_ids = cp.get("externalIds") or {}
                        arxiv_id = external_ids.get("ArXiv") or cp.get("paperId", "")[:12]
                        citation_count = cp.get("citationCount") or 0
                        year = str(cp.get("year") or "Recent")
                        authors = [a.get("name") for a in (cp.get("authors") or []) if a.get("name")]
                        pdf_url = f"https://arxiv.org/abs/{arxiv_id}" if external_ids.get("ArXiv") else ""

                        content = (
                            f"Title: {title}\n"
                            f"Authors: {', '.join(authors) if authors else 'Unknown'}\n"
                            f"Cites Reference: {seed_id}\n"
                            f"Citation Count: {citation_count}\n"
                            f"Abstract: {abstract}"
                        )

                        papers.append(PaperMetadata(
                            id=arxiv_id,
                            title=title,
                            authors=authors,
                            summary=abstract,
                            published=year,
                            pdf_url=pdf_url,
                            text_content=content,
                            citation_count=citation_count,
                            cites_seed=seed_id
                        ))
                    logger.info(f"Fetched {len(papers)} papers that cite seed paper {seed_id}.")
        except Exception as e:
            logger.warning(f"Error fetching citing papers for {seed_id}: {e}")

        return papers

    def search_papers(
        self,
        query: str,
        seed_citation: Optional[str] = None,
        narrow_keywords: Optional[str] = None,
        title_keyword: Optional[str] = None,
        abstract_keyword: Optional[str] = None,
        min_citations: int = 0,
        max_results: int = 5,
        download_pdf: bool = False
    ) -> List[PaperMetadata]:
        """Search papers with seed citation graph traversal and keyword narrowing."""

        # -------------------------------------------------------------
        # BRANCH 1: User specified a seed citation (Forward Citation Search)
        # -------------------------------------------------------------
        if seed_citation and seed_citation.strip():
            seed_info = self.resolve_seed_citation(seed_citation)
            if seed_info:
                seed_id = seed_info["arxiv_id"]
                logger.info(f"Resolved seed citation '{seed_citation}' -> ArXiv ID: {seed_id} ('{seed_info['title']}')")

                # Fetch all papers that cite this specific landmark paper
                citing_candidates = self.fetch_papers_citing_seed(seed_id=seed_id, limit=80)

                if citing_candidates:
                    # Filter by user keywords (narrowing down)
                    filtered = []
                    keywords_to_check = []
                    if narrow_keywords and narrow_keywords.strip():
                        keywords_to_check.extend([k.strip().lower() for k in narrow_keywords.split(",") if k.strip()])
                    if query and query.strip() and query.strip().lower() != "retrieval augmented generation":
                        keywords_to_check.extend([k.strip().lower() for k in query.split() if len(k) > 2])

                    for p in citing_candidates:
                        text_to_search = f"{p.title} {p.summary}".lower()

                        # Check general keywords
                        if keywords_to_check:
                            if not any(kw in text_to_search for kw in keywords_to_check):
                                continue

                        # Check title keyword
                        if title_keyword and title_keyword.strip().lower() not in p.title.lower():
                            continue

                        # Check abstract keyword
                        if abstract_keyword and abstract_keyword.strip().lower() not in p.summary.lower():
                            continue

                        # Check minimum citations
                        if min_citations > 0 and p.citation_count < min_citations:
                            continue

                        filtered.append(p)

                    # Sort by citation impact descending
                    filtered.sort(key=lambda p: p.citation_count, reverse=True)
                    results = filtered[:max_results]
                    logger.info(f"Citation graph search: Narrowed {len(citing_candidates)} citing papers to {len(results)} matching keywords.")
                    if results:
                        return results
            else:
                logger.warning(f"Could not resolve seed citation '{seed_citation}'. Falling back to standard ArXiv search.")

        # -------------------------------------------------------------
        # BRANCH 2: Standard ArXiv Query with Keyword & Citation Filters
        # -------------------------------------------------------------
        composite_query = self.build_query(
            query=query,
            title_keyword=title_keyword,
            abstract_keyword=abstract_keyword
        )
        logger.info(f"Querying ArXiv with query='{composite_query}' (min_citations={min_citations}, target_limit={max_results})...")

        fetch_limit = max(max_results * 4, 16) if (min_citations > 0 or narrow_keywords) else max_results

        client = arxiv.Client(page_size=min(fetch_limit, 20), delay_seconds=0.5, num_retries=2)
        search = arxiv.Search(query=composite_query, max_results=fetch_limit, sort_by=arxiv.SortCriterion.Relevance)

        candidates: List[PaperMetadata] = []
        try:
            for paper in client.results(search):
                paper_id = paper.get_short_id()
                authors = [author.name for author in paper.authors]
                published = paper.published.strftime("%Y-%m-%d")
                citations = self.fetch_citation_count(paper_id)

                content = (
                    f"Title: {paper.title}\n"
                    f"Authors: {', '.join(authors)}\n"
                    f"Citations: {citations}\n"
                    f"Abstract: {paper.summary}"
                )

                candidates.append(PaperMetadata(
                    id=paper_id,
                    title=paper.title,
                    authors=authors,
                    summary=paper.summary,
                    published=published,
                    pdf_url=paper.pdf_url,
                    text_content=content,
                    categories=paper.categories,
                    citation_count=citations
                ))
        except Exception as e:
            logger.error(f"Error querying ArXiv results: {e}")

        # Narrow with narrow_keywords if provided
        filtered = candidates
        if narrow_keywords and narrow_keywords.strip():
            kw_list = [k.strip().lower() for k in narrow_keywords.split(",") if k.strip()]
            filtered = [
                p for p in filtered
                if any(kw in f"{p.title} {p.summary}".lower() for kw in kw_list)
            ]

        # Filter by minimum citations
        if min_citations > 0:
            filtered = [p for p in filtered if p.citation_count >= min_citations]

        filtered.sort(key=lambda p: p.citation_count, reverse=True)
        results = filtered[:max_results]

        # Download PDF if requested
        if download_pdf:
            for p in results:
                try:
                    pdf_path = self.download_dir / f"{p.id}.pdf"
                    if not pdf_path.exists():
                        search_single = arxiv.Search(id_list=[p.id], max_results=1)
                        for sp in client.results(search_single):
                            sp.download_pdf(dirpath=str(self.download_dir), filename=f"{p.id}.pdf")
                            break
                    extracted_text = self._extract_text_from_pdf(pdf_path)
                    if extracted_text:
                        p.text_content += f"\n\nFull Content:\n{extracted_text}"
                except Exception as e:
                    logger.warning(f"Could not parse PDF for {p.id}: {e}")

        logger.info(f"Successfully prepared {len(results)} papers for indexing.")
        return results

    def _extract_text_from_pdf(self, pdf_path: Path, max_pages: int = 10) -> str:
        """Extract text from downloaded PDF file up to max_pages."""
        reader = PdfReader(str(pdf_path))
        extracted_pages = []
        for i, page in enumerate(reader.pages[:max_pages]):
            text = page.extract_text()
            if text:
                extracted_pages.append(text)
        return "\n".join(extracted_pages)
