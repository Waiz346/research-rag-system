"""ArXiv paper fetcher and PDF text extractor."""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import arxiv
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

class ArxivCollector:
    """Collects research papers from ArXiv and extracts their contents."""

    def __init__(self, download_dir: Optional[Path] = None):
        self.download_dir = download_dir or settings.DOWNLOADS_DIR
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def search_papers(
        self,
        query: str,
        max_results: int = 5,
        download_pdf: bool = False
    ) -> List[PaperMetadata]:
        """Search ArXiv by query string and optionally download/extract PDF text."""
        logger.info(f"Querying ArXiv for '{query}' (limit={max_results})...")
        
        client = arxiv.Client(
            page_size=max(max_results, 5),
            delay_seconds=0.5,
            num_retries=2
        )
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results: List[PaperMetadata] = []
        for paper in client.results(search):
            paper_id = paper.get_short_id()
            authors = [author.name for author in paper.authors]
            published = paper.published.strftime("%Y-%m-%d")
            
            content = f"Title: {paper.title}\nAuthors: {', '.join(authors)}\nAbstract: {paper.summary}"
            
            if download_pdf:
                try:
                    pdf_path = self.download_dir / f"{paper_id}.pdf"
                    if not pdf_path.exists():
                        paper.download_pdf(dirpath=str(self.download_dir), filename=f"{paper_id}.pdf")
                    extracted_text = self._extract_text_from_pdf(pdf_path)
                    if extracted_text:
                        content += f"\n\nFull Content:\n{extracted_text}"
                except Exception as e:
                    logger.warning(f"Could not download or parse PDF for {paper_id}: {e}")

            meta = PaperMetadata(
                id=paper_id,
                title=paper.title,
                authors=authors,
                summary=paper.summary,
                published=published,
                pdf_url=paper.pdf_url,
                text_content=content,
                categories=paper.categories
            )
            results.append(meta)

        logger.info(f"Retrieved {len(results)} papers from ArXiv.")
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
