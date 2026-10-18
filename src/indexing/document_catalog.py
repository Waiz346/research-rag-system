"""Lightweight Document Catalog & Disk-Backed Storage Manager.

Optimized for memory efficiency:
- Keeps full document texts on disk (`data/documents/{doc_id}.txt`).
- Maintains a lightweight manifest (`data/documents/manifest.json`) for instant listing without RAM bloat.
- Provides lazy on-demand retrieval when a user opens or inspects a specific paper.
"""
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.config import settings

logger = logging.getLogger(__name__)

DOCUMENTS_DIR = settings.DATA_DIR / "documents"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = DOCUMENTS_DIR / "manifest.json"

@dataclass
class DocumentSummary:
    doc_id: str
    title: str
    source_type: str                  # "arxiv", "pdf", "text", "markdown"
    filename: str
    authors: str = "Unknown"
    citation_count: int = 0
    chunk_count: int = 0
    sections: List[str] = field(default_factory=list)
    created_at: str = ""
    char_count: int = 0
    summary: str = ""
    key_topics: List[str] = field(default_factory=list)
    suggested_questions: List[str] = field(default_factory=list)

@dataclass
class DocumentDetail:
    summary: DocumentSummary
    full_text: str


def _extract_source_guide(title: str, full_text: str, sections: List[str]) -> Dict[str, Any]:
    """Extract a concise NotebookLM-style Source Guide (summary, key_topics, suggested_questions) from document text."""
    import re
    clean_text = re.sub(r"\s+", " ", (full_text or "")).strip()

    # 1. Extract summary (prefer Abstract or opening 2-3 substantive sentences)
    summary = ""
    if clean_text:
        abs_match = re.search(r"(?:abstract|summary)[:\s—-]+(.{80,550}?)(?:\s+\d+\.\s+|\s+introduction|\Z)", clean_text, re.IGNORECASE)
        raw_excerpt = abs_match.group(1).strip() if abs_match else clean_text[:500]
        # Trim to sentence boundary if possible
        sentences = re.split(r"(?<=[.!?])\s+", raw_excerpt)
        summary = " ".join(sentences[:3]).strip()
        if len(summary) > 420:
            summary = summary[:417].rsplit(" ", 1)[0] + "..."
    if not summary:
        summary = f"Indexed source covering {title}."

    # 2. Extract key topics from title and capitalized technical phrases
    stop = {
        "the", "and", "for", "with", "from", "using", "via", "based", "into", "over", "under",
        "study", "paper", "model", "models", "method", "methods", "approach", "system", "analysis",
        "general", "abstract", "introduction", "conclusion", "references", "methodology", "experiments"
    }
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", f"{title} {' '.join(sections or [])} {summary[:250]}")
    topics: List[str] = []
    seen_lower = set()
    for w in words:
        wl = w.lower()
        if wl not in stop and wl not in seen_lower and len(w) >= 4:
            seen_lower.add(wl)
            topics.append(w)
        if len(topics) >= 6:
            break

    # 3. Build 3 source-specific suggested questions
    short_title = title[:65] + ("..." if len(title) > 65 else "")
    suggested_questions = [
        f"What are the core contributions and main findings of \"{short_title}\"?",
        f"What methodology or architecture is proposed in \"{short_title}\"?",
        f"What experimental results or limitations are discussed in \"{short_title}\"?"
    ]
    return {
        "summary": summary,
        "key_topics": topics,
        "suggested_questions": suggested_questions
    }


class DocumentCatalog:
    """Manages paper metadata catalog and lazy full-text retrieval on disk."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or DOCUMENTS_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.storage_dir / "manifest.json"
        self._ensure_manifest()

    def _ensure_manifest(self):
        if not self.manifest_file.exists():
            with open(self.manifest_file, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _load_manifest(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(self.manifest_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading manifest: {e}. Reinitializing.")
            return {}

    def _save_manifest(self, data: Dict[str, Dict[str, Any]]):
        temp_file = self.storage_dir / "manifest.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        temp_file.replace(self.manifest_file)

    def _enrich_item_guide(self, doc_id: str, item: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure a manifest item has summary, key_topics, and suggested_questions."""
        if item.get("summary") and item.get("key_topics") and item.get("suggested_questions"):
            return item
        text_file = self.storage_dir / f"{doc_id}.txt"
        sample_text = ""
        if text_file.exists():
            try:
                with open(text_file, "r", encoding="utf-8", errors="replace") as f:
                    sample_text = f.read(2500)
            except Exception:
                sample_text = ""
        guide = _extract_source_guide(
            title=item.get("title", doc_id),
            full_text=sample_text,
            sections=item.get("sections", [])
        )
        item["summary"] = guide["summary"]
        item["key_topics"] = guide["key_topics"]
        item["suggested_questions"] = guide["suggested_questions"]
        return item

    def list_documents(self) -> List[DocumentSummary]:
        """Return lightweight metadata summaries (including Source Guides) for all papers."""
        manifest = self._load_manifest()
        summaries: List[DocumentSummary] = []
        dirty = False
        for doc_id, item in manifest.items():
            if not item.get("summary"):
                self._enrich_item_guide(doc_id, item)
                dirty = True
            summaries.append(DocumentSummary(
                doc_id=doc_id,
                title=item.get("title", doc_id),
                source_type=item.get("source_type", "unknown"),
                filename=item.get("filename", ""),
                authors=item.get("authors", "Unknown"),
                citation_count=int(item.get("citation_count", 0)),
                chunk_count=int(item.get("chunk_count", 0)),
                sections=item.get("sections", []),
                created_at=item.get("created_at", ""),
                char_count=int(item.get("char_count", 0)),
                summary=item.get("summary", ""),
                key_topics=item.get("key_topics", []),
                suggested_questions=item.get("suggested_questions", [])
            ))
        if dirty:
            self._save_manifest(manifest)
        # Sort by creation date descending
        summaries.sort(key=lambda s: s.created_at, reverse=True)
        return summaries

    def get_document(self, doc_id: str) -> Optional[DocumentDetail]:
        """Lazy load full text and details for a specific paper on-demand from disk."""
        manifest = self._load_manifest()
        item = manifest.get(doc_id)
        if not item:
            return None

        text_file = self.storage_dir / f"{doc_id}.txt"
        full_text = ""
        if text_file.exists():
            try:
                with open(text_file, "r", encoding="utf-8", errors="replace") as f:
                    full_text = f.read()
            except Exception as e:
                logger.error(f"Error reading document text file for {doc_id}: {e}")

        if not item.get("summary"):
            guide = _extract_source_guide(item.get("title", doc_id), full_text[:3000], item.get("sections", []))
            item.update(guide)
            self._save_manifest(manifest)

        summary = DocumentSummary(
            doc_id=doc_id,
            title=item.get("title", doc_id),
            source_type=item.get("source_type", "unknown"),
            filename=item.get("filename", ""),
            authors=item.get("authors", "Unknown"),
            citation_count=int(item.get("citation_count", 0)),
            chunk_count=int(item.get("chunk_count", 0)),
            sections=item.get("sections", []),
            created_at=item.get("created_at", ""),
            char_count=len(full_text),
            summary=item.get("summary", ""),
            key_topics=item.get("key_topics", []),
            suggested_questions=item.get("suggested_questions", [])
        )
        return DocumentDetail(summary=summary, full_text=full_text)

    def register_document(
        self,
        doc_id: str,
        title: str,
        full_text: str,
        source_type: str = "arxiv",
        filename: Optional[str] = None,
        authors: str = "Unknown",
        citation_count: int = 0,
        chunk_count: int = 0,
        sections: Optional[List[str]] = None
    ) -> DocumentSummary:
        """Save document text to disk and record metadata + Source Guide in lightweight manifest."""
        # 1. Save full text file to disk
        text_file = self.storage_dir / f"{doc_id}.txt"
        with open(text_file, "w", encoding="utf-8", errors="replace") as f:
            f.write(full_text)

        # 2. Extract NotebookLM-style Source Guide
        dedup_sections = list(dict.fromkeys(sections or ["General"]))
        guide = _extract_source_guide(title=title, full_text=full_text[:3500], sections=dedup_sections)

        # 3. Update manifest
        manifest = self._load_manifest()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        manifest[doc_id] = {
            "doc_id": doc_id,
            "title": title,
            "source_type": source_type,
            "filename": filename or f"{doc_id}.txt",
            "authors": authors,
            "citation_count": citation_count,
            "chunk_count": chunk_count,
            "sections": dedup_sections,
            "created_at": now_str,
            "char_count": len(full_text),
            "summary": guide["summary"],
            "key_topics": guide["key_topics"],
            "suggested_questions": guide["suggested_questions"]
        }
        self._save_manifest(manifest)
        logger.info(f"Registered document '{doc_id}' in catalog ({len(full_text)} chars, {chunk_count} chunks).")

        return DocumentSummary(
            doc_id=doc_id,
            title=title,
            source_type=source_type,
            filename=filename or f"{doc_id}.txt",
            authors=authors,
            citation_count=citation_count,
            chunk_count=chunk_count,
            sections=dedup_sections,
            created_at=now_str,
            char_count=len(full_text),
            summary=guide["summary"],
            key_topics=guide["key_topics"],
            suggested_questions=guide["suggested_questions"]
        )

    def delete_document(self, doc_id: str) -> bool:
        """Remove document text from disk and delete manifest record."""
        manifest = self._load_manifest()
        if doc_id not in manifest:
            return False

        del manifest[doc_id]
        self._save_manifest(manifest)

        text_file = self.storage_dir / f"{doc_id}.txt"
        if text_file.exists():
            try:
                text_file.unlink()
            except Exception as e:
                logger.warning(f"Could not remove text file {text_file}: {e}")

        return True

    def sync_from_chroma(self, chroma_collection) -> int:
        """Sync existing papers from ChromaDB metadata into manifest if manifest is empty."""
        manifest = self._load_manifest()
        if manifest:
            return len(manifest)

        try:
            # Query ONLY metadatas to maintain strict memory efficiency (no vectors, no text)
            stored = chroma_collection.get(include=["metadatas"])
            if not stored or not stored.get("metadatas"):
                return 0

            metas = stored["metadatas"]
            docs_grouped: Dict[str, Dict[str, Any]] = {}

            for m in metas:
                if not m:
                    continue
                d_id = m.get("doc_id")
                if not d_id:
                    continue

                if d_id not in docs_grouped:
                    docs_grouped[d_id] = {
                        "title": m.get("title", d_id),
                        "authors": m.get("authors", "Unknown"),
                        "citation_count": int(m.get("citation_count", 0)),
                        "sections": set(),
                        "chunk_count": 0
                    }
                docs_grouped[d_id]["chunk_count"] += 1
                if m.get("section"):
                    docs_grouped[d_id]["sections"].add(m.get("section"))

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for d_id, info in docs_grouped.items():
                manifest[d_id] = {
                    "doc_id": d_id,
                    "title": info["title"],
                    "source_type": "arxiv",
                    "filename": f"{d_id}.txt",
                    "authors": info["authors"],
                    "citation_count": info["citation_count"],
                    "chunk_count": info["chunk_count"],
                    "sections": list(info["sections"]) or ["General"],
                    "created_at": now_str,
                    "char_count": 0
                }

            self._save_manifest(manifest)
            logger.info(f"Synchronized {len(manifest)} existing documents from ChromaDB into catalog.")
            return len(manifest)
        except Exception as e:
            logger.warning(f"Could not sync from ChromaDB: {e}")
            return 0
