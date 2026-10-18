"""Text chunker with academic paper section awareness and sliding window fallback.

Inspired by RAGFlow's Paper Template & DeepDoc layout parsing:
- Recognizes canonical scientific paper sections (Abstract, Intro, Method, Experiments, Conclusion).
- Isolates or excludes reference sections to prevent citation list noise from polluting retrieval.
- Preserves section hierarchy metadata for visual grounding and traceable citations.
"""
import re
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from src.config import settings

logger = logging.getLogger(__name__)

# Canonical scientific paper & everyday coursework section regex matching patterns
SECTION_REGEX = re.compile(
    r"(?im)^(?:\s*(?:[0-9IVXLCDM]+\.?\s+)?(Abstract|Introduction|Overview|Executive\s+Summary|Summary|Key\s+Takeaways|Syllabus|Course\s+Schedule|Schedule|Grading(?:\s+Policy)?|Assignments?|Requirements|Deliverables|Related\s+Work|Background|Methodology|Method|Approach|Architecture|System\s+Model|Model|Experiments?|Experimental\s+Setup|Results|Evaluation|Discussion|Ablation(?:\s+Studies?)?|Conclusions?|Future\s+Work|Transcript|References|Bibliography)(?:\s*[:\n\r]|\s*$))"
)

PAGE_MARKER_REGEX = re.compile(r"(?:\[|#{1,3}\s*|\b)(Page|Slide)\s+(\d+)\b", re.IGNORECASE)

# Canonical normalization map
SECTION_CANONICAL_MAP = {
    "abstract": "Abstract",
    "executive summary": "Overview",
    "overview": "Overview",
    "summary": "Summary",
    "key takeaways": "Summary",
    "syllabus": "Syllabus",
    "course schedule": "Schedule",
    "schedule": "Schedule",
    "grading": "Grading",
    "grading policy": "Grading",
    "assignment": "Assignments",
    "assignments": "Assignments",
    "requirements": "Requirements",
    "deliverables": "Requirements",
    "transcript": "Transcript",
    "introduction": "Introduction",
    "related work": "Related Work",
    "background": "Related Work",
    "methodology": "Methodology",
    "method": "Methodology",
    "approach": "Methodology",
    "architecture": "Methodology",
    "system model": "Methodology",
    "model": "Methodology",
    "experiment": "Experiments",
    "experiments": "Experiments",
    "experimental setup": "Experiments",
    "results": "Experiments",
    "evaluation": "Experiments",
    "discussion": "Discussion",
    "ablation": "Discussion",
    "ablation study": "Discussion",
    "ablation studies": "Discussion",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "future work": "Conclusion",
    "references": "References",
    "bibliography": "References",
}

@dataclass
class TextChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    section: str = "General"

class DocumentChunker:
    """Splits text documents into semantically coherent, section-aware chunks."""

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def split_text(
        self,
        text: str,
        base_metadata: Dict[str, Any],
        section_aware: bool = True,
        exclude_references: bool = True
    ) -> List[TextChunk]:
        """Split text into chunks. Uses section-aware paper template if headers exist, else sliding window."""
        if not text or not text.strip():
            return []

        if section_aware:
            sections = self._extract_paper_sections(text)
            if len(sections) > 1 or (len(sections) == 1 and sections[0][0] != "General"):
                logger.debug(f"Detected {len(sections)} paper sections: {[s[0] for s in sections]}")
                return self._chunk_sections(sections, base_metadata, exclude_references=exclude_references)

        # Fallback to standard sliding window
        return self._sliding_window_split(text, base_metadata, section_name="General")

    def _extract_paper_sections(self, text: str) -> List[Tuple[str, str]]:
        """Identify academic paper sections based on headings."""
        matches = list(SECTION_REGEX.finditer(text))
        if not matches:
            # Check for Markdown headers like ## Introduction, # Abstract
            md_matches = list(re.finditer(r"(?im)^#{1,3}\s+(.+)$", text))
            if md_matches:
                sections: List[Tuple[str, str]] = []
                for i, m in enumerate(md_matches):
                    sec_name = m.group(1).strip()
                    start = m.start()
                    end = md_matches[i + 1].start() if i + 1 < len(md_matches) else len(text)
                    sec_text = text[start:end].strip()
                    canonical = self._canonicalize_section_name(sec_name)
                    sections.append((canonical, sec_text))
                return sections
            return [("General", text)]

        sections: List[Tuple[str, str]] = []

        # If there is content before the first matched section (e.g. Title, Authors, Citations)
        first_start = matches[0].start()
        header_preamble = text[:first_start].strip()

        for i, m in enumerate(matches):
            raw_name = m.group(1).strip().lower()
            canonical_name = self._canonicalize_section_name(raw_name)
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sec_text = text[start:end].strip()

            # Attach header preamble (Title, Authors) to Abstract for unified metadata context
            if i == 0 and header_preamble and canonical_name == "Abstract":
                sec_text = f"{header_preamble}\n\n{sec_text}"
                header_preamble = ""

            sections.append((canonical_name, sec_text))

        # If header preamble wasn't attached to abstract, prepend it as Header section
        if header_preamble:
            sections.insert(0, ("Header", header_preamble))

        return sections

    def _canonicalize_section_name(self, raw_name: str) -> str:
        """Map raw section heading to canonical paper section name."""
        cleaned = re.sub(r"^[0-9IVXLCDM\.\s]+", "", raw_name).strip().lower()
        for key, val in SECTION_CANONICAL_MAP.items():
            if key in cleaned:
                return val
        return raw_name.title()

    @staticmethod
    def _detect_location_meta(text: str, active_page: int = 0, active_label: str = "") -> Tuple[int, str]:
        """Detect [Page N] or [Slide N] marker in text and return updated (page_num, label)."""
        matches = list(PAGE_MARKER_REGEX.finditer(text))
        if matches:
            last_m = matches[-1]
            kind = last_m.group(1).title()
            num = int(last_m.group(2))
            return num, f"{kind} {num}"
        return active_page, active_label

    def _chunk_sections(
        self,
        sections: List[Tuple[str, str]],
        base_metadata: Dict[str, Any],
        exclude_references: bool = True
    ) -> List[TextChunk]:
        """Create structured chunks for each detected section."""
        all_chunks: List[TextChunk] = []
        doc_id = base_metadata.get("doc_id", "doc")
        global_chunk_idx = 0
        active_page = 0
        active_label = ""

        for sec_name, sec_text in sections:
            is_ref = (sec_name == "References")
            if is_ref and exclude_references:
                logger.debug(f"Excluding References section from chunk index for doc '{doc_id}'.")
                continue

            cleaned = re.sub(r"\s+", " ", sec_text).strip()
            if not cleaned:
                continue

            active_page, active_label = self._detect_location_meta(cleaned, active_page, active_label)

            # If section is reasonably sized (e.g., Abstract or Short Summary), keep as 1 atomic chunk
            if len(cleaned) <= int(self.chunk_size * 1.3):
                chunk_id = f"{doc_id}_{sec_name.lower().replace(' ', '_')}_{global_chunk_idx}"
                meta = {
                    **base_metadata,
                    "section": sec_name,
                    "chunk_index": global_chunk_idx,
                    "is_reference": is_ref,
                    "start_char": 0,
                    "end_char": len(cleaned),
                }
                if active_page > 0:
                    meta["page_number"] = active_page
                    meta["location_label"] = active_label
                meta = self._clean_metadata(meta)
                # Prefix with section tag for semantic clarity in LLM & retrieval
                tagged_text = f"[{sec_name}] {cleaned}" if not cleaned.startswith(f"[{sec_name}]") else cleaned
                all_chunks.append(TextChunk(
                    chunk_id=chunk_id,
                    text=tagged_text,
                    metadata=meta,
                    section=sec_name
                ))
                global_chunk_idx += 1
            else:
                # Split large section into overlapping chunks within section boundary
                sec_chunks = self._sliding_window_split(
                    cleaned,
                    base_metadata,
                    section_name=sec_name,
                    start_index=global_chunk_idx,
                    initial_page=active_page,
                    initial_label=active_label
                )
                all_chunks.extend(sec_chunks)
                global_chunk_idx += len(sec_chunks)

        return all_chunks

    def _sliding_window_split(
        self,
        text: str,
        base_metadata: Dict[str, Any],
        section_name: str = "General",
        start_index: int = 0,
        initial_page: int = 0,
        initial_label: str = ""
    ) -> List[TextChunk]:
        """Split text via sliding window with sentence snapping and page/slide tracking."""
        cleaned_text = re.sub(r"\s+", " ", text).strip()
        if not cleaned_text:
            return []

        chunks: List[TextChunk] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        chunk_idx = start_index
        doc_id = base_metadata.get("doc_id", "doc")
        sec_slug = re.sub(r"[^a-z0-9_]", "_", section_name.lower())
        active_page = initial_page
        active_label = initial_label

        while start < len(cleaned_text):
            end = min(start + self.chunk_size, len(cleaned_text))

            # Try to snap to sentence end if within reasonable distance
            if end < len(cleaned_text):
                period_pos = cleaned_text.rfind(". ", start, end)
                if period_pos != -1 and (period_pos - start) > (self.chunk_size * 0.6):
                    end = period_pos + 1

            chunk_content = cleaned_text[start:end].strip()
            if chunk_content:
                active_page, active_label = self._detect_location_meta(chunk_content, active_page, active_label)
                meta = {
                    **base_metadata,
                    "section": section_name,
                    "chunk_index": chunk_idx,
                    "start_char": start,
                    "end_char": end,
                    "is_reference": (section_name == "References")
                }
                if active_page > 0:
                    meta["page_number"] = active_page
                    meta["location_label"] = active_label
                meta = self._clean_metadata(meta)
                if section_name == "General":
                    chunk_id = f"{doc_id}_chunk_{chunk_idx}"
                else:
                    chunk_id = f"{doc_id}_{sec_slug}_{chunk_idx}"

                # Prefix with section tag if not already general
                tagged_text = f"[{section_name}] {chunk_content}" if section_name != "General" else chunk_content

                chunks.append(TextChunk(
                    chunk_id=chunk_id,
                    text=tagged_text,
                    metadata=meta,
                    section=section_name
                ))
                chunk_idx += 1

            start += step
            if end == len(cleaned_text):
                break

        return chunks

    @staticmethod
    def _clean_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
        """Convert list or non-primitive values to strings for vector DB storage."""
        cleaned = {}
        for k, v in meta.items():
            if isinstance(v, list):
                cleaned[k] = ", ".join(str(item) for item in v)
            elif v is None:
                cleaned[k] = ""
            else:
                cleaned[k] = v
        return cleaned
