"""Universal Document, Office File, Web URL, YouTube Transcript & Multimodal Parser.

Supports everyday file formats with zero external C-compiler dependencies:
- PDF (.pdf) with [Page N] boundary markers & optional Vision/OCR fallback for scanned pages
- PowerPoint (.pptx) with [Slide N] markers and speaker notes via OpenXML ZIP parsing
- Word Documents (.docx) with heading-to-Markdown conversion and table extraction
- Excel Spreadsheets (.xlsx) & CSV/TSV (.csv, .tsv) with Markdown table formatting
- Jupyter Notebooks (.ipynb) with markdown, code cells, and execution outputs
- Web HTML (.html, .htm) & Source Code (.py, .js, .ts, .sql, .json, .md, .txt)
- Images (.png, .jpg, .jpeg, .webp) via Multimodal Vision LLM or metadata fallback
- Live Web URLs & YouTube Video Transcripts (timestamped captions + descriptions)
"""
import csv
import hashlib
import html
import io
import json
import logging
import re
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_UPLOAD_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".markdown",
    ".docx", ".pptx", ".xlsx", ".csv", ".tsv",
    ".ipynb", ".html", ".htm", ".json",
    ".py", ".js", ".ts", ".sql", ".yaml", ".yml",
    ".png", ".jpg", ".jpeg", ".webp"
}


def compute_deterministic_doc_id(
    prefix: str,
    identifier: str,
    workspace_id: str = "default",
    content: Optional[str] = None
) -> str:
    """Generate a deterministic document ID to prevent duplicate indexing on re-import."""
    clean_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", identifier.strip())[:48].strip("_") or "doc"
    ws_part = "" if (not workspace_id or workspace_id == "default") else f"{re.sub(r'[^a-zA-Z0-9]', '', workspace_id)[:12]}_"
    if content:
        digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:8]
        return f"{prefix}_{ws_part}{clean_id}_{digest}"
    digest = hashlib.sha256(identifier.encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"{prefix}_{ws_part}{clean_id}_{digest}"


class _ArticleHTMLParser(HTMLParser):
    """Lightweight HTML-to-Markdown extractor stripping scripts, styles, and navigation noise."""

    SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "header", "iframe", "form"}
    HEADING_MAP = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### "}
    BLOCK_TAGS = {"p", "div", "section", "article", "blockquote", "pre", "tr", "ul", "ol"}

    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self._in_title = False
        self.title: str = ""
        self._chunks: List[str] = []
        self._in_td = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        t = tag.lower()
        if t in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        if t == "title":
            self._in_title = True
        elif t in self.HEADING_MAP:
            self._chunks.append(f"\n\n{self.HEADING_MAP[t]}")
        elif t == "li":
            self._chunks.append("\n- ")
        elif t in ("br", "hr"):
            self._chunks.append("\n")
        elif t in ("td", "th"):
            self._in_td = True
            self._chunks.append(" | ")
        elif t in self.BLOCK_TAGS:
            self._chunks.append("\n\n")

    def handle_endtag(self, tag: str):
        t = tag.lower()
        if t in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if self._skip_depth > 0:
            return
        if t == "title":
            self._in_title = False
        elif t in self.HEADING_MAP:
            self._chunks.append("\n")
        elif t in ("td", "th"):
            self._in_td = False
        elif t == "tr":
            self._chunks.append(" |\n")

    def handle_data(self, data: str):
        if self._skip_depth > 0:
            return
        cleaned = re.sub(r"[ \t\r\f\v]+", " ", data)
        if self._in_title:
            self.title += cleaned.strip()
        else:
            self._chunks.append(cleaned)

    def get_markdown(self) -> str:
        raw = "".join(self._chunks)
        raw = html.unescape(raw)
        # Collapse excessive blank lines
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def parse_pdf_bytes(
    pdf_bytes: bytes,
    vision_ocr_fn: Optional[Callable[[bytes, str], str]] = None
) -> Tuple[str, int]:
    """Extract text from PDF bytes preserving [Page N] markers for page-accurate citations."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages_output: List[str] = []
    total_pages = len(reader.pages)

    for idx, page in enumerate(reader.pages, start=1):
        try:
            txt = (page.extract_text() or "").strip()
        except Exception as e:
            logger.debug(f"Error extracting PDF page {idx}: {e}")
            txt = ""

        if txt:
            pages_output.append(f"[Page {idx}]\n{txt}")

    combined = "\n\n".join(pages_output).strip()

    # If PDF had pages but extracted almost no text (scanned PDF) and a Vision OCR callback is provided
    if total_pages > 0 and len(combined) < 40 and vision_ocr_fn is not None:
        try:
            ocr_text = vision_ocr_fn(pdf_bytes, "application/pdf")
            if ocr_text and ocr_text.strip():
                return f"[Page 1]\n{ocr_text.strip()}", total_pages
        except Exception as e:
            logger.warning(f"Vision OCR fallback on scanned PDF skipped: {e}")

    return combined, total_pages


def parse_docx_bytes(docx_bytes: bytes) -> str:
    """Extract structured Markdown text and tables from a Word (.docx) OpenXML archive."""
    try:
        with zipfile.ZipFile(io.BytesIO(docx_bytes)) as zf:
            if "word/document.xml" not in zf.namelist():
                raise ValueError("Invalid .docx file: missing word/document.xml")
            xml_bytes = zf.read("word/document.xml")
    except Exception as e:
        raise ValueError(f"Could not open .docx archive: {e}")

    root = ET.fromstring(xml_bytes)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    body = root.find("w:body", ns)
    if body is None:
        return ""

    blocks: List[str] = []

    for child in body:
        tag_name = child.tag.split("}")[-1]
        if tag_name == "p":
            # Detect paragraph heading style if present
            style_prefix = ""
            pPr = child.find("w:pPr", ns)
            if pPr is not None:
                pStyle = pPr.find("w:pStyle", ns)
                if pStyle is not None:
                    val = (pStyle.attrib.get(f"{{{ns['w']}}}val") or "").lower()
                    if "heading1" in val or val == "title":
                        style_prefix = "# "
                    elif "heading2" in val or val == "subtitle":
                        style_prefix = "## "
                    elif "heading3" in val:
                        style_prefix = "### "

            runs = [t.text for t in child.findall(".//w:t", ns) if t.text]
            para_text = "".join(runs).strip()
            if para_text:
                blocks.append(f"{style_prefix}{para_text}")

        elif tag_name == "tbl":
            rows_md: List[str] = []
            for r_idx, row in enumerate(child.findall(".//w:tr", ns)):
                cells: List[str] = []
                for cell in row.findall(".//w:tc", ns):
                    c_texts = [t.text for t in cell.findall(".//w:t", ns) if t.text]
                    cells.append(" ".join("".join(c_texts).split()))
                if any(cells):
                    rows_md.append("| " + " | ".join(cells) + " |")
                    if r_idx == 0:
                        rows_md.append("| " + " | ".join(["---"] * len(cells)) + " |")
            if rows_md:
                blocks.append("\n".join(rows_md))

    return "\n\n".join(blocks).strip()


def parse_pptx_bytes(pptx_bytes: bytes) -> str:
    """Extract slide-by-slide text and speaker notes from a PowerPoint (.pptx) archive."""
    try:
        with zipfile.ZipFile(io.BytesIO(pptx_bytes)) as zf:
            names = zf.namelist()
            slide_files = [
                n for n in names
                if re.match(r"^ppt/slides/slide\d+\.xml$", n)
            ]
            # Sort slides numerically (slide1.xml, slide2.xml, ...)
            slide_files.sort(key=lambda x: int(re.search(r"slide(\d+)\.xml$", x).group(1)))

            slides_output: List[str] = []
            for slide_path in slide_files:
                slide_num = int(re.search(r"slide(\d+)\.xml$", slide_path).group(1))
                xml_data = zf.read(slide_path)
                s_root = ET.fromstring(xml_data)

                # Extract all text runs <a:t>
                texts: List[str] = []
                for elem in s_root.iter():
                    if elem.tag.endswith("}t") and elem.text and elem.text.strip():
                        texts.append(elem.text.strip())

                # Check for matching speaker notes
                notes_path = f"ppt/notesSlides/notesSlide{slide_num}.xml"
                notes_texts: List[str] = []
                if notes_path in names:
                    try:
                        n_root = ET.fromstring(zf.read(notes_path))
                        for elem in n_root.iter():
                            if elem.tag.endswith("}t") and elem.text and elem.text.strip():
                                t_val = elem.text.strip()
                                if t_val != str(slide_num):
                                    notes_texts.append(t_val)
                    except Exception:
                        pass

                if texts or notes_texts:
                    title_line = texts[0] if texts else f"Slide {slide_num}"
                    body_lines = "\n".join(f"- {t}" for t in texts[1:]) if len(texts) > 1 else ""
                    notes_block = f"\nSpeaker Notes: {' '.join(notes_texts)}" if notes_texts else ""
                    slide_section = f"[Slide {slide_num}]\n## Slide {slide_num}: {title_line}\n{body_lines}{notes_block}".strip()
                    slides_output.append(slide_section)

            return "\n\n".join(slides_output).strip()
    except Exception as e:
        raise ValueError(f"Could not parse .pptx presentation: {e}")


def parse_xlsx_bytes(xlsx_bytes: bytes) -> str:
    """Extract worksheet tables from an Excel (.xlsx) OpenXML archive."""
    try:
        with zipfile.ZipFile(io.BytesIO(xlsx_bytes)) as zf:
            names = zf.namelist()
            shared_strings: List[str] = []
            if "xl/sharedStrings.xml" in names:
                ss_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
                for si in ss_root.iter():
                    if si.tag.endswith("}si"):
                        parts = [t.text for t in si.iter() if t.tag.endswith("}t") and t.text]
                        shared_strings.append("".join(parts))

            sheet_files = sorted([n for n in names if re.match(r"^xl/worksheets/sheet\d+\.xml$", n)])
            sheets_output: List[str] = []

            for s_idx, s_path in enumerate(sheet_files, start=1):
                s_root = ET.fromstring(zf.read(s_path))
                rows_data: List[List[str]] = []

                for row_el in s_root.iter():
                    if row_el.tag.endswith("}row"):
                        row_vals: List[str] = []
                        for cell_el in row_el:
                            if not cell_el.tag.endswith("}c"):
                                continue
                            c_type = cell_el.attrib.get("t", "")
                            v_el = next((ch for ch in cell_el if ch.tag.endswith("}v")), None)
                            is_el = next((ch for ch in cell_el if ch.tag.endswith("}is")), None)
                            val_str = ""
                            if c_type == "s" and v_el is not None and v_el.text:
                                idx = int(v_el.text)
                                if 0 <= idx < len(shared_strings):
                                    val_str = shared_strings[idx]
                            elif is_el is not None:
                                val_str = "".join(t.text for t in is_el.iter() if t.tag.endswith("}t") and t.text)
                            elif v_el is not None and v_el.text:
                                val_str = v_el.text
                            row_vals.append(val_str.strip())
                        if any(row_vals):
                            rows_data.append(row_vals)

                if rows_data:
                    md_lines = [f"## Sheet {s_idx}"]
                    for r_i, r_vals in enumerate(rows_data[:500]):
                        md_lines.append("| " + " | ".join(r_vals) + " |")
                        if r_i == 0:
                            md_lines.append("| " + " | ".join(["---"] * len(r_vals)) + " |")
                    sheets_output.append("\n".join(md_lines))

            return "\n\n".join(sheets_output).strip()
    except Exception as e:
        raise ValueError(f"Could not parse .xlsx spreadsheet: {e}")


def parse_csv_bytes(raw_bytes: bytes, delimiter: str = ",") -> str:
    """Parse CSV or TSV bytes into a structured Markdown table with column summary."""
    text = raw_bytes.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = [r for r in reader if any(cell.strip() for cell in r)]
    if not rows:
        return ""

    headers = [h.strip() for h in rows[0]]
    lines = [
        f"## Tabular Dataset ({len(rows) - 1} rows, Columns: {', '.join(headers)})",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    for r in rows[1:500]:
        cleaned_cells = [c.strip().replace("\n", " ") for c in r]
        lines.append("| " + " | ".join(cleaned_cells) + " |")

    return "\n".join(lines)


def parse_ipynb_bytes(raw_bytes: bytes) -> str:
    """Extract markdown narrative, code cells, and outputs from a Jupyter Notebook (.ipynb)."""
    data = json.loads(raw_bytes.decode("utf-8", errors="replace"))
    cells = data.get("cells", [])
    blocks: List[str] = []

    for idx, cell in enumerate(cells, start=1):
        c_type = cell.get("cell_type", "")
        source = cell.get("source", [])
        src_text = "".join(source) if isinstance(source, list) else str(source)
        if not src_text.strip():
            continue

        if c_type == "markdown":
            blocks.append(src_text.strip())
        elif c_type == "code":
            code_block = f"### Code Cell {idx}\n```python\n{src_text.strip()}\n```"
            outputs_text: List[str] = []
            for out in cell.get("outputs", []):
                if "text" in out:
                    t = "".join(out["text"]) if isinstance(out["text"], list) else str(out["text"])
                    if t.strip():
                        outputs_text.append(t.strip()[:600])
                elif "data" in out and "text/plain" in out["data"]:
                    tp = out["data"]["text/plain"]
                    t = "".join(tp) if isinstance(tp, list) else str(tp)
                    if t.strip():
                        outputs_text.append(t.strip()[:600])
            if outputs_text:
                code_block += "\nOutput:\n" + "\n".join(outputs_text)
            blocks.append(code_block)

    return "\n\n".join(blocks).strip()


def parse_any_file(
    filename: str,
    raw_bytes: bytes,
    mime_type: Optional[str] = None,
    vision_ocr_fn: Optional[Callable[[bytes, str], str]] = None
) -> Tuple[str, str]:
    """Universal file dispatcher returning (extracted_text, normalized_source_type).

    Supports PDF, DOCX, PPTX, XLSX, CSV, TSV, IPYNB, HTML, JSON, Code, TXT, MD, and Images.
    """
    ext = Path(filename).suffix.lower()
    mime = (mime_type or "").lower()

    if ext == ".pdf" or mime == "application/pdf":
        text, _ = parse_pdf_bytes(raw_bytes, vision_ocr_fn=vision_ocr_fn)
        return text, "pdf"

    if ext == ".docx" or "wordprocessingml" in mime:
        return parse_docx_bytes(raw_bytes), "docx"

    if ext == ".pptx" or "presentationml" in mime:
        return parse_pptx_bytes(raw_bytes), "pptx"

    if ext == ".xlsx" or "spreadsheetml" in mime:
        return parse_xlsx_bytes(raw_bytes), "xlsx"

    if ext == ".csv" or mime == "text/csv":
        return parse_csv_bytes(raw_bytes, delimiter=","), "csv"

    if ext == ".tsv" or mime == "text/tab-separated-values":
        return parse_csv_bytes(raw_bytes, delimiter="\t"), "tsv"

    if ext == ".ipynb":
        return parse_ipynb_bytes(raw_bytes), "ipynb"

    if ext in (".html", ".htm") or mime == "text/html":
        parser = _ArticleHTMLParser()
        parser.feed(raw_bytes.decode("utf-8", errors="replace"))
        return parser.get_markdown(), "html"

    if ext in (".png", ".jpg", ".jpeg", ".webp") or mime.startswith("image/"):
        if vision_ocr_fn is not None:
            ocr_res = vision_ocr_fn(raw_bytes, mime or f"image/{ext.lstrip('.')}")
            if ocr_res and ocr_res.strip():
                return ocr_res.strip(), "image"
        return f"[Image File: {filename}] (Configure Gemini or OpenAI Vision API key to enable full OCR text extraction.)", "image"

    # Default text / markdown / source code
    decoded = raw_bytes.decode("utf-8", errors="replace")
    src_type = ext.lstrip(".") if ext else "txt"
    return decoded, src_type


def _extract_youtube_id(url: str) -> Optional[str]:
    """Extract 11-character YouTube video ID from watch, short, or youtu.be URLs."""
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if "youtu.be" in host:
        vid = parsed.path.lstrip("/").split("/")[0]
        return vid if len(vid) == 11 else None
    if "youtube.com" in host:
        if parsed.path.startswith("/watch"):
            qs = urllib.parse.parse_qs(parsed.query)
            v = qs.get("v", [None])[0]
            return v if v and len(v) == 11 else None
        for prefix in ("/shorts/", "/embed/", "/v/", "/live/"):
            if parsed.path.startswith(prefix):
                vid = parsed.path[len(prefix):].split("/")[0]
                return vid if len(vid) == 11 else None
    return None


def _format_timestamp(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def extract_from_url(url: str, timeout: float = 20.0) -> Tuple[str, str, str]:
    """Fetch and extract structured text from a Web URL, ArXiv link, PDF link, or YouTube video.

    Returns:
        Tuple of (title, extracted_text, source_type)
    """
    clean_url = (url or "").strip()
    if not clean_url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9"
    }

    # 1. Check if YouTube Video URL
    yt_id = _extract_youtube_id(clean_url)
    if yt_id:
        watch_url = f"https://www.youtube.com/watch?v={yt_id}"
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
            res = client.get(watch_url)
            html_text = res.text

            title_match = re.search(r"<title>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
            raw_title = html.unescape(title_match.group(1)).replace("- YouTube", "").strip() if title_match else f"YouTube Video {yt_id}"

            # Extract player response JSON for description & caption tracks
            desc = ""
            transcript_lines: List[str] = []

            pr_match = re.search(r"ytInitialPlayerResponse\s*=\s*(\{.+?\})\s*;", html_text)
            if pr_match:
                try:
                    pr_data = json.loads(pr_match.group(1))
                    vd = pr_data.get("videoDetails", {})
                    if vd.get("title"):
                        raw_title = vd["title"]
                    desc = vd.get("shortDescription", "")

                    captions = (
                        pr_data.get("captions", {})
                        .get("playerCaptionsTracklistRenderer", {})
                        .get("captionTracks", [])
                    )
                    if captions:
                        # Prefer English track if available
                        track = next((c for c in captions if c.get("languageCode", "").startswith("en")), captions[0])
                        base_url = track.get("baseUrl")
                        if base_url:
                            t_res = client.get(base_url)
                            if t_res.status_code == 200 and t_res.text:
                                c_root = ET.fromstring(t_res.text)
                                for node in c_root.findall(".//text"):
                                    start_sec = float(node.attrib.get("start", "0"))
                                    line_txt = html.unescape(node.text or "").replace("\n", " ").strip()
                                    if line_txt:
                                        transcript_lines.append(f"[{_format_timestamp(start_sec)}] {line_txt}")
                except Exception as e:
                    logger.debug(f"YouTube caption parse fallback: {e}")

            sections = [f"# {raw_title}\nSource URL: {watch_url}"]
            if desc:
                sections.append(f"## Overview & Description\n{desc}")
            if transcript_lines:
                sections.append("## Transcript\n" + "\n".join(transcript_lines))
            elif not desc:
                # Fallback to HTML parser
                p = _ArticleHTMLParser()
                p.feed(html_text)
                md = p.get_markdown()
                if md:
                    sections.append(f"## Page Content\n{md[:4000]}")

            full_text = "\n\n".join(sections).strip()
            return raw_title, full_text, "youtube"

    # 2. General HTTP/HTTPS URL (PDF or HTML Article)
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        res = client.get(clean_url)
        res.raise_for_status()
        content_type = (res.headers.get("content-type") or "").lower()

        if "application/pdf" in content_type or clean_url.lower().endswith(".pdf"):
            text, _ = parse_pdf_bytes(res.content)
            url_path = Path(urllib.parse.urlparse(clean_url).path).stem or "Web_PDF"
            title = url_path.replace("_", " ").replace("-", " ").title()
            return title, text, "pdf_url"

        parser = _ArticleHTMLParser()
        parser.feed(res.text)
        extracted_md = parser.get_markdown()
        title = parser.title.strip() if parser.title.strip() else urllib.parse.urlparse(clean_url).netloc
        full_text = f"# {title}\nSource URL: {clean_url}\n\n{extracted_md}".strip()
        return title, full_text, "web_url"
