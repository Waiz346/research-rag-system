"""Formula-aware PDF and ArXiv HTML extractor.

Fixes mangled mathematical formulas in scientific papers by:
1. Extracting native LaTeX (<math alttext="...">) from ArXiv HTML when available.
2. Reconstructing embedded MathType / Word Form XObjects (/Meta*) in PDFs by
   computing true horizontal (x) and vertical (y) glyph coordinates from PDF
   Tm/Tc/TJ operators, mapping Symbol Private Use Area (U+F000..U+F0FF) glyphs
   to LaTeX, and grouping subscripts/superscripts into $...$ LaTeX notation.
"""
import html
import io
import logging
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import httpx
from pypdf import PdfReader
from pypdf.generic import ByteStringObject, ContentStream
import pypdf._page as _pypdf_page

logger = logging.getLogger(__name__)

Font = _pypdf_page.Font

# Symbol font Private Use Area (U+F000..U+F0FF) & Unicode math -> LaTeX mapping
PUA_TO_LATEX: Dict[str, str] = {
    "\uf0e5": r"\sum",
    "∑": r"\sum",
    "\uf0b4": r" \times ",
    "×": r" \times ",
    "\uf0ce": r" \in ",
    "∈": r" \in ",
    "\uf0cf": r" \notin ",
    "\uf02d": "-",
    "\u2212": "-",
    "\uf02b": "+",
    "\uf03d": "=",
    "\uf028": "(",
    "\uf029": ")",
    "\uf05b": "[",
    "\uf05d": "]",
    "\uf0be": r" \to ",
    "→": r" \to ",
    "\uf0bb": r" \approx ",
    "≈": r" \approx ",
    "\uf0b9": r" \neq ",
    "≠": r" \neq ",
    "\uf0a3": r" \le ",
    "≤": r" \le ",
    "\uf0b3": r" \ge ",
    "≥": r" \ge ",
    "\uf0b1": r" \pm ",
    "±": r" \pm ",
    "\uf0d7": r" \cdot ",
    "⋅": r" \cdot ",
    "\uf0a5": r"\infty",
    "∞": r"\infty",
    "\uf0f2": r"\int",
    "∫": r"\int",
    "\uf0b6": r"\partial",
    "∂": r"\partial",
    "\uf0d6": r"\sqrt",
    "√": r"\sqrt",
    "\uf0c4": r" \otimes ",
    "⊗": r" \otimes ",
    "\uf0c5": r" \oplus ",
    "⊕": r" \oplus ",
    "\uf0c7": r" \cap ",
    "∩": r" \cap ",
    "\uf0c8": r" \cup ",
    "∪": r" \cup ",
    "\uf0b5": r" \propto ",
    "∝": r" \propto ",
    "ξ": r"\xi",
    "ε": r"\varepsilon",
    "λ": r"\lambda",
    "μ": r"\mu",
    "σ": r"\sigma",
    "ω": r"\omega",
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "θ": r"\theta",
    "π": r"\pi",
    "ρ": r"\rho",
    "τ": r"\tau",
    "φ": r"\phi",
    "ψ": r"\psi",
    "η": r"\eta",
    "κ": r"\kappa",
    "ν": r"\nu",
    "χ": r"\chi",
    "ζ": r"\zeta",
    "Γ": r"\Gamma",
    "Δ": r"\Delta",
    "Θ": r"\Theta",
    "Λ": r"\Lambda",
    "Π": r"\Pi",
    "Σ": r"\Sigma",
    "Φ": r"\Phi",
    "Ψ": r"\Psi",
    "Ω": r"\Omega",
    "ℝ": r"\mathbb{R}",
}


def _decode_pdf_item(item: Any, cur_font: Optional[Any]) -> str:
    """Decode a PDF string or ByteStringObject using the active pypdf Font."""
    if isinstance(item, ByteStringObject) or (cur_font and getattr(cur_font, "encoding", None) == "utf-16-be"):
        raw_bytes = bytes(item) if isinstance(item, ByteStringObject) else getattr(item, "original_bytes", b"")
        codes = raw_bytes.decode("utf-16-be", errors="replace")
    else:
        codes = str(item)

    if not cur_font:
        return codes

    cmap = getattr(cur_font, "character_map", {}) or {}
    return "".join(cmap.get(c, c) for c in codes)


def reconstruct_xobject_latex(obj: Any, reader: PdfReader) -> Optional[str]:
    """Reconstruct a MathType / Word Form XObject into clean inline LaTeX ($...$)."""
    try:
        resources = obj.get("/Resources")
        if not resources:
            return None
        res_obj = resources.get_object()
        font_dict = res_obj.get("/Font")
        if not font_dict:
            return None

        fonts: Dict[str, Any] = {}
        for fname, fobj in font_dict.get_object().items():
            try:
                fonts[fname] = Font.from_font_resource(fobj.get_object())
            except Exception:
                continue

        cs = ContentStream(obj, reader)
        cur_font = None
        fsize = 12.0
        tc = 0.0
        x, y = 0.0, 0.0
        last_cm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        glyphs: List[Dict[str, Any]] = []

        for operands, op in cs.operations:
            if op == b"Tf" and len(operands) >= 2:
                cur_font = fonts.get(str(operands[0]))
                fsize = float(operands[1])
            elif op == b"Tc" and len(operands) >= 1:
                tc = float(operands[0])
            elif op == b"Tm" and len(operands) >= 6:
                x, y = float(operands[4]), float(operands[5])
            elif op in (b"Td", b"TD") and len(operands) >= 2:
                x += float(operands[0])
                y += float(operands[1])
            elif op == b"cm" and len(operands) >= 6:
                last_cm = [float(v) for v in operands[:6]]
            elif op == b"Do":
                # MathType embeds blackboard-bold R (\mathbb{R}) as an inline image inside the formula XObject
                glyphs.append({"x": float(last_cm[4]), "y": float(y), "s": 12.0, "ch": "ℝ"})
            elif op in (b"TJ", b"Tj") and operands:
                arr = operands[0] if op == b"TJ" else [operands[0]]
                for item in arr:
                    if isinstance(item, (int, float)):
                        x -= (float(item) / 1000.0) * fsize
                    else:
                        decoded = _decode_pdf_item(item, cur_font)
                        widths = getattr(cur_font, "character_widths", {}) if cur_font else {}
                        for ch in decoded:
                            w = (widths.get(ch, 500) / 1000.0) * fsize
                            if ch.strip():
                                glyphs.append({"x": float(x), "y": float(y), "s": float(fsize), "ch": ch})
                            x += w + tc

        if not glyphs:
            return None

        glyphs.sort(key=lambda g: g["x"])

        # Determine baseline y and font size from primary math characters
        main_glyphs = [g for g in glyphs if 9.0 <= g["s"] <= 15.5]
        if not main_glyphs:
            main_glyphs = glyphs
        base_y = statistics.median([g["y"] for g in main_glyphs])
        base_s = statistics.median([g["s"] for g in main_glyphs])

        out: List[str] = []
        i = 0
        while i < len(glyphs):
            g = glyphs[i]
            ch_mapped = PUA_TO_LATEX.get(g["ch"], g["ch"])

            # 1. Large summation / integral operator with lower & upper limits
            if g["ch"] in ("\uf0e5", "∑", "\uf0f2", "∫") or g["s"] > base_s * 1.35:
                sub_parts: List[str] = []
                sup_parts: List[str] = []
                j = i + 1
                while j < len(glyphs) and glyphs[j]["s"] < base_s * 0.82:
                    if glyphs[j]["y"] < base_y - 1.2:
                        sub_parts.append(PUA_TO_LATEX.get(glyphs[j]["ch"], glyphs[j]["ch"]))
                    elif glyphs[j]["y"] > base_y + 1.8:
                        sup_parts.append(PUA_TO_LATEX.get(glyphs[j]["ch"], glyphs[j]["ch"]))
                    else:
                        break
                    j += 1
                tok = ch_mapped
                if sub_parts:
                    tok += "_{" + "".join(sub_parts).strip() + "}"
                if sup_parts:
                    tok += "^{" + "".join(sup_parts).strip() + "}"
                out.append(tok + " ")
                i = j
                continue

            # 2. Subscript or Superscript sequence
            if g["s"] < base_s * 0.82 and abs(g["y"] - base_y) > 1.3:
                is_sup = g["y"] > base_y
                script_glyphs = []
                j = i
                while (
                    j < len(glyphs)
                    and glyphs[j]["s"] < base_s * 0.82
                    and ((glyphs[j]["y"] > base_y) == is_sup)
                ):
                    script_glyphs.append(glyphs[j])
                    j += 1

                script_base_s = max(sg["s"] for sg in script_glyphs)
                script_str = ""
                k = 0
                while k < len(script_glyphs):
                    sg = script_glyphs[k]
                    if sg["s"] < script_base_s * 0.85:
                        nested = []
                        while k < len(script_glyphs) and script_glyphs[k]["s"] < script_base_s * 0.85:
                            nested.append(PUA_TO_LATEX.get(script_glyphs[k]["ch"], script_glyphs[k]["ch"]))
                            k += 1
                        script_str += "_{" + "".join(nested).strip() + "}"
                    else:
                        script_str += PUA_TO_LATEX.get(sg["ch"], sg["ch"])
                        k += 1

                op_char = "^" if is_sup else "_"
                clean_script = re.sub(r"\s+", " ", script_str).strip()
                out.append(f"{op_char}{{{clean_script}}}")
                i = j
                continue

            out.append(ch_mapped)
            i += 1

        latex = "".join(out)
        # Convert || x || to \| x \| for cleaner LaTeX
        latex = latex.replace("||", r"\|")
        latex = re.sub(r"\s+", " ", latex).strip()
        return f"${latex}$"
    except Exception as e:
        logger.debug(f"Could not reconstruct XObject formula: {e}")
        return None


def extract_page_with_formulas(page: Any, reader: PdfReader) -> str:
    """Extract text from a single PDF page while reconstructing any MathType/Form XObjects inline."""
    resources = page.get("/Resources")
    if not resources:
        return page.extract_text() or ""

    res_obj = resources.get_object()
    xobj_dict = res_obj.get("/XObject")

    # Map Form XObject names (e.g. '/Meta36') -> reconstructed LaTeX ('$...$')
    xobj_latex_map: Dict[str, str] = {}
    if xobj_dict:
        for xname, xval in xobj_dict.get_object().items():
            try:
                xo = xval.get_object()
                if xo.get("/Subtype") == "/Form":
                    latex_str = reconstruct_xobject_latex(xo, reader)
                    if latex_str:
                        xobj_latex_map[str(xname)] = latex_str
            except Exception:
                continue

    # If the page has no Form XObject formulas, use standard extract_text with PUA cleanup
    if not xobj_latex_map:
        raw = page.extract_text() or ""
        return clean_pua_symbols(raw)

    # Walk the page's top-level ContentStream in exact reading order, combining body text and /Meta* XObjects
    font_dict = res_obj.get("/Font")
    fonts: Dict[str, Any] = {}
    if font_dict:
        for fname, fobj in font_dict.get_object().items():
            try:
                fonts[fname] = Font.from_font_resource(fobj.get_object())
            except Exception:
                continue

    contents_obj = page.get("/Contents")
    if not contents_obj:
        return clean_pua_symbols(page.extract_text() or "")
    cs = ContentStream(contents_obj.get_object(), reader)
    cur_font = None
    fsize = 11.0
    cur_y: Optional[float] = None
    pieces: List[str] = []

    for operands, op in cs.operations:
        if op == b"Tf" and len(operands) >= 2:
            cur_font = fonts.get(str(operands[0]))
            fsize = float(operands[1])
        elif op == b"Tm" and len(operands) >= 6:
            new_y = float(operands[5])
            if cur_y is not None and abs(new_y - cur_y) > 4.0:
                pieces.append("\n")
            cur_y = new_y
        elif op in (b"Td", b"TD") and len(operands) >= 2:
            dy = float(operands[1])
            if abs(dy) > 4.0:
                pieces.append("\n")
            if cur_y is not None:
                cur_y += dy
        elif op == b"T*":
            pieces.append("\n")
        elif op in (b"TJ", b"Tj") and operands:
            arr = operands[0] if op == b"TJ" else [operands[0]]
            seg = []
            for item in arr:
                if isinstance(item, (int, float)):
                    if item < -180:
                        seg.append(" ")
                else:
                    seg.append(_decode_pdf_item(item, cur_font))
            pieces.append("".join(seg))
        elif op == b"Do" and operands:
            xname = str(operands[0])
            if xname in xobj_latex_map:
                pieces.append(f" {xobj_latex_map[xname]} ")

    text = "".join(pieces)
    text = clean_pua_symbols(text)
    # Collapse excessive spaces while preserving paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_pua_symbols(text: str) -> str:
    """Replace any stray Symbol font Private Use Area characters with clean Unicode/LaTeX equivalents."""
    if not text:
        return ""
    out = text.replace("\x00", "")
    for pua_char, replacement in PUA_TO_LATEX.items():
        if ord(pua_char) >= 0xF000 and pua_char in out:
            out = out.replace(pua_char, replacement)
    return out


def extract_pdf_with_formulas(pdf_source: Any, max_pages: int = 35) -> str:
    """Extract text and reconstructed LaTeX formulas from a PDF path or bytes."""
    try:
        if isinstance(pdf_source, (str, Path)):
            reader = PdfReader(str(pdf_source))
        elif isinstance(pdf_source, bytes):
            reader = PdfReader(io.BytesIO(pdf_source))
        else:
            reader = PdfReader(pdf_source)

        extracted_pages: List[str] = []
        for page in reader.pages[:max_pages]:
            try:
                page_text = extract_page_with_formulas(page, reader)
                if page_text and page_text.strip():
                    extracted_pages.append(page_text)
            except Exception as e:
                logger.debug(f"Fallback to standard extract_text on page: {e}")
                fallback = clean_pua_symbols(page.extract_text() or "")
                if fallback.strip():
                    extracted_pages.append(fallback)
        return "\n\n".join(extracted_pages)
    except Exception as e:
        logger.warning(f"Error extracting PDF with formulas: {e}")
        return ""


def extract_arxiv_html_with_formulas(paper_id: str) -> Optional[str]:
    """Fetch ArXiv's native HTML version (if available) and convert <math alttext='...'> directly to LaTeX."""
    clean_id = paper_id.strip()
    urls = [
        f"https://arxiv.org/html/{clean_id}",
        f"https://arxiv.org/html/{re.sub(r'v\\d+$', '', clean_id)}",
    ]
    for url in urls:
        try:
            with httpx.Client(follow_redirects=True, timeout=20.0) as client:
                res = client.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Research-RAG/1.0"},
                )
                if res.status_code != 200 or "ltx_" not in res.text:
                    continue

                raw_html = res.text
                # Isolate <article> if present
                art_match = re.search(r"<article[^>]*>([\s\S]*?)</article>", raw_html, re.IGNORECASE)
                body_html = art_match.group(1) if art_match else raw_html

                # Strip script/style/nav/header/footer
                body_html = re.sub(r"<(script|style|nav|footer)[^>]*>[\s\S]*?</\1>", "", body_html, flags=re.IGNORECASE)

                # Convert <math ... alttext="...">...</math> to $...$ or $$...$$
                def _math_repl(m: re.Match) -> str:
                    attrs = m.group(1) or ""
                    inner = m.group(2) or ""
                    alt_m = re.search(r'alttext="([^"]+)"', attrs)
                    if alt_m:
                        latex_str = html.unescape(alt_m.group(1)).strip()
                    else:
                        latex_str = html.unescape(re.sub(r"<[^>]+>", " ", inner)).strip()
                    if not latex_str:
                        return ""
                    is_block = 'display="block"' in attrs
                    delim = "$$" if is_block else "$"
                    return f" {delim}{latex_str}{delim} "

                body_html = re.sub(r"<math([^>]*)>([\s\S]*?)</math>", _math_repl, body_html, flags=re.IGNORECASE)

                # Convert headings and paragraph breaks to Markdown
                body_html = re.sub(r"<h[12][^>]*>([\s\S]*?)</h[12]>", r"\n\n## \1\n\n", body_html, flags=re.IGNORECASE)
                body_html = re.sub(r"<h[3456][^>]*>([\s\S]*?)</h[3456]>", r"\n\n### \1\n\n", body_html, flags=re.IGNORECASE)
                body_html = re.sub(r"</(p|div|section|tr|li|figcaption|blockquote)>", "\n\n", body_html, flags=re.IGNORECASE)
                body_html = re.sub(r"<(td|th)[^>]*>", " | ", body_html, flags=re.IGNORECASE)

                # Strip remaining HTML tags
                clean_txt = html.unescape(re.sub(r"<[^>]+>", " ", body_html)).replace("\x00", "")
                clean_txt = re.sub(r"[ \t]+", " ", clean_txt)
                clean_txt = re.sub(r" *\n *", "\n", clean_txt)
                clean_txt = re.sub(r"\n{3,}", "\n\n", clean_txt).strip()

                if len(clean_txt) > 1500:
                    logger.info(
                        f"Extracted native LaTeX-preserving HTML from {url} ({len(clean_txt):,} chars)"
                    )
                    return clean_txt
        except Exception as e:
            logger.debug(f"ArXiv HTML check skipped for {url}: {e}")
    return None

