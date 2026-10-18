"""1-Click Everyday Study & Research Studio Tools and Obsidian/BibTeX Export Engine.

Provides NotebookLM-style studio transformations over indexed workspace documents:
1. study_guide: Exam Study Guide, Key Concepts & Formula Cheat Sheet
2. quiz: Practice Quiz (MCQs & Short Answer with grounded solutions)
3. comparison_matrix: Side-by-side Literature & Method Comparison Matrix
4. action_items: Deadlines, Syllabus Requirements & Action Items Extractor
5. executive_briefing: 2-Minute Spoken Audio Briefing Script & Executive Summary
6. Markdown / Obsidian / BibTeX Exporter
"""
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from src.rag.generator import LLMGenerator


STUDIO_ACTION_TITLES = {
    "study_guide": "📝 Comprehensive Study Guide & Concept Cheat Sheet",
    "quiz": "🎯 Interactive Practice Quiz & Flashcards",
    "comparison_matrix": "📊 Literature & Methodology Comparison Matrix",
    "action_items": "📌 Key Deadlines, Syllabus & Action Items Digest",
    "executive_briefing": "🎙️ Audio Overview — Two-Host Deep Dive (Alex & Sam)",
    "audio_briefing": "🎙️ Audio Overview — Two-Host Deep Dive (Alex & Sam)",
}


def generate_notebook_guide(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Synthesize a NotebookLM-style Notebook Guide (overview, key_topics, suggested_questions) across active documents."""
    if not documents:
        return {
            "source_count": 0,
            "overview": "Add documents, lecture slides, web links, or ArXiv papers to generate an automatic Notebook Guide.",
            "key_topics": [],
            "suggested_questions": []
        }

    titles = [d.get("title", "Untitled") for d in documents[:8]]
    topics_seen: List[str] = []
    summaries: List[str] = []

    for idx, d in enumerate(documents[:6], start=1):
        for t in (d.get("key_topics") or []):
            if t not in topics_seen and len(topics_seen) < 8:
                topics_seen.append(t)
        s = (d.get("summary") or "").strip()
        if not s:
            txt = re.sub(r"\s+", " ", (d.get("full_text") or "")).strip()
            s = txt[:180] + ("..." if len(txt) > 180 else "")
        if s:
            first_sent = re.split(r"(?<=[.!?])\s+", s)[0]
            summaries.append(f"**{d.get('title', f'Source {idx}')}** ([{idx}]): {first_sent}")

    if len(documents) == 1:
        overview = f"This notebook contains **1 active source** (*{titles[0]}*). {summaries[0] if summaries else ''}"
    else:
        lead_titles = ", ".join(f"*{t[:50]}{'...' if len(t) > 50 else ''}*" for t in titles[:3])
        more = f" and {len(documents) - 3} more" if len(documents) > 3 else ""
        overview = (
            f"This notebook synthesizes **{len(documents)} sources** ({lead_titles}{more}). "
            + (" ".join(summaries[:2]) if summaries else "")
        )

    # Dynamic suggested questions grounded in the actual active documents
    suggested: List[str] = []
    if len(titles) >= 2:
        t0 = titles[0][:50] + ("..." if len(titles[0]) > 50 else "")
        t1 = titles[1][:50] + ("..." if len(titles[1]) > 50 else "")
        suggested.append(f"How do \"{t0}\" and \"{t1}\" differ in their core approach and results?")
        suggested.append("What are the overarching themes and shared methodologies across all sources in this notebook?")
    if titles:
        t0 = titles[0][:55] + ("..." if len(titles[0]) > 55 else "")
        suggested.append(f"What is the primary problem solved by \"{t0}\" and how is it evaluated?")
    if topics_seen:
        suggested.append(f"Explain the role of {', '.join(topics_seen[:3])} across the indexed documents.")

    return {
        "source_count": len(documents),
        "overview": overview,
        "key_topics": topics_seen[:8],
        "suggested_questions": suggested[:4]
    }


def generate_studio_artifact(
    action_type: str,
    documents: List[Dict[str, Any]],
    generator: LLMGenerator,
    custom_focus: Optional[str] = None
) -> Dict[str, Any]:
    """Generate a structured everyday study or research artifact from workspace documents."""
    start = time.perf_counter()
    clean_action = (action_type or "study_guide").strip().lower()
    if clean_action not in STUDIO_ACTION_TITLES:
        clean_action = "study_guide"

    title = STUDIO_ACTION_TITLES[clean_action]

    if not documents:
        return {
            "action_type": clean_action,
            "title": title,
            "content": "No documents are currently available in the selected scope. Please ingest or select at least one document first.",
            "sources": [],
            "latency_ms": 0.0
        }

    # Build source context blocks and citation metadata from up to 8 documents
    selected_docs = documents[:8]
    context_blocks: List[str] = []
    sources_meta: List[Dict[str, Any]] = []

    for idx, doc in enumerate(selected_docs, start=1):
        doc_id = doc.get("doc_id", f"doc_{idx}")
        doc_title = doc.get("title", f"Document {idx}")
        authors = doc.get("authors", "Unknown")
        src_type = doc.get("source_type", "document")
        sections = doc.get("sections") or ["General"]
        full_text = (doc.get("full_text") or "").strip()
        excerpt = full_text[:1800] if full_text else f"Title: {doc_title} ({src_type})"

        context_blocks.append(
            f"[Source {idx}] Title: {doc_title} | Authors: {authors} | Type: {src_type} | Sections: {', '.join(sections[:5])}\n"
            f"Excerpt:\n{excerpt}"
        )
        sources_meta.append({
            "index": idx,
            "paper_id": str(doc_id),
            "title": str(doc_title),
            "snippet": excerpt[:280] + ("..." if len(excerpt) > 280 else ""),
            "score": 0.95,
            "citation_count": int(doc.get("citation_count", 0)),
            "section": ", ".join(sections[:2]) if sections else "Overview",
            "dense_score": 0.95,
            "bm25_score": 0.95,
            "search_type": "studio",
            "metadata": {
                "doc_id": doc_id,
                "title": doc_title,
                "authors": authors,
                "source_type": src_type
            }
        })

    combined_context = "\n\n---\n\n".join(context_blocks)
    focus_clause = f"\nSpecific User Focus: {custom_focus.strip()}\n" if custom_focus and custom_focus.strip() else ""

    if generator.provider == "mock":
        content = _mock_studio_artifact(clean_action, selected_docs, custom_focus)
    else:
        prompt = _build_studio_prompt(clean_action, combined_context, focus_clause)
        content = generator.generate(
            prompt,
            system_prompt="You are an expert academic synthesizer and study coach. Cite sources using [Source X] tags."
        )

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return {
        "action_type": clean_action,
        "title": title,
        "content": content,
        "sources": sources_meta,
        "latency_ms": latency_ms
    }


def _build_studio_prompt(action_type: str, context: str, focus_clause: str) -> str:
    if action_type == "study_guide":
        task = (
            "Create a structured, high-yield **Exam Study Guide & Cheat Sheet** from these sources:\n"
            "1. **Core Concepts & Definitions**: Clear bullet points explaining key terminology and mechanisms.\n"
            "2. **Architecture, Methodology & Formulas**: Step-by-step breakdown of algorithms, equations, or workflows.\n"
            "3. **High-Yield Exam Takeaways**: Key comparisons and common pitfalls to remember.\n"
            "4. **Self-Check Review Checklist**: Actionable study checklist.\n"
            "Cite every point with [Source X]."
        )
    elif action_type == "quiz":
        task = (
            "Create an **Interactive Practice Quiz & Flashcard Set** based strictly on the provided sources:\n"
            "- Generate **5 Multiple-Choice Questions (MCQs)** (with options A, B, C, D) testing conceptual understanding and empirical facts.\n"
            "- Generate **3 Short-Answer / Synthesis Questions**.\n"
            "- At the bottom, include an **Answer Key & Detailed Explanations** section citing [Source X] for every answer."
        )
    elif action_type == "comparison_matrix":
        task = (
            "Create a comprehensive **Literature & Methodology Comparison Matrix** comparing all provided sources:\n"
            "1. Output a Markdown Table with columns: `| Source | Core Problem / Objective | Methodology / Architecture | Key Results / Findings | Limitations / Notes |`.\n"
            "2. Follow the table with a **Cross-Source Synthesis** highlighting where the documents complement or contrast with each other, citing [Source X]."
        )
    elif action_type == "action_items":
        task = (
            "Scan all provided course materials, announcements, syllabi, and notes to extract a **Key Deadlines, Deliverables & Action Items Digest**:\n"
            "1. **Upcoming Deadlines, Exams & Milestones**: Any dates, midterm/exam topics, or submission rules mentioned.\n"
            "2. **Required Readings & Technical Topics**: Core topics, chapters, or methods you must master.\n"
            "3. **Prioritized Action Checklist**: Concrete next steps for the student/researcher, citing [Source X]."
        )
    else:  # executive_briefing / audio_briefing (NotebookLM Two-Host Deep Dive)
        task = (
            "Write an engaging **NotebookLM-Style Two-Host Deep Dive Audio Overview** dialogue between **Alex** (lead synthesizer) and **Sam** (curious analytical co-host) unpacking these sources:\n"
            "- Format every spoken turn strictly as `**Alex:** ...` or `**Sam:** ...` on its own paragraph so the two-voice audio engine can alternate voices.\n"
            "- Have Alex and Sam naturally discuss the core problem, key breakthroughs, methodology, and how [Source 1], [Source 2], etc. connect or contrast.\n"
            "- Keep the tone insightful, conversational, and grounded strictly in the sources with inline [Source X] citations (6–10 turns total)."
        )

    return f"Source Documents:\n{context}\n{focus_clause}\nTask:\n{task}"


def _mock_studio_artifact(
    action_type: str,
    docs: List[Dict[str, Any]],
    custom_focus: Optional[str] = None
) -> str:
    """Deterministic, source-grounded Studio generator for offline/mock mode."""
    focus_line = f"**Focus Topic:** *{custom_focus}*\n\n" if custom_focus else ""
    source_bullets = []
    for idx, d in enumerate(docs, start=1):
        t = d.get("title", f"Document {idx}")
        txt = re.sub(r"\s+", " ", (d.get("full_text") or "")).strip()
        first_sent = txt[:200] + ("..." if len(txt) > 200 else "") if txt else "Core domain reference."
        source_bullets.append((idx, t, first_sent, d.get("source_type", "doc"), d.get("authors", "Unknown")))

    if action_type == "study_guide":
        concepts = "\n".join(
            f"- **{t}** ([Source {idx}]): {sent}"
            for idx, t, sent, _, _ in source_bullets
        )
        return (
            f"{focus_line}### 1. Core Concepts & High-Yield Definitions\n{concepts}\n\n"
            f"### 2. Key Mechanisms & Methodological Takeaways\n"
            f"- **Hybrid Retrieval & Grounding**: Combining dense semantic representations with lexical keyword matching (BM25 + RRF) prevents terminology dropout [Source 1].\n"
            f"- **Section & Page Provenance**: Structuring documents into canonical sections (`Abstract`, `Methodology`, `Experiments`) improves context precision [Source 1].\n\n"
            f"### 3. Exam & Research Review Checklist\n"
            + "\n".join(f"- [ ] Review core methodology and empirical claims in *{t}* [Source {idx}]" for idx, t, _, _, _ in source_bullets)
        )

    if action_type == "quiz":
        q_lines = [f"{focus_line}### Part 1: Practice Multiple-Choice Questions\n"]
        a_lines = ["\n### Part 2: Verified Answer Key & Explanations\n"]
        for idx, t, sent, _, _ in source_bullets[:5]:
            q_lines.append(
                f"**Q{idx}. According to *{t}*, what is the primary contribution or focus described in the text?**\n"
                f"- A) Unconstrained random sampling without external grounding\n"
                f"- B) {sent[:110]}\n"
                f"- C) Purely manual rule-based heuristics without indexing\n"
                f"- D) Hardware overclocking without algorithmic changes\n"
            )
            a_lines.append(f"- **Q{idx} Correct Answer: B** — Grounded directly in *{t}* [Source {idx}]: \"{sent[:140]}\"")
        return "\n".join(q_lines + a_lines)

    if action_type == "comparison_matrix":
        rows = [
            f"{focus_line}### Cross-Document Comparison Matrix\n",
            "| Source | Title | Format / Authors | Key Extracted Summary & Findings |",
            "| :--- | :--- | :--- | :--- |"
        ]
        for idx, t, sent, stype, auth in source_bullets:
            safe_t = t.replace("|", "/")
            safe_s = sent[:120].replace("|", "/")
            rows.append(f"| **[Source {idx}]** | *{safe_t}* | `{stype.upper()}` ({auth}) | {safe_s} [Source {idx}] |")
        rows.append(
            "\n### Cross-Source Synthesis\n"
            "Across the analyzed workspace documents, the primary themes converge on grounded knowledge retrieval, "
            "structured methodology, and empirical validation [Source 1]."
        )
        return "\n".join(rows)

    if action_type == "action_items":
        items = [f"{focus_line}### Extracted Coursework, Deadlines & Action Items\n"]
        for idx, t, sent, stype, _ in source_bullets:
            items.append(f"- **[Source {idx}] {t}** (`{stype.upper()}`): {sent}")
        items.append(
            "\n### Recommended Next Steps\n"
            "1. **Verify Key Dates & Scope**: Cross-check all announcements and syllabus requirements in [Source 1].\n"
            "2. **Run Practice Quiz**: Use the *Practice Quiz* studio tool to test recall over these materials."
        )
        return "\n".join(items)

    # executive_briefing / audio_briefing: Two-Host Deep Dive (Alex & Sam)
    turns = [
        f"{focus_line}**Alex:** Welcome back to the Deep Dive! Today we're unpacking **{len(source_bullets)} source(s)** in your notebook to connect the big ideas and practical takeaways.",
        f"**Sam:** Right, and what immediately stood out to me looking across these documents is how concrete the methodology is. Let's start with *{source_bullets[0][1]}* [Source 1]."
    ]
    for idx, t, sent, _, _ in source_bullets[:4]:
        speaker = "Alex" if idx % 2 == 1 else "Sam"
        turns.append(f"**{speaker}:** In *{t}*, the core insight is: {sent} [Source {idx}].")
    turns.append(
        "**Sam:** That ties everything together nicely—and since every claim here is linked to the source passages, you can click any citation badge or press **Read Aloud** to listen to our two-voice breakdown."
    )
    return "\n\n".join(turns)


def format_markdown_export(
    title: str,
    query: str,
    answer: str,
    sources: List[Dict[str, Any]],
    workspace_id: str = "default",
    include_bibtex: bool = True,
    obsidian_compatible: bool = True
) -> str:
    """Format a RAG synthesis or Studio report as an Obsidian-ready Markdown note with BibTeX & APA citations."""
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_slug = datetime.now().strftime("%Y-%m-%d")

    lines: List[str] = []
    if obsidian_compatible:
        lines.extend([
            "---",
            f'title: "{title.replace(chr(34), chr(39))}"',
            f"date: {date_slug}",
            f"workspace: {workspace_id}",
            "tags:",
            "  - nexusrag",
            "  - research-synthesis",
            f"  - {workspace_id}",
            "---",
            ""
        ])

    lines.append(f"# {title}\n")
    if query:
        lines.append(f"> **Query / Prompt:** {query}\n")
    lines.append(f"*Generated by NexusRAG Workbench on {now_iso} (Workspace: `{workspace_id}`)*\n")
    lines.append("## Synthesis & Findings\n")
    lines.append(answer.strip() + "\n")

    if sources:
        lines.append("## Cited Sources & Provenance\n")
        for s in sources:
            idx = s.get("index", 1)
            s_title = s.get("title", "Untitled")
            paper_id = s.get("paper_id", "doc")
            sec = s.get("section", "General")
            score = s.get("score", 0.0)
            snippet = (s.get("snippet") or "").replace("\n", " ").strip()
            wikilink = f"[[{s_title}]]" if obsidian_compatible else f"**{s_title}**"
            lines.append(
                f"- **[Source {idx}]** {wikilink} (`{paper_id}` | Section: *{sec}* | Score: `{score:.2f}`)\n"
                f"  > {snippet}"
            )
        lines.append("")

    if include_bibtex and sources:
        lines.append("## BibTeX & Reference Entries\n")
        lines.append("```bibtex")
        seen_papers = set()
        for s in sources:
            paper_id = str(s.get("paper_id", "doc"))
            if paper_id in seen_papers:
                continue
            seen_papers.add(paper_id)
            meta = s.get("metadata") or {}
            s_title = s.get("title", "Untitled")
            authors = meta.get("authors") or "Author, Unknown"
            published = str(meta.get("published") or date_slug)[:4]
            cite_key = re.sub(r"[^a-zA-Z0-9]", "_", paper_id)[:24]
            lines.extend([
                f"@article{{{cite_key},",
                f"  title = {{{s_title}}},",
                f"  author = {{{authors}}},",
                f"  year = {{{published}}},",
                f"  note = {{Indexed in NexusRAG ({paper_id})}}",
                "}"
            ])
        lines.append("```\n")

    return "\n".join(lines)
