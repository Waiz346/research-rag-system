"""Interactive NotebookLM-Style UI for the Research RAG Intelligence system.

Features:
- Source Library: Upload custom PDF, TXT, MD files + ArXiv citation search.
- Memory-Efficient Document List: Displays lightweight metadata without loading heavy full texts.
- On-Demand Content Inspector: View full paper text and section hierarchy lazily when requested.
- Conversational Studio: Multi-turn chat with source filtering and traceable section citations.
- Dynamic Model Switcher: Hot-swap between Qwen, GLM, Gemini, OpenAI, Ollama, and Mock.
"""
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Research RAG Studio",
    page_icon="📓",
    layout="wide"
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "inspect_doc_id" not in st.session_state:
    st.session_state.inspect_doc_id = None
if "selected_doc_ids" not in st.session_state:
    st.session_state.selected_doc_ids = []

# Top Header
st.title("📓 Research RAG Studio")
st.caption("NotebookLM-style conversational research workbench powered by Hybrid Retrieval (ChromaDB + BM25) and Section-Aware Grounding.")

# ==============================================================================
# SIDEBAR: System Status, LLM Switcher & Quality Controls
# ==============================================================================
with st.sidebar:
    st.header("⚙️ System Status")
    try:
        health_res = requests.get(f"{API_URL}/health", timeout=3)
        if health_res.status_code == 200:
            hdata = health_res.json()
            st.success(f"Backend Online ({hdata['environment']})")
            st.metric("Total Indexed Chunks", hdata["indexed_documents_count"])
            st.info(f"Active Model: `{hdata['llm_provider']}`")
        else:
            st.error("API returned error.")
    except Exception:
        st.warning("⚠️ API is offline. Start FastAPI backend with:\n`uvicorn src.api.main:app --reload`")

    st.markdown("---")
    st.subheader("🤖 LLM Model Switcher")

    PROVIDER_OPTIONS = {
        "Qwen (Alibaba / DashScope)": {
            "key": "qwen",
            "default_model": "qwen-plus",
            "models": ["qwen-plus", "qwen-turbo", "qwen-max", "qwen2.5-72b-instruct", "qwen2.5-7b-instruct"],
            "default_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "key_prompt": "DashScope API Key (sk-...)"
        },
        "GLM (Zhipu AI / BigModel)": {
            "key": "glm",
            "default_model": "glm-4-flash",
            "models": ["glm-4-flash", "glm-4-plus", "glm-4-air", "glm-4"],
            "default_url": "https://open.bigmodel.cn/api/paas/v4",
            "key_prompt": "Zhipu AI Key (api_key)"
        },
        "Google Gemini": {
            "key": "gemini",
            "default_model": "gemini-2.5-flash",
            "models": ["gemini-2.5-flash", "gemini-1.5-pro", "gemini-3.8-flash"],
            "default_url": "https://generativelanguage.googleapis.com/v1beta",
            "key_prompt": "Google AI Studio Key (AIzaSy...)"
        },
        "OpenAI": {
            "key": "openai",
            "default_model": "gpt-4o-mini",
            "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            "default_url": "https://api.openai.com/v1",
            "key_prompt": "OpenAI API Key (sk-...)"
        },
        "Ollama (Local)": {
            "key": "ollama",
            "default_model": "qwen2.5:7b",
            "models": ["qwen2.5:7b", "llama3", "mistral", "deepseek-r1:7b"],
            "default_url": "http://localhost:11434/v1",
            "key_prompt": "No key needed (Local)"
        },
        "Custom OpenAI-Compatible (SiliconFlow / vLLM)": {
            "key": "custom",
            "default_model": "Qwen/Qwen2.5-7B-Instruct",
            "models": ["Qwen/Qwen2.5-7B-Instruct", "deepseek-ai/DeepSeek-V3", "THUDM/glm-4-9b-chat"],
            "default_url": "https://api.siliconflow.cn/v1",
            "key_prompt": "API Key (sk-...)"
        },
        "Mock Engine (Offline Testing)": {
            "key": "mock",
            "default_model": "mock-eval-engine",
            "models": ["mock-eval-engine"],
            "default_url": "",
            "key_prompt": "No key needed"
        }
    }

    selected_provider_label = st.selectbox(
        "Provider",
        options=list(PROVIDER_OPTIONS.keys()),
        index=0
    )
    provider_info = PROVIDER_OPTIONS[selected_provider_label]

    model_choice = st.selectbox(
        "Model Name",
        options=provider_info["models"] + ["Custom..."]
    )
    if model_choice == "Custom...":
        chosen_model = st.text_input("Enter model identifier", value=provider_info["default_model"])
    else:
        chosen_model = model_choice

    if provider_info["key"] not in ["mock", "ollama"]:
        entered_api_key = st.text_input(
            provider_info["key_prompt"],
            type="password",
            help="Your credentials are maintained in memory for this session without rewriting disk files."
        )
    else:
        entered_api_key = ""

    with st.expander("⚙️ Advanced Endpoint Settings", expanded=False):
        chosen_base_url = st.text_input("Base URL", value=provider_info["default_url"])

    if st.button("⚡ Switch & Test Model", use_container_width=True):
        with st.spinner("Connecting to model..."):
            try:
                update_payload = {
                    "provider": provider_info["key"],
                    "model_name": chosen_model,
                    "api_key": entered_api_key if entered_api_key else None,
                    "base_url": chosen_base_url if chosen_base_url else None
                }
                up_res = requests.post(f"{API_URL}/config/llm", json=update_payload, timeout=10)
                if up_res.status_code == 200:
                    test_res = requests.post(f"{API_URL}/config/llm/test", timeout=25)
                    if test_res.status_code == 200:
                        tdata = test_res.json()
                        if tdata["success"]:
                            st.success(f"✅ {tdata['message']}")
                        else:
                            st.warning(f"⚠️ Switched to {chosen_model}, but test ping failed: {tdata.get('error') or tdata.get('message')}")
                    st.rerun()
                else:
                    st.error(f"Failed to update config: {up_res.text}")
            except Exception as e:
                st.error(f"Error switching model: {e}")

    st.markdown("---")
    st.subheader("🎯 Retrieval & Quality Controls")

    search_strategy = st.selectbox(
        "🔍 Search Strategy",
        options=["Hybrid (Dense + BM25)", "Dense Only (ChromaDB)", "Sparse Only (BM25)"],
        index=0,
        help="Hybrid applies Reciprocal Rank Fusion (RRF) to blend conceptual vector embeddings with exact keyword matches."
    )
    mode_key = "hybrid"
    if "Dense Only" in search_strategy:
        mode_key = "dense"
    elif "Sparse Only" in search_strategy:
        mode_key = "sparse"

    score_threshold = st.slider(
        "🛡️ Quality Gate (Relevance Cutoff)",
        min_value=0.00,
        max_value=0.80,
        value=0.35,
        step=0.05,
        help="RAGFlow Quality Gate: Reject low-relevance chunks below this cutoff to prevent hallucinations."
    )

    top_k = st.slider("Top Chunks per Query", min_value=1, max_value=8, value=3)

    use_hyde = st.checkbox(
        "🔮 HyDE (Hypothetical Embeddings)",
        value=True,
        help="Synthesizes a hypothetical passage before search to align vocabulary with scientific literature."
    )

    st.markdown("---")
    if st.button("🗑️ Reset Entire Vector Index", use_container_width=True):
        try:
            requests.post(f"{API_URL}/reset")
            st.session_state.messages = []
            st.session_state.selected_doc_ids = []
            st.session_state.inspect_doc_id = None
            st.warning("Index cleared.")
            st.rerun()
        except Exception as e:
            st.error(str(e))

# ==============================================================================
# MAIN WORKSPACE: Two-Column NotebookLM Layout
# ==============================================================================
col_sources, col_studio = st.columns([1.1, 1.9], gap="large")

# ------------------------------------------------------------------------------
# COLUMN 1: Sources Shelf (Upload, Ingest, Lightweight List & Lazy Inspection)
# ------------------------------------------------------------------------------
with col_sources:
    st.subheader("📚 Sources")

    # Add Sources Tabs
    tab_upload, tab_arxiv = st.tabs(["📤 Upload Files", "🌐 Fetch ArXiv"])

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload Research Paper or Notes",
            type=["pdf", "txt", "md"],
            help="Directly parse and index your own PDF papers, Markdown notes, or text files."
        )
        if uploaded_file is not None:
            if st.button("📥 Index Uploaded File", use_container_width=True, type="primary"):
                with st.spinner(f"Extracting sections and indexing '{uploaded_file.name}'..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        up_res = requests.post(f"{API_URL}/documents/upload", files=files, timeout=60)
                        if up_res.status_code == 200:
                            up_data = up_res.json()
                            st.success(f"Indexed **{up_data['title']}** ({up_data['chunks_indexed']} chunks)!")
                            st.rerun()
                        else:
                            st.error(f"Upload error: {up_res.text}")
                    except Exception as e:
                        st.error(f"Failed to upload: {e}")

    with tab_arxiv:
        arxiv_topic = st.text_input("Topic / Keyword", value="Retrieval Augmented Generation")
        seed_cite = st.text_input("Landmark Citation (Optional)", placeholder="e.g. 2005.11401", help="Target papers that cite a specific landmark publication.")
        narrow_kw = st.text_input("Narrow Keywords (Optional)", placeholder="e.g. hallucination, medical")

        with st.expander("🎯 Citation & Section Options"):
            min_cite_count = st.number_input("Minimum Citations", min_value=0, max_value=50000, value=0, step=10)
            max_p = st.slider("Paper Limit", min_value=1, max_value=5, value=2)
            sec_chunk = st.checkbox("Section-Aware Chunking", value=True)
            ex_ref = st.checkbox("Exclude Reference List", value=True)
            dl_pdf = st.checkbox("Download Full PDF (Slow)", value=False)

        if st.button("📥 Fetch & Index from ArXiv", use_container_width=True):
            with st.spinner("Fetching papers from ArXiv and Semantic Scholar..."):
                try:
                    payload = {
                        "query": arxiv_topic,
                        "seed_citation": seed_cite.strip() if seed_cite else None,
                        "narrow_keywords": narrow_kw.strip() if narrow_kw else None,
                        "min_citations": int(min_cite_count),
                        "max_papers": max_p,
                        "section_aware": sec_chunk,
                        "exclude_references": ex_ref,
                        "download_pdf": dl_pdf
                    }
                    res = requests.post(f"{API_URL}/ingest", json=payload, timeout=120)
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"Indexed {data['chunks_indexed']} chunks from {data['papers_fetched']} papers!")
                        st.rerun()
                    else:
                        st.error(f"Ingestion error: {res.text}")
                except Exception as e:
                    st.error(f"Failed to ingest: {e}")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # Memory-Efficient Document List (Reads lightweight manifest metadata only)
    # --------------------------------------------------------------------------
    st.markdown("#### 📑 Indexed Sources Library")
    st.caption("💡 *Memory-efficient catalog: Metadata loads instantly without loading heavy full texts into RAM.*")

    try:
        doc_res = requests.get(f"{API_URL}/documents", timeout=5)
        documents = doc_res.json() if doc_res.status_code == 200 else []
    except Exception:
        documents = []

    if not documents:
        st.info("No sources indexed yet. Upload a file above or fetch papers from ArXiv!")
    else:
        st.write(f"Total Sources: **{len(documents)}**")

        for doc in documents:
            doc_id = doc["doc_id"]
            title = doc["title"]
            cite_count = doc.get("citation_count", 0)
            chunk_count = doc.get("chunk_count", 0)
            sections = doc.get("sections", [])
            src_type = doc.get("source_type", "pdf").upper()

            with st.container(border=True):
                col_info, col_actions = st.columns([3, 1.2])

                with col_info:
                    # Checkbox to filter chat scope to this specific document
                    is_selected = st.checkbox(
                        f"**{title}**",
                        key=f"chk_{doc_id}",
                        value=(doc_id in st.session_state.selected_doc_ids),
                        help="Check to focus conversation specifically on this source."
                    )
                    if is_selected and doc_id not in st.session_state.selected_doc_ids:
                        st.session_state.selected_doc_ids.append(doc_id)
                    elif not is_selected and doc_id in st.session_state.selected_doc_ids:
                        st.session_state.selected_doc_ids.remove(doc_id)

                    cite_info = f" • ⭐ {cite_count:,} citations" if cite_count > 0 else ""
                    st.caption(f"`{src_type}` • {chunk_count} chunks{cite_info}")

                    if sections:
                        sec_badges = " ".join([f"`{s}`" for s in sections[:4]])
                        st.markdown(f"🏷️ {sec_badges}")

                with col_actions:
                    if st.button("👁️ View", key=f"view_{doc_id}", use_container_width=True):
                        st.session_state.inspect_doc_id = doc_id

                    if st.button("🗑️", key=f"del_{doc_id}", help="Delete source"):
                        try:
                            requests.delete(f"{API_URL}/documents/{doc_id}")
                            if doc_id in st.session_state.selected_doc_ids:
                                st.session_state.selected_doc_ids.remove(doc_id)
                            if st.session_state.inspect_doc_id == doc_id:
                                st.session_state.inspect_doc_id = None
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

    # --------------------------------------------------------------------------
    # On-Demand Document Content Inspector (Lazy Loaded)
    # --------------------------------------------------------------------------
    if st.session_state.inspect_doc_id:
        st.markdown("---")
        with st.spinner("Loading document text on demand from disk..."):
            try:
                detail_res = requests.get(f"{API_URL}/documents/{st.session_state.inspect_doc_id}", timeout=5)
                if detail_res.status_code == 200:
                    detail = detail_res.json()
                    summary = detail["summary"]
                    full_text = detail["full_text"]

                    with st.expander(f"📖 Inspecting: {summary['title']}", expanded=True):
                        col_close, _ = st.columns([1, 4])
                        with col_close:
                            if st.button("✖️ Close Inspector", key="close_inspector"):
                                st.session_state.inspect_doc_id = None
                                st.rerun()

                        st.markdown(f"**Document ID:** `{summary['doc_id']}`")
                        st.markdown(f"**Authors:** {summary.get('authors', 'Unknown')}")
                        st.markdown(f"**Indexed Sections:** {', '.join(summary.get('sections', []))}")
                        st.markdown(f"**Word Count:** ~{len(full_text.split()):,} words ({len(full_text):,} characters)")

                        st.markdown("##### Full Document Content:")
                        st.text_area(
                            "Document Text",
                            value=full_text,
                            height=350,
                            disabled=True,
                            key=f"text_area_{summary['doc_id']}"
                        )
            except Exception as e:
                st.error(f"Error loading document content: {e}")

# ------------------------------------------------------------------------------
# COLUMN 2: NotebookLM Conversational Studio
# ------------------------------------------------------------------------------
with col_studio:
    st.subheader("💬 Conversational Studio")

    # Source Scope Badge
    if st.session_state.selected_doc_ids:
        active_names = [d["title"] for d in documents if d["doc_id"] in st.session_state.selected_doc_ids]
        st.info(f"🎯 **Active Scope:** Filtering queries to **{len(active_names)}** selected sources ({', '.join(active_names[:2])}{'...' if len(active_names) > 2 else ''})")
    else:
        st.caption("🌐 **Active Scope:** Searching across all indexed sources in library.")

    # Render Conversation History
    chat_container = st.container(height=520)

    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                """
                <div style="text-align: center; color: #888; padding: 60px 20px;">
                    <h3>👋 Welcome to Research RAG Studio</h3>
                    <p>Ask a question about your indexed papers or uploaded documents.</p>
                    <p><i>Example: "What is RAG and what are its key components?", "Compare the methodologies of the selected papers"</i></p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

                    # If assistant message has source citations, render them
                    if msg.get("sources"):
                        with st.expander(f"📑 View {len(msg['sources'])} Cited Passages & Confidence Breakdown"):
                            for src in msg["sources"]:
                                sec = src.get("section", "General")
                                c_count = src.get("citation_count", 0)
                                cite_str = f" • ⭐ {c_count:,} citations" if c_count > 0 else ""
                                st.markdown(f"**Source {src['index']}:** *{src['title']}* `[🏷️ {sec}]` (Score: `{src['score']:.2f}`{cite_str})")
                                if src.get("dense_score") or src.get("bm25_score"):
                                    st.caption(f"Score Breakdown: Fused `{src['score']:.2f}` | Dense `{src.get('dense_score', 0):.2f}` | BM25 `{src.get('bm25_score', 0):.2f}`")
                                st.markdown(f"> {src['snippet']}")
                                st.markdown("---")

                    if msg.get("is_low_confidence"):
                        st.caption("⚠️ *Response generated with low retrieval confidence.*")

    # Chat Controls & Clear Button
    col_input, col_clear = st.columns([5, 1])
    with col_clear:
        if st.button("🧹 Clear", use_container_width=True, help="Clear conversation history"):
            st.session_state.messages = []
            st.rerun()

    # Chat Input Box
    user_prompt = st.chat_input("Ask a question about your papers...")

    if user_prompt:
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # Prepare Payload
        history_payload = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]
        doc_filter = st.session_state.selected_doc_ids if st.session_state.selected_doc_ids else None

        query_payload = {
            "question": user_prompt,
            "top_k": top_k,
            "use_hyde": use_hyde,
            "search_mode": mode_key,
            "score_threshold": score_threshold,
            "hybrid_alpha": 0.5,
            "history": history_payload,
            "doc_ids": doc_filter
        }

        with st.spinner("Retrieving relevant passages and synthesizing response..."):
            try:
                res = requests.post(f"{API_URL}/query", json=query_payload, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    assistant_msg = {
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data.get("sources", []),
                        "confidence_score": data.get("confidence_score", 0.0),
                        "is_low_confidence": data.get("is_low_confidence", False),
                        "filtered_out_count": data.get("filtered_out_count", 0),
                        "latency_ms": data.get("latency_ms", 0.0),
                        "search_mode": data.get("search_mode", "hybrid")
                    }
                    st.session_state.messages.append(assistant_msg)
                    st.rerun()
                else:
                    st.error(f"API Error: {res.text}")
            except Exception as e:
                st.error(f"Failed to communicate with API: {e}")
