"""Interactive Streamlit UI for the Research RAG Intelligence system."""
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Research RAG Intelligence",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Research RAG Intelligence System")
st.caption("End-to-end scientific literature ingestion, grounded retrieval, HyDE query expansion, and LLM-as-a-Judge evaluations.")

# Sidebar: System status & settings
with st.sidebar:
    st.header("⚙️ System Status")
    try:
        health_res = requests.get(f"{API_URL}/health", timeout=3)
        if health_res.status_code == 200:
            hdata = health_res.json()
            st.success(f"API Online ({hdata['environment']})")
            st.metric("Total Indexed Chunks", hdata["indexed_documents_count"])
            st.info(f"LLM Provider: `{hdata['llm_provider']}`")
        else:
            st.error("API returned error.")
    except Exception:
        st.warning("⚠️ API is offline. Start FastAPI backend with:\n`uvicorn src.api.main:app --reload`")

    st.markdown("---")
    st.subheader("📚 Ingest ArXiv Literature")
    ingest_topic = st.text_input("Topic / Keyword", value="Retrieval Augmented Generation")
    max_papers = st.slider("Number of papers", min_value=1, max_value=5, value=2)
    download_pdf = st.checkbox("Download & Parse Full PDF (Slow)", value=False)
    st.caption("💡 Leave PDF parsing unchecked for instant abstract indexing.")

    if st.button("📥 Fetch & Index Papers", use_container_width=True):
        with st.spinner("Fetching papers from ArXiv and indexing into vector store..."):
            try:
                res = requests.post(
                    f"{API_URL}/ingest",
                    json={"query": ingest_topic, "max_papers": max_papers, "download_pdf": download_pdf},
                    timeout=180
                )
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"Indexed {data['chunks_indexed']} chunks from {data['papers_fetched']} papers!")
                    for title in data["paper_titles"]:
                        st.write(f"- {title}")
                    st.rerun()
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")

    st.markdown("---")
    if st.button("🗑️ Reset Vector Database", use_container_width=True):
        try:
            requests.post(f"{API_URL}/reset")
            st.warning("Index cleared.")
            st.rerun()
        except Exception as e:
            st.error(str(e))

# Main Tabs
tab_query, tab_eval = st.tabs(["💬 Literature Q&A", "📊 Automated RAG Evals (LLM Judge)"])

with tab_query:
    st.subheader("Ask a Research Question")
    user_query = st.text_input(
        "Question",
        placeholder="e.g. What is RAG and how does it work?"
    )
    col_k, col_hyde = st.columns([1, 1])
    with col_k:
        top_k = st.slider("Top-K retrieved documents", min_value=1, max_value=8, value=3)
    with col_hyde:
        use_hyde = st.checkbox("🔮 Use HyDE (Hypothetical Document Embeddings)", value=True, help="Synthesizes a hypothetical academic passage before vector search to dramatically improve semantic alignment.")

    if st.button("🔍 Generate Grounded Answer", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving sources, verifying intent, and synthesizing answer..."):
                try:
                    res = requests.post(
                        f"{API_URL}/query",
                        json={"question": user_query, "top_k": top_k, "use_hyde": use_hyde},
                        timeout=45
                    )
                    if res.status_code == 200:
                        data = res.json()

                        # Display Confidence Alert if retrieval is weak or narrow
                        if data.get("is_low_confidence"):
                            st.warning(f"⚠️ **Relevance Gating Alert:** {data.get('warning_note')}")

                        st.markdown(
                            f"**Latency:** `{data['latency_ms']} ms` | "
                            f"**Retrieved Chunks:** `{data['total_retrieved']}` | "
                            f"**Top Match Score:** `{data.get('confidence_score', 0):.2f}`"
                        )
                        
                        if data.get("hyde_passage"):
                            with st.expander("🔮 View HyDE Hypothetical Passage Used for Retrieval"):
                                st.info(data["hyde_passage"])

                        st.markdown("### Answer")
                        st.write(data["answer"])

                        st.markdown("### 📑 Verified Source Citations")
                        if not data["sources"]:
                            st.info("No sources retrieved.")
                        else:
                            for src in data["sources"]:
                                with st.expander(f"Source {src['index']}: {src['title']} (Similarity: {src['score']:.2f})"):
                                    st.markdown(f"**Paper ID:** `{src['paper_id']}`")
                                    st.markdown(f"**Excerpt:** {src['snippet']}")
                                    if "pdf_url" in src["metadata"]:
                                        st.markdown(f"[View on ArXiv]({src['metadata']['pdf_url']})")
                    else:
                        st.error(f"API Error: {res.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

with tab_eval:
    st.subheader("Automated Quality Assessment (RAG Triad)")
    st.caption("Evaluated semantically using **LLM-as-a-Judge** (powered by Gemini Pro) to assess true conceptual relevance.")
    eval_query = st.text_input(
        "Query to benchmark",
        value=user_query if user_query else "What is RAG?"
    )

    if st.button("🧪 Run LLM-as-a-Judge Benchmark"):
        with st.spinner("Executing semantic RAG Triad evaluation with Gemini Pro..."):
            try:
                res = requests.post(
                    f"{API_URL}/evaluate",
                    json={"question": eval_query, "top_k": 3, "use_hyde": True},
                    timeout=60
                )
                if res.status_code == 200:
                    report = res.json()
                    st.markdown(f"**Evaluation Engine:** `{report.get('evaluation_mode', 'LLM Judge').upper()}`")
                    
                    st.markdown("### 🏆 Overall Composite Score")
                    composite = report["composite_score"]
                    st.progress(composite)
                    st.metric("Composite Score", f"{composite * 100:.1f} / 100")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("#### Context Relevance")
                        st.metric("Score", f"{report['context_relevance']['score'] * 100:.1f}%")
                        st.caption(report['context_relevance']['rationale'])

                    with col2:
                        st.markdown("#### Faithfulness (Groundedness)")
                        st.metric("Score", f"{report['faithfulness']['score'] * 100:.1f}%")
                        st.caption(report['faithfulness']['rationale'])

                    with col3:
                        st.markdown("#### Answer Relevance")
                        st.metric("Score", f"{report['answer_relevance']['score'] * 100:.1f}%")
                        st.caption(report['answer_relevance']['rationale'])

                    with st.expander("View Evaluated System Output"):
                        st.write(report["answer"])
                else:
                    st.error(f"Evaluation error: {res.text}")
            except Exception as e:
                st.error(f"Could not connect to evaluation service: {e}")
