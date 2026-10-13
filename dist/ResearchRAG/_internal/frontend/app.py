"""Antigravity-Inspired Interactive RAG Workbench.

Hierarchical Structure:
- Projects / Workspaces: Top-level domain workspaces (e.g. Biomedical RAG, Financial Forecasting).
- Project Conversations: Multi-turn chat threads residing inside the active project.
- Project Knowledge Base: Documents & full-text contexts stored online in PostgreSQL under the active project.
- Real-time Cloud Persistence: All conversation turns, citations, and file contexts persist directly to Supabase.
"""
import os
import re
from datetime import datetime
from typing import Optional, List, Dict, Any
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Antigravity Research Studio",
    page_icon="🪐",
    layout="wide"
)

# ==============================================================================
# Session State Initialization
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "inspect_doc_id" not in st.session_state:
    st.session_state.inspect_doc_id = None
if "selected_doc_ids" not in st.session_state:
    st.session_state.selected_doc_ids = []
if "gdrive_folder_id" not in st.session_state:
    st.session_state.gdrive_folder_id = "root"
if "gdrive_folder_path" not in st.session_state:
    st.session_state.gdrive_folder_path = [("root", "My Drive")]
if "gdrive_search_query" not in st.session_state:
    st.session_state.gdrive_search_query = ""
if "login_in_progress" not in st.session_state:
    st.session_state.login_in_progress = False
if "login_auth_url" not in st.session_state:
    st.session_state.login_auth_url = None
if "gdrive_folders_page" not in st.session_state:
    st.session_state.gdrive_folders_page = 1
if "gdrive_files_page" not in st.session_state:
    st.session_state.gdrive_files_page = 1
if "gdrive_page_size" not in st.session_state:
    st.session_state.gdrive_page_size = 5
if "gdrive_sort_by" not in st.session_state:
    st.session_state.gdrive_sort_by = "mod_desc"
if "gdrive_selected_ids" not in st.session_state:
    st.session_state.gdrive_selected_ids = []
if "sources_page" not in st.session_state:
    st.session_state.sources_page = 1
if "sources_page_size" not in st.session_state:
    st.session_state.sources_page_size = 5
if "active_workspace_id" not in st.session_state:
    st.session_state.active_workspace_id = None
if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = None
if "loaded_conversation_id" not in st.session_state:
    st.session_state.loaded_conversation_id = None
if "show_new_ws_dialog" not in st.session_state:
    st.session_state.show_new_ws_dialog = False

# ==============================================================================
# Fetch Workspaces & Synchronize Active Project
# ==============================================================================
workspaces = []
try:
    ws_res = requests.get(f"{API_URL}/workspaces", timeout=6)
    if ws_res.status_code == 200:
        workspaces = ws_res.json().get("workspaces", [])
except Exception:
    pass

if workspaces:
    ws_ids = [w["id"] for w in workspaces]
    
    # Synchronize if selectbox in sidebar was changed
    sel_ws = st.session_state.get("sidebar_project_selector")
    if sel_ws and sel_ws in ws_ids and sel_ws != st.session_state.active_workspace_id:
        st.session_state.active_workspace_id = sel_ws
        st.session_state.active_conversation_id = None
        st.session_state.loaded_conversation_id = None
        st.session_state.messages = []

    if st.session_state.active_workspace_id not in ws_ids:
        # Default to first workspace or one with active conversations
        st.session_state.active_workspace_id = ws_ids[0]
        st.session_state.active_conversation_id = None
        st.session_state.loaded_conversation_id = None
        st.session_state.messages = []

    active_ws = next((w for w in workspaces if w["id"] == st.session_state.active_workspace_id), workspaces[0])
else:
    active_ws = {"id": "default", "name": "General Research", "conversations": []}

threads = active_ws.get("conversations", [])
thread_ids = [t["id"] for t in threads]
if threads:
    if st.session_state.active_conversation_id not in thread_ids:
        st.session_state.active_conversation_id = thread_ids[0]
        st.session_state.loaded_conversation_id = None
        st.session_state.messages = []
else:
    st.session_state.active_conversation_id = None
    st.session_state.loaded_conversation_id = None
    st.session_state.messages = []

# If conversation thread changed, not loaded, or messages are empty while thread has messages, fetch from database
should_fetch_messages = False
if st.session_state.active_conversation_id:
    if st.session_state.loaded_conversation_id != st.session_state.active_conversation_id:
        should_fetch_messages = True
    elif not st.session_state.messages and any(t.get("id") == st.session_state.active_conversation_id and t.get("message_count", 0) > 0 for t in threads):
        should_fetch_messages = True

if should_fetch_messages:
    try:
        c_detail_res = requests.get(f"{API_URL}/conversations/{st.session_state.active_conversation_id}", timeout=10)
        if c_detail_res.status_code == 200:
            conv_data = c_detail_res.json()
            st.session_state.messages = conv_data.get("messages", [])
            st.session_state.loaded_conversation_id = st.session_state.active_conversation_id
    except Exception:
        pass

# ==============================================================================
# SIDEBAR: Antigravity Project Navigator & Conversations Hub
# ==============================================================================
with st.sidebar:
    st.markdown("### 🪐 Antigravity Studio")
    st.caption("Project-driven research grounded in online cloud knowledge.")
    st.markdown("---")

    # 1. Project Switcher
    st.markdown("#### 📁 Active Project")
    if workspaces:
        ws_options = {w["id"]: f"📁 {w['name']}" for w in workspaces}
        curr_ws_idx = list(ws_options.keys()).index(st.session_state.active_workspace_id) if st.session_state.active_workspace_id in ws_options else 0
        
        def on_project_selector_change():
            sel = st.session_state.get("sidebar_project_selector")
            if sel and sel != st.session_state.get("active_workspace_id"):
                st.session_state.active_workspace_id = sel
                st.session_state.active_conversation_id = None
                st.session_state.loaded_conversation_id = None
                st.session_state.messages = []
                st.session_state.selected_doc_ids = []
                st.session_state.inspect_doc_id = None

        chosen_ws = st.selectbox(
            "Select Project",
            options=list(ws_options.keys()),
            format_func=lambda wid: ws_options.get(wid, wid),
            index=curr_ws_idx,
            key="sidebar_project_selector",
            on_change=on_project_selector_change,
            label_visibility="collapsed"
        )
        if chosen_ws != st.session_state.active_workspace_id:
            st.session_state.active_workspace_id = chosen_ws
            st.session_state.active_conversation_id = None
            st.session_state.loaded_conversation_id = None
            st.session_state.messages = []
            st.session_state.selected_doc_ids = []
            st.session_state.inspect_doc_id = None
            st.rerun()

    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        if st.button("➕ New Project", use_container_width=True, key="btn_new_project_sidebar"):
            st.session_state.show_new_ws_dialog = not st.session_state.show_new_ws_dialog
            st.rerun()
    with col_p2:
        is_default = (st.session_state.active_workspace_id == "default")
        if st.button("🗑️", key="btn_del_project_sidebar", help="Delete this project", disabled=is_default, use_container_width=True):
            try:
                requests.delete(f"{API_URL}/workspaces/{st.session_state.active_workspace_id}", timeout=5)
                st.session_state.active_workspace_id = None
                if "sidebar_project_selector" in st.session_state:
                    st.session_state.sidebar_project_selector = "default"
                st.session_state.active_conversation_id = None
                st.session_state.loaded_conversation_id = None
                st.session_state.messages = []
                st.session_state.selected_doc_ids = []
                st.session_state.inspect_doc_id = None
                st.rerun()
            except Exception as ex:
                st.error(f"Error: {ex}")

    # Inline New Project Form
    if st.session_state.show_new_ws_dialog:
        with st.form("form_create_project_sidebar", clear_on_submit=True):
            st.markdown("##### ➕ Create New Project")
            p_name = st.text_input("Project Name", placeholder="e.g. Biomedical RAG, Financial Study")
            p_desc = st.text_input("Scope / Description", placeholder="Scope of this research project")
            sub_c1, sub_c2 = st.columns([1, 1])
            with sub_c1:
                if st.form_submit_button("Create", type="primary", use_container_width=True):
                    if p_name.strip():
                        try:
                            resp = requests.post(f"{API_URL}/workspaces", json={"name": p_name.strip(), "description": p_desc.strip()}, timeout=5)
                            if resp.status_code == 200:
                                p_data = resp.json()
                                st.session_state.active_workspace_id = p_data["id"]
                                st.session_state.sidebar_project_selector = p_data["id"]
                                st.session_state.active_conversation_id = p_data["conversations"][0]["id"] if p_data.get("conversations") else None
                                st.session_state.loaded_conversation_id = None
                                st.session_state.messages = []
                                st.session_state.selected_doc_ids = []
                                st.session_state.inspect_doc_id = None
                                st.session_state.show_new_ws_dialog = False
                                st.rerun()
                        except Exception as ex:
                            st.error(str(ex))
            with sub_c2:
                if st.form_submit_button("Cancel", use_container_width=True):
                    st.session_state.show_new_ws_dialog = False
                    st.rerun()

    st.markdown("---")

    # 2. Project Conversations (Threads)
    col_th_head, col_th_btn = st.columns([2.5, 1.5])
    with col_th_head:
        st.markdown(f"#### 💬 Chats ({len(threads)})")
    with col_th_btn:
        if st.button("➕ New Chat", use_container_width=True, key="btn_new_chat_sidebar"):
            if st.session_state.active_workspace_id:
                try:
                    c_res = requests.post(
                        f"{API_URL}/workspaces/{st.session_state.active_workspace_id}/conversations",
                        json={"title": f"Conversation {len(threads) + 1}"},
                        timeout=5
                    )
                    if c_res.status_code == 200:
                        new_c = c_res.json()
                        st.session_state.active_conversation_id = new_c["id"]
                        st.session_state.loaded_conversation_id = new_c["id"]
                        st.session_state.messages = []
                        st.rerun()
                except Exception as ex:
                    st.error(f"Error: {ex}")

    # List conversations for active project
    if not threads:
        st.caption("No conversations in this project yet.")
    else:
        for t in threads:
            is_active = (t["id"] == st.session_state.active_conversation_id)
            prefix = "👉 " if is_active else "🗨️ "
            msg_cnt = t.get("message_count", 0)
            btn_label = f"{prefix}{t['title']} ({msg_cnt})"

            c_tb1, c_tb2 = st.columns([4, 1])
            with c_tb1:
                if st.button(
                    btn_label,
                    key=f"btn_thread_{t['id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    st.session_state.active_conversation_id = t["id"]
                    st.session_state.loaded_conversation_id = None
                    st.session_state.messages = []
                    st.rerun()
            with c_tb2:
                if len(threads) > 1:
                    if st.button("🗑️", key=f"del_thread_{t['id']}", help="Delete chat thread"):
                        try:
                            requests.delete(f"{API_URL}/conversations/{t['id']}", timeout=5)
                            if is_active:
                                st.session_state.active_conversation_id = None
                                st.session_state.loaded_conversation_id = None
                                st.session_state.messages = []
                            st.rerun()
                        except Exception:
                            pass

    st.markdown("---")

    # 3. Cloud Database & Engine Settings
    with st.expander("⚙️ Engine & Database Settings", expanded=False):
        try:
            health_res = requests.get(f"{API_URL}/health", timeout=3)
            if health_res.status_code == 200:
                hdata = health_res.json()
                st.success(f"Backend Online ({hdata['environment']})")
                st.info(f"Storage: `{hdata.get('database_engine', 'PostgreSQL')}`")
                st.caption(f"Active Model: `{hdata['llm_provider']}`")
            else:
                st.error("API returned error.")
        except Exception:
            st.warning("⚠️ API is offline.")

        PROVIDER_OPTIONS = {
            "Google Gemini": {
                "key": "gemini",
                "default_model": "gemini-3.5-flash-lite",
                "models": ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-latest", "gemini-3.8-flash"],
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
            "Qwen (Alibaba)": {
                "key": "qwen",
                "default_model": "qwen-plus",
                "models": ["qwen-plus", "qwen-turbo", "qwen-max", "qwen2.5-72b-instruct"],
                "default_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                "key_prompt": "DashScope Key"
            },
            "GLM (Zhipu AI)": {
                "key": "glm",
                "default_model": "glm-4-flash",
                "models": ["glm-4-flash", "glm-4-plus", "glm-4-air"],
                "default_url": "https://open.bigmodel.cn/api/paas/v4",
                "key_prompt": "Zhipu AI Key"
            },
            "Ollama (Local)": {
                "key": "ollama",
                "default_model": "qwen2.5:7b",
                "models": ["qwen2.5:7b", "llama3", "mistral", "deepseek-r1:7b"],
                "default_url": "http://localhost:11434/v1",
                "key_prompt": "No key needed"
            },
            "Mock Engine": {
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
        chosen_model = model_choice if model_choice != "Custom..." else st.text_input("Model identifier", value=provider_info["default_model"])

        entered_api_key = st.text_input(
            provider_info["key_prompt"],
            type="password"
        ) if provider_info["key"] not in ["mock", "ollama"] else ""

        if st.button("⚡ Switch Model", use_container_width=True):
            try:
                requests.post(f"{API_URL}/config/llm", json={
                    "provider": provider_info["key"],
                    "model_name": chosen_model,
                    "api_key": entered_api_key or None
                }, timeout=10)
                st.success(f"Switched to {chosen_model}!")
                st.rerun()
            except Exception as e:
                st.error(str(e))

        st.markdown("---")
        search_strategy = st.selectbox(
            "Search Strategy",
            options=["Hybrid (Dense + BM25)", "Dense Only (ChromaDB)", "Sparse Only (BM25)"],
            index=0
        )
        mode_key = "hybrid" if "Hybrid" in search_strategy else ("dense" if "Dense" in search_strategy else "sparse")
        score_threshold = st.slider("Quality Cutoff", 0.0, 0.8, 0.35, 0.05)
        top_k = st.slider("Top Chunks", 1, 8, 3)
        use_hyde = st.checkbox("🔮 HyDE Query Expansion", value=True)

# Set defaults if not defined in sidebar
if 'mode_key' not in locals():
    mode_key = "hybrid"
if 'score_threshold' not in locals():
    score_threshold = 0.35
if 'top_k' not in locals():
    top_k = 3
if 'use_hyde' not in locals():
    use_hyde = True

# ==============================================================================
# MAIN WORKSPACE: Project Header & Two-Column Layout
# ==============================================================================
active_project_name = active_ws.get("name", "General Research")
active_thread_obj = next((t for t in threads if t["id"] == st.session_state.active_conversation_id), None)
active_thread_title = active_thread_obj["title"] if active_thread_obj else "Chat"

col_header_title, col_header_badge = st.columns([3.5, 1.5])
with col_header_title:
    st.markdown(f"### 📁 {active_project_name} › 💬 {active_thread_title}")
    st.caption(f"Project Description: *{active_ws.get('description', 'General research workspace')}*")
with col_header_badge:
    st.markdown(
        "<div style='text-align: right; padding-top: 8px;'><span style='background-color: #2e7d32; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.82em;'>☁️ Cloud PostgreSQL Synced</span></div>",
        unsafe_allow_html=True
    )

st.markdown("---")

col_files, col_studio = st.columns([1.1, 1.9], gap="large")

# ==============================================================================
# COLUMN 1: Project Knowledge Base (Files Stored Online in Database)
# ==============================================================================
with col_files:
    # Fetch documents for active workspace
    project_docs = []
    try:
        curr_wid = st.session_state.active_workspace_id or "default"
        docs_res = requests.get(f"{API_URL}/documents?workspace_id={curr_wid}", timeout=5)
        if docs_res.status_code == 200:
            project_docs = docs_res.json()
    except Exception:
        project_docs = []

    st.markdown(f"#### 📚 Project Knowledge Files ({len(project_docs)})")
    st.caption("All files and sections are stored online in PostgreSQL under this project.")

    # --------------------------------------------------------------------------
    # Add Files to Project Expander
    # --------------------------------------------------------------------------
    with st.expander("➕ Add Files to Project", expanded=False):
        tab_up, tab_arx, tab_gd, tab_cls = st.tabs(["📤 Upload", "🌐 ArXiv", "☁️ Google Drive", "🎓 Classroom"])

        # Tab 1: Upload File
        with tab_up:
            up_file = st.file_uploader(
                "Upload Paper or Notes",
                type=["pdf", "txt", "md"],
                key="uploader_input",
                help="Stores file text and metadata directly into PostgreSQL under this project."
            )
            if up_file is not None:
                if st.button("📥 Index into Project", use_container_width=True, type="primary"):
                    with st.spinner(f"Indexing '{up_file.name}' into project..."):
                        try:
                            f_data = {"file": (up_file.name, up_file.getvalue(), up_file.type)}
                            post_data = {"workspace_id": st.session_state.active_workspace_id or "default"}
                            up_res = requests.post(f"{API_URL}/documents/upload", files=f_data, data=post_data, timeout=60)
                            if up_res.status_code == 200:
                                res_json = up_res.json()
                                st.success(f"Indexed **{res_json['title']}** into project database!")
                                st.rerun()
                            else:
                                st.error(f"Upload error: {up_res.text}")
                        except Exception as ex:
                            st.error(f"Error: {ex}")

        # Tab 2: ArXiv Fetch
        with tab_arx:
            arx_scope = st.radio(
                "Content Scope",
                options=["📥 Full Paper (Complete PDF)", "📄 Abstract Only (Quick)"],
                index=0,
                horizontal=True,
                help="Full Paper downloads and parses complete PDF sections (Introduction, Methods, Results, Discussion). Abstract Only fetches summary for quick indexing.",
                key="arx_scope_in"
            )
            download_pdf_val = (arx_scope == "📥 Full Paper (Complete PDF)")

            arx_q = st.text_input("ArXiv Topic, Title or ID", value="Retrieval Augmented Generation", help="Enter a search topic, paper title, or direct ArXiv ID like 2005.11401", key="arx_query_in")
            arx_seed = st.text_input("Landmark Seed Citation (Optional)", placeholder="e.g. 2005.11401 to find citing papers", help="Specify a seminal paper to discover forward citations citing it", key="arx_seed_in")
            arx_kw = st.text_input("Narrow Keywords", placeholder="e.g. reasoning, medical, hallucination", key="arx_kw_in")
            col_a1, col_a2 = st.columns([1, 1])
            with col_a1:
                arx_limit = st.slider("Paper Limit", 1, 5, 2, key="arx_limit_in")
            with col_a2:
                arx_min_c = st.number_input("Min Citations", 0, 50000, 0, step=10, key="arx_min_c_in")

            if st.button("📥 Fetch into Project", use_container_width=True, key="btn_fetch_arx"):
                spinner_msg = "Downloading complete PDF(s) and indexing all sections into project..." if download_pdf_val else "Fetching paper abstracts into project database..."
                with st.spinner(spinner_msg):
                    try:
                        payload = {
                            "query": arx_q,
                            "seed_citation": arx_seed.strip() if arx_seed else None,
                            "narrow_keywords": arx_kw.strip() if arx_kw else None,
                            "min_citations": int(arx_min_c),
                            "max_papers": arx_limit,
                            "download_pdf": download_pdf_val,
                            "workspace_id": st.session_state.active_workspace_id or "default",
                            "section_aware": True,
                            "exclude_references": True
                        }
                        arx_res = requests.post(f"{API_URL}/ingest", json=payload, timeout=180)
                        if arx_res.status_code == 200:
                            data = arx_res.json()
                            num_chunks = data.get("chunks_indexed", 0)
                            st.success(f"Indexed {data['papers_fetched']} paper(s) ({num_chunks} chunks) into project!")
                            st.rerun()
                        else:
                            st.error(f"Ingestion error: {arx_res.text}")
                    except Exception as ex:
                        st.error(f"Error: {ex}")

        # Tab 3: Google Drive
        with tab_gd:
            @st.fragment
            def render_gdrive_tab(active_wid: str):
                try:
                    gd_stat_res = requests.get(f"{API_URL}/gdrive/status", timeout=4)
                    gd_stat = gd_stat_res.json() if gd_stat_res.status_code == 200 else {"connected": False, "message": "Offline"}
                except Exception:
                    gd_stat = {"connected": False, "message": "Offline"}

                st.caption(f"Google Drive Status: `{'🟢 Connected' if gd_stat.get('connected') else '⚪ Disconnected'}`")

                # Quick Account Switcher & Connect
                try:
                    acc_res = requests.get(f"{API_URL}/gdrive/accounts", timeout=4)
                    acc_list = acc_res.json().get("accounts", []) if acc_res.status_code == 200 else []
                except Exception:
                    acc_list = []

                if acc_list:
                    active_ac = gd_stat.get("account_email")
                    em_list = [a["email"] for a in acc_list]
                    sel_em = st.selectbox("Saved Google Account", options=em_list, index=em_list.index(active_ac) if active_ac in em_list else 0, key="gd_acc_sel")
                    if st.button("Switch Account", use_container_width=True, key="btn_switch_acc"):
                        requests.post(f"{API_URL}/gdrive/accounts/switch", json={"email": sel_em}, timeout=10)
                        st.rerun(scope="fragment")

                col_g1, col_g2 = st.columns([1, 1])
                with col_g1:
                    if st.button("➕ Connect Google", use_container_width=True, key="btn_conn_gd_tab"):
                        l_res = requests.post(f"{API_URL}/gdrive/accounts/login", timeout=10)
                        if l_res.status_code == 200:
                            st.session_state.login_in_progress = True
                            st.session_state.login_auth_url = l_res.json().get("auth_url")
                            st.rerun(scope="fragment")
                with col_g2:
                    if st.button("🎭 Demo Mode", use_container_width=True, key="btn_demo_gd_tab"):
                        requests.post(f"{API_URL}/gdrive/connect", json={"auth_type": "demo"}, timeout=10)
                        st.session_state.gdrive_folder_id = "root"
                        st.session_state.gdrive_folder_path = [("root", "My Drive")]
                        st.session_state.gdrive_search_query = ""
                        st.rerun(scope="fragment")

                if st.session_state.get("login_in_progress") and st.session_state.get("login_auth_url"):
                    st.link_button("🌐 Authorize Google Account", st.session_state["login_auth_url"], type="primary", use_container_width=True)
                    if st.button("🔄 Complete Sign-in", use_container_width=True, key="btn_comp_signin"):
                        st.session_state.login_in_progress = False
                        st.rerun(scope="fragment")

                if not gd_stat.get("connected"):
                    return

                st.markdown("---")

                # --------------------------------------------------------------
                # Search Google Drive
                # --------------------------------------------------------------
                c_sq, c_sbtn = st.columns([3.2, 1.2])
                with c_sq:
                    gd_q_val = st.text_input(
                        "Search Google Drive",
                        value=st.session_state.get("gdrive_search_query", ""),
                        placeholder="🔍 Search Drive (filename or content)...",
                        key="gd_search_input_field",
                        label_visibility="collapsed"
                    )
                with c_sbtn:
                    if gd_q_val.strip():
                        if st.button("✖️ Clear", key="btn_clear_gd_s", use_container_width=True):
                            st.session_state.gdrive_search_query = ""
                            st.rerun(scope="fragment")
                    else:
                        if st.button("🔍 Search", key="btn_run_gd_s", use_container_width=True):
                            st.session_state.gdrive_search_query = gd_q_val.strip()
                            st.rerun(scope="fragment")

                # --------------------------------------------------------------
                # Sorting & Helpers
                # --------------------------------------------------------------
                def sort_drive_items(items, sort_key):
                    if sort_key == "mod_desc":
                        with_dt = [it for it in items if it.get("modified_time") or it.get("modified")]
                        no_dt = [it for it in items if not (it.get("modified_time") or it.get("modified"))]
                        with_dt.sort(key=lambda it: str(it.get("modified_time") or it.get("modified")), reverse=True)
                        no_dt.sort(key=lambda it: (it.get("name") or "").lower())
                        return with_dt + no_dt
                    elif sort_key == "mod_asc":
                        with_dt = [it for it in items if it.get("modified_time") or it.get("modified")]
                        no_dt = [it for it in items if not (it.get("modified_time") or it.get("modified"))]
                        with_dt.sort(key=lambda it: str(it.get("modified_time") or it.get("modified")), reverse=False)
                        no_dt.sort(key=lambda it: (it.get("name") or "").lower())
                        return with_dt + no_dt
                    elif sort_key == "name_asc":
                        return sorted(items, key=lambda it: (it.get("name") or "").lower())
                    elif sort_key == "name_desc":
                        return sorted(items, key=lambda it: (it.get("name") or "").lower(), reverse=True)
                    return items

                def format_drive_date(dt_str: Optional[str]) -> str:
                    if not dt_str:
                        return "No date"
                    try:
                        clean = dt_str.replace("Z", "+00:00")
                        dt = datetime.fromisoformat(clean)
                        return dt.strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        return str(dt_str)[:16].replace("T", " ")

                # Sort selector
                c_sort_lbl, c_sort_ctrl = st.columns([1.6, 2.4])
                with c_sort_lbl:
                    st.markdown("**Sort Items:**")
                with c_sort_ctrl:
                    SORT_OPTIONS = {
                        "🕒 Modified: Newest First": "mod_desc",
                        "🕒 Modified: Oldest First": "mod_asc",
                        "🔤 Name: A to Z": "name_asc",
                        "🔤 Name: Z to A": "name_desc"
                    }
                    inv_map = {v: k for k, v in SORT_OPTIONS.items()}
                    curr_label = inv_map.get(st.session_state.gdrive_sort_by, "🕒 Modified: Newest First")
                    chosen_sort_label = st.selectbox(
                        "Sort Order",
                        options=list(SORT_OPTIONS.keys()),
                        index=list(SORT_OPTIONS.keys()).index(curr_label),
                        key="gd_sort_selector",
                        label_visibility="collapsed"
                    )
                    chosen_code = SORT_OPTIONS[chosen_sort_label]
                    if chosen_code != st.session_state.gdrive_sort_by:
                        st.session_state.gdrive_sort_by = chosen_code
                        st.session_state.gdrive_folders_page = 1
                        st.session_state.gdrive_files_page = 1
                        st.rerun(scope="fragment")

                PAGE_SIZE = 5

                # --------------------------------------------------------------
                # Mode A: Search Results View (Categorized + Paginated + Sorted)
                # --------------------------------------------------------------
                active_search = gd_q_val.strip()
                if active_search:
                    st.markdown(f"**Search Results for:** *'{active_search}'*")
                    try:
                        s_res = requests.get(f"{API_URL}/gdrive/search", params={"query": active_search}, timeout=15)
                        if s_res.status_code == 200:
                            s_data = s_res.json()
                            s_folders = sort_drive_items(s_data.get("folders", []), st.session_state.gdrive_sort_by)
                            s_files = sort_drive_items(s_data.get("files", []), st.session_state.gdrive_sort_by)

                            if not s_folders and not s_files:
                                st.info(f"No files or folders found matching '{active_search}'.")
                                return

                            cat_s_f, cat_s_d = st.tabs([f"📁 Matching Folders ({len(s_folders)})", f"📄 Matching Files ({len(s_files)})"])

                            # Search Folders Category
                            with cat_s_f:
                                if not s_folders:
                                    st.info("No matching folders found.")
                                else:
                                    tot_sf_pages = max(1, (len(s_folders) + PAGE_SIZE - 1) // PAGE_SIZE)
                                    if st.session_state.gdrive_folders_page > tot_sf_pages:
                                        st.session_state.gdrive_folders_page = tot_sf_pages
                                    cur_sf_p = st.session_state.gdrive_folders_page

                                    sf_start = (cur_sf_p - 1) * PAGE_SIZE
                                    sf_end = min(sf_start + PAGE_SIZE, len(s_folders))
                                    page_sf = s_folders[sf_start:sf_end]

                                    if tot_sf_pages > 1:
                                        p_c1, p_c2, p_c3 = st.columns([1, 2, 1])
                                        with p_c1:
                                            if st.button("◀ Prev", key="btn_prev_sf", disabled=(cur_sf_p <= 1), use_container_width=True):
                                                st.session_state.gdrive_folders_page = cur_sf_p - 1
                                                st.rerun(scope="fragment")
                                        with p_c2:
                                            st.markdown(f"<div style='text-align: center; padding-top: 6px; font-size: 0.85em; color: #bbb;'>Page <b>{cur_sf_p}</b> of <b>{tot_sf_pages}</b> ({sf_start+1}-{sf_end} of {len(s_folders)})</div>", unsafe_allow_html=True)
                                        with p_c3:
                                            if st.button("Next ▶", key="btn_next_sf", disabled=(cur_sf_p >= tot_sf_pages), use_container_width=True):
                                                st.session_state.gdrive_folders_page = cur_sf_p + 1
                                                st.rerun(scope="fragment")
                                    else:
                                        st.caption(f"Showing all {len(s_folders)} matching folders (5 per page).")

                                    for sf in page_sf:
                                        with st.container(border=True):
                                            sfc1, sfc2 = st.columns([3.2, 1.1])
                                            with sfc1:
                                                st.write(f"📁 **{sf['name']}**")
                                                st.caption(f"🕒 Modified: `{format_drive_date(sf.get('modified_time') or sf.get('modified'))}`")
                                            with sfc2:
                                                if st.button("Open", key=f"s_open_{sf['id']}", use_container_width=True):
                                                    st.session_state.gdrive_folder_id = sf["id"]
                                                    st.session_state.gdrive_folder_path = [("root", "My Drive"), (sf["id"], sf["name"])]
                                                    st.session_state.gdrive_search_query = ""
                                                    st.session_state.gdrive_folders_page = 1
                                                    st.session_state.gdrive_files_page = 1
                                                    st.rerun(scope="fragment")

                            # Search Files Category
                            with cat_s_d:
                                if not s_files:
                                    st.info("No matching files found.")
                                else:
                                    tot_sd_pages = max(1, (len(s_files) + PAGE_SIZE - 1) // PAGE_SIZE)
                                    if st.session_state.gdrive_files_page > tot_sd_pages:
                                        st.session_state.gdrive_files_page = tot_sd_pages
                                    cur_sd_p = st.session_state.gdrive_files_page

                                    sd_start = (cur_sd_p - 1) * PAGE_SIZE
                                    sd_end = min(sd_start + PAGE_SIZE, len(s_files))
                                    page_sd = s_files[sd_start:sd_end]

                                    if tot_sd_pages > 1:
                                        p_c1, p_c2, p_c3 = st.columns([1, 2, 1])
                                        with p_c1:
                                            if st.button("◀ Prev", key="btn_prev_sd", disabled=(cur_sd_p <= 1), use_container_width=True):
                                                st.session_state.gdrive_files_page = cur_sd_p - 1
                                                st.rerun(scope="fragment")
                                        with p_c2:
                                            st.markdown(f"<div style='text-align: center; padding-top: 6px; font-size: 0.85em; color: #bbb;'>Page <b>{cur_sd_p}</b> of <b>{tot_sd_pages}</b> ({sd_start+1}-{sd_end} of {len(s_files)})</div>", unsafe_allow_html=True)
                                        with p_c3:
                                            if st.button("Next ▶", key="btn_next_sd", disabled=(cur_sd_p >= tot_sd_pages), use_container_width=True):
                                                st.session_state.gdrive_files_page = cur_sd_p + 1
                                                st.rerun(scope="fragment")
                                    else:
                                        st.caption(f"Showing all {len(s_files)} matching files (5 per page).")

                                    for sfile in page_sd:
                                        with st.container(border=True):
                                            f_icon = sfile.get("icon", "📄")
                                            sz_str = f" • {sfile['size'] / 1024:.1f} KB" if sfile.get("size") else ""
                                            st.write(f"{f_icon} **{sfile['name']}**{sz_str}")
                                            st.caption(f"🕒 Modified: `{format_drive_date(sfile.get('modified_time') or sfile.get('modified'))}`")

                                            fc1, fc2 = st.columns([1.5, 1])
                                            with fc1:
                                                if st.button("📥 Import", key=f"s_imp_{sfile['id']}", use_container_width=True, type="primary"):
                                                    with st.spinner(f"Importing '{sfile['name']}' into project..."):
                                                        try:
                                                            imp_res = requests.post(f"{API_URL}/gdrive/import", json={
                                                                "file_id": sfile["id"],
                                                                "title": sfile["name"].rsplit(".", 1)[0],
                                                                "workspace_id": active_wid,
                                                                "section_aware": True
                                                            }, timeout=60)
                                                            if imp_res.status_code == 200:
                                                                st.success(f"Indexed **{sfile['name']}**!")
                                                                st.rerun(scope="app")
                                                            else:
                                                                st.error(f"Import failed: {imp_res.text}")
                                                        except Exception as ex:
                                                            st.error(f"Error: {ex}")
                                            with fc2:
                                                if sfile.get("web_view_link"):
                                                    st.link_button("🔗 Link", sfile["web_view_link"], use_container_width=True)
                        else:
                            st.error(f"Search failed: {s_res.text}")
                    except Exception as ex:
                        st.error(f"Error searching Drive: {ex}")
                    return

                # --------------------------------------------------------------
                # Mode B: Normal Folder Browser (Categorized + Paginated + Sorted)
                # --------------------------------------------------------------
                path_str = " / ".join([name for _, name in st.session_state.gdrive_folder_path])
                st.markdown(f"**Path:** 📁 `{path_str}`")

                c_up, c_refresh = st.columns([1, 1])
                with c_up:
                    can_go_up = len(st.session_state.gdrive_folder_path) > 1
                    if st.button("⬆️ Up One Folder", use_container_width=True, disabled=not can_go_up, key="btn_gd_up"):
                        if can_go_up:
                            st.session_state.gdrive_folder_path.pop()
                            st.session_state.gdrive_folder_id = st.session_state.gdrive_folder_path[-1][0]
                            st.session_state.gdrive_folders_page = 1
                            st.session_state.gdrive_files_page = 1
                            st.rerun(scope="fragment")
                with c_refresh:
                    if st.button("🔄 Refresh", use_container_width=True, key="btn_gd_refresh"):
                        st.rerun(scope="fragment")

                # Direct folder jump
                c_jump_in, c_jump_btn = st.columns([3, 1])
                with c_jump_in:
                    jump_id_input = st.text_input("Jump to Folder ID/URL", placeholder="Folder ID or drive URL...", key="gd_jump_box", label_visibility="collapsed")
                with c_jump_btn:
                    if st.button("Go", use_container_width=True, key="btn_gd_jump"):
                        target_id = jump_id_input.strip()
                        if "/folders/" in target_id:
                            target_id = target_id.split("/folders/")[1].split("?")[0].split("/")[0]
                        if target_id:
                            st.session_state.gdrive_folder_id = target_id
                            st.session_state.gdrive_folder_path = [(target_id, f"Folder {target_id[:8]}...")]
                            st.session_state.gdrive_folders_page = 1
                            st.session_state.gdrive_files_page = 1
                            st.rerun(scope="fragment")

                # Fetch contents of current folder
                try:
                    curr_fid = st.session_state.gdrive_folder_id
                    b_res = requests.get(f"{API_URL}/gdrive/browse?folder_id={curr_fid}", timeout=10)
                    if b_res.status_code == 200:
                        b_data = b_res.json()
                        raw_folders = b_data.get("folders", [])
                        raw_files = b_data.get("files", [])

                        if not raw_folders and not raw_files:
                            st.info("📂 This folder is empty.")
                            return

                        # Sort according to selected sort order
                        sorted_folders = sort_drive_items(raw_folders, st.session_state.gdrive_sort_by)
                        sorted_files = sort_drive_items(raw_files, st.session_state.gdrive_sort_by)

                        # Category Tabs: Folders vs Files
                        cat_folders, cat_files = st.tabs([f"📁 Folders ({len(sorted_folders)})", f"📄 Files ({len(sorted_files)})"])

                        # ------------------------------------------------------
                        # Category 1: Subfolders Listing (5 per page)
                        # ------------------------------------------------------
                        with cat_folders:
                            if not sorted_folders:
                                st.info("📂 No subfolders in this directory.")
                            else:
                                tot_f_pages = max(1, (len(sorted_folders) + PAGE_SIZE - 1) // PAGE_SIZE)
                                if st.session_state.gdrive_folders_page > tot_f_pages:
                                    st.session_state.gdrive_folders_page = tot_f_pages
                                cur_f_p = st.session_state.gdrive_folders_page

                                f_start = (cur_f_p - 1) * PAGE_SIZE
                                f_end = min(f_start + PAGE_SIZE, len(sorted_folders))
                                page_folders = sorted_folders[f_start:f_end]

                                if tot_f_pages > 1:
                                    p_col1, p_col2, p_col3 = st.columns([1, 2, 1])
                                    with p_col1:
                                        if st.button("◀ Prev", key="btn_prev_f", disabled=(cur_f_p <= 1), use_container_width=True):
                                            st.session_state.gdrive_folders_page = cur_f_p - 1
                                            st.rerun(scope="fragment")
                                    with p_col2:
                                        st.markdown(
                                            f"<div style='text-align: center; padding-top: 6px; font-size: 0.85em; color: #bbb;'>"
                                            f"Page <b>{cur_f_p}</b> of <b>{tot_f_pages}</b> ({f_start+1}-{f_end} of {len(sorted_folders)})"
                                            f"</div>",
                                            unsafe_allow_html=True
                                        )
                                    with p_col3:
                                        if st.button("Next ▶", key="btn_next_f", disabled=(cur_f_p >= tot_f_pages), use_container_width=True):
                                            st.session_state.gdrive_folders_page = cur_f_p + 1
                                            st.rerun(scope="fragment")
                                else:
                                    st.caption(f"Showing all {len(sorted_folders)} folders (5 per page).")

                                for f in page_folders:
                                    with st.container(border=True):
                                        f_col1, f_col2 = st.columns([3.2, 1.1])
                                        with f_col1:
                                            st.write(f"📁 **{f['name']}**")
                                            st.caption(f"🕒 Modified: `{format_drive_date(f.get('modified_time') or f.get('modified'))}`")
                                        with f_col2:
                                            if st.button("Open", key=f"f_open_{f['id']}", use_container_width=True):
                                                st.session_state.gdrive_folder_id = f["id"]
                                                st.session_state.gdrive_folder_path.append((f["id"], f["name"]))
                                                st.session_state.gdrive_folders_page = 1
                                                st.session_state.gdrive_files_page = 1
                                                st.rerun(scope="fragment")

                        # ------------------------------------------------------
                        # Category 2: Documents & Files Listing (5 per page)
                        # ------------------------------------------------------
                        with cat_files:
                            if not sorted_files:
                                st.info("📄 No files in this directory.")
                            else:
                                tot_d_pages = max(1, (len(sorted_files) + PAGE_SIZE - 1) // PAGE_SIZE)
                                if st.session_state.gdrive_files_page > tot_d_pages:
                                    st.session_state.gdrive_files_page = tot_d_pages
                                cur_d_p = st.session_state.gdrive_files_page

                                d_start = (cur_d_p - 1) * PAGE_SIZE
                                d_end = min(d_start + PAGE_SIZE, len(sorted_files))
                                page_files = sorted_files[d_start:d_end]

                                # Batch import header option
                                b_c1, b_c2 = st.columns([2.5, 1.5])
                                with b_c1:
                                    st.caption(f"Showing {d_start+1}-{d_end} of {len(sorted_files)} files (5 per page).")
                                with b_c2:
                                    if st.button(f"📥 Batch Page ({len(page_files)})", key="btn_batch_page_gd", use_container_width=True, help="Batch import all 5 files on this page into project database"):
                                        with st.spinner(f"Batch indexing {len(page_files)} files..."):
                                            try:
                                                p_ids = [it["id"] for it in page_files]
                                                b_res = requests.post(f"{API_URL}/gdrive/import/batch", json={"file_ids": p_ids, "workspace_id": active_wid}, timeout=120)
                                                if b_res.status_code == 200:
                                                    b_res_data = b_res.json()
                                                    st.success(f"Indexed {b_res_data.get('indexed_count', len(p_ids))} files into project database!")
                                                    st.rerun(scope="app")
                                                else:
                                                    st.error(f"Batch import failed: {b_res.text}")
                                            except Exception as ex:
                                                st.error(str(ex))

                                if tot_d_pages > 1:
                                    p_col1, p_col2, p_col3 = st.columns([1, 2, 1])
                                    with p_col1:
                                        if st.button("◀ Prev", key="btn_prev_d", disabled=(cur_d_p <= 1), use_container_width=True):
                                            st.session_state.gdrive_files_page = cur_d_p - 1
                                            st.rerun(scope="fragment")
                                    with p_col2:
                                        st.markdown(
                                            f"<div style='text-align: center; padding-top: 6px; font-size: 0.85em; color: #bbb;'>"
                                            f"Page <b>{cur_d_p}</b> of <b>{tot_d_pages}</b> ({d_start+1}-{d_end} of {len(sorted_files)})"
                                            f"</div>",
                                            unsafe_allow_html=True
                                        )
                                    with p_col3:
                                        if st.button("Next ▶", key="btn_next_d", disabled=(cur_d_p >= tot_d_pages), use_container_width=True):
                                            st.session_state.gdrive_files_page = cur_d_p + 1
                                            st.rerun(scope="fragment")

                                for file_item in page_files:
                                    with st.container(border=True):
                                        icon = file_item.get("icon", "📄")
                                        size_str = f" • {file_item['size'] / 1024:.1f} KB" if file_item.get("size") else ""
                                        st.write(f"{icon} **{file_item['name']}**{size_str}")
                                        st.caption(f"🕒 Modified: `{format_drive_date(file_item.get('modified_time') or file_item.get('modified'))}`")

                                        fc1, fc2 = st.columns([1.5, 1])
                                        with fc1:
                                            if st.button("📥 Import into Project", key=f"b_imp_{file_item['id']}", use_container_width=True, type="primary"):
                                                with st.spinner(f"Downloading and indexing '{file_item['name']}' into project database..."):
                                                    try:
                                                        imp_res = requests.post(f"{API_URL}/gdrive/import", json={
                                                            "file_id": file_item["id"],
                                                            "title": file_item["name"].rsplit(".", 1)[0],
                                                            "workspace_id": active_wid,
                                                            "section_aware": True
                                                        }, timeout=60)
                                                        if imp_res.status_code == 200:
                                                            st.success(f"Indexed **{file_item['name']}** into project database!")
                                                            st.rerun(scope="app")
                                                        else:
                                                            st.error(f"Import failed: {imp_res.text}")
                                                    except Exception as ex:
                                                        st.error(f"Error: {ex}")
                                        with fc2:
                                            if file_item.get("web_view_link"):
                                                st.link_button("🔗 Drive Link", file_item["web_view_link"], use_container_width=True)
                    else:
                        st.error(f"Could not browse folder: {b_res.text}")
                except Exception as e:
                    st.error(f"Failed to fetch Google Drive contents: {e}")

            render_gdrive_tab(st.session_state.active_workspace_id or "default")

        # Tab 4: Google Classroom
        with tab_cls:
            @st.fragment
            def render_classroom_tab(active_wid: str):
                if "classroom_selected_course_id" not in st.session_state:
                    st.session_state.classroom_selected_course_id = None
                if "classroom_selected_items" not in st.session_state:
                    st.session_state.classroom_selected_items = []

                # 1. Status & Connectivity Validation
                try:
                    c_stat_res = requests.get(f"{API_URL}/classroom/status", timeout=6)
                    c_stat = c_stat_res.json() if c_stat_res.status_code == 200 else {
                        "connected": False, "valid": False, "message": "Offline", "auth_type": "none"
                    }
                except Exception as ex:
                    c_stat = {"connected": False, "valid": False, "message": f"Offline: {ex}", "auth_type": "none"}

                # Status Badge & Validation Indicator
                is_valid = c_stat.get("valid", False)
                auth_t = c_stat.get("auth_type", "none")
                ac_em = c_stat.get("account_email", "")
                lat = c_stat.get("latency_ms")
                lat_str = f" ({lat:.0f}ms)" if lat is not None else ""

                if is_valid:
                    stat_badge = f"🟢 Connected as {ac_em}{lat_str}"
                elif c_stat.get("error") == "ACCESS_TOKEN_SCOPE_INSUFFICIENT":
                    stat_badge = "⚠️ Insufficient Scopes (Drive Only)"
                else:
                    stat_badge = "⚪ Disconnected"

                st.caption(f"Google Classroom Status: `{stat_badge}`")

                # Show specific guidance if scopes are insufficient
                if c_stat.get("error") == "ACCESS_TOKEN_SCOPE_INSUFFICIENT":
                    st.warning(
                        f"⚠️ **Google Classroom Permissions Required**\n\n"
                        f"Account `{ac_em}` is only authorized for Google Drive and lacks Google Classroom scopes. "
                        f"Please sign in with a Google Cloud Client ID that has Classroom scopes enabled, or provide an OAuth Access Token below."
                    )

                # Account Switcher
                try:
                    c_acc_res = requests.get(f"{API_URL}/classroom/accounts", timeout=4)
                    c_acc_list = c_acc_res.json().get("accounts", []) if c_acc_res.status_code == 200 else []
                except Exception:
                    c_acc_list = []

                if c_acc_list:
                    acc_options = {}
                    for a in c_acc_list:
                        em = a["email"]
                        badge = "✅" if a.get("has_classroom_scopes") else "⚠️ Drive Only"
                        acc_options[em] = f"{badge} {em}"

                    cur_em_idx = list(acc_options.keys()).index(ac_em) if ac_em in acc_options else 0
                    sel_acc = st.selectbox(
                        "Saved Google Accounts",
                        options=list(acc_options.keys()),
                        format_func=lambda e: acc_options.get(e, e),
                        index=cur_em_idx,
                        key="cls_acc_sel"
                    )
                    c_sw_col1, c_sw_col2 = st.columns([1, 1])
                    with c_sw_col1:
                        if st.button("Switch Account", use_container_width=True, key="btn_switch_cls_acc"):
                            requests.post(f"{API_URL}/classroom/accounts/switch", json={"email": sel_acc}, timeout=10)
                            st.session_state.classroom_selected_course_id = None
                            st.session_state.classroom_selected_items = []
                            st.rerun(scope="fragment")
                    with c_sw_col2:
                        if st.button("🔍 Verify Test", use_container_width=True, key="btn_test_cls_conn"):
                            t_res = requests.get(f"{API_URL}/classroom/status", timeout=8)
                            if t_res.status_code == 200:
                                td = t_res.json()
                                if td.get("valid"):
                                    st.success(f"Verified! Latency: {td.get('latency_ms')}ms • {td.get('courses_count')} courses")
                                else:
                                    st.error(f"Validation failed: {td.get('message')}")
                            else:
                                st.error(f"Test error: {t_res.text}")

                # Authentication Controls (Access Token & Google Cloud Credentials)
                if not is_valid:
                    st.markdown("---")
                    st.markdown("#### 🔐 Google Classroom Authentication")

                    tab_auth_tok, tab_auth_oauth = st.tabs(["🔑 Access Token", "🌐 Google Cloud OAuth App"])

                    with tab_auth_tok:
                        st.caption("Connect directly using a Google OAuth 2.0 access token (e.g. from Google OAuth Playground).")
                        tok_in = st.text_input("OAuth Access Token", type="password", key="cls_tok_in", placeholder="ya29.a0...", help="Enter token with classroom.courses.readonly and classroom.announcements.readonly scopes")
                        if st.button("Connect with Access Token", type="primary", use_container_width=True, key="btn_conn_tok_cls"):
                            if tok_in.strip():
                                with st.spinner("Validating access token with Google Classroom API..."):
                                    try:
                                        c_res = requests.post(f"{API_URL}/classroom/connect", json={"auth_type": "access_token", "access_token": tok_in.strip()}, timeout=15)
                                        if c_res.status_code == 200:
                                            cd = c_res.json()
                                            if cd.get("valid"):
                                                st.success(f"Connected as **{cd.get('account_email')}**!")
                                                st.rerun(scope="fragment")
                                            else:
                                                st.error(f"Validation failed: {cd.get('message')}")
                                        else:
                                            try:
                                                err_data = c_res.json()
                                                err_msg = err_data.get("detail", c_res.text)
                                            except Exception:
                                                err_msg = c_res.text
                                            st.error(f"Connection error: {err_msg}")
                                    except Exception as ex:
                                        st.error(f"Error: {ex}")
                            else:
                                st.error("Please paste an OAuth access token.")

                        with st.expander("ℹ️ How to get an Access Token (in 30 seconds)", expanded=False):
                            st.markdown(
                                """
                                1. Open [Google OAuth 2.0 Playground](https://developers.google.com/oauthplayground).
                                2. In **Step 1**, scroll down to **Input your own scopes** and paste:
                                   ```text
                                   https://www.googleapis.com/auth/classroom.courses.readonly https://www.googleapis.com/auth/classroom.announcements.readonly https://www.googleapis.com/auth/classroom.courseworkmaterials.readonly https://www.googleapis.com/auth/drive.readonly
                                   ```
                                3. Click **Authorize APIs** and choose your Google Account.
                                4. In **Step 2**, click **Exchange authorization code for tokens** (the blue button).
                                5. Copy the **Access token** field (which starts with `ya29.`) and paste it above.
                                *(⚠️ Note: Do not copy the Authorization code starting with `4/`)*
                                """
                            )

                    with tab_auth_oauth:
                        st.caption("Sign in with your Google Account using your Google Cloud Project's OAuth 2.0 Client credentials.")
                        c_id_in = st.text_input("Google Cloud Client ID", key="cls_cid_in", placeholder="107100...apps.googleusercontent.com")
                        c_sec_in = st.text_input("Google Cloud Client Secret", type="password", key="cls_csec_in", placeholder="GOCSPX-...")
                        if st.button("🌐 Sign in with Google Classroom", type="primary", use_container_width=True, key="btn_login_oauth_cls"):
                            if c_id_in.strip() and c_sec_in.strip():
                                requests.post(f"{API_URL}/classroom/oauth-config", json={"client_id": c_id_in.strip(), "client_secret": c_sec_in.strip()}, timeout=5)
                            l_res = requests.post(
                                f"{API_URL}/classroom/accounts/login",
                                params={"custom_client_id": c_id_in.strip() or None, "custom_client_secret": c_sec_in.strip() or None},
                                timeout=10
                            )
                            if l_res.status_code == 200:
                                st.session_state.cls_login_in_progress = True
                                st.session_state.cls_login_auth_url = l_res.json().get("auth_url")
                                st.rerun(scope="fragment")
                            else:
                                st.error(f"Login error: {l_res.text}")

                    if st.session_state.get("cls_login_in_progress") and st.session_state.get("cls_login_auth_url"):
                        st.link_button("🌐 Open Google Account Approval", st.session_state["cls_login_auth_url"], type="primary", use_container_width=True)
                        if st.button("🔄 Complete Classroom Sign-in", use_container_width=True, key="btn_comp_cls_signin"):
                            st.session_state.cls_login_in_progress = False
                            st.rerun(scope="fragment")

                    return

                st.markdown("---")

                # 2. Fetch and Select Course
                try:
                    courses_res = requests.get(f"{API_URL}/classroom/courses", timeout=12)
                    courses = courses_res.json().get("courses", []) if courses_res.status_code == 200 else []
                except Exception as ex:
                    st.error(f"Failed to fetch courses: {ex}")
                    return

                if not courses:
                    st.info(f"No active courses found in Google Classroom account '{ac_em}'.")
                    return

                course_ids = [c["id"] for c in courses]
                if not st.session_state.classroom_selected_course_id or st.session_state.classroom_selected_course_id not in course_ids:
                    st.session_state.classroom_selected_course_id = course_ids[0]

                course_map = {c["id"]: c for c in courses}
                c_idx = course_ids.index(st.session_state.classroom_selected_course_id)
                selected_cid = st.selectbox(
                    "Select Classroom Course",
                    options=course_ids,
                    format_func=lambda cid: f"📚 {course_map[cid]['name']} ({course_map[cid].get('section', 'Class')})",
                    index=c_idx,
                    key="cls_course_select"
                )
                if selected_cid != st.session_state.classroom_selected_course_id:
                    st.session_state.classroom_selected_course_id = selected_cid
                    st.session_state.classroom_selected_items = []
                    st.rerun(scope="fragment")

                active_course = course_map[selected_cid]
                with st.container(border=True):
                    st.markdown(f"**{active_course['name']}**")
                    meta_parts = []
                    if active_course.get("section"): meta_parts.append(f"Section: `{active_course['section']}`")
                    if active_course.get("room"): meta_parts.append(f"Room: `{active_course['room']}`")
                    if active_course.get("course_state"): meta_parts.append(f"State: `{active_course['course_state']}`")
                    if meta_parts:
                        st.caption(" • ".join(meta_parts))
                    if active_course.get("alternate_link"):
                        st.link_button("🌐 Open Course in Classroom", active_course["alternate_link"], use_container_width=True)

                # 3. Fetch Course Stream (Announcements, Materials, Coursework)
                try:
                    stream_res = requests.get(f"{API_URL}/classroom/courses/{selected_cid}/stream", timeout=12)
                    if stream_res.status_code != 200:
                        st.error(f"Error loading stream: {stream_res.text}")
                        return
                    stream = stream_res.json()
                except Exception as ex:
                    st.error(f"Failed to load course stream: {ex}")
                    return

                announcements = stream.get("announcements", [])
                materials = stream.get("coursework_materials", [])
                coursework = stream.get("coursework", [])

                # Collect all selectable items across stream for batch operations
                all_selectable_items = []
                for a in announcements:
                    if a.get("text", "").strip():
                        all_selectable_items.append({
                            "key": f"ann_txt_{a['id']}",
                            "item_type": "announcement_text",
                            "course_id": selected_cid,
                            "course_name": active_course["name"],
                            "title": f"[{active_course['name']}] Announcement - {a.get('created_at', '')[:10]}",
                            "raw_text": a["text"],
                            "drive_file_id": None
                        })
                    for m in a.get("materials", []):
                        if m.get("type") == "drive_file":
                            all_selectable_items.append({
                                "key": f"mat_drive_{m['id']}",
                                "item_type": "drive_file",
                                "course_id": selected_cid,
                                "course_name": active_course["name"],
                                "title": m.get("title", "Attached Document"),
                                "drive_file_id": m.get("drive_file_id"),
                                "mime_type": m.get("mime_type")
                            })

                for cm in materials:
                    for m in cm.get("materials", []):
                        if m.get("type") == "drive_file":
                            all_selectable_items.append({
                                "key": f"cm_drive_{m['id']}",
                                "item_type": "drive_file",
                                "course_id": selected_cid,
                                "course_name": active_course["name"],
                                "title": m.get("title", cm.get("title")),
                                "drive_file_id": m.get("drive_file_id"),
                                "mime_type": m.get("mime_type")
                            })

                for cw in coursework:
                    for m in cw.get("materials", []):
                        if m.get("type") == "drive_file":
                            all_selectable_items.append({
                                "key": f"cw_drive_{m['id']}",
                                "item_type": "drive_file",
                                "course_id": selected_cid,
                                "course_name": active_course["name"],
                                "title": m.get("title", cw.get("title")),
                                "drive_file_id": m.get("drive_file_id"),
                                "mime_type": m.get("mime_type")
                            })

                # Batch Ingestion Bar
                sel_keys = set(st.session_state.classroom_selected_items)
                selected_objects = [it for it in all_selectable_items if it["key"] in sel_keys]

                b_col1, b_col2, b_col3 = st.columns([1.5, 1, 1])
                with b_col1:
                    st.caption(f"Selected: **{len(selected_objects)}** / {len(all_selectable_items)} document(s)")
                with b_col2:
                    if st.button("Select All", key="btn_cls_sel_all", use_container_width=True):
                        st.session_state.classroom_selected_items = [it["key"] for it in all_selectable_items]
                        st.rerun(scope="fragment")
                with b_col3:
                    if st.button("Clear", key="btn_cls_clear", use_container_width=True):
                        st.session_state.classroom_selected_items = []
                        st.rerun(scope="fragment")

                if selected_objects:
                    if st.button(f"📥 Index Selected ({len(selected_objects)}) into Project", type="primary", use_container_width=True, key="btn_cls_batch_index"):
                        with st.spinner(f"Ingesting {len(selected_objects)} documents from Classroom into project..."):
                            try:
                                payload = {
                                    "items": selected_objects,
                                    "workspace_id": active_wid,
                                    "section_aware": True,
                                    "exclude_references": True
                                }
                                imp_res = requests.post(f"{API_URL}/classroom/import/batch", json=payload, timeout=120)
                                if imp_res.status_code == 200:
                                    res_data = imp_res.json()
                                    st.success(f"Indexed {res_data.get('total_succeeded', 0)} documents into project database!")
                                    st.session_state.classroom_selected_items = []
                                    st.rerun(scope="app")
                                else:
                                    st.error(f"Import error: {imp_res.text}")
                            except Exception as ex:
                                st.error(f"Error: {ex}")

                st.markdown("---")

                # Stream Tabs: Announcements, Materials, Coursework
                c_tab_ann, c_tab_mat, c_tab_cw = st.tabs([
                    f"📢 Announcements ({len(announcements)})",
                    f"📚 Course Materials ({len(materials)})",
                    f"📝 Coursework ({len(coursework)})"
                ])

                # --------------------------------------------------------------
                # Sub-Tab 1: Announcements & Attached Files
                # --------------------------------------------------------------
                with c_tab_ann:
                    if not announcements:
                        st.info("No announcements posted in this course yet.")
                    else:
                        for ann in announcements:
                            with st.container(border=True):
                                ann_date = ann.get("created_at", "")[:10]
                                st.markdown(f"**📢 Notice ({ann_date})**")
                                st.write(ann.get("text", ""))

                                # Attached files to this announcement
                                att_mats = ann.get("materials", [])
                                if att_mats:
                                    st.markdown("##### 📎 Attached Files & Materials")
                                    for m in att_mats:
                                        m_type = m.get("type")
                                        m_title = m.get("title", "Attachment")
                                        m_key = f"mat_drive_{m['id']}"

                                        if m_type == "drive_file":
                                            icon = "📕" if "pdf" in (m.get("mime_type") or "").lower() or m_title.endswith(".pdf") else "📄"
                                            c_chk, c_info, c_act = st.columns([0.6, 2.6, 1.2])
                                            with c_chk:
                                                is_chk = st.checkbox(
                                                    "select",
                                                    value=(m_key in sel_keys),
                                                    key=f"chk_cls_{m_key}",
                                                    label_visibility="collapsed"
                                                )
                                                if is_chk and m_key not in sel_keys:
                                                    st.session_state.classroom_selected_items.append(m_key)
                                                    st.rerun(scope="fragment")
                                                elif not is_chk and m_key in sel_keys:
                                                    st.session_state.classroom_selected_items.remove(m_key)
                                                    st.rerun(scope="fragment")
                                            with c_info:
                                                st.write(f"{icon} **{m_title}**")
                                            with c_act:
                                                if st.button("Index", key=f"btn_single_{m_key}", use_container_width=True):
                                                    with st.spinner(f"Indexing '{m_title}'..."):
                                                        p_load = {
                                                            "course_id": selected_cid,
                                                            "course_name": active_course["name"],
                                                            "item_type": "drive_file",
                                                            "drive_file_id": m.get("drive_file_id"),
                                                            "title": m_title,
                                                            "mime_type": m.get("mime_type"),
                                                            "workspace_id": active_wid
                                                        }
                                                        s_res = requests.post(f"{API_URL}/classroom/import", json=p_load, timeout=60)
                                                        if s_res.status_code == 200:
                                                            st.success(f"Indexed **{m_title}**!")
                                                            st.rerun(scope="app")
                                                        else:
                                                            st.error(s_res.text)
                                        elif m_type == "link":
                                            st.markdown(f"🔗 [{m_title}]({m.get('url')})")

                                # Option to index announcement text itself
                                txt_key = f"ann_txt_{ann['id']}"
                                c_tchk, c_tbtn = st.columns([3.2, 1.2])
                                with c_tchk:
                                    is_tchk = st.checkbox(
                                        "📌 Include notice text in RAG indexing",
                                        value=(txt_key in sel_keys),
                                        key=f"chk_ann_txt_{ann['id']}"
                                    )
                                    if is_tchk and txt_key not in sel_keys:
                                        st.session_state.classroom_selected_items.append(txt_key)
                                        st.rerun(scope="fragment")
                                    elif not is_tchk and txt_key in sel_keys:
                                        st.session_state.classroom_selected_items.remove(txt_key)
                                        st.rerun(scope="fragment")
                                with c_tbtn:
                                    if st.button("Index Notice", key=f"btn_ann_txt_{ann['id']}", use_container_width=True):
                                        with st.spinner("Indexing notice..."):
                                            p_load = {
                                                "course_id": selected_cid,
                                                "course_name": active_course["name"],
                                                "item_type": "announcement_text",
                                                "title": f"[{active_course['name']}] Announcement - {ann_date}",
                                                "raw_text": ann.get("text", ""),
                                                "workspace_id": active_wid
                                            }
                                            s_res = requests.post(f"{API_URL}/classroom/import", json=p_load, timeout=60)
                                            if s_res.status_code == 200:
                                                st.success("Notice indexed into project database!")
                                                st.rerun(scope="app")
                                            else:
                                                st.error(s_res.text)

                # --------------------------------------------------------------
                # Sub-Tab 2: Course Materials (Lecture Slides, Notes)
                # --------------------------------------------------------------
                with c_tab_mat:
                    if not materials:
                        st.info("No coursework materials found.")
                    else:
                        for cm in materials:
                            with st.container(border=True):
                                st.markdown(f"**📚 {cm.get('title')}**")
                                if cm.get("description"):
                                    st.caption(cm["description"])
                                for m in cm.get("materials", []):
                                    if m.get("type") == "drive_file":
                                        m_title = m.get("title", cm.get("title"))
                                        m_key = f"cm_drive_{m['id']}"
                                        icon = "📕" if "pdf" in (m.get("mime_type") or "").lower() or m_title.endswith(".pdf") else "📄"

                                        c_chk, c_info, c_act = st.columns([0.6, 2.6, 1.2])
                                        with c_chk:
                                            is_chk = st.checkbox(
                                                "select",
                                                value=(m_key in sel_keys),
                                                key=f"chk_cls_{m_key}",
                                                label_visibility="collapsed"
                                            )
                                            if is_chk and m_key not in sel_keys:
                                                st.session_state.classroom_selected_items.append(m_key)
                                                st.rerun(scope="fragment")
                                            elif not is_chk and m_key in sel_keys:
                                                st.session_state.classroom_selected_items.remove(m_key)
                                                st.rerun(scope="fragment")
                                        with c_info:
                                            st.write(f"{icon} **{m_title}**")
                                        with c_act:
                                            if st.button("Index", key=f"btn_single_{m_key}", use_container_width=True):
                                                with st.spinner(f"Indexing '{m_title}'..."):
                                                    p_load = {
                                                        "course_id": selected_cid,
                                                        "course_name": active_course["name"],
                                                        "item_type": "drive_file",
                                                        "drive_file_id": m.get("drive_file_id"),
                                                        "title": m_title,
                                                        "mime_type": m.get("mime_type"),
                                                        "workspace_id": active_wid
                                                    }
                                                    s_res = requests.post(f"{API_URL}/classroom/import", json=p_load, timeout=60)
                                                    if s_res.status_code == 200:
                                                        st.success(f"Indexed **{m_title}**!")
                                                        st.rerun(scope="app")
                                                    else:
                                                        st.error(s_res.text)

                # --------------------------------------------------------------
                # Sub-Tab 3: Coursework & Assignments
                # --------------------------------------------------------------
                with c_tab_cw:
                    if not coursework:
                        st.info("No assignments or coursework items found.")
                    else:
                        for cw in coursework:
                            with st.container(border=True):
                                due_str = f" • Due: `{cw['due_date']}`" if cw.get("due_date") else ""
                                pts_str = f" • Points: `{cw['max_points']}`" if cw.get("max_points") else ""
                                st.markdown(f"**📝 {cw.get('title')}**{due_str}{pts_str}")
                                if cw.get("description"):
                                    st.caption(cw["description"])

                                for m in cw.get("materials", []):
                                    if m.get("type") == "drive_file":
                                        m_title = m.get("title", cw.get("title"))
                                        m_key = f"cw_drive_{m['id']}"
                                        icon = "📕" if "pdf" in (m.get("mime_type") or "").lower() or m_title.endswith(".pdf") else "📄"

                                        c_chk, c_info, c_act = st.columns([0.6, 2.6, 1.2])
                                        with c_chk:
                                            is_chk = st.checkbox(
                                                "select",
                                                value=(m_key in sel_keys),
                                                key=f"chk_cls_{m_key}",
                                                label_visibility="collapsed"
                                            )
                                            if is_chk and m_key not in sel_keys:
                                                st.session_state.classroom_selected_items.append(m_key)
                                                st.rerun(scope="fragment")
                                            elif not is_chk and m_key in sel_keys:
                                                st.session_state.classroom_selected_items.remove(m_key)
                                                st.rerun(scope="fragment")
                                        with c_info:
                                            st.write(f"{icon} **{m_title}**")
                                        with c_act:
                                            if st.button("Index", key=f"btn_single_{m_key}", use_container_width=True):
                                                with st.spinner(f"Indexing '{m_title}'..."):
                                                    p_load = {
                                                        "course_id": selected_cid,
                                                        "course_name": active_course["name"],
                                                        "item_type": "drive_file",
                                                        "drive_file_id": m.get("drive_file_id"),
                                                        "title": m_title,
                                                        "mime_type": m.get("mime_type"),
                                                        "workspace_id": active_wid
                                                    }
                                                    s_res = requests.post(f"{API_URL}/classroom/import", json=p_load, timeout=60)
                                                    if s_res.status_code == 200:
                                                        st.success(f"Indexed **{m_title}**!")
                                                        st.rerun(scope="app")
                                                    else:
                                                        st.error(s_res.text)

            render_classroom_tab(st.session_state.active_workspace_id or "default")


    # --------------------------------------------------------------------------
    # Project Files List with Scoping, Inspection & Deletion (AJAX Fragment)
    # --------------------------------------------------------------------------
    @st.fragment
    def render_knowledge_files(curr_wid: str):
        # Fetch documents for active workspace
        project_docs = []
        try:
            docs_res = requests.get(f"{API_URL}/documents?workspace_id={curr_wid}", timeout=5)
            if docs_res.status_code == 200:
                project_docs = docs_res.json()
        except Exception:
            project_docs = []

        if not project_docs:
            st.info("No files in this project yet. Use **➕ Add Files to Project** above to upload PDFs, search ArXiv, or connect Google Drive.")
            return

        # Search filter and items per page
        c_f1, c_f2 = st.columns([3, 1.5])
        with c_f1:
            f_filter = st.text_input("Filter files", placeholder="🔍 Filter project files...", key="f_filter_in", label_visibility="collapsed")
        with c_f2:
            f_ps = st.selectbox("Per page", options=[5, 10, 20], index=0, key="f_ps_select", label_visibility="collapsed")
            st.session_state.sources_page_size = f_ps

        filtered_docs = [
            d for d in project_docs
            if not f_filter or (
                f_filter.lower() in d.get("title", "").lower()
                or f_filter.lower() in d.get("authors", "").lower()
                or any(f_filter.lower() in s.lower() for s in d.get("sections", []))
            )
        ]

        total_d = len(filtered_docs)
        page_sz = st.session_state.sources_page_size
        tot_pages = max(1, (total_d + page_sz - 1) // page_sz)
        if st.session_state.sources_page > tot_pages:
            st.session_state.sources_page = tot_pages
        cur_p = st.session_state.sources_page

        p_start = (cur_p - 1) * page_sz
        p_end = min(p_start + page_sz, total_d)
        page_docs = filtered_docs[p_start:p_end]

        # Active Scope Indicator (AJAX live feedback)
        sel_in_project = [did for did in st.session_state.selected_doc_ids if did in [d["doc_id"] for d in project_docs]]
        if sel_in_project:
            st.markdown(
                f"<div style='background: rgba(33, 150, 243, 0.12); border-left: 3px solid #2196f3; padding: 6px 12px; border-radius: 4px; font-size: 0.86em; margin-bottom: 8px; color: #90caf9;'>"
                f"🎯 <b>Grounding Scope:</b> <b>{len(sel_in_project)}</b> of <b>{len(project_docs)}</b> files selected"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='background: rgba(255, 255, 255, 0.04); border-left: 3px solid #757575; padding: 6px 12px; border-radius: 4px; font-size: 0.86em; margin-bottom: 8px; color: #aaa;'>"
                "🌐 <b>Grounding Scope:</b> Searching across all files in this project"
                "</div>",
                unsafe_allow_html=True
            )

        # Selection Toolbar
        col_t1, col_t2, col_t3, col_t4 = st.columns([1.1, 1.1, 1.1, 1.7])
        with col_t1:
            if st.button("☑️ Page", key="btn_sel_page_pf", use_container_width=True, help="Select files on this page"):
                for d in page_docs:
                    did = d["doc_id"]
                    if did not in st.session_state.selected_doc_ids:
                        st.session_state.selected_doc_ids.append(did)
                    st.session_state[f"chk_f_{did}"] = True
                st.rerun(scope="fragment")

        with col_t2:
            if st.button(f"☑️ All ({total_d})", key="btn_sel_all_pf", use_container_width=True, help="Select all filtered files"):
                for d in filtered_docs:
                    did = d["doc_id"]
                    if did not in st.session_state.selected_doc_ids:
                        st.session_state.selected_doc_ids.append(did)
                    st.session_state[f"chk_f_{did}"] = True
                st.rerun(scope="fragment")

        with col_t3:
            if st.button("◻️ Clear", key="btn_desel_pf", use_container_width=True, help="Clear selection"):
                for did in list(st.session_state.selected_doc_ids):
                    st.session_state[f"chk_f_{did}"] = False
                st.session_state.selected_doc_ids = []
                st.rerun(scope="fragment")

        with col_t4:
            del_targets = [did for did in st.session_state.selected_doc_ids if did in [d["doc_id"] for d in filtered_docs]]
            if st.button(f"🗑️ ({len(del_targets)})", key="btn_batch_del_pf", disabled=(len(del_targets) == 0), use_container_width=True):
                try:
                    requests.post(f"{API_URL}/documents/batch-delete", json={"doc_ids": del_targets}, timeout=15)
                    for did in del_targets:
                        if did in st.session_state.selected_doc_ids:
                            st.session_state.selected_doc_ids.remove(did)
                        st.session_state[f"chk_f_{did}"] = False
                    if st.session_state.inspect_doc_id in del_targets:
                        st.session_state.inspect_doc_id = None
                    st.success(f"Deleted {len(del_targets)} files.")
                    st.rerun(scope="fragment")
                except Exception as ex:
                    st.error(str(ex))

        # Pagination Bar
        if tot_pages > 1:
            col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
            with col_p1:
                if st.button("◀ Prev", key="btn_pg_prev", disabled=(cur_p <= 1), use_container_width=True):
                    st.session_state.sources_page = cur_p - 1
                    st.rerun(scope="fragment")
            with col_p2:
                st.markdown(f"<div style='text-align: center; padding-top: 6px; font-size: 0.88em; color: #aaa;'>Page {cur_p} of {tot_pages}</div>", unsafe_allow_html=True)
            with col_p3:
                if st.button("Next ▶", key="btn_pg_next", disabled=(cur_p >= tot_pages), use_container_width=True):
                    st.session_state.sources_page = cur_p + 1
                    st.rerun(scope="fragment")

        # Callback for individual checkbox
        def on_checkbox_toggle(doc_id: str):
            is_chk = st.session_state.get(f"chk_f_{doc_id}", False)
            if is_chk and doc_id not in st.session_state.selected_doc_ids:
                st.session_state.selected_doc_ids.append(doc_id)
            elif not is_chk and doc_id in st.session_state.selected_doc_ids:
                st.session_state.selected_doc_ids.remove(doc_id)

        # Render File Cards
        for doc in page_docs:
            did = doc["doc_id"]
            dtitle = doc["title"]
            cite_cnt = doc.get("citation_count", 0)
            chunk_cnt = doc.get("chunk_count", 0)
            secs = doc.get("sections", [])
            stype = doc.get("source_type", "pdf").upper()

            with st.container(border=True):
                col_dinfo, col_dact = st.columns([3.2, 1.1])
                with col_dinfo:
                    # Sync widget state key with selected_doc_ids if not already in session_state
                    if f"chk_f_{did}" not in st.session_state:
                        st.session_state[f"chk_f_{did}"] = (did in st.session_state.selected_doc_ids)

                    st.checkbox(
                        f"**{dtitle}**",
                        key=f"chk_f_{did}",
                        on_change=on_checkbox_toggle,
                        args=(did,),
                        help="Select to ground conversational queries in this document."
                    )

                    cite_badge = f" • ⭐ {cite_cnt:,} citations" if cite_cnt > 0 else ""
                    st.caption(f"`{stype}` • {chunk_cnt} chunks{cite_badge}")
                    if secs:
                        st.markdown(" ".join([f"`🏷️ {s}`" for s in secs[:3]]))

                with col_dact:
                    is_inspecting = (st.session_state.get("inspect_doc_id") == did)
                    if st.button("👁️ View" if not is_inspecting else "✖️ Close", key=f"btn_view_{did}", use_container_width=True):
                        if is_inspecting:
                            st.session_state.inspect_doc_id = None
                        else:
                            st.session_state.inspect_doc_id = did
                        st.rerun(scope="fragment")

                    if st.button("🗑️", key=f"btn_delf_{did}", help="Delete file"):
                        try:
                            requests.delete(f"{API_URL}/documents/{did}", timeout=10)
                            if did in st.session_state.selected_doc_ids:
                                st.session_state.selected_doc_ids.remove(did)
                            st.session_state[f"chk_f_{did}"] = False
                            if st.session_state.inspect_doc_id == did:
                                st.session_state.inspect_doc_id = None
                            st.rerun(scope="fragment")
                        except Exception as ex:
                            st.error(str(ex))

                # Inline file inspector if opened
                if st.session_state.get("inspect_doc_id") == did:
                    with st.spinner("Fetching full file content from online PostgreSQL database..."):
                        try:
                            detail_res = requests.get(f"{API_URL}/documents/{did}", timeout=6)
                            if detail_res.status_code == 200:
                                detail = detail_res.json()
                                dsum = detail["summary"]
                                ftext = detail["full_text"]
                                st.markdown("---")
                                st.caption(f"Authors: **{dsum.get('authors', 'Unknown')}** • Sections: `{', '.join(dsum.get('sections', []))}` • Words: ~{len(ftext.split()):,} ({len(ftext):,} chars)")
                                st.text_area("Full Document Text (Online DB)", value=ftext, height=240, disabled=True, key=f"insp_ta_{did}")
                        except Exception as ex:
                            st.error(f"Error loading document: {ex}")

    render_knowledge_files(st.session_state.active_workspace_id or "default")

# ==============================================================================
# COLUMN 2: Antigravity Conversation Canvas & On-Demand Inspector
# ==============================================================================
with col_studio:
    # --------------------------------------------------------------------------
    # Scope Banner & Controls
    # --------------------------------------------------------------------------
    col_sc_info, col_sc_clear = st.columns([4, 1])
    with col_sc_info:
        if st.session_state.selected_doc_ids:
            active_names = [d["title"] for d in project_docs if d["doc_id"] in st.session_state.selected_doc_ids]
            st.info(f"🎯 **Active Grounding Scope:** Scoped to **{len(active_names)}** selected files ({', '.join(active_names[:2])}{'...' if len(active_names) > 2 else ''})")
        else:
            st.caption("🌐 **Active Grounding Scope:** Searching across all files in this project.")
    with col_sc_clear:
        if st.button("🧹 Clear Chat", use_container_width=True, help="Clear chat history for this conversation"):
            if st.session_state.active_conversation_id:
                try:
                    requests.delete(f"{API_URL}/conversations/{st.session_state.active_conversation_id}/messages", timeout=5)
                except Exception:
                    pass
            st.session_state.messages = []
            st.rerun()



    # --------------------------------------------------------------------------
    # Conversation Turns Stream
    # --------------------------------------------------------------------------
    chat_container = st.container(height=480)

    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                f"""
                <div style="text-align: center; color: #888; padding: 60px 20px;">
                    <h3>🪐 Antigravity Project Canvas</h3>
                    <p>You are chatting inside project <b>{active_project_name}</b>.</p>
                    <p>Ask questions about your project files, or compare methodologies across multiple documents.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

                    # If assistant message has source citations, render traceable citation cards
                    if msg.get("sources"):
                        valid_sources = [s for s in msg["sources"] if isinstance(s, dict)]
                        if valid_sources:
                            with st.expander(f"📑 View {len(valid_sources)} Cited Passages & Traceable Badges"):
                                for src in valid_sources:
                                    sec = src.get("section", "General")
                                    c_count = src.get("citation_count", 0) or 0
                                    cite_str = f" • ⭐ {c_count:,} citations" if c_count > 0 else ""
                                    idx_val = src.get("index", "?")
                                    title_val = src.get("title", "Untitled Document")
                                    raw_sc = src.get("score")
                                    sc_fmt = f"{float(raw_sc):.2f}" if raw_sc is not None else "N/A"
                                    st.markdown(f"**Source {idx_val}:** *{title_val}* `[🏷️ {sec}]` (Score: `{sc_fmt}`{cite_str})")
                                    if src.get("dense_score") is not None or src.get("bm25_score") is not None:
                                        d_sc = float(src["dense_score"]) if src.get("dense_score") is not None else 0.0
                                        b_sc = float(src["bm25_score"]) if src.get("bm25_score") is not None else 0.0
                                        st.caption(f"Fused: `{sc_fmt}` | Dense: `{d_sc:.2f}` | BM25: `{b_sc:.2f}`")
                                    if src.get("snippet"):
                                        st.markdown(f"> {src['snippet']}")
                                    st.markdown("---")

                    if msg.get("is_low_confidence"):
                        st.caption("⚠️ *Generated with low retrieval confidence.*")

    # --------------------------------------------------------------------------
    # Chat Input Box (Scoped to Active Project & Saved to Supabase)
    # --------------------------------------------------------------------------
    user_prompt = st.chat_input("Ask a question about your project files...")

    if user_prompt:
        # Append User Message to UI State
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # Persist User Message to Database
        if st.session_state.active_conversation_id:
            try:
                requests.post(
                    f"{API_URL}/conversations/{st.session_state.active_conversation_id}/messages",
                    json={"role": "user", "content": user_prompt},
                    timeout=5
                )
            except Exception:
                pass

        # Build payload with conversation history & project scoping
        history_payload = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]
        doc_filter = st.session_state.selected_doc_ids if st.session_state.selected_doc_ids else None

        query_payload = {
            "question": user_prompt,
            "workspace_id": st.session_state.active_workspace_id or "default",
            "top_k": top_k,
            "use_hyde": use_hyde,
            "search_mode": mode_key,
            "score_threshold": score_threshold,
            "hybrid_alpha": 0.5,
            "history": history_payload,
            "doc_ids": doc_filter
        }

        with st.spinner("Searching project knowledge and synthesizing grounded response..."):
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

                    # Persist Assistant Message to Database
                    if st.session_state.active_conversation_id:
                        try:
                            requests.post(
                                f"{API_URL}/conversations/{st.session_state.active_conversation_id}/messages",
                                json={
                                    "role": "assistant",
                                    "content": data["answer"],
                                    "sources": data.get("sources", []),
                                    "confidence_score": data.get("confidence_score", 0.0),
                                    "is_low_confidence": data.get("is_low_confidence", False)
                                },
                                timeout=5
                            )
                        except Exception:
                            pass
                    st.rerun()
                else:
                    st.error(f"API Error: {res.text}")
            except Exception as e:
                st.error(f"Error querying API: {e}")
