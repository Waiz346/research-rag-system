"""Google Drive Connector for Research RAG Intelligence System.

Enables browsing Google Drive folders, inspecting file metadata,
and directly importing/indexing PDFs, Docs, and text files into ChromaDB + BM25.

Supports multiple authentication modes:
1. Service Account JSON credentials (enterprise/production)
2. OAuth 2.0 Access Token
3. Google Cloud API Key (for public/shared folders)
4. Local `gdrive` CLI bridge
5. Interactive Demo Mode (pre-populated with canonical research papers)
"""
import io
import os
import json
import logging
import threading
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Canonical Google Drive MIME types
MIME_FOLDER = "application/vnd.google-apps.folder"
MIME_DOC = "application/vnd.google-apps.document"
MIME_PDF = "application/pdf"
MIME_TEXT = "text/plain"
MIME_MARKDOWN = "text/markdown"

# Demo Mode Mock Data (Canonical Research Literature)
DEMO_FOLDERS = {
    "root": {
        "name": "My Drive",
        "parent_id": None,
        "folders": [
            {"id": "folder_dl", "name": "Deep Learning & Transformer Foundations", "modified": "2024-03-15T10:00:00Z"},
            {"id": "folder_rag", "name": "Retrieval-Augmented Generation (RAG) Archive", "modified": "2024-04-10T14:30:00Z"},
            {"id": "folder_notes", "name": "Lab Notes & Evaluation Benchmarks", "modified": "2024-05-02T09:15:00Z"},
        ],
        "files": [
            {"id": "demo_overview", "name": "Research Roadmap & Knowledge Base Overview.md", "mime": "text/markdown", "size": 4200, "modified": "2024-05-01T12:00:00Z"}
        ]
    },
    "folder_dl": {
        "name": "Deep Learning & Transformer Foundations",
        "parent_id": "root",
        "folders": [],
        "files": [
            {"id": "demo_attention", "name": "Attention Is All You Need (Vaswani et al.).pdf", "mime": "application/pdf", "size": 2210000, "modified": "2024-01-10T08:00:00Z"},
            {"id": "demo_lora", "name": "LoRA: Low-Rank Adaptation of Large Language Models (Hu et al.).pdf", "mime": "application/pdf", "size": 1850000, "modified": "2024-02-14T11:20:00Z"},
        ]
    },
    "folder_rag": {
        "name": "Retrieval-Augmented Generation (RAG) Archive",
        "parent_id": "root",
        "folders": [],
        "files": [
            {"id": "demo_rag_lewis", "name": "Retrieval-Augmented Generation for Knowledge-Intensive Tasks (Lewis et al.).pdf", "mime": "application/pdf", "size": 3100000, "modified": "2024-03-20T16:45:00Z"},
            {"id": "demo_self_rag", "name": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (Asai et al.).pdf", "mime": "application/pdf", "size": 2400000, "modified": "2024-04-05T10:30:00Z"},
            {"id": "demo_dpr", "name": "Dense Passage Retrieval for Open-Domain Question Answering (Karpukhin et al.).pdf", "mime": "application/pdf", "size": 1950000, "modified": "2024-03-12T09:10:00Z"}
        ]
    },
    "folder_notes": {
        "name": "Lab Notes & Evaluation Benchmarks",
        "parent_id": "root",
        "folders": [],
        "files": [
            {"id": "demo_rag_eval", "name": "RAG Triad & Faithfulness Evaluation Protocols.txt", "mime": "text/plain", "size": 8900, "modified": "2024-04-28T18:00:00Z"},
            {"id": "demo_bm25_tuning", "name": "Hybrid Dense-Sparse Reciprocal Rank Fusion Guide.md", "mime": "text/markdown", "size": 6500, "modified": "2024-04-29T14:15:00Z"}
        ]
    }
}

DEMO_FILE_CONTENTS = {
    "demo_overview": (
        "# Research Roadmap & Knowledge Base Overview\n\n"
        "## Abstract\nThis laboratory dossier outlines empirical benchmarks and methodology for neural dense retrieval "
        "and hybrid lexical search across scholarly corpora.\n\n"
        "## Introduction\nLarge language models require continuous grounding on validated scientific papers to counteract hallucinations. "
        "Our system indexes research publications with section-aware decomposition.\n\n"
        "## Methodology\nWe apply dense passage embeddings coupled with BM25 lexical token matching using Reciprocal Rank Fusion.\n\n"
        "## Conclusion\nHybrid retrieval achieves higher precision and ground truth recall across specialized domains."
    ),
    "demo_attention": (
        "Title: Attention Is All You Need\nAuthors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit\n\n"
        "1. Abstract\nThe dominant sequence transduction models are based on complex recurrent or convolutional neural networks. "
        "We propose the Transformer, a model architecture eschewing recurrence and entirely relying on an attention mechanism.\n\n"
        "2. Introduction\nRecurrent neural networks factor computation along the symbol positions of the input and output sequences. "
        "In this work we introduce multi-head self-attention that connects all positions in constant time operations.\n\n"
        "3. Methodology\nThe Transformer follows an encoder-decoder architecture using stacked self-attention and point-wise fully connected layers. "
        "Multi-head attention allows the model to jointly attend to information from different representation subspaces.\n\n"
        "4. Experiments\nOn the WMT 2014 English-to-German translation task, the big transformer model achieves 28.4 BLEU, establishing a new state of the art.\n\n"
        "5. Conclusion\nThe Transformer is the first sequence transduction model based entirely on self-attention."
    ),
    "demo_lora": (
        "Title: LoRA: Low-Rank Adaptation of Large Language Models\nAuthors: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu\n\n"
        "1. Abstract\nAn important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to specific tasks. "
        "We propose Low-Rank Adaptation (LoRA), which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer.\n\n"
        "2. Introduction\nFull fine-tuning becomes prohibitive as model parameters reach hundreds of billions. LoRA reduces the number of trainable parameters by up to 10,000 times.\n\n"
        "3. Methodology\nWe constrain the parameter update by representing it with a low-rank decomposition: W + Delta W = W + B * A, where B and A have rank r << d.\n\n"
        "4. Experiments\nLoRA matches or outperforms full fine-tuning on RoBERTa, DeBERTa, and GPT-3 175B while drastically reducing GPU memory requirements.\n\n"
        "5. Conclusion\nLoRA provides modular task adaptation with zero added inference latency."
    ),
    "demo_rag_lewis": (
        "Title: Retrieval-Augmented Generation for Knowledge-Intensive Tasks\nAuthors: Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin\n\n"
        "1. Abstract\nLarge pre-trained language models store factual knowledge in their parameters, but their ability to access and manipulate precise knowledge is limited. "
        "We explore general-purpose fine-tuning recipes for Retrieval-Augmented Generation (RAG) combining pre-trained parametric and non-parametric memory.\n\n"
        "2. Introduction\nParametric memory stores knowledge in neural weights, while non-parametric memory uses a dense vector index of Wikipedia passages.\n\n"
        "3. Methodology\nWe evaluate RAG-Sequence, where the model uses the same retrieved document to generate the complete sequence, and RAG-Token, which can condition tokens on different documents.\n\n"
        "4. Experiments\nOn Natural Questions, WebQuestions, and CuratedTREC, RAG models outperform pure parametric models and generate more factual, specific responses.\n\n"
        "5. Conclusion\nRAG combines the adaptability of non-parametric knowledge retrieval with the expressive synthesis capabilities of seq2seq generators."
    ),
    "demo_self_rag": (
        "Title: Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection\nAuthors: Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi\n\n"
        "1. Abstract\nRetrieval-Augmented Generation often suffers from retrieving unhelpful information or failing to verify factual consistency. "
        "We introduce Self-RAG, a framework that trains an LM to adaptively retrieve passages on demand and critique its own generations via special reflection tokens.\n\n"
        "2. Introduction\nIndiscriminate retrieval adds computational cost and can mislead language models. Self-RAG decides whether retrieval is needed dynamically.\n\n"
        "3. Methodology\nThe model generates reflection tokens: [Retrieve], [IsRel] (relevance), [IsSup] (groundedness support), and [IsUse] (utility). During inference, beam search selects generation paths that maximize factual critique scores.\n\n"
        "4. Experiments\nSelf-RAG significantly outperforms standard RAG on open-domain QA, reasoning benchmarks, and fact-checking suites.\n\n"
        "5. Conclusion\nSelf-reflection tokens allow end-to-end control of factual grounding in conversational generation."
    ),
    "demo_dpr": (
        "Title: Dense Passage Retrieval for Open-Domain Question Answering\nAuthors: Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu\n\n"
        "1. Abstract\nOpen-domain question answering relies on efficient passage retrieval to select candidate contexts. We show that retrieval can be practically implemented using dense representations learned by dual-encoders.\n\n"
        "2. Introduction\nTraditional systems rely on TF-IDF or BM25 sparse lexical matching. Dense Passage Retrieval (DPR) maps questions and passages to continuous vectors.\n\n"
        "3. Methodology\nWe train dual BERT encoders using in-batch negative sampling and contrastive loss to maximize inner product between questions and relevant passages.\n\n"
        "4. Experiments\nDPR outperforms strong Lucene BM25 baselines on five QA datasets, achieving 78.4% top-20 passage retrieval accuracy on Natural Questions.\n\n"
        "5. Conclusion\nDense representations provide superior semantic matching for open-domain information extraction."
    ),
    "demo_rag_eval": (
        "Subject: RAG Triad & Faithfulness Evaluation Protocols\n"
        "Section: Methodology\n"
        "1. Context Relevance: Verifies that retrieved text chunks directly align with core query concepts.\n"
        "2. Faithfulness (Groundedness): Ensures that claims made in the synthesis are strictly grounded in retrieved evidence.\n"
        "3. Answer Relevance: Assesses whether the model directly resolves the user's intent without drifting."
    ),
    "demo_bm25_tuning": (
        "# Hybrid Dense-Sparse Reciprocal Rank Fusion Guide\n\n"
        "## Methodology\n"
        "1. Dense Retrieval: Captures semantic intent and paraphrasing via cosine distance over embeddings.\n"
        "2. BM25 Sparse Search: Guards against acronym and exact identifier dropouts.\n"
        "3. Reciprocal Rank Fusion (RRF): Score = (1.0 / (60 + Rank_dense)) + (1.0 / (60 + Rank_bm25))."
    )
}

def get_detected_token_path() -> Optional[Path]:
    """Search common locations for a valid authorized Google token."""
    candidates = [
        Path.home() / ".gemini" / "antigravity" / "plugin_data" / "gdocs" / "token.json",
        Path("token.json"),
        Path.home() / ".gdrive" / "token.json"
    ]
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            pass
    return None

def get_oauth_client_info() -> Tuple[Optional[str], Optional[str]]:
    """Retrieve Google OAuth client_id and client_secret from saved accounts or token.json."""
    acc_dir = Path("data/gdrive_accounts")
    if acc_dir.exists():
        for f in acc_dir.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    d = json.load(fp)
                if d.get("client_id") and d.get("client_secret"):
                    return d["client_id"], d["client_secret"]
            except Exception:
                pass

    tp = get_detected_token_path()
    if tp:
        try:
            with open(tp, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            if d.get("client_id") and d.get("client_secret"):
                return d["client_id"], d["client_secret"]
        except Exception:
            pass

    return None, None

class GoogleDriveConnector:
    """Manages Google Drive authentication, folder hierarchy navigation, and document importing."""

    def __init__(self):
        self.auth_type = "demo"  # "demo", "user_account", "api_key", "access_token", "service_account", "gdrive_cli"
        self.service = None
        self.account_email: Optional[str] = None
        self.api_key: Optional[str] = None
        self.access_token: Optional[str] = None
        self.credentials_info: Optional[Dict[str, Any]] = None
        self.login_state: Dict[str, Any] = {"status": "idle", "email": None, "message": None, "error": None}
        self.login_thread: Optional[threading.Thread] = None
        self.active_server = None

    def connect(
        self,
        auth_type: str = "demo",
        api_key: Optional[str] = None,
        access_token: Optional[str] = None,
        service_account_json: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Configure authentication with Google Drive."""
        self.auth_type = auth_type.lower()
        self.api_key = api_key
        self.access_token = access_token
        self.service = None
        self.account_email = None

        if self.auth_type in ["user_account", "google_account", "auto"]:
            token_path = get_detected_token_path()
            if not token_path:
                raise ValueError("No authorized Google Account found on this machine. Authenticate or use Demo mode.")
            try:
                from google.oauth2.credentials import Credentials
                from googleapiclient.discovery import build

                with open(token_path, "r", encoding="utf-8") as f:
                    d = json.load(f)

                token_data = d.get("token", {})
                creds = Credentials(
                    token=token_data.get("access_token"),
                    refresh_token=token_data.get("refresh_token"),
                    token_uri="https://oauth2.googleapis.com/token",
                    client_id=d.get("client_id"),
                    client_secret=d.get("client_secret"),
                    scopes=d.get("scopes")
                )
                self.service = build("drive", "v3", credentials=creds)
                self.auth_type = "user_account"
                self.account_email = d.get("email", "Google Account")
                return {
                    "status": "success",
                    "auth_type": "user_account",
                    "account": self.account_email,
                    "message": f"Connected to Google Drive as {self.account_email}"
                }
            except Exception as e:
                logger.error(f"Failed to authenticate with user Google Account: {e}")
                raise RuntimeError(f"User Google Account connection failed: {e}")

        if self.auth_type == "demo":
            self.account_email = "researcher-demo@google.drive"
            logger.info("Google Drive initialized in Demo Mode.")
            return {"status": "success", "auth_type": "demo", "message": "Connected to Google Drive (Interactive Demo Mode)"}

        if self.auth_type == "service_account":
            if not service_account_json:
                raise ValueError("Service account JSON content or file path is required.")
            try:
                from google.oauth2 import service_account
                from googleapiclient.discovery import build

                # Check if it's a file path or direct JSON string
                if os.path.exists(service_account_json):
                    creds = service_account.Credentials.from_service_account_file(
                        service_account_json,
                        scopes=["https://www.googleapis.com/auth/drive.readonly"]
                    )
                else:
                    data = json.loads(service_account_json)
                    creds = service_account.Credentials.from_service_account_info(
                        data,
                        scopes=["https://www.googleapis.com/auth/drive.readonly"]
                    )
                self.service = build("drive", "v3", credentials=creds)
                self.account_email = getattr(creds, "service_account_email", "service-account")
                return {"status": "success", "auth_type": "service_account", "account": self.account_email, "message": f"Connected as {self.account_email}"}
            except Exception as e:
                logger.error(f"Failed to authenticate with Service Account: {e}")
                raise RuntimeError(f"Service Account authentication failed: {e}")

        if self.auth_type == "access_token":
            if not access_token:
                raise ValueError("OAuth access token is required.")
            try:
                from google.oauth2.credentials import Credentials
                from googleapiclient.discovery import build
                creds = Credentials(token=access_token)
                self.service = build("drive", "v3", credentials=creds)
                return {"status": "success", "auth_type": "access_token", "message": "Connected via OAuth Access Token"}
            except Exception as e:
                raise RuntimeError(f"OAuth token authentication failed: {e}")

        if self.auth_type == "api_key":
            if not api_key:
                raise ValueError("Google API key is required.")
            try:
                from googleapiclient.discovery import build
                self.service = build("drive", "v3", developerKey=api_key)
                return {"status": "success", "auth_type": "api_key", "message": "Connected via API Key (Public Shared Folders)"}
            except Exception as e:
                raise RuntimeError(f"API Key initialization failed: {e}")

        if self.auth_type == "gdrive_cli":
            # Test if local gdrive CLI is available and authenticated
            try:
                res = subprocess.run(["gdrive", "auth", "status"], capture_output=True, text=True, check=False)
                if "Logged in" in res.stdout:
                    return {"status": "success", "auth_type": "gdrive_cli", "message": "Connected via local gdrive CLI"}
                else:
                    return {"status": "warning", "auth_type": "gdrive_cli", "message": "gdrive CLI not logged in. Run 'gdrive auth login' in terminal."}
            except Exception as e:
                raise RuntimeError(f"Local gdrive CLI not found on PATH: {e}")

        raise ValueError(f"Unsupported auth_type: {auth_type}")

    def list_saved_accounts(self) -> List[Dict[str, Any]]:
        """List all saved Google accounts available for quick switching."""
        acc_dir = Path("data/gdrive_accounts")
        acc_dir.mkdir(parents=True, exist_ok=True)

        # Proactively check if token.json has an account not yet in acc_dir
        tp = get_detected_token_path()
        if tp:
            try:
                with open(tp, "r", encoding="utf-8") as fp:
                    d = json.load(fp)
                email = d.get("email")
                if email:
                    target_file = acc_dir / f"{email}.json"
                    if not target_file.exists():
                        with open(target_file, "w", encoding="utf-8") as out_fp:
                            json.dump(d, out_fp, indent=2)
            except Exception:
                pass

        accounts = []
        for f in acc_dir.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    d = json.load(fp)
                email = d.get("email", f.stem)
                name = d.get("name")
                is_active = (self.auth_type == "user_account" and self.account_email == email)
                accounts.append({
                    "email": email,
                    "name": name,
                    "is_active": is_active,
                    "updated_at": d.get("updated_at")
                })
            except Exception as e:
                logger.warning(f"Error reading account file {f}: {e}")

        # Sort: active first, then alphabetical
        accounts.sort(key=lambda x: (not x["is_active"], x["email"]))
        return accounts

    def switch_account(self, email: str) -> Dict[str, Any]:
        """Instantly switch active Google Drive session to a saved account."""
        acc_dir = Path("data/gdrive_accounts")
        target = acc_dir / f"{email}.json"
        if not target.exists():
            raise FileNotFoundError(f"Saved account credentials for '{email}' not found.")

        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            with open(target, "r", encoding="utf-8") as fp:
                d = json.load(fp)

            token_data = d.get("token", {})
            creds = Credentials(
                token=token_data.get("access_token"),
                refresh_token=token_data.get("refresh_token"),
                token_uri="https://oauth2.googleapis.com/token",
                client_id=d.get("client_id"),
                client_secret=d.get("client_secret"),
                scopes=d.get("scopes")
            )
            self.service = build("drive", "v3", credentials=creds)
            self.auth_type = "user_account"
            self.account_email = email
            logger.info(f"Switched Google Drive active account to: {email}")
            return {
                "status": "success",
                "email": email,
                "message": f"Successfully switched to Google Account: {email}"
            }
        except Exception as e:
            logger.error(f"Failed to switch account to {email}: {e}")
            raise RuntimeError(f"Failed to switch Google account to {email}: {e}")

    def start_browser_login(self) -> Dict[str, Any]:
        """Launch browser-based OAuth flow with Google Account Chooser."""
        # If a login flow is already active, close the existing server so we can start fresh
        if self.active_server:
            try:
                self.active_server.server_close()
            except Exception:
                pass
            self.active_server = None

        client_id, client_secret = get_oauth_client_info()
        if not client_id or not client_secret:
            raise ValueError("No Google OAuth client ID/secret found to initialize sign-in.")

        from google_auth_oauthlib.flow import InstalledAppFlow, _RedirectWSGIApp, _ExclusiveWSGIServer
        import wsgiref.simple_server
        from wsgiref.simple_server import WSGIRequestHandler

        client_config = {
            "installed": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost", "http://127.0.0.1"]
            }
        }

        # Check registered scopes in existing token to avoid 403 restricted_client
        registered_client_scopes = []
        tp = get_detected_token_path()
        if tp:
            try:
                with open(tp, "r", encoding="utf-8") as fp:
                    registered_client_scopes = json.load(fp).get("scopes", [])
            except Exception:
                pass

        if not registered_client_scopes:
            acc_dir = Path("data/gdrive_accounts")
            if acc_dir.exists():
                for f in acc_dir.glob("*.json"):
                    try:
                        with open(f, "r", encoding="utf-8") as fp:
                            registered_client_scopes = json.load(fp).get("scopes", [])
                        if registered_client_scopes:
                            break
                    except Exception:
                        pass

        # Use full drive scope if registered (drive.readonly triggers 403 restricted_client if unregistered)
        if "https://www.googleapis.com/auth/drive" in registered_client_scopes:
            drive_scope = "https://www.googleapis.com/auth/drive"
        elif "https://www.googleapis.com/auth/drive.readonly" in registered_client_scopes:
            drive_scope = "https://www.googleapis.com/auth/drive.readonly"
        else:
            drive_scope = "https://www.googleapis.com/auth/drive"

        scopes = [
            drive_scope,
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
            "openid"
        ]
        flow = InstalledAppFlow.from_client_config(client_config, scopes=scopes)
        wsgi_app = _RedirectWSGIApp("Authentication successful! You can now close this tab and return to Research RAG.")
        local_server = wsgiref.simple_server.make_server(
            "localhost",
            0,
            wsgi_app,
            server_class=_ExclusiveWSGIServer,
            handler_class=WSGIRequestHandler
        )
        self.active_server = local_server
        port = local_server.server_port
        flow.redirect_uri = f"http://localhost:{port}/"
        auth_url, _ = flow.authorization_url(prompt="select_account", access_type="offline")

        self.login_state = {
            "status": "in_progress",
            "auth_url": auth_url,
            "email": None,
            "message": "Sign-in prompt opened. If not shown automatically, click the button below.",
            "error": None
        }

        # Attempt to open browser automatically using Windows shell or webbrowser
        try:
            if hasattr(os, "startfile"):
                os.startfile(auth_url)
            else:
                import webbrowser
                webbrowser.open(auth_url, new=1, autoraise=True)
        except Exception as e:
            logger.warning(f"Could not open browser automatically via shell: {e}")

        def _auth_worker():
            try:
                from googleapiclient.discovery import build
                local_server.timeout = 300  # 5 minutes timeout
                local_server.handle_request()

                if not hasattr(wsgi_app, "last_request_uri") or not wsgi_app.last_request_uri:
                    raise TimeoutError("Timed out or received no response from Google sign-in.")

                authorization_response = wsgi_app.last_request_uri.replace("http", "https")
                flow.fetch_token(authorization_response=authorization_response)
                creds = flow.credentials

                # Query profile name & email
                oauth2_service = build("oauth2", "v2", credentials=creds)
                uinfo = oauth2_service.userinfo().get().execute()
                email = uinfo.get("email") or "google_user"
                name = uinfo.get("name")

                # Persist to data/gdrive_accounts/{email}.json
                save_data = {
                    "email": email,
                    "name": name,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "token": {
                        "access_token": creds.token,
                        "refresh_token": creds.refresh_token,
                        "expiry": creds.expiry.isoformat() if creds.expiry else None
                    },
                    "scopes": creds.scopes,
                    "updated_at": datetime.now().isoformat()
                }
                acc_path = Path("data/gdrive_accounts") / f"{email}.json"
                acc_path.parent.mkdir(parents=True, exist_ok=True)
                with open(acc_path, "w", encoding="utf-8") as out_fp:
                    json.dump(save_data, out_fp, indent=2)

                # Switch active account immediately
                self.switch_account(email)
                self.login_state = {
                    "status": "completed",
                    "auth_url": None,
                    "email": email,
                    "message": f"Successfully connected and switched to Google Account: {email}",
                    "error": None
                }
            except Exception as e:
                logger.error(f"Google OAuth sign-in error: {e}")
                self.login_state = {
                    "status": "error",
                    "auth_url": None,
                    "email": None,
                    "message": None,
                    "error": str(e)
                }
            finally:
                try:
                    local_server.server_close()
                except Exception:
                    pass

        self.login_thread = threading.Thread(target=_auth_worker, daemon=True)
        self.login_thread.start()

        return {
            "status": "in_progress",
            "auth_url": auth_url,
            "message": "Sign-in prompt opened in browser. If not shown, click the button below."
        }

    def get_login_status(self) -> Dict[str, Any]:
        """Return status of pending browser login."""
        return self.login_state

    def delete_saved_account(self, email: str) -> Dict[str, Any]:
        """Remove a saved Google account."""
        acc_path = Path("data/gdrive_accounts") / f"{email}.json"
        if acc_path.exists():
            acc_path.unlink()
            if self.account_email == email:
                remaining = self.list_saved_accounts()
                if remaining:
                    self.switch_account(remaining[0]["email"])
                else:
                    self.connect(auth_type="demo")
            return {"status": "success", "message": f"Removed account {email}"}
        return {"status": "not_found", "message": f"Account {email} not found."}

    def get_status(self) -> Dict[str, Any]:
        """Return current connector status."""
        token_path = get_detected_token_path()
        detected_email = None
        if token_path:
            try:
                with open(token_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    detected_email = d.get("email")
            except Exception:
                pass

        is_connected = (self.auth_type in ["demo", "user_account"] and (self.service is not None or self.auth_type == "demo")) or (self.service is not None)
        active_label = self.account_email or self.auth_type.upper()

        return {
            "connected": is_connected,
            "auth_type": self.auth_type,
            "account_email": self.account_email or detected_email,
            "has_detected_account": token_path is not None,
            "detected_email": detected_email,
            "message": f"Active: {active_label}"
        }

    def list_folder_contents(self, folder_id: str = "root") -> Dict[str, Any]:
        """List files and subfolders inside the specified Google Drive folder."""
        # 1. Demo Mode
        if self.auth_type == "demo":
            curr = DEMO_FOLDERS.get(folder_id)
            if not curr:
                curr = DEMO_FOLDERS["root"]
                folder_id = "root"

            folders = [
                {
                    "id": f["id"],
                    "name": f["name"],
                    "mime_type": MIME_FOLDER,
                    "is_folder": True,
                    "size": None,
                    "modified_time": f["modified"],
                    "web_view_link": None,
                    "icon": "📁"
                }
                for f in curr.get("folders", [])
            ]

            files = [
                {
                    "id": f["id"],
                    "name": f["name"],
                    "mime_type": f["mime"],
                    "is_folder": False,
                    "size": f["size"],
                    "modified_time": f["modified"],
                    "web_view_link": f"https://drive.google.com/file/d/{f['id']}/view",
                    "icon": "📄" if "pdf" in f["mime"] else "📝"
                }
                for f in curr.get("files", [])
            ]

            return {
                "current_folder_id": folder_id,
                "current_folder_name": curr["name"],
                "parent_id": curr.get("parent_id"),
                "folders": folders,
                "files": files,
                "total_items": len(folders) + len(files)
            }

        # 2. Live Google Drive API Client
        if self.service is not None:
            try:
                # Query children of folder_id
                query = f"'{folder_id}' in parents and trashed = false"
                results = self.service.files().list(
                    q=query,
                    pageSize=100,
                    fields="files(id, name, mimeType, size, modifiedTime, webViewLink, iconLink)",
                    orderBy="folder, name"
                ).execute()

                raw_items = results.get("files", [])
                folders = []
                files = []

                for item in raw_items:
                    is_folder = item.get("mimeType") == MIME_FOLDER
                    file_item = {
                        "id": item["id"],
                        "name": item["name"],
                        "mime_type": item.get("mimeType", "application/octet-stream"),
                        "is_folder": is_folder,
                        "size": int(item["size"]) if item.get("size") else None,
                        "modified_time": item.get("modifiedTime"),
                        "web_view_link": item.get("webViewLink"),
                        "icon": "📁" if is_folder else ("📄" if "pdf" in item.get("mimeType", "") else "📝")
                    }
                    if is_folder:
                        folders.append(file_item)
                    else:
                        files.append(file_item)

                # Get current folder name
                folder_name = "My Drive"
                parent_id = None
                if folder_id != "root":
                    try:
                        folder_meta = self.service.files().get(
                            fileId=folder_id,
                            fields="id, name, parents"
                        ).execute()
                        folder_name = folder_meta.get("name", folder_id)
                        parents = folder_meta.get("parents", [])
                        parent_id = parents[0] if parents else "root"
                    except Exception:
                        folder_name = folder_id

                return {
                    "current_folder_id": folder_id,
                    "current_folder_name": folder_name,
                    "parent_id": parent_id,
                    "folders": folders,
                    "files": files,
                    "total_items": len(folders) + len(files)
                }
            except Exception as e:
                logger.error(f"Error querying Google Drive API: {e}")
                raise RuntimeError(f"Drive API error: {e}")

        # 3. gdrive CLI bridge
        if self.auth_type == "gdrive_cli":
            try:
                cmd = ["gdrive", "readonly", "ls", folder_id if folder_id != "root" else ""]
                cmd = [c for c in cmd if c]
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
                # Basic parsing of gdrive ls output
                folders = []
                files = []
                for line in lines[1:]:  # skip header
                    parts = line.split()
                    if len(parts) >= 2:
                        fid = parts[0]
                        fname = " ".join(parts[1:])
                        is_folder = "dir" in line.lower() or "folder" in line.lower()
                        item = {
                            "id": fid,
                            "name": fname,
                            "mime_type": MIME_FOLDER if is_folder else "application/pdf",
                            "is_folder": is_folder,
                            "size": None,
                            "modified_time": None,
                            "web_view_link": f"https://drive.google.com/file/d/{fid}/view",
                            "icon": "📁" if is_folder else "📄"
                        }
                        if is_folder:
                            folders.append(item)
                        else:
                            files.append(item)

                return {
                    "current_folder_id": folder_id,
                    "current_folder_name": "Google Drive",
                    "parent_id": "root" if folder_id != "root" else None,
                    "folders": folders,
                    "files": files,
                    "total_items": len(folders) + len(files)
                }
            except Exception as e:
                logger.warning(f"gdrive CLI execution failed: {e}")

        return {
            "current_folder_id": folder_id,
            "current_folder_name": "Not Connected",
            "parent_id": None,
            "folders": [],
            "files": [],
            "total_items": 0,
            "error": "Google Drive is not connected. Please choose an authentication mode and click Connect."
        }

    def search_drive(self, query: str) -> Dict[str, Any]:
        """Search files and folders in Google Drive by name, keywords, or content."""
        query_clean = (query or "").strip()
        if not query_clean:
            return {"query": "", "folders": [], "files": [], "total_items": 0}

        q_lower = query_clean.lower()

        # 1. Demo Mode
        if self.auth_type == "demo":
            matching_folders = []
            matching_files = []
            seen_folder_ids = set()
            seen_file_ids = set()

            for folder_id, f_data in DEMO_FOLDERS.items():
                # Check subfolders
                for subf in f_data.get("folders", []):
                    if subf["id"] not in seen_folder_ids and q_lower in subf["name"].lower():
                        seen_folder_ids.add(subf["id"])
                        matching_folders.append({
                            "id": subf["id"],
                            "name": subf["name"],
                            "mime_type": MIME_FOLDER,
                            "is_folder": True,
                            "size": None,
                            "modified_time": subf.get("modified"),
                            "web_view_link": None,
                            "icon": "📁"
                        })

                # Check files
                for f in f_data.get("files", []):
                    fid = f["id"]
                    if fid in seen_file_ids:
                        continue
                    file_text = DEMO_FILE_CONTENTS.get(fid, "")
                    if q_lower in f["name"].lower() or q_lower in file_text.lower():
                        seen_file_ids.add(fid)
                        matching_files.append({
                            "id": fid,
                            "name": f["name"],
                            "mime_type": f.get("mime", MIME_PDF),
                            "is_folder": False,
                            "size": f.get("size"),
                            "modified_time": f.get("modified"),
                            "web_view_link": f"https://drive.google.com/file/d/{fid}/view",
                            "icon": "📄" if "pdf" in f.get("mime", "") else "📝"
                        })

            return {
                "query": query_clean,
                "folders": matching_folders,
                "files": matching_files,
                "total_items": len(matching_folders) + len(matching_files)
            }

        # 2. Live Google Drive API Client
        if self.service is not None:
            try:
                safe_q = query_clean.replace("\\", "\\\\").replace("'", "\\'")
                drive_q = f"name contains '{safe_q}' and trashed = false"
                results = self.service.files().list(
                    q=drive_q,
                    pageSize=50,
                    fields="files(id, name, mimeType, size, modifiedTime, webViewLink, iconLink)",
                    orderBy="folder, name"
                ).execute()

                raw_items = results.get("files", [])
                folders = []
                files = []

                for item in raw_items:
                    is_folder = item.get("mimeType") == MIME_FOLDER
                    file_item = {
                        "id": item["id"],
                        "name": item["name"],
                        "mime_type": item.get("mimeType", "application/octet-stream"),
                        "is_folder": is_folder,
                        "size": int(item["size"]) if item.get("size") else None,
                        "modified_time": item.get("modifiedTime"),
                        "web_view_link": item.get("webViewLink"),
                        "icon": "📁" if is_folder else ("📄" if "pdf" in item.get("mimeType", "") else "📝")
                    }
                    if is_folder:
                        folders.append(file_item)
                    else:
                        files.append(file_item)

                return {
                    "query": query_clean,
                    "folders": folders,
                    "files": files,
                    "total_items": len(folders) + len(files)
                }
            except Exception as e:
                logger.error(f"Error searching Google Drive API: {e}")
                raise RuntimeError(f"Drive API search error: {e}")

        # 3. gdrive CLI bridge
        if self.auth_type == "gdrive_cli":
            try:
                cmd = ["gdrive", "readonly", "ls"]
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
                folders = []
                files = []
                for line in lines[1:]:
                    if q_lower in line.lower():
                        parts = line.split()
                        if len(parts) >= 2:
                            fid = parts[0]
                            fname = " ".join(parts[1:])
                            is_folder = "dir" in line.lower() or "folder" in line.lower()
                            item = {
                                "id": fid,
                                "name": fname,
                                "mime_type": MIME_FOLDER if is_folder else "application/pdf",
                                "is_folder": is_folder,
                                "size": None,
                                "modified_time": None,
                                "web_view_link": f"https://drive.google.com/file/d/{fid}/view",
                                "icon": "📁" if is_folder else "📄"
                            }
                            if is_folder:
                                folders.append(item)
                            else:
                                files.append(item)
                return {
                    "query": query_clean,
                    "folders": folders,
                    "files": files,
                    "total_items": len(folders) + len(files)
                }
            except Exception as e:
                logger.warning(f"gdrive CLI search failed: {e}")

        return {
            "query": query_clean,
            "folders": [],
            "files": [],
            "total_items": 0,
            "error": "Google Drive is not connected."
        }

    def download_file_content(self, file_id: str) -> Tuple[str, str, str]:
        """Download and extract text content from a Google Drive file.

        Returns:
            Tuple of (filename, text_content, detected_mime_type)
        """
        # 1. Demo Mode
        if self.auth_type == "demo":
            if file_id in DEMO_FILE_CONTENTS:
                # Find file metadata
                name = "research_paper.txt"
                mime = MIME_TEXT
                for folder in DEMO_FOLDERS.values():
                    for f in folder.get("files", []):
                        if f["id"] == file_id:
                            name = f["name"]
                            mime = f["mime"]
                            break
                return (name, DEMO_FILE_CONTENTS[file_id], mime)
            raise FileNotFoundError(f"Demo file ID {file_id} not found.")

        # 2. Live API
        if self.service is not None:
            try:
                meta = self.service.files().get(
                    fileId=file_id,
                    fields="id, name, mimeType"
                ).execute()
                name = meta.get("name", f"gdrive_{file_id}")
                mime = meta.get("mimeType", "")

                text_content = ""
                # Google Doc: Export as Plain Text
                if mime == MIME_DOC:
                    export_req = self.service.files().export_media(fileId=file_id, mimeType="text/plain")
                    text_content = export_req.execute().decode("utf-8", errors="replace")
                # PDF: Download binary and extract with pypdf
                elif mime == MIME_PDF or name.lower().endswith(".pdf"):
                    from pypdf import PdfReader
                    get_req = self.service.files().get_media(fileId=file_id)
                    pdf_bytes = get_req.execute()
                    reader = PdfReader(io.BytesIO(pdf_bytes))
                    extracted = [p.extract_text() for p in reader.pages if p.extract_text()]
                    text_content = "\n\n".join(extracted)
                # Plain Text / Markdown: Download directly
                else:
                    get_req = self.service.files().get_media(fileId=file_id)
                    raw_bytes = get_req.execute()
                    text_content = raw_bytes.decode("utf-8", errors="replace")

                return (name, text_content, mime)
            except Exception as e:
                logger.error(f"Error downloading file {file_id} from Google Drive: {e}")
                raise RuntimeError(f"Failed to download Google Drive file: {e}")

        # 3. gdrive CLI
        if self.auth_type == "gdrive_cli":
            try:
                out_path = Path("data/downloads") / f"gdrive_{file_id}.tmp"
                subprocess.run(["gdrive", "readonly", "download", file_id, "--destination", str(out_path)], check=True)
                with open(out_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                return (f"gdrive_{file_id}.txt", content, MIME_TEXT)
            except Exception as e:
                raise RuntimeError(f"gdrive CLI download failed: {e}")

        raise RuntimeError("Google Drive is not connected.")

# Global singleton instance
_gdrive_connector: Optional[GoogleDriveConnector] = None

def get_gdrive_connector() -> GoogleDriveConnector:
    """Retrieve or initialize global GoogleDriveConnector singleton."""
    global _gdrive_connector
    if _gdrive_connector is None:
        _gdrive_connector = GoogleDriveConnector()
        # Automatically connect to user's authorized Google account if token is detected
        token_path = get_detected_token_path()
        if token_path:
            try:
                _gdrive_connector.connect(auth_type="user_account")
                logger.info(f"Auto-connected Google Drive as {_gdrive_connector.account_email}")
            except Exception as e:
                logger.warning(f"Could not auto-connect Google account ({e}), falling back to Demo mode.")
                _gdrive_connector.connect(auth_type="demo")
        else:
            _gdrive_connector.connect(auth_type="demo")
    return _gdrive_connector
