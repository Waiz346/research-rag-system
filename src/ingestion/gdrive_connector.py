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

class GoogleDriveConnector:
    """Manages Google Drive authentication, folder hierarchy navigation, and document importing."""

    def __init__(self):
        self.auth_type = "demo"  # "demo", "api_key", "access_token", "service_account", "gdrive_cli"
        self.service = None
        self.account_email: Optional[str] = None
        self.api_key: Optional[str] = None
        self.access_token: Optional[str] = None
        self.credentials_info: Optional[Dict[str, Any]] = None

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

    def get_status(self) -> Dict[str, Any]:
        """Return current connector status."""
        return {
            "connected": self.auth_type == "demo" or self.service is not None,
            "auth_type": self.auth_type,
            "account_email": self.account_email,
            "message": f"Active Mode: {self.auth_type.upper()}"
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
                raise RuntimeError(f"gdrive CLI execution failed: {e}")

        raise RuntimeError("Google Drive is not connected. Connect first.")

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
        # Default to Demo mode so it works out-of-the-box
        _gdrive_connector.connect(auth_type="demo")
    return _gdrive_connector
