"""Workspace, folder, and multi-thread conversation persistence store.

Supports online PostgreSQL databases (Supabase, Neon, AWS RDS, CockroachDB) via pg8000
and seamlessly falls back to local SQLite when DATABASE_URL is not configured.
"""
import re
import json
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Boolean, Text,
    DateTime, ForeignKey, event, desc
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session

from src.config import settings

logger = logging.getLogger(__name__)

Base = declarative_base()


def now_utc():
    return datetime.now(timezone.utc)


class Workspace(Base):
    """Workspace / Folder containing related research topics and conversation threads."""
    __tablename__ = "workspaces"

    id = Column(String(64), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=now_utc)
    updated_at = Column(DateTime(timezone=True), default=now_utc, onupdate=now_utc)

    conversations = relationship(
        "Conversation",
        back_populates="workspace",
        cascade="all, delete-orphan",
        order_by="desc(Conversation.updated_at)"
    )
    documents = relationship(
        "Document",
        back_populates="workspace",
        cascade="all, delete-orphan",
        order_by="desc(Document.created_at)"
    )


class Conversation(Base):
    """Sub-conversation thread belonging to a parent Workspace."""
    __tablename__ = "conversations"

    id = Column(String(64), primary_key=True)
    workspace_id = Column(String(64), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False, default="New Conversation")
    created_at = Column(DateTime(timezone=True), default=now_utc)
    updated_at = Column(DateTime(timezone=True), default=now_utc, onupdate=now_utc)

    workspace = relationship("Workspace", back_populates="conversations")
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at"
    )


class Message(Base):
    """Individual message in a conversation thread with citations and confidence metadata."""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(64), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(32), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    sources_json = Column(Text, nullable=True)  # JSON-encoded cited passages
    confidence_score = Column(Float, default=0.0)
    is_low_confidence = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=now_utc)

    conversation = relationship("Conversation", back_populates="messages")


class Document(Base):
    """File / Document context and metadata stored directly in the database under a Workspace."""
    __tablename__ = "documents"

    id = Column(String(128), primary_key=True)  # doc_id
    workspace_id = Column(String(64), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    filename = Column(String(255), nullable=False)
    source_type = Column(String(32), default="upload")  # "arxiv", "gdrive", "upload", "pdf", etc.
    authors = Column(String(500), default="Unknown")
    citation_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    char_count = Column(Integer, default=0)
    sections_json = Column(Text, nullable=True)  # JSON-encoded list of section names
    full_text = Column(Text, nullable=False)     # Complete file context stored online in database!
    created_at = Column(DateTime(timezone=True), default=now_utc)
    updated_at = Column(DateTime(timezone=True), default=now_utc, onupdate=now_utc)

    workspace = relationship("Workspace", back_populates="documents")


def normalize_database_url(url: Optional[str], data_dir: Path) -> str:
    """Normalize database connection string for SQLAlchemy with pure-Python drivers."""
    if not url or not url.strip():
        db_path = (data_dir / "workspaces.db").resolve()
        return f"sqlite:///{db_path}"

    clean = url.strip()
    # Normalize Postgres URLs to use pg8000 (pure-Python, no C compiler needed on Windows)
    if clean.startswith("postgres://"):
        clean = clean.replace("postgres://", "postgresql+pg8000://", 1)
    elif clean.startswith("postgresql://") and not clean.startswith("postgresql+"):
        clean = clean.replace("postgresql://", "postgresql+pg8000://", 1)
    return clean


class WorkspaceStore:
    """Manages CRUD operations for Workspaces, sub-conversations, and messages."""

    def __init__(self, db_url: Optional[str] = None):
        self.raw_url = db_url or getattr(settings, "DATABASE_URL", None)
        self.resolved_url = normalize_database_url(self.raw_url, settings.DATA_DIR)
        
        connect_args = {}
        if self.resolved_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False

        self.engine = create_engine(
            self.resolved_url,
            connect_args=connect_args,
            pool_pre_ping=True
        )

        # Enable SQLite Foreign Key cascade delete support
        if self.resolved_url.startswith("sqlite"):
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._init_db()

    def _init_db(self):
        """Create tables if not existing, seed default workspace, and migrate local files if needed."""
        Base.metadata.create_all(bind=self.engine)
        self.get_or_create_default_workspace()
        self._migrate_local_manifest_if_needed()

    @contextmanager
    def get_session(self) -> Session:
        session: Session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()

    def _workspace_to_dict(self, ws: Workspace, include_conversations: bool = False) -> Dict[str, Any]:
        conv_list = []
        if include_conversations and hasattr(ws, "conversations"):
            conv_list = [self._conversation_to_dict(c, include_messages=False) for c in ws.conversations]
        doc_count = len(ws.documents) if hasattr(ws, "documents") else 0
        return {
            "id": ws.id,
            "name": ws.name,
            "description": ws.description or "",
            "created_at": ws.created_at.isoformat() if ws.created_at else "",
            "updated_at": ws.updated_at.isoformat() if ws.updated_at else "",
            "conversation_count": len(ws.conversations) if hasattr(ws, "conversations") else 0,
            "document_count": doc_count,
            "conversations": conv_list
        }

    def _document_to_dict(self, doc: Document, include_full_text: bool = False) -> Dict[str, Any]:
        sections = []
        if doc.sections_json:
            try:
                sections = json.loads(doc.sections_json)
            except Exception:
                sections = ["General"]
        res = {
            "doc_id": doc.id,
            "workspace_id": doc.workspace_id,
            "title": doc.title,
            "filename": doc.filename,
            "source_type": doc.source_type,
            "authors": doc.authors or "Unknown",
            "citation_count": doc.citation_count or 0,
            "chunk_count": doc.chunk_count or 0,
            "char_count": doc.char_count or 0,
            "sections": sections,
            "created_at": doc.created_at.isoformat() if doc.created_at else "",
        }
        if include_full_text:
            res["full_text"] = doc.full_text or ""
        return res

    def _conversation_to_dict(self, conv: Conversation, include_messages: bool = False) -> Dict[str, Any]:
        msg_list = []
        if include_messages and hasattr(conv, "messages"):
            msg_list = [self._message_to_dict(m) for m in conv.messages]
        return {
            "id": conv.id,
            "workspace_id": conv.workspace_id,
            "title": conv.title,
            "created_at": conv.created_at.isoformat() if conv.created_at else "",
            "updated_at": conv.updated_at.isoformat() if conv.updated_at else "",
            "message_count": len(conv.messages) if hasattr(conv, "messages") else 0,
            "messages": msg_list
        }

    def _message_to_dict(self, msg: Message) -> Dict[str, Any]:
        sources = []
        if msg.sources_json:
            try:
                sources = json.loads(msg.sources_json)
            except Exception:
                sources = []
        return {
            "id": msg.id,
            "conversation_id": msg.conversation_id,
            "role": msg.role,
            "content": msg.content,
            "sources": sources,
            "confidence_score": msg.confidence_score or 0.0,
            "is_low_confidence": bool(msg.is_low_confidence),
            "created_at": msg.created_at.isoformat() if msg.created_at else ""
        }

    def get_or_create_default_workspace(self) -> Dict[str, Any]:
        """Ensure default workspace and initial thread exists."""
        with self.get_session() as session:
            ws = session.query(Workspace).filter(Workspace.id == "default").first()
            if not ws:
                ws = Workspace(
                    id="default",
                    name="General Research",
                    description="Default workspace for multi-source research & synthesis."
                )
                session.add(ws)
                session.commit()
                session.refresh(ws)

                conv = Conversation(
                    id=str(uuid.uuid4())[:8],
                    workspace_id="default",
                    title="General Research Chat"
                )
                session.add(conv)
                session.commit()
                session.refresh(ws)
            return self._workspace_to_dict(ws, include_conversations=True)

    def list_workspaces(self) -> List[Dict[str, Any]]:
        """List all workspaces ordered by updated_at descending."""
        with self.get_session() as session:
            workspaces = session.query(Workspace).order_by(desc(Workspace.updated_at)).all()
            return [self._workspace_to_dict(w, include_conversations=True) for w in workspaces]

    def create_workspace(self, name: str, description: Optional[str] = None) -> Dict[str, Any]:
        """Create a new workspace / folder with an initial conversation thread."""
        name_clean = name.strip() or "Untitled Workspace"
        base_slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", name_clean.lower())[:24].strip("_")
        ws_id = f"ws_{base_slug}_{str(uuid.uuid4())[:6]}"

        with self.get_session() as session:
            ws = Workspace(
                id=ws_id,
                name=name_clean,
                description=description.strip() if description else ""
            )
            session.add(ws)
            session.commit()

            # Seed an initial conversation
            initial_conv = Conversation(
                id=str(uuid.uuid4())[:8],
                workspace_id=ws_id,
                title="Conversation 1"
            )
            session.add(initial_conv)
            session.commit()
            session.refresh(ws)
            return self._workspace_to_dict(ws, include_conversations=True)

    def get_workspace(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        """Get workspace by ID including its conversation threads."""
        with self.get_session() as session:
            ws = session.query(Workspace).filter(Workspace.id == workspace_id).first()
            if not ws:
                return None
            return self._workspace_to_dict(ws, include_conversations=True)

    def delete_workspace(self, workspace_id: str) -> bool:
        """Delete a workspace and all its cascaded conversations and messages."""
        with self.get_session() as session:
            ws = session.query(Workspace).filter(Workspace.id == workspace_id).first()
            if not ws:
                return False
            session.delete(ws)
            session.commit()

            # If default was deleted or no workspaces remain, re-create default
            remaining = session.query(Workspace).count()
            if remaining == 0:
                self.get_or_create_default_workspace()
            return True

    def list_conversations(self, workspace_id: str) -> List[Dict[str, Any]]:
        """List all conversations within a workspace."""
        with self.get_session() as session:
            convs = session.query(Conversation).filter(
                Conversation.workspace_id == workspace_id
            ).order_by(desc(Conversation.updated_at)).all()
            return [self._conversation_to_dict(c, include_messages=False) for c in convs]

    def create_conversation(self, workspace_id: str, title: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Create a new conversation thread inside a workspace."""
        with self.get_session() as session:
            ws = session.query(Workspace).filter(Workspace.id == workspace_id).first()
            if not ws:
                return None

            conv_id = str(uuid.uuid4())[:8]
            conv_title = (title.strip() if title else "") or f"Conversation {len(ws.conversations) + 1}"
            conv = Conversation(
                id=conv_id,
                workspace_id=workspace_id,
                title=conv_title
            )
            ws.updated_at = now_utc()
            session.add(conv)
            session.commit()
            session.refresh(conv)
            return self._conversation_to_dict(conv, include_messages=True)

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation by ID including its complete message history."""
        with self.get_session() as session:
            conv = session.query(Conversation).filter(Conversation.id == conversation_id).first()
            if not conv:
                return None
            return self._conversation_to_dict(conv, include_messages=True)

    def update_conversation_title(self, conversation_id: str, title: str) -> Optional[Dict[str, Any]]:
        """Update conversation title."""
        with self.get_session() as session:
            conv = session.query(Conversation).filter(Conversation.id == conversation_id).first()
            if not conv:
                return None
            conv.title = title.strip()
            conv.updated_at = now_utc()
            session.commit()
            session.refresh(conv)
            return self._conversation_to_dict(conv, include_messages=False)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages."""
        with self.get_session() as session:
            conv = session.query(Conversation).filter(Conversation.id == conversation_id).first()
            if not conv:
                return False
            ws_id = conv.workspace_id
            session.delete(conv)
            session.commit()

            # Ensure workspace has at least one conversation thread
            remaining = session.query(Conversation).filter(Conversation.workspace_id == ws_id).count()
            if remaining == 0:
                new_conv = Conversation(
                    id=str(uuid.uuid4())[:8],
                    workspace_id=ws_id,
                    title="Conversation 1"
                )
                session.add(new_conv)
                session.commit()
            return True

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        confidence_score: float = 0.0,
        is_low_confidence: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Add a message turn to a conversation and touch timestamps."""
        with self.get_session() as session:
            conv = session.query(Conversation).filter(Conversation.id == conversation_id).first()
            if not conv:
                return None

            # Auto-title conversation on first user query if it has a default title
            if role == "user" and (conv.title.startswith("Conversation") or conv.title == "New Conversation"):
                short_title = content.strip().replace("\n", " ")
                if len(short_title) > 36:
                    short_title = short_title[:33] + "..."
                conv.title = short_title

            sources_str = json.dumps(sources) if sources else None
            msg = Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                sources_json=sources_str,
                confidence_score=confidence_score,
                is_low_confidence=is_low_confidence
            )
            conv.updated_at = now_utc()

            # Also update parent workspace updated_at
            ws = session.query(Workspace).filter(Workspace.id == conv.workspace_id).first()
            if ws:
                ws.updated_at = now_utc()

            session.add(msg)
            session.commit()
            session.refresh(msg)
            return self._message_to_dict(msg)

    def clear_conversation_messages(self, conversation_id: str) -> bool:
        """Clear all messages in a conversation."""
        with self.get_session() as session:
            conv = session.query(Conversation).filter(Conversation.id == conversation_id).first()
            if not conv:
                return False
            session.query(Message).filter(Message.conversation_id == conversation_id).delete()
            conv.updated_at = now_utc()
            session.commit()
            return True

    # ==========================================================================
    # Online Document / File Persistence Layer
    # ==========================================================================

    def _migrate_local_manifest_if_needed(self):
        """If database documents table is empty, migrate existing files from local disk/manifest."""
        try:
            with self.get_session() as session:
                count = session.query(Document).count()
                if count > 0:
                    return

                manifest_path = settings.DATA_DIR / "documents" / "manifest.json"
                if not manifest_path.exists():
                    return

                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)

                if not manifest:
                    return

                for doc_id, item in manifest.items():
                    text_path = settings.DATA_DIR / "documents" / f"{doc_id}.txt"
                    full_text = ""
                    if text_path.exists():
                        try:
                            with open(text_path, "r", encoding="utf-8", errors="replace") as tf:
                                full_text = tf.read()
                        except Exception:
                            pass

                    doc = Document(
                        id=doc_id,
                        workspace_id="default",
                        title=item.get("title", doc_id),
                        filename=item.get("filename", f"{doc_id}.txt"),
                        source_type=item.get("source_type", "arxiv"),
                        authors=item.get("authors", "Unknown"),
                        citation_count=int(item.get("citation_count", 0)),
                        chunk_count=int(item.get("chunk_count", 0)),
                        char_count=len(full_text) or int(item.get("char_count", 0)),
                        sections_json=json.dumps(item.get("sections", ["General"])),
                        full_text=full_text,
                        created_at=now_utc(),
                        updated_at=now_utc()
                    )
                    session.add(doc)
                session.commit()
                logger.info(f"Successfully migrated {len(manifest)} local documents into online database table 'documents'.")
        except Exception as e:
            logger.warning(f"Could not migrate local manifest to database: {e}")

    def register_document(
        self,
        doc_id: str,
        workspace_id: str,
        title: str,
        full_text: str,
        source_type: str = "upload",
        filename: Optional[str] = None,
        authors: str = "Unknown",
        citation_count: int = 0,
        chunk_count: int = 0,
        sections: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Save full document context and metadata directly into the database under a workspace."""
        with self.get_session() as session:
            # Verify or default workspace
            ws = session.query(Workspace).filter(Workspace.id == workspace_id).first()
            if not ws:
                ws = session.query(Workspace).filter(Workspace.id == "default").first()
                if not ws:
                    ws = Workspace(
                        id="default",
                        name="General Research",
                        description="Default workspace for multi-source research & synthesis."
                    )
                    session.add(ws)
                    session.commit()
                    session.refresh(ws)
                workspace_id = ws.id

            sections_clean = list(dict.fromkeys(sections or ["General"]))
            sections_str = json.dumps(sections_clean)
            fname = filename or f"{doc_id}.txt"
            char_count = len(full_text)

            doc = session.query(Document).filter(Document.id == doc_id).first()
            if doc:
                doc.workspace_id = workspace_id
                doc.title = title
                doc.full_text = full_text
                doc.source_type = source_type
                doc.filename = fname
                doc.authors = authors
                doc.citation_count = citation_count
                doc.chunk_count = chunk_count
                doc.char_count = char_count
                doc.sections_json = sections_str
                doc.updated_at = now_utc()
            else:
                doc = Document(
                    id=doc_id,
                    workspace_id=workspace_id,
                    title=title,
                    full_text=full_text,
                    source_type=source_type,
                    filename=fname,
                    authors=authors,
                    citation_count=citation_count,
                    chunk_count=chunk_count,
                    char_count=char_count,
                    sections_json=sections_str,
                    created_at=now_utc(),
                    updated_at=now_utc()
                )
                session.add(doc)

            ws.updated_at = now_utc()
            session.commit()
            session.refresh(doc)
            return self._document_to_dict(doc, include_full_text=False)

    def list_documents(self, workspace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List documents stored in database, optionally scoped to a workspace."""
        with self.get_session() as session:
            q = session.query(Document)
            if workspace_id:
                q = q.filter(Document.workspace_id == workspace_id)
            docs = q.order_by(desc(Document.created_at)).all()
            return [self._document_to_dict(d, include_full_text=False) for d in docs]

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full document text context and metadata directly from database."""
        with self.get_session() as session:
            doc = session.query(Document).filter(Document.id == doc_id).first()
            if not doc:
                return None
            return self._document_to_dict(doc, include_full_text=True)

    def delete_document(self, doc_id: str) -> bool:
        """Delete document from database."""
        with self.get_session() as session:
            doc = session.query(Document).filter(Document.id == doc_id).first()
            if not doc:
                return False
            ws_id = doc.workspace_id
            session.delete(doc)
            ws = session.query(Workspace).filter(Workspace.id == ws_id).first()
            if ws:
                ws.updated_at = now_utc()
            session.commit()
            return True

    def batch_delete_documents(self, doc_ids: List[str]) -> List[str]:
        """Batch delete documents from database."""
        deleted = []
        with self.get_session() as session:
            for did in doc_ids:
                doc = session.query(Document).filter(Document.id == did).first()
                if doc:
                    session.delete(doc)
                    deleted.append(did)
            session.commit()
        return deleted


_workspace_store = None


def get_workspace_store() -> WorkspaceStore:
    """Singleton getter for WorkspaceStore."""
    global _workspace_store
    if _workspace_store is None:
        _workspace_store = WorkspaceStore()
    return _workspace_store
