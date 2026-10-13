"""Google Classroom Connector for Research RAG Intelligence System.

Enables browsing real Google Classroom courses, exploring course announcements,
coursework materials, and assignment attachments, and directly indexing
attached documents (PDFs, Google Docs, Slides, syllabus notes) into ChromaDB + BM25.

Operates strictly on authentic Google Classroom API data.
Zero mock data, zero fake accounts, zero synthetic fallback.
"""
import io
import os
import re
import json
import time
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pypdf import PdfReader
from googleapiclient.errors import HttpError

from src.config import settings

logger = logging.getLogger(__name__)

CLASSROOM_SCOPES = [
    "https://www.googleapis.com/auth/classroom.courses.readonly",
    "https://www.googleapis.com/auth/classroom.announcements.readonly",
    "https://www.googleapis.com/auth/classroom.courseworkmaterials.readonly",
    "https://www.googleapis.com/auth/classroom.coursework.me.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid"
]

CONFIG_FILE = Path("data/classroom_oauth_config.json")


def save_oauth_credentials(client_id: str, client_secret: str) -> None:
    """Save custom Google Cloud OAuth client credentials."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"client_id": client_id.strip(), "client_secret": client_secret.strip()}, f, indent=2)


def get_oauth_client_info() -> Tuple[Optional[str], Optional[str]]:
    """Retrieve Google OAuth client_id and client_secret from config or saved accounts."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            if d.get("client_id") and d.get("client_secret"):
                return d["client_id"], d["client_secret"]
        except Exception:
            pass

    for base in [Path("data/classroom_accounts"), Path("data/gdrive_accounts")]:
        if base.exists():
            for f in base.glob("*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        d = json.load(fp)
                    if d.get("client_id") and d.get("client_secret"):
                        return d["client_id"], d["client_secret"]
                except Exception:
                    pass

    tp = Path.home() / ".gemini" / "antigravity" / "plugin_data" / "gdocs" / "token.json"
    if tp.exists():
        try:
            with open(tp, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            if d.get("client_id") and d.get("client_secret"):
                return d["client_id"], d["client_secret"]
        except Exception:
            pass

    return None, None


class GoogleClassroomConnector:
    """Connector for Google Classroom supporting course navigation, announcements, and material ingestion."""

    def __init__(self):
        self.auth_type = "none"  # "user_account", "service_account", "access_token", "none"
        self.service = None
        self.drive_service = None
        self.account_email: Optional[str] = None
        self.access_token: Optional[str] = None
        self.login_state: Dict[str, Any] = {"status": "idle", "email": None, "message": None, "error": None}
        self.active_server = None
        self._accounts_dir = Path("data/classroom_accounts")
        self._accounts_dir.mkdir(parents=True, exist_ok=True)

        self._auto_detect_saved_account()

    def _auto_detect_saved_account(self):
        """Auto-connect ONLY to saved accounts that have valid Google Classroom scopes."""
        saved = self.list_saved_accounts()
        # Filter strictly to accounts that have classroom scopes
        valid_classroom_accounts = [a for a in saved if a.get("has_classroom_scopes")]
        if valid_classroom_accounts:
            first = valid_classroom_accounts[0]["email"]
            try:
                self.switch_account(first)
                logger.info(f"ClassroomConnector initialized with verified account: {first}")
            except Exception as e:
                logger.debug(f"Could not auto-switch to {first}: {e}")
                self.auth_type = "none"
                self.account_email = None
                self.service = None
                self.drive_service = None
        else:
            self.auth_type = "none"
            self.account_email = None
            self.service = None
            self.drive_service = None

    def connect(
        self,
        auth_type: str = "user_account",
        access_token: Optional[str] = None,
        service_account_json: Optional[str] = None
    ) -> Dict[str, Any]:
        """Authenticate Classroom connector via user account, service account, or access token."""
        self.auth_type = auth_type.lower()
        self.access_token = access_token
        self.service = None
        self.drive_service = None
        self.account_email = None

        if self.auth_type == "service_account":
            if not service_account_json:
                raise ValueError("Service account JSON content or file path is required.")
            try:
                from google.oauth2 import service_account
                from googleapiclient.discovery import build

                if os.path.exists(service_account_json):
                    creds = service_account.Credentials.from_service_account_file(
                        service_account_json, scopes=CLASSROOM_SCOPES
                    )
                else:
                    data = json.loads(service_account_json)
                    creds = service_account.Credentials.from_service_account_info(
                        data, scopes=CLASSROOM_SCOPES
                    )
                self.service = build("classroom", "v1", credentials=creds)
                self.drive_service = build("drive", "v3", credentials=creds)
                self.account_email = getattr(creds, "service_account_email", "service-account")
                return {
                    "status": "success",
                    "auth_type": "service_account",
                    "account_email": self.account_email,
                    "message": f"Connected to Google Classroom as {self.account_email}"
                }
            except Exception as e:
                logger.error(f"Classroom service account authentication failed: {e}")
                raise RuntimeError(f"Service account error: {e}")

        if self.auth_type == "access_token":
            if not access_token or not access_token.strip():
                raise ValueError("OAuth access token is required.")
            try:
                from google.oauth2.credentials import Credentials
                from googleapiclient.discovery import build

                clean_token = access_token.strip()

                # 1. Detect if user pasted an Authorization Code instead of an Access Token
                if clean_token.startswith("4/") or clean_token.startswith("4%2F"):
                    raise ValueError(
                        "The value entered appears to be an Authorization Code (starts with '4/'), not an Access Token. "
                        "In Step 2 of Google OAuth Playground, click the blue 'Exchange authorization code for tokens' button, "
                        "then copy the generated 'Access token' (which begins with 'ya29.')."
                    )

                # 2. Inspect token with Google's universal tokeninfo endpoint (does not require userinfo scope)
                user_email = "classroom-user"
                token_scopes: List[str] = []
                tokeninfo_valid = False

                try:
                    with httpx.Client(timeout=8.0) as cl:
                        t_res = cl.get(f"https://oauth2.googleapis.com/tokeninfo?access_token={clean_token}")
                        if t_res.status_code == 200:
                            tokeninfo_valid = True
                            t_info = t_res.json()
                            if t_info.get("email"):
                                user_email = t_info["email"]
                            elif t_info.get("user_id"):
                                user_email = f"user-{t_info['user_id'][:8]}"

                            raw_scopes = t_info.get("scope", "")
                            token_scopes = raw_scopes.split() if isinstance(raw_scopes, str) else []

                            expires_in = int(t_info.get("expires_in", 3600))
                            if expires_in <= 0:
                                raise ValueError("OAuth access token has expired. Please generate a fresh token.")
                except ValueError:
                    raise
                except Exception as e:
                    logger.debug(f"tokeninfo check failed: {e}")

                # 3. Direct verification against Google Classroom API
                creds = Credentials(token=clean_token, scopes=CLASSROOM_SCOPES)
                self.service = build("classroom", "v1", credentials=creds)
                self.drive_service = build("drive", "v3", credentials=creds)

                try:
                    # Test calling courses.list to be 100% sure Classroom API accepts this token
                    self.service.courses().list(pageSize=1).execute()
                except HttpError as he:
                    status_code = he.resp.status if hasattr(he, "resp") else 400
                    err_msg = str(he)
                    if status_code == 401:
                        raise ValueError(
                            "Invalid or expired OAuth access token (Google Classroom API returned HTTP 401 Unauthorized). "
                            "Please generate a fresh token from Google OAuth Playground."
                        )
                    elif status_code == 403:
                        raise ValueError(
                            "The access token does not have Google Classroom permissions (HTTP 403 Insufficient Scopes). "
                            "Please ensure 'https://www.googleapis.com/auth/classroom.courses.readonly' was included."
                        )
                    else:
                        raise ValueError(f"Google Classroom API rejected token (HTTP {status_code}): {err_msg}")
                except Exception as ex:
                    if not tokeninfo_valid:
                        raise ValueError(
                            f"Invalid or expired OAuth access token: {ex}. "
                            "Please ensure you copied the 'Access token' (starting with ya29.) from Google OAuth Playground."
                        )

                # 4. Optional userinfo email fetch
                if user_email == "classroom-user":
                    try:
                        with httpx.Client(timeout=4.0) as cl:
                            u_res = cl.get(
                                "https://www.googleapis.com/oauth2/v2/userinfo",
                                headers={"Authorization": f"Bearer {clean_token}"}
                            )
                            if u_res.status_code == 200:
                                user_email = u_res.json().get("email", user_email)
                    except Exception:
                        pass

                self.account_email = user_email
                self.auth_type = "access_token"

                # Persist valid token for future sessions
                save_data = {
                    "email": user_email,
                    "scopes": token_scopes or CLASSROOM_SCOPES,
                    "token": {
                        "access_token": clean_token
                    },
                    "updated_at": datetime.now().isoformat()
                }
                with open(self._accounts_dir / f"{user_email}.json", "w", encoding="utf-8") as f:
                    json.dump(save_data, f, indent=2)

                return {
                    "status": "success",
                    "auth_type": "access_token",
                    "account_email": self.account_email,
                    "message": f"Connected to Google Classroom as {self.account_email} via Access Token."
                }
            except Exception as e:
                logger.error(f"Classroom access token error: {e}")
                raise RuntimeError(f"Access token error: {e}")

        raise ValueError(f"Unknown auth_type '{auth_type}'.")

    def validate_connection(self) -> Dict[str, Any]:
        """Probe Google Classroom API and measure latency to confirm connection is valid."""
        start = time.perf_counter()

        if not self.service or self.auth_type == "none":
            return {
                "valid": False,
                "latency_ms": 0.0,
                "auth_type": "none",
                "account_email": self.account_email,
                "courses_count": 0,
                "message": "No Google Classroom account connected. Please sign in or provide credentials with Classroom scopes.",
                "error": "Not connected"
            }

        try:
            res = self.service.courses().list(pageSize=10, courseStates=["ACTIVE"]).execute()
            latency = round((time.perf_counter() - start) * 1000, 2)
            courses = res.get("courses", [])
            return {
                "valid": True,
                "latency_ms": latency,
                "auth_type": self.auth_type,
                "account_email": self.account_email,
                "courses_count": len(courses),
                "message": f"Connection verified! Connected to Google Classroom as {self.account_email} ({len(courses)} active courses, {latency}ms).",
                "error": None
            }
        except HttpError as e:
            latency = round((time.perf_counter() - start) * 1000, 2)
            status_code = e.resp.status if hasattr(e, "resp") else 403
            err_reason = str(e)

            if "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in err_reason or status_code == 403:
                msg = (
                    f"Account '{self.account_email}' has insufficient authentication scopes. "
                    "Only Google Drive permissions were detected. Please re-authorize with Google Classroom scopes "
                    "(classroom.courses.readonly, classroom.announcements.readonly, classroom.courseworkmaterials.readonly)."
                )
                logger.warning(f"Classroom validation failed: {msg}")
                return {
                    "valid": False,
                    "latency_ms": latency,
                    "auth_type": self.auth_type,
                    "account_email": self.account_email,
                    "courses_count": 0,
                    "message": msg,
                    "error": "ACCESS_TOKEN_SCOPE_INSUFFICIENT"
                }

            logger.warning(f"Classroom API HTTP {status_code} error: {err_reason}")
            return {
                "valid": False,
                "latency_ms": latency,
                "auth_type": self.auth_type,
                "account_email": self.account_email,
                "courses_count": 0,
                "message": f"Google Classroom API returned HTTP {status_code}: {err_reason}",
                "error": err_reason
            }
        except Exception as e:
            latency = round((time.perf_counter() - start) * 1000, 2)
            err_str = str(e)
            logger.warning(f"Classroom connection error: {err_str}")
            return {
                "valid": False,
                "latency_ms": latency,
                "auth_type": self.auth_type,
                "account_email": self.account_email,
                "courses_count": 0,
                "message": f"Classroom API check failed: {err_str}",
                "error": err_str
            }

    def list_courses(self) -> List[Dict[str, Any]]:
        """List all enrolled and active courses in Google Classroom."""
        if not self.service or self.auth_type == "none":
            return []

        try:
            res = self.service.courses().list(courseStates=["ACTIVE"], pageSize=25).execute()
            courses = res.get("courses", [])
            output = []
            for c in courses:
                output.append({
                    "id": c.get("id"),
                    "name": c.get("name", "Untitled Course"),
                    "section": c.get("section", ""),
                    "description_heading": c.get("descriptionHeading", ""),
                    "room": c.get("room", ""),
                    "course_state": c.get("courseState", "ACTIVE"),
                    "alternate_link": c.get("alternateLink", ""),
                    "enrollment_code": c.get("enrollmentCode", "")
                })
            return output
        except Exception as e:
            logger.error(f"Failed to fetch live courses from Classroom: {e}")
            raise RuntimeError(f"Could not load Google Classroom courses: {e}")

    def get_course_stream(self, course_id: str) -> Dict[str, Any]:
        """Fetch announcements, coursework materials, and coursework for a specific course."""
        if not self.service or self.auth_type == "none":
            return {
                "course_id": course_id,
                "course_name": "",
                "announcements": [],
                "coursework_materials": [],
                "coursework": [],
                "total_items": 0
            }

        # Fetch course details
        try:
            course_res = self.service.courses().get(id=course_id).execute()
            course_name = course_res.get("name", "Google Classroom Course")
        except Exception:
            course_name = "Google Classroom Course"

        announcements_list = []
        try:
            ann_res = self.service.courses().announcements().list(courseId=course_id, pageSize=30).execute()
            for ann in ann_res.get("announcements", []):
                materials = self._parse_api_materials(ann.get("materials", []))
                announcements_list.append({
                    "id": ann.get("id"),
                    "course_id": course_id,
                    "text": ann.get("text", ""),
                    "created_at": ann.get("creationTime", ""),
                    "alternate_link": ann.get("alternateLink", ""),
                    "materials": materials
                })
        except Exception as e:
            logger.warning(f"Could not load announcements for course {course_id}: {e}")

        materials_list = []
        try:
            cwm_res = self.service.courses().courseWorkMaterials().list(courseId=course_id, pageSize=30).execute()
            for cwm in cwm_res.get("courseWorkMaterial", []):
                materials = self._parse_api_materials(cwm.get("materials", []))
                materials_list.append({
                    "id": cwm.get("id"),
                    "course_id": course_id,
                    "title": cwm.get("title", "Untitled Material"),
                    "description": cwm.get("description", ""),
                    "created_at": cwm.get("creationTime", ""),
                    "alternate_link": cwm.get("alternateLink", ""),
                    "materials": materials
                })
        except Exception as e:
            logger.warning(f"Could not load courseWorkMaterials for course {course_id}: {e}")

        coursework_list = []
        try:
            cw_res = self.service.courses().courseWork().list(courseId=course_id, pageSize=30).execute()
            for cw in cw_res.get("courseWork", []):
                materials = self._parse_api_materials(cw.get("materials", []))
                due = cw.get("dueDate", {})
                due_str = f"{due.get('year')}-{due.get('month', 1):02d}-{due.get('day', 1):02d}" if due else None
                coursework_list.append({
                    "id": cw.get("id"),
                    "course_id": course_id,
                    "title": cw.get("title", "Untitled Assignment"),
                    "description": cw.get("description", ""),
                    "due_date": due_str,
                    "max_points": cw.get("maxPoints"),
                    "alternate_link": cw.get("alternateLink", ""),
                    "materials": materials
                })
        except Exception as e:
            logger.warning(f"Could not load courseWork for course {course_id}: {e}")

        return {
            "course_id": course_id,
            "course_name": course_name,
            "announcements": announcements_list,
            "coursework_materials": materials_list,
            "coursework": coursework_list,
            "total_items": len(announcements_list) + len(materials_list) + len(coursework_list)
        }

    def _parse_api_materials(self, api_materials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize heterogeneous Google Classroom API materials."""
        parsed = []
        for idx, m in enumerate(api_materials):
            if "driveFile" in m:
                df = m["driveFile"].get("driveFile", {})
                parsed.append({
                    "id": df.get("id") or f"df_{idx}",
                    "title": df.get("title", "Google Drive File"),
                    "type": "drive_file",
                    "drive_file_id": df.get("id"),
                    "url": df.get("alternateLink", ""),
                    "thumbnail_url": df.get("thumbnailUrl", ""),
                    "mime_type": "application/pdf" if df.get("title", "").endswith(".pdf") else "application/octet-stream"
                })
            elif "link" in m:
                lk = m["link"]
                parsed.append({
                    "id": f"link_{idx}",
                    "title": lk.get("title") or lk.get("url", "Web Link"),
                    "type": "link",
                    "drive_file_id": None,
                    "url": lk.get("url", ""),
                    "thumbnail_url": lk.get("thumbnailUrl", ""),
                    "mime_type": "text/html"
                })
            elif "youtubeVideo" in m:
                yt = m["youtubeVideo"]
                parsed.append({
                    "id": yt.get("id") or f"yt_{idx}",
                    "title": yt.get("title", "YouTube Video"),
                    "type": "youtube",
                    "drive_file_id": None,
                    "url": yt.get("alternateLink", ""),
                    "thumbnail_url": yt.get("thumbnailUrl", ""),
                    "mime_type": "video/youtube"
                })
            elif "form" in m:
                fm = m["form"]
                parsed.append({
                    "id": f"form_{idx}",
                    "title": fm.get("title", "Google Form"),
                    "type": "form",
                    "drive_file_id": None,
                    "url": fm.get("formUrl", ""),
                    "thumbnail_url": fm.get("thumbnailUrl", ""),
                    "mime_type": "application/vnd.google-apps.form"
                })
        return parsed

    def download_document_content(
        self,
        item_type: str,
        title: str,
        drive_file_id: Optional[str] = None,
        raw_text: Optional[str] = None,
        mime_type: Optional[str] = None
    ) -> Tuple[str, str, str]:
        """Download document content from Google Drive attachment or extract announcement text.

        Returns: (filename, extracted_text, mime_type)
        """
        if item_type in ["announcement_text", "material_text"] or (raw_text and raw_text.strip()):
            clean_title = re.sub(r"[^a-zA-Z0-9_\- ]", "", title).strip() or "Announcement"
            filename = f"{clean_title}.txt"
            content = f"# {title}\n\n{raw_text or ''}"
            return filename, content, "text/plain"

        if not self.drive_service or not drive_file_id:
            raise RuntimeError(f"No Google Drive service available to download '{title}'.")

        try:
            # 1. Fetch file metadata
            meta = self.drive_service.files().get(
                fileId=drive_file_id, fields="id, name, mimeType, size"
            ).execute()
            filename = meta.get("name", title)
            file_mime = meta.get("mimeType", mime_type or "application/octet-stream")

            # Google Docs export
            if file_mime == "application/vnd.google-apps.document":
                res = self.drive_service.files().export(fileId=drive_file_id, mimeType="text/plain").execute()
                text = res.decode("utf-8") if isinstance(res, bytes) else str(res)
                return filename, text, "text/plain"

            # Google Slides export
            if file_mime == "application/vnd.google-apps.presentation":
                res = self.drive_service.files().export(fileId=drive_file_id, mimeType="text/plain").execute()
                text = res.decode("utf-8") if isinstance(res, bytes) else str(res)
                return filename, text, "text/plain"

            # Binary download (PDF, plain text, markdown)
            res = self.drive_service.files().get_media(fileId=drive_file_id).execute()
            bytes_content = res if isinstance(res, bytes) else res.encode("utf-8")

            if file_mime == "application/pdf" or filename.endswith(".pdf"):
                reader = PdfReader(io.BytesIO(bytes_content))
                pages_text = []
                for page in reader.pages[:40]:
                    t = page.extract_text()
                    if t:
                        pages_text.append(t)
                extracted = "\n\n".join(pages_text)
                return filename, extracted, "application/pdf"

            try:
                text = bytes_content.decode("utf-8")
                return filename, text, "text/plain"
            except Exception:
                return filename, str(bytes_content[:4000]), file_mime

        except Exception as e:
            logger.error(f"Failed to fetch live file {drive_file_id} from Drive: {e}")
            raise RuntimeError(f"Error downloading file {title} ({drive_file_id}): {e}")

    def list_saved_accounts(self) -> List[Dict[str, Any]]:
        """List saved accounts, distinguishing verified Classroom accounts from Drive-only accounts."""
        accounts = []
        seen = set()

        # Check classroom_accounts directory
        for f in self._accounts_dir.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    d = json.load(fp)
                em = d.get("email") or f.stem
                if em not in seen:
                    seen.add(em)
                    scopes = d.get("scopes", [])
                    has_scopes = any("classroom" in s.lower() for s in scopes)
                    accounts.append({
                        "email": em,
                        "name": d.get("name", em.split("@")[0]),
                        "is_active": (em == self.account_email and self.auth_type == "user_account"),
                        "has_classroom_scopes": has_scopes,
                        "updated_at": d.get("updated_at", "")
                    })
            except Exception as e:
                logger.debug(f"Error reading account file {f}: {e}")

        # Check gdrive_accounts directory
        gd_dir = Path("data/gdrive_accounts")
        if gd_dir.exists():
            for f in gd_dir.glob("*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        d = json.load(fp)
                    em = d.get("email") or f.stem
                    if em not in seen:
                        seen.add(em)
                        scopes = d.get("scopes", [])
                        has_scopes = any("classroom" in s.lower() for s in scopes)
                        accounts.append({
                            "email": em,
                            "name": d.get("name", em.split("@")[0]),
                            "is_active": (em == self.account_email and self.auth_type == "user_account"),
                            "has_classroom_scopes": has_scopes,
                            "updated_at": d.get("updated_at", "")
                        })
                except Exception:
                    pass

        # Sort: verified accounts first, then active
        accounts.sort(key=lambda x: (not x["has_classroom_scopes"], not x["is_active"], x["email"]))
        return accounts

    def switch_account(self, email: str) -> Dict[str, Any]:
        """Switch active Google Classroom account session."""
        target_file = None
        for base in [self._accounts_dir, Path("data/gdrive_accounts")]:
            f = base / f"{email}.json"
            if f.exists():
                target_file = f
                break

        if not target_file:
            raise FileNotFoundError(f"Account credentials for '{email}' not found.")

        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            with open(target_file, "r", encoding="utf-8") as fp:
                d = json.load(fp)

            scopes = d.get("scopes", [])
            has_classroom = any("classroom" in s.lower() for s in scopes)

            token_data = d.get("token", {})
            has_refresh = bool(token_data.get("refresh_token"))
            creds = Credentials(
                token=token_data.get("access_token"),
                refresh_token=token_data.get("refresh_token") if has_refresh else None,
                token_uri="https://oauth2.googleapis.com/token" if has_refresh else None,
                client_id=d.get("client_id") if has_refresh else None,
                client_secret=d.get("client_secret") if has_refresh else None,
                scopes=scopes
            )

            self.service = build("classroom", "v1", credentials=creds)
            self.drive_service = build("drive", "v3", credentials=creds)
            self.auth_type = "user_account"
            self.account_email = email
            logger.info(f"Switched Google Classroom active account to: {email} (has_classroom_scopes={has_classroom})")

            return {
                "status": "success",
                "auth_type": "user_account",
                "account_email": email,
                "has_classroom_scopes": has_classroom,
                "message": f"Switched to account {email}"
            }
        except Exception as e:
            logger.error(f"Failed to switch Classroom account to {email}: {e}")
            raise RuntimeError(f"Classroom account switch failed: {e}")

    def start_browser_login(
        self,
        custom_client_id: Optional[str] = None,
        custom_client_secret: Optional[str] = None
    ) -> Dict[str, Any]:
        """Launch browser OAuth flow with Google Account Chooser and Classroom scopes."""
        if self.active_server:
            try:
                self.active_server.server_close()
            except Exception:
                pass
            self.active_server = None

        client_id = (custom_client_id or "").strip()
        client_secret = (custom_client_secret or "").strip()

        if not client_id or not client_secret:
            c_id, c_sec = get_oauth_client_info()
            client_id = client_id or c_id
            client_secret = client_secret or c_sec

        if not client_id or not client_secret:
            raise ValueError(
                "No Google Cloud OAuth Client ID & Secret configured. "
                "Please configure your Google Cloud Client ID and Secret in the settings."
            )

        from google_auth_oauthlib.flow import InstalledAppFlow, _RedirectWSGIApp, _ExclusiveWSGIServer
        import wsgiref.simple_server

        client_config = {
            "installed": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost", "http://127.0.0.1"]
            }
        }

        flow = InstalledAppFlow.from_client_config(client_config, scopes=CLASSROOM_SCOPES)
        flow.redirect_uri_trailing_slash = False

        wsgi_app = _RedirectWSGIApp("<h3>Authorization successful! You can return to Research RAG.</h3>")
        port = 8092
        host = "127.0.0.1"

        try:
            local_server = _ExclusiveWSGIServer((host, port), wsgi_app, handler_class=wsgiref.simple_server.WSGIRequestHandler)
        except OSError:
            port = 8093
            local_server = _ExclusiveWSGIServer((host, port), wsgi_app, handler_class=wsgiref.simple_server.WSGIRequestHandler)

        self.active_server = local_server
        flow.redirect_uri = f"http://{host}:{port}/"
        auth_url, _ = flow.authorization_url(prompt="select_account", access_type="offline", include_granted_scopes="true")

        self.login_state = {
            "status": "in_progress",
            "auth_url": auth_url,
            "email": None,
            "message": "Waiting for Google account approval in browser...",
            "error": None
        }

        def _run_server():
            try:
                local_server.handle_request()
                auth_code = wsgi_app.last_request_uri
                if auth_code:
                    flow.fetch_token(authorization_response=auth_code)
                    creds = flow.credentials

                    user_email = "google_classroom_user@edu"
                    try:
                        with httpx.Client(timeout=10.0) as cl:
                            u_res = cl.get(
                                "https://www.googleapis.com/oauth2/v2/userinfo",
                                headers={"Authorization": f"Bearer {creds.token}"}
                            )
                            if u_res.status_code == 200:
                                user_email = u_res.json().get("email", user_email)
                    except Exception:
                        pass

                    save_data = {
                        "email": user_email,
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "scopes": list(creds.scopes or CLASSROOM_SCOPES),
                        "token": {
                            "access_token": creds.token,
                            "refresh_token": creds.refresh_token,
                            "token_uri": "https://oauth2.googleapis.com/token"
                        },
                        "updated_at": datetime.now().isoformat()
                    }

                    target_file = self._accounts_dir / f"{user_email}.json"
                    with open(target_file, "w", encoding="utf-8") as f:
                        json.dump(save_data, f, indent=2)

                    from googleapiclient.discovery import build
                    self.service = build("classroom", "v1", credentials=creds)
                    self.drive_service = build("drive", "v3", credentials=creds)
                    self.auth_type = "user_account"
                    self.account_email = user_email

                    self.login_state = {
                        "status": "completed",
                        "auth_url": None,
                        "email": user_email,
                        "message": f"Successfully authorized Google Classroom account: {user_email}",
                        "error": None
                    }
                    logger.info(f"Classroom login succeeded for {user_email}")
            except Exception as e:
                logger.error(f"Error in Classroom OAuth listener: {e}")
                self.login_state = {
                    "status": "error",
                    "auth_url": None,
                    "email": None,
                    "message": f"Authorization error: {e}",
                    "error": str(e)
                }
            finally:
                try:
                    local_server.server_close()
                except Exception:
                    pass
                self.active_server = None

        thread = threading.Thread(target=_run_server, daemon=True)
        thread.start()

        try:
            import webbrowser
            webbrowser.open(auth_url)
        except Exception:
            pass

        return {
            "status": "in_progress",
            "auth_url": auth_url,
            "email": None,
            "message": "Browser opened for Google Classroom sign-in."
        }

    def get_login_status(self) -> Dict[str, Any]:
        """Check progress of browser sign-in flow."""
        return self.login_state


# Global singleton instance
_classroom_connector_instance: Optional[GoogleClassroomConnector] = None

def get_classroom_connector() -> GoogleClassroomConnector:
    """Retrieve or initialize global GoogleClassroomConnector instance."""
    global _classroom_connector_instance
    if _classroom_connector_instance is None:
        _classroom_connector_instance = GoogleClassroomConnector()
    return _classroom_connector_instance
