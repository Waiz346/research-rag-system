"""Desktop Application Runner for NexusRAG Research Intelligence Studio.

Launches the unified FastAPI backend + standalone HTML5/ES6 Workbench in a
background thread, then opens a native desktop window using Microsoft Edge
WebView2 (via pywebview). Gracefully falls back to the default web browser
if WebView2 is unavailable.
"""
import os
import sys
import time
import socket
import logging
import threading
import urllib.request
from pathlib import Path

# Configure paths for development or PyInstaller frozen bundle
if getattr(sys, "frozen", False):
    BUNDLE_DIR = Path(sys._MEIPASS)
    BASE_DIR = Path(sys.executable).parent
else:
    BUNDLE_DIR = Path(__file__).resolve().parent
    BASE_DIR = BUNDLE_DIR

# Ensure persistent data directories exist next to the application
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "documents").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "downloads").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "chroma_db").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DesktopApp")


def find_free_port(preferred_port: int, max_attempts: int = 50) -> int:
    """Find an available localhost port starting from preferred_port."""
    for port in range(preferred_port, preferred_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred_port


def wait_for_url(url: str, timeout_seconds: float = 15.0) -> bool:
    """Poll URL until it responds with HTTP 200 or timeout expires."""
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HealthCheck/1.0"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def start_fastapi_server(port: int):
    """Run FastAPI server (serving both API and standalone UI) in a background daemon thread."""
    try:
        import uvicorn
        from src.api.main import app as fastapi_app

        config = uvicorn.Config(
            app=fastapi_app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False
        )
        server = uvicorn.Server(config)
        server.run()
    except Exception as e:
        logger.error(f"FastAPI server encountered an error: {e}", exc_info=True)


def main():
    logger.info("Initializing NexusRAG Desktop Research Workbench...")

    # 1. Allocate available port for unified FastAPI + Web Workbench server
    api_port = find_free_port(8000)
    app_url = f"http://127.0.0.1:{api_port}"
    os.environ["API_URL"] = app_url

    logger.info(f"Starting Unified NexusRAG Server on {app_url}...")
    api_thread = threading.Thread(target=start_fastapi_server, args=(api_port,), daemon=True)
    api_thread.start()

    # 2. Wait for server to become ready
    logger.info("Waiting for server to initialize...")
    if not wait_for_url(f"{app_url}/health", timeout_seconds=12.0):
        logger.warning("Server took longer than expected to start, continuing anyway...")

    logger.info("Server ready. Launching desktop application window...")

    # 3. Launch Native Window (PyWebView)
    try:
        import webview

        webview.create_window(
            title="NexusRAG // Interactive Web RAG Research Workbench",
            url=app_url,
            width=1440,
            height=900,
            min_size=(960, 640),
            confirm_close=False,
            text_select=True
        )
        webview.start()
        logger.info("Desktop window closed. Exiting application.")
    except Exception as e:
        logger.warning(f"Native PyWebView window could not open ({e}). Opening in default browser...")
        import webbrowser
        webbrowser.open(app_url)
        print("\n" + "=" * 60)
        print(" NexusRAG Research Workbench is running!")
        print(f" URL: {app_url}")
        print(" Press Ctrl+C in this terminal to exit.")
        print("=" * 60 + "\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down server...")

    sys.exit(0)


if __name__ == "__main__":
    main()
