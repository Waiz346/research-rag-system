"""Desktop Application Runner for Research RAG Intelligence Studio.

Launches both FastAPI and Streamlit in background threads, then opens a
native desktop window using Microsoft Edge WebView2 (via pywebview).
Gracefully falls back to the default web browser if WebView2 is unavailable.
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
        time.sleep(0.3)
    return False

def start_fastapi_server(port: int):
    """Run FastAPI server in a background daemon thread."""
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

def start_streamlit_server(port: int, api_url: str):
    """Run Streamlit frontend in a background daemon thread with thread-safe signal overrides."""
    try:
        import streamlit.web.bootstrap as bootstrap
        from streamlit import config as st_config

        # Crucial for running Streamlit inside a worker thread
        bootstrap._set_up_signal_handler = lambda server: None

        st_config.set_option("server.port", port)
        st_config.set_option("server.headless", True)
        st_config.set_option("server.fileWatcherType", "none")
        st_config.set_option("global.developmentMode", False)
        st_config.set_option("client.toolbarMode", "minimal")
        st_config.set_option("browser.gatherUsageStats", False)

        app_script = BUNDLE_DIR / "frontend" / "app.py"
        if not app_script.exists():
            raise FileNotFoundError(f"Frontend app script not found at: {app_script}")

        os.environ["API_URL"] = api_url
        bootstrap.run(str(app_script), False, [], {})
    except Exception as e:
        logger.error(f"Streamlit server encountered an error: {e}", exc_info=True)

def main():
    logger.info("Initializing Research RAG Desktop Studio...")
    
    # 1. Allocate available ports
    api_port = find_free_port(8000)
    st_port = find_free_port(8501)
    
    api_url = f"http://127.0.0.1:{api_port}"
    st_url = f"http://127.0.0.1:{st_port}"
    os.environ["API_URL"] = api_url

    logger.info(f"Starting Backend API on {api_url}...")
    api_thread = threading.Thread(target=start_fastapi_server, args=(api_port,), daemon=True)
    api_thread.start()

    logger.info(f"Starting Streamlit Studio on {st_url}...")
    st_thread = threading.Thread(target=start_streamlit_server, args=(st_port, api_url), daemon=True)
    st_thread.start()

    # 2. Wait for servers to become ready
    logger.info("Waiting for servers to initialize...")
    if not wait_for_url(f"{api_url}/health", timeout_seconds=12.0):
        logger.warning("Backend API took longer than expected to start, continuing anyway...")
    if not wait_for_url(f"{st_url}/_stcore/health", timeout_seconds=15.0):
        logger.warning("Streamlit took longer than expected to start, continuing anyway...")

    logger.info("Servers ready. Launching desktop application window...")

    # 3. Launch Native Window (PyWebView)
    try:
        import webview
        
        window = webview.create_window(
            title="Research RAG Intelligence Studio",
            url=st_url,
            width=1380,
            height=880,
            min_size=(960, 640),
            confirm_close=False,
            text_select=True
        )
        # webview.start blocks until the window is closed
        webview.start()
        logger.info("Desktop window closed. Exiting application.")
    except Exception as e:
        logger.warning(f"Native PyWebView window could not open ({e}). Opening in default browser...")
        import webbrowser
        webbrowser.open(st_url)
        print("\n" + "=" * 60)
        print(" Research RAG Studio is running!")
        print(f" URL: {st_url}")
        print(" Press Ctrl+C in this terminal to exit.")
        print("=" * 60 + "\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down servers...")

    sys.exit(0)

if __name__ == "__main__":
    main()
