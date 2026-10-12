# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification file for Research RAG Intelligence Studio."""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None

import os
# Base repository directory (SPECPATH is injected by PyInstaller)
ROOT_DIR = Path(SPECPATH).resolve()
os.chdir(ROOT_DIR)

# 1. Collect all package data and hidden imports for complex dynamic packages
streamlit_datas, streamlit_binaries, streamlit_hidden = collect_all('streamlit')
chromadb_datas, chromadb_binaries, chromadb_hidden = collect_all('chromadb')
webview_datas, webview_binaries, webview_hidden = collect_all('webview')
uvicorn_datas, uvicorn_binaries, uvicorn_hidden = collect_all('uvicorn')

# 2. Local application source data
app_datas = [
    (str(ROOT_DIR / 'frontend'), 'frontend'),
    (str(ROOT_DIR / 'src'), 'src'),
]

# Merge datas and binaries
all_datas = app_datas + streamlit_datas + chromadb_datas + webview_datas + uvicorn_datas
all_binaries = streamlit_binaries + chromadb_binaries + webview_binaries + uvicorn_binaries

# 3. Explicit hidden imports to prevent runtime missing-module exceptions
all_hiddenimports = list(set(
    streamlit_hidden + chromadb_hidden + webview_hidden + uvicorn_hidden + [
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'fastapi',
        'starlette',
        'pydantic',
        'pydantic_settings',
        'pypdf',
        'rank_bm25',
        'arxiv',
        'requests',
        'httpx',
        'dotenv',
        'clr_loader',
        'pythonnet',
        'webview.platforms.winforms',
        'webview.platforms.edgechromium',
        'src.api.main',
        'src.api.schemas',
        'src.indexing.vector_store',
        'src.indexing.document_catalog',
        'src.ingestion.chunker',
        'src.ingestion.arxiv_collector',
        'src.ingestion.gdrive_connector',
        'googleapiclient',
        'google.auth',
        'google.oauth2',
        'src.rag.generator',
        'src.rag.pipeline',
        'src.evals.evaluator',
        'src.config',
    ]
))

a = Analysis(
    ['desktop_app.py'],
    pathex=[str(ROOT_DIR)],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'pytest',
        'IPython',
        'jupyter',
        'matplotlib',
        'tkinter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ResearchRAG',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Keep console enabled initially so any unexpected launch logs are visible
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ResearchRAG',
)
