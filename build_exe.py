"""Automated Build Script for Research RAG Desktop Executable.

Runs PyInstaller using build_desktop.spec, creates dist/ResearchRAG bundle,
and sets up initial runtime folders and sample configuration.
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

# Force UTF-8 stdout if possible on Windows
if sys.platform == "win32" and sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    root_dir = Path(__file__).resolve().parent
    os.chdir(root_dir)
    print("=" * 65)
    print(" [*] Building Research RAG Intelligence Studio Desktop App")
    print("=" * 65)
    print(f"Working Directory: {root_dir}")

    # Check PyInstaller availability
    try:
        import PyInstaller
        print(f"Detected PyInstaller Version: {PyInstaller.__version__}")
    except ImportError:
        print("[!] PyInstaller not found. Installing pyinstaller and pywebview...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller", "pywebview"])

    # Clean previous build artifacts
    build_dir = root_dir / "build"
    dist_dir = root_dir / "dist"
    if build_dir.exists():
        print("Cleaning previous build/ cache...")
        shutil.rmtree(build_dir, ignore_errors=True)

    # Execute PyInstaller
    spec_path = root_dir / "build_desktop.spec"
    dist_dir = root_dir / "dist"
    build_dir = root_dir / "build"
    print(f"Compiling application with {spec_path.name}...")
    cmd = [
        sys.executable, "-m", "PyInstaller",
        str(spec_path),
        "--noconfirm",
        "--clean",
        "--distpath", str(dist_dir),
        "--workpath", str(build_dir),
    ]
    
    result = subprocess.run(cmd, cwd=root_dir)
    if result.returncode != 0:
        print("\n[!] Build failed! Review compiler logs above.")
        sys.exit(result.returncode)

    output_app_dir = dist_dir / "ResearchRAG"
    output_exe = output_app_dir / "ResearchRAG.exe"

    if output_exe.exists():
        # Setup initial runtime data directories inside output app folder
        (output_app_dir / "data" / "documents").mkdir(parents=True, exist_ok=True)
        (output_app_dir / "data" / "chroma_db").mkdir(parents=True, exist_ok=True)
        (output_app_dir / "data" / "downloads").mkdir(parents=True, exist_ok=True)

        # Copy sample .env if exists
        env_sample = root_dir / ".env"
        target_env = output_app_dir / ".env"
        if env_sample.exists() and not target_env.exists():
            shutil.copy(env_sample, target_env)

        # Create 1-click launcher batch file
        launcher_bat = output_app_dir / "Launch_Research_RAG.bat"
        with open(launcher_bat, "w", encoding="utf-8") as f:
            f.write("@echo off\n")
            f.write("cd /d \"%~dp0\"\n")
            f.write("start \"\" \"ResearchRAG.exe\"\n")

        print("\n" + "=" * 65)
        print(" [OK] BUILD COMPLETE!")
        print(f" Output Folder: {output_app_dir}")
        print(f" Executable:    {output_exe}")
        print("=" * 65)
        print("\nTo launch:")
        print(f"1. Open the folder: {output_app_dir}")
        print(f"2. Double-click ResearchRAG.exe (or Launch_Research_RAG.bat)")
        print("=" * 65 + "\n")
    else:
        print(f"\n[!] Expected output executable not found at: {output_exe}")
        sys.exit(1)

if __name__ == "__main__":
    main()
