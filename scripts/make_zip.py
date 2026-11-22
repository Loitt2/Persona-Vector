"""Packt die Abgabe-Zip. Lässt Umgebung, API-Key, Git-Daten und den großen, ungenutzten Datensatz weg."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT.parent / "Persona-Vector_Abgabe.zip"
SKIP_DIRS = {".git", ".venv", "__pycache__", ".ipynb_checkpoints", ".ruff_cache"}
SKIP_FILES = {".env", ".git", "dataset.zip", "VORTRAG.md", "ZUSAMMENFASSUNG.md", "CODE_RUNDGANG.md"}  # dataset.zip: 58 MB Finetuning-Daten des Original-Repos, hier nicht verwendet
SKIP_SUFFIXES = {".one", ".pyc"}

with zipfile.ZipFile(TARGET, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or SKIP_DIRS & set(relative.parts):
            continue
        if path.name in SKIP_FILES or path.suffix in SKIP_SUFFIXES:
            continue
        archive.write(path, Path(ROOT.name) / relative)

print(f"{TARGET} | {TARGET.stat().st_size / 1e6:.1f} MB | {len(zipfile.ZipFile(TARGET).namelist())} Dateien")
