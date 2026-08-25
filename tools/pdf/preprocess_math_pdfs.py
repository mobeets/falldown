"""Math-preserving PDF queue processor.

Converts research PDFs from papers/queue/ into LaTeX-preserving Markdown in
papers/md/ and archives the source PDFs in papers/archive/.

Run with the interpreter that has marker-pdf installed (Python 3.11 here):
    C:\\Users\\manik\\AppData\\Local\\Programs\\Python\\Python311\\python.exe tools/pdf/preprocess_math_pdfs.py
"""

import os
import shutil
from pathlib import Path

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
QUEUE_DIR = _REPO_ROOT / "papers" / "queue"
ARCHIVE_DIR = _REPO_ROOT / "papers" / "archive"
MD_DIR = _REPO_ROOT / "papers" / "md"


def process_math_paper_queue(
    queue_dir: Path | str = QUEUE_DIR,
    archive_dir: Path | str = ARCHIVE_DIR,
    md_dir: Path | str = MD_DIR,
):
    queue_dir = Path(queue_dir)
    archive_dir = Path(archive_dir)
    md_dir = Path(md_dir)

    # Create required directories if they don't exist
    queue_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = [f for f in os.listdir(queue_dir) if f.endswith(".pdf")]

    if not pdf_files:
        print("No new PDFs found in ./papers/queue/")
        return

    print(f"Found {len(pdf_files)} paper(s) to convert...")
    converter = PdfConverter(artifact_dict=create_model_dict())

    for filename in pdf_files:
        pdf_path = queue_dir / filename

        # Convert layout and parse equations into LaTeX
        rendered = converter(str(pdf_path))
        full_text, _, _ = text_from_rendered(rendered)

        # Save Markdown file
        out_filename = Path(filename).stem + ".md"
        md_path = md_dir / out_filename
        md_path.write_text(full_text, encoding="utf-8")

        # Move converted PDF to the archive folder
        shutil.move(str(pdf_path), str(archive_dir / filename))
        print(f"Successfully processed {filename} -> {out_filename} (PDF archived)")

    print("\nDone. Next step: run `graphify extract . --backend gemini --max-concurrency 1` "
          "(from the repo root) to ingest the new papers into the merged root graph.")


if __name__ == "__main__":
    process_math_paper_queue()
