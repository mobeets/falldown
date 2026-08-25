import os
import shutil
from marker.convert import convert_single_pdf
from marker.models import load_all_models

def process_math_paper_queue(
    queue_dir="./papers_queue",
    archive_dir="./papers_archive",
    md_dir="./papers_md"
):
    # Create required directories if they don't exist
    os.makedirs(queue_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)
    os.makedirs(md_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(queue_dir) if f.endswith(".pdf")]

    if not pdf_files:
        print("No new PDFs found in ./papers_queue/")
        return

    print(f"Found {len(pdf_files)} paper(s) to convert...")
    models = load_all_models()

    for filename in pdf_files:
        pdf_path = os.path.join(queue_dir, filename)

        # Convert layout and parse equations into LaTeX
        full_text, _, _ = convert_single_pdf(pdf_path, models)

        # Save Markdown file
        out_filename = os.path.splitext(filename)[0] + ".md"
        md_path = os.path.join(md_dir, out_filename)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        # Move converted PDF to the archive folder
        shutil.move(pdf_path, os.path.join(archive_dir, filename))
        print(f"Successfully processed {filename} -> {out_filename} (PDF archived)")

if __name__ == "__main__":
    process_math_paper_queue()