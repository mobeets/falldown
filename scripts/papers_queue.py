import os
import shutil
from marker.convert import convert_single_pdf
from marker.models import load_all_models

def process_paper_queue(
    queue_dir="./papers_queue", 
    archive_dir="./papers_archive", 
    md_dir="./papers_md"
):
    os.makedirs(queue_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)
    os.makedirs(md_dir, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(queue_dir) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("No new papers in queue.")
        return

    print(f"Processing {len(pdf_files)} paper(s)...")
    models = load_all_models()
    
    for filename in pdf_files:
        pdf_path = os.path.join(queue_dir, filename)
        
        # 1. Convert PDF to Markdown with LaTeX math syntax
        full_text, _, _ = convert_single_pdf(pdf_path, models)
        
        # 2. Save Markdown output
        out_filename = os.path.splitext(filename)[0] + ".md"
        with open(os.path.join(md_dir, out_filename), "w", encoding="utf-8") as f:
            f.write(full_text)
            
        # 3. Move processed PDF to archive
        shutil.move(pdf_path, os.path.join(archive_dir, filename))
        print(f"Processed and archived: {filename}")

if __name__ == "__main__":
    process_paper_queue()