import re
from pathlib import Path
from pypdf import PdfReader
import config


def load_pdf(filepath: str | Path) -> list[dict]:
    """
    Load a single PDF and extract text page by page.

    Returns a list of dicts:
    [
        {
            "text": "page text...",
            "page_number": 1,
            "filename": "document.pdf"
        },
        ...
    ]
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"PDF not found: {filepath}")

    filename = filepath.name
    pages = []

    try:
        reader = PdfReader(filepath)
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF '{filename}': {e}")

    for page_index, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
        except Exception as e:
            if config.DEBUG:
                print(f"[DEBUG] Could not extract text from page {page_index + 1} in {filename}: {e}")
            continue

        if not text or not text.strip():
            if config.DEBUG:
                print(f"[DEBUG] Skipping empty page {page_index + 1} in {filename}")
            continue

        text = _normalize_text(text)

        pages.append({
            "text": text,
            "page_number": page_index + 1,
            "filename": filename
        })

    if config.DEBUG:
        print(f"[DEBUG] Loaded {len(pages)} pages from {filename}")

    return pages


def load_all_pdfs(docs_path: str | Path = config.DOCS_PATH) -> list[dict]:
    """
    Load all PDFs from the docs folder.

    Returns a flat list of page dicts across all documents,
    sorted deterministically by filename.
    """
    docs_path = Path(docs_path)

    if not docs_path.exists():
        raise FileNotFoundError(f"Docs folder not found: {docs_path}")

    pdf_files = sorted([
        f for f in docs_path.iterdir()
        if f.is_file() and f.suffix.lower() == ".pdf"
    ])

    if not pdf_files:
        raise ValueError(f"No PDF files found in: {docs_path}")

    if config.DEBUG:
        print(f"[DEBUG] Found {len(pdf_files)} PDF(s) in {docs_path}")

    all_pages = []

    for pdf_file in pdf_files:
        try:
            pages = load_pdf(pdf_file)
            all_pages.extend(pages)
        except RuntimeError as e:
            print(f"[WARN] Skipping unreadable PDF: {e}")

    print(f"[INFO] Total pages loaded: {len(all_pages)}")

    return all_pages


def _normalize_text(text: str) -> str:
    """
    Clean common PDF extraction artifacts.

    - Collapse repeated spaces
    - Collapse 3+ consecutive newlines into two
    - Strip leading/trailing whitespace
    """
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()