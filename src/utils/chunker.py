import re
import config


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Take a list of page dicts from pdf_loader and return
    a flat list of chunk dicts with full metadata.

    Input:
    [
        {
            "text": "page text...",
            "page_number": 1,
            "filename": "document.pdf"
        },
        ...
    ]

    Output:
    [
        {
            "text": "chunk text...",
            "metadata": {
                "filename": "document.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "char_count": 342
            },
            "id": "document.pdf_p1_c0"
        },
        ...
    ]
    """
    all_chunks = []

    for page in pages:
        chunks = _chunk_text(
            text=page["text"],
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )

        for chunk_index, chunk_text in enumerate(chunks):
            chunk_id = _build_id(
                filename=page["filename"],
                page_number=page["page_number"],
                chunk_index=chunk_index
            )

            all_chunks.append({
                "text": chunk_text,
                "metadata": {
                    "filename": page["filename"],
                    "page_number": page["page_number"],
                    "chunk_index": chunk_index,
                    "char_count": len(chunk_text)
                },
                "id": chunk_id
            })

        if config.DEBUG:
            print(f"[DEBUG] {page['filename']} p{page['page_number']} → {len(chunks)} chunk(s)")

    if all_chunks and config.DEBUG:
        avg_len = sum(c["metadata"]["char_count"] for c in all_chunks) / len(all_chunks)
        print(f"[DEBUG] Total chunks created : {len(all_chunks)}")
        print(f"[DEBUG] Average chunk length : {avg_len:.0f} characters")

    return all_chunks


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split a single string into overlapping chunks by character count.

    Uses a sliding window:
    - window size = chunk_size
    - step size   = chunk_size - chunk_overlap

    Attempts to avoid mid-word splits by backing up to the nearest
    whitespace. Falls back to hard split if no whitespace is found.

    Raises:
        ValueError: if chunk_size or chunk_overlap values are invalid.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})")

    chunks = []
    step = chunk_size - chunk_overlap
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # try to back up to nearest whitespace to avoid mid-word split
            boundary = text.rfind(" ", start, end)
            if boundary != -1:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


def _build_id(filename: str, page_number: int, chunk_index: int) -> str:
    """
    Build a deterministic, filesystem-safe chunk ID.

    Format: "document.pdf_p1_c0"

    Sanitizes filename to remove characters that could
    cause issues in ChromaDB or filesystem paths.
    """
    safe_filename = re.sub(r"[^\w.\-]", "_", filename)
    return f"{safe_filename}_p{page_number}_c{chunk_index}"