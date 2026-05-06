try:
    import filetype as _filetype
except ImportError:
    _filetype = None
import re
from pathlib import Path

from pypdf import PdfReader

import config
from src.captioners.base import BaseImageCaptioner
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import pdfplumber
except ImportError:  # pragma: no cover - optional dependency
    pdfplumber = None


def load_pdf(
    filepath: str | Path,
    captioner: BaseImageCaptioner | None = None,
) -> list[dict]:
    """
    Load a single PDF and extract retrievable content block by block.

    Returns a list of dicts:
    [
        {
            "text": "page text, table text, or image description...",
            "page_number": 1,
            "filename": "document.pdf",
            "content_type": "page_text" | "table" | "image",
            "content_index": 0,
            "source": "pypdf/plain" | "pdfplumber" | "openai-compatible/...",
        },
        ...
    ]
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"PDF not found: {filepath}")

    filename = filepath.name
    content_blocks = []

    try:
        reader = PdfReader(filepath)
    except Exception as exc:
        raise RuntimeError(f"Failed to read PDF '{filename}': {exc}") from exc

    plumber_pdf = _open_pdfplumber(filepath)

    try:
        for page_index, page in enumerate(reader.pages):
            page_number = page_index + 1
            page_content = []

            page_text = _extract_page_text(page, filename, page_number)
            if page_text:
                page_content.append(
                    {
                        "text": page_text,
                        "page_number": page_number,
                        "filename": filename,
                        "content_type": "page_text",
                        "content_index": 0,
                        "source": f"pypdf/{config.PDF_TEXT_EXTRACTION_MODE}",
                    }
                )

            if plumber_pdf is not None:
                page_content.extend(
                    _extract_page_tables(
                        plumber_pdf.pages[page_index],
                        filename,
                        page_number,
                    )
                )

            page_content.extend(
                _extract_page_images(
                    page=page,
                    pdf_name=filepath.stem,
                    filename=filename,
                    page_number=page_number,
                    captioner=captioner,
                )
            )

            if not page_content and config.DEBUG:
                logger.debug(f"Skipping empty page {page_number} in {filename}")

            content_blocks.extend(page_content)
    finally:
        if plumber_pdf is not None:
            plumber_pdf.close()

    if config.DEBUG:
        logger.debug(f"Loaded {len(content_blocks)} content block(s) from {filename}")

    return content_blocks


def load_all_pdfs(docs_path: str | Path = config.DOCS_PATH) -> list[dict]:
    """
    Load all PDFs from the docs folder.

    Returns a flat list of content blocks across all documents,
    sorted deterministically by filename.
    """
    docs_path = Path(docs_path)

    if not docs_path.exists():
        raise FileNotFoundError(f"Docs folder not found: {docs_path}")

    pdf_files = sorted(
        [
            f
            for f in docs_path.iterdir()
            if f.is_file() and f.suffix.lower() == ".pdf"
        ]
    )

    if not pdf_files:
        raise ValueError(f"No PDF files found in: {docs_path}")

    if config.DEBUG:
        logger.debug(f"Found {len(pdf_files)} PDF(s) in {docs_path}")

    all_content_blocks = []
    captioner = _load_image_captioner()

    for pdf_file in pdf_files:
        try:
            content_blocks = load_pdf(pdf_file, captioner=captioner)
            all_content_blocks.extend(content_blocks)
        except RuntimeError as exc:
            logger.warning(f"Skipping unreadable PDF: {exc}")

    if all_content_blocks:
        by_type = _content_type_counts(all_content_blocks)
        logger.info(
            "Total content blocks loaded: "
            f"{len(all_content_blocks)} "
            f"(page_text={by_type['page_text']}, "
            f"table={by_type['table']}, image={by_type['image']})"
        )
    else:
        logger.info("Total content blocks loaded: 0")

    return all_content_blocks


def _extract_page_text(page, filename: str, page_number: int) -> str | None:
    """Extract normalized text from a page using the configured mode."""
    try:
        text = page.extract_text(extraction_mode=config.PDF_TEXT_EXTRACTION_MODE)
    except TypeError:
        text = page.extract_text()
    except Exception as exc:
        if config.DEBUG:
            logger.debug(
                "Could not extract text from page "
                f"{page_number} in {filename}: {exc}"
            )
        return None

    if not text or not text.strip():
        return None

    preserve_layout = config.PDF_TEXT_EXTRACTION_MODE == "layout"
    return _normalize_text(text, preserve_layout=preserve_layout)


def _extract_page_tables(plumber_page, filename: str, page_number: int) -> list[dict]:
    """Extract tables from a pdfplumber page and render them as text."""
    if not config.PDF_EXTRACT_TABLES:
        return []

    try:
        raw_tables = plumber_page.extract_tables()
    except Exception as exc:
        logger.warning(
            f"Table extraction failed for {filename} page {page_number}: {exc}"
        )
        return []

    content_blocks = []

    for table_index, raw_table in enumerate(raw_tables):
        table = _clean_table(raw_table)
        if not table:
            continue

        table_text, row_count, column_count = _table_to_text(
            table=table,
            filename=filename,
            page_number=page_number,
        )
        if not table_text:
            continue

        content_blocks.append(
            {
                "text": table_text,
                "page_number": page_number,
                "filename": filename,
                "content_type": "table",
                "content_index": table_index,
                "source": "pdfplumber",
                "row_count": row_count,
                "column_count": column_count,
            }
        )

    return content_blocks


def _extract_page_images(
    page,
    pdf_name: str,
    filename: str,
    page_number: int,
    captioner: BaseImageCaptioner | None,
) -> list[dict]:
    """
    Extract embedded images from a PDF page.

    Image blocks are always returned so they can be embedded as raw image
    vectors later. A captioner is optional and only affects whether an
    additional text representation is available for text retrieval.
    """
    if not config.PDF_EXTRACT_IMAGES:
        return []

    content_blocks = []
    images = list(getattr(page, "images", []))

    for image_index, image_file in enumerate(images):
        image_bytes = getattr(image_file, "data", b"")
        if not image_bytes:
            continue

        extension = _guess_image_extension(
            image_name=getattr(image_file, "name", ""),
            image_bytes=image_bytes,
        )
        mime_type = _extension_to_mime_type(extension)
        asset_path = ""

        if config.PDF_SAVE_EXTRACTED_IMAGES:
            asset_path = _save_image_asset(
                pdf_name=pdf_name,
                page_number=page_number,
                image_index=image_index,
                image_bytes=image_bytes,
                extension=extension,
            )

        description_text = ""
        caption_source = ""

        if captioner is not None:
            try:
                description = captioner.describe(
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    filename=filename,
                    page_number=page_number,
                    image_index=image_index,
                )
                if description.strip():
                    description_text = _image_description_to_text(
                        description=description,
                        filename=filename,
                        page_number=page_number,
                        image_index=image_index,
                    )
                    caption_source = captioner.name()
            except Exception as exc:
                logger.warning(
                    "Image captioning failed for "
                    f"{filename} page {page_number} image {image_index + 1}: {exc}"
                )
        elif config.DEBUG:
            logger.debug(
                "No captioner configured for "
                f"{filename} page {page_number} image {image_index + 1}. "
                "Image will still be available for raw image-vector ingestion."
            )

        content_block = {
            "text": description_text,
            "page_number": page_number,
            "filename": filename,
            "content_type": "image",
            "content_index": image_index,
            "source": "pypdf/image",
            "mime_type": mime_type,
            "image_bytes": image_bytes,
        }
        if asset_path:
            content_block["asset_path"] = asset_path
        if caption_source:
            content_block["caption_source"] = caption_source

        content_blocks.append(content_block)

    return content_blocks


def _open_pdfplumber(filepath: Path):
    """Open a PDF with pdfplumber if the feature and dependency are available."""
    if not config.PDF_EXTRACT_TABLES:
        return None

    if pdfplumber is None:
        logger.warning(
            "PDF table extraction is enabled but pdfplumber is not installed. "
            "Install pdfplumber to index tables."
        )
        return None

    try:
        return pdfplumber.open(filepath)
    except Exception as exc:
        logger.warning(f"Could not open '{filepath.name}' with pdfplumber: {exc}")
        return None


def _load_image_captioner() -> BaseImageCaptioner | None:
    """Instantiate the configured image captioner, or return None."""
    if config.IMAGE_CAPTION_PROVIDER == "none":
        return None

    try:
        if config.IMAGE_CAPTION_PROVIDER == "openai_compatible":
            from src.captioners.openai_compatible import (
                OpenAICompatibleVisionCaptioner,
            )

            return OpenAICompatibleVisionCaptioner()
    except Exception as exc:
        logger.warning(f"Image captioning disabled: {exc}")
        return None

    logger.warning(
        f"Unknown image caption provider '{config.IMAGE_CAPTION_PROVIDER}'. "
        "Skipping image indexing."
    )
    return None


def _table_to_text(
    table: list[list[str]],
    filename: str,
    page_number: int,
) -> tuple[str, int, int]:
    """Render extracted table rows into retrieval-friendly plain text."""
    if not table:
        return "", 0, 0

    column_count = max(len(row) for row in table)
    normalized_rows = [row + [""] * (column_count - len(row)) for row in table]

    if len(normalized_rows) == 1:
        headers = [f"column_{index + 1}" for index in range(column_count)]
        data_rows = normalized_rows
    else:
        raw_headers = normalized_rows[0]
        headers = [
            cell if cell else f"column_{index + 1}"
            for index, cell in enumerate(raw_headers)
        ]
        data_rows = normalized_rows[1:]

    lines = [f"Table extracted from {filename} page {page_number}."]
    lines.append(f"Header row: {' | '.join(headers)}")

    for row_number, row in enumerate(data_rows, start=1):
        row_pairs = [
            f"{headers[index]} = {value}"
            for index, value in enumerate(row)
            if value
        ]
        if row_pairs:
            lines.append(f"Row {row_number}: " + "; ".join(row_pairs))

    return "\n".join(lines).strip(), len(data_rows), column_count


def _image_description_to_text(
    description: str,
    filename: str,
    page_number: int,
    image_index: int,
) -> str:
    """Add source context around an image description before embedding."""
    return (
        f"Image extracted from {filename} page {page_number} "
        f"(image {image_index + 1}).\n"
        f"{description.strip()}"
    )


def _clean_table(raw_table: list[list[str | None]] | None) -> list[list[str]]:
    """Normalize pdfplumber table cells and drop empty rows."""
    if not raw_table:
        return []

    cleaned_rows = []
    for row in raw_table:
        values = [_normalize_table_cell(cell) for cell in (row or [])]
        if any(values):
            cleaned_rows.append(values)

    return cleaned_rows


def _normalize_text(text: str, preserve_layout: bool = False) -> str:
    """
    Clean common PDF extraction artifacts.

    - Collapse repeated spaces for plain-text extraction
    - Collapse 3+ consecutive newlines into two
    - Strip leading/trailing whitespace
    """
    if preserve_layout:
        lines = [re.sub(r"[ \t]+$", "", line) for line in text.splitlines()]
        text = "\n".join(lines)
    else:
        text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_table_cell(cell: str | None) -> str:
    """Normalize an extracted table cell into a compact single line."""
    if cell is None:
        return ""

    text = str(cell).replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _guess_image_extension(image_name: str, image_bytes: bytes) -> str:
    """Infer an image extension from the filename or raw bytes."""
    suffix = Path(image_name).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}:
        return suffix

    if _filetype is not None:
        kind = _filetype.guess(image_bytes)
        if kind is not None:
            ext = kind.extension
            return ".jpg" if ext == "jpeg" else f".{ext}"
    return ".bin"


def _extension_to_mime_type(extension: str) -> str:
    """Map a file extension to a MIME type for vision APIs."""
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".webp": "image/webp",
    }.get(extension.lower(), "application/octet-stream")


def _save_image_asset(
    pdf_name: str,
    page_number: int,
    image_index: int,
    image_bytes: bytes,
    extension: str,
) -> str:
    """Persist an extracted image to disk and return its relative path."""
    image_dir = Path(config.PDF_IMAGE_OUTPUT_PATH) / _safe_filename(pdf_name)
    image_dir.mkdir(parents=True, exist_ok=True)

    output_path = image_dir / (
        f"page_{page_number:04d}_image_{image_index + 1:03d}{extension}"
    )
    output_path.write_bytes(image_bytes)
    return output_path.as_posix()


def _content_type_counts(content_blocks: list[dict]) -> dict[str, int]:
    """Count content blocks by type for logging."""
    counts = {"page_text": 0, "table": 0, "image": 0}
    for block in content_blocks:
        content_type = block.get("content_type", "page_text")
        counts[content_type] = counts.get(content_type, 0) + 1
    return counts


def _safe_filename(value: str) -> str:
    """Convert a filename fragment into a filesystem-safe directory name."""
    return re.sub(r"[^\w.\-]", "_", value)
