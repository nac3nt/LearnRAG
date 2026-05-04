import re

from src.embeddings.image_base import ImageEmbeddingInput


def build_image_vector_items(content_blocks: list[dict]) -> list[dict]:
    """
    Convert image content blocks into vector-ready items.

    Output items carry:
    - `input`    -> ImageEmbeddingInput for the image embedder
    - `record`   -> Chroma-compatible document dict for the vector store
    """
    items = []

    for block in content_blocks:
        if block.get("content_type") != "image":
            continue

        image_bytes = block.get("image_bytes", b"")
        if not image_bytes:
            continue

        filename = block["filename"]
        page_number = block["page_number"]
        image_index = block.get("content_index", 0)
        text = (block.get("text") or "").strip()

        input_item = ImageEmbeddingInput(
            image_bytes=image_bytes,
            mime_type=block.get("mime_type", "application/octet-stream"),
            filename=filename,
            page_number=page_number,
            image_index=image_index,
            asset_path=block.get("asset_path", ""),
            text=text,
        )

        metadata = {
            key: value
            for key, value in block.items()
            if key not in {"text", "image_bytes"}
        }
        metadata["has_text_representation"] = bool(text)
        metadata["vector_modality"] = "image"

        record_text = text or (
            f"Image extracted from {filename} page {page_number} "
            f"(image {image_index + 1})."
        )

        items.append(
            {
                "input": input_item,
                "record": {
                    "id": _build_image_id(
                        filename=filename,
                        page_number=page_number,
                        image_index=image_index,
                    ),
                    "text": record_text,
                    "metadata": metadata,
                },
            }
        )

    return items


def _build_image_id(filename: str, page_number: int, image_index: int) -> str:
    """Build a deterministic ID for an image vector record."""
    safe_filename = re.sub(r"[^\w.\-]", "_", filename)
    return f"{safe_filename}_p{page_number}_image_{image_index}"
