import time

import config
from src.embeddings.image_factory import load_image_embedder
from src.loaders.pdf_loader import load_all_pdfs
from src.utils.chunker import chunk_pages
from src.utils.image_records import build_image_vector_items
from src.utils.logger import get_logger
from src.vectordb.chroma_store import ChromaStore

logger = get_logger(__name__)


def run_ingestion(reset: bool = False) -> dict:
    """
    Full ingestion pipeline that prepares the knowledge base once.

    Steps:
        1. Load all PDFs from data/docs/
        2. Convert text-like content into chunks
        3. Embed text chunks
        4. Embed extracted images
        5. Upsert text and image vectors into ChromaDB

    Re-running is safe because duplicate IDs are overwritten.

    Args:
        reset: If True, wipes the ChromaDB collections before ingestion.
               Use during development when re-ingesting from scratch.

    Returns:
        Summary dict:
        {
            "pages_loaded"          : int,
            "content_blocks_loaded" : int,
            "image_blocks_loaded"   : int,
            "chunks_created"        : int,
            "vectors_stored"        : int,
            "image_vectors_stored"  : int,
            "duration_sec"          : float,
            "timings_sec"           : {
                "load"        : float,
                "chunk"       : float,
                "text_embed"  : float,
                "image_embed" : float,
                "store"       : float,
                "total"       : float
            }
        }
    """
    pipeline_start = time.perf_counter()
    timings = {
        "load": 0.0,
        "chunk": 0.0,
        "text_embed": 0.0,
        "image_embed": 0.0,
        "store": 0.0,
    }

    logger.info("=" * 52)
    logger.info("Starting ingestion pipeline")
    logger.info(f"Docs path       : {config.DOCS_PATH}")
    logger.info(f"Chroma path     : {config.CHROMA_PATH}")
    logger.info(f"Text collection : {config.CHROMA_COLLECTION}")
    logger.info(f"Image collection: {config.CHROMA_IMAGE_COLLECTION}")
    logger.info(f"Chunk strategy  : {config.CHUNK_STRATEGY}")
    if config.CHUNK_STRATEGY == "semantic":
        logger.info(
            "Chunk budget    : "
            f"min={config.CHUNK_MIN_TOKENS} | "
            f"max={config.CHUNK_MAX_TOKENS} approx. tokens"
        )
        logger.info(
            "Semantic split  : "
            f"similarity<{config.CHUNK_SIMILARITY_THRESHOLD}"
        )
        logger.info(
            "Fallback split  : "
            f"{config.CHUNK_SIZE} chars | overlap {config.CHUNK_OVERLAP} chars"
        )
    else:
        logger.info(f"Chunk size      : {config.CHUNK_SIZE} characters")
        logger.info(f"Chunk overlap   : {config.CHUNK_OVERLAP} characters")
    logger.info("Text embedder   : NVIDIA NIM")
    logger.info(f"Text model      : {config.NIM_EMBED_MODEL}")
    logger.info(f"Image embedder  : {config.IMAGE_EMBED_PROVIDER}")
    logger.info("=" * 52)

    text_store = ChromaStore()
    image_store = ChromaStore(collection_name=config.CHROMA_IMAGE_COLLECTION)

    if reset:
        logger.warning("Reset mode enabled - wiping existing collections.")
        text_store.reset()
        image_store.reset()

    logger.info("Step 1/5 - Loading PDFs...")
    step_start = time.perf_counter()
    content_blocks = load_all_pdfs(config.DOCS_PATH)
    timings["load"] = _elapsed_seconds(step_start)
    page_count = _count_unique_pages(content_blocks)
    image_block_count = sum(
        1 for block in content_blocks if block.get("content_type") == "image"
    )
    logger.info(
        f"Pages loaded : {page_count} | "
        f"content blocks: {len(content_blocks)} | "
        f"image blocks: {image_block_count} "
        f"({timings['load']:.2f}s)"
    )

    if not content_blocks:
        logger.error("No pages loaded. Add PDF files to data/docs/ and retry.")
        return _summary(0, 0, 0, 0, 0, 0, pipeline_start, timings)

    logger.info("Step 2/5 - Chunking text content...")
    step_start = time.perf_counter()
    chunks = chunk_pages(content_blocks)
    timings["chunk"] = _elapsed_seconds(step_start)
    logger.info(f"Text chunks created : {len(chunks)} ({timings['chunk']:.2f}s)")

    logger.info("Step 3/5 - Embedding text chunks...")
    text_vectors = []
    if chunks:
        step_start = time.perf_counter()
        text_embedder = _load_embedder()
        logger.info(
            f"Text embedder : {text_embedder.name()} | dim: {text_embedder.dimension()}"
        )

        texts = [chunk["text"] for chunk in chunks]
        text_vectors = text_embedder.embed_batch(texts)
        timings["text_embed"] = _elapsed_seconds(step_start)
        logger.info(
            "Text embeddings produced : "
            f"{len(text_vectors)} ({timings['text_embed']:.2f}s)"
        )
    else:
        logger.warning("No text chunks produced. Text retrieval will be empty.")

    logger.info("Step 4/5 - Embedding extracted images...")
    step_start = time.perf_counter()
    image_items = build_image_vector_items(content_blocks)
    image_vectors = []
    image_embedder = load_image_embedder()

    if image_items and image_embedder is not None:
        logger.info(
            f"Image embedder: {image_embedder.name()} | dim: {image_embedder.dimension()}"
        )
        image_vectors = image_embedder.embed_image_batch(
            [item["input"] for item in image_items]
        )
        logger.info(f"Image embeddings produced : {len(image_vectors)}")
    elif image_items:
        logger.warning(
            "Image blocks were extracted but no image embedder is configured. "
            "Set IMAGE_EMBED_PROVIDER=custom and IMAGE_EMBEDDER_CLASS=... to "
            "ingest first-class image vectors."
        )
    else:
        logger.info("No image blocks available for image-vector ingestion.")

    timings["image_embed"] = _elapsed_seconds(step_start)
    logger.info(f"Image embedding step : {timings['image_embed']:.2f}s")

    logger.info("Step 5/5 - Storing vectors in ChromaDB...")
    step_start = time.perf_counter()

    text_before = text_store.count()
    if chunks and text_vectors:
        text_store.upsert(chunks, text_vectors)
    text_after = text_store.count()

    image_before = image_store.count()
    image_records = [item["record"] for item in image_items]
    if image_records and image_vectors:
        image_store.upsert(image_records, image_vectors)
    image_after = image_store.count()

    timings["store"] = _elapsed_seconds(step_start)

    logger.info(f"Text vectors before : {text_before}")
    logger.info(f"Text vectors after  : {text_after}")
    logger.info(f"Image vectors before: {image_before}")
    logger.info(f"Image vectors after : {image_after} ({timings['store']:.2f}s)")

    summary = _summary(
        pages=page_count,
        content_blocks=len(content_blocks),
        image_blocks=image_block_count,
        chunks=len(chunks),
        stored=text_after,
        image_stored=image_after,
        start=pipeline_start,
        timings=timings,
    )

    logger.info("=" * 52)
    logger.info("Ingestion complete.")
    logger.info(f"Pages loaded         : {summary['pages_loaded']}")
    logger.info(f"Content blocks loaded: {summary['content_blocks_loaded']}")
    logger.info(f"Image blocks loaded  : {summary['image_blocks_loaded']}")
    logger.info(f"Text chunks created  : {summary['chunks_created']}")
    logger.info(f"Text vectors stored  : {summary['vectors_stored']}")
    logger.info(f"Image vectors stored : {summary['image_vectors_stored']}")
    logger.info(
        "Timing metrics  : "
        f"load={summary['timings_sec']['load']:.2f}s | "
        f"chunk={summary['timings_sec']['chunk']:.2f}s | "
        f"text_embed={summary['timings_sec']['text_embed']:.2f}s | "
        f"image_embed={summary['timings_sec']['image_embed']:.2f}s | "
        f"store={summary['timings_sec']['store']:.2f}s | "
        f"total={summary['timings_sec']['total']:.2f}s"
    )
    logger.info(f"Total time      : {summary['duration_sec']}s")
    logger.info("=" * 52)

    return summary


def _load_embedder():
    """Load the NVIDIA NIM embedder used by the ingestion pipeline."""
    from src.embeddings.nim_embedder import NIMEmbedder

    return NIMEmbedder()


def _elapsed_seconds(start: float) -> float:
    """Return elapsed time in seconds, rounded for logs and summaries."""
    return round(time.perf_counter() - start, 2)


def _summary(
    pages: int,
    content_blocks: int,
    image_blocks: int,
    chunks: int,
    stored: int,
    image_stored: int,
    start: float,
    timings: dict[str, float],
) -> dict:
    """Build and return the ingestion summary dict."""
    total = _elapsed_seconds(start)

    return {
        "pages_loaded": pages,
        "content_blocks_loaded": content_blocks,
        "image_blocks_loaded": image_blocks,
        "chunks_created": chunks,
        "vectors_stored": stored,
        "image_vectors_stored": image_stored,
        "duration_sec": total,
        "timings_sec": {
            "load": timings["load"],
            "chunk": timings["chunk"],
            "text_embed": timings["text_embed"],
            "image_embed": timings["image_embed"],
            "store": timings["store"],
            "total": total,
        },
    }


def _count_unique_pages(content_blocks: list[dict]) -> int:
    """Count distinct source pages represented in the loaded content blocks."""
    return len(
        {
            (block["filename"], block["page_number"])
            for block in content_blocks
        }
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the RAG ingestion pipeline.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe ChromaDB collections before ingesting.",
    )
    args = parser.parse_args()

    try:
        run_ingestion(reset=args.reset)
    except Exception as exc:
        logger.critical(f"Ingestion failed: {exc}", exc_info=True)
        raise SystemExit(1)
