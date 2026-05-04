import time
from src.loaders.pdf_loader import load_all_pdfs
from src.utils.chunker import chunk_pages
from src.utils.logger import get_logger
from src.vectordb.chroma_store import ChromaStore
import config

logger = get_logger(__name__)


def run_ingestion(reset: bool = False) -> dict:
    """
    Full ingestion pipeline that prepares the knowledge base once.

    Steps:
        1. Load all PDFs from data/docs/
        2. Chunk pages into overlapping segments
        3. Embed chunks using NVIDIA NIM
        4. Upsert vectors and metadata into ChromaDB

    Re-running is safe because duplicate IDs are overwritten.

    Args:
        reset: If True, wipes the ChromaDB collection before ingestion.
               Use during development when re-ingesting from scratch.

    Returns:
        Summary dict:
        {
            "pages_loaded"   : int,
            "chunks_created" : int,
            "vectors_stored" : int,
            "duration_sec"   : float,
            "timings_sec"    : {
                "load"  : float,
                "chunk" : float,
                "embed" : float,
                "store" : float,
                "total" : float
            }
        }
    """
    pipeline_start = time.perf_counter()
    timings = {
        "load": 0.0,
        "chunk": 0.0,
        "embed": 0.0,
        "store": 0.0,
    }

    logger.info("=" * 52)
    logger.info("Starting ingestion pipeline")
    logger.info(f"Docs path       : {config.DOCS_PATH}")
    logger.info(f"Chroma path     : {config.CHROMA_PATH}")
    logger.info(f"Chunk size      : {config.CHUNK_SIZE} characters")
    logger.info(f"Chunk overlap   : {config.CHUNK_OVERLAP} characters")
    logger.info("Embed provider  : NVIDIA NIM")
    logger.info(f"Embed model     : {config.NIM_EMBED_MODEL}")
    logger.info("=" * 52)

    if reset:
        logger.warning("Reset mode enabled - wiping existing collection.")
        store = ChromaStore()
        store.reset()

    logger.info("Step 1/4 - Loading PDFs...")
    step_start = time.perf_counter()
    pages = load_all_pdfs(config.DOCS_PATH)
    timings["load"] = _elapsed_seconds(step_start)
    logger.info(f"Pages loaded : {len(pages)} ({timings['load']:.2f}s)")

    if not pages:
        logger.error("No pages loaded. Add PDF files to data/docs/ and retry.")
        return _summary(0, 0, 0, pipeline_start, timings)

    logger.info("Step 2/4 - Chunking pages...")
    step_start = time.perf_counter()
    chunks = chunk_pages(pages)
    timings["chunk"] = _elapsed_seconds(step_start)
    logger.info(f"Chunks created : {len(chunks)} ({timings['chunk']:.2f}s)")

    if not chunks:
        logger.error("No chunks produced. Check chunk size settings in .env.")
        return _summary(len(pages), 0, 0, pipeline_start, timings)

    logger.info("Step 3/4 - Embedding chunks...")
    step_start = time.perf_counter()
    embedder = _load_embedder()
    logger.info(f"Embedder : {embedder.name()} | dim: {embedder.dimension()}")

    texts = [chunk["text"] for chunk in chunks]
    vectors = embedder.embed_batch(texts)
    timings["embed"] = _elapsed_seconds(step_start)
    logger.info(f"Embeddings produced : {len(vectors)} ({timings['embed']:.2f}s)")

    logger.info("Step 4/4 - Storing in ChromaDB...")
    step_start = time.perf_counter()
    store = ChromaStore()
    before = store.count()
    store.upsert(chunks, vectors)
    after = store.count()
    timings["store"] = _elapsed_seconds(step_start)

    logger.info(f"Chunks before : {before}")
    logger.info(f"Chunks after  : {after} ({timings['store']:.2f}s)")

    summary = _summary(len(pages), len(chunks), after, pipeline_start, timings)

    logger.info("=" * 52)
    logger.info("Ingestion complete.")
    logger.info(f"Pages loaded    : {summary['pages_loaded']}")
    logger.info(f"Chunks created  : {summary['chunks_created']}")
    logger.info(f"Vectors stored  : {summary['vectors_stored']}")
    logger.info(
        "Timing metrics  : "
        f"load={summary['timings_sec']['load']:.2f}s | "
        f"chunk={summary['timings_sec']['chunk']:.2f}s | "
        f"embed={summary['timings_sec']['embed']:.2f}s | "
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
    chunks: int,
    stored: int,
    start: float,
    timings: dict[str, float]
) -> dict:
    """Build and return the ingestion summary dict."""
    total = _elapsed_seconds(start)

    return {
        "pages_loaded": pages,
        "chunks_created": chunks,
        "vectors_stored": stored,
        "duration_sec": total,
        "timings_sec": {
            "load": timings["load"],
            "chunk": timings["chunk"],
            "embed": timings["embed"],
            "store": timings["store"],
            "total": total,
        }
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the RAG ingestion pipeline.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe ChromaDB collection before ingesting."
    )
    args = parser.parse_args()

    try:
        run_ingestion(reset=args.reset)
    except Exception as exc:
        logger.critical(f"Ingestion failed: {exc}", exc_info=True)
        raise SystemExit(1)
