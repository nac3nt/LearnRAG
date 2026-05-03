import time
from src.utils.logger import get_logger
from src.loaders.pdf_loader import load_all_pdfs
from src.utils.chunker import chunk_pages
from src.vectordb.chroma_store import ChromaStore
import config

logger = get_logger(__name__)


def run_ingestion(reset: bool = False) -> dict:
    """
    Full ingestion pipeline — runs once to prepare the knowledge base.

    Steps:
        1. Load all PDFs from data/docs/
        2. Chunk pages into overlapping segments
        3. Embed chunks using configured embedder
        4. Upsert vectors and metadata into ChromaDB

    Re-running is safe — duplicate IDs are overwritten, not duplicated.

    Args:
        reset: If True, wipes the ChromaDB collection before ingestion.
               Use during development when re-ingesting from scratch.

    Returns:
        Summary dict:
        {
            "pages_loaded"   : int,
            "chunks_created" : int,
            "vectors_stored" : int,
            "duration_sec"   : float
        }
    """
    pipeline_start = time.perf_counter()

    logger.info("=" * 52)
    logger.info("Starting ingestion pipeline")
    logger.info(f"Docs path       : {config.DOCS_PATH}")
    logger.info(f"Chroma path     : {config.CHROMA_PATH}")
    logger.info(f"Chunk size      : {config.CHUNK_SIZE} characters")
    logger.info(f"Chunk overlap   : {config.CHUNK_OVERLAP} characters")
    logger.info(f"Embed mode      : {config.EMBED_MODE}")
    logger.info("=" * 52)

    # Optional reset
    if reset:
        logger.warning("Reset mode enabled — wiping existing collection.")
        store = ChromaStore()
        store.reset()

    # Step 1: Load
    logger.info("Step 1/4 - Loading PDFs...")
    step_start = time.perf_counter()

    pages = load_all_pdfs(config.DOCS_PATH)

    logger.info(f"Pages loaded : {len(pages)} ({_elapsed(step_start)})")

    if not pages:
        logger.error("No pages loaded. Add PDF files to data/docs/ and retry.")
        return _summary(0, 0, 0, pipeline_start)

    # Step 2: Chunk
    logger.info("Step 2/4 - Chunking pages...")
    step_start = time.perf_counter()

    chunks = chunk_pages(pages)

    logger.info(f"Chunks created : {len(chunks)} ({_elapsed(step_start)})")

    if not chunks:
        logger.error("No chunks produced. Check chunk size settings in .env.")
        return _summary(len(pages), 0, 0, pipeline_start)

    # Step 3: Embed
    logger.info("Step 3/4 - Embedding chunks...")
    step_start = time.perf_counter()

    embedder = _load_embedder()
    logger.info(f"Embedder : {embedder.name()} | dim: {embedder.dimension()}")

    texts   = [c["text"] for c in chunks]
    vectors = embedder.embed_batch(texts)

    logger.info(f"Embeddings produced : {len(vectors)} ({_elapsed(step_start)})")

    # Step 4: Store 
    logger.info("Step 4/4 - Storing in ChromaDB...")
    step_start = time.perf_counter()

    store = ChromaStore()
    before = store.count()

    store.upsert(chunks, vectors)

    after = store.count()
    logger.info(f"Chunks before : {before}")
    logger.info(f"Chunks after  : {after} ({_elapsed(step_start)})")

    summary = _summary(len(pages), len(chunks), after, pipeline_start)

    logger.info("=" * 52)
    logger.info("Ingestion complete.")
    logger.info(f"Pages loaded    : {summary['pages_loaded']}")
    logger.info(f"Chunks created  : {summary['chunks_created']}")
    logger.info(f"Vectors stored  : {summary['vectors_stored']}")
    logger.info(f"Total time      : {summary['duration_sec']}s")
    logger.info("=" * 52)

    return summary


def _load_embedder():
    """
    Load the configured embedder based on config.EMBED_MODE.

    Returns:
        SentenceEmbedder if EMBED_MODE = "sentence_transformers"
        OllamaEmbedder   if EMBED_MODE = "ollama"

    Raises:
        ValueError: if EMBED_MODE is not a recognized value.
    """
    if config.EMBED_MODE == "sentence_transformers":
        from src.embeddings.sentence_embedder import SentenceEmbedder
        return SentenceEmbedder()

    if config.EMBED_MODE == "ollama":
        from src.embeddings.ollama_embedder import OllamaEmbedder # type: ignore
        return OllamaEmbedder()

    raise ValueError(
        f"Unknown EMBED_MODE: '{config.EMBED_MODE}'. "
        f"Expected 'sentence_transformers' or 'ollama'."
    )


def _elapsed(start: float) -> str:
    """Return human-readable elapsed time since start."""
    return f"{time.perf_counter() - start:.2f}s"


def _summary(
    pages: int,
    chunks: int,
    stored: int,
    start: float
) -> dict:
    """Build and return the ingestion summary dict."""
    return {
        "pages_loaded"   : pages,
        "chunks_created" : chunks,
        "vectors_stored" : stored,
        "duration_sec"   : round(time.perf_counter() - start, 2)
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
    except Exception as e:
        logger.critical(f"Ingestion failed: {e}", exc_info=True)
        raise SystemExit(1)