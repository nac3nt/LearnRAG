import chromadb
from chromadb.config import Settings
from src.embeddings.base import Vector
from src.utils.logger import get_logger
from src.utils.vectors import normalize_vector, normalize_vectors
from src.vectordb.base import BaseVectorDB
import config

logger = get_logger(__name__)


class ChromaStore(BaseVectorDB):
    """
    Local persistent vector store using ChromaDB.

    Stores chunk vectors and metadata to disk at config.CHROMA_PATH.
    Survives restarts without re-ingestion between sessions.

    Collection is created on first run and reused on subsequent runs.
    Uses cosine similarity for all queries, as required by BaseVectorDB.

    Score semantics:
        query() returns a 'score' field derived from ChromaDB's cosine
        distance: score = 1 - distance.
        Higher score = more similar to the query.
    """

    def __init__(self) -> None:
        if config.DEBUG:
            logger.debug(f"Initializing ChromaDB at: {config.CHROMA_PATH}")

        self._client = chromadb.PersistentClient(
            path=config.CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False)
        )

        self._collection = self._client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": config.CHROMA_DISTANCE_SPACE}
        )

        if config.DEBUG:
            logger.debug(f"Collection    : {config.CHROMA_COLLECTION}")
            logger.debug(f"Chunks stored : {self.count()}")

    def name(self) -> str:
        """Return human-readable store identifier."""
        return f"chromadb/{config.CHROMA_COLLECTION}"

    def upsert(self, chunks: list[dict], vectors: list[Vector]) -> None:
        """
        Insert or update chunks and their vectors in ChromaDB.

        Duplicate-safe: re-ingesting the same document overwrites
        existing entries by ID rather than creating duplicates.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks and vectors must be the same length. "
                f"Got {len(chunks)} chunks and {len(vectors)} vectors."
            )

        if not chunks:
            if config.DEBUG:
                logger.debug("Upsert called with empty list - skipping.")
            return

        _validate_vector_dimensions(vectors)
        normalized_vectors = normalize_vectors(vectors)

        batch_size = config.CHROMA_UPSERT_BATCH_SIZE
        total = len(chunks)
        num_batches = (total + batch_size - 1) // batch_size

        for batch_index in range(num_batches):
            start = batch_index * batch_size
            end = min(start + batch_size, total)

            batch_chunks = chunks[start:end]
            batch_vectors = normalized_vectors[start:end]

            ids = [c["id"] for c in batch_chunks]
            documents = [c["text"] for c in batch_chunks]
            metadatas = [c["metadata"] for c in batch_chunks]

            self._collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=batch_vectors,  # type: ignore
                metadatas=metadatas
            )

            if config.DEBUG:
                logger.debug(
                    f"Upserted batch {batch_index + 1}/{num_batches} "
                    f"({len(batch_chunks)} chunks) -> total stored: {self.count()}"
                )

        logger.info(f"Upsert complete - {total} chunk(s) written to {self.name()}")

    def query(
        self,
        vector: Vector,
        top_k: int,
        filters: dict | None = None
    ) -> list[dict]:
        """
        Find the most similar chunks to a query vector.

        Returns results ordered by similarity descending.
        score = 1 - cosine_distance. Higher = more similar.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        if self.count() == 0:
            if config.DEBUG:
                logger.debug("Query called on empty collection - returning []")
            return []

        normalized_vector = normalize_vector(vector)
        result_count = min(top_k, self.count())

        query_params = {
            "query_embeddings": [normalized_vector],
            "n_results": result_count,
            "include": ["documents", "metadatas", "distances"]
        }

        if filters:
            query_params["where"] = filters

        raw = self._collection.query(**query_params)

        ids = raw["ids"][0]
        documents = raw["documents"][0]  # type: ignore
        metadatas = raw["metadatas"][0]  # type: ignore
        distances = raw["distances"][0]  # type: ignore

        results = []

        for id_, text, metadata, distance in zip(ids, documents, metadatas, distances):
            score = round(1 - distance, 4)
            results.append({
                "id": id_,
                "text": text,
                "metadata": metadata,
                "score": score
            })

        if config.DEBUG:
            for result in results:
                logger.debug(
                    f"Retrieved -> {result['id']} "
                    f"| score: {result['score']} "
                    f"| page: {result['metadata'].get('page_number')} "
                    f"| file: {result['metadata'].get('filename')}"
                )

        return results

    def count(self) -> int:
        """Return total number of chunks in the collection."""
        return self._collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection, wiping all stored data."""
        logger.warning(
            f"Resetting collection '{config.CHROMA_COLLECTION}' - all data will be lost."
        )

        try:
            self._client.delete_collection(name=config.CHROMA_COLLECTION)
        except Exception as exc:
            if config.DEBUG:
                logger.debug(f"Delete collection skipped: {exc}")

        self._collection = self._client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": config.CHROMA_DISTANCE_SPACE}
        )

        logger.info(f"Collection reset complete. Chunks stored: {self.count()}")


def _validate_vector_dimensions(vectors: list[Vector]) -> None:
    """
    Ensure all vectors in a batch have identical dimensions.

    Raises:
        ValueError: if any vector dimension differs from the first.
    """
    if not vectors:
        return

    expected = len(vectors[0])

    if expected == 0:
        raise ValueError("Vectors must not be empty.")

    for i, vector in enumerate(vectors[1:], start=1):
        if len(vector) != expected:
            raise ValueError(
                f"Inconsistent vector dimensions at index {i}. "
                f"Expected {expected}, got {len(vector)}."
            )
