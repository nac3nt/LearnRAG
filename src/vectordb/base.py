from abc import ABC, abstractmethod
from src.embeddings.base import Vector


class BaseVectorDB(ABC):
    """
    Abstract base class for all vector database implementations.

    Defines the minimum contract every vector store must fulfill.
    Concrete implementations handle their own client setup,
    collection management, and error handling.

    Current implementations:
        - ChromaStore  (chroma_store.py)  → local persistent ChromaDB

    Score semantics:
        All query() implementations must return cosine similarity scores
        in the range [0.0, 1.0] where:
            1.0  = identical direction (perfect match)
            0.0  = orthogonal (no semantic relation)
           <0.0  = should not occur with normalized vectors

        Higher is always better. Implementations that use distance-based
        metrics internally (e.g. L2) must convert to similarity scores
        before returning so callers never have to care about the
        underlying metric.
    """

    @abstractmethod
    def name(self) -> str:
        """
        Return a human-readable identifier for this vector store.

        Used in logs and debug output.

        Example return values:
            "chromadb/rag_docs"
        """
        pass

    @abstractmethod
    def upsert(self, chunks: list[dict], vectors: list[Vector]) -> None:
        """
        Insert or update chunks and their vectors in the store.

        Uses upsert semantics — if a chunk ID already exists,
        it is overwritten. If it does not exist, it is inserted.
        This makes ingestion duplicate-safe and re-runnable.

        Args:
            chunks:  List of chunk dicts from chunker.py.
                     Each dict must contain:
                         - "id"       (str)  deterministic chunk ID
                         - "text"     (str)  raw chunk text
                         - "metadata" (dict) filename, page_number,
                                             chunk_index, char_count
            vectors: List of embedding vectors, one per chunk,
                     in the same order as chunks.

        Raises:
            ValueError: if chunks and vectors lengths do not match.
        """
        pass

    @abstractmethod
    def query(
        self,
        vector: Vector,
        top_k: int,
        filters: dict | None = None
    ) -> list[dict]:
        """
        Find the most similar chunks to a query vector.

        Args:
            vector:  Embedded query vector (normalized).
            top_k:   Number of results to return.
            filters: Optional metadata filters for narrowing results.
                     Not implemented yet — reserved for future use.
                     Example future usage:
                         {"filename": "report.pdf"}
                         {"page_number": 3}

        Returns:
            List of result dicts, ordered by cosine similarity descending.
            Each dict contains:
                - "id"       (str)   chunk ID
                - "text"     (str)   chunk text
                - "metadata" (dict)  filename, page_number,
                                     chunk_index, char_count
                - "score"    (float) cosine similarity in [0.0, 1.0]
                                     higher = more relevant
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Return the total number of chunks stored in the collection.

        Used for logging and sanity checks after ingestion.

        Returns:
            Integer count of stored chunks.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Delete and recreate the collection, removing all stored data.

        Destructive operation — use only during development
        or when re-ingesting from scratch.

        Implementations must log a clear warning before executing.
        """
        pass