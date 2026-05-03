from abc import ABC, abstractmethod

Vector  = list[float]         # single embedding vector
Vectors = list[list[float]]   # batch of embedding vectors


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedding implementations.

    Defines the minimum contract every embedder must fulfill.
    Concrete implementations handle their own model loading,
    network calls, and error handling.

    Current implementations:
        - SentenceEmbedder  (sentence_embedder.py)  → sentence-transformers, local
        - OllamaEmbedder    (ollama_embedder.py)    → Ollama REST API
    """

    @abstractmethod
    def name(self) -> str:
        """
        Return a human-readable identifier for this embedder.

        Used in logs and debug output to identify which
        provider and model is active.

        Example return values:
            "sentence-transformers/all-MiniLM-L6-v2"
            "ollama/nomic-embed-text"
        """
        pass

    @abstractmethod
    def embed(self, text: str) -> Vector:
        """
        Embed a single string and return its vector.

        Used at query time to embed the user's question
        before similarity search.

        Args:
            text: The input string to embed.

        Returns:
            A Vector (list of floats) of length self.dimension().
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> Vectors:
        """
        Embed a list of strings and return a list of vectors.

        Used at ingestion time to embed all chunks efficiently.
        Implementations should use native batching where available
        rather than looping over embed().

        Args:
            texts: List of input strings to embed.

        Returns:
            A Vectors (list of Vectors), one per input string,
            in the same order as the input list.
        """
        pass

    @abstractmethod
    def dimension(self) -> int:
        """
        Return the dimensionality of vectors produced by this embedder.

        Must return a static, known value — not discovered lazily
        after the first call. This value is used by ChromaDB when
        creating a collection and must be consistent across all
        embed() and embed_batch() calls.

        Returns:
            An integer representing vector length.
            Example: 384 for all-MiniLM-L6-v2
                     768 for nomic-embed-text

        Note:
            If dimension is unknown before the first call, raise
            NotImplementedError with a clear message rather than
            returning a wrong value silently.
        """
        pass