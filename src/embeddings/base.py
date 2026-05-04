from abc import ABC, abstractmethod

Vector = list[float]
Vectors = list[list[float]]


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedding implementations.

    Defines the minimum contract every embedder must fulfill.
    Concrete implementations handle their own model loading,
    network calls, and error handling.

    Current implementations:
        - NIMEmbedder (nim_embedder.py) -> NVIDIA NIM embeddings API
    """

    @abstractmethod
    def name(self) -> str:
        """
        Return a human-readable identifier for this embedder.

        Used in logs and debug output to identify which
        provider and model is active.

        Example return value:
            "nim/nvidia/nv-embedqa-e5-v5"
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

        Must return a static, known value, or probe once and cache
        the result before returning a stable value to the caller.

        Returns:
            An integer representing vector length.
            Example: 1024 for nv-embedqa-e5-v5
        """
        pass
