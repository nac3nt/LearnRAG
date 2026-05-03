import torch
from sentence_transformers import SentenceTransformer
from src.embeddings.base import BaseEmbedder, Vector, Vectors
import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SentenceEmbedder(BaseEmbedder):
    """
    Local embedder using the sentence-transformers library.

    No network calls. Model runs entirely on local CPU/GPU.
    Used as the default embedder in Phase 2.

    Swap to OllamaEmbedder by setting EMBED_MODE=ollama in .env
    """

    # known static dimensions per model
    _DIMENSIONS: dict[str, int] = {
        "all-MiniLM-L6-v2"        : 384,
        "all-mpnet-base-v2"        : 768,
        "paraphrase-MiniLM-L6-v2" : 384,
    }

    def __init__(self) -> None:
        model_name  = config.SENTENCE_MODEL
        self._device = self._resolve_device(config.EMBED_DEVICE)

        if config.DEBUG:
            logger.debug(f"Loading sentence-transformers model : {model_name}")
            logger.debug(f"Device                              : {self._device}")

        try:
            self._model = SentenceTransformer(model_name, device=self._device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SentenceTransformer model '{model_name}': {e}"
            )

        self._model_name     = model_name
        self._probed_dim: int | None = None   # cache for unknown model dimensions

        if config.DEBUG:
            logger.debug(f"Model loaded    : {self.name()}")
            logger.debug(f"Dimension       : {self.dimension()}")
            logger.debug(f"Batch size      : {config.EMBED_BATCH_SIZE}")


    def name(self) -> str:
        """
        Return human-readable embedder identifier.

        Example: "sentence-transformers/all-MiniLM-L6-v2"
        """
        return f"sentence-transformers/{self._model_name}"

    def embed(self, text: str) -> Vector:
        """
        Embed a single string.

        Used at query time for the user's question.

        Args:
            text: Input string to embed.

        Returns:
            Normalized Vector of length self.dimension().
        """
        self._validate_text(text, index=None)

        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.tolist() # type: ignore

    def embed_batch(self, texts: list[str]) -> Vectors:
        """
        Embed a list of strings efficiently using native batching.

        Used at ingestion time to embed all chunks.
        Preserves input order in output.

        Args:
            texts: List of strings to embed.

        Returns:
            Vectors — one normalized vector per input string, same order.
        """
        if not texts:
            raise ValueError("Cannot embed an empty list.")

        for i, text in enumerate(texts):
            self._validate_text(text, index=i)

        if config.DEBUG:
            logger.debug(f"Embedding batch of {len(texts)} chunk(s) "
                         f"[batch_size={config.EMBED_BATCH_SIZE}]...")

        embeddings = self._model.encode(
            texts,
            batch_size=config.EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=config.DEBUG
        )
        return embeddings.tolist() # type: ignore

    def dimension(self) -> int:
        """
        Return static embedding dimension for the active model.

        Checks known dimensions table first. If model is unknown,
        probes with a test string once and caches the result for
        all subsequent calls.

        Returns:
            Integer vector length.
        """
        if self._model_name in self._DIMENSIONS:
            return self._DIMENSIONS[self._model_name]

        if self._probed_dim is not None:
            return self._probed_dim

        if config.DEBUG:
            logger.debug(f"'{self._model_name}' not in known dimensions "
                         f"table — probing model...")

        probe = self._model.encode("probe", convert_to_numpy=True)
        self._probed_dim = len(probe)

        if config.DEBUG:
            logger.debug(f"Probed dimension: {self._probed_dim} (cached)")

        return self._probed_dim
    

    @staticmethod
    def _resolve_device(setting: str) -> str:
        """
        Resolve the compute device to use for inference.

        "auto"  → "cuda" if available, else "cpu"
        "cuda"  → forced CUDA (raises at model load if unavailable)
        "cpu"   → forced CPU

        Args:
            setting: Value from config.EMBED_DEVICE

        Returns:
            Resolved device string: "cuda" or "cpu"
        """
        if setting == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return setting

    @staticmethod
    def _validate_text(text: str, index: int | None) -> None:
        """
        Validate a single text input before embedding.

        Args:
            text:  The string to validate.
            index: Position in batch (None for single embed calls).

        Raises:
            ValueError: if text is empty or whitespace-only.
        """
        if not text or not text.strip():
            location = f"index {index}" if index is not None else "input"
            raise ValueError(f"Empty or whitespace-only text at {location}.")