import time
import requests
from src.embeddings.base import BaseEmbedder, Vector, Vectors
from src.utils.logger import get_logger
from src.utils.vectors import normalize_vectors
import config

logger = get_logger(__name__)

# NIM models and their known embedding dimensions
_NIM_DIMENSIONS: dict[str, int] = {
    "nvidia/nv-embedqa-e5-v5": 1024,
    "nvidia/llama-3.2-nv-embedqa-1b-v2": 2048,
    "nvidia/nv-embed-v1": 4096,
    "baai/bge-m3": 1024,
}


class NIMEmbedder(BaseEmbedder):
    """
    Remote embedder using NVIDIA NIM inference endpoints.

    Calls the NIM-hosted embedding API via the OpenAI-compatible
    /v1/embeddings endpoint. Requires a valid NVIDIA_API_KEY in .env.
    This is the project's default embedder.

    Recommended models:
        nvidia/nv-embedqa-e5-v5           (1024-dim, best quality)
        nvidia/llama-3.2-nv-embedqa-1b-v2 (2048-dim, fast)
        baai/bge-m3                        (1024-dim, multilingual)
    """

    def __init__(self) -> None:
        self._api_key = config.NIM_API_KEY
        self._model = config.NIM_EMBED_MODEL
        self._base_url = config.NIM_BASE_URL.rstrip("/")
        self._endpoint = f"{self._base_url}/v1/embeddings"
        self._batch_size = config.EMBED_BATCH_SIZE
        self._probed_dim: int | None = None

        if not self._api_key:
            raise ValueError(
                "NVIDIA_API_KEY is not set. "
                "Add it to your .env file to use NIM embeddings."
            )

        if config.DEBUG:
            logger.debug("NIM embedder initialised")
            logger.debug(f"Model    : {self._model}")
            logger.debug(f"Endpoint : {self._endpoint}")
            logger.debug(f"Dim      : {self.dimension()}")

    def name(self) -> str:
        return f"nim/{self._model}"

    def embed(self, text: str) -> Vector:
        """Embed a single query string via the NIM API."""
        self._validate_text(text, index=None)
        return self._call_api([text], input_type="query")[0]

    def embed_batch(self, texts: list[str]) -> Vectors:
        """
        Embed a list of passage strings via the NIM API.

        Splits into sub-batches of EMBED_BATCH_SIZE to stay within
        request limits while preserving input order.
        """
        if not texts:
            raise ValueError("Cannot embed an empty list.")

        for i, text in enumerate(texts):
            self._validate_text(text, index=i)

        if config.DEBUG:
            logger.debug(
                f"NIM embed_batch: {len(texts)} chunk(s) "
                f"[batch_size={self._batch_size}]"
            )

        all_vectors: Vectors = []
        for i in range(0, len(texts), self._batch_size):
            sub_batch = texts[i:i + self._batch_size]
            if config.DEBUG:
                logger.debug(
                    f"  Sending sub-batch {i // self._batch_size + 1}: "
                    f"{len(sub_batch)} chunk(s)"
                )
            all_vectors.extend(self._call_api(sub_batch, input_type="passage"))

        return all_vectors

    def dimension(self) -> int:
        """
        Return the embedding dimension for the active NIM model.

        Checks the known-dimensions table first. If the model is
        unknown, probes the API once with a test string and caches
        the result.
        """
        if self._model in _NIM_DIMENSIONS:
            return _NIM_DIMENSIONS[self._model]

        if self._probed_dim is not None:
            return self._probed_dim

        if config.DEBUG:
            logger.debug(
                f"'{self._model}' not in known NIM dimensions table "
                f"- probing API..."
            )

        probe = self._call_api(["probe"], input_type="query")
        self._probed_dim = len(probe[0])

        if config.DEBUG:
            logger.debug(f"Probed dimension: {self._probed_dim} (cached)")

        return self._probed_dim

    def _call_api(self, texts: list[str], input_type: str) -> Vectors:
        """
        POST to the NIM /v1/embeddings endpoint and return normalized vectors.

        Retries retryable failures with exponential backoff.

        Raises:
            RuntimeError: on repeated HTTP failures or unexpected response shape.
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "input": texts,
            "input_type": input_type,
        }

        backoff_seconds = config.NIM_INITIAL_BACKOFF_SECONDS
        max_attempts = config.NIM_MAX_RETRIES + 1

        for attempt in range(1, max_attempts + 1):
            response: requests.Response | None = None

            try:
                response = requests.post(
                    self._endpoint,
                    headers=headers,
                    json=payload,
                    timeout=config.NIM_REQUEST_TIMEOUT_SEC,
                )
                response.raise_for_status()

                data = response.json()
                items = data["data"]
                items.sort(key=lambda item: item["index"])
                vectors = [item["embedding"] for item in items]
                return normalize_vectors(vectors)
            except requests.exceptions.Timeout as exc:
                error_message = (
                    f"NIM API request timed out after "
                    f"{config.NIM_REQUEST_TIMEOUT_SEC:g}s. "
                    "Check NIM_BASE_URL or try again."
                )
                should_retry = True
                cause = exc
            except requests.exceptions.HTTPError as exc:
                status_code = exc.response.status_code if exc.response else "unknown"
                response_text = exc.response.text[:300] if exc.response else str(exc)
                error_message = (
                    f"NIM API HTTP error {status_code}: {response_text}"
                )
                should_retry = bool(
                    exc.response
                    and exc.response.status_code in config.NIM_RETRYABLE_STATUS_CODES
                )
                cause = exc
            except requests.exceptions.RequestException as exc:
                error_message = f"NIM API connection error: {exc}"
                should_retry = True
                cause = exc
            except (KeyError, TypeError, ValueError) as exc:
                response_text = response.text[:300] if response is not None else ""
                raise RuntimeError(
                    f"Unexpected NIM API response shape: {exc}\n"
                    f"Response: {response_text}"
                ) from exc

            if attempt == max_attempts or not should_retry:
                raise RuntimeError(error_message) from cause

            logger.warning(
                f"NIM API call failed on attempt {attempt}/{max_attempts}: "
                f"{error_message} Retrying in {backoff_seconds:.1f}s."
            )
            time.sleep(backoff_seconds)
            backoff_seconds *= 2

        raise RuntimeError("NIM API request failed unexpectedly.")

    @staticmethod
    def _validate_text(text: str, index: int | None) -> None:
        if not text or not text.strip():
            location = f"index {index}" if index is not None else "input"
            raise ValueError(f"Empty or whitespace-only text at {location}.")
