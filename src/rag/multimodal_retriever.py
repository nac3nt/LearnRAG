from src.embeddings.base import BaseEmbedder
from src.embeddings.image_base import BaseImageEmbedder
from src.embeddings.image_factory import load_image_embedder
from src.embeddings.nim_embedder import NIMEmbedder
from src.utils.logger import get_logger
from src.vectordb.chroma_store import ChromaStore
import config

logger = get_logger(__name__)


class MultimodalRetriever:
    """
    Retrieve from text and image vector collections and merge the results.

    Image results are only included when an image embedder is configured and
    it supports text queries in the same embedding space as the ingested
    image vectors.
    """

    def __init__(
        self,
        text_store: ChromaStore | None = None,
        text_embedder: BaseEmbedder | None = None,
        image_store: ChromaStore | None = None,
        image_embedder: BaseImageEmbedder | None = None,
    ) -> None:
        # Store arguments without triggering any network calls or API validation.
        # NIMEmbedder and ChromaStore are constructed lazily on first use via properties.
        self._text_store_arg    = text_store
        self._text_embedder_arg = text_embedder
        self._image_store_arg   = image_store
        self._image_embedder_arg = image_embedder

        # Cached resolved instances (populated on first access)
        self.__text_store:    ChromaStore | None        = None
        self.__text_embedder: BaseEmbedder | None       = None
        self.__image_store:   ChromaStore | None        = None
        self.__image_embedder: BaseImageEmbedder | None = None
        self.__image_embedder_resolved: bool            = False

    @property
    def _text_store(self) -> ChromaStore:
        if self.__text_store is None:
            self.__text_store = self._text_store_arg or ChromaStore()
        return self.__text_store

    @property
    def _text_embedder(self) -> BaseEmbedder:
        if self.__text_embedder is None:
            self.__text_embedder = self._text_embedder_arg or NIMEmbedder()
        return self.__text_embedder

    @property
    def _image_store(self) -> ChromaStore:
        if self.__image_store is None:
            self.__image_store = self._image_store_arg or ChromaStore(
                collection_name=config.CHROMA_IMAGE_COLLECTION
            )
        return self.__image_store

    @property
    def _image_embedder(self) -> BaseImageEmbedder | None:
        if not self.__image_embedder_resolved:
            self.__image_embedder = (
                self._image_embedder_arg
                if self._image_embedder_arg is not None
                else load_image_embedder()
            )
            self.__image_embedder_resolved = True
        return self.__image_embedder

    def retrieve(
        self,
        query: str,
        top_k: int = config.TOP_K,
        filters: dict | None = None,
        include_images: bool = True,
    ) -> list[dict]:
        """Retrieve the best text and image matches for a text query."""
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        results = []

        text_vector = self._text_embedder.embed(query)
        text_results = self._text_store.query(text_vector, top_k=top_k, filters=filters)
        results.extend(_tag_results(text_results, retrieval_modality="text"))

        if include_images:
            image_results = self._retrieve_image_results(
                query=query,
                top_k=top_k,
                filters=filters,
            )
            results.extend(image_results)

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]

    def _retrieve_image_results(
        self,
        query: str,
        top_k: int,
        filters: dict | None,
    ) -> list[dict]:
        """Retrieve image-vector matches for a text query when supported."""
        if self._image_embedder is None:
            return []

        if not self._image_embedder.supports_text_queries():
            logger.warning(
                "Image embedder '%s' does not support text queries. "
                "Skipping image-vector retrieval.",
                self._image_embedder.name(),
            )
            return []

        if self._image_store.count() == 0:
            return []

        image_vector = self._image_embedder.embed_query(query)
        image_results = self._image_store.query(
            image_vector,
            top_k=top_k,
            filters=filters,
        )
        return _tag_results(image_results, retrieval_modality="image")


def _tag_results(results: list[dict], retrieval_modality: str) -> list[dict]:
    """Annotate raw store results with their retrieval modality."""
    tagged = []
    for result in results:
        tagged_result = dict(result)
        metadata = dict(tagged_result.get("metadata", {}))
        metadata["retrieval_modality"] = retrieval_modality
        tagged_result["metadata"] = metadata
        tagged.append(tagged_result)
    return tagged
