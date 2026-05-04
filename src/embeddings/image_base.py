from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.embeddings.base import Vector, Vectors


@dataclass(frozen=True)
class ImageEmbeddingInput:
    """Raw image input and source metadata for multimodal embedding."""

    image_bytes: bytes
    mime_type: str
    filename: str
    page_number: int
    image_index: int
    asset_path: str = ""
    text: str = ""


class BaseImageEmbedder(ABC):
    """
    Abstract base class for image embedders used in multimodal retrieval.

    Implementations are expected to embed raw image bytes, and when
    retrieval should support text queries against image vectors, they
    must also expose embed_query() in the same vector space.
    """

    @abstractmethod
    def name(self) -> str:
        """Return a human-readable identifier for this embedder."""
        pass

    @abstractmethod
    def embed_image(self, image: ImageEmbeddingInput) -> Vector:
        """Embed a single image and return its vector."""
        pass

    @abstractmethod
    def embed_image_batch(self, images: list[ImageEmbeddingInput]) -> Vectors:
        """Embed a batch of images and return vectors in input order."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> Vector:
        """
        Embed a text query into the same space as image vectors.

        This is what enables text-to-image retrieval, which is the key
        requirement for mixing image vectors into standard enterprise
        search flows.
        """
        pass

    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality produced by this embedder."""
        pass

    @abstractmethod
    def supports_text_queries(self) -> bool:
        """Return True when text queries can search the image vector space."""
        pass
