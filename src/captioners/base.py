from abc import ABC, abstractmethod


class BaseImageCaptioner(ABC):
    """Convert an image into retrievable text."""

    @abstractmethod
    def name(self) -> str:
        """Return a human-readable identifier for the captioner."""
        pass

    @abstractmethod
    def describe(
        self,
        image_bytes: bytes,
        mime_type: str,
        filename: str,
        page_number: int,
        image_index: int,
    ) -> str:
        """
        Describe an image as plain text suitable for embedding.

        Args:
            image_bytes: Raw image bytes extracted from the PDF.
            mime_type: MIME type such as image/png.
            filename: Source PDF filename.
            page_number: 1-based source page number.
            image_index: 0-based image position on the page.

        Returns:
            Plain text description or OCR result for embedding.
        """
        pass
