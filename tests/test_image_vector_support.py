import unittest

from src.embeddings.base import BaseEmbedder
from src.embeddings.image_base import BaseImageEmbedder, ImageEmbeddingInput
from src.rag.multimodal_retriever import MultimodalRetriever
from src.utils.chunker import chunk_pages
from src.utils.image_records import build_image_vector_items


class ImageVectorItemTests(unittest.TestCase):
    def test_build_image_vector_items_preserves_image_metadata(self) -> None:
        content_blocks = [
            {
                "text": "",
                "filename": "report.pdf",
                "page_number": 2,
                "content_type": "image",
                "content_index": 0,
                "source": "pypdf/image",
                "mime_type": "image/png",
                "asset_path": "data/extracted_images/report/page_0002_image_001.png",
                "image_bytes": b"\x89PNGtest",
            }
        ]

        items = build_image_vector_items(content_blocks)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["record"]["metadata"]["vector_modality"], "image")
        self.assertFalse(items[0]["record"]["metadata"]["has_text_representation"])
        self.assertEqual(items[0]["input"].mime_type, "image/png")
        self.assertEqual(items[0]["input"].filename, "report.pdf")
        self.assertNotIn("image_bytes", items[0]["record"]["metadata"])

    def test_chunk_pages_skips_image_blocks_without_text(self) -> None:
        content_blocks = [
            {
                "text": "",
                "filename": "report.pdf",
                "page_number": 2,
                "content_type": "image",
                "content_index": 0,
                "source": "pypdf/image",
                "mime_type": "image/png",
                "image_bytes": b"\x89PNGtest",
            }
        ]

        chunks = chunk_pages(content_blocks)

        self.assertEqual(chunks, [])


class MultimodalRetrieverTests(unittest.TestCase):
    def test_retrieve_merges_text_and_image_results(self) -> None:
        text_store = _StubStore(
            [
                {
                    "id": "text-1",
                    "text": "Revenue improved in enterprise accounts.",
                    "metadata": {"filename": "report.pdf"},
                    "score": 0.71,
                }
            ]
        )
        image_store = _StubStore(
            [
                {
                    "id": "image-1",
                    "text": "Image extracted from report.pdf page 4 (image 1).",
                    "metadata": {"filename": "report.pdf", "asset_path": "data/img.png"},
                    "score": 0.93,
                }
            ]
        )

        retriever = MultimodalRetriever(
            text_store=text_store,
            text_embedder=_StubTextEmbedder(),
            image_store=image_store,
            image_embedder=_StubImageEmbedder(),
        )

        results = retriever.retrieve("revenue chart", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "image-1")
        self.assertEqual(results[0]["metadata"]["retrieval_modality"], "image")
        self.assertEqual(results[1]["metadata"]["retrieval_modality"], "text")


class _StubTextEmbedder(BaseEmbedder):
    def name(self) -> str:
        return "stub/text"

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def dimension(self) -> int:
        return 2


class _StubImageEmbedder(BaseImageEmbedder):
    def name(self) -> str:
        return "stub/image"

    def embed_image(self, image: ImageEmbeddingInput) -> list[float]:
        return [0.0, 1.0]

    def embed_image_batch(self, images: list[ImageEmbeddingInput]) -> list[list[float]]:
        return [[0.0, 1.0] for _ in images]

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 1.0]

    def dimension(self) -> int:
        return 2

    def supports_text_queries(self) -> bool:
        return True


class _StubStore:
    def __init__(self, results: list[dict]) -> None:
        self._results = results

    def query(self, vector, top_k: int, filters=None) -> list[dict]:
        return self._results[:top_k]

    def count(self) -> int:
        return len(self._results)


if __name__ == "__main__":
    unittest.main()
