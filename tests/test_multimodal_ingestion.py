import unittest

import config
from src.loaders.pdf_loader import _normalize_text, _table_to_text
from src.utils.chunker import chunk_pages


class PdfLoaderHelperTests(unittest.TestCase):
    def test_normalize_text_preserves_layout_spacing_when_requested(self) -> None:
        raw = "Quarter   Revenue   Growth\nQ1        10        5%\n\n\n"

        normalized = _normalize_text(raw, preserve_layout=True)

        self.assertIn("Quarter   Revenue   Growth", normalized)
        self.assertNotIn("\n\n\n", normalized)

    def test_table_to_text_renders_headers_and_rows(self) -> None:
        table = [
            ["Quarter", "Revenue", "Growth"],
            ["Q1", "$10M", "5%"],
            ["Q2", "$12M", "20%"],
        ]

        text, row_count, column_count = _table_to_text(
            table=table,
            filename="report.pdf",
            page_number=4,
        )

        self.assertEqual(row_count, 2)
        self.assertEqual(column_count, 3)
        self.assertIn("Table extracted from report.pdf page 4.", text)
        self.assertIn("Header row: Quarter | Revenue | Growth", text)
        self.assertIn("Row 1: Quarter = Q1; Revenue = $10M; Growth = 5%", text)


class ChunkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_settings = {
            "CHUNK_STRATEGY": config.CHUNK_STRATEGY,
            "CHUNK_MAX_TOKENS": config.CHUNK_MAX_TOKENS,
            "CHUNK_MIN_TOKENS": config.CHUNK_MIN_TOKENS,
            "CHUNK_SIMILARITY_THRESHOLD": config.CHUNK_SIMILARITY_THRESHOLD,
            "CHUNK_TABLE_ROW_OVERLAP": config.CHUNK_TABLE_ROW_OVERLAP,
            "CHUNK_SIZE": config.CHUNK_SIZE,
            "CHUNK_OVERLAP": config.CHUNK_OVERLAP,
        }

        config.CHUNK_STRATEGY = "semantic"
        config.CHUNK_MAX_TOKENS = 28
        config.CHUNK_MIN_TOKENS = 10
        config.CHUNK_SIMILARITY_THRESHOLD = 0.12
        config.CHUNK_TABLE_ROW_OVERLAP = 1
        config.CHUNK_SIZE = 140
        config.CHUNK_OVERLAP = 20

    def tearDown(self) -> None:
        for name, value in self._original_settings.items():
            setattr(config, name, value)

    def test_chunk_pages_preserves_content_metadata(self) -> None:
        config.CHUNK_MAX_TOKENS = 80

        content_blocks = [
            {
                "text": "Table extracted from report.pdf page 2.\n"
                "Header row: Region | Revenue\n"
                "Row 1: Region = North; Revenue = 100\n"
                "Row 2: Region = South; Revenue = 120",
                "filename": "report.pdf",
                "page_number": 2,
                "content_type": "table",
                "content_index": 1,
                "source": "pdfplumber",
                "row_count": 2,
                "column_count": 2,
            }
        ]

        chunks = chunk_pages(content_blocks)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["metadata"]["content_type"], "table")
        self.assertEqual(chunks[0]["metadata"]["content_index"], 1)
        self.assertEqual(chunks[0]["metadata"]["row_count"], 2)
        self.assertEqual(chunks[0]["metadata"]["chunk_strategy"], "semantic")
        self.assertTrue(chunks[0]["id"].endswith("_table_1_c0"))

    def test_semantic_chunking_starts_new_chunk_for_new_heading(self) -> None:
        content_blocks = [
            {
                "text": (
                    "Executive Summary\n\n"
                    "Revenue improved across enterprise accounts because renewal rates "
                    "rose and upsell activity accelerated in the second quarter.\n\n"
                    "Operational Risks\n\n"
                    "Migration delays increased after two platform teams were moved onto "
                    "security remediation work."
                ),
                "filename": "report.pdf",
                "page_number": 1,
                "content_type": "page_text",
                "content_index": 0,
                "source": "pypdf/plain",
            }
        ]

        chunks = chunk_pages(content_blocks)

        self.assertEqual(len(chunks), 2)
        self.assertIn("Executive Summary", chunks[0]["text"])
        self.assertIn("Revenue improved", chunks[0]["text"])
        self.assertIn("Operational Risks", chunks[1]["text"])
        self.assertIn("Migration delays", chunks[1]["text"])

    def test_semantic_chunking_splits_large_paragraph_by_sentence(self) -> None:
        config.CHUNK_MAX_TOKENS = 16
        config.CHUNK_MIN_TOKENS = 6

        content_blocks = [
            {
                "text": (
                    "Revenue increased across all regions because enterprise renewals "
                    "came in above forecast. The commercial segment also improved after "
                    "pricing changes were rolled out. Support costs stayed elevated due "
                    "to backlog reduction work."
                ),
                "filename": "report.pdf",
                "page_number": 3,
                "content_type": "page_text",
                "content_index": 0,
                "source": "pypdf/plain",
            }
        ]

        chunks = chunk_pages(content_blocks)

        self.assertGreater(len(chunks), 1)
        self.assertIn("Revenue increased across all regions", chunks[0]["text"])
        self.assertTrue(
            any("Support costs stayed elevated" in chunk["text"] for chunk in chunks)
        )


if __name__ == "__main__":
    unittest.main()
