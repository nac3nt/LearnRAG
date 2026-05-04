import re
from dataclasses import dataclass

import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_NUMERIC_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*[\).]?\s+\S+")
_BULLET_RE = re.compile("^(?:[-*]|\\u2022|(?:\\d+|[a-zA-Z])[\\).])\\s+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


@dataclass(frozen=True)
class _TextUnit:
    """A semantic text unit used during chunk assembly."""

    text: str
    kind: str
    token_count: int


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Take a list of content-block dicts from pdf_loader and return
    a flat list of chunk dicts with full metadata.

    Input:
    [
        {
            "text": "page text, table text, or image description...",
            "page_number": 1,
            "filename": "document.pdf",
            "content_type": "page_text" | "table" | "image",
            "content_index": 0,
        },
        ...
    ]

    Output:
    [
        {
            "text": "chunk text...",
            "metadata": {
                "filename": "document.pdf",
                "page_number": 1,
                "content_type": "table",
                "content_index": 2,
                "chunk_index": 0,
                "char_count": 342,
                "token_count": 61,
                "chunk_strategy": "semantic"
            },
            "id": "document.pdf_p1_table_2_c0"
        },
        ...
    ]
    """
    all_chunks = []

    for page in pages:
        content_type = page.get("content_type", "page_text")
        chunks = _chunk_content(
            text=page["text"],
            content_type=content_type,
        )

        for chunk_index, chunk_text in enumerate(chunks):
            chunk_id = _build_id(
                filename=page["filename"],
                page_number=page["page_number"],
                content_type=content_type,
                content_index=page.get("content_index", 0),
                chunk_index=chunk_index,
            )

            metadata = {key: value for key, value in page.items() if key != "text"}
            metadata["chunk_index"] = chunk_index
            metadata["char_count"] = len(chunk_text)
            metadata["token_count"] = _estimate_token_count(chunk_text)
            metadata["chunk_strategy"] = config.CHUNK_STRATEGY

            all_chunks.append(
                {
                    "text": chunk_text,
                    "metadata": metadata,
                    "id": chunk_id,
                }
            )

        if config.DEBUG:
            logger.debug(
                f"{page['filename']} p{page['page_number']} -> {len(chunks)} chunk(s)"
            )

    if all_chunks and config.DEBUG:
        avg_len = sum(c["metadata"]["char_count"] for c in all_chunks) / len(all_chunks)
        avg_tokens = sum(c["metadata"]["token_count"] for c in all_chunks) / len(all_chunks)
        logger.debug(f"Total chunks created : {len(all_chunks)}")
        logger.debug(f"Average chunk length : {avg_len:.0f} characters")
        logger.debug(f"Average chunk tokens : {avg_tokens:.0f}")

    return all_chunks


def _chunk_content(text: str, content_type: str) -> list[str]:
    """Chunk a content block using the configured strategy."""
    if config.CHUNK_STRATEGY == "fixed":
        return _chunk_content_fixed(text=text, content_type=content_type)

    if content_type == "table":
        return _chunk_table_text_semantic(
            text=text,
            max_tokens=config.CHUNK_MAX_TOKENS,
            row_overlap=config.CHUNK_TABLE_ROW_OVERLAP,
        )

    if content_type == "image":
        return _chunk_text_by_sentences(
            text=text,
            max_tokens=config.CHUNK_MAX_TOKENS,
            min_tokens=config.CHUNK_MIN_TOKENS,
        )

    return _chunk_page_text_semantic(
        text=text,
        max_tokens=config.CHUNK_MAX_TOKENS,
        min_tokens=config.CHUNK_MIN_TOKENS,
        similarity_threshold=config.CHUNK_SIMILARITY_THRESHOLD,
    )


def _chunk_content_fixed(text: str, content_type: str) -> list[str]:
    """Apply the legacy fixed-width chunker."""
    if content_type == "table":
        return _chunk_table_text_fixed(
            text=text,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )

    return _chunk_text_fixed(
        text=text,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )


def _chunk_page_text_semantic(
    text: str,
    max_tokens: int,
    min_tokens: int,
    similarity_threshold: float,
) -> list[str]:
    """Chunk page text by semantic blocks and soft token budgets."""
    units = _split_page_text_units(text)
    if not units:
        return []

    chunks: list[str] = []
    current_units: list[_TextUnit] = []

    for unit in units:
        for fragment in _split_oversized_unit(unit, max_tokens, min_tokens):
            if fragment.kind == "heading":
                if current_units and _current_has_body(current_units):
                    chunks.append(_join_units(current_units))
                current_units = [fragment]
                continue

            if not current_units:
                current_units = [fragment]
                continue

            current_tokens = sum(item.token_count for item in current_units)
            candidate_tokens = current_tokens + fragment.token_count

            if _current_expects_follow_up(current_units):
                current_units.append(fragment)
                continue

            anchor_text = _semantic_anchor_text(current_units)
            similarity = _semantic_similarity(anchor_text, fragment.text)

            if candidate_tokens > max_tokens and current_tokens >= min_tokens:
                chunks.append(_join_units(current_units))
                current_units = [fragment]
                continue

            if current_tokens >= min_tokens and similarity < similarity_threshold:
                chunks.append(_join_units(current_units))
                current_units = [fragment]
                continue

            current_units.append(fragment)

    if current_units:
        chunks.append(_join_units(current_units))

    return [chunk for chunk in chunks if chunk.strip()]


def _chunk_text_by_sentences(text: str, max_tokens: int, min_tokens: int) -> list[str]:
    """Split oversized free text on sentence boundaries before falling back."""
    unit = _make_unit(text=text, kind="paragraph")
    fragments = _split_oversized_unit(unit, max_tokens, min_tokens)
    return [fragment.text for fragment in fragments if fragment.text.strip()]


def _chunk_table_text_semantic(
    text: str,
    max_tokens: int,
    row_overlap: int,
) -> list[str]:
    """
    Chunk tables by row while preserving the header row in every chunk.

    Tables are already structured text, so the semantic strategy is to keep
    the table title and header stable while packing related rows together
    under a soft token budget.
    """
    if _estimate_token_count(text) <= max_tokens:
        return [text]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 2:
        return _chunk_text_by_sentences(
            text=text,
            max_tokens=max_tokens,
            min_tokens=1,
        )

    prefix_lines = lines[:2]
    row_lines = lines[2:]
    chunks = []
    current_rows: list[str] = []

    for row in row_lines:
        row_token_count = _estimate_token_count(row)

        if row_token_count > max_tokens:
            if current_rows:
                chunks.append("\n".join(prefix_lines + current_rows))
                current_rows = []

            oversized_parts = _chunk_text_by_sentences(
                text=row,
                max_tokens=max_tokens,
                min_tokens=1,
            )
            for part in oversized_parts:
                chunks.append("\n".join(prefix_lines + [part]))
            continue

        candidate_rows = current_rows + [row]
        candidate_text = "\n".join(prefix_lines + candidate_rows)

        if current_rows and _estimate_token_count(candidate_text) > max_tokens:
            chunks.append("\n".join(prefix_lines + current_rows))
            overlap_rows = current_rows[-row_overlap:] if row_overlap > 0 else []
            current_rows = overlap_rows + [row]

            while (
                len(current_rows) > 1
                and _estimate_token_count("\n".join(prefix_lines + current_rows)) > max_tokens
            ):
                current_rows.pop(0)
        else:
            current_rows.append(row)

    if current_rows:
        chunks.append("\n".join(prefix_lines + current_rows))

    return chunks


def _split_page_text_units(text: str) -> list[_TextUnit]:
    """Split page text into headings, bullets, and paragraph units."""
    units = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return

        paragraph_text = _join_paragraph_lines(paragraph_lines)
        if paragraph_text:
            units.append(_make_unit(text=paragraph_text, kind="paragraph"))
        paragraph_lines.clear()

    for raw_line in text.splitlines():
        stripped = raw_line.strip()

        if not stripped:
            flush_paragraph()
            continue

        if _is_heading_line(stripped):
            flush_paragraph()
            units.append(_make_unit(text=_normalize_inline_text(stripped), kind="heading"))
            continue

        if _is_bullet_line(stripped):
            flush_paragraph()
            units.append(_make_unit(text=_normalize_inline_text(stripped), kind="bullet"))
            continue

        paragraph_lines.append(stripped)

    flush_paragraph()

    if not units and text.strip():
        units.append(_make_unit(text=_normalize_inline_text(text), kind="paragraph"))

    return units


def _split_oversized_unit(
    unit: _TextUnit,
    max_tokens: int,
    min_tokens: int,
) -> list[_TextUnit]:
    """Break a large unit into sentence-level fragments if needed."""
    if unit.token_count <= max_tokens:
        return [unit]

    sentences = _split_sentences(unit.text)
    if len(sentences) <= 1:
        return [
            _make_unit(text=chunk, kind=unit.kind)
            for chunk in _chunk_text_fixed(
                text=unit.text,
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
            )
        ]

    fragments: list[_TextUnit] = []
    current_sentences: list[str] = []

    for sentence in sentences:
        sentence_tokens = _estimate_token_count(sentence)

        if sentence_tokens > max_tokens:
            if current_sentences:
                fragments.append(
                    _make_unit(
                        text=" ".join(current_sentences),
                        kind=unit.kind,
                    )
                )
                current_sentences = []

            fragments.extend(
                _make_unit(text=chunk, kind=unit.kind)
                for chunk in _chunk_text_fixed(
                    text=sentence,
                    chunk_size=config.CHUNK_SIZE,
                    chunk_overlap=config.CHUNK_OVERLAP,
                )
            )
            continue

        candidate_sentences = current_sentences + [sentence]
        candidate_tokens = _estimate_token_count(" ".join(candidate_sentences))
        current_tokens = _estimate_token_count(" ".join(current_sentences))

        if current_sentences and candidate_tokens > max_tokens and current_tokens >= min_tokens:
            fragments.append(
                _make_unit(
                    text=" ".join(current_sentences),
                    kind=unit.kind,
                )
            )
            current_sentences = [sentence]
            continue

        current_sentences.append(sentence)

    if current_sentences:
        fragments.append(
            _make_unit(
                text=" ".join(current_sentences),
                kind=unit.kind,
            )
        )

    return fragments


def _chunk_text_fixed(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split a single string into overlapping chunks by character count.

    Uses a sliding window:
    - window size = chunk_size
    - step size   = chunk_size - chunk_overlap

    Attempts to avoid mid-word splits by backing up to the nearest
    whitespace. Falls back to hard split if no whitespace is found.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})")

    chunks = []
    step = chunk_size - chunk_overlap
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary != -1:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


def _chunk_table_text_fixed(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Chunk tables by line while preserving the header row in every chunk.

    This is the legacy table chunker used when CHUNK_STRATEGY=fixed.
    """
    if len(text) <= chunk_size:
        return [text]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 2:
        return _chunk_text_fixed(text, chunk_size, chunk_overlap)

    prefix_lines = lines[:2]
    row_lines = lines[2:]
    chunks = []
    start_index = 0
    overlap_rows = max(1, chunk_overlap // 80) if chunk_overlap > 0 else 0

    while start_index < len(row_lines):
        current_lines = prefix_lines.copy()
        cursor = start_index

        while cursor < len(row_lines):
            candidate = current_lines + [row_lines[cursor]]
            if len("\n".join(candidate)) > chunk_size and len(current_lines) > len(prefix_lines):
                break

            current_lines.append(row_lines[cursor])
            cursor += 1

            if len("\n".join(current_lines)) >= chunk_size:
                break

        chunks.append("\n".join(current_lines))

        if cursor >= len(row_lines):
            break

        start_index = max(start_index + 1, cursor - overlap_rows)

    return chunks


def _build_id(
    filename: str,
    page_number: int,
    content_type: str,
    content_index: int,
    chunk_index: int,
) -> str:
    """
    Build a deterministic, filesystem-safe chunk ID.

    Format: "document.pdf_p1_table_0_c0"

    Sanitizes filename to remove characters that could
    cause issues in ChromaDB or filesystem paths.
    """
    safe_filename = re.sub(r"[^\w.\-]", "_", filename)
    safe_content_type = re.sub(r"[^\w.\-]", "_", content_type)
    return (
        f"{safe_filename}_p{page_number}_"
        f"{safe_content_type}_{content_index}_c{chunk_index}"
    )


def _make_unit(text: str, kind: str) -> _TextUnit:
    """Create a text unit with a cached token count."""
    normalized = text.strip()
    return _TextUnit(
        text=normalized,
        kind=kind,
        token_count=_estimate_token_count(normalized),
    )


def _join_units(units: list[_TextUnit]) -> str:
    """Join text units into a chunk while preserving simple structure."""
    return "\n\n".join(unit.text for unit in units if unit.text.strip()).strip()


def _join_paragraph_lines(lines: list[str]) -> str:
    """Join wrapped PDF lines back into a paragraph-like unit."""
    text = " ".join(line.strip() for line in lines if line.strip())
    return _normalize_inline_text(text)


def _normalize_inline_text(text: str) -> str:
    """Collapse repeated whitespace inside a single semantic unit."""
    return re.sub(r"\s+", " ", text).strip()


def _estimate_token_count(text: str) -> int:
    """Estimate token count using word-like and punctuation tokens."""
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-like fragments."""
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [_normalize_inline_text(part) for part in parts if part.strip()]


def _is_heading_line(text: str) -> bool:
    """Heuristically detect section headings in extracted PDF text."""
    stripped = text.strip()
    if not stripped or _is_bullet_line(stripped):
        return False

    token_count = len(_WORD_RE.findall(stripped))
    if token_count == 0 or token_count > 14 or len(stripped) > 100:
        return False

    if _NUMERIC_HEADING_RE.match(stripped):
        return True

    if stripped.endswith(":"):
        return True

    if stripped.endswith((".", "!", "?", ";")):
        return False

    alpha_tokens = [token for token in stripped.split() if any(ch.isalpha() for ch in token)]
    if not alpha_tokens:
        return False

    title_like = sum(token[0].isupper() for token in alpha_tokens)
    uppercase_words = sum(token.isupper() for token in alpha_tokens)
    title_ratio = title_like / len(alpha_tokens)
    uppercase_ratio = uppercase_words / len(alpha_tokens)
    return title_ratio >= 0.8 or uppercase_ratio >= 0.6


def _is_bullet_line(text: str) -> bool:
    """Detect bullet or enumerated list items."""
    return bool(_BULLET_RE.match(text.strip()))


def _current_expects_follow_up(units: list[_TextUnit]) -> bool:
    """Return True when the current chunk should absorb the next unit."""
    return bool(units) and units[-1].kind == "heading"


def _current_has_body(units: list[_TextUnit]) -> bool:
    """Return True when a chunk contains non-heading content."""
    return any(unit.kind != "heading" for unit in units)


def _semantic_anchor_text(units: list[_TextUnit]) -> str:
    """Use the latest non-heading units as the semantic comparison anchor."""
    body_units = [unit.text for unit in units if unit.kind != "heading"]
    if body_units:
        return "\n\n".join(body_units[-2:])
    return units[-1].text if units else ""


def _semantic_similarity(left: str, right: str) -> float:
    """Compute a simple lexical Jaccard similarity between two texts."""
    left_terms = _content_terms(left)
    right_terms = _content_terms(right)

    if not left_terms or not right_terms:
        return 0.0

    intersection = left_terms & right_terms
    union = left_terms | right_terms
    return len(intersection) / len(union)


def _content_terms(text: str) -> set[str]:
    """Extract comparable content terms from text, excluding stopwords."""
    terms = {
        token.lower()
        for token in _WORD_RE.findall(text)
        if token.lower() not in _STOPWORDS
    }
    return terms
