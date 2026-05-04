import os

from dotenv import load_dotenv

load_dotenv()


def _parse_int_set(raw: str) -> set[int]:
    """Parse a comma-separated list of integers into a set."""
    values = {int(part.strip()) for part in raw.split(",") if part.strip()}
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


_DEFAULT_RETRYABLE_STATUS_CODES = "408,409,425,429,500,502,503,504"


# Paths
DOCS_PATH = os.getenv("DOCS_PATH", "data/docs")
CHROMA_PATH = os.getenv("CHROMA_PATH", "data/chroma")

# Chunking
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "semantic").lower()
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "220"))
CHUNK_MIN_TOKENS = int(os.getenv("CHUNK_MIN_TOKENS", "80"))
CHUNK_SIMILARITY_THRESHOLD = float(
    os.getenv("CHUNK_SIMILARITY_THRESHOLD", "0.18")
)
CHUNK_TABLE_ROW_OVERLAP = int(os.getenv("CHUNK_TABLE_ROW_OVERLAP", "1"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # fallback characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # fallback characters

# PDF extraction
PDF_TEXT_EXTRACTION_MODE = os.getenv("PDF_TEXT_EXTRACTION_MODE", "plain").lower()
PDF_EXTRACT_TABLES = os.getenv("PDF_EXTRACT_TABLES", "true").lower() == "true"
PDF_EXTRACT_IMAGES = os.getenv("PDF_EXTRACT_IMAGES", "true").lower() == "true"
PDF_SAVE_EXTRACTED_IMAGES = os.getenv(
    "PDF_SAVE_EXTRACTED_IMAGES",
    "true",
).lower() == "true"
PDF_IMAGE_OUTPUT_PATH = os.getenv("PDF_IMAGE_OUTPUT_PATH", "data/extracted_images")

# Embedding (NVIDIA NIM)
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
NIM_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com")
NIM_EMBED_MODEL = os.getenv("NIM_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
NIM_REQUEST_TIMEOUT_SEC = float(os.getenv("NIM_REQUEST_TIMEOUT_SEC", "60"))
NIM_MAX_RETRIES = int(os.getenv("NIM_MAX_RETRIES", "3"))
NIM_INITIAL_BACKOFF_SECONDS = float(
    os.getenv("NIM_INITIAL_BACKOFF_SECONDS", "1.0")
)
NIM_RETRYABLE_STATUS_CODES = _parse_int_set(
    os.getenv(
        "NIM_RETRYABLE_STATUS_CODES",
        _DEFAULT_RETRYABLE_STATUS_CODES,
    )
)

# Image captioning / vision
IMAGE_CAPTION_PROVIDER = os.getenv("IMAGE_CAPTION_PROVIDER", "none").lower()
VISION_API_KEY = os.getenv("VISION_API_KEY", "")
VISION_BASE_URL = os.getenv("VISION_BASE_URL", "")
VISION_MODEL = os.getenv("VISION_MODEL", "")
VISION_REQUEST_TIMEOUT_SEC = float(
    os.getenv("VISION_REQUEST_TIMEOUT_SEC", str(NIM_REQUEST_TIMEOUT_SEC))
)
VISION_MAX_RETRIES = int(os.getenv("VISION_MAX_RETRIES", str(NIM_MAX_RETRIES)))
VISION_INITIAL_BACKOFF_SECONDS = float(
    os.getenv(
        "VISION_INITIAL_BACKOFF_SECONDS",
        str(NIM_INITIAL_BACKOFF_SECONDS),
    )
)
VISION_RETRYABLE_STATUS_CODES = _parse_int_set(
    os.getenv(
        "VISION_RETRYABLE_STATUS_CODES",
        _DEFAULT_RETRYABLE_STATUS_CODES,
    )
)
VISION_DESCRIPTION_PROMPT = os.getenv(
    "VISION_DESCRIPTION_PROMPT",
    (
        "Describe the business-relevant contents of this PDF image for "
        "retrieval. Transcribe visible text, summarize charts, tables, and "
        "diagrams, and keep the answer factual and concise."
    ),
)

# Retrieval
TOP_K = int(os.getenv("TOP_K", "3"))  # number of chunks returned

# ChromaDB
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_docs")
CHROMA_UPSERT_BATCH_SIZE = int(os.getenv("CHROMA_UPSERT_BATCH_SIZE", "100"))
CHROMA_DISTANCE_SPACE = os.getenv("CHROMA_DISTANCE_SPACE", "cosine")

# Debug
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

# Logging
LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "DEBUG" if DEBUG else "INFO").upper()
LOG_FORMAT = os.getenv(
    "APP_LOG_FORMAT",
    "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
)
LOG_DATE_FORMAT = os.getenv("APP_LOG_DATE_FORMAT", "%H:%M:%S")
LOG_TO_FILE = os.getenv("APP_LOG_TO_FILE", "false").lower() == "true"
LOG_FILE = os.getenv("APP_LOG_FILE", "logs/app.log")
LOG_FILE_ENCODING = os.getenv("APP_LOG_FILE_ENCODING", "utf-8")

# Validation
if CHUNK_SIZE <= 0:
    raise ValueError("CHUNK_SIZE must be greater than 0")

if CHUNK_OVERLAP < 0:
    raise ValueError("CHUNK_OVERLAP must be at least 0")

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

if CHUNK_STRATEGY not in {"fixed", "semantic"}:
    raise ValueError("CHUNK_STRATEGY must be one of: fixed, semantic")

if CHUNK_MAX_TOKENS <= 0:
    raise ValueError("CHUNK_MAX_TOKENS must be greater than 0")

if CHUNK_MIN_TOKENS <= 0:
    raise ValueError("CHUNK_MIN_TOKENS must be greater than 0")

if CHUNK_MIN_TOKENS > CHUNK_MAX_TOKENS:
    raise ValueError("CHUNK_MIN_TOKENS must be <= CHUNK_MAX_TOKENS")

if not 0 <= CHUNK_SIMILARITY_THRESHOLD <= 1:
    raise ValueError("CHUNK_SIMILARITY_THRESHOLD must be in [0, 1]")

if CHUNK_TABLE_ROW_OVERLAP < 0:
    raise ValueError("CHUNK_TABLE_ROW_OVERLAP must be at least 0")

if PDF_TEXT_EXTRACTION_MODE not in {"plain", "layout"}:
    raise ValueError("PDF_TEXT_EXTRACTION_MODE must be one of: plain, layout")

if EMBED_BATCH_SIZE < 1:
    raise ValueError("EMBED_BATCH_SIZE must be at least 1")

if CHROMA_DISTANCE_SPACE not in {"cosine", "l2", "ip"}:
    raise ValueError("CHROMA_DISTANCE_SPACE must be one of: cosine, l2, ip")

if NIM_REQUEST_TIMEOUT_SEC <= 0:
    raise ValueError("NIM_REQUEST_TIMEOUT_SEC must be greater than 0")

if NIM_MAX_RETRIES < 0:
    raise ValueError("NIM_MAX_RETRIES must be at least 0")

if NIM_INITIAL_BACKOFF_SECONDS <= 0:
    raise ValueError("NIM_INITIAL_BACKOFF_SECONDS must be greater than 0")

if any(code < 100 or code > 599 for code in NIM_RETRYABLE_STATUS_CODES):
    raise ValueError("NIM_RETRYABLE_STATUS_CODES must contain valid HTTP status codes")

if IMAGE_CAPTION_PROVIDER not in {"none", "openai_compatible"}:
    raise ValueError(
        "IMAGE_CAPTION_PROVIDER must be one of: none, openai_compatible"
    )

if VISION_REQUEST_TIMEOUT_SEC <= 0:
    raise ValueError("VISION_REQUEST_TIMEOUT_SEC must be greater than 0")

if VISION_MAX_RETRIES < 0:
    raise ValueError("VISION_MAX_RETRIES must be at least 0")

if VISION_INITIAL_BACKOFF_SECONDS <= 0:
    raise ValueError("VISION_INITIAL_BACKOFF_SECONDS must be greater than 0")

if any(code < 100 or code > 599 for code in VISION_RETRYABLE_STATUS_CODES):
    raise ValueError("VISION_RETRYABLE_STATUS_CODES must contain valid HTTP status codes")

if LOG_LEVEL not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}:
    raise ValueError(
        "LOG_LEVEL must be one of: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET"
    )

if TOP_K < 1:
    raise ValueError("TOP_K must be at least 1")
