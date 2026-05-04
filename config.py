import os
from dotenv import load_dotenv

load_dotenv()


def _parse_int_set(raw: str) -> set[int]:
    """Parse a comma-separated list of integers into a set."""
    values = {int(part.strip()) for part in raw.split(",") if part.strip()}
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


# Paths
DOCS_PATH = os.getenv("DOCS_PATH", "data/docs")
CHROMA_PATH = os.getenv("CHROMA_PATH", "data/chroma")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # characters

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
        "408,409,425,429,500,502,503,504",
    )
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

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

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

if LOG_LEVEL not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}:
    raise ValueError(
        "LOG_LEVEL must be one of: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET"
    )

if TOP_K < 1:
    raise ValueError("TOP_K must be at least 1")
