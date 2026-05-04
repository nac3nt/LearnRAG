import os
from dotenv import load_dotenv

load_dotenv()

# Paths
DOCS_PATH         = os.getenv("DOCS_PATH",   "data/docs")
CHROMA_PATH       = os.getenv("CHROMA_PATH", "data/chroma")

# Chunking
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE",    "500"))  # characters
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP", "50"))   # characters

# Embedding (NVIDIA NIM)
EMBED_BATCH_SIZE  = int(os.getenv("EMBED_BATCH_SIZE", "32"))
NIM_API_KEY       = os.getenv("NVIDIA_API_KEY", "")
NIM_BASE_URL      = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com")
NIM_EMBED_MODEL   = os.getenv("NIM_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")

# Retrieval
TOP_K             = int(os.getenv("TOP_K", "3"))  # number of chunks returned

# ChromaDB
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_docs")
CHROMA_UPSERT_BATCH_SIZE = int(os.getenv("CHROMA_UPSERT_BATCH_SIZE", "100"))

# Debug
DEBUG             = os.getenv("DEBUG", "true").lower() == "true"

# Logging
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "false").lower() == "true"
LOG_FILE    = os.getenv("LOG_FILE", "logs/app.log")

# Validation
if CHUNK_SIZE <= 0:
    raise ValueError("CHUNK_SIZE must be greater than 0")

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

if EMBED_BATCH_SIZE < 1:
    raise ValueError("EMBED_BATCH_SIZE must be at least 1")

if TOP_K < 1:
    raise ValueError("TOP_K must be at least 1")
