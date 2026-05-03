import os
from dotenv import load_dotenv

load_dotenv()

# Paths
DOCS_PATH         = os.getenv("DOCS_PATH",   "data/docs")
CHROMA_PATH       = os.getenv("CHROMA_PATH", "data/chroma")

# Chunking
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE",    "500"))  # characters
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP", "50"))   # characters

# Embedding
EMBED_MODE        = os.getenv("EMBED_MODE", "sentence_transformers") # options: "sentence_transformers" | "ollama"
SENTENCE_MODEL    = os.getenv("SENTENCE_MODEL",    "all-MiniLM-L6-v2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
EMBED_BATCH_SIZE   = int(os.getenv("EMBED_BATCH_SIZE", "32"))
EMBED_DEVICE       = os.getenv("EMBED_DEVICE", "auto") # options: "auto" | "cuda" | "cpu"

# Ollama
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL         = os.getenv("LLM_MODEL",       "llama3.2:3b")

# Retrieval
TOP_K             = int(os.getenv("TOP_K", "3"))  # number of chunks returned

# ChromaDB
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_docs")
CHROMA_UPSERT_BATCH_SIZE = int(os.getenv("CHROMA_UPSERT_BATCH_SIZE", "100"))

# Debug
DEBUG             = os.getenv("DEBUG", "true").lower() == "true"

# Validation
if CHUNK_SIZE <= 0:
    raise ValueError("CHUNK_SIZE must be greater than 0")

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

if TOP_K < 1:
    raise ValueError("TOP_K must be at least 1")