import os
from dotenv import load_dotenv

load_dotenv()

# ======================
# Ollama (Local LLM)
# ======================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# ======================
# Vector Store
# ======================
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

# ======================
# Chunking & Retrieval
# ======================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "3"))

# Embedding model (local, free)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"