"""
Build the vector index from documents in the data/ folder.
Run this once before starting the API (or whenever documents change).

Usage:
    python scripts/build_index.py
"""

import sys
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from app.config import CHROMA_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL


def load_documents():
    """Load all .md and .txt files from the data folder (Windows-friendly)."""
    data_path = project_root / "data"
    print(f"Looking for documents in: {data_path}")

    if not data_path.exists():
        print(f"ERROR: data folder not found at {data_path}")
        return []

    documents = []
    # Support both .md and .txt files
    for pattern in ("**/*.md", "**/*.txt"):
        for file_path in data_path.glob(pattern):
            try:
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = str(file_path.relative_to(project_root))
                documents.extend(docs)
                print(f"  ✓ Loaded: {file_path.relative_to(project_root)}")
            except Exception as e:
                print(f"  ✗ Failed to load {file_path}: {e}")

    return documents


def build_vector_store():
    print("Loading documents from data/ ...")
    documents = load_documents()

    if not documents:
        print("\nNo documents found in data/ folder!")
        print("Please make sure the following structure exists:")
        print("  data/")
        print("  ├── hr/")
        print("  │   ├── leave_policy.txt")
        print("  │   └── onboarding.txt")
        print("  ├── it/")
        print("  │   ├── password_reset.txt")
        print("  │   └── laptop_request.txt")
        print("  └── faq/")
        print("      ├── general_faq.txt")
        print("      └── benefits.txt")
        return

    print(f"\nLoaded {len(documents)} documents.")

    print("Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print(f"Creating embeddings with {EMBEDDING_MODEL} (this may take a minute)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Remove old index if exists
    chroma_path = project_root / CHROMA_PATH
    if chroma_path.exists():
        print(f"Removing old index at {chroma_path} ...")
        shutil.rmtree(chroma_path)

    print(f"Building and saving Chroma index to {CHROMA_PATH} ...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(chroma_path),
    )

    print(f"\n✅ Successfully indexed {len(chunks)} chunks into {CHROMA_PATH}")
    print("You can now start the API with:")
    print("    uvicorn app.main:app --reload")
    print("Make sure Ollama is running: ollama serve")


if __name__ == "__main__":
    build_vector_store()