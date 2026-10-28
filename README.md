# Company Knowledge Assistant (Ollama Edition)

**End-to-End Capstone Project**  
RAG + FastAPI + Unit Testing + CI/CD (GitHub Actions) + AWS Deployment  
**Powered by local Ollama LLM** (no OpenAI / paid API required)

A beginner-friendly company internal knowledge base Q&A system that answers questions about HR policies, IT guides, and FAQs using Retrieval-Augmented Generation (RAG) with a **completely local** LLM.

---

## Features

- Fully local & private (embeddings + LLM run on your machine)
- Document loading, chunking, embedding & indexing (ChromaDB)
- FastAPI REST API with `/ask` and `/health` endpoints
- Source citations for every answer
- Graceful refusal for questions outside the knowledge base
- Unit tests + GitHub Actions CI
- Docker-ready

---

## Prerequisites

1. **Python 3.10+**
2. **Ollama** installed → https://ollama.com
3. Pull a model (recommended for beginners):

```bash
ollama pull llama3.2
# or
ollama pull mistral
# or
ollama pull phi3
```

Start Ollama (if not already running as a service):

```bash
ollama serve
```

---

## Project Structure

```
company-knowledge-assistant/
├── data/                   # Sample knowledge base (HR, IT, FAQ)
├── app/
│   ├── main.py             # FastAPI application
│   ├── rag_pipeline.py     # RAG logic (uses ChatOllama)
│   ├── config.py
│   └── schemas.py
├── scripts/
│   └── build_index.py      # Build the vector store
├── tests/
├── .github/workflows/
│   └── ci-cd.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick Start (Local)

### 1. Unzip and create virtual environment

```bash
cd company-knowledge-assistant
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment (optional)

```bash
cp .env.example .env
# Default values work out of the box if Ollama runs on localhost:11434
```

### 3. Build the vector index

```bash
python scripts/build_index.py
```

### 4. Start the API

```bash
uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** and try the interactive Swagger UI.

### Example Request

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days of annual leave do employees get?"}'
```

---

## Changing the Ollama Model

Edit `.env` (or set environment variables):

```env
OLLAMA_MODEL=mistral
# or llama3.1, phi3, gemma2, etc.
```

Make sure you have pulled the model first:

```bash
ollama pull mistral
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Docker Notes

The application container does **not** include Ollama.  
You must run Ollama separately (on the host or in another container).

Example:

```bash
# Terminal 1 – Ollama
ollama serve

# Terminal 2 – App
docker build -t company-knowledge-assistant .
docker run -p 8000:8000 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  company-knowledge-assistant
```

---

## Sample Questions to Try

- How many days of annual leave do employees get?
- How do I reset my password?
- What is the process to request a laptop?
- What are the working hours?
- What wellness benefits are available?
- Who won the last FIFA World Cup?  ← should refuse

---

## Tech Stack

| Layer              | Technology                          |
|--------------------|-------------------------------------|
| Framework          | FastAPI + Uvicorn                   |
| RAG Orchestration  | LangChain                           |
| Vector Store       | ChromaDB                            |
| Embeddings         | sentence-transformers (local)       |
| LLM                | **Ollama** (local)                  |
| Testing            | pytest                              |
| CI/CD              | GitHub Actions                      |
| Container          | Docker                              |

---

## Advantages of Using Ollama

- Completely free
- Data never leaves your machine
- No API keys or rate limits
- Works offline
- Easy to switch models (`llama3.2`, `mistral`, `phi3`, etc.)

---

## License

MIT – feel free to use this as a starting point for your own projects.
