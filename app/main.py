from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from app.schemas import QueryRequest, QueryResponse, Source, HealthResponse
from app.rag_pipeline import get_qa_chain

# Global variable to hold the QA chain
qa_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global qa_chain
    print("Loading RAG pipeline with Ollama...")
    
    # TEMPORARY: Let the real error appear
    qa_chain = get_qa_chain()
    print("RAG pipeline loaded successfully (using Ollama).")
    
    yield
    print("Shutting down...")


app = FastAPI(
    title="Company Knowledge Assistant",
    description="RAG-powered Q&A system for internal company documents (HR, IT, FAQ) using local Ollama LLM",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    status = "healthy" if qa_chain is not None else "degraded"
    return HealthResponse(
        status=status,
        llm_provider="ollama",
    )


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    """
    Ask a question to the company knowledge base.
    Returns an answer grounded in the documents + source references.
    Uses local Ollama model (no external API calls).
    """
    # 1. Validate input first (so tests get correct status codes)
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 2. Then check if the knowledge base is ready
    if qa_chain is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Knowledge base is not ready. "
                "Please build the index (python scripts/build_index.py) "
                "and ensure Ollama is running (ollama serve)."
            ),
        )

    try:
        result = qa_chain.invoke({"query": question})
        answer = result["result"]

        sources = []
        for doc in result.get("source_documents", []):
            content = doc.page_content
            if len(content) > 350:
                content = content[:350] + "..."
            sources.append(
                Source(
                    content=content,
                    source=doc.metadata.get("source", "unknown"),
                )
            )

        is_supported = "I don't have enough information" not in answer

        return QueryResponse(
            answer=answer,
            sources=sources,
            is_supported=is_supported,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}. Is Ollama running?",
        )


@app.get("/")
def root():
    return {
        "message": "Company Knowledge Assistant API (Powered by Ollama)",
        "docs": "/docs",
        "health": "/health",
        "ask": "POST /ask",
        "llm": "Ollama (local)",
    }