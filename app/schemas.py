from pydantic import BaseModel, Field
from typing import List


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask the knowledge base")


class Source(BaseModel):
    content: str
    source: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    is_supported: bool


class HealthResponse(BaseModel):
    status: str
    message: str = "Company Knowledge Assistant is running"
    llm_provider: str = "ollama"