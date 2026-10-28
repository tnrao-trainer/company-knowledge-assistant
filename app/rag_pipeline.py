#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.config import (
    CHROMA_PATH,
    TOP_K,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)


def get_embeddings():
    """Return the embedding model (local HuggingFace model - free)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_llm():
    """Return the Ollama LLM (runs completely locally)."""
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )


def get_vectorstore():
    """Load the existing Chroma vector store."""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )


def format_docs(docs):
    """Helper to format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def get_qa_chain():
    """
    Build and return a modern LCEL-based RAG chain using Ollama.
    Returns a chain that accepts {"query": "..."} and returns
    a dict with 'result' and 'source_documents'.
    """
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful Company Knowledge Assistant.
Use ONLY the following pieces of context to answer the question.
If the answer cannot be found in the context, clearly say:
"I don't have enough information in the company knowledge base to answer this question."

Do not make up information. Always be concise and professional.

Context:
{context}

Question: {question}

Answer:"""
    )

    # Modern LCEL chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Wrapper to keep the same interface as the old RetrievalQA
    class QAChainWrapper:
        def invoke(self, inputs: dict):
            query = inputs.get("query") or inputs.get("question")
            # Get source documents
            source_documents = retriever.invoke(query)
            # Get the answer
            answer = rag_chain.invoke(query)
            return {
                "result": answer,
                "source_documents": source_documents,
            }

        def __call__(self, inputs: dict):
            return self.invoke(inputs)

    return QAChainWrapper()