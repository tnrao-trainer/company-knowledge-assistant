FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by sentence-transformers / chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the vector index at image build time
RUN python scripts/build_index.py

EXPOSE 8000

# Note: Ollama must be running on the host or in another container.
# When running this container, pass:
#   -e OLLAMA_BASE_URL=http://host.docker.internal:11434
# (or the actual Ollama service URL)

RUN useradd --create-home appuser
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]