
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create a non-root user and grant ownership of application files
RUN useradd --create-home appuser \
    && mkdir -p /app/chroma_db \
    && chown -R appuser:appuser /app

# Run the application and build the index as the same user
USER appuser

# Build ChromaDB index
RUN python scripts/build_index.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
