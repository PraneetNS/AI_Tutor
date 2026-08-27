FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY ai_tutor/ ./ai_tutor/
COPY scripts/ ./scripts/

EXPOSE 8000

CMD ["uvicorn", "ai_tutor.api:app", "--host", "0.0.0.0", "--port", "8000"]
