# ── Build stage: Python 3.11 + Java 17 (required by PySpark 3.4+) ──────────
FROM python:3.11-slim

# Install Java (PySpark runtime dependency)
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Working directory inside the container
WORKDIR /app

# Install Python dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Default command: run the full pipeline
CMD ["python", "main.py"]
