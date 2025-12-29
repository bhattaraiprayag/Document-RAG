FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (curl for health checks, ffmpeg for media processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (both punkt and punkt_tab for compatibility with different NLTK versions)
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Copy application
COPY app/ ./app/

# Create directories
RUN mkdir -p /app/uploads

# Expose port
EXPOSE 8000

# Run the backend
CMD ["python", "-m", "app.main"]
