FROM python:3.10-slim

# Install system dependencies for Playwright and general requirements
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    # Dependencies for Playwright
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libatspi2.0-0 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libxcb1 \
    libxkbcommon0 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create data directories that will be mounted
RUN mkdir -p /app/data/jobs /app/data/resumes /app/data/lancedb \
    /app/data/uploads /app/data/pipeline_output /app/data/.cache

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browser without system deps (we installed them manually)
RUN python -m playwright install chromium

# Copy application code
COPY . .

# Expose Flask port
EXPOSE 5000

CMD ["python", "app.py"]
