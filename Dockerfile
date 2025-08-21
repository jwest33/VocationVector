FROM python:3.10-slim

# Install system dependencies for Playwright and general requirements
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    # Core dependencies for Playwright/Chromium
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
    # Additional dependencies often needed for headless Chrome
    libglib2.0-0 \
    libgtk-3-0 \
    libxshmfence1 \
    libglu1-mesa \
    # Fonts for headless browser
    fonts-liberation \
    fonts-noto-color-emoji \
    fonts-noto-cjk \
    fonts-liberation2 \
    fonts-ipafont-gothic \
    fonts-wqy-zenhei \
    fonts-thai-tlwg \
    fonts-freefont-ttf \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create data directories that will be mounted
RUN mkdir -p /app/data/jobs /app/data/resumes /app/data/lancedb \
    /app/data/uploads /app/data/pipeline_output /app/data/.cache

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright and browsers
# Note: install-deps may fail on some systems, but we've already installed the deps manually
RUN pip install playwright && \
    python -m playwright install chromium

# Copy application code
COPY . .

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Expose Flask port
EXPOSE 5000

# Start app directly - docker-compose depends_on and healthcheck will handle ordering
CMD ["python", "app.py"]
