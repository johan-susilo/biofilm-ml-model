# Biofilm Prediction API - Docker Image
# Multi-stage build for optimized production image

# Build stage - includes development tools and model training capabilities
FROM python:3.10-slim-bullseye as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Production stage - minimal runtime image
FROM python:3.10-slim-bullseye as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/venv/bin:$PATH" \
    BIOFILM_API_HOST=0.0.0.0 \
    BIOFILM_API_PORT=8000

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r biofilm && useradd -r -g biofilm biofilm

# Create working directory and set ownership
WORKDIR /app
RUN chown -R biofilm:biofilm /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=biofilm:biofilm . .

# Switch to non-root user
USER biofilm

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - use uvicorn with the package entrypoint
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Labels for metadata
LABEL maintainer="iGEM Team" \
      version="1.0.0" \
      description="Biofilm Prediction API with ML models" \
      org.opencontainers.image.source="https://github.com/igem-team/biofilm-prediction" \
      org.opencontainers.image.documentation="https://github.com/igem-team/biofilm-prediction/blob/main/README.md"
