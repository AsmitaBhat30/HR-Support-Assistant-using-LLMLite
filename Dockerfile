# 1. Choose a slim base image
FROM python:3.11-slim

# 2. Set environment variables to prevent Python from buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Create a non-root user for security
RUN useradd -m -u 1000 appuser

# 4. Set working directory
WORKDIR /app

# 5. Install system dependencies (needed for psycopg2 compilation and uv)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 6. Install uv globally
RUN pip install uv

# 7. Copy dependency files and install Python packages
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# 8. Copy application code
COPY src/ ./src/
COPY .env .

# 9. Switch to non-root user
USER appuser

# 10. Expose port
EXPOSE 8000

# 11. Run application
# host 0.0.0.0 is required for Docker to expose the port externally
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]