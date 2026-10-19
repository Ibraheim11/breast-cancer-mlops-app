# ==========================================
# LLMOps Assignment 02 - FastAPI Dockerfile
# ==========================================

# 1. Use Python 3.12
FROM python:3.12-slim

# 2. Install uv using pip
RUN python -m pip install --no-cache-dir uv==0.11.31

# 3. Set working directory
WORKDIR /app

# 4. Environment configuration
ENV PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# 5. Copy dependency files
COPY pyproject.toml uv.lock .python-version ./

# 6. Install production dependencies
RUN uv sync --locked --no-dev --no-install-project --python /usr/local/bin/python

# 7. Copy FastAPI application
COPY production_app/ ./production_app/

# 8. Copy the previously trained model
COPY models/breast_cancer_pipeline.joblib ./models/breast_cancer_pipeline.joblib

# 9. Expose FastAPI port
EXPOSE 8000

# 10. Start FastAPI server
CMD ["uvicorn", "production_app.api:app", "--host", "0.0.0.0", "--port", "8000"]