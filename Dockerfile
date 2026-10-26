# Assignment 03 - Dockerized FastAPI Backend

# ----------------------------------------
# 1. Python Base Image
# ----------------------------------------

FROM python:3.12-slim


# ----------------------------------------
# 2. Install uv
# ----------------------------------------

RUN python -m pip install --no-cache-dir uv==0.11.31


# ----------------------------------------
# 3. Set Working Directory
# ----------------------------------------

WORKDIR /app


# ----------------------------------------
# 4. Environment Variables
# ----------------------------------------

ENV PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    MODEL_PATH=models/breast_cancer_final_pipeline.joblib \
    PATH="/app/.venv/bin:$PATH"


# ----------------------------------------
# 5. Copy Dependency Files
# ----------------------------------------

COPY pyproject.toml uv.lock .python-version ./


# ----------------------------------------
# 6. Install Dependencies
# ----------------------------------------

RUN uv sync \
    --locked \
    --no-dev \
    --no-install-project \
    --python /usr/local/bin/python


# ----------------------------------------
# 7. Copy FastAPI Application
# ----------------------------------------

COPY production_app/ ./production_app/


# ----------------------------------------
# 8. Copy Final Trained Model
# ----------------------------------------

COPY models/breast_cancer_final_pipeline.joblib \
     ./models/breast_cancer_final_pipeline.joblib


# ----------------------------------------
# 9. Copy Selected Feature Metadata
# ----------------------------------------

COPY models/final_selected_features.json \
     ./models/final_selected_features.json


# ----------------------------------------
# 10. Expose API Port
# ----------------------------------------

EXPOSE 8000


# ----------------------------------------
# 11. Start FastAPI
# ----------------------------------------

CMD ["uvicorn", "production_app.api:app", "--host", "0.0.0.0", "--port", "8000"]