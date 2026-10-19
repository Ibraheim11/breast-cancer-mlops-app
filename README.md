# Breast Cancer Classification — Production ML & FastAPI

**LLMOps | Assignment 01 + Assignment 02**  
**Assignment 01:** From Notebook to Production Python  
**Assignment 02:** From Saved Model to a Validated ML API

This repository demonstrates how a machine-learning experiment is turned into a reproducible Python application and then served through a **validated FastAPI backend**. Assignment 02 extends the **same project and saved scikit-learn pipeline** from Assignment 01; it does not replace the model or retrain it during API requests.

The API runs locally or inside a **Docker container**. Docker is an additional learning exercise; cloud deployment, Docker deployment, an automated API test suite, and a user interface are **not required** by Assignment 02.

> **Educational use only.** The breast-cancer classifier is a coursework demonstration, not a medical diagnostic service.

## Contents

1. [Project overview](#project-overview)
2. [Dataset and model](#dataset-and-model)
3. [Project structure](#project-structure)
4. [Quick start: local API](#quick-start-local-api)
5. [Assignment 01: training and CLI inference](#assignment-01-training-and-cli-inference)
6. [Assignment 01: evaluation results](#assignment-01-evaluation-results)
7. [Assignment 02: API design](#assignment-02-api-design)
8. [Required request features](#required-request-features)
9. [API requests and responses](#api-requests-and-responses)
10. [Validation and error handling](#validation-and-error-handling)
11. [Docker: build and run](#docker-build-and-run)
12. [Code quality](#code-quality)
13. [Reproducibility and security](#reproducibility-and-security)
14. [Troubleshooting](#troubleshooting)
15. [Final verification checklist](#final-verification-checklist)

---

## Project overview

### What Assignment 01 implemented

- Explored the **Breast Cancer Wisconsin Diagnostic** dataset in `notebooks/01_exploration.ipynb`.
- Split the data into training and test sets, preserving class proportions.
- Built a `scikit-learn` **Pipeline**: `StandardScaler` → `LogisticRegression`.
- Evaluated the classifier and saved the **fitted pipeline** with `joblib`.
- Refactored the experiment into reusable Python modules with configuration, logging, type hints, and error handling.
- Added a CLI that can **train** or **load a model and predict**.

### What Assignment 02 adds

- `production_app/schemas.py`: Pydantic request and response models for **one observation containing all 30 original features**.
- `production_app/api.py`: FastAPI application with **`GET /health`** and **`POST /predict`**.
- Validation of feature names, types, numeric constraints, and unexpected input fields.
- Controlled API failures (including **422** for invalid requests and **503** when the saved model is unavailable).
- Automatic interactive API documentation at **`/docs`**.
- An optional **Dockerfile** and **`.dockerignore`** to run the same saved-model API in a Linux container.

### Serving architecture

```text
HTTP/JSON client
       |
       v
FastAPI  (/health, /predict)
       |
       v
Pydantic validates the 30 input features
       |
       v
DataFrame in the trained feature order
       |
       v
Cached, previously fitted joblib pipeline
       |
       v
Prediction: class ID + readable label (JSON)
```

**The API does not run `train_model()`, `fit()`, or `fit_transform()` on incoming requests.** The saved model is loaded on first use and cached in the process for subsequent requests. The model file must exist and be compatible with the installed scikit-learn version.

---

## Dataset and model

The data comes from `sklearn.datasets.load_breast_cancer`; no external dataset download is necessary.

| Property | Value |
| --- | --- |
| Samples | 569 |
| Input features | 30 numeric measurements |
| Target classes | 2 |
| Missing values | 0 |
| Class `0` | `malignant` |
| Class `1` | `benign` |
| Training/test split | 80% / 20%, stratified |
| Random seed | 42 (default) |
| Preprocessing | `StandardScaler` |
| Estimator | `LogisticRegression(max_iter=1000)` |
| Serialized artifact | `models/breast_cancer_pipeline.joblib` |

The scaler is **fitted only on training data** as part of the scikit-learn Pipeline, reducing the risk of data leakage. The saved artifact includes both the fitted scaler and classifier, so inference applies the same transformations automatically.

---

## Project structure

```text
ml-production-project/
├── notebooks/
│   └── 01_exploration.ipynb       # Assignment 01 experimentation
├── production_app/
│   ├── __init__.py
│   ├── config.py                 # Runtime settings and model path
│   ├── logging_config.py        # Operational logging
│   ├── data.py                  # Dataset loading and splitting
│   ├── preprocessing.py         # StandardScaler
│   ├── training.py              # Pipeline creation and fitting
│   ├── evaluation.py            # Classification metrics
│   ├── model_service.py         # Save / load / predict
│   ├── main.py                  # Training and CLI prediction flows
│   ├── schemas.py               # Assignment 02: Pydantic schemas
│   └── api.py                   # Assignment 02: FastAPI endpoints
├── models/
│   └── breast_cancer_pipeline.joblib  # Existing fitted model
├── run.py                       # Assignment 01 CLI entry point
├── Dockerfile                   # Optional Docker packaging
├── .dockerignore
├── .env.example                 # Safe example configuration
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

The notebook is kept as an experiment record; the reusable application logic lives in `production_app/`. The CLI entry point `run.py` remains available after adding the API.

---

## Quick start: local API

### Prerequisites

- **Git**
- **uv** (Python package and environment manager)
- Python **3.12.7** (pinned in `.python-version`; `uv` can manage this version)

### 1. Clone the existing repository

```bash
git clone https://github.com/Ibraheim11/llmops-assignment-01-production-python.git
cd llmops-assignment-01-production-python
```

This is the **same repository** used for Assignment 01.

### 2. Install the dependencies

```bash
uv sync
```

`uv` reads `pyproject.toml` and `uv.lock` and creates or updates `.venv`. For a strict lockfile check, use `uv sync --locked`.

### 3. Configure the environment (optional for the documented defaults)

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

Expected example settings:

```dotenv
APP_ENV=development
LOG_LEVEL=INFO
MODEL_PATH=models/breast_cancer_pipeline.joblib
TEST_SIZE=0.20
RANDOM_SEED=42
```

The actual `.env` is local and **must not be committed**. The default `MODEL_PATH` points to the saved model inside this repository.

### 4. Ensure the saved model exists

The API **reuses the Assignment 01 fitted artifact**:

```text
models/breast_cancer_pipeline.joblib
```

If this artifact is present after cloning, **skip training**. If it is missing (for example, in a clone that excludes generated artifacts), create it once using the original CLI:

```bash
uv run python run.py train
```

Training is a **one-time artifact-preparation step**, not part of normal API startup or inference.

### 5. Start the FastAPI server

```bash
uv run uvicorn production_app.api:app --host 127.0.0.1 --port 8000
```

For local development with automatic code reload, you can add `--reload`:

```bash
uv run uvicorn production_app.api:app --reload
```

Keep the terminal open while testing. Stop the server with **Ctrl+C**.

### 6. Open the interactive documentation

- Swagger UI: <http://127.0.0.1:8000/docs>
- OpenAPI schema: <http://127.0.0.1:8000/openapi.json>
- Health endpoint: <http://127.0.0.1:8000/health>

In Swagger UI, expand an endpoint, choose **Try it out**, enter the request body where applicable, and select **Execute**.

---

## Assignment 01: training and CLI inference

Training, evaluation, and saving the fitted pipeline:

```bash
uv run python run.py train
```

CLI prediction with the saved pipeline:

```bash
uv run python run.py predict --sample-index 0
uv run python run.py predict --sample-index 100
```

The prediction command loads the artifact with `joblib.load()` and does **not** train the model again.

```text
Dataset -> train/test split -> StandardScaler + LogisticRegression
                                       |
                                  train and evaluate
                                       |
                                save fitted pipeline
                                       |
                               load model -> predict
```

## Assignment 01: evaluation results

Recorded results from a successful training run, evaluated on **114 test samples**:

| Metric | Result |
| --- | ---: |
| Accuracy | **0.9825** (98.25%) |
| Weighted precision | **0.9825** |
| Weighted recall | **0.9825** |
| Weighted F1-score | **0.9825** |

Confusion matrix (rows = actual `[malignant, benign]`; columns = predicted `[malignant, benign]`):

```text
[[41  1]
 [ 1 71]]
```

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| Malignant | 0.98 | 0.98 | 0.98 | 42 |
| Benign | 0.99 | 0.99 | 0.99 | 72 |

Two of the 114 test observations were misclassified. The displayed metrics are from **one recorded run**; to reproduce them using the current configuration, run `uv run python run.py train`. The emphasis of this assignment is correct engineering, validation, and reproducibility rather than hyperparameter optimization.

---

## Assignment 02: API design

### `GET /health`

Checks whether the API can load its configured saved model. This is **model readiness**, not just confirmation that the web server is listening.

| Case | HTTP status | Meaning |
| --- | --- | --- |
| Model available and compatible | `200 OK` | Service ready |
| Model unavailable or incompatible | `503 Service Unavailable` | Service cannot serve predictions |

Healthy response (`200`):

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

Unhealthy response (`503`):

```json
{
  "status": "unhealthy",
  "model_loaded": false
}
```

Check in a browser at <http://127.0.0.1:8000/health> or run:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get
```

### `POST /predict`

Accepts **one observation as a JSON object** with all 30 required features, validates it with Pydantic, builds a DataFrame in the saved pipeline's expected column order, and returns a class ID and readable label.

Example successful response (`200 OK`):

```json
{
  "predicted_class": 0,
  "class_label": "malignant"
}
```

Prediction labels follow the original scikit-learn classes: **`0 = malignant`** and **`1 = benign`**. This endpoint does not return probabilities; a prediction score is optional in Assignment 02 and is **not exposed by this implementation**.

The model is loaded once per running API process on first use and then cached. Restart the server after replacing the artifact so the process can load the new file.

---

## Required request features

`PredictionRequest` in `production_app/schemas.py` defines **30 required numeric features** with the same names used by the saved pipeline. External JSON keys contain spaces; the Python schema uses aliases for those names.

**Every feature is required.** Values must be JSON numbers, finite, and non-negative. Numeric strings, missing features, unknown extra keys, negative values, `NaN`, and infinity are rejected where applicable. Names and order are checked against the fitted model when it is loaded; the API reconstructs the input columns in model order.

| Group | Required JSON keys (in trained order) |
| --- | --- |
| Mean measurements (10) | `mean radius`, `mean texture`, `mean perimeter`, `mean area`, `mean smoothness`, `mean compactness`, `mean concavity`, `mean concave points`, `mean symmetry`, `mean fractal dimension` |
| Error measurements (10) | `radius error`, `texture error`, `perimeter error`, `area error`, `smoothness error`, `compactness error`, `concavity error`, `concave points error`, `symmetry error`, `fractal dimension error` |
| Worst measurements (10) | `worst radius`, `worst texture`, `worst perimeter`, `worst area`, `worst smoothness`, `worst compactness`, `worst concavity`, `worst concave points`, `worst symmetry`, `worst fractal dimension` |

### Full valid JSON example (all 30 features)

The following example uses **observation 0** of `sklearn.datasets.load_breast_cancer`. Paste it into Swagger UI's `POST /predict` request body.

```json
{
  "mean radius": 17.99,
  "mean texture": 10.38,
  "mean perimeter": 122.8,
  "mean area": 1001.0,
  "mean smoothness": 0.1184,
  "mean compactness": 0.2776,
  "mean concavity": 0.3001,
  "mean concave points": 0.1471,
  "mean symmetry": 0.2419,
  "mean fractal dimension": 0.07871,
  "radius error": 1.095,
  "texture error": 0.9053,
  "perimeter error": 8.589,
  "area error": 153.4,
  "smoothness error": 0.006399,
  "compactness error": 0.04904,
  "concavity error": 0.05373,
  "concave points error": 0.01587,
  "symmetry error": 0.03003,
  "fractal dimension error": 0.006193,
  "worst radius": 25.38,
  "worst texture": 17.33,
  "worst perimeter": 184.6,
  "worst area": 2019.0,
  "worst smoothness": 0.1622,
  "worst compactness": 0.6656,
  "worst concavity": 0.7119,
  "worst concave points": 0.2654,
  "worst symmetry": 0.4601,
  "worst fractal dimension": 0.1189
}
```

Expected response for this saved model (`200 OK`):

```json
{
  "predicted_class": 0,
  "class_label": "malignant"
}
```

**Send a request with Swagger:** open `/docs` → `POST /predict` → **Try it out** → paste the complete JSON → **Execute**.

**Send a request from a terminal:** save the JSON above in a local file named `sample_request.json` and run:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" --data-binary @sample_request.json
```

In PowerShell, use `curl.exe` instead of the `curl` alias if necessary:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" --data-binary "@sample_request.json"
```

The `sample_request.json` file is only needed for this command-line example; you can paste the JSON directly into Swagger without creating a file.

---

## Validation and error handling

FastAPI applies the `PredictionRequest` schema **before** inference. These are the expected HTTP responses:

| Test | Example | HTTP status |
| --- | --- | --- |
| Valid request | All 30 numeric features | `200` |
| Missing feature | Omit `mean radius` | `422` |
| Incorrect type | Set `mean radius` to `"hello"` | `422` |
| Numeric string (strict input) | Set `mean radius` to `"17.99"` | `422` |
| Negative measurement | Set `mean radius` to `-10` | `422` |
| Unexpected field | Add `"random_feature": 100` | `422` |
| Model missing or cannot load | Start API with an invalid model path | `503` |
| Other inference failure | Internal prediction error | Controlled `500` |

For validation errors, FastAPI returns a `422` JSON body describing which field failed and why. For example, a missing field can return a body with this structure (the exact details may vary with library versions):

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "mean radius"],
      "msg": "Field required"
    }
  ]
}
```

Model-unavailable prediction response (`503`):

```json
{
  "detail": "Prediction service is temporarily unavailable."
}
```

An unexpected inference failure produces a generic error response instead of disclosing a Python stack trace, local path, or secret to the client. Details needed for debugging belong in **server-side logs**.

### Manually test model unavailability (no file deletion)

1. Stop any previous server on port `8001`.
2. In a **new PowerShell terminal**, from the project root, set a temporary invalid model path and start a second server:

   ```powershell
   $env:MODEL_PATH = "models/nonexistent_model_for_test.joblib"
   uv run uvicorn production_app.api:app --host 127.0.0.1 --port 8001
   ```

3. Open <http://127.0.0.1:8001/docs>. `GET /health` should return **503** with `"model_loaded": false`. `POST /predict` with the **valid 30-feature JSON** should also return **503**.
4. Stop that server with **Ctrl+C**, then clear the temporary variable in that terminal:

   ```powershell
   Remove-Item Env:MODEL_PATH
   ```

The separate port/process is important because the normal API process **caches** its loaded model; changing the model path without restarting a process does not replace the cached pipeline.

Assignment 02 requires **manual endpoint checks**. No new automated API test suite, exported Postman collection, or screenshots are necessary for submission.

---

## Docker: build and run

Docker packaging is an **optional extension** to this assignment. It runs the FastAPI application and **existing fitted pipeline** in an isolated Linux container.

### Prerequisites

- Docker Desktop or an equivalent Docker Engine running **Linux containers**.
- The fitted model artifact at `models/breast_cancer_pipeline.joblib`.

Verify Docker:

```bash
docker --version
docker info
docker run --rm hello-world
```

### Build the image

From the project root:

```bash
docker build -t breast-cancer-api:task02 .
```

The Dockerfile uses `python:3.12-slim`, installs **uv**, synchronizes production dependencies from `uv.lock`, copies `production_app/` and the **previously trained** `breast_cancer_pipeline.joblib`, and starts Uvicorn. It **does not retrain the model** while building or running the container.

The `.dockerignore` excludes local environments, `.env` files, Git history, notebooks, caches, and other unnecessary development files. **Do not include secrets** in an image.

> **Image-build note:** The uv installation in this project uses `python -m pip install --no-cache-dir uv==0.11.31` because pulling `uv` from GitHub Container Registry previously failed in this environment. Both approaches are valid; the current Dockerfile uses the `pip` approach.

### Run the container

```bash
docker run --rm --name breast-cancer-api-task02 -p 8002:8000 breast-cancer-api:task02
```

- `8000`: Uvicorn port **inside** the container.
- `8002`: port on the **host machine**, avoiding conflict with a local API on `8000`.
- `--rm`: automatically removes the container when it stops.

Keep the terminal open. Open:

- Docker Swagger UI: <http://127.0.0.1:8002/docs>
- Docker health check: <http://127.0.0.1:8002/health>

The Docker endpoints use the **same API schemas, 30-feature JSON body, and status codes** as the local API. To use the terminal example, replace port `8000` with `8002`.

### Verify container behavior

1. `GET /health` → **200** with `"model_loaded": true`.
2. `POST /predict` with the complete valid JSON above → **200** with class `0`, label `"malignant"`.
3. `POST /predict` with a negative `"mean radius"` or missing feature → **422**.
4. Confirm in server logs that the pipeline was **loaded**, not trained.

Check the running container from another terminal:

```bash
docker ps
```

Stop it with **Ctrl+C** in the attached terminal or from another terminal:

```bash
docker stop breast-cancer-api-task02
```

With `--rm`, it is removed after stopping. The **Docker image** remains available for later runs.

### Model and configuration inside Docker

The container's configured model path resolves to `/app/models/breast_cancer_pipeline.joblib`. No local `.env` is copied into the image, and the default environment values are sufficient for the provided artifact. If you need different non-sensitive runtime settings, pass them explicitly with Docker environment variables (for example, `-e LOG_LEVEL=DEBUG`). Avoid baking credentials into images.

---

## Code quality

The project reuses Assignment 01's Ruff, Black, and pre-commit setup.

Run from the **repository root**:

```bash
uv run ruff check .
uv run black --check .
uv run pre-commit run --all-files
```

All configured checks should **pass before submitting**. Ruff may fix import order with `uv run ruff check . --fix`, and Black can format Python files with `uv run black .`; rerun the checks afterward. Some Black installations skip `.ipynb` files unless the optional Jupyter support is installed; successful Python source checks are still meaningful.

Manual API checks are sufficient for Assignment 02; no new automated test suite is required.

---

## Reproducibility and security

- `.python-version` records the intended Python version (`3.12.7` for local development).
- `pyproject.toml` declares the dependencies and tool configuration.
- `uv.lock` makes dependency resolution reproducible. Commit it together with `pyproject.toml`.
- `.env.example` documents safe configuration defaults; `.env` stays local.
- `models/breast_cancer_pipeline.joblib` is a **trusted, previously fitted artifact**. Never load an untrusted joblib/pickle file.
- The prediction endpoint does **not** call any training function.
- API validation happens before inference; errors returned to clients avoid internal filesystem paths, tracebacks, and secrets.
- Do not commit `.venv/`, `.env`, credentials, caches, or unnecessary generated artifacts.

### Reproduce from a fresh clone

```bash
git clone https://github.com/Ibraheim11/llmops-assignment-01-production-python.git
cd llmops-assignment-01-production-python
uv sync
```

If the saved model is already in `models/`, start the API directly:

```bash
uv run uvicorn production_app.api:app --host 127.0.0.1 --port 8000
```

If the artifact is **missing**, create it **once** with:

```bash
uv run python run.py train
```

Then start the API and test `/health`, `/predict` and `/docs`. The Docker route is optional and follows the steps in [Docker: build and run](#docker-build-and-run).

---

## Troubleshooting

| Problem | Likely cause | Action |
| --- | --- | --- |
| Browser says `ERR_CONNECTION_REFUSED` at `/docs` | Server not running or wrong port | Start Uvicorn; keep terminal open; use port `8000` locally or `8002` for Docker. |
| `ModuleNotFoundError` | Environment not synchronized or wrong working directory | Run `uv sync` in the repository root and use `uv run ...`. |
| `/health` returns `503` | Model artifact missing, invalid, or incompatible | Check `MODEL_PATH` and artifact; prepare a trusted fitted artifact with the original training command if needed, then restart the server. |
| `/predict` returns `422` | Missing/extra field, wrong type, or invalid numeric value | Compare the request to the 30-feature example; use numbers, not numeric strings. |
| `/predict` returns `500` | Unexpected inference failure | Review server logs for diagnostics; do not expose traceback information to clients. |
| API still uses an old model after replacing the file | Model cached per process | Restart Uvicorn or the Docker container. |
| `docker: command not found` | Docker not installed or not on `PATH` | Install/start Docker Desktop and reopen the terminal. |
| `Cannot connect to Docker daemon` | Docker Engine stopped | Start Docker Desktop and verify `docker info`. |
| Docker build fails while pulling `ghcr.io` assets | Registry/DNS/network access problem | Use the `pip install uv==0.11.31` Dockerfile variant documented above and retry the build. |
| `Unable to find image breast-cancer-api:task02 locally` | Build did not succeed | Complete `docker build -t breast-cancer-api:task02 .` before `docker run`. |
| Port `8000` is already in use | Another server is listening | Stop the other server or run Docker with `-p 8002:8000`. |
| `black --check` reports missing `SRC` | No source path provided | Run `uv run black --check .` including the trailing dot. |

---

## Final verification checklist

Before submitting the **updated existing GitHub repository**:

- [ ] Assignment 01 notebook, CLI, configuration, and model logic are intact.
- [ ] A trusted saved model is present or can be created once using the original CLI.
- [ ] `GET /health` returns **200** with the model available and **503** when unavailable.
- [ ] `POST /predict` returns **200** for the complete 30-feature request.
- [ ] Pydantic rejects missing features, invalid types/values, and extra keys with **422**.
- [ ] Inference loads the saved pipeline **without retraining**.
- [ ] Swagger UI at `/docs` works; Docker checks work if Docker is included.
- [ ] `uv run ruff check .` passes.
- [ ] `uv run black --check .` passes.
- [ ] `uv run pre-commit run --all-files` passes.
- [ ] `.env`, `.venv/`, secrets, and other unnecessary files are not tracked by Git.
- [ ] This entire README renders correctly on **GitHub**, including the final section and closed code fences.
- [ ] All Assignment 02 changes are committed and pushed to the **same** repository.

**Submission:** [LLMOps Production Python Repository](https://github.com/Ibraheim11/llmops-assignment-01-production-python) — submit the updated repository link through Google Classroom.

### Technologies

Python 3.12 · scikit-learn · pandas · NumPy · joblib · python-dotenv · uv · FastAPI · Pydantic · Uvicorn · Docker · Ruff · Black · pre-commit · Git/GitHub
