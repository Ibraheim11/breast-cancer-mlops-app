# Breast Cancer Classification — Production ML Application

**LLMOps | Assignments 01, 02 & 03**  
**Stack:** Python 3.12 · scikit-learn · FastAPI · Pydantic · Streamlit · Docker · pytest · uv

> **Educational demonstration only.** This classifier is **not clinically validated**, does not provide a diagnosis, and must not be used to make medical decisions.

## Overview

This repository evolves the **same machine-learning project** across three assignments:

| Assignment | Deliverable |
| --- | --- |
| **01 — From Notebook to Production Python** | Reproducible training, evaluation, a saved scikit-learn pipeline, modular code, and CLI train/predict commands. |
| **02 — From Saved Model to a Validated ML API** | FastAPI `GET /health` and `POST /predict`, Pydantic validation, safe error responses, saved-model inference, and Docker support. |
| **03 — From Model API to a Production-Ready App** | Training-only feature selection, comparison with the 30-feature baseline, a **25-feature model**, Streamlit HTTP client, Dockerized API, and automated pytest coverage. |

**Final selected model:** `StandardScaler` → `LogisticRegression`, using **25 of the 30 original input measurements**. Selection uses `SelectKBest(f_classif)`; the number of features was chosen with **5-fold stratified cross-validation on training data only**. The fitted pipeline is saved and reused for inference: **no training occurs inside FastAPI or Streamlit**.

## Architecture

```text
Breast Cancer Wisconsin dataset (569 observations, 30 input features)
      |
      +--> stratified train/test split (80/20; random seed 42)
              |
              +--> training data: SelectKBest + 5-fold CV chooses k=25
                         |
                         +--> selected-feature training: StandardScaler + LogisticRegression
                                     |
                                     +--> evaluate on held-out test set
                                     +--> save fitted 25-feature pipeline and feature-name metadata

User --> Streamlit (editable 25-feature form; localhost:8501)
             |
             | HTTP POST /predict (JSON)
             v
         FastAPI + Pydantic (localhost:8006 with Docker)
             |
             v
         Cached, fitted 25-feature joblib pipeline
             |
             v
         JSON {"predicted_class": 0/1, "class_label": "malignant"/"benign"}
             |
             v
         Streamlit result
```

The Streamlit UI **does not import/load the fitted model** and **never retrains**. It only reads the selected-feature *names* and example input values from JSON files, sends requests to the FastAPI service, and displays responses.

## Dataset and labels

- Source: `sklearn.datasets.load_breast_cancer` (no external dataset download required).
- **569** observations; **30** numeric measurements; **2** classes; no missing values.
- Class **`0` = `malignant`**; class **`1` = `benign`**.
- Reproducible stratified **80% training / 20% test** split, random seed **42**.
- Preprocessing: `StandardScaler` fitted **only on training data**.
- Classifier: `LogisticRegression`; unchanged classifier family from Assignments 01 and 02.

## Feature-selection research and rationale

The assignment requires researching **all three families** but implementing only one:

| Family | How it works | Example | Strength | Limitation |
| --- | --- | --- | --- | --- |
| **Filter** | Ranks/removes features using statistics calculated from training data, independently of repeatedly fitting the final predictor. | ANOVA F-test via `SelectKBest(f_classif)` | Simple and computationally efficient. | Evaluates features individually and may retain correlated/redundant measurements. |
| **Wrapper** | Tests feature subsets by repeatedly training and evaluating a predictive estimator. | RFE, RFECV | Can account for the predictor's behavior. | Usually more computationally expensive and may overfit without careful validation. |
| **Embedded** | Selects/shrinks features during the estimator's fitting procedure. | L1-regularized logistic regression, `SelectFromModel` | Ties selection to model training. | Depends on the estimator/regularization and can be sensitive to settings. |

**Chosen method:** Filter → **ANOVA F-test / SelectKBest**, appropriate for these numeric features and binary labels. It is straightforward to reproduce and explain. ANOVA's univariate scores do not model interactions among measurements, which is a limitation.

### Leakage prevention and experiment design

1. Split the dataset into **training and held-out test** data **before** fitting a selector.
2. For `k` selection, place `SelectKBest` **inside** the scikit-learn `Pipeline` passed to `GridSearchCV`. Every cross-validation fold refits the selector using **that fold's training portion only**.
3. Compare candidate counts `k ∈ {8, 10, 12, 15, 20, 25}` using **5-fold stratified CV** and **weighted F1**.
4. Choose `k=25` based on training CV, refit on the full training split, and evaluate the final reduced-feature model on the held-out test split.
5. Record the final selected-feature names and **their exact order**. Train and save a final `StandardScaler` + `LogisticRegression` pipeline with *only* those 25 DataFrame columns. Inference must not guess, invent, zero-fill, or request the five removed features.

An initial exploratory **10-feature** model was also tested; its lower score motivated a training-only CV search. That 10-feature hold-out result is documented below for transparency; the selected `k` is the winner of the **training cross-validation**, not chosen by maximizing held-out test metrics. Since an earlier exploratory result was inspected, the held-out results should be interpreted as educational evidence rather than a pristine, never-inspected benchmark.

### Cross-validation results (training data only)

| Candidate k | Mean weighted F1 | Standard deviation |
| ---: | ---: | ---: |
| 8 | 0.9516 | 0.0191 |
| 10 | 0.9472 | 0.0162 |
| 12 | 0.9516 | 0.0132 |
| 15 | 0.9450 | 0.0155 |
| 20 | 0.9757 | 0.0083 |
| **25** | **0.9825** | **0.0131** |

The original **30-feature baseline** obtained **0.9780** mean weighted F1 under the training-only CV evaluation. The 25-feature candidate obtained **0.9825**. These are descriptive results on this dataset, not a guarantee of generalization.

### Held-out test results (114 observations)

| Metric | Original baseline: 30 features | Final selected: 25 features |
| --- | ---: | ---: |
| Accuracy | 0.9825 (98.25%) | 0.9737 (97.37%) |
| Weighted precision | 0.9825 | 0.9740 |
| Weighted recall | 0.9825 | 0.9737 |
| Weighted F1 | 0.9825 | 0.9737 |
| Input measurements | 30 | **25** |

**Reduced-feature result:** 5 fewer measurements (**16.7% reduction**) with a **0.88-percentage-point drop** in held-out accuracy relative to the original baseline. This is a trade-off, **not an improvement in held-out accuracy**.

Confusion matrices, with rows = true `[malignant, benign]`, columns = predicted `[malignant, benign]`:

```text
Baseline (30):        Final (25):
[[41  1]              [[41  1]
 [ 1 71]]              [ 2 70]]
```

The initial **10-feature exploratory model** scored accuracy **0.9474**, weighted precision **0.9485**, weighted recall **0.9474**, and weighted F1 **0.9476** on the held-out samples.

### Final selected input fields — **25**, in order

The external FastAPI JSON keys contain **spaces**, matching the dataset column names and Pydantic aliases:

| # | Feature | # | Feature |
| ---: | --- | ---: | --- |
| 1 | `mean radius` | 14 | `concavity error` |
| 2 | `mean texture` | 15 | `concave points error` |
| 3 | `mean perimeter` | 16 | `worst radius` |
| 4 | `mean area` | 17 | `worst texture` |
| 5 | `mean smoothness` | 18 | `worst perimeter` |
| 6 | `mean compactness` | 19 | `worst area` |
| 7 | `mean concavity` | 20 | `worst smoothness` |
| 8 | `mean concave points` | 21 | `worst compactness` |
| 9 | `mean symmetry` | 22 | `worst concavity` |
| 10 | `radius error` | 23 | `worst concave points` |
| 11 | `perimeter error` | 24 | `worst symmetry` |
| 12 | `area error` | 25 | `worst fractal dimension` |
| 13 | `compactness error` | | |

**Removed fields (not accepted by the final API):** `mean fractal dimension`, `texture error`, `smoothness error`, `symmetry error`, `fractal dimension error`.

The authoritative runtime sources are `models/final_selected_features.json` and `feature_names_in_` on the saved model. The exact schema in `production_app/schemas.py` must match **both names and order**.

## Project structure

```text
ml-production-project/
├── notebooks/01_exploration.ipynb             # Assignment 01
├── production_app/
│   ├── __init__.py
│   ├── config.py                               # Env configuration
│   ├── logging_config.py
│   ├── data.py                                 # Dataset and split
│   ├── preprocessing.py
│   ├── feature_selection.py                    # SelectKBest utility (Task 03)
│   ├── training.py                             # Fitted scaler/classifier pipeline
│   ├── evaluation.py                           # Metrics
│   ├── model_service.py                        # Save/load/predict
│   ├── main.py                                 # Selected-feature CLI workflows
│   ├── schemas.py                              # Exact 25-field Pydantic request
│   └── api.py                                  # GET /health, POST /predict
├── models/
│   ├── breast_cancer_pipeline.joblib           # Original 30-feature baseline
│   ├── breast_cancer_final_pipeline.joblib     # Final 25-feature fitted model
│   └── final_selected_features.json            # Selected input names/order
├── examples/sample_25_features.json           # Sample 0, real data values
├── tests/
│   ├── test_model.py
│   └── test_api.py
├── feature_selection_experiment.py             # Exploratory 10-feature comparison
├── feature_selection_cv.py                     # CV search and comparison
├── streamlit_app.py                            # HTTP client (no ML training/loading)
├── run.py                                      # Train/predict CLI
├── Dockerfile                                  # FastAPI only
├── .dockerignore
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

Some intermediate model artifacts produced by experiments (`breast_cancer_selected_pipeline.joblib`, `breast_cancer_cv_selected_pipeline.joblib` and the corresponding JSON files) are **not required** to run the final application and may be kept locally. The repository should contain the final artifact or a reliable documented path to reproduce it.

## Install and configure (Windows PowerShell)

Requirements: **Git**, **uv**, compatible **Python 3.12**; **Docker Desktop** for container testing. The project uses a Python version recorded in `.python-version`.

```powershell
git clone https://github.com/Ibraheim11/llmops-assignment-01-production-python.git
cd llmops-assignment-01-production-python
uv sync --locked
Copy-Item .env.example .env
```

Check `.env` and ensure these settings target the **final** model (example):

```dotenv
APP_ENV=development
LOG_LEVEL=INFO
MODEL_PATH=models/breast_cancer_final_pipeline.joblib
TEST_SIZE=0.20
RANDOM_SEED=42
```

`.env` is local configuration and must **not** be committed. The `MODEL_PATH` default and the `.env.example` file should likewise target the 25-feature artifact. A conflicting operating-system `MODEL_PATH` takes precedence; clear it in PowerShell with `Remove-Item Env:MODEL_PATH -ErrorAction SilentlyContinue` if necessary.

### Create/recreate the final fitted model (train once)

If `models/breast_cancer_final_pipeline.joblib` or its feature metadata is missing, prepare it **before** starting the API or building Docker:

```powershell
uv run python run.py train
```

Expected outputs:

- `models/breast_cancer_final_pipeline.joblib` (trained scaler + classifier, 25 columns).
- `models/final_selected_features.json` (25 names in model order).
- Console metrics evaluated on the test data.

The CLI uses `SelectKBest(f_classif)` with the final `k=25` established by the documented training-only CV experiment. **Feature count re-search is separate** (`uv run python feature_selection_cv.py`); the standard `train` command builds the final model without selecting `k` using test data.

Use the saved final artifact to predict **without training**:

```powershell
uv run python run.py predict --sample-index 0
```

For the documented final model, sample `0` returns class `0` (`malignant`). For new versions, inspect the actual output rather than assuming a class.

To reproduce the original-versus-selected comparison:

```powershell
uv run python feature_selection_experiment.py
uv run python feature_selection_cv.py
```

Those scripts create separately named candidate artifacts and do not overwrite the original `models/breast_cancer_pipeline.joblib` baseline. The original baseline remains as a reference from Assignments 01 and 02.

## FastAPI — local run

Start the backend **in one terminal**:

```powershell
uv run uvicorn production_app.api:app --host 127.0.0.1 --port 8005
```

- Swagger UI: <http://127.0.0.1:8005/docs>
- Health: <http://127.0.0.1:8005/health>
- Predict: `POST http://127.0.0.1:8005/predict`
- Stop: **Ctrl+C** in the server terminal.

### `GET /health`

`200 OK` when the configured saved pipeline is loadable:

```json
{"status": "healthy", "model_loaded": true}
```

`503 Service Unavailable` when the model is unavailable:

```json
{"status": "unhealthy", "model_loaded": false}
```

### `POST /predict` — complete valid **25-feature** JSON

The example uses **observation 0** from the Wisconsin dataset and can be copied into Swagger UI or saved as `examples/sample_25_features.json`:

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
  "radius error": 1.095,
  "perimeter error": 8.589,
  "area error": 153.4,
  "compactness error": 0.04904,
  "concavity error": 0.05373,
  "concave points error": 0.01587,
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

Expected example response (`200 OK`) for the documented final model:

```json
{"predicted_class": 0, "class_label": "malignant"}
```

PowerShell request (when the API is running on **8005**):

```powershell
Invoke-RestMethod `
    -Uri "http://127.0.0.1:8005/predict" `
    -Method POST `
    -ContentType "application/json" `
    -Body (Get-Content .\examples\sample_25_features.json -Raw)
```

The request must contain **exactly the 25 selected fields** as finite, non-negative numeric JSON values. Pydantic forbids unknown fields, and the backend enforces exact schema/model feature-order consistency. The five removed fields are **not accepted**.

| Scenario | HTTP code | Behavior |
| --- | ---: | --- |
| Valid 25-feature request | `200` | JSON predicted class + label. |
| Missing selected field | `422` | Validation error. |
| Extra/removed field | `422` | Validation error. |
| Invalid type / numeric string / negative value | `422` | Validation error. |
| Unavailable saved model | `503` | Safe error message, no internal traceback/path. |
| Unexpected inference failure | `500` | Controlled generic error. |

A model-unavailable prediction normally returns:

```json
{"detail": "Prediction service is temporarily unavailable."}
```

The saved model is cached inside the running backend process. **Restart FastAPI after replacing an artifact.** There is **no fitting/training in API handlers**.

## Streamlit — editable browser UI

Start FastAPI first, then **in a second terminal**:

```powershell
uv run streamlit run streamlit_app.py --server.port 8501
```

Open <http://localhost:8501>. The page displays all **25 selected features** as editable non-negative numeric fields, with demonstration values from `examples/sample_25_features.json`, plus a **Predict** button.

- The **FastAPI Base URL** is configurable in the sidebar; default is `http://127.0.0.1:8005` (local API).
- The same setting can be supplied as environment variable `API_BASE_URL` before starting Streamlit.
- Clicking **Predict** sends a JSON `POST` over HTTP and displays the model's class ID and label.
- The UI handles connection failure, request timeout, validation error (`422`), model unavailable (`503`), and other HTTP problems with user-friendly messages.
- **Streamlit runs locally; Docker packages FastAPI only.** The UI never trains, loads, or directly runs the estimator.

Stop Streamlit with **Ctrl+C** in its terminal.

## Docker — FastAPI backend only

**Prerequisite:** Docker Desktop/Linux containers running, with the final fitted artifact and metadata already present. The image copies those files; it does **not** train anything. Never copy `.env`, secrets, virtual environments, Git metadata, or notebooks into the runtime image.

From the repository root:

```powershell
# Prepare the artifact if it is not present yet
uv run python run.py train

# Build the Assignment 03 API image
docker build -t breast-cancer-api:task03 .

# Start container (host:8006 -> container:8000)
docker run --rm --name breast-cancer-api-task03 -p 8006:8000 breast-cancer-api:task03
```

In **another terminal**:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8006/health"
Invoke-RestMethod `
    -Uri "http://127.0.0.1:8006/predict" `
    -Method POST `
    -ContentType "application/json" `
    -Body (Get-Content .\examples\sample_25_features.json -Raw)
```

- Dockerized Swagger: <http://127.0.0.1:8006/docs>
- In the Streamlit sidebar change **FastAPI Base URL** to `http://127.0.0.1:8006`; submit the example input and inspect the returned prediction.
- Stop with **Ctrl+C** in the Docker terminal, or in another terminal: `docker stop breast-cancer-api-task03`.
- `--rm` removes the stopped **container**, not the built image.

Docker Compose, Kubernetes, cloud deployment, and CI/CD are outside the Assignment 03 required scope.

## Automated tests — pytest

The test suite uses **in-process FastAPI `TestClient`**, temporary files, and mocking. It does not require a running Uvicorn server, Streamlit process, or Docker container. Testing dependencies are installed via the `uv` development dependency group.

```powershell
uv sync --locked
uv run python -m pytest -v
uv run python -m pytest -q
```

The intended coverage comprises:

- Training workflow writes a model and selected-feature metadata to **temporary paths**, not production artifacts.
- Loading the final saved model and predicting **without fitting**.
- CLI prediction remains functional.
- The **25 aliases and order** in Pydantic exactly match the model.
- `GET /health` responds `200` when ready.
- `POST /predict` responds `200` with structured class/label for valid data.
- Missing, extra, malformed, and negative features produce `422`.
- A simulated missing model yields controlled `503` responses, without deleting the real file.

Do **not** report a successful test count until the suite has passed locally. If a test fails, inspect the traceback and fix the root cause before submission.

## Code quality and final verification

```powershell
uv run ruff check .
uv run black --check .
uv run pre-commit run --all-files
uv run python -m pytest -q
uv run python run.py predict --sample-index 0
```

Before submission also verify the **real backend path**, the schema/model agreement, and Git hygiene:

```powershell
uv run python -c "from production_app.config import settings; print(settings.model_path)"
uv run python -c "from production_app.schemas import PredictionRequest; from production_app.model_service import load_model; from production_app.config import settings; m=load_model(settings.model_path); s=[f.alias or n for n,f in PredictionRequest.model_fields.items()]; print(len(s),len(m.feature_names_in_), s==list(m.feature_names_in_)); assert s==list(m.feature_names_in_)"
git status
```

Expect **`25 25 True`** for the final schema/model check. Keep `.venv`, `.env`, caches, secrets, and any test-generated temporary files out of Git. Commit `pyproject.toml`, the resolved `uv.lock`, source, the selected model or reproducible training path, metadata, README, Docker files, examples, notebook, and tests.

### Fresh-clone verification

The reviewer should be able to: **clone → `uv sync --locked` → copy `.env.example` → train if needed → CLI predict → test → run API → run Streamlit → Docker build/run → call `/health` and `/predict`**. Execute this sequence in a separate folder before submitting on Google Classroom. If the `models/` artifact is not committed, perform `uv run python run.py train` first.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| API returns `503` | `MODEL_PATH` points to `models/breast_cancer_final_pipeline.joblib`; create artifact using CLI and restart the API. |
| Model loads but feature schema mismatch | Compare Pydantic aliases with `model.feature_names_in_` in exact order; check for stale 30-feature artifact. |
| API returns `422` | Supply exactly the final 25 numeric measurement keys, no unknown or removed fields. |
| Streamlit says cannot connect | Ensure backend is running; local API URL uses **8005**, Docker API URL uses **8006**. |
| Docker build cannot `COPY` model | Run training **before** `docker build`; do not exclude `models/` in `.dockerignore`. |
| Docker connection refused | Start Docker Desktop, verify `docker ps`, and use `-p 8006:8000`. |
| `uv` warns about wrong `VIRTUAL_ENV` | You may have activated a different project's environment. Clear the variable in PowerShell: `Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue`. |
| pytest import/dependency errors | Run `uv sync --locked`; check `pytest` and `httpx` in the dev dependency group. |
| Black shows notebook-skipping warning | This is not necessarily a failure; check exit status and, if needed, install Jupyter extras. |

## Submission and further work

Submit the **same** repository used for Assignments 01 and 02:

<https://github.com/Ibraheim11/llmops-assignment-01-production-python>

**Required for Assignment 03:** feature-selection comparison and leakage prevention, selected model with consistent API/UI inputs, functioning Streamlit → HTTP → FastAPI flow, working Dockerized backend, pytest, updated uv lockfile, and complete reproducibility instructions.

**After instructor acceptance (portfolio phase):** record a brief demo showing the live Streamlit UI and Dockerized API first, then a real prediction, then a short explanation of the code architecture. Optionally share the repository/demo on LinkedIn and credit instructors using their actual profiles.

> **Limitations:** This small dataset and its selected features are for educational model-serving practice; reported cross-validation/test scores are not clinical validation. The API intentionally provides class labels only, not probabilities or medical recommendations.
