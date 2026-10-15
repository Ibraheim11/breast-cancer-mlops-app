Breast Cancer Classification - Production Python Project

Assignment 01: From Notebook to Production Python

This project was developed as part of the Production Python Foundations module.

The goal of this assignment is to transform an experimental Machine Learning notebook into a clean, reproducible, production-style Python application.

The main focus is not achieving the highest possible model accuracy, but engineering the project in a structured, maintainable, and reproducible way.

Project Goal

The project builds a binary classification system using the Breast Cancer Wisconsin Dataset.

The workflow starts with experimentation inside a Jupyter Notebook and then moves the validated reusable logic into separate Python modules.

The final application supports:

Dataset loading

Dataset inspection

Train/test splitting

Data preprocessing

Model training

Model evaluation

Model serialization

Loading a saved model

Prediction without retraining

Configuration management

Environment variables

Logging

Exception handling

Type hints

Code quality automation

Dataset

The project uses the Breast Cancer Wisconsin Dataset available directly from scikit-learn.

Dataset Information

Samples: 569

Features: 30

Classes: 2

Missing values: 0

Target Classes

0 → Malignant

1 → Benign

Machine Learning Model

The selected model is:

Logistic Regression

The preprocessing step is:

StandardScaler

Both are combined using a scikit-learn Pipeline.

Input Features
      ↓
StandardScaler
      ↓
LogisticRegression
      ↓
Prediction

Using a Pipeline keeps preprocessing and prediction consistent and helps prevent data leakage because the scaler is fitted using the training data only.

Machine Learning Workflow

Load Dataset
    ↓
Inspect Dataset
    ↓
Separate Features and Target
    ↓
Train/Test Split
    ↓
Preprocessing
    ↓
Train Logistic Regression
    ↓
Evaluate Model
    ↓
Save Fitted Pipeline
    ↓
Reload Saved Pipeline
    ↓
Prediction

Project Structure

ml-production-project/
│
├── notebooks/
│   └── 01_exploration.ipynb
│
├── production_app/
│   ├── __init__.py
│   ├── config.py
│   ├── logging_config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── training.py
│   ├── evaluation.py
│   ├── model_service.py
│   └── main.py
│
├── models/
│   └── breast_cancer_pipeline.joblib
│
├── run.py
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md

Module Responsibilities

data.py

Responsible for:

Loading the Breast Cancer dataset

Separating features and target

Creating the train/test split

preprocessing.py

Defines the reusable preprocessing step:

StandardScaler

training.py

Responsible for:

Creating the Logistic Regression model

Combining preprocessing and model inside a Pipeline

Fitting the Pipeline using training data

evaluation.py

Calculates classification metrics including:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Classification Report

model_service.py

Responsible for:

Saving the fitted Pipeline using joblib

Loading the saved Pipeline

Making predictions using the saved model without retraining

config.py

Loads application settings such as:

Model path

Application environment

Logging level

Test size

Random seed

logging_config.py

Configures application logging and provides reusable loggers.

main.py

Orchestrates the training and prediction workflows.

run.py

Provides the command-line entry point for running the application.

Setup and Installation

1. Clone the Repository

git clone https://github.com/Ibraheim11/llmops-assignment-01-production-python.git

Enter the project directory:

cd llmops-assignment-01-production-python

2. Install Dependencies

This project uses uv for Python environment and dependency management.

Synchronize the project environment using:

uv sync

This installs the dependencies required by the project using:

pyproject.toml

uv.lock

3. Configure Environment Variables

The project contains a safe example configuration file:

.env.example

On Windows PowerShell, create the local .env file using:

Copy-Item .env.example .env

The default configuration is:

APP_ENV=development
LOG_LEVEL=INFO
MODEL_PATH=models/breast_cancer_pipeline.joblib
TEST_SIZE=0.20
RANDOM_SEED=42

The .env file is ignored by Git and should not be committed.

Training

To train the model, evaluate it, and save the fitted Pipeline, run:

uv run python run.py train

The training workflow performs:

Load Dataset
    ↓
Train/Test Split
    ↓
Create Pipeline
    ↓
Fit Preprocessing + Model
    ↓
Evaluate Model
    ↓
Save Fitted Pipeline

The trained Pipeline is saved to:

models/breast_cancer_pipeline.joblib

Prediction

After the model has been trained and saved, predictions can be made using:

uv run python run.py predict --sample-index 0

Another example:

uv run python run.py predict --sample-index 100

The prediction workflow performs:

Load Saved Pipeline
      ↓
Select Input Sample
      ↓
Predict
      ↓
Return Result

Important

The training command only needs to be run once to create the saved model artifact.

After the model has been saved, the prediction command can be executed multiple times without retraining the model.

Train Once
    ↓
Save Pipeline
    ↓
Load Pipeline
    ↓
Predict Many Times

The prediction workflow uses joblib.load() to load the existing fitted Pipeline.

No fit() operation occurs during prediction.

Evaluation Results

Results from one training run:

Accuracy : 0.9825
Precision: 0.9825
Recall   : 0.9825
F1 Score : 0.9825

Confusion Matrix:

[[41  1]
 [ 1 71]]

Classification summary:

Class

Precision

Recall

F1-score

Support

Malignant

0.98

0.98

0.98

42

Benign

0.99

0.99

0.99

72

The model achieved approximately 98.25% accuracy on the test set.

These results are generated by running:

uv run python run.py train

Code Quality

The project uses three code-quality tools:

Ruff

Black

pre-commit

Ruff

Used for linting and detecting code-quality issues.

Run:

uv run ruff check .

Expected result:

All checks passed!

Black

Used for consistent Python code formatting.

Run:

uv run black --check .

pre-commit

Runs the configured quality checks automatically.

Run all configured hooks manually using:

uv run pre-commit run --all-files

All hooks should pass before submission.

Notebook Experimentation

The initial experimentation is preserved in:

notebooks/01_exploration.ipynb

The notebook includes:

Dataset loading

Dataset dimensions

Feature names

Class labels

Missing-value inspection

Class distribution

Train/test split

Baseline Pipeline

Model training

Evaluation

Sample prediction

Saving the fitted Pipeline

Reloading the Pipeline

Prediction without retraining

The final Python application runs independently from the notebook.

Reproducibility

The project contains the following files to make the environment reproducible:

.python-version
pyproject.toml
uv.lock
.env.example

A new developer can reproduce the project using:

uv sync

Then train and save the model:

uv run python run.py train

Then make predictions using the saved model:

uv run python run.py predict --sample-index 0

Quick Start

For a new user who just cloned the repository:

uv sync
uv run python run.py train
uv run python run.py predict --sample-index 0

The training command creates the saved model artifact.

After that, the prediction command can be run whenever needed without retraining.

Final Workflow

Notebook Experimentation
        ↓
Validation
        ↓
Refactor into Python Modules
        ↓
Configuration
        ↓
Logging
        ↓
Training
        ↓
Evaluation
        ↓
Save Pipeline
        ↓
Load Pipeline
        ↓
Prediction Without Retraining
        ↓
Code Quality Checks
        ↓
Git / GitHub

Technologies Used

Python 3.12

scikit-learn

pandas

NumPy

joblib

python-dotenv

uv

Ruff

Black

pre-commit

Git

GitHub