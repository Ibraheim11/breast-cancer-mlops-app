# Breast Cancer Classification - Production Python Project

## Assignment 01: From Notebook to Production Python

This project was developed as part of the Production Python Foundations module.

The goal is to transform an experimental Machine Learning notebook into a clean, reproducible, production-style Python application.

The main focus is software engineering and project organization rather than maximizing model accuracy.

---

## Project Goal

Build a binary classification system using the Breast Cancer Wisconsin dataset.

The project starts with experimentation in a Jupyter Notebook and then refactors the validated workflow into reusable Python modules.

The final application supports:

- Dataset loading
- Train/test splitting
- Data preprocessing
- Model training
- Model evaluation
- Model serialization
- Saved-model loading
- Prediction without retraining
- Configuration management
- Logging
- Exception handling
- Code quality automation

---

## Dataset

The project uses the **Breast Cancer Wisconsin Dataset** provided by scikit-learn.

### Dataset Information

- Samples: 569
- Features: 30
- Classes: 2
- Missing values: 0

### Target Classes

- 0: Malignant
- 1: Benign

---

## Machine Learning Model

The model used is:

**Logistic Regression**

The preprocessing step is:

**StandardScaler**

They are combined using a scikit-learn Pipeline:

```text
StandardScaler
      ↓
LogisticRegression