# FactoryGuard AI

**IoT Predictive Maintenance Engine for Industrial Robotic Arms**

FactoryGuard AI predicts catastrophic machine failures up to **24 hours in advance** using time-series sensor telemetry and imbalance-aware machine learning.  
It is designed to help manufacturing teams shift from reactive maintenance to planned interventions.

---

## Executive Summary

- Problem: Rare failures in large robotic fleets cause high unplanned downtime costs.
- Approach: Engineer temporal features from live sensor streams and train high-precision classifiers.
- Outcome: Real-time failure-risk scoring API + operator dashboard for maintenance decisions.
- Deployment target: Low-latency inference (`< 50ms` p95 objective).

---

## Business Use Case

A critical plant floor operates ~500 robotic arms with:

- `vibration`
- `temperature`
- `pressure`

FactoryGuard AI estimates failure probability in the next 24 hours so the maintenance team can:

- Prioritize at-risk assets.
- Schedule preventive maintenance.
- Reduce false alarms through precision-oriented thresholding.
- Minimize unscheduled stoppages and revenue loss.

---

## Core Features

### 1) Time-Series Feature Engineering

- Rolling Mean (`1h`, `6h`, `12h`)
- Exponential Moving Average (`1h`, `6h`, `12h`)
- Rolling Standard Deviation (`1h`, `6h`, `12h`)
- Lag Features (`t-1`, `t-2`)

### 2) Modeling Strategy

- Baseline Models:
  - Logistic Regression
  - Random Forest
- Production Models:
  - XGBoost (default)
  - LightGBM
- Hyperparameter tuning with Optuna

### 3) Imbalance Handling

- Primary metric: **PR-AUC** (not accuracy)
- Class weighting via `scale_pos_weight`
- Optional SMOTE path for comparison

### 4) Explainability

- SHAP-based local feature attribution
- JSON output for top contributing factors

### 5) Deployment

- Flask API for real-time scoring
- `joblib` model + metadata bundling
- Latency benchmark utility
- Dashboard UI for non-technical users

---

## Architecture Overview

```text
Raw Sensor Data (per arm, timestamped)
           |
           v
Feature Engineering Pipeline
(rolling stats + EMA + lag features)
           |
           v
Model Training & Tuning
(baseline + XGBoost/LightGBM + PR-AUC optimization)
           |
           v
Serialized Bundle
(model + feature config + metadata + threshold)
           |
           v
Flask Inference API (/predict)
           |
           v
Operator Dashboard + Maintenance Decision
```

---

## Repository Structure

```text
.
|-- api/
|   |-- __init__.py
|   |-- app.py
|-- scripts/
|   |-- latency_check.py
|-- src/
|   |-- __init__.py
|   |-- explainability.py
|   |-- feature_engineering.py
|   |-- model_trainer.py
|   |-- train_pipeline.py
|-- requirements.txt
|-- README.md
```

---

## Tech Stack

- Python 3.11+
- Pandas, NumPy
- scikit-learn
- imbalanced-learn
- XGBoost
- LightGBM
- Optuna
- SHAP
- Flask
- joblib

---

## Quick Start

### 1) Create environment and install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Train the model

If `data/sensor_timeseries.csv` is unavailable, synthetic data is generated automatically.

```bash
python -m src.train_pipeline --model-family xgboost --n-trials 5
```

Optional alternatives:

```bash
python -m src.train_pipeline --model-family lightgbm --n-trials 10
python -m src.train_pipeline --model-family xgboost --use-smote --n-trials 10
```

Generated artifacts:

- `models/factoryguard_model.joblib`
- `reports/training_metrics.json`

### 3) Start API + UI

```bash
python -m api.app
```

Open in browser:

- `http://127.0.0.1:8000/` (dashboard)
- `http://127.0.0.1:8000/health` (health endpoint)

---

## API Reference

### `GET /health`

Service health check.

### `POST /predict`

Predicts 24h failure probability from recent rows.

Request body:

```json
{
  "rows": [
    {
      "arm_id": 101,
      "timestamp": "2026-02-27T08:00:00Z",
      "vibration": 0.61,
      "temperature": 74.1,
      "pressure": 29.0
    },
    {
      "arm_id": 101,
      "timestamp": "2026-02-27T09:00:00Z",
      "vibration": 0.66,
      "temperature": 76.8,
      "pressure": 29.6
    }
  ]
}
```

Response body:

```json
{
  "product": "FactoryGuard AI",
  "failure_probability_24h": 0.43,
  "predicted_failure_24h": 1,
  "decision_threshold": 0.11,
  "latency_ms": 8.7
}
```

---

## Explainability Workflow

Generate SHAP explanation artifact:

```bash
python -m src.explainability --model-path models/factoryguard_model.joblib --input-csv data/sensor_timeseries.csv
```

Output:

- `reports/shap_local_explanation.json`

---

## Performance & Validation

Latency benchmark:

```bash
python scripts/latency_check.py
```

Outputs include:

- mean latency
- p95 latency
- max latency
- pass/fail against `< 50ms` p95 target

Model training outputs include:

- baseline PR-AUC
- best tuned model PR-AUC
- optimized threshold
- training metadata

---

## Production Hardening Checklist

- Replace Flask dev server with Gunicorn/Uvicorn behind reverse proxy.
- Add auth/rate limiting for API endpoints.
- Add model versioning and rollback strategy.
- Add CI pipeline for:
  - training metric regression checks
  - API contract tests
  - latency guardrails
- Integrate structured logging + monitoring dashboards.

---

## License

Internal / project-specific.  
If publishing publicly, add a proper OSS license (MIT/Apache-2.0/etc.).

