# FactoryGuard AI

IoT Predictive Maintenance Engine for manufacturing plants.

FactoryGuard AI predicts catastrophic robotic arm failures up to 24 hours in advance using time-series sensor telemetry (`vibration`, `temperature`, `pressure`). The system is designed for high-precision alerting to reduce false alarms and avoid expensive unplanned downtime.

## Key Capabilities

- Time-series feature engineering:
  - Rolling mean, EMA, and rolling standard deviation over `1h`, `6h`, `12h`
  - Lag features: `t-1`, `t-2`
- Modeling pipeline:
  - Baselines: Logistic Regression, Random Forest
  - Production model: XGBoost (default) or LightGBM
  - Hyperparameter tuning with Optuna
- Imbalance-aware training:
  - PR-AUC as primary metric (not accuracy)
  - `scale_pos_weight` class weighting
  - Optional SMOTE
- Explainability:
  - SHAP local explanation export
- Deployment:
  - Flask API for real-time scoring
  - Model + metadata serialization via `joblib`
  - Latency check script (p95 target validation)

## Tech Stack

- Python 3.11+
- Pandas, NumPy
- scikit-learn, imbalanced-learn
- XGBoost, LightGBM, Optuna
- SHAP
- Flask

## Repository Structure

```text
.
|-- api/
|   |-- app.py
|-- scripts/
|   |-- latency_check.py
|-- src/
|   |-- explainability.py
|   |-- feature_engineering.py
|   |-- model_trainer.py
|   |-- train_pipeline.py
|-- requirements.txt
|-- README.md
```

## Quick Start

### 1) Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Train model

If `data/sensor_timeseries.csv` is not available, the pipeline generates synthetic training data automatically.

```bash
python -m src.train_pipeline --model-family xgboost --n-trials 5
```

Optional:

```bash
python -m src.train_pipeline --model-family lightgbm --use-smote --n-trials 10
```

Generated artifacts:

- `models/factoryguard_model.joblib`
- `reports/training_metrics.json`

### 3) Run API + UI

```bash
python -m api.app
```

Open:

- `http://127.0.0.1:8000/` (dashboard UI)
- `http://127.0.0.1:8000/health` (health check)

## API Contract

### POST `/predict`

Request:

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

Response:

```json
{
  "product": "FactoryGuard AI",
  "failure_probability_24h": 0.43,
  "predicted_failure_24h": 1,
  "decision_threshold": 0.11,
  "latency_ms": 8.7
}
```

## Explainability (SHAP)

```bash
python -m src.explainability --model-path models/factoryguard_model.joblib --input-csv data/sensor_timeseries.csv
```

Output:

- `reports/shap_local_explanation.json`

## Latency Benchmark

```bash
python scripts/latency_check.py
```

Reports:

- mean latency
- p95 latency
- max latency
- pass/fail for `p95 < 50ms`

## Production Notes

- Current Flask server is for development/demo. Use Gunicorn/Uvicorn + reverse proxy for production.
- Keep model file path configurable via `FACTORYGUARD_MODEL_PATH`.
- Add CI checks for training metrics regression and API contract tests.

## License

Internal / project-specific. Add your preferred license before publishing publicly.
