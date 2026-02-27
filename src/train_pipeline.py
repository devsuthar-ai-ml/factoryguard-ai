from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.feature_engineering import FeatureEngineer, SensorConfig
from src.model_trainer import ModelTrainer


@dataclass
class BundleMeta:
    feature_columns: list[str]
    id_column: str
    timestamp_column: str
    sensor_columns: list[str]
    model_name: str
    threshold: float
    pr_auc_validation: float
    baseline_scores: dict[str, float]
    best_params: dict[str, object]


def _find_best_precision_threshold(y_true, y_prob, min_precision: float = 0.90) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    best = 0.5
    best_recall = -1.0
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if p >= min_precision and r > best_recall:
            best_recall = r
            best = float(t)
    return best


def _synthesize_training_data(rows: int = 120000, arms: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01", periods=rows // arms, freq="h", tz="UTC")
    frames = []
    for arm in range(arms):
        size = len(timestamps)
        base_v = np.zeros(size, dtype=float)
        base_t = np.zeros(size, dtype=float)
        base_p = np.zeros(size, dtype=float)
        base_v[0] = rng.normal(0.50, 0.05)
        base_t[0] = rng.normal(66.0, 2.2)
        base_p[0] = rng.normal(28.0, 1.4)
        for i in range(1, size):
            base_v[i] = 0.90 * base_v[i - 1] + 0.10 * rng.normal(0.50, 0.05)
            base_t[i] = 0.92 * base_t[i - 1] + 0.08 * rng.normal(66.0, 2.2)
            base_p[i] = 0.90 * base_p[i - 1] + 0.10 * rng.normal(28.0, 1.4)

        base_v += rng.normal(0, 0.015, size)
        base_t += rng.normal(0, 0.6, size)
        base_p += rng.normal(0, 0.5, size)

        event_roll = float(rng.random())
        if event_roll < 0.10:
            n_events = 1
        elif event_roll < 0.11:
            n_events = 2
        else:
            n_events = 0
        event_idx = []
        if n_events > 0:
            event_idx = sorted(
                rng.choice(np.arange(48, size - 1), size=n_events, replace=False).tolist()
            )

        for e in event_idx:
            start = max(e - 24, 0)
            ramp = np.linspace(0.0, 1.0, e - start + 1)
            base_v[start : e + 1] += 0.25 * ramp + rng.normal(0, 0.01, len(ramp))
            base_t[start : e + 1] += 18.0 * ramp + rng.normal(0, 0.5, len(ramp))
            base_p[start : e + 1] += 6.0 * ramp + rng.normal(0, 0.3, len(ramp))

        # Label at time t: failure will happen in next 24h.
        y = np.zeros(size, dtype=int)
        for e in event_idx:
            win_start = max(e - 24, 0)
            y[win_start:e] = 1

        base_v = np.clip(base_v, 0.2, 1.7)
        base_t = np.clip(base_t, 40.0, 130.0)
        base_p = np.clip(base_p, 15.0, 50.0)
        frames.append(
            pd.DataFrame(
                {
                    "arm_id": arm,
                    "timestamp": timestamps,
                    "vibration": base_v,
                    "temperature": base_t,
                    "pressure": base_p,
                    "failure_24h": y,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="FactoryGuard AI training pipeline")
    parser.add_argument("--input-csv", default="data/sensor_timeseries.csv")
    parser.add_argument("--output-model", default="models/factoryguard_model.joblib")
    parser.add_argument("--output-report", default="reports/training_metrics.json")
    parser.add_argument("--model-family", choices=["xgboost", "lightgbm"], default="xgboost")
    parser.add_argument("--use-smote", action="store_true")
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)

    if os.path.exists(args.input_csv):
        df = pd.read_csv(args.input_csv)
    else:
        df = _synthesize_training_data()

    cfg = SensorConfig()
    fe = FeatureEngineer(cfg)
    df_feat = fe.transform(df)

    target_col = "failure_24h"
    if target_col not in df_feat.columns:
        raise ValueError("Input data must include target column: failure_24h")

    features = fe.feature_columns()
    X = df_feat[features]
    y = df_feat[target_col].astype(int)

    trainer = ModelTrainer(random_state=42)
    result = trainer.train_production_model(
        X,
        y,
        model_family=args.model_family,
        use_smote=args.use_smote,
        n_trials=args.n_trials,
    )

    # Threshold is tuned toward high precision to minimize maintenance false alarms.
    y_prob_full = result.model.predict_proba(X)[:, 1]
    train_pr_auc = float(average_precision_score(y, y_prob_full))
    threshold = _find_best_precision_threshold(y, y_prob_full, min_precision=0.90)

    bundle_meta = BundleMeta(
        feature_columns=features,
        id_column=cfg.id_column,
        timestamp_column=cfg.timestamp_column,
        sensor_columns=list(cfg.sensor_columns),
        model_name=result.model_name,
        threshold=threshold,
        pr_auc_validation=result.production_score,
        baseline_scores=result.baseline_scores,
        best_params=result.best_params,
    )

    bundle = {
        "model": result.model,
        "feature_engineer_config": asdict(cfg),
        "metadata": asdict(bundle_meta),
    }
    joblib.dump(bundle, args.output_model)

    report = {
        "training_pr_auc_full": train_pr_auc,
        "validation_pr_auc_best": result.production_score,
        "model_name": result.model_name,
        "baseline_scores": result.baseline_scores,
        "best_params": result.best_params,
        "threshold": threshold,
        "use_smote": bool(args.use_smote),
        "rows": int(len(df_feat)),
        "positive_rate": float(y.mean()),
    }
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
