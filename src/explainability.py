from __future__ import annotations

import argparse
import json
import os

import joblib
import pandas as pd
import shap

from src.feature_engineering import FeatureEngineer, SensorConfig


def main():
    parser = argparse.ArgumentParser(description="Generate SHAP explainability artifacts")
    parser.add_argument("--model-path", default="models/factoryguard_model.joblib")
    parser.add_argument("--input-csv", default="data/sensor_timeseries.csv")
    parser.add_argument("--output-json", default="reports/shap_local_explanation.json")
    parser.add_argument("--sample-index", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    bundle = joblib.load(args.model_path)
    model = bundle["model"]
    cfg = SensorConfig(**bundle["feature_engineer_config"])
    metadata = bundle["metadata"]

    df = pd.read_csv(args.input_csv)
    fe = FeatureEngineer(cfg)
    feat_df = fe.transform(df)
    feature_cols = metadata["feature_columns"]
    X = feat_df[feature_cols]

    idx = args.sample_index if args.sample_index >= 0 else len(X) - 1
    idx = min(max(idx, 0), len(X) - 1)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.iloc[[idx]])
    vals = shap_values[0] if isinstance(shap_values, list) else shap_values[0]

    contrib = (
        pd.DataFrame({"feature": feature_cols, "shap_value": vals})
        .assign(abs_shap=lambda x: x["shap_value"].abs())
        .sort_values("abs_shap", ascending=False)
        .head(8)
    )

    output = {
        "sample_index": int(idx),
        "top_contributors": contrib.to_dict(orient="records"),
        "n_features": len(feature_cols),
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
