from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import optuna
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split


def _safe_pr_auc(y_true, y_prob) -> float:
    return float(average_precision_score(y_true, y_prob))


@dataclass
class TrainingResult:
    baseline_scores: Dict[str, float]
    production_score: float
    model_name: str
    model: object
    best_params: Dict[str, object]


class ModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def train_baselines(
        self, X_train, y_train, X_val, y_val
    ) -> Tuple[Dict[str, float], Dict[str, object]]:
        models = {
            "logistic_regression": LogisticRegression(
                max_iter=1000, class_weight="balanced", n_jobs=None
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=250,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=self.random_state,
                n_jobs=-1,
            ),
        }
        scores: Dict[str, float] = {}
        fitted: Dict[str, object] = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)[:, 1]
            scores[name] = _safe_pr_auc(y_val, y_prob)
            fitted[name] = model
        return scores, fitted

    def _build_xgboost(
        self, trial: optuna.Trial, scale_pos_weight: float, random_state: int
    ):
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            n_estimators=trial.suggest_int("n_estimators", 250, 800),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
        )

    def _build_lightgbm(
        self, trial: optuna.Trial, scale_pos_weight: float, random_state: int
    ):
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="binary",
            metric="average_precision",
            n_estimators=trial.suggest_int("n_estimators", 250, 900),
            num_leaves=trial.suggest_int("num_leaves", 16, 256),
            max_depth=trial.suggest_int("max_depth", -1, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 80),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
        )

    def train_production_model(
        self,
        X,
        y,
        model_family: str = "xgboost",
        use_smote: bool = False,
        n_trials: int = 30,
        val_size: float = 0.2,
    ) -> TrainingResult:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=val_size,
            random_state=self.random_state,
            stratify=y,
        )

        baseline_scores, _ = self.train_baselines(X_train, y_train, X_val, y_val)

        pos = max(int(y_train.sum()), 1)
        neg = max(int((y_train == 0).sum()), 1)
        scale_pos_weight = neg / pos

        if use_smote:
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        def objective(trial: optuna.Trial) -> float:
            if model_family == "lightgbm":
                model = self._build_lightgbm(
                    trial, scale_pos_weight, self.random_state
                )
            else:
                model = self._build_xgboost(trial, scale_pos_weight, self.random_state)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)[:, 1]
            return _safe_pr_auc(y_val, y_prob)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        if model_family == "lightgbm":
            best_model = self._build_lightgbm(
                optuna.trial.FixedTrial(study.best_params),
                scale_pos_weight,
                self.random_state,
            )
            model_name = "lightgbm"
        else:
            best_model = self._build_xgboost(
                optuna.trial.FixedTrial(study.best_params),
                scale_pos_weight,
                self.random_state,
            )
            model_name = "xgboost"

        best_model.fit(X, y)
        return TrainingResult(
            baseline_scores=baseline_scores,
            production_score=study.best_value,
            model_name=model_name,
            model=best_model,
            best_params=study.best_params,
        )
