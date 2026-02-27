from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd


@dataclass(frozen=True)
class SensorConfig:
    sensor_columns: Iterable[str] = ("vibration", "temperature", "pressure")
    rolling_hours: Iterable[int] = (1, 6, 12)
    lag_steps: Iterable[int] = (1, 2)
    id_column: str = "arm_id"
    timestamp_column: str = "timestamp"


class FeatureEngineer:
    def __init__(self, config: SensorConfig | None = None):
        self.config = config or SensorConfig()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {
            self.config.id_column,
            self.config.timestamp_column,
            *self.config.sensor_columns,
        }
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        work = df.copy()
        work[self.config.timestamp_column] = pd.to_datetime(
            work[self.config.timestamp_column], utc=True
        )
        work = work.sort_values([self.config.id_column, self.config.timestamp_column])

        engineered_parts: List[pd.DataFrame] = []
        for arm_id, group in work.groupby(self.config.id_column, group_keys=False):
            grouped = group.copy()
            grouped = grouped.set_index(self.config.timestamp_column)
            for sensor in self.config.sensor_columns:
                for h in self.config.rolling_hours:
                    window = f"{h}h"
                    grouped[f"{sensor}_roll_mean_{h}h"] = (
                        grouped[sensor].rolling(window=window, min_periods=1).mean()
                    )
                    grouped[f"{sensor}_ema_{h}h"] = grouped[sensor].ewm(
                        span=max(h, 2), adjust=False, min_periods=1
                    ).mean()
                    grouped[f"{sensor}_roll_std_{h}h"] = (
                        grouped[sensor]
                        .rolling(window=window, min_periods=2)
                        .std()
                        .fillna(0.0)
                    )
                for lag in self.config.lag_steps:
                    grouped[f"{sensor}_lag_{lag}"] = grouped[sensor].shift(lag)
            grouped[self.config.id_column] = arm_id
            grouped = grouped.reset_index()
            engineered_parts.append(grouped)

        out = pd.concat(engineered_parts, ignore_index=True)
        out = out.sort_values([self.config.id_column, self.config.timestamp_column])
        for sensor in self.config.sensor_columns:
            for lag in self.config.lag_steps:
                col = f"{sensor}_lag_{lag}"
                out[col] = out.groupby(self.config.id_column)[col].bfill()
                out[col] = out[col].fillna(out[sensor])
        return out

    def feature_columns(self) -> List[str]:
        cols: List[str] = []
        for sensor in self.config.sensor_columns:
            cols.append(sensor)
            for h in self.config.rolling_hours:
                cols.extend(
                    [
                        f"{sensor}_roll_mean_{h}h",
                        f"{sensor}_ema_{h}h",
                        f"{sensor}_roll_std_{h}h",
                    ]
                )
            for lag in self.config.lag_steps:
                cols.append(f"{sensor}_lag_{lag}")
        return cols
