from __future__ import annotations

import json
import statistics
import time

import pandas as pd

from api.app import create_app


def build_payload(n_rows: int = 16):
    ts = pd.date_range("2026-01-01", periods=n_rows, freq="h", tz="UTC")
    rows = []
    for t in ts:
        rows.append(
            {
                "arm_id": 123,
                "timestamp": t.isoformat(),
                "vibration": 0.62,
                "temperature": 74.5,
                "pressure": 29.1,
            }
        )
    return {"rows": rows}


def main():
    app = create_app()
    payload = build_payload()
    latencies = []
    with app.test_client() as client:
        for _ in range(200):
            t0 = time.perf_counter()
            resp = client.post("/predict", data=json.dumps(payload), content_type="application/json")
            dt = (time.perf_counter() - t0) * 1000.0
            if resp.status_code != 200:
                raise RuntimeError(f"Request failed: {resp.status_code}, {resp.data}")
            latencies.append(dt)
    p95 = sorted(latencies)[int(0.95 * len(latencies)) - 1]
    print(
        json.dumps(
            {
                "runs": len(latencies),
                "mean_ms": statistics.mean(latencies),
                "p95_ms": p95,
                "max_ms": max(latencies),
                "pass_under_50ms_p95": p95 < 50.0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
