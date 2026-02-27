from __future__ import annotations

import os
import time
from typing import Any, Dict

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

from src.feature_engineering import FeatureEngineer, SensorConfig


DEFAULT_MODEL_PATH = os.getenv("FACTORYGUARD_MODEL_PATH", "models/factoryguard_model.joblib")


def _load_bundle(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Train first with src/train_pipeline.py")
    return joblib.load(path)


def create_app(model_path: str = DEFAULT_MODEL_PATH) -> Flask:
    app = Flask(__name__)
    bundle = _load_bundle(model_path)
    model = bundle["model"]
    cfg = SensorConfig(**bundle["feature_engineer_config"])
    metadata = bundle["metadata"]
    fe = FeatureEngineer(cfg)
    threshold = float(metadata["threshold"])
    feature_columns = metadata["feature_columns"]

    @app.get("/")
    def index():
        return render_template_string(
            """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FactoryGuard AI | Predictive Maintenance Command Center</title>
  <style>
    :root{
      --base:#061119;
      --surface:#0a1b2a;
      --surface-soft:#102538;
      --line:#214663;
      --ink:#e8f2fd;
      --muted:#9bb6cf;
      --brand:#00c2ff;
      --brand2:#16f5b5;
      --danger:#ff5d73;
      --warn:#ffba08;
      --ok:#2ad38f;
      --field:#081726;
    }
    *{box-sizing:border-box}
    body{
      margin:0;
      font-family:"Aptos","Segoe UI Variable Text","Verdana",sans-serif;
      color:var(--ink);
      background:
        radial-gradient(900px 500px at 0% -20%, rgba(0,194,255,.22), transparent 55%),
        radial-gradient(800px 400px at 100% -10%, rgba(22,245,181,.18), transparent 55%),
        linear-gradient(165deg,#040b12,#081420 55%,#07121d);
      min-height:100vh;
    }
    .shell{max-width:1220px;margin:18px auto;padding:0 14px 24px}
    .hero{
      border:1px solid var(--line);
      background:linear-gradient(180deg, rgba(10,27,42,.92), rgba(10,27,42,.72));
      border-radius:20px;
      padding:18px;
      box-shadow:0 24px 44px rgba(0,0,0,.28);
      animation:fade .5s ease;
    }
    .hero-top{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap}
    h1{margin:0;font-size:31px;letter-spacing:.3px}
    .sub{margin:7px 0 0;color:var(--muted);max-width:760px}
    .chips{display:flex;gap:8px;flex-wrap:wrap}
    .chip{
      border:1px solid #2a6286;background:#10283c;color:#d3e8fb;
      padding:7px 11px;border-radius:999px;font-size:12px
    }
    .hero-grid{display:grid;grid-template-columns:repeat(3,minmax(180px,1fr));gap:10px;margin-top:14px}
    .metric{
      background:linear-gradient(180deg,#10273b,#0d2031);
      border:1px solid #234f6f;border-radius:12px;padding:10px 12px
    }
    .metric span{display:block;color:var(--muted);font-size:12px;margin-bottom:4px}
    .metric b{font-size:17px}
    .layout{margin-top:14px;display:grid;grid-template-columns:1fr 1fr;gap:14px}
    .card{
      border:1px solid var(--line);
      background:linear-gradient(180deg, rgba(16,37,56,.93), rgba(11,28,43,.9));
      border-radius:16px;padding:16px;
    }
    .card h2{margin:0 0 7px;font-size:20px}
    .copy{margin:0 0 12px;color:var(--muted);font-size:14px}
    .fields{display:grid;grid-template-columns:repeat(2,minmax(120px,1fr));gap:10px}
    label{display:block;font-size:12px;color:#a7c3db;margin-bottom:5px}
    input{
      width:100%;padding:10px;border-radius:10px;border:1px solid #2b5c7f;
      background:var(--field);color:var(--ink);outline:none
    }
    input:focus{border-color:var(--brand);box-shadow:0 0 0 3px rgba(0,194,255,.2)}
    .presets{display:flex;gap:8px;flex-wrap:wrap;margin-top:11px}
    .preset{
      border:1px solid #3a7094;background:#10263a;color:#cbe2f6;border-radius:999px;
      padding:7px 11px;font-size:12px;cursor:pointer
    }
    .run{
      margin-top:12px;width:100%;border:0;border-radius:11px;padding:12px;font-weight:700;cursor:pointer;
      color:#032033;background:linear-gradient(90deg,var(--brand),var(--brand2))
    }
    .run:disabled{opacity:.6;cursor:not-allowed}
    .ring-box{display:grid;place-items:center;padding:4px 0 10px}
    .ring{
      width:220px;height:220px;border-radius:50%;
      background:conic-gradient(var(--ok) 0deg, var(--ok) 36deg, #1a3b55 36deg 360deg);
      display:grid;place-items:center;position:relative;transition:all .35s ease
    }
    .ring::after{
      content:"";width:168px;height:168px;border-radius:50%;
      background:radial-gradient(circle at 50% 35%, #123452 0%, #0a1f32 70%);
      border:1px solid #2e5a7b
    }
    .center{position:absolute;z-index:1;text-align:center}
    .center b{font-size:37px;display:block;line-height:1}
    .center span{font-size:12px;color:var(--muted)}
    .state{text-align:center;font-size:18px;font-weight:700}
    .action{text-align:center;color:#c9ddf0;font-size:13px;margin-top:5px;min-height:20px}
    .kv{display:grid;grid-template-columns:repeat(2,minmax(130px,1fr));gap:8px;margin-top:10px}
    .k{border:1px solid #285473;background:#0e2235;padding:10px;border-radius:10px}
    .k span{display:block;color:var(--muted);font-size:12px}
    .k b{font-size:15px}
    .json{margin-top:10px;max-height:120px;overflow:auto;border-radius:10px;padding:10px;font-size:12px;
      background:#071321;border:1px solid #244964;color:#d0e6fb}
    .story{
      margin-top:14px;
      border:1px solid var(--line); border-radius:16px; padding:14px;
      background:linear-gradient(180deg, rgba(8,23,36,.92), rgba(8,23,36,.78));
    }
    .story h3{margin:0 0 8px;font-size:18px}
    .steps{display:grid;grid-template-columns:repeat(3,minmax(160px,1fr));gap:10px}
    .step{border:1px solid #255171;border-radius:12px;padding:10px;background:#0b1f31}
    .step b{font-size:14px}
    .step p{margin:6px 0 0;color:var(--muted);font-size:13px}
    @keyframes fade{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
    @media (max-width:980px){
      .layout{grid-template-columns:1fr}
      .hero-grid{grid-template-columns:1fr 1fr}
      .steps{grid-template-columns:1fr}
    }
    @media (max-width:620px){
      .fields{grid-template-columns:1fr}
      .hero-grid{grid-template-columns:1fr}
      .ring{width:190px;height:190px}
      .ring::after{width:142px;height:142px}
      .center b{font-size:31px}
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="hero-top">
        <div>
          <h1>FactoryGuard AI Command Center</h1>
          <p class="sub">
            This project predicts catastrophic robotic-arm failure up to 24 hours early using live sensor trends
            (vibration, temperature, pressure), so maintenance teams can schedule repairs before downtime happens.
          </p>
        </div>
        <div class="chips">
          <div class="chip">500 Robotic Arms</div>
          <div class="chip">24h Early Warning</div>
          <div class="chip">PR-AUC Optimized</div>
        </div>
      </div>
      <div class="hero-grid">
        <div class="metric"><span>Business Goal</span><b>Prevent Unplanned Downtime</b></div>
        <div class="metric"><span>How Model Learns</span><b>Rolling Stats + Lag Features</b></div>
        <div class="metric"><span>Decision Focus</span><b>High Precision Alerts</b></div>
      </div>
    </section>

    <section class="layout">
      <article class="card">
        <h2>Operator Input Panel</h2>
        <p class="copy">Enter latest readings. The system will score failure risk and recommend action in simple language.</p>
        <form id="predict-form">
          <div class="fields">
            <div>
              <label for="arm_id">Robotic Arm ID</label>
              <input id="arm_id" type="number" value="101" required>
            </div>
            <div>
              <label for="vibration">Vibration Level</label>
              <input id="vibration" type="number" step="0.01" value="0.66" required>
            </div>
            <div>
              <label for="temperature">Temperature (C)</label>
              <input id="temperature" type="number" step="0.1" value="76.8" required>
            </div>
            <div>
              <label for="pressure">Pressure</label>
              <input id="pressure" type="number" step="0.1" value="29.6" required>
            </div>
          </div>
          <div class="presets">
            <button class="preset" type="button" data-preset="normal">Normal Operation</button>
            <button class="preset" type="button" data-preset="watch">Rising Stress</button>
            <button class="preset" type="button" data-preset="critical">Critical Pattern</button>
          </div>
          <button class="run" id="predict-btn" type="submit">Run Predictive Analysis</button>
        </form>
      </article>

      <article class="card">
        <h2>Maintenance Decision Output</h2>
        <p class="copy">Single-screen summary for non-technical teams: risk %, status, and what to do now.</p>
        <div class="ring-box">
          <div class="ring" id="risk-ring">
            <div class="center"><b id="risk-value">0%</b><span>Failure Risk (24h)</span></div>
          </div>
        </div>
        <div class="state" id="risk-state">Ready</div>
        <div class="action" id="action-note">Submit data to receive recommendation.</div>
        <div class="kv">
          <div class="k"><span>Decision Threshold</span><b id="k-threshold">-</b></div>
          <div class="k"><span>Response Latency</span><b id="k-latency">-</b></div>
        </div>
        <pre class="json" id="out">No prediction yet.</pre>
      </article>
    </section>

    <section class="story">
      <h3>How This Project Works</h3>
      <div class="steps">
        <div class="step">
          <b>1. Sensor Pattern Analysis</b>
          <p>The engine builds rolling averages, variance, and lag behavior over 1h, 6h, and 12h windows.</p>
        </div>
        <div class="step">
          <b>2. AI Failure Prediction</b>
          <p>Class-imbalance aware XGBoost/LightGBM scores probability of failure in the next 24 hours.</p>
        </div>
        <div class="step">
          <b>3. Actionable Guidance</b>
          <p>Output is translated into practical maintenance action: continue, inspect, or intervene immediately.</p>
        </div>
      </div>
    </section>
  </div>

  <script>
    const form = document.getElementById("predict-form");
    const out = document.getElementById("out");
    const ring = document.getElementById("risk-ring");
    const riskValue = document.getElementById("risk-value");
    const riskState = document.getElementById("risk-state");
    const actionNote = document.getElementById("action-note");
    const kThreshold = document.getElementById("k-threshold");
    const kLatency = document.getElementById("k-latency");
    const predictBtn = document.getElementById("predict-btn");
    const presets = {
      normal: { vibration: 0.52, temperature: 69.2, pressure: 27.8 },
      watch: { vibration: 0.70, temperature: 80.0, pressure: 31.0 },
      critical: { vibration: 0.75, temperature: 78.0, pressure: 30.5 }
    };

    function applyPreset(name) {
      const p = presets[name];
      if (!p) return;
      document.getElementById("vibration").value = p.vibration;
      document.getElementById("temperature").value = p.temperature;
      document.getElementById("pressure").value = p.pressure;
    }

    document.querySelectorAll(".preset").forEach((btn) => {
      btn.addEventListener("click", () => applyPreset(btn.dataset.preset));
    });

    function renderRisk(prob) {
      const pct = Math.max(0, Math.min(100, prob * 100));
      const deg = (pct / 100) * 360;
      let tone = "var(--ok)";
      let title = "Healthy";
      let msg = "No immediate maintenance action required.";
      if (pct >= 65) {
        tone = "var(--danger)";
        title = "Critical";
        msg = "Schedule intervention now. High failure probability within 24 hours.";
      } else if (pct >= 35) {
        tone = "var(--warn)";
        title = "Attention";
        msg = "Plan inspection in the next maintenance window.";
      }
      ring.style.background = `conic-gradient(${tone} 0deg, ${tone} ${deg}deg, #1a3b55 ${deg}deg 360deg)`;
      riskValue.textContent = `${pct.toFixed(1)}%`;
      riskState.textContent = title;
      actionNote.textContent = msg;
    }

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const armId = Number(document.getElementById("arm_id").value);
      const vib = Number(document.getElementById("vibration").value);
      const temp = Number(document.getElementById("temperature").value);
      const press = Number(document.getElementById("pressure").value);
      const now = new Date();
      const rows = [];
      for (let i = 12; i >= 0; i -= 1) {
        const ts = new Date(now.getTime() - i * 60 * 60 * 1000);
        const trend = i / 12;
        rows.push({
          arm_id: armId,
          timestamp: ts.toISOString(),
          vibration: Number((vib * (0.90 + 0.10 * (1 - trend))).toFixed(4)),
          temperature: Number((temp - 3.5 * trend).toFixed(4)),
          pressure: Number((press - 1.2 * trend).toFixed(4))
        });
      }
      predictBtn.disabled = true;
      out.textContent = "Running analysis...";
      try {
        const res = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ rows })
        });
        const data = await res.json();
        if (!res.ok) {
          out.textContent = JSON.stringify(data, null, 2);
          actionNote.textContent = "Input error. Please verify all fields.";
          return;
        }

        const risk = Number(data.failure_probability_24h || 0);
        renderRisk(risk);
        kThreshold.textContent = `${(Number(data.decision_threshold || 0) * 100).toFixed(1)}%`;
        kLatency.textContent = `${Number(data.latency_ms || 0).toFixed(2)} ms`;
        out.textContent = JSON.stringify(data, null, 2);
      } catch (err) {
        out.textContent = "Request failed: " + err;
        actionNote.textContent = "Connection issue. Confirm API server is running.";
      } finally {
        predictBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
            """
        )

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "FactoryGuard AI"})

    @app.post("/predict")
    def predict():
        start = time.perf_counter()
        payload: Dict[str, Any] = request.get_json(silent=True) or {}
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) == 0:
            return jsonify({"error": "Payload must contain non-empty `rows` list."}), 400

        frame = pd.DataFrame(rows)
        needed = {cfg.id_column, cfg.timestamp_column, *cfg.sensor_columns}
        missing = sorted(needed - set(frame.columns))
        if missing:
            return jsonify({"error": f"Missing required keys in rows: {missing}"}), 400

        feat = fe.transform(frame)
        latest = feat.sort_values([cfg.id_column, cfg.timestamp_column]).tail(1)
        x = latest[feature_columns]
        prob = float(model.predict_proba(x)[:, 1][0])
        pred = int(prob >= threshold)

        latency_ms = (time.perf_counter() - start) * 1000.0
        return jsonify(
            {
                "product": "FactoryGuard AI",
                "failure_probability_24h": prob,
                "predicted_failure_24h": pred,
                "decision_threshold": threshold,
                "latency_ms": latency_ms,
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
