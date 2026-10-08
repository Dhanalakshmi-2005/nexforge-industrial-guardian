"""Isolation Forest anomaly detection over sensor-window features.

Honest scope: the model learns what NORMAL simulated behaviour looks like and
flags windows that deviate from it. It does not predict failures.
"""
import os

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from modules.simulator import MACHINES, SENSOR_NUMERIC, Simulator
from utils.database import BASE_DIR

MODEL_PATH = os.path.join(BASE_DIR, "models", "anomaly_model.pkl")
WINDOW = 10
MIN_WINDOWS = 20
FEATURES = [
    "mean_temperature", "temp_std", "mean_vibration", "vibration_deviation",
    "rpm_variation", "pressure_variation", "current_variation",
]
VIB_BASELINE = 2.1


def _empty_features():
    return pd.DataFrame(columns=["machine_id", "timestamp"] + FEATURES)


def build_features(df):
    """Compute rolling sensor-window features for each machine."""
    if df is None or df.empty:
        return _empty_features()
    frames = []
    for machine, g in df.sort_values("timestamp").groupby("machine_id"):
        g = g.reset_index(drop=True)
        roll = g[SENSOR_NUMERIC].rolling(WINDOW, min_periods=WINDOW)
        mean, std = roll.mean(), roll.std()
        f = pd.DataFrame({
            "machine_id": machine,
            "timestamp": g["timestamp"],
            "mean_temperature": mean["temperature"],
            "temp_std": std["temperature"],
            "mean_vibration": mean["vibration"],
            "vibration_deviation": mean["vibration"] - VIB_BASELINE,
            "rpm_variation": std["rpm"] / mean["rpm"] * 100,
            "pressure_variation": std["pressure"],
            "current_variation": std["current"],
        })
        frames.append(f.dropna(subset=FEATURES))
    return pd.concat(frames, ignore_index=True) if frames else _empty_features()


def train_model(normal_df=None, seed=42, steps=600):
    """Train on simulated NORMAL data and save the bundle to models/."""
    if normal_df is None:
        sim = Simulator(seed=seed)
        normal_df = pd.concat([sim.tick() for _ in range(steps)], ignore_index=True)
    feats = build_features(normal_df)
    if len(feats) < 50:
        raise ValueError("Not enough normal data to train the anomaly model.")
    X = feats[FEATURES]
    model = IsolationForest(n_estimators=200, contamination=0.01, random_state=seed)
    model.fit(X)
    baseline = {f: {"median": float(X[f].median()), "p995": float(X[f].quantile(0.995))}
                for f in FEATURES}
    bundle = {"model": model, "baseline": baseline, "features": FEATURES,
              "trained_on_windows": int(len(X))}
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        try:
            bundle = joblib.load(MODEL_PATH)
            if isinstance(bundle, dict) and "model" in bundle:
                return bundle
        except Exception:
            pass
    return train_model()


def score_windows(df, bundle):
    """Return feature windows with anomaly_score (higher = more unusual) and is_anomaly."""
    feats = build_features(df)
    if feats.empty:
        return feats
    X = feats[bundle["features"]]
    feats = feats.copy()
    feats["anomaly_score"] = -bundle["model"].score_samples(X)
    feats["is_anomaly"] = bundle["model"].predict(X) == -1
    return feats.reset_index(drop=True)


def assess_machine(df, bundle, machine):
    """Assess one machine: anomaly status, confidence, and plain-language evidence."""
    sub = df[df["machine_id"] == machine] if df is not None and not df.empty else df
    scored = score_windows(sub, bundle)
    result = {
        "machine": machine, "has_data": not scored.empty, "windows": len(scored),
        "is_anomaly": False, "confirmed": False, "confidence": "LOW", "score": None,
        "recent_flags": 0, "reasons": [], "metrics": {"temp_increase_pct": 0.0},
    }
    if scored.empty:
        result["reasons"] = ["Insufficient data for reliable assessment."]
        return result

    last = scored.iloc[-1]
    b = bundle["baseline"]
    recent3 = int(scored["is_anomaly"].tail(3).sum())
    recent5 = int(scored["is_anomaly"].tail(5).sum())
    recent10 = int(scored["is_anomaly"].tail(10).sum())
    confirmed = recent3 >= 2  # a single flagged window is not enough to alert

    if len(scored) < MIN_WINDOWS:
        confidence = "LOW"
    elif confirmed:
        confidence = "HIGH" if recent5 >= 3 else "MEDIUM"
    else:
        confidence = "HIGH" if recent5 == 0 else "MEDIUM"

    t_base = b["mean_temperature"]["median"]
    t_pct = float((last["mean_temperature"] - t_base) / t_base * 100)
    reasons = []
    if confirmed:
        if t_pct > 5:
            reasons.append(f"Temperature increased {t_pct:.0f}% above baseline")
        v_lim = b["mean_vibration"]["p995"]
        if last["mean_vibration"] > v_lim:
            reasons.append(f"Vibration exceeded normal range ({last['mean_vibration']:.2f} vs limit {v_lim:.2f})")
        if last["rpm_variation"] > b["rpm_variation"]["p995"]:
            reasons.append("RPM became unstable (variation above normal)")
        if last["pressure_variation"] > b["pressure_variation"]["p995"]:
            reasons.append("Pressure fluctuating beyond normal range")
        if last["current_variation"] > b["current_variation"]["p995"]:
            reasons.append("Current draw fluctuating beyond normal range")
        if not reasons:
            reasons.append("Combined sensor pattern differs from learned normal behaviour")
    else:
        reasons.append("No sustained anomaly; readings are within learned normal behaviour")

    result.update({
        "is_anomaly": confirmed, "confirmed": confirmed, "confidence": confidence,
        "score": float(last["anomaly_score"]), "recent_flags": recent10, "reasons": reasons,
        "metrics": {"temp_increase_pct": t_pct,
                    "vibration_mean": float(last["mean_vibration"]),
                    "rpm_cv": float(last["rpm_variation"])},
    })
    return result
