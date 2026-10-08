"""Shared constants and input validation helpers."""
import os

import pandas as pd

DEMO_BANNER = "DEMO DATA — simulated factory signals, not real plant data"
SENSOR_NUMERIC = ["temperature", "vibration", "pressure", "rpm", "current"]
SENSOR_REQUIRED = ["timestamp", "machine_id"] + SENSOR_NUMERIC
RISK_EMOJI = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}
CONFIDENCE_EMOJI = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}


def safe_read_csv(path):
    """Return a DataFrame, or None if the file is missing or unreadable."""
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        return None
    return None


def validate_sensor_csv(raw):
    """Validate an uploaded sensor CSV. Returns (clean_df or None, message)."""
    if raw is None or raw.empty:
        return None, "The uploaded file is empty."
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in SENSOR_REQUIRED if c not in df.columns]
    if missing:
        return None, ("Missing required column(s): " + ", ".join(missing)
                      + ". Expected: " + ", ".join(SENSOR_REQUIRED))
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    except Exception:
        return None, "Column 'timestamp' could not be read as dates."
    df["machine_id"] = df["machine_id"].astype(str).str.strip()
    for c in SENSOR_NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    before = len(df)
    df = df.dropna(subset=SENSOR_REQUIRED)
    if df.empty:
        return None, "No valid numeric sensor rows were found."
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    skipped = before - len(df)
    msg = f"Validated {len(df)} rows across {df['machine_id'].nunique()} machine(s)."
    if skipped:
        msg += f" {skipped} invalid row(s) skipped."
    return df, msg
