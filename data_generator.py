"""Creates demo CSV datasets (simulated) in the data/ folder if they are missing."""
import os

import pandas as pd

from modules.simulator import MACHINES, SENSOR_NUMERIC, Simulator
from utils.database import BASE_DIR

DATA_DIR = os.path.join(BASE_DIR, "data")
DEMO_FILES = ["machines.csv", "sensor_data.csv", "production_data.csv", "maintenance_history.csv"]

MAINTENANCE_ROWS = [
    {"date": "2026-01-08", "machine_id": "Machine B", "event_type": "Inspection",
     "confirmed_issue": "Bearing wear confirmed", "action_taken": "Bearing replaced",
     "downtime_min": 95, "technician_notes": "Vibration signature matched early bearing wear."},
    {"date": "2026-01-19", "machine_id": "Machine A", "event_type": "Routine service",
     "confirmed_issue": "", "action_taken": "Lubrication", "downtime_min": 20,
     "technician_notes": "No issues found."},
    {"date": "2026-02-02", "machine_id": "Machine C", "event_type": "Inspection",
     "confirmed_issue": "", "action_taken": "None", "downtime_min": 0,
     "technician_notes": "False alarm from sensor drift; recalibrated."},
    {"date": "2026-02-14", "machine_id": "Machine B", "event_type": "Inspection",
     "confirmed_issue": "Coupling misalignment", "action_taken": "Realigned coupling",
     "downtime_min": 60, "technician_notes": "Temperature rise traced to misalignment."},
    {"date": "2026-03-03", "machine_id": "Machine A", "event_type": "Routine service",
     "confirmed_issue": "", "action_taken": "Filter change", "downtime_min": 15,
     "technician_notes": "Scheduled service."},
    {"date": "2026-03-21", "machine_id": "Machine C", "event_type": "Inspection",
     "confirmed_issue": "Bearing wear confirmed", "action_taken": "Bearing replaced",
     "downtime_min": 110, "technician_notes": "Confirmed after repeated vibration alerts."},
]


def _simulated_history(steps=480, seed=42):
    sim = Simulator(seed=seed, start=pd.Timestamp("2026-01-05 06:00:00").to_pydatetime())
    return pd.concat([sim.tick() for _ in range(steps)], ignore_index=True)


def generate_demo_files():
    os.makedirs(DATA_DIR, exist_ok=True)
    hist = _simulated_history()
    ts = pd.to_datetime(hist["timestamp"])

    sensors = hist[["machine_id"] + SENSOR_NUMERIC].copy()
    sensors.insert(0, "timestamp", ts.dt.strftime("%Y-%m-%d %H:%M:%S"))
    sensors.to_csv(os.path.join(DATA_DIR, "sensor_data.csv"), index=False)

    prod = (hist.assign(timestamp=ts.dt.strftime("%Y-%m-%d %H:00:00"))
                .groupby(["timestamp", "machine_id"], as_index=False)[["production_count", "rejected"]]
                .sum())
    prod["reject_rate"] = (prod["rejected"] / prod["production_count"].clip(lower=1)).round(4)
    prod.to_csv(os.path.join(DATA_DIR, "production_data.csv"), index=False)

    pd.DataFrame([
        {"machine_id": m, "line": f"Line {i + 1}", "location": "Demo hall", "status": "Running"}
        for i, m in enumerate(MACHINES)
    ]).to_csv(os.path.join(DATA_DIR, "machines.csv"), index=False)

    pd.DataFrame(MAINTENANCE_ROWS).to_csv(os.path.join(DATA_DIR, "maintenance_history.csv"), index=False)
    return DATA_DIR


def ensure_demo_data():
    if all(os.path.exists(os.path.join(DATA_DIR, f)) for f in DEMO_FILES):
        return DATA_DIR
    return generate_demo_files()
