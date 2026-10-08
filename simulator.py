"""Virtual factory simulator. Produces timestamped sensor and production readings.

All values are SIMULATED for demonstration. They are not real plant data.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

MACHINES = ["Machine A", "Machine B", "Machine C"]
SENSOR_NUMERIC = ["temperature", "vibration", "pressure", "rpm", "current"]
BASELINE = {"temperature": 60.0, "vibration": 2.1, "pressure": 5.0, "rpm": 1500.0, "current": 12.0}
NOISE = {"temperature": 0.6, "vibration": 0.08, "pressure": 0.05, "rpm": 12.0, "current": 0.2}
RAMP_STEPS = 12           # ticks for an injected anomaly to reach full strength
BASE_REJECT_RATE = 0.02   # 2% normal reject rate
BASE_PRODUCTION = 120     # units per interval


class Simulator:
    def __init__(self, seed=None, start=None):
        self.rng = np.random.default_rng(seed)
        self.clock = start or (datetime.now().replace(microsecond=0) - timedelta(minutes=120))
        self.anomaly_age = {m: None for m in MACHINES}

    def inject_anomaly(self, machine):
        self.anomaly_age[machine] = 0

    def reset(self):
        self.anomaly_age = {m: None for m in MACHINES}

    def is_anomalous(self, machine):
        return self.anomaly_age[machine] is not None

    def _progress(self, machine):
        age = self.anomaly_age[machine]
        return 0.0 if age is None else min(1.0, age / RAMP_STEPS)

    def tick(self, minutes=1):
        """Advance the clock and return one row per machine."""
        self.clock = self.clock + timedelta(minutes=minutes)
        rows = []
        n = self.rng.normal
        for m in MACHINES:
            p = self._progress(m)
            produced = max(0, int(n(BASE_PRODUCTION * (1 - 0.1 * p), 4)))
            reject_p = min(1.0, BASE_REJECT_RATE + 0.10 * p)
            rows.append({
                "timestamp": self.clock,
                "machine_id": m,
                "temperature": BASELINE["temperature"] + n(0, NOISE["temperature"]) + 12.0 * p,
                "vibration": BASELINE["vibration"] + n(0, NOISE["vibration"]) + 1.5 * p,
                "pressure": BASELINE["pressure"] + n(0, NOISE["pressure"]) + 0.6 * p,
                "rpm": n(BASELINE["rpm"] - 90.0 * p, NOISE["rpm"] * (1 + 8 * p)),
                "current": BASELINE["current"] + n(0, NOISE["current"]) + 1.8 * p,
                "production_count": produced,
                "rejected": int(self.rng.binomial(produced, reject_p)),
            })
        for m in MACHINES:
            if self.anomaly_age[m] is not None:
                self.anomaly_age[m] += 1
        return pd.DataFrame(rows)
