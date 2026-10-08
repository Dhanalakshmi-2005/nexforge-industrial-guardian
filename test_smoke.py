"""Smoke tests for NEXFORGE. Run:  python tests/test_smoke.py   (or pytest)"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

from modules.anomaly_detection import assess_machine, train_model  # noqa: E402
from modules.maintenance import create_ticket, list_tickets  # noqa: E402
from modules.quality_inspection import analyze_image, make_sample_image  # noqa: E402
from modules.risk_engine import compute_risk  # noqa: E402
from modules.simulator import Simulator  # noqa: E402
from utils.data_generator import ensure_demo_data  # noqa: E402
from utils.database import init_db  # noqa: E402
from utils.helpers import validate_sensor_csv  # noqa: E402


def test_anomaly_only_flags_injected_machine():
    bundle = train_model()
    sim = Simulator(seed=3)
    pre = pd.concat([sim.tick() for _ in range(60)], ignore_index=True)
    sim.inject_anomaly("Machine B")
    post = pd.concat([sim.tick() for _ in range(25)], ignore_index=True)
    df = pd.concat([pre, post], ignore_index=True)
    assert assess_machine(df, bundle, "Machine B")["is_anomaly"]
    assert not assess_machine(df, bundle, "Machine A")["is_anomaly"]
    assert not assess_machine(df, bundle, "Machine C")["is_anomaly"]


def test_quality_inspection_demo_cases():
    assert analyze_image(make_sample_image("good"))["status"] == "PASSED"
    assert analyze_image(make_sample_image("scratch"))["status"] == "ANOMALY"
    assert analyze_image(make_sample_image("blemish"))["status"] == "ANOMALY"
    assert analyze_image(None)["status"] == "INSUFFICIENT"


def test_risk_engine_matches_spec_example():
    r = compute_risk("Machine B", True, True, True, 0, 0, "HIGH")
    assert r["score"] == 80 and r["level"] == "HIGH"
    assert r["recommendation"] == "INSPECT MACHINE B"


def test_csv_validation():
    bad, msg = validate_sensor_csv(pd.DataFrame({"timestamp": ["2026-01-01"], "machine_id": ["A"]}))
    assert bad is None and "Missing required column" in msg
    good = pd.DataFrame({"timestamp": ["2026-01-01 00:00", "2026-01-01 00:01"], "machine_id": ["A", "A"],
                         "temperature": [60, 61], "vibration": [2.1, 2.2], "pressure": [5, 5],
                         "rpm": [1500, 1502], "current": [12, 12]})
    clean, _ = validate_sensor_csv(good)
    assert clean is not None and len(clean) == 2


def test_database_and_tickets():
    init_db()
    ensure_demo_data()
    label = create_ticket("Machine B", "HIGH", "test", "test evidence", "INSPECT MACHINE B")
    assert label.startswith("TKT-")
    assert not list_tickets().empty


def test_app_renders_all_pages():
    from streamlit.testing.v1 import AppTest
    PAGE_NAMES = ["📊 Machine Health", "👁️ Quality Inspection", "🧠 Risk-to-Action Engine", "🔧 Maintenance", "📈 Analytics", "📜 History", "⚙️ Settings / About"]
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=90).run()
    assert not at.exception, at.exception
    for name in PAGE_NAMES:
        at.sidebar.radio[0].set_value(name).run()
        assert not at.exception, (name, at.exception)
    at.sidebar.radio[0].set_value("🏠 Command Center").run()
    at.sidebar.radio[0].set_value("📊 Machine Health").run()
    buttons = {b.label: b for b in at.button}
    buttons["⚠ Inject Anomaly"].click().run()
    assert not at.exception, at.exception


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("All smoke tests passed.")
