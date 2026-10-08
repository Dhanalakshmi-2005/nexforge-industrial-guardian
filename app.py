"""NEXFORGE: AI-powered industrial intelligence prototype (Streamlit entry point).

Run with:  streamlit run app.py
All factory data is SIMULATED demo data unless a CSV is uploaded in Settings.
"""
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.anomaly_detection import assess_machine, load_or_train_model, score_windows, train_model
from modules.maintenance import (STATUSES, create_ticket, list_tickets, parse_ticket_id,
                                 prior_confirmed_issues, record_outcome, update_status)
from modules.quality_inspection import DEMO_LABEL, analyze_image, make_sample_image
from modules.risk_engine import LEVEL_ORDER, compute_risk
from modules.simulator import BASELINE, MACHINES, Simulator
from utils.data_generator import DATA_DIR, ensure_demo_data, generate_demo_files
from utils.database import DB_PATH, execute, executemany, init_db, query_df
from utils.helpers import CONFIDENCE_EMOJI, DEMO_BANNER, RISK_EMOJI, safe_read_csv, validate_sensor_csv

st.set_page_config(page_title="NEXFORGE", page_icon="🏭", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
.nf-hero {background: linear-gradient(120deg,#0b1d2a,#12324a 60%,#0f2b1f); border:1px solid #1f4d63;
  border-radius:16px; padding:22px 26px; margin-bottom:14px;}
.nf-title {font-size:2.2rem; font-weight:800; letter-spacing:.08em; color:#e8f7ff;}
.nf-sub {color:#7fd1ea; font-size:1rem; letter-spacing:.04em;}
.nf-tag {color:#9fb7c4; font-size:.9rem; margin-top:4px;}
.nf-card {background:#0f1c26; border:1px solid #1d3a4a; border-radius:12px; padding:14px 16px; margin-bottom:10px;}
.nf-label {color:#8aa6b5; font-size:.75rem; text-transform:uppercase; letter-spacing:.08em;}
.nf-value {color:#f0fbff; font-size:1.6rem; font-weight:700;}
.nf-sub2 {color:#8aa6b5; font-size:.8rem;}
.nf-demo {display:inline-block; background:#3a2a06; color:#ffc857; border:1px solid #7a5a12;
  border-radius:999px; padding:2px 12px; font-size:.72rem; font-weight:700; letter-spacing:.08em;}
.nf-alert {border-left:4px solid #ff5d5d; background:#2a1417; padding:10px 14px; border-radius:8px; color:#ffd5d5; margin-bottom:8px;}
.nf-ok {border-left:4px solid #3ddc97; background:#0e2a1f; padding:10px 14px; border-radius:8px; color:#c8ffe6; margin-bottom:8px;}
.nf-warn {border-left:4px solid #ffc857; background:#2a2310; padding:10px 14px; border-radius:8px; color:#fff1c2; margin-bottom:8px;}
</style>
"""

PAGES = [
    "🏠 Command Center", "📊 Machine Health", "👁️ Quality Inspection",
    "🧠 Risk-to-Action Engine", "🔧 Maintenance", "📈 Analytics",
    "📜 History", "⚙️ Settings / About",
]


# ---------------------------------------------------------------- setup ----
@st.cache_resource(show_spinner=False)
def setup_storage():
    init_db()
    ensure_demo_data()
    return True


@st.cache_resource(show_spinner="Training anomaly model on simulated normal data…")
def get_model():
    return load_or_train_model()


def start_fresh_simulation():
    sim = Simulator(seed=11)
    warm = pd.concat([sim.tick() for _ in range(60)], ignore_index=True)
    st.session_state.sim = sim
    st.session_state.history = warm
    st.session_state.quality = {}
    st.session_state.alerted = set()
    st.session_state.qi_last = None


# -------------------------------------------------------------- helpers ----
def card(label, value, sub=""):
    st.markdown(f'<div class="nf-card"><div class="nf-label">{label}</div>'
                f'<div class="nf-value">{value}</div><div class="nf-sub2">{sub}</div></div>',
                unsafe_allow_html=True)


def demo_badge():
    st.markdown('<span class="nf-demo">DEMO DATA</span> <span class="nf-sub2">Simulated factory signals</span>',
                unsafe_allow_html=True)


def style_fig(fig, height=300, title=None):
    fig.update_layout(template="plotly_dark", height=height, title=title,
                      margin=dict(l=10, r=10, t=40 if title else 15, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.2))
    return fig


def persist_readings(new):
    rows = [(r.machine_id, r.timestamp.strftime("%Y-%m-%d %H:%M:%S"), r.temperature, r.vibration,
             r.pressure, r.rpm, r.current) for r in new.itertuples(index=False)]
    executemany("INSERT INTO sensor_readings (machine_id, timestamp, temperature, vibration, pressure, rpm, current) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)


def production_change_pct(df, machine):
    if df is None or not {"production_count", "rejected"}.issubset(df.columns):
        return 0.0
    sub = df[df["machine_id"] == machine]
    if len(sub) < 40:
        return 0.0
    base, recent = sub.iloc[: len(sub) // 2], sub.tail(20)
    rate_base = base["rejected"].sum() / max(base["production_count"].sum(), 1)
    rate_recent = recent["rejected"].sum() / max(recent["production_count"].sum(), 1)
    if rate_base <= 0:
        return 0.0
    return float((rate_recent - rate_base) / rate_base * 100)


def quality_defect(machine):
    q = st.session_state.quality.get(machine)
    return bool(q and q.get("status") == "ANOMALY")


def machine_risk(machine, a, df):
    return compute_risk(
        machine=machine,
        machine_anomaly=a["is_anomaly"],
        quality_defect=quality_defect(machine),
        repeated_anomaly=a["recent_flags"] >= 3,
        reject_increase_pct=production_change_pct(df, machine),
        prior_confirmed_issues=prior_confirmed_issues(machine),
        data_confidence=a["confidence"],
    )


def machine_health(a):
    if not a["has_data"]:
        return None
    drift = max(0.0, a["metrics"].get("temp_increase_pct", 0.0))
    penalty = (45 if a["is_anomaly"] else 0) + min(30.0, drift * 1.5)
    return int(max(0, round(100 - penalty)))


def evidence_for(machine, a):
    items = list(a["reasons"]) if a["is_anomaly"] else []
    q = st.session_state.quality.get(machine)
    if q and q.get("status") == "ANOMALY":
        items.append("Quality inspection: " + (", ".join(q["defects"]) or "anomaly"))
    prior = prior_confirmed_issues(machine)
    if prior:
        items.append(f"{prior} earlier inspection(s) confirmed an issue")
    change = production_change_pct(st.session_state.history, machine)
    if change >= 10:
        items.append(f"Reject rate up {change:.0f}% vs baseline")
    return items or ["No abnormal evidence in current data"]


def open_ticket(machine):
    df = st.session_state.history
    a = assess_machine(df, get_model(), machine)
    risk = machine_risk(machine, a, df)
    return create_ticket(
        machine, risk["level"],
        "; ".join(a["reasons"]) if a["is_anomaly"] else "No sustained sensor anomaly",
        "; ".join(evidence_for(machine, a)),
        risk["recommendation"],
    )


def analyze_all(df, bundle):
    return {m: assess_machine(df, bundle, m) for m in MACHINES}


def log_new_alerts(bundle):
    df = st.session_state.history
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    for m in MACHINES:
        a = assess_machine(df, bundle, m)
        if a["is_anomaly"]:
            if m not in st.session_state.alerted:
                risk = machine_risk(m, a, df)
                execute("INSERT INTO alerts (timestamp, machine_id, alert_type, severity, evidence, risk_score) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (now, m, "Sensor anomaly", risk["level"], "; ".join(a["reasons"]), float(risk["score"])))
                st.session_state.alerted.add(m)
        else:
            st.session_state.alerted.discard(m)
        execute("UPDATE machines SET status = ? WHERE machine_name = ?",
                ("Anomaly" if a["is_anomaly"] else "Running", m))


def simulate(steps, bundle):
    new = pd.concat([st.session_state.sim.tick() for _ in range(steps)], ignore_index=True)
    st.session_state.history = pd.concat([st.session_state.history, new], ignore_index=True).tail(3000)
    persist_readings(new)
    log_new_alerts(bundle)


def inject(machine, bundle):
    st.session_state.sim.inject_anomaly(machine)
    simulate(15, bundle)
    st.toast(f"Anomaly injected into {machine}")


def sensor_chart(df, machine, col, title, baseline=None):
    sub = df[df["machine_id"] == machine].tail(180)
    fig = go.Figure(go.Scatter(x=sub["timestamp"], y=sub[col], mode="lines", line=dict(color="#4cc9f0")))
    if baseline is not None:
        fig.add_hline(y=baseline, line_dash="dot", line_color="#8aa6b5")
    return style_fig(fig, height=260, title=title)


def _date_bounds(picked, dmin, dmax):
    if isinstance(picked, (tuple, list)):
        if len(picked) == 2:
            return picked[0], picked[1]
        if len(picked) == 1:
            return picked[0], picked[0]
        return dmin, dmax
    return picked, picked


# ---------------------------------------------------------------- pages ----
def page_command_center(bundle):
    st.markdown('<div class="nf-hero"><div class="nf-title">NEXFORGE</div>'
                '<div class="nf-sub">AI-Powered Industrial Intelligence</div>'
                '<div class="nf-tag">Predict • Detect • Explain • Act</div></div>', unsafe_allow_html=True)
    demo_badge()
    df = st.session_state.history
    analysis = analyze_all(df, bundle)
    risks = {m: machine_risk(m, analysis[m], df) for m in MACHINES}

    healths = [machine_health(analysis[m]) for m in MACHINES]
    health = round(sum(healths) / len(healths))
    q_scores = [100 - q["defect_score"] for q in st.session_state.quality.values()
                if q.get("defect_score") is not None]
    quality_txt = f"{round(sum(q_scores) / len(q_scores))}%" if q_scores else "—"
    alerts = sum(analysis[m]["is_anomaly"] for m in MACHINES) + sum(quality_defect(m) for m in MACHINES)
    overall = max((risks[m]["level"] for m in MACHINES), key=LEVEL_ORDER.index)
    pending = int(query_df("SELECT COUNT(*) AS n FROM maintenance_tickets WHERE status != 'Completed'")["n"].iloc[0])

    cols = st.columns(6)
    with cols[0]: card("Machine health", f"{health}%", "average of monitored machines")
    with cols[1]: card("Quality score", quality_txt, "run an inspection to update")
    with cols[2]: card("Active alerts", alerts, "anomalies + quality defects")
    with cols[3]: card("Overall risk", f"{RISK_EMOJI[overall]} {overall}", "highest machine risk")
    with cols[4]: card("Machines monitored", len(MACHINES), "virtual machines")
    with cols[5]: card("Pending inspections", pending, "tickets not yet completed")

    st.subheader("Machine status")
    tiles = st.columns(len(MACHINES))
    for col, m in zip(tiles, MACHINES):
        a = analysis[m]
        status = "⚠️ ANOMALY" if a["is_anomaly"] else "✅ NORMAL"
        with col:
            st.markdown(f'<div class="nf-card"><div class="nf-label">{m}</div>'
                        f'<div class="nf-value" style="font-size:1.2rem">{status}</div>'
                        f'<div class="nf-sub2">Confidence {CONFIDENCE_EMOJI[a["confidence"]]} {a["confidence"]} · '
                        f'Risk {RISK_EMOJI[risks[m]["level"]]} {risks[m]["level"]}</div></div>',
                        unsafe_allow_html=True)

    fig = go.Figure()
    for m in MACHINES:
        sub = df[df["machine_id"] == m].tail(120)
        fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["vibration"], mode="lines", name=m))
    st.plotly_chart(style_fig(fig, height=280, title="Vibration across machines (simulated)"), width="stretch")

    st.markdown("### Final summary")
    top = max(MACHINES, key=lambda m: LEVEL_ORDER.index(risks[m]["level"]) * 100 + risks[m]["score"])
    r = risks[top]
    st.markdown(f'<div class="nf-card"><div class="nf-value" style="font-size:1.3rem">NEXFORGE</div>'
                f'<div class="nf-sub2">Predict. Detect. Explain. Act.</div><br>'
                f'<b>Machine risk:</b> {RISK_EMOJI[r["level"]]} {r["level"]} ({top})<br>'
                f'<b>Quality:</b> {"ANOMALY DETECTED" if quality_defect(top) else "No anomaly recorded"}<br>'
                f'<b>Recommended action:</b> {r["recommendation"]}</div>', unsafe_allow_html=True)
    can_ticket = r["level"] in ("HIGH", "CRITICAL")
    if st.button("CREATE INSPECTION TICKET", type="primary", disabled=not can_ticket, key="cc_ticket"):
        st.success(f"Ticket {open_ticket(top)} created for {top}.")
    if not can_ticket:
        st.caption("Ticket creation is enabled when risk is HIGH or CRITICAL.")


def page_machine_health(bundle):
    st.header("📊 Machine Health")
    demo_badge()
    machine = st.selectbox("Select machine", MACHINES, index=1, key="mh_machine")
    b1, b2, b3 = st.columns(3)
    if b1.button("▶ Start Simulation", width="stretch"):
        simulate(10, bundle)
    if b2.button("⚠ Inject Anomaly", width="stretch"):
        inject(machine, bundle)
    if b3.button("↺ Reset Simulation", width="stretch"):
        start_fresh_simulation()
        st.rerun()

    df = st.session_state.history
    analysis = analyze_all(df, bundle)
    rows = []
    for m in MACHINES:
        a = analysis[m]
        latest = df[df["machine_id"] == m].iloc[-1]
        risk = machine_risk(m, a, df)
        rows.append({
            "Machine": m,
            "Status": "⚠️ Anomaly" if a["is_anomaly"] else "✅ Normal",
            "Temperature (°C)": round(latest["temperature"], 1),
            "Vibration": round(latest["vibration"], 2),
            "Pressure": round(latest["pressure"], 2),
            "RPM": round(latest["rpm"]),
            "Current": round(latest["current"], 2),
            "Anomaly score": round(a["score"], 3) if a["score"] is not None else None,
            "Confidence": f"{CONFIDENCE_EMOJI[a['confidence']]} {a['confidence']}",
            "Risk": f"{RISK_EMOJI[risk['level']]} {risk['level']}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    a = analysis[machine]
    if a["is_anomaly"]:
        st.markdown(f'<div class="nf-alert"><b>⚠️ ANOMALY DETECTED — {machine}</b><br>'
                    f'Confidence {CONFIDENCE_EMOJI[a["confidence"]]} {a["confidence"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="nf-ok">✅ {machine}: operating within learned normal behaviour</div>',
                    unsafe_allow_html=True)
    with st.expander("Why did this alert happen? (Evidence)", expanded=a["is_anomaly"]):
        for reason in a["reasons"]:
            st.markdown(f"- {reason}")
        st.caption("Evidence comes from an Isolation Forest trained on simulated normal behaviour. "
                   "It highlights unusual patterns; it does not predict failure.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(sensor_chart(df, machine, "temperature", "Temperature over time", BASELINE["temperature"]),
                        width="stretch")
    with c2:
        st.plotly_chart(sensor_chart(df, machine, "vibration", "Vibration over time", BASELINE["vibration"]),
                        width="stretch")
    with c3:
        st.plotly_chart(sensor_chart(df, machine, "rpm", "RPM over time", BASELINE["rpm"]),
                        width="stretch")
    scored = score_windows(df[df["machine_id"] == machine], bundle)
    if not scored.empty:
        fig = go.Figure(go.Scatter(x=scored["timestamp"], y=scored["anomaly_score"], mode="lines",
                                   line=dict(color="#ffc857")))
        st.plotly_chart(style_fig(fig, height=220, title="Anomaly score (higher = more unusual)"),
                        width="stretch")


def page_quality():
    st.header("👁️ AI Visual Quality Inspection")
    st.caption(DEMO_LABEL)
    machine = st.selectbox("Product line / machine", MACHINES, index=1, key="qi_machine")
    up = st.file_uploader("Upload product image", type=["png", "jpg", "jpeg", "bmp"], key="qi_upload")
    s1, s2, s3 = st.columns(3)
    sample_kind = None
    if s1.button("Sample: good product", width="stretch"):
        sample_kind = "good"
    if s2.button("Sample: scratched product", width="stretch"):
        sample_kind = "scratch"
    if s3.button("Sample: surface defect", width="stretch"):
        sample_kind = "blemish"
    if sample_kind:
        st.session_state.qi_sample = sample_kind

    image = None
    if up is not None:
        image = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            st.error("This file could not be read as an image. Try PNG or JPG.")
    elif st.session_state.get("qi_sample"):
        image = make_sample_image(st.session_state.qi_sample)
    if image is None:
        st.info("Upload a product image, or use one of the sample images above.")

    if image is not None and st.button("Run inspection", type="primary", key="qi_run"):
        result = analyze_image(image)
        result["machine"] = machine
        st.session_state.quality[machine] = result
        st.session_state.qi_last = {"machine": machine, "result": result,
                                    "original": cv2.cvtColor(image, cv2.COLOR_BGR2RGB)}
        conf_val = {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3}[result["confidence"]]
        execute("INSERT INTO quality_results (timestamp, machine_id, result, defect_type, confidence) "
                "VALUES (?, ?, ?, ?, ?)",
                (pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), machine, result["status"],
                 ", ".join(result["defects"]) or None, conf_val))

    last = st.session_state.get("qi_last")
    if last:
        res = last["result"]
        if res["status"] == "PASSED":
            st.markdown('<div class="nf-ok">✅ QUALITY PASSED</div>', unsafe_allow_html=True)
        elif res["status"] == "ANOMALY":
            st.markdown('<div class="nf-alert">⚠️ QUALITY ANOMALY</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="nf-warn">⚠️ Insufficient data for reliable assessment.</div>',
                        unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            st.image(last["original"], caption="Product image", width="stretch")
        with right:
            if res["annotated_rgb"] is not None:
                st.image(res["annotated_rgb"], caption="Visual explanation (demo overlay)", width="stretch")
        m1, m2, m3 = st.columns(3)
        m1.metric("Defect score", "—" if res["defect_score"] is None else f"{res['defect_score']}/100")
        m2.metric("Confidence", f"{CONFIDENCE_EMOJI[res['confidence']]} {res['confidence']}")
        m3.metric("Possible defects", ", ".join(res["defects"]) or "None")
        st.markdown(f"**Explanation:** {res['explanation']}")


def page_risk(bundle):
    st.header("🧠 Risk-to-Action Engine")
    st.caption("Alert → Evidence → Risk → Recommended action. Rules are transparent; there is no black-box score.")
    df = st.session_state.history
    machine = st.selectbox("Machine", MACHINES, index=1, key="ra_machine")
    a = assess_machine(df, bundle, machine)
    risk = machine_risk(machine, a, df)

    st.markdown(f'<div class="nf-card"><div class="nf-label">Risk score</div>'
                f'<div class="nf-value">{risk["score"]}/100 · {RISK_EMOJI[risk["level"]]} {risk["level"]}</div>'
                f'<div class="nf-sub2">{risk["reason"]}</div></div>', unsafe_allow_html=True)
    for w in risk["warnings"]:
        st.markdown(f'<div class="nf-warn">{w}</div>', unsafe_allow_html=True)

    st.subheader("How the score was built")
    if risk["breakdown"]:
        bd = pd.DataFrame(risk["breakdown"], columns=["Signal", "Points"])
        st.dataframe(bd, hide_index=True, width="stretch")
        st.caption("Rule points are summed and capped at 100.")
    else:
        st.info("No risk signals triggered.")

    st.subheader("Recommendation")
    if risk["level"] in ("HIGH", "CRITICAL"):
        st.markdown(f'<div class="nf-alert"><b>🔧 {risk["recommendation"]}</b></div>', unsafe_allow_html=True)
    elif risk["level"] == "MEDIUM":
        st.markdown(f'<div class="nf-warn"><b>👀 {risk["recommendation"]}</b></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="nf-ok"><b>✅ {risk["recommendation"]}</b></div>', unsafe_allow_html=True)
    st.markdown("**Evidence**")
    for item in evidence_for(machine, a):
        st.markdown(f"- {item}")
    if risk["steps"]:
        st.markdown("**Recommended actions**")
        for i, step in enumerate(risk["steps"], 1):
            st.markdown(f"{i}. {step}")

    can_ticket = risk["level"] in ("HIGH", "CRITICAL")
    if st.button("Create Inspection Ticket", type="primary", disabled=not can_ticket, key="ra_ticket"):
        st.success(f"Ticket {open_ticket(machine)} created for {machine}.")

    st.subheader("Cross-signal correlation")
    sub = df[df["machine_id"] == machine].tail(120).copy()
    if len(sub) > 20 and {"production_count", "rejected"}.issubset(sub.columns):
        sub["reject_rate_pct"] = sub["rejected"] / sub["production_count"].clip(lower=1) * 100
        fig = go.Figure()
        for col, label in [("vibration", "Vibration"), ("temperature", "Temperature"), ("reject_rate_pct", "Reject rate")]:
            base = sub[col].iloc[:20].mean()
            change = (sub[col] - base) / max(abs(base), 1e-9) * 100
            fig.add_trace(go.Scatter(x=sub["timestamp"], y=change, mode="lines", name=label))
        st.plotly_chart(style_fig(fig, height=300, title="Change from early baseline (%)"), width="stretch")
        st.caption("Signals coincide and should be investigated. This shows timing overlap only, not causation.")
    else:
        st.info("Not enough readings for a correlation view yet.")


def page_maintenance():
    st.header("🔧 Maintenance")
    tickets = list_tickets()
    if tickets.empty:
        st.info("No inspection tickets yet. Create one from the Risk-to-Action Engine.")
        return
    chosen = st.multiselect("Filter by status", STATUSES, default=STATUSES, key="mt_status")
    view = tickets[tickets["status"].isin(chosen)]
    st.dataframe(view[["ticket_id", "machine_id", "created_at", "risk_level", "detected_anomaly",
                       "recommendation", "status", "confirmed_issue", "outcome"]],
                 hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Update status")
        tid = st.selectbox("Ticket", tickets["ticket_id"].tolist(), key="mt_tid")
        new_status = st.selectbox("New status", STATUSES, key="mt_new_status")
        if st.button("Update status", key="mt_update"):
            update_status(parse_ticket_id(tid), new_status)
            st.rerun()
    with right:
        st.subheader("Record inspection outcome")
        with st.form("outcome_form"):
            tid2 = st.selectbox("Ticket", tickets["ticket_id"].tolist(), key="mt_tid2")
            confirmed = st.text_input("Confirmed issue (leave blank if none)", placeholder="e.g. Bearing wear confirmed")
            action = st.text_input("Action taken", placeholder="e.g. Bearing inspection")
            downtime = st.number_input("Downtime (minutes)", min_value=0.0, value=0.0, step=5.0)
            outcome = st.selectbox("Outcome", ["Issue confirmed", "No issue found", "Monitoring recommended"])
            notes = st.text_area("Technician notes")
            if st.form_submit_button("Save outcome"):
                record_outcome(parse_ticket_id(tid2), confirmed, action, downtime, outcome, notes)
                st.success("Outcome saved. Ticket marked Completed.")
                st.rerun()
    st.caption("Recorded outcomes are stored in SQLite for future analysis and tuning of alert rules.")


def page_analytics():
    st.header("📈 Analytics")
    demo_badge()
    df = st.session_state.history
    if df.empty:
        st.info("No sensor history yet.")
        return
    dmin, dmax = df["timestamp"].min().date(), df["timestamp"].max().date()
    picked = st.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax, key="an_dates")
    start, end = _date_bounds(picked, dmin, dmax)
    d = df[(df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)].copy()
    if d.empty:
        st.info("No data in the selected range.")
        return
    d["reject_rate_pct"] = d["rejected"] / d["production_count"].clip(lower=1) * 100
    d["reject_smooth"] = d.groupby("machine_id")["reject_rate_pct"].transform(lambda s: s.rolling(15, min_periods=1).mean())

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(d, x="timestamp", y="vibration", color="machine_id", title="Machine anomaly trend (vibration)")
        st.plotly_chart(style_fig(fig, height=300), width="stretch")
    with c2:
        fig = px.line(d, x="timestamp", y="reject_smooth", color="machine_id", title="Reject rate (%, smoothed)")
        st.plotly_chart(style_fig(fig, height=300), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        agg = d.groupby("machine_id")[["temperature", "vibration", "rpm"]].mean().reset_index()
        fig = px.bar(agg.melt(id_vars="machine_id"), x="machine_id", y="value", color="variable",
                     barmode="group", title="Machine comparison (averages)")
        st.plotly_chart(style_fig(fig, height=300), width="stretch")
    with c4:
        alerts = query_df("SELECT machine_id, severity FROM alerts")
        if alerts.empty:
            st.info("No alerts logged yet. Run the simulation and inject an anomaly.")
        else:
            fig = px.histogram(alerts, x="machine_id", color="severity", title="Alert frequency and risk distribution")
            st.plotly_chart(style_fig(fig, height=300), width="stretch")

    c5, c6 = st.columns(2)
    with c5:
        qr = query_df("SELECT result, COUNT(*) AS n FROM quality_results GROUP BY result")
        if qr.empty:
            st.info("No quality inspections yet.")
        else:
            st.plotly_chart(style_fig(px.bar(qr, x="result", y="n", title="Quality results"), height=260),
                            width="stretch")
    with c6:
        tk = query_df("SELECT status, COUNT(*) AS n FROM maintenance_tickets GROUP BY status")
        if tk.empty:
            st.info("No maintenance tickets yet.")
        else:
            st.plotly_chart(style_fig(px.bar(tk, x="status", y="n", title="Maintenance tickets by status"), height=260),
                            width="stretch")

    st.subheader("Maintenance feedback loop")
    hist = safe_read_csv(f"{DATA_DIR}/maintenance_history.csv")
    if hist is not None:
        st.markdown("Past inspections (demo dataset):")
        st.dataframe(hist, hide_index=True, width="stretch")
    st.caption("Inspection results, confirmed issues, downtime, and technician notes form a feedback loop. "
               "Over time this can help label anomalies and tune alert rules. The demo data here is simulated.")


def page_history():
    st.header("📜 History")
    tickets = list_tickets()
    alerts = query_df("SELECT * FROM alerts ORDER BY id DESC")
    if tickets.empty and alerts.empty:
        st.info("No history yet. Run the simulation, then create a ticket.")
        return

    f1, f2, f3, f4 = st.columns(4)
    machine = f1.selectbox("Machine", ["All"] + MACHINES, key="hi_machine")
    risk = f2.multiselect("Risk", LEVEL_ORDER, default=LEVEL_ORDER, key="hi_risk")
    status = f3.multiselect("Ticket status", STATUSES, default=STATUSES, key="hi_status")
    query = f4.text_input("Search", key="hi_search")

    def _filter(df, machine_col, risk_col, ts_col):
        if df.empty:
            return df
        out = df.copy()
        if machine != "All":
            out = out[out[machine_col] == machine]
        if risk_col in out.columns:
            out = out[out[risk_col].isin(risk)]
        if query:
            mask = out.astype(str).apply(lambda col: col.str.contains(query, case=False, regex=False)).any(axis=1)
            out = out[mask]
        return out

    t_view = _filter(tickets, "machine_id", "risk_level", "created_at")
    if not t_view.empty:
        t_view = t_view[t_view["status"].isin(status)]
    st.subheader("Maintenance tickets")
    st.dataframe(t_view[["ticket_id", "created_at", "machine_id", "detected_anomaly", "risk_level",
                         "recommendation", "status", "outcome"]] if not t_view.empty else t_view,
                 hide_index=True, width="stretch")

    a_view = _filter(alerts, "machine_id", "severity", "timestamp")
    st.subheader("Alerts")
    st.dataframe(a_view[["timestamp", "machine_id", "alert_type", "severity", "evidence", "risk_score"]]
                 if not a_view.empty else a_view, hide_index=True, width="stretch")


def page_settings(bundle):
    st.header("⚙️ Settings / About")
    st.markdown("""
**NEXFORGE** is a hackathon prototype for AI-assisted smart-factory monitoring.
It combines anomaly detection on sensor windows, a demo visual-inspection pipeline,
a transparent risk engine, and a maintenance ticket workflow.

**Limitations.** All factory data is simulated unless you upload a CSV. The anomaly model is
trained on simulated normal behaviour and has not been validated on real plant data. The vision
pipeline is rule-based and labelled as a demo. Real deployment would need representative
factory data, validation, safety review, and integration testing.
""")
    st.caption(f"Database: {DB_PATH}")
    st.caption(f"Model: Isolation Forest, trained on {bundle.get('trained_on_windows', '?')} simulated normal windows")

    st.subheader("Data")
    c1, c2 = st.columns(2)
    if c1.button("Regenerate demo datasets", width="stretch"):
        generate_demo_files()
        st.success("Demo CSV files regenerated in the data/ folder.")
    if c2.button("Retrain anomaly model", width="stretch"):
        train_model()
        get_model.clear()
        st.success("Anomaly model retrained on simulated normal data.")

    st.subheader("Upload real sensor CSV (optional)")
    st.caption("Required columns: timestamp, machine_id, temperature, vibration, pressure, rpm, current")
    up = st.file_uploader("Sensor CSV", type=["csv"], key="csv_up")
    if up is not None:
        try:
            raw = pd.read_csv(up)
        except Exception as exc:
            st.error(f"The file could not be read as CSV: {exc}")
            return
        clean, msg = validate_sensor_csv(raw)
        if clean is None:
            st.error(msg)
            return
        st.success(msg)
        st.dataframe(clean.head(20), hide_index=True, width="stretch")
        for m in clean["machine_id"].unique():
            a = assess_machine(clean, bundle, m)
            st.markdown(f"**{m}**: {'⚠️ anomaly' if a['is_anomaly'] else '✅ normal'} · "
                        f"confidence {CONFIDENCE_EMOJI[a['confidence']]} {a['confidence']}")
            for reason in a["reasons"]:
                st.markdown(f"- {reason}")
        st.caption("Uploaded files have no production columns, so reject-rate evidence is not included.")


# --------------------------------------------------------------- router ----
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    try:
        setup_storage()
        bundle = get_model()
    except Exception as exc:
        st.error(f"NEXFORGE could not start: {exc}")
        st.stop()

    if "sim" not in st.session_state:
        start_fresh_simulation()

    with st.sidebar:
        st.markdown("## 🏭 NEXFORGE")
        st.caption("Predict • Detect • Explain • Act")
        page = st.radio("Navigate", PAGES, label_visibility="collapsed")
        demo_badge()
        st.caption("Simulated factory signals. Not validated on real plant data.")

    try:
        if page == PAGES[0]:
            page_command_center(bundle)
        elif page == PAGES[1]:
            page_machine_health(bundle)
        elif page == PAGES[2]:
            page_quality()
        elif page == PAGES[3]:
            page_risk(bundle)
        elif page == PAGES[4]:
            page_maintenance()
        elif page == PAGES[5]:
            page_analytics()
        elif page == PAGES[6]:
            page_history()
        else:
            page_settings(bundle)
    except Exception as exc:
        st.error(f"This page hit an unexpected problem: {exc}")


main()
