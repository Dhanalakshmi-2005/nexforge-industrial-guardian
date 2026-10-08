"""Transparent Risk-to-Action engine. Every point comes from a visible rule."""

LEVEL_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
RISK_BANDS = [(90, "CRITICAL"), (60, "HIGH"), (30, "MEDIUM"), (0, "LOW")]

INSPECT_STEPS = [
    "Inspect vibration / bearing condition",
    "Check temperature trend",
    "Review recent maintenance history",
    "Verify affected products",
]


def level_for(score):
    for threshold, name in RISK_BANDS:
        if score >= threshold:
            return name
    return "LOW"


def compute_risk(machine, machine_anomaly, quality_defect, repeated_anomaly,
                 reject_increase_pct, prior_confirmed_issues, data_confidence):
    breakdown = []
    if machine_anomaly:
        breakdown.append(("Machine anomaly detected", 30))
    if quality_defect:
        breakdown.append(("Quality defect detected", 30))
    if repeated_anomaly:
        breakdown.append(("Repeated anomaly in recent readings", 20))
    if reject_increase_pct >= 10:
        breakdown.append((f"Reject rate up {reject_increase_pct:.0f}% vs baseline", 10))
    if prior_confirmed_issues > 0:
        breakdown.append(("Earlier inspection confirmed a similar issue", 10))

    score = min(100, sum(points for _, points in breakdown))
    level = level_for(score)

    warnings = []
    if data_confidence == "LOW":
        warnings.append("Low data confidence: treat this score as indicative only.")
    elif data_confidence == "MEDIUM":
        warnings.append("Medium data confidence: verify on the floor before acting.")

    if machine_anomaly and quality_defect:
        reason = ("Machine anomaly coincides with a quality defect. The signals coincide and "
                  "should be investigated together; this does not prove a cause.")
    elif machine_anomaly:
        reason = "Sensor anomaly detected without a matching quality defect."
    elif quality_defect:
        reason = "Quality defect detected without a confirmed sensor anomaly."
    else:
        reason = "No abnormal signals in the current data."

    if level in ("HIGH", "CRITICAL"):
        recommendation, steps = f"INSPECT {machine.upper()}", INSPECT_STEPS
    elif level == "MEDIUM":
        recommendation, steps = f"MONITOR {machine.upper()}", ["Keep watching the sensor trend", "Re-check after the next production interval"]
    else:
        recommendation, steps = "CONTINUE MONITORING", []

    return {"score": score, "level": level, "breakdown": breakdown, "warnings": warnings,
            "reason": reason, "recommendation": recommendation, "steps": steps}
