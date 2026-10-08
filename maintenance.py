"""Inspection ticket workflow stored in SQLite."""
from datetime import datetime

import pandas as pd

from utils.database import execute, query_df

STATUSES = ["Open", "In Progress", "Completed"]


def ticket_label(tid):
    return f"TKT-{int(tid):04d}"


def parse_ticket_id(label):
    return int(str(label).split("-")[-1])


def create_ticket(machine, risk_level, detected_anomaly, evidence, recommendation):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_id = execute(
        "INSERT INTO maintenance_tickets (machine_id, created_at, risk_level, detected_anomaly, "
        "evidence, recommendation, status) VALUES (?, ?, ?, ?, ?, ?, 'Open')",
        (machine, now, risk_level, detected_anomaly, evidence, recommendation),
    )
    return ticket_label(new_id)


def list_tickets(machine=None, status=None):
    sql = "SELECT * FROM maintenance_tickets WHERE 1=1"
    params = []
    if machine:
        sql += " AND machine_id = ?"
        params.append(machine)
    if status:
        sql += " AND status = ?"
        params.append(status)
    df = query_df(sql + " ORDER BY id DESC", tuple(params))
    ids = df["id"].apply(ticket_label) if not df.empty else pd.Series(dtype=object)
    df.insert(0, "ticket_id", ids)
    return df


def update_status(ticket_int, status):
    execute("UPDATE maintenance_tickets SET status = ? WHERE id = ?", (status, ticket_int))


def record_outcome(ticket_int, confirmed_issue, action_taken, downtime_min, outcome, notes):
    execute(
        "UPDATE maintenance_tickets SET confirmed_issue = ?, action_taken = ?, downtime_min = ?, "
        "outcome = ?, notes = ?, status = 'Completed' WHERE id = ?",
        (confirmed_issue, action_taken, float(downtime_min), outcome, notes, ticket_int),
    )


def prior_confirmed_issues(machine):
    df = query_df(
        "SELECT COUNT(*) AS n FROM maintenance_tickets WHERE machine_id = ? "
        "AND confirmed_issue IS NOT NULL AND TRIM(confirmed_issue) != ''",
        (machine,),
    )
    return int(df["n"].iloc[0])
