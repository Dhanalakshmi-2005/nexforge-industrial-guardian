"""SQLite helpers for NEXFORGE."""
import os
import sqlite3
from contextlib import closing

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "nexforge.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS machines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_name TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL DEFAULT 'Running'
);
CREATE TABLE IF NOT EXISTS sensor_readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    temperature REAL, vibration REAL, pressure REAL, rpm REAL, current REAL
);
CREATE TABLE IF NOT EXISTS quality_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    machine_id TEXT,
    result TEXT NOT NULL,
    defect_type TEXT,
    confidence REAL
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    machine_id TEXT NOT NULL,
    alert_type TEXT,
    severity TEXT,
    evidence TEXT,
    risk_score REAL
);
CREATE TABLE IF NOT EXISTS maintenance_tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    risk_level TEXT,
    detected_anomaly TEXT,
    evidence TEXT,
    recommendation TEXT,
    status TEXT NOT NULL DEFAULT 'Open',
    confirmed_issue TEXT,
    action_taken TEXT,
    downtime_min REAL DEFAULT 0,
    outcome TEXT,
    notes TEXT
);
"""


def get_connection():
    os.makedirs(DB_DIR, exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    with closing(get_connection()) as conn:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT OR IGNORE INTO machines (machine_name, status) VALUES (?, 'Running')",
            [("Machine A",), ("Machine B",), ("Machine C",)],
        )
        conn.commit()


def execute(sql, params=()):
    with closing(get_connection()) as conn:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.lastrowid


def executemany(sql, rows):
    with closing(get_connection()) as conn:
        conn.executemany(sql, rows)
        conn.commit()


def query_df(sql, params=()):
    with closing(get_connection()) as conn:
        return pd.read_sql_query(sql, conn, params=params)
