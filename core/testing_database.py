"""SQLite persistence for offline DRS testing jobs."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_jobs (
    id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    options_json TEXT NOT NULL,
    video_a_path TEXT NOT NULL,
    video_b_path TEXT,
    created_at_ms REAL NOT NULL,
    updated_at_ms REAL NOT NULL,
    result_json TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS tracking_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    camera_id INTEGER NOT NULL,
    frame_id INTEGER NOT NULL,
    timestamp_ms REAL NOT NULL,
    x REAL,
    y REAL,
    confidence REAL NOT NULL,
    predicted INTEGER NOT NULL,
    FOREIGN KEY(job_id) REFERENCES analysis_jobs(id)
);

CREATE INDEX IF NOT EXISTS idx_tracking_job_frame
ON tracking_points(job_id, camera_id, frame_id);
"""


class TestingDatabase:
    """Small local database for uploaded delivery tests and DRS outputs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def create_job(self, job_id: str, mode: str, options: dict[str, Any], video_a: Path, video_b: Path | None) -> None:
        now = time.time() * 1000.0
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO analysis_jobs
                (id, mode, status, options_json, video_a_path, video_b_path, created_at_ms, updated_at_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, mode, "queued", json.dumps(options), str(video_a), str(video_b) if video_b else None, now, now),
            )

    def update_job(self, job_id: str, status: str, result: dict[str, Any] | None = None, error: str | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE analysis_jobs
                SET status = ?, result_json = COALESCE(?, result_json), error = ?, updated_at_ms = ?
                WHERE id = ?
                """,
                (status, json.dumps(result) if result is not None else None, error, time.time() * 1000.0, job_id),
            )

    def insert_tracking(self, job_id: str, points: list[dict[str, Any]]) -> None:
        if not points:
            return
        rows = [
            (
                job_id,
                int(point.get("camera_id", 0)),
                int(point.get("frame_id", 0)),
                float(point.get("timestamp_ms", 0.0)),
                point.get("x"),
                point.get("y"),
                float(point.get("confidence", 0.0)),
                1 if point.get("predicted") else 0,
            )
            for point in points
        ]
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO tracking_points
                (job_id, camera_id, frame_id, timestamp_ms, x, y, confidence, predicted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM analysis_jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["options"] = json.loads(item.pop("options_json"))
        item["result"] = json.loads(item["result_json"]) if item.get("result_json") else None
        return item
