"""
HealthLens AI — Database Layer
SQLite-based storage for reports and history.
"""

import sqlite3
import json
import os
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.getenv("DB_PATH", "healthlens.db")
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    file_id     TEXT PRIMARY KEY,
                    filename    TEXT NOT NULL,
                    upload_time TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    doc_type    TEXT,
                    patient_name TEXT,
                    doc_date    TEXT,
                    created_at  TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_reports_upload_time
                ON reports(upload_time DESC)
            """)
            conn.commit()

    def save_report(self, file_id: str, result: dict):
        try:
            # Extract quick-access fields
            extraction = result.get("extraction") or {}
            patient = extraction.get("patient") or {}
            doc_type = extraction.get("document_type", "other")
            patient_name = patient.get("name")
            doc_date = extraction.get("document_date")

            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO reports
                        (file_id, filename, upload_time, result_json, doc_type, patient_name, doc_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_id,
                        result.get("filename", "unknown"),
                        result.get("upload_time", datetime.now().isoformat()),
                        json.dumps(result),
                        doc_type,
                        patient_name,
                        doc_date,
                    )
                )
                conn.commit()
        except Exception as e:
            logger.error(f"DB save error: {e}")

    def get_report(self, file_id: str) -> dict | None:
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT result_json FROM reports WHERE file_id = ?",
                    (file_id,)
                ).fetchone()
                if row:
                    return json.loads(row["result_json"])
        except Exception as e:
            logger.error(f"DB get error: {e}")
        return None

    def get_all_reports(self, limit: int = 50) -> list:
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    """
                    SELECT file_id, filename, upload_time, doc_type, patient_name, doc_date
                    FROM reports
                    ORDER BY upload_time DESC
                    LIMIT ?
                    """,
                    (limit,)
                ).fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"DB list error: {e}")
        return []

    def get_previous_report(self, patient_name: str | None, doc_type: str, exclude_file_id: str) -> dict | None:
        """Fetch the most recent report for the same patient and document type, for comparison."""
        if not patient_name:
            return None
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    """
                    SELECT result_json FROM reports
                    WHERE patient_name = ?
                      AND doc_type = ?
                      AND file_id != ?
                    ORDER BY upload_time DESC
                    LIMIT 1
                    """,
                    (patient_name, doc_type, exclude_file_id)
                ).fetchone()
                if row:
                    return json.loads(row["result_json"])
        except Exception as e:
            logger.error(f"DB previous report error: {e}")
        return None

    def delete_report(self, file_id: str):
        try:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM reports WHERE file_id = ?", (file_id,))
                conn.commit()
        except Exception as e:
            logger.error(f"DB delete error: {e}")
