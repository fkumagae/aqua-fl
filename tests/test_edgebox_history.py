"""History payload used by the technical dashboard."""

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fluxos.edgebox.edgebox_node import read_dashboard_history


class DashboardHistoryTests(unittest.TestCase):
    def test_hour_day_week_and_time_labels(self):
        now = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory(prefix="aquafl_dashboard_history_") as temp_name:
            path = Path(temp_name) / "metrics.jsonl"
            with path.open("w", encoding="utf-8") as output:
                for days, minutes, cpu in (
                    (8, 0, 10), (2, 0, 20), (0, 720, 30),
                    (0, 45, 40), (0, 10, 50), (0, 1, 60),
                ):
                    stamp = now - timedelta(days=days, minutes=minutes)
                    sample = {
                        "timestamp": stamp.isoformat().replace("+00:00", "Z"),
                        "cpu_percent": cpu,
                        "ram_percent": 30,
                        "temperature_c": 50,
                        "inferred_risk": 0.2,
                    }
                    output.write(json.dumps(sample) + "\n")
                output.write("invalid JSON\n")

            history = read_dashboard_history(path, now=now)
            self.assertEqual([point["cpu_percent"] for point in history["samples"]], [40, 50, 60])
            self.assertEqual(len(history["today"]), 4)
            self.assertTrue(any(point["cpu_percent"] == 20 for point in history["week"]))
            self.assertFalse(any(point["cpu_percent"] == 10 for point in history["week"]))
            self.assertIsNotNone(datetime.fromisoformat(history["samples"][0]["timestamp"]).tzinfo)

    def test_missing_file_gives_empty_periods(self):
        with tempfile.TemporaryDirectory(prefix="aquafl_dashboard_history_") as temp_name:
            history = read_dashboard_history(Path(temp_name) / "missing.jsonl")
        for key in ("samples", "today", "week"):
            self.assertEqual(history[key], [])

    def test_indexed_database_and_latest_unsynced_sample(self):
        now = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc).astimezone()
        with tempfile.TemporaryDirectory(prefix="aquafl_dashboard_history_") as temp_name:
            database_path = Path(temp_name) / "edgebox.db"
            connection = sqlite3.connect(database_path)
            connection.execute(
                "CREATE TABLE metrics (timestamp TEXT UNIQUE, cpu_percent REAL, "
                "ram_percent REAL, temperature_c REAL, inferred_risk REAL)"
            )
            for days, minutes, cpu in ((2, 0, 20), (0, 720, 30), (0, 45, 40), (0, 10, 50)):
                timestamp = (now - timedelta(days=days, minutes=minutes)).replace(tzinfo=None)
                connection.execute(
                    "INSERT INTO metrics VALUES (?, ?, ?, ?, ?)",
                    (timestamp.isoformat(timespec="seconds"), cpu, 30, 50, 0.2),
                )
            connection.commit()
            connection.close()

            latest_sample = {
                "timestamp": (now - timedelta(minutes=1)).replace(tzinfo=None).isoformat(timespec="seconds"),
                "cpu_percent": 60,
                "ram_percent": 30,
                "temperature_c": 50,
                "inferred_risk": 0.2,
            }
            history = read_dashboard_history(
                now=now, database_path=database_path, latest_sample=latest_sample,
            )
            self.assertEqual([point["cpu_percent"] for point in history["samples"]], [40, 50, 60])
            self.assertEqual(len(history["today"]), 3)
            self.assertTrue(any(point["cpu_percent"] == 20 for point in history["week"]))
            self.assertIsNotNone(datetime.fromisoformat(history["today"][0]["timestamp"]).tzinfo)


if __name__ == "__main__":
    unittest.main()
