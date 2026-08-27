import sqlite3
from typing import List
from pathlib import Path
from app.schemas import MetricsRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rolling_metrics (
  ts_utc TEXT NOT NULL,
  window_sec INTEGER NOT NULL,
  n INTEGER NOT NULL,
  mean REAL NOT NULL,
  stdev REAL NOT NULL,
  drift_risk TEXT NOT NULL,
  source TEXT NOT NULL,
  request_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_metrics_ts ON rolling_metrics(ts_utc DESC);
"""

class SqliteMetricsStore:
    def __init__(self, path: str = "rolling_store.db") -> None:
        self.path = Path(path)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path.as_posix(), check_same_thread=False)

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as cx:
            cx.executescript(_SCHEMA)

    def save(self, record: MetricsRecord) -> None:
        self.init()
        with self._conn() as cx:
            cx.execute(
                """INSERT INTO rolling_metrics
                (ts_utc, window_sec, n, mean, stdev, drift_risk, source, request_id)
                VALUES (?,?,?,?,?,?,?,?)""",
                (
                    record.ts_utc.isoformat(),
                    record.window_sec,
                    record.n,
                    record.mean,
                    record.stdev,
                    record.drift_risk,
                    record.source,
                    record.request_id,
                ),
            )

    def read_latest(self, limit: int = 100) -> List[MetricsRecord]:
        self.init()
        query_limit = -1 if limit <= 0 else limit
        with self._conn() as cx:
            cur = cx.execute(
                """SELECT ts_utc, window_sec, n, mean, stdev, drift_risk, source, request_id
                   FROM rolling_metrics
                   ORDER BY ts_utc DESC
                   LIMIT ?""",
                (query_limit,),
            )
            rows = cur.fetchall()
        from datetime import datetime as _dt
        return [
            MetricsRecord(
                ts_utc=_dt.fromisoformat(r[0]),
                window_sec=int(r[1]),
                n=int(r[2]),
                mean=float(r[3]),
                stdev=float(r[4]),
                drift_risk=r[5],  # type: ignore
                source=r[6],
                request_id=r[7],
            )
            for r in rows
        ]
