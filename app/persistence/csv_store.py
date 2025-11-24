import os
import csv
from typing import List, Optional

from app.schemas import MetricsRecord


class CsvMetricsStore:
    """
    CSV persistence for Coherence Engine metrics.

    NOTE: API & UI speak Phase 2 names. Internally we still expose MetricsRecord
    to avoid touching callers; we map new<->old fields at the boundary.
    """

    HEADER_PHASE2 = [
        "ts_utc",
        "window_sec",
        "n",
        "interactionStability",
        "signalVolatility",
        "trustContinuityRiskLevel",
        "coherenceTrend",
        "source",
        "request_id",
    ]

    HEADER_LEGACY = [
        "ts_utc",
        "window_sec",
        "n",
        "mean",
        "stdev",
        "drift_risk",
        "source",
        "request_id",
    ]

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(path)

    # -------------------- Public API --------------------

    def init(self) -> None:
        """
        Ensure file exists and has Phase 2 header.
        If legacy header is found, rewrite to Phase 2 with column rename.
        """
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

        # Create empty with Phase 2 header
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            self._write_header(self.HEADER_PHASE2)
            return

        header, rows = self._read_raw()

        if header == self.HEADER_PHASE2:
            return

        if header == self.HEADER_LEGACY:
            # Upgrade legacy -> Phase 2
            upgraded = []
            for r in rows:
                # Pad/truncate to legacy length
                r = (r + [""] * len(self.HEADER_LEGACY))[: len(self.HEADER_LEGACY)]
                mapping = {
                    "ts_utc": r[0],
                    "window_sec": r[1],
                    "n": r[2],
                    "interactionStability": r[3],           # mean -> interactionStability
                    "signalVolatility": r[4],                # stdev -> signalVolatility
                    "trustContinuityRiskLevel": r[5],        # drift_risk -> trustContinuityRiskLevel
                    "coherenceTrend": "",                    # not recorded in legacy; leave blank
                    "source": r[6],
                    "request_id": r[7],
                }
                upgraded.append([mapping[k] for k in self.HEADER_PHASE2])

            with open(self.path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(self.HEADER_PHASE2)
                w.writerows(upgraded)
            return

        # Unknown header: rewrite and append best-effort rows
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.HEADER_PHASE2)
            # Try to map by position if row length matches; otherwise pad blanks
            for r in rows:
                new_r = ([""] * len(self.HEADER_PHASE2))
                # Best-effort: place ts/window/n and source if present
                if len(r) >= 1: new_r[0] = r[0]
                if len(r) >= 2: new_r[1] = r[1]
                if len(r) >= 3: new_r[2] = r[2]
                if len(r) >= 4: new_r[7] = r[3]  # guess 'source'
                if len(r) >= 5: new_r[8] = r[4]  # guess 'request_id'
                w.writerow(new_r)

    def save(self, rec: MetricsRecord) -> None:
        """
        Append a record. MetricsRecord uses legacy names; map to Phase 2 columns on write.
        - mean                -> interactionStability
        - stdev               -> signalVolatility
        - drift_risk          -> trustContinuityRiskLevel
        - (optional) rec.coherence_trend if you later add it to MetricsRecord
        """
        self.init()
        row = [
            (rec.ts_utc.isoformat() if hasattr(rec.ts_utc, "isoformat") else rec.ts_utc) if rec.ts_utc else "",
            int(rec.window_sec) if rec.window_sec is not None else "",
            int(rec.n) if rec.n is not None else "",
            float(rec.mean) if rec.mean is not None else "",
            float(rec.stdev) if rec.stdev is not None else "",
            str(rec.drift_risk) if rec.drift_risk is not None else "",
            getattr(rec, "coherence_trend", "") or "",
            str(rec.source) if rec.source is not None else "",
            str(rec.request_id) if rec.request_id is not None else "",
        ]
        with open(self.path, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(row)

    def read_latest(self, limit: int = 50) -> List[MetricsRecord]:
        """
        Returns most recent `limit` rows as MetricsRecord objects (legacy field names),
        mapping Phase 2 columns back to MetricsRecord for compatibility.
        """
        header, rows = self._read_raw()
        if not rows:
            return []

        tail = rows[-limit:] if limit > 0 else rows[:]
        out: List[MetricsRecord] = []

        if header == self.HEADER_PHASE2:
            idx = {k: i for i, k in enumerate(self.HEADER_PHASE2)}
            for r in tail:
                # Guard on row length
                r = (r + [""] * len(self.HEADER_PHASE2))[: len(self.HEADER_PHASE2)]
                out.append(
                    MetricsRecord(
                        ts_utc=r[idx["ts_utc"]] or None,
                        window_sec=int(r[idx["window_sec"]]) if r[idx["window_sec"]] else None,
                        n=int(r[idx["n"]]) if r[idx["n"]] else None,
                        mean=float(r[idx["interactionStability"]]) if r[idx["interactionStability"]] else None,
                        stdev=float(r[idx["signalVolatility"]]) if r[idx["signalVolatility"]] else None,
                        drift_risk=r[idx["trustContinuityRiskLevel"]] or None,
                        source=r[idx["source"]] or None,
                        request_id=r[idx["request_id"]] or None,
                    )
                )
            return out

        if header == self.HEADER_LEGACY:
            idx = {k: i for i, k in enumerate(self.HEADER_LEGACY)}
            for r in tail:
                r = (r + [""] * len(self.HEADER_LEGACY))[: len(self.HEADER_LEGACY)]
                out.append(
                    MetricsRecord(
                        ts_utc=r[idx["ts_utc"]] or None,
                        window_sec=int(r[idx["window_sec"]]) if r[idx["window_sec"]] else None,
                        n=int(r[idx["n"]]) if r[idx["n"]] else None,
                        mean=float(r[idx["mean"]]) if r[idx["mean"]] else None,
                        stdev=float(r[idx["stdev"]]) if r[idx["stdev"]] else None,
                        drift_risk=r[idx["drift_risk"]] or None,
                        source=r[idx["source"]] or None,
                        request_id=r[idx["request_id"]] or None,
                    )
                )
            return out

        # Unknown header: best-effort parse for ts/window/n
        for r in tail:
            r = list(r)
            out.append(
                MetricsRecord(
                    ts_utc=r[0] if len(r) > 0 else None,
                    window_sec=int(r[1]) if len(r) > 1 and r[1].isdigit() else None,
                    n=int(r[2]) if len(r) > 2 and str(r[2]).isdigit() else None,
                    mean=None,
                    stdev=None,
                    drift_risk=None,
                    source=r[3] if len(r) > 3 else None,
                    request_id=r[4] if len(r) > 4 else None,
                )
            )
        return out

    # -------------------- Internals --------------------

    def _write_header(self, header: List[str]) -> None:
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(header)

    def _read_raw(self) -> (List[str], List[List[str]]):
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            return (self.HEADER_PHASE2, [])
        with open(self.path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                return (self.HEADER_PHASE2, [])
            rows = [row for row in reader]
        return (header, rows)
