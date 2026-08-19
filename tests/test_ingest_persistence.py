import asyncio
from datetime import datetime, timezone

from app.ingest.signal_client import SignalClient
from app.persistence.csv_store import CsvMetricsStore, load_series
from app.schemas import MetricsRecord


def test_signal_client_loads_normalized_mock_fixture():
    client = SignalClient("http://unused", None, mock_path="data/mock_signals.json")

    records, meta = asyncio.run(client.fetch_summary())

    assert meta["pages_fetched"] == 1
    assert records
    assert all(0 <= record.sentinelScore <= 1 for record in records)


def test_csv_store_round_trip_and_series_loader(tmp_path):
    path = tmp_path / "rolling.csv"
    store = CsvMetricsStore(str(path))
    store.save(
        MetricsRecord(
            ts_utc=datetime.now(timezone.utc),
            window_sec=3600,
            n=2,
            mean=0.84,
            stdev=0.05,
            drift_risk="low",
            source="signal_api",
        )
    )

    records = store.read_latest()
    assert len(records) == 1
    assert records[0].mean == 0.84
    assert load_series(3600, str(path)) == [0.84]