import sqlite3
from pathlib import Path

import pandas as pd

from my_etf.utils.data_health import collect_data_health, eligible_universe


def _write_table(conn, table_name, start, periods, close=10.0):
    dates = pd.date_range(start=start, periods=periods, freq="D")
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": 10000,
        "amount": 100000,
        "MA5": close,
    })
    df.to_sql(table_name, conn, index=False)


def test_collect_data_health_excludes_unofficial_stale_and_short_tables(tmp_path: Path):
    db_path = tmp_path / "health.db"
    conn = sqlite3.connect(db_path)
    try:
        _write_table(conn, "etf_510300", "2026-01-01", 30)
        _write_table(conn, "etf_512780", "2025-01-01", 30)
        _write_table(conn, "etf_159189", "2026-01-20", 5)
        _write_table(conn, "etf_512780_test", "2026-01-01", 30)
    finally:
        conn.close()

    health = collect_data_health(
        db_path=str(db_path),
        required_feature_cols=["MA5"],
        max_staleness_days=10,
        min_history_days=20,
    )

    by_code = health.set_index("code")
    assert by_code.loc["510300", "is_eligible"] == True
    assert "stale_data" in by_code.loc["512780", "exclude_reason"]
    assert "insufficient_history" in by_code.loc["159189", "exclude_reason"]
    assert "unofficial_table" in by_code.loc["512780_test", "exclude_reason"]

    eligible, _ = eligible_universe(
        db_path=str(db_path),
        required_feature_cols=["MA5"],
        max_staleness_days=10,
        min_history_days=20,
    )
    assert eligible == ["510300"]
