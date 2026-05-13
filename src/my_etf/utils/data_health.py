"""Data quality checks and universe snapshots for ETF workflows."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from ..config import ALL_FEATURE_COLS, BACKTEST_CONFIG, DATA_DIR, DB_PATH


BASE_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']


def _table_name_to_code(table_name: str) -> str:
    return table_name.replace('etf_', '', 1)


def _is_official_code(code: str) -> bool:
    return code.isdigit() and len(code) == 6


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _existing_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({_quote_identifier(table_name)})").fetchall()
    return [row[1] for row in rows]


def _get_etf_tables(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'etf_%' ORDER BY name"
    ).fetchall()
    return [row[0] for row in rows]


def _reference_date(conn: sqlite3.Connection, tables: Iterable[str]) -> Optional[pd.Timestamp]:
    latest_dates = []
    for table_name in tables:
        code = _table_name_to_code(table_name)
        if not _is_official_code(code):
            continue
        try:
            value = conn.execute(
                f"SELECT MAX(date) FROM {_quote_identifier(table_name)}"
            ).fetchone()[0]
        except sqlite3.Error:
            continue
        if value:
            latest_dates.append(pd.to_datetime(value))
    return max(latest_dates) if latest_dates else None


def collect_data_health(
    codes: Optional[Iterable[str]] = None,
    db_path: str = DB_PATH,
    required_feature_cols: Optional[List[str]] = None,
    max_staleness_days: Optional[int] = None,
    min_history_days: Optional[int] = None,
) -> pd.DataFrame:
    """Return one health row per ETF table, including exclusion reason."""
    required_feature_cols = required_feature_cols or ALL_FEATURE_COLS
    max_staleness_days = (
        BACKTEST_CONFIG['MAX_STALENESS_DAYS']
        if max_staleness_days is None else max_staleness_days
    )
    min_history_days = (
        BACKTEST_CONFIG['MIN_HISTORY_DAYS']
        if min_history_days is None else min_history_days
    )

    wanted_codes = set(codes) if codes is not None else None
    rows = []
    conn = sqlite3.connect(db_path)
    try:
        tables = _get_etf_tables(conn)
        ref_date = _reference_date(conn, tables)

        for table_name in tables:
            code = _table_name_to_code(table_name)
            if wanted_codes is not None and code not in wanted_codes:
                continue

            official = _is_official_code(code)
            reason_parts = []
            if not official:
                reason_parts.append('unofficial_table')

            columns = _existing_columns(conn, table_name)
            missing_features = [
                col for col in required_feature_cols
                if col not in columns and col not in BASE_COLUMNS
            ]
            missing_feature_ratio = (
                len(missing_features) / len(required_feature_cols)
                if required_feature_cols else 0.0
            )

            try:
                count, min_date, max_date = conn.execute(
                    f"SELECT COUNT(*), MIN(date), MAX(date) FROM {_quote_identifier(table_name)}"
                ).fetchone()
            except sqlite3.Error as exc:
                rows.append({
                    'code': code,
                    'table_name': table_name,
                    'is_official': official,
                    'is_eligible': False,
                    'exclude_reason': f'read_error:{exc}',
                })
                continue

            min_ts = pd.to_datetime(min_date) if min_date else pd.NaT
            max_ts = pd.to_datetime(max_date) if max_date else pd.NaT
            staleness_days = (
                int((ref_date - max_ts).days)
                if ref_date is not None and not pd.isna(max_ts) else None
            )
            calendar_days = (
                int((max_ts - min_ts).days) + 1
                if not pd.isna(min_ts) and not pd.isna(max_ts) else 0
            )
            missing_calendar_days = max(calendar_days - int(count or 0), 0)

            if count < min_history_days:
                reason_parts.append('insufficient_history')
            if staleness_days is not None and staleness_days > max_staleness_days:
                reason_parts.append('stale_data')
            if missing_feature_ratio > 0:
                reason_parts.append('missing_features')

            abnormal_price_count = 0
            abnormal_amount_count = 0
            if 'close' in columns:
                abnormal_price_count = conn.execute(
                    f"SELECT COUNT(*) FROM {_quote_identifier(table_name)} WHERE close <= 0"
                ).fetchone()[0]
            if 'amount' in columns:
                abnormal_amount_count = conn.execute(
                    f"SELECT COUNT(*) FROM {_quote_identifier(table_name)} WHERE amount < 0"
                ).fetchone()[0]
            if abnormal_price_count:
                reason_parts.append('abnormal_price')
            if abnormal_amount_count:
                reason_parts.append('abnormal_amount')

            rows.append({
                'code': code,
                'table_name': table_name,
                'is_official': official,
                'rows': int(count or 0),
                'min_date': min_date,
                'max_date': max_date,
                'reference_date': ref_date.strftime('%Y-%m-%d') if ref_date is not None else None,
                'staleness_days': staleness_days,
                'missing_calendar_days': missing_calendar_days,
                'missing_feature_count': len(missing_features),
                'missing_feature_ratio': round(missing_feature_ratio, 4),
                'abnormal_price_count': int(abnormal_price_count),
                'abnormal_amount_count': int(abnormal_amount_count),
                'is_eligible': len(reason_parts) == 0,
                'exclude_reason': ';'.join(reason_parts),
            })
    finally:
        conn.close()

    return pd.DataFrame(rows)


def eligible_universe(
    codes: Optional[Iterable[str]] = None,
    db_path: str = DB_PATH,
    required_feature_cols: Optional[List[str]] = None,
    max_staleness_days: Optional[int] = None,
    min_history_days: Optional[int] = None,
) -> Tuple[List[str], pd.DataFrame]:
    """Return eligible ETF codes and the full health table."""
    health_df = collect_data_health(
        codes=codes,
        db_path=db_path,
        required_feature_cols=required_feature_cols,
        max_staleness_days=max_staleness_days,
        min_history_days=min_history_days,
    )
    if health_df.empty:
        return [], health_df
    eligible = health_df[health_df['is_eligible']]['code'].tolist()
    return eligible, health_df


def write_universe_snapshot(
    health_df: pd.DataFrame,
    workflow_name: str,
    output_dir: str = DATA_DIR,
) -> str:
    """Persist the exact ETF universe and exclusion reasons for a workflow."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(output_dir, f'universe_snapshot_{workflow_name}_{timestamp}.json')
    payload = {
        'workflow': workflow_name,
        'created_at': datetime.now().isoformat(),
        'eligible_count': int(health_df['is_eligible'].sum()) if not health_df.empty else 0,
        'total_count': int(len(health_df)),
        'items': health_df.to_dict(orient='records'),
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return path
