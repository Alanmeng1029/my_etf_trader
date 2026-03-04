"""数据库操作工具"""
import sqlite3
import pandas as pd
from typing import List, Optional
from ..config import DB_PATH, DB_MIN_DATE, FEATURE_COLS, ALL_FEATURE_COLS


def get_connection():
    """获取数据库连接"""
    return sqlite3.connect(DB_PATH)


def read_etf_data(code: str, min_date: str = None, use_advanced: bool = False) -> pd.DataFrame:
    """
    读取ETF数据

    Args:
        code: ETF代码
        min_date: 最小日期（格式：YYYY-MM-DD），默认为DB_MIN_DATE
        use_advanced: 是否读取高级指标

    Returns:
        包含ETF数据的DataFrame
    """
    if min_date is None:
        min_date = DB_MIN_DATE

    conn = get_connection()
    try:
        # 首先检查表的列名（英文或中文）
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info(etf_{code})")
        table_columns = [row[1] for row in cursor.fetchall()]

        # 判断使用中文还是英文列名
        has_chinese_cols = '日期' in table_columns

        if has_chinese_cols:
            # 使用中文列名
            # 映射中文列名到英文列名（用于select列表）
            col_mapping = {
                '日期': '日期',
                '开盘': '开盘',
                '收盘': '收盘',
                '最高': '最高',
                '最低': '最低',
                '成交量': '成交量',
                '成交额': '成交额',
            }
        else:
            # 使用英文列名
            col_mapping = {
                'date': 'date',
                'open': 'open',
                'close': 'close',
                'high': 'high',
                'low': 'low',
                'volume': 'volume',
                'amount': 'amount',
            }

        # 选择要读取的列
        if use_advanced:
            # 使用所有特征列（已包含OHLCV）
            columns_to_read = ALL_FEATURE_COLS
        else:
            columns_to_read = FEATURE_COLS

        # 构建查询：使用表中的实际列名，只选择表中存在的列
        select_cols = []
        # 首先确保包含日期列
        date_col = 'date' if 'date' in table_columns else '日期'
        select_cols.append(date_col)

        for col in columns_to_read:
            if col in col_mapping:
                actual_col = col_mapping[col]
            else:
                actual_col = col
            # 只添加表中存在的列（避免重复添加日期列）
            if actual_col in table_columns and actual_col != date_col:
                select_cols.append(actual_col)

        if len(select_cols) == 1:
            # 如果只有日期列，至少添加收盘价
            close_col = 'close' if 'close' in table_columns else '收盘'
            if close_col in table_columns:
                select_cols.append(close_col)

        cols_str = ", ".join(select_cols)
        date_col = col_mapping.get('date', col_mapping.get('日期', 'date'))
        query = f"""
            SELECT {cols_str}
            FROM etf_{code}
            WHERE {date_col} >= DATE('{min_date}')
            ORDER BY {date_col} ASC
        """

        df = pd.read_sql(query, conn)

        # 重命名中文列名为英文（所有中文列名统一转为英文）
        if has_chinese_cols:
            # 基础OHLCV列
            df = df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close',
                               '最高': 'high', '最低': 'low',
                               '成交量': 'volume', '成交额': 'amount'})
        else:
            # 英文列名 - 无需重命名
            pass
    except Exception as e:
        # 如果读取失败，记录错误并返回空DataFrame
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from .logger import setup_logger
        logger = setup_logger("database", "database.log")
        logger.warning(f"读取ETF {code} 数据失败: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()

    return df


def get_etf_price(code: str, start_date: str = '2015-01-01') -> pd.DataFrame:
    """
    获取ETF价格数据（用于回测）

    Args:
        code: ETF代码
        start_date: 起始日期

    Returns:
        包含日期和收盘价的DataFrame
    """
    conn = get_connection()
    try:
        # 检查表的列名
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info(etf_{code})")
        table_columns = [row[1] for row in cursor.fetchall()]

        # 判断使用中文还是英文列名
        date_col = '日期' if '日期' in table_columns else 'date'
        close_col = '收盘' if '收盘' in table_columns else 'close'

        query = f"""
            SELECT {date_col}, {close_col}
            FROM etf_{code}
            WHERE {date_col} >= '{start_date}'
            ORDER BY {date_col} ASC
        """
        df = pd.read_sql(query, conn)

        # 重命名中文列名为英文
        df = df.rename(columns={
            '日期': 'date',
            '收盘': 'close'
        })
    finally:
        conn.close()

    return df


def get_all_etf_tables() -> List[str]:
    """
    获取数据库中所有ETF表

    Returns:
        ETF表名列表（如 ['etf_510050', 'etf_159915']）
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'etf_%' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        return tables
    finally:
        conn.close()


def table_has_column(conn, table_name: str, column_name: str) -> bool:
    """检查表是否存在指定的列"""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return column_name in columns


def add_indicator_columns(conn, table_name: str, indicator_cols: List[str]) -> int:
    """
    添加指标列到表

    Args:
        conn: 数据库连接
        table_name: 表名
        indicator_cols: 要添加的列名列表

    Returns:
        成功添加的列数
    """
    added_count = 0
    for col in indicator_cols:
        if not table_has_column(conn, table_name, col):
            try:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} REAL")
                added_count += 1
            except sqlite3.OperationalError as e:
                print(f"  警告: 表 {table_name} 添加列 {col} 失败: {e}")
    return added_count
