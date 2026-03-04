# -*- coding: utf-8 -*-
"""
ETF择时算子计算和数据库更新脚本
支持计算常见的技术指标（MA、MACD、KDJ、RSI、BOLL）并更新到数据库
"""
import argparse
import logging
import os
from typing import List, Optional

import pandas as pd

from ..config import DB_PATH
from ..utils.logger import setup_logger
from ..utils.constants import (
    ALL_INDICATORS,
    INDICATOR_COLUMNS,
    MA_PERIODS,
    MACD_SHORT,
    MACD_LONG,
    MACD_MID,
    KDJ_N,
    KDJ_M1,
    KDJ_M2,
    RSI_PERIOD,
    BOLL_PERIOD,
    BOLL_NUM_STD,
)

# 日志系统
logger = setup_logger("etf_indicators", "indicators.log")


# 指标计算函数

def _safe_col(df: pd.DataFrame, name_candidates: List[str]) -> Optional[str]:
    """安全获取列名，兼容不同的列名格式"""
    for c in name_candidates:
        if c in df.columns:
            return c
    return None


def calculate_ma(df: pd.DataFrame, periods: List[int] = MA_PERIODS) -> pd.DataFrame:
    """
    计算移动平均线（MA）

    Args:
        df: 包含OHLCV数据的DataFrame
        periods: MA周期列表

    Returns:
        添加了MA列的DataFrame
    """
    df_out = df.copy()
    close_col = _safe_col(df_out, ["收盘", "收盘价", "close", "Close"])

    if not close_col:
        logger.warning("未找到收盘价列，跳过MA计算")
        return df

    for period in periods:
        df_out[f'MA{period}'] = df_out[close_col].rolling(window=period, min_periods=1).mean()

    return df_out


def calculate_macd(df: pd.DataFrame, short: int = MACD_SHORT, long: int = MACD_LONG, mid: int = MACD_MID) -> pd.DataFrame:
    """
    计算MACD指标（指数平滑异同移动平均线）

    Args:
        df: 包含OHLCV数据的DataFrame
        short: 短周期
        long: 长周期
        mid: 信号线周期

    Returns:
        添加了dif, dea, macd列的DataFrame
    """
    df_out = df.copy()
    close_col = _safe_col(df_out, ["收盘", "收盘价", "close", "Close"])

    if not close_col:
        logger.warning("未找到收盘价列，跳过MACD计算")
        return df

    # 计算DIF, DEA, MACD
    ema_short = df_out[close_col].ewm(span=short, adjust=False).mean()
    ema_long = df_out[close_col].ewm(span=long, adjust=False).mean()
    df_out['dif'] = ema_short - ema_long
    df_out['dea'] = df_out['dif'].ewm(span=mid, adjust=False).mean()
    df_out['macd'] = (df_out['dif'] - df_out['dea']) * 2

    return df_out


def calculate_kdj(df: pd.DataFrame, n: int = KDJ_N, m1: int = KDJ_M1, m2: int = KDJ_M2) -> pd.DataFrame:
    """
    计算KDJ指标（随机指标）

    Args:
        df: 包含OHLCV数据的DataFrame
        n: RSV周期
        m1: K值平滑周期
        m2: D值平滑周期

    Returns:
        添加了k, d, j列的DataFrame
    """
    df_out = df.copy()
    close_col = _safe_col(df_out, ["收盘", "收盘价", "close", "Close"])
    high_col = _safe_col(df_out, ["最高", "最高价", "high", "High"])
    low_col = _safe_col(df_out, ["最低", "最低价", "low", "Low"])

    if not all([close_col, high_col, low_col]):
        logger.warning("未找到完整的OHLC列，跳过KDJ计算")
        return df

    # 计算RSV
    low_n = df_out[low_col].rolling(n, min_periods=1).min()
    high_n = df_out[high_col].rolling(n, min_periods=1).max()
    rsv = (df_out[close_col] - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(0)  # 处理分母为0的情况

    # 计算K, D, J
    df_out['k'] = rsv.ewm(com=m1 - 1, adjust=False).mean()
    df_out['d'] = df_out['k'].ewm(com=m2 - 1, adjust=False).mean()
    df_out['j'] = 3 * df_out['k'] - 2 * df_out['d']

    return df_out


def calculate_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """
    计算RSI指标（相对强弱指标）

    Args:
        df: 包含OHLCV数据的DataFrame
        period: RSI周期

    Returns:
        添加了rsi列的DataFrame
    """
    df_out = df.copy()
    close_col = _safe_col(df_out, ["收盘", "收盘价", "close", "Close"])

    if not close_col:
        logger.warning("未找到收盘价列，跳过RSI计算")
        return df

    # 计算价格变化
    delta = df_out[close_col].diff()

    # 分离上涨和下跌
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # 计算平均涨跌幅（使用EMA）
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    # 计算RSI
    rs = avg_gain / avg_loss.replace(0, float('inf'))
    df_out['rsi'] = 100 - (100 / (1 + rs))

    return df_out


def calculate_boll(df: pd.DataFrame, period: int = BOLL_PERIOD, num_std: float = BOLL_NUM_STD) -> pd.DataFrame:
    """
    计算布林带（BOLL）

    Args:
        df: 包含OHLCV数据的DataFrame
        period: 中轨周期
        num_std: 标准差倍数

    Returns:
        添加了boll_upper, boll_middle, boll_lower列的DataFrame
    """
    df_out = df.copy()
    close_col = _safe_col(df_out, ["收盘", "收盘价", "close", "Close"])

    if not close_col:
        logger.warning("未找到收盘价列，跳过BOLL计算")
        return df

    # 计算中轨
    df_out['boll_middle'] = df_out[close_col].rolling(window=period, min_periods=1).mean()

    # 计算标准差
    std = df_out[close_col].rolling(window=period, min_periods=1).std()

    # 计算上下轨
    df_out['boll_upper'] = df_out['boll_middle'] + (std * num_std)
    df_out['boll_lower'] = df_out['boll_middle'] - (std * num_std)

    return df_out


def calculate_all_indicators(df: pd.DataFrame, indicators_to_calc: Optional[List[str]] = None) -> pd.DataFrame:
    """
    计算所有指定的指标

    Args:
        df: 包含OHLCV数据的DataFrame
        indicators_to_calc: 要计算的指标列表，None表示计算所有指标

    Returns:
        添加了所有指标列的DataFrame
    """
    if indicators_to_calc is None:
        indicators_to_calc = ALL_INDICATORS

    df_out = df.copy()

    for indicator in indicators_to_calc:
        indicator_lower = indicator.lower()
        if indicator_lower == "ma":
            df_out = calculate_ma(df_out)
        elif indicator_lower == "macd":
            df_out = calculate_macd(df_out)
        elif indicator_lower == "kdj":
            df_out = calculate_kdj(df_out)
        elif indicator_lower == "rsi":
            df_out = calculate_rsi(df_out)
        elif indicator_lower == "boll":
            df_out = calculate_boll(df_out)
        elif indicator_lower == "all":
            # 递归调用计算所有指标
            return calculate_all_indicators(df_out, ALL_INDICATORS)
        else:
            logger.warning(f"未知指标: {indicator}，跳过")

    return df_out


# 数据库操作函数

from ..utils.database import get_all_etf_tables, table_has_column, add_indicator_columns
import sqlite3


def update_table_indicators(conn, table_name: str, df: pd.DataFrame, indicator_cols: List[str]) -> int:
    """
    更新表中的指标值

    Args:
        conn: 数据库连接
        table_name: 表名
        df: 包含指标值的DataFrame
        indicator_cols: 要更新的指标列名列表

    Returns:
        成功更新的行数
    """
    if df.empty:
        return 0

    # 检查日期列是否存在
    date_col = _safe_col(df, ["日期", "date", "Date"])
    if not date_col:
        logger.error(f"DataFrame中未找到日期列")
        return 0

    # 检查表中的日期列名
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    table_columns = [row[1] for row in cursor.fetchall()]
    table_date_col = '日期' if '日期' in table_columns else 'date'

    # 构建UPDATE语句（使用表中的实际日期列名）
    set_clause = ", ".join([f"{col} = ?" for col in indicator_cols])
    update_sql = f"UPDATE {table_name} SET {set_clause} WHERE {table_date_col} = ?"

    # 准备数据
    updated_count = 0
    cursor = conn.cursor()

    for _, row in df.iterrows():
        # 获取日期值
        date_value = row[date_col]
        if pd.isna(date_value):
            continue

        # 确保日期为字符串格式
        if isinstance(date_value, pd.Timestamp):
            date_value = date_value.strftime('%Y-%m-%d')
        else:
            date_value = str(date_value)

        # 准备参数
        params = []
        for col in indicator_cols:
            val = row.get(col, None)
            if pd.isna(val):
                params.append(None)
            else:
                params.append(float(val))
        params.append(date_value)

        try:
            cursor.execute(update_sql, params)
            if cursor.rowcount > 0:
                updated_count += 1
        except sqlite3.Error as e:
            logger.error(f"更新 {table_name} 日期 {date_value} 失败: {e}")

    return updated_count


def process_etf_table(
    etf_code: str,
    db_path: str = DB_PATH,
    indicators: Optional[List[str]] = None,
    force: bool = False,
    dry_run: bool = False
) -> bool:
    """
    处理单个ETF表：计算并更新指标

    Args:
        etf_code: ETF代码（如 510050）
        db_path: 数据库路径
        indicators: 要计算的指标列表，None表示所有指标
        force: 强制重新计算所有指标
        dry_run: 试运行模式，不实际修改数据库

    Returns:
        处理是否成功
    """
    if indicators is None:
        indicators = ALL_INDICATORS

    table_name = f"etf_{etf_code}"

    logger.info(f"{'='*60}")
    logger.info(f"处理ETF: {etf_code} (表: {table_name})")
    logger.info(f"计算指标: {', '.join(indicators)}")
    logger.info(f"强制重新计算: {force}，试运行: {dry_run}")

    if dry_run:
        logger.info(f"[DRY-RUN] 将处理 {table_name}")
        return True

    # 检查数据库和表是否存在
    if not os.path.exists(db_path):
        logger.error(f"数据库不存在: {db_path}")
        return False

    conn = sqlite3.connect(db_path)
    try:
        # 检查表是否存在
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        )
        if cursor.fetchone() is None:
            logger.error(f"表不存在: {table_name}")
            return False

        # 从数据库读取数据
        logger.info(f"从数据库读取 {table_name} 数据...")
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)

        if df.empty:
            logger.warning(f"表 {table_name} 为空，跳过")
            return False

        logger.info(f"读取到 {len(df)} 条记录")

        # 检查必要的列是否存在（支持中英文列名）
        required_col_mapping = {
            "日期": ["日期", "date", "Date"],
            "开盘": ["开盘", "open", "Open"],
            "收盘": ["收盘", "close", "Close"],
            "最高": ["最高", "high", "High"],
            "最低": ["最低", "low", "Low"]
        }
        missing_cols = []
        for col_name, candidates in required_col_mapping.items():
            if not any(c in df.columns for c in candidates):
                missing_cols.append(col_name)
        if missing_cols:
            logger.error(f"表 {table_name} 缺少必要的列: {missing_cols}")
            return False

        # 确定需要添加的列
        cols_to_add = []
        for indicator in indicators:
            indicator_lower = indicator.lower()
            if indicator_lower in INDICATOR_COLUMNS:
                for col in INDICATOR_COLUMNS[indicator_lower]:
                    if not table_has_column(conn, table_name, col):
                        cols_to_add.append(col)

        # 添加新列
        if cols_to_add:
            logger.info(f"添加新列: {', '.join(cols_to_add)}")
            add_indicator_columns(conn, table_name, cols_to_add)
        elif not force:
            logger.info("所有指标列已存在，跳过列添加")
        else:
            logger.info("强制模式：将重新计算所有指标")

        # 计算指标
        logger.info("计算指标...")
        df_with_indicators = calculate_all_indicators(df, indicators)

        # 确定要更新的列
        cols_to_update = []
        for indicator in indicators:
            indicator_lower = indicator.lower()
            if indicator_lower in INDICATOR_COLUMNS:
                cols_to_update.extend(INDICATOR_COLUMNS[indicator_lower])

        cols_to_update = list(set(cols_to_update))  # 去重

        # 更新数据
        logger.info(f"更新 {len(cols_to_update)} 个指标列...")
        updated_count = update_table_indicators(conn, table_name, df_with_indicators, cols_to_update)
        logger.info(f"成功更新 {updated_count} 条记录的指标值")

        # 提交事务
        conn.commit()
        logger.info("数据库更新已提交")

        # 统计指标列的非空值
        stats_msg = "指标统计: "
        for col in sorted(cols_to_update):
            count = df_with_indicators[col].notna().sum()
            stats_msg += f"{col}={count} "
        logger.info(stats_msg)

        return True

    except Exception as e:
        logger.error(f"处理 {table_name} 时发生错误: {e}")
        return False
    finally:
        conn.close()


# 辅助函数

def load_etf_list_from_env() -> List[str]:
    """从环境变量或.env文件加载ETF列表"""
    from ..config import get_etf_codes
    return get_etf_codes()


def process_all_etfs(
    etf_list: List[str],
    indicators: Optional[List[str]] = None,
    force: bool = False,
    dry_run: bool = False
) -> None:
    """
    批量处理ETF列表

    Args:
        etf_list: ETF代码列表
        indicators: 要计算的指标列表
        force: 强制重新计算所有指标
        dry_run: 试运行模式
    """
    if not etf_list:
        logger.warning("ETF列表为空，未执行任何操作")
        return

    logger.info(f"开始批量处理 {len(etf_list)} 个ETF")
    if dry_run:
        logger.info("[DRY-RUN] 试运行模式，不会修改数据库")

    success_count = 0
    fail_count = 0

    for etf_code in etf_list:
        try:
            from datetime import datetime
            start_time = datetime.now()
            if process_etf_table(etf_code, indicators=indicators, force=force, dry_run=dry_run):
                success_count += 1
            else:
                fail_count += 1
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"ETF {etf_code} 处理完成，耗时: {elapsed:.2f}秒")
        except Exception as e:
            fail_count += 1
            logger.error(f"ETF {etf_code} 处理失败: {e}")

    logger.info(f"{'='*60}")
    logger.info(f"批量处理完成: 成功 {success_count}，失败 {fail_count}")
    logger.info(f"{'='*60}")


# 命令行接口

def main():
    parser = argparse.ArgumentParser(
        description="ETF择时算子计算和数据库更新脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 计算所有ETF的所有指标
  python %(prog)s --all

  # 计算指定ETF的特定指标
  python %(prog)s --code 510050 --indicators ma macd

  # 计算.env列表中所有ETF的指标
  python %(prog)s --list --indicators kdj rsi boll

  # 强制重新计算指标
  python %(prog)s --code 510050 --force

  # 试运行查看将要执行的操作
  python %(prog)s --all --dry-run
        """
    )

    # ETF选择参数
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', help='处理数据库中所有ETF表')
    group.add_argument('--code', type=str, help='处理指定单个ETF代码')
    group.add_argument('--list', action='store_true', help='处理.env中配置的ETF列表')

    # 指标选择参数
    parser.add_argument(
        '--indicators',
        type=str,
        nargs='+',
        choices=ALL_INDICATORS + ['all'],
        default=['all'],
        help=f'指定要计算的指标（默认: all），可选: {", ".join(ALL_INDICATORS)}'
    )

    # 其他参数
    parser.add_argument('--force', action='store_true', help='强制重新计算所有指标')
    parser.add_argument('--dry-run', action='store_true', help='试运行模式，不实际修改数据库')
    parser.add_argument(
        '--db-path',
        type=str,
        default=DB_PATH,
        help=f'指定数据库路径（默认: {DB_PATH}）'
    )

    args = parser.parse_args()

    # 处理indicators参数
    if 'all' in [ind.lower() for ind in args.indicators]:
        indicators_to_calc = ALL_INDICATORS
    else:
        indicators_to_calc = args.indicators

    # 确定要处理的ETF列表
    if args.all:
        etf_list = get_all_etf_tables()
        etf_list = [table.replace('etf_', '') for table in etf_list]
    elif args.code:
        etf_list = [args.code]
    elif args.list:
        etf_list = load_etf_list_from_env()
    else:
        parser.print_help()
        return

    if not etf_list:
        logger.warning("没有找到需要处理的ETF")
        return

    # 执行处理
    try:
        from datetime import datetime
        start_time = datetime.now()
        process_all_etfs(etf_list, indicators=indicators_to_calc, force=args.force, dry_run=args.dry_run)
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"总耗时: {elapsed:.2f}秒")
    except Exception as e:
        logger.error(f"执行失败: {e}")
        raise


if __name__ == '__main__':
    main()
