# -*- coding: utf-8 -*-
"""
高级择时因子计算模块
提供23个高级技术指标用于增强择时能力
"""
import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from .calculator import _safe_col
from ..utils.logger import setup_logger

logger = setup_logger("advanced_indicators", "advanced_indicators.log")


# ============================================================================
# 市场波动率因子 (3个)
# ============================================================================

def calculate_volatility(df: pd.DataFrame, periods: List[int] = [20, 60]) -> pd.DataFrame:
    """
    计算历史波动率因子

    Args:
        df: 包含OHLCV数据的DataFrame
        periods: 计算周期列表

    Returns:
        添加了volatility_20d, volatility_60d列的DataFrame
    """
    df_out = df.copy()
    close_col = _safe_col(df_out, ["收盘", "收盘价", "close", "Close"])

    if not close_col:
        logger.warning("未找到收盘价列，跳过波动率计算")
        return df

    # 计算收益率
    returns = df_out[close_col].pct_change()

    # 计算不同周期的波动率（年化）
    for period in periods:
        volatility = returns.rolling(window=period, min_periods=1).std() * np.sqrt(252)
        df_out[f'volatility_{period}d'] = volatility * 100  # 转换为百分比

    return df_out


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    计算平均真实波幅 (ATR - Average True Range)

    Args:
        df: 包含OHLCV数据的DataFrame
        period: ATR周期

    Returns:
        添加了atr列的DataFrame
    """
    df_out = df.copy()
    close_col = _safe_col(df_out, ["收盘", "收盘价", "close", "Close"])
    high_col = _safe_col(df_out, ["最高", "最高价", "high", "High"])
    low_col = _safe_col(df_out, ["最低", "最低价", "low", "Low"])

    if not all([close_col, high_col, low_col]):
        logger.warning("未找到完整的OHLC列，跳过ATR计算")
        return df

    # 计算真实波幅 (True Range)
    high_low = df_out[high_col] - df_out[low_col]
    high_close = np.abs(df_out[high_col] - df_out[close_col].shift())
    low_close = np.abs(df_out[low_col] - df_out[close_col].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # 计算ATR (使用Wilders Smoothing)
    df_out['atr'] = tr.ewm(alpha=1/period, adjust=False).mean()

    return df_out


# ============================================================================
# 价格动量因子 (4个)
# ============================================================================

def calculate_momentum(df: pd.DataFrame, periods: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
    """
    计算价格动量因子

    Args:
        df: 包含OHLCV数据的DataFrame
        periods: 动量周期列表

    Returns:
        添加了momentum_5d, momentum_10d, momentum_20d, momentum_60d列的DataFrame
    """
    df_out = df.copy()
    close_col = _safe_col(df_out, ["收盘", "收盘价", "close", "Close"])

    if not close_col:
        logger.warning("未找到收盘价列，跳过动量计算")
        return df

    for period in periods:
        momentum = (df_out[close_col] - df_out[close_col].shift(period)) / df_out[close_col].shift(period) * 100
        df_out[f'momentum_{period}d'] = momentum

    return df_out


def calculate_acceleration(df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
    """
    计算加速度因子（动量的变化率，二阶导数）

    Args:
        df: 包含OHLCV数据的DataFrame
        period: 动量周期

    Returns:
        添加了acceleration列的DataFrame
    """
    df_out = df.copy()
    close_col = _safe_col(df_out, ["收盘", "收盘价", "close", "Close"])

    if not close_col:
        logger.warning("未找到收盘价列，跳过加速度计算")
        return df

    # 计算动量
    momentum = (df_out[close_col] - df_out[close_col].shift(period)) / df_out[close_col].shift(period)
    # 计算加速度（动量的变化）
    df_out['acceleration'] = momentum.diff() * 100

    return df_out


def calculate_roc(df: pd.DataFrame, periods: List[int] = [5, 20]) -> pd.DataFrame:
    """
    计算变化率 (ROC - Rate of Change)

    Args:
        df: 包含OHLCV数据的DataFrame
        periods: ROC周期列表

    Returns:
        添加了roc_5d, roc_20d列的DataFrame
    """
    df_out = df.copy()
    close_col = _safe_col(df_out, ["收盘", "收盘价", "close", "Close"])

    if not close_col:
        logger.warning("未找到收盘价列，跳过ROC计算")
        return df

    for period in periods:
        roc = ((df_out[close_col] - df_out[close_col].shift(period)) / df_out[close_col].shift(period)) * 100
        df_out[f'roc_{period}d'] = roc

    return df_out


# ============================================================================
# 成交量/流动性因子 (4个)
# ============================================================================

def calculate_volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    计算成交量比率因子

    Args:
        df: 包含OHLCV数据的DataFrame
        period: 平均成交量周期

    Returns:
        添加了volume_ratio列的DataFrame
    """
    df_out = df.copy()
    volume_col = _safe_col(df_out, ["成交量", "volume", "Volume"])

    if not volume_col:
        logger.warning("未找到成交量列，跳过成交量比率计算")
        return df

    avg_volume = df_out[volume_col].rolling(window=period, min_periods=1).mean()
    df_out['volume_ratio'] = df_out[volume_col] / avg_volume

    return df_out


def calculate_volume_surge(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    计算成交量突然放大因子（标准差倍数）

    Args:
        df: 包含OHLCV数据的DataFrame
        period: 统计周期

    Returns:
        添加了volume_surge列的DataFrame
    """
    df_out = df.copy()
    volume_col = _safe_col(df_out, ["成交量", "volume", "Volume"])

    if not volume_col:
        logger.warning("未找到成交量列，跳过成交量放大计算")
        return df

    avg_volume = df_out[volume_col].rolling(window=period, min_periods=1).mean()
    std_volume = df_out[volume_col].rolling(window=period, min_periods=1).std()

    df_out['volume_surge'] = (df_out[volume_col] - avg_volume) / std_volume.replace(0, 1)

    return df_out


def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算能量潮指标 (OBV - On Balance Volume)

    Args:
        df: 包含OHLCV数据的DataFrame

    Returns:
        添加了obv列的DataFrame
    """
    df_out = df.copy()
    close_col = _safe_col(df_out, ["收盘", "收盘价", "close", "Close"])
    volume_col = _safe_col(df_out, ["成交量", "volume", "Volume"])

    if not all([close_col, volume_col]):
        logger.warning("未找到收盘价和成交量列，跳过OBV计算")
        return df

    # 计算OBV
    price_change = df_out[close_col].diff()
    obv = (np.where(price_change > 0, df_out[volume_col],
                   np.where(price_change < 0, -df_out[volume_col], 0)))
    df_out['obv'] = obv.cumsum()

    return df_out


def calculate_ad(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算累积/派发线 (AD - Accumulation/Distribution Line)

    Args:
        df: 包含OHLCV数据的DataFrame

    Returns:
        添加了ad列的DataFrame
    """
    df_out = df.copy()
    close_col = _safe_col(df_out, ["收盘", "收盘价", "close", "Close"])
    high_col = _safe_col(df_out, ["最高", "最高价", "high", "High"])
    low_col = _safe_col(df_out, ["最低", "最低价", "low", "Low"])
    volume_col = _safe_col(df_out, ["成交量", "volume", "Volume"])

    if not all([close_col, high_col, low_col, volume_col]):
        logger.warning("未找到完整的OHLCV列，跳过AD计算")
        return df

    # 计算AD
    high_low = df_out[high_col] - df_out[low_col]
    close_low = df_out[close_col] - df_out[low_col]
    high_close = df_out[high_col] - df_out[close_col]

    clv = ((close_low - high_close) / high_low.replace(0, 1)) * df_out[volume_col]
    df_out['ad'] = clv.cumsum()

    return df_out


# ============================================================================
# 相对强弱因子 (3个)
# ============================================================================

def calculate_rsi_sma(df: pd.DataFrame, period: int = 14, sma_period: int = 5) -> pd.DataFrame:
    """
    计算RSI与其移动均值的差异

    Args:
        df: 包含OHLCV数据的DataFrame（需要已计算rsi列）
        period: RSI周期
        sma_period: RSI的SMA周期

    Returns:
        添加了rsi_sma列的DataFrame
    """
    df_out = df.copy()

    if 'rsi' not in df_out.columns:
        logger.warning("未找到rsi列，跳过rsi_sma计算")
        return df

    # 计算RSI的移动平均
    rsi_sma = df_out['rsi'].rolling(window=sma_period, min_periods=1).mean()
    df_out['rsi_sma'] = df_out['rsi'] - rsi_sma

    return df_out


def calculate_price_vs_ma(df: pd.DataFrame, periods: List[int] = [20, 60]) -> pd.DataFrame:
    """
    计算价格相对均线的位置

    Args:
        df: 包含OHLCV数据的DataFrame
        periods: MA周期列表

    Returns:
        添加了price_vs_ma20, price_vs_ma60列的DataFrame
    """
    df_out = df.copy()
    close_col = _safe_col(df_out, ["收盘", "收盘价", "close", "Close"])

    if not close_col:
        logger.warning("未找到收盘价列，跳过价格vs MA计算")
        return df

    for period in periods:
        ma_col = f'MA{period}'
        if ma_col in df_out.columns:
            df_out[f'price_vs_ma{period}'] = (df_out[close_col] - df_out[ma_col]) / df_out[ma_col] * 100
        else:
            logger.warning(f"未找到{ma_col}列，跳过计算")

    return df_out


# ============================================================================
# 布林带位置因子 (2个)
# ============================================================================

def calculate_boll_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算布林带位置因子 (价格在布林带中的位置)

    Args:
        df: 包含OHLCV数据的DataFrame（需要已计算boll_upper, boll_lower）

    Returns:
        添加了boll_position列的DataFrame
    """
    df_out = df.copy()
    close_col = _safe_col(df_out, ["收盘", "收盘价", "close", "Close"])

    if not all(col in df_out.columns for col in ['boll_upper', 'boll_lower']):
        logger.warning("未找到布林带列，跳过boll_position计算")
        return df

    # 计算(价格 - 下轨) / (上轨 - 下轨)
    boll_range = df_out['boll_upper'] - df_out['boll_lower']
    df_out['boll_position'] = (df_out[close_col] - df_out['boll_lower']) / boll_range.replace(0, 1)

    return df_out


def calculate_boll_width(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算布林带宽度百分比

    Args:
        df: 包含OHLCV数据的DataFrame（需要已计算boll_upper, boll_middle, boll_lower）

    Returns:
        添加了boll_width_pct列的DataFrame
    """
    df_out = df.copy()

    if not all(col in df_out.columns for col in ['boll_upper', 'boll_middle', 'boll_lower']):
        logger.warning("未找到布林带列，跳过boll_width计算")
        return df

    # 计算布林带宽度百分比
    boll_range = df_out['boll_upper'] - df_out['boll_lower']
    df_out['boll_width_pct'] = boll_range / df_out['boll_middle'].replace(0, 1) * 100

    return df_out


# ============================================================================
# 时间特征 (5个)
# ============================================================================

def calculate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算时间特征

    Args:
        df: 包含日期列的DataFrame

    Returns:
        添加了day_of_week, month_of_year, quarter, is_month_end, is_quarter_end列的DataFrame
    """
    df_out = df.copy()
    date_col = _safe_col(df_out, ["日期", "date", "Date"])

    if not date_col:
        logger.warning("未找到日期列，跳过时间特征计算")
        return df

    # 转换日期列
    if not pd.api.types.is_datetime64_any_dtype(df_out[date_col]):
        df_out[date_col] = pd.to_datetime(df_out[date_col])

    # 星期几 (0=周一, 6=周日)
    df_out['day_of_week'] = df_out[date_col].dt.dayofweek

    # 月份 (1-12)
    df_out['month_of_year'] = df_out[date_col].dt.month

    # 季度 (1-4)
    df_out['quarter'] = df_out[date_col].dt.quarter

    # 是否月末（最后5个交易日）
    df_out['is_month_end'] = 0
    for i in range(len(df_out)):
        if i >= len(df_out) - 5:
            df_out.iloc[i, df_out.columns.get_loc('is_month_end')] = 1

    # 是否季末（3, 6, 9, 12月月末）
    df_out['is_quarter_end'] = 0
    for i in range(len(df_out)):
        if i >= len(df_out) - 5 and df_out.iloc[i][date_col].month in [3, 6, 9, 12]:
            df_out.iloc[i, df_out.columns.get_loc('is_quarter_end')] = 1

    return df_out


# ============================================================================
# 市场环境因子 (2个)
# ============================================================================

def calculate_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算市场状态因子

    Args:
        df: 包含OHLCV数据的DataFrame（需要已计算MA20和MA60）

    Returns:
        添加了is_bullish, market_regime列的DataFrame
    """
    df_out = df.copy()

    if not all(col in df_out.columns for col in ['MA20', 'MA60']):
        logger.warning("未找到MA20或MA60列，跳过市场状态计算")
        return df

    # 是否处于上升趋势 (MA20 > MA60)
    df_out['is_bullish'] = (df_out['MA20'] > df_out['MA60']).astype(int)

    # 市场状态分类 (0=熊市, 1=震荡, 2=牛市)
    ma_diff = (df_out['MA20'] - df_out['MA60']) / df_out['MA60'] * 100
    df_out['market_regime'] = pd.cut(ma_diff,
                                     bins=[-np.inf, -2, 2, np.inf],
                                     labels=[0, 1, 2]).astype(int)

    return df_out


# ============================================================================
# 风险指标 (2个)
# ============================================================================

def calculate_drawdown(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    计算回撤因子

    Args:
        df: 包含OHLCV数据的DataFrame
        period: 回撤计算周期

    Returns:
        添加了drawdown_20d, drawdown_from_high列的DataFrame
    """
    df_out = df.copy()
    close_col = _safe_col(df_out, ["收盘", "收盘价", "close", "Close"])

    if not close_col:
        logger.warning("未找到收盘价列，跳过回撤计算")
        return df

    # 计算滚动最高点
    rolling_high = df_out[close_col].rolling(window=period, min_periods=1).max()
    df_out[f'drawdown_{period}d'] = (df_out[close_col] - rolling_high) / rolling_high * 100

    # 计算从历史最高点的回撤
    cummax = df_out[close_col].cummax()
    df_out['drawdown_from_high'] = (df_out[close_col] - cummax) / cummax * 100

    return df_out


# ============================================================================
# 汇总函数
# ============================================================================

def calculate_all_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有高级指标

    Args:
        df: 包含OHLCV数据的DataFrame

    Returns:
        添加了所有高级指标列的DataFrame
    """
    df_out = df.copy()

    # 市场波动率因子
    df_out = calculate_volatility(df_out)
    df_out = calculate_atr(df_out)

    # 价格动量因子
    df_out = calculate_momentum(df_out)
    df_out = calculate_acceleration(df_out)
    df_out = calculate_roc(df_out)

    # 成交量因子
    df_out = calculate_volume_ratio(df_out)
    df_out = calculate_volume_surge(df_out)
    df_out = calculate_obv(df_out)
    df_out = calculate_ad(df_out)

    # 相对强弱因子
    df_out = calculate_rsi_sma(df_out)
    df_out = calculate_price_vs_ma(df_out)

    # 布林带位置因子
    df_out = calculate_boll_position(df_out)
    df_out = calculate_boll_width(df_out)

    # 时间特征
    df_out = calculate_time_features(df_out)

    # 市场环境因子
    df_out = calculate_market_regime(df_out)

    # 风险指标
    df_out = calculate_drawdown(df_out)

    return df_out


# ============================================================================
# 数据库操作函数
# ============================================================================

from ..utils.database import table_has_column, add_indicator_columns
import sqlite3

ADVANCED_INDICATOR_COLUMNS = [
    # 市场波动率因子 (3个)
    'volatility_20d', 'volatility_60d', 'atr',
    # 价格动量因子 (6个)
    'momentum_5d', 'momentum_10d', 'momentum_20d', 'momentum_60d',
    'acceleration', 'roc_5d', 'roc_20d',
    # 成交量因子 (4个)
    'volume_ratio', 'volume_surge', 'obv', 'ad',
    # 相对强弱因子 (3个)
    'rsi_sma', 'price_vs_ma20', 'price_vs_ma60',
    # 布林带位置因子 (2个)
    'boll_position', 'boll_width_pct',
    # 时间特征 (5个)
    'day_of_week', 'month_of_year', 'quarter', 'is_month_end', 'is_quarter_end',
    # 市场环境因子 (2个)
    'is_bullish', 'market_regime',
    # 风险指标 (2个)
    'drawdown_20d', 'drawdown_from_high',
]


def update_advanced_indicators_to_db(conn, table_name: str, df: pd.DataFrame) -> int:
    """
    更新表中的高级指标值

    Args:
        conn: 数据库连接
        table_name: 表名
        df: 包含指标值的DataFrame

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

    # 确定需要更新的列
    cols_to_update = []
    for col in ADVANCED_INDICATOR_COLUMNS:
        if col in df.columns:
            cols_to_update.append(col)

    if not cols_to_update:
        logger.warning("没有找到任何高级指标列")
        return 0

    # 构建UPDATE语句（使用表中的实际日期列名）
    set_clause = ", ".join([f"{col} = ?" for col in cols_to_update])
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
        for col in cols_to_update:
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


def process_etf_advanced_indicators(
    etf_code: str,
    db_path: str,
    force: bool = False,
    dry_run: bool = False
) -> bool:
    """
    处理单个ETF表：计算并更新高级指标

    Args:
        etf_code: ETF代码（如 510050）
        db_path: 数据库路径
        force: 强制重新计算所有指标
        dry_run: 试运行模式

    Returns:
        处理是否成功
    """
    table_name = f"etf_{etf_code}"

    logger.info(f"{'='*60}")
    logger.info(f"处理高级指标: {etf_code} (表: {table_name})")
    logger.info(f"强制重新计算: {force}，试运行: {dry_run}")

    if dry_run:
        logger.info(f"[DRY-RUN] 将处理 {table_name}")
        return True

    # 检查数据库和表是否存在
    import os
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

        # 确定需要添加的列
        cols_to_add = []
        for col in ADVANCED_INDICATOR_COLUMNS:
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

        # 计算高级指标
        logger.info("计算高级指标...")
        df_with_indicators = calculate_all_advanced_indicators(df)

        # 更新数据
        logger.info(f"更新 {len(ADVANCED_INDICATOR_COLUMNS)} 个指标列...")
        updated_count = update_advanced_indicators_to_db(conn, table_name, df_with_indicators)
        logger.info(f"成功更新 {updated_count} 条记录的指标值")

        # 提交事务
        conn.commit()
        logger.info("数据库更新已提交")

        return True

    except Exception as e:
        logger.error(f"处理 {table_name} 时发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        conn.close()


def process_all_etfs_advanced(
    etf_list: list,
    db_path: str,
    force: bool = False,
    dry_run: bool = False
) -> None:
    """
    批量处理ETF列表的高级指标

    Args:
        etf_list: ETF代码列表
        db_path: 数据库路径
        force: 强制重新计算
        dry_run: 试运行模式
    """
    if not etf_list:
        logger.warning("ETF列表为空，未执行任何操作")
        return

    logger.info(f"开始批量处理 {len(etf_list)} 个ETF的高级指标")
    if dry_run:
        logger.info("[DRY-RUN] 试运行模式，不会修改数据库")

    success_count = 0
    fail_count = 0

    for etf_code in etf_list:
        try:
            from datetime import datetime
            start_time = datetime.now()
            if process_etf_advanced_indicators(etf_code, db_path, force=force, dry_run=dry_run):
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


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    import argparse
    from ..config import DB_PATH, get_etf_codes
    from ..utils.database import get_all_etf_tables

    parser = argparse.ArgumentParser(
        description="高级择时因子计算和数据库更新脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 计算所有ETF的高级指标
  python %(prog)s --all

  # 计算指定ETF的高级指标
  python %(prog)s --code 510050

  # 计算.env列表中所有ETF的高级指标
  python %(prog)s --list

  # 强制重新计算
  python %(prog)s --all --force

  # 试运行
  python %(prog)s --all --dry-run
        """
    )

    # ETF选择参数
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', help='处理数据库中所有ETF表')
    group.add_argument('--code', type=str, help='处理指定单个ETF代码')
    group.add_argument('--list', action='store_true', help='处理.env中配置的ETF列表')

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

    # 确定要处理的ETF列表
    if args.all:
        etf_list = get_all_etf_tables()
        etf_list = [table.replace('etf_', '') for table in etf_list]
    elif args.code:
        etf_list = [args.code]
    elif args.list:
        etf_list = get_etf_codes()
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
        process_all_etfs_advanced(etf_list, DB_PATH, force=args.force, dry_run=args.dry_run)
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"总耗时: {elapsed:.2f}秒")
    except Exception as e:
        logger.error(f"执行失败: {e}")
        raise


if __name__ == '__main__':
    main()
