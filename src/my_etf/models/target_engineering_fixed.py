# -*- coding: utf-8 -*-
"""
目标工程模块（修复版）
计算绝对收益率和超额收益率（相对于基准ETF）
"""
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..config import BENCHMARK_ETF
from ..utils.database import read_etf_data
from ..utils.logger import setup_logger

logger = setup_logger("target_engineering", "target_engineering.log")


# ============================================================================
# 绝对收益率目标
# ============================================================================

def create_absolute_return_target(df: pd.DataFrame, days: int = 5) -> pd.DataFrame:
    """
    创建绝对收益率目标

    Args:
        df: 包含收盘价的DataFrame
        days: 预测天数

    Returns:
        添加了week_return列的DataFrame
    """
    df = df.copy()
    close_col = 'close' if 'close' in df.columns else '收盘'

    # 计算未来5天的收益率（T+5日）
    abs_return = (df[close_col].shift(-days) - df[close_col]) / df[close_col] * 100
    df['week_return'] = abs_return

    # 移除最后5天的数据（无法计算目标）
    df = df.iloc[:-days]

    return df


# ============================================================================
# 超额收益率目标
# ============================================================================

def get_benchmark_returns(benchmark_code: str = BENCHMARK_ETF,
                         min_date: str = '2015-01-01',
                         horizon: int = 5) -> Optional[pd.Series]:
    """
    获取基准ETF的未来收益率序列

    Args:
        benchmark_code: 基准ETF代码
        min_date: 最小日期
        horizon: 预测天数

    Returns:
        包含基准收益率的Series，索引为日期
    """
    try:
        df = read_etf_data(benchmark_code, min_date=min_date)
        close_col = 'close' if 'close' in df.columns else '收盘'

        # 计算基准收益率
        benchmark_returns = (df[close_col].shift(-horizon) - df[close_col]) / df[close_col] * 100

        # 设置日期为索引
        date_col = 'date' if 'date' in df.columns else '日期'
        if date_col in df.columns:
            benchmark_returns.index = pd.to_datetime(df[date_col])

        return benchmark_returns
    except Exception as e:
        logger.error(f"获取基准 {benchmark_code} 收益率失败: {e}")
        return pd.Series(dtype=np.float64)


def create_excess_return_target(df: pd.DataFrame,
                                 benchmark_returns: Optional[pd.Series] = None,
                                 benchmark_code: str = BENCHMARK_ETF,
                                 horizon: int = 5) -> pd.DataFrame:
    """
    创建超额收益率目标（ETF收益 - 基准收益）

    Args:
        df: 包含收盘价的DataFrame
        benchmark_returns: 基准收益率Series（如果为None则自动计算）
        benchmark_code: 基准ETF代码
        horizon: 预测天数

    Returns:
        添加了excess_return列的DataFrame
    """
    df_out = df.copy()
    close_col = 'close' if 'close' in df_out.columns else '收盘'
    date_col = 'date' if 'date' in df_out.columns else '日期'

    # 确保close_col返回Series
    close_prices = df_out[close_col]
    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.iloc[:, 0]

    # 计算ETF绝对收益率
    etf_returns = (close_prices.shift(-horizon) - close_prices) / close_prices * 100
    # 确保etf_returns是Series
    if isinstance(etf_returns, pd.DataFrame):
        etf_returns = etf_returns.iloc[:, 0]

    # 获取基准收益率
    if benchmark_returns is None:
        benchmark_returns = get_benchmark_returns(benchmark_code, horizon=horizon)

    if benchmark_returns.empty:
        logger.warning(f"无法获取基准 {benchmark_code} 收益率，使用绝对收益率")
        df_out['excess_return'] = etf_returns.values
    elif date_col in df_out.columns:
        # 使用merge对齐数据
        df_dates = pd.to_datetime(df_out[date_col])
        benchmark_dates = pd.to_datetime(benchmark_returns.index)

        # 创建临时DataFrame用于对齐
        df_temp = pd.DataFrame({
            'etf_return': etf_returns.values,
            'df_date': df_dates,
            'bm_date': benchmark_dates
        })

        # 按日期合并
        merged = pd.merge(df_temp, df_temp, left_on='df_date', right_on='bm_date', how='left')

        # 填充缺失的基准收益
        merged['bm_return'] = merged['bm_return'].ffill()

        df_out['excess_return'] = merged['etf_return'].values - merged['bm_return'].values
    else:
        logger.warning(f"DataFrame中未找到日期列，无法对齐基准收益率")
        df_out['excess_return'] = etf_returns.values

    return df_out
