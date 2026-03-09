# -*- coding: utf-8 -*-
"""
ETF预测策略回测模块
基于预测数据进行周调仓回测，计算收益率、回撤、夏普比率等指标
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
from pathlib import Path

from ..config import DATA_DIR, REPORTS_DIR, BACKTEST_CONFIG, STRATEGY_CONFIGS
from ..utils.logger import setup_logger
from ..utils.database import get_etf_price, get_connection

# -*- coding: utf-8 -*-
"""
ETF预测策略回测模块
基于预测数据进行周调仓回测，计算收益率、回撤、夏普比率等指标
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
from pathlib import Path

from ..config import DATA_DIR, REPORTS_DIR, BACKTEST_CONFIG, STRATEGY_CONFIGS
from ..utils.logger import setup_logger
from ..utils.database import get_etf_price, get_connection
def get_last_trading_day_last_week() -> str:
    """
    获取上周最后一个交易日（周五）

    Returns:
        上周最后一个交易日的字符串，格式为 'YYYY-MM-DD'
    """
    today = datetime.now().date()
    weekday = today.weekday()  # 0=周一, 1=周二, ..., 6=周日

    if weekday < 5:
        # 工作日：上周五 = 今天 - (weekday + 3) 天
        last_friday = today - timedelta(days=weekday + 3)
    else:
        # 周末（周六或周日）：上周五 = 今天 - 2 天
        last_friday = today - timedelta(days=2)

    return last_friday.strftime('%Y-%m-%d')
logger = setup_logger("etf_backtest", "backtest.log")


def load_etf_names(names_file: str) -> Dict[str, str]:
    """加载ETF中文名映射"""
    try:
        with open(names_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"警告: 未找到ETF名称文件 {names_file}")
        return {}
    except json.JSONDecodeError:
        logger.warning(f"警告: ETF名称文件格式错误 {names_file}")
        return {}


# 数据加载

def load_latest_predictions(data_dir: str = DATA_DIR,
                          backtest_start_date: str = '2025-01-01',
                          model_type: str = 'separate',
                          use_validation: bool = False) -> pd.DataFrame:
    """
    加载所有ETF的最新预测数据，并过滤：
    1. 只保留预测日期 >= backtest_start_date 的预测
    2. 排除预测开始日期晚于backtest_start_date的ETF

    Args:
        data_dir: 数据目录
        backtest_start_date: 回测起始日期
        model_type: 模型类型 ('separate', 'unified', 'classification', 'classification_validation')
        use_validation: 是否使用验证集模型（仅用于classification）

    Returns:
        回归模型: DataFrame with columns: date, code, actual_return, predicted_return
        分类模型: DataFrame with columns: date, code, actual_return, predicted_class, class_4_prob
    """
    all_predictions = []
    excluded_etfs = []

    for etf_dir in Path(data_dir).iterdir():
        if etf_dir.is_dir() and etf_dir.name not in ['summary', 'reports']:
            # 根据模型类型选择文件
            if model_type == 'separate':
                csv_files = sorted(etf_dir.glob(f"{etf_dir.name}_{etf_dir.name}_*_separate.csv"))
            elif model_type == 'classification':
                # 根据use_validation决定使用哪种模型
                if use_validation:
                    csv_files = sorted(etf_dir.glob(f"{etf_dir.name}_{etf_dir.name}_*_validation_classification.csv"))
                else:
                    csv_files = sorted(etf_dir.glob(f"{etf_dir.name}_{etf_dir.name}_*_classification.csv"))
                    # 排除validation文件
                    csv_files = [f for f in csv_files if 'validation' not in f.name]
            elif model_type == 'classification_validation':
                # 明确使用验证集模型
                csv_files = sorted(etf_dir.glob(f"{etf_dir.name}_{etf_dir.name}_*_validation_classification.csv"))
            else:  # unified
                csv_files = sorted(etf_dir.glob(f"{etf_dir.name}_*_unified.csv"))

            if csv_files:
                latest_file = csv_files[-1]
                try:
                    df = pd.read_csv(latest_file)

                    # 获取该ETF的最早预测日期
                    earliest_date = df['date'].min()

                    # 如果最早预测日期晚于回测起始日期，排除该ETF
                    if earliest_date > backtest_start_date:
                        excluded_etfs.append(etf_dir.name)
                        logger.info(f"  排除 {etf_dir.name}: 预测开始日期 {earliest_date} 晚于 {backtest_start_date}")
                        continue

                    # 只保留回测起始日期之后的预测
                    df_filtered = df[df['date'] >= backtest_start_date].copy()

                    if not df_filtered.empty:
                        all_predictions.append(df_filtered)
                        logger.info(f"  加载预测: {etf_dir.name} ({model_type}, 最早预测: {earliest_date}, {len(df_filtered)} 条有效记录)")
                except Exception as e:
                    logger.warning(f"  警告: 无法读取 {latest_file}: {e}")

    if excluded_etfs:
        logger.info(f"\n已排除 {len(excluded_etfs)} 个ETF (预测开始日期晚于 {backtest_start_date})")

    if all_predictions:
        return pd.concat(all_predictions, ignore_index=True)
    logger.warning(f"警告: 未找到任何{model_type}预测数据")
    return pd.DataFrame()


# 回测核心逻辑

def run_backtest(predictions_df: pd.DataFrame,
                db_path: str,
                backtest_start_date: str = '2025-01-01',
                top_n: int = 5,
                rebalance_days: int = 5,
                initial_capital: float = 100000,
                strategy_name: str = 'TOP5',
                transaction_rate: float = 0.0003) -> pd.DataFrame:
    """
    运行回测策略

    策略说明：
    - 从backtest_start_date开始回测
    - 每rebalance_days个交易日调仓一次
    - 选择预测收益最高的前N个ETF
    - 等权分配资金
    - 扣除交易费用（买入和卖出均收取）

    Args:
        predictions_df: 预测数据DataFrame
        db_path: 数据库路径
        backtest_start_date: 回测起始日期
        top_n: 选择ETF的数量
        rebalance_days: 调仓周期（交易日数）
        initial_capital: 初始资金
        strategy_name: 策略名称
        transaction_rate: 交易费率（如0.0003表示0.03%）

    Returns:
        回测结果DataFrame，包含每日净值、收益、持仓等信息
    """
    # 转换日期格式
    predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.strftime('%Y-%m-%d')

    # 获取所有预测日期（去重排序）
    all_dates = sorted(predictions_df['date'].unique())

    if not all_dates:
        logger.warning("没有找到预测数据")
        return pd.DataFrame()

    # 从指定的回测起始日期开始
    logger.info(f"回测起始日期: {backtest_start_date}")

    # 加载所有ETF的价格数据
    logger.info("加载价格数据...")
    price_data = {}
    for code in predictions_df['code'].unique():
        price_df = get_etf_price(code, start_date='2015-01-01')
        if not price_df.empty:
            price_df['date'] = pd.to_datetime(price_df['date']).dt.strftime('%Y-%m-%d')
            price_data[code] = price_df
            logger.info(f"  {code}: {price_df['date'].min()} ~ {price_df['date'].max()}")
        else:
            logger.info(f"  {code}: 无价格数据")

    if not price_data:
        logger.error("错误: 没有可用的价格数据")
        return pd.DataFrame()

    # 找到价格数据的最新日期范围
    all_price_dates = set()
    for df in price_data.values():
        all_price_dates.update(df['date'].unique())
    all_price_dates = sorted(list(all_price_dates))

    # 找到起始日期或之后的第一个交易日的位置
    try:
        start_idx = all_price_dates.index(backtest_start_date)
        actual_start_date = backtest_start_date
    except ValueError:
        actual_start_date = None
        for i, date in enumerate(all_price_dates):
            if date >= backtest_start_date:
                start_idx = i
                actual_start_date = date
                break

        if actual_start_date is None:
            logger.error(f"错误: 找不到 {backtest_start_date} 之后的交易日")
            logger.error(f"可用日期范围: {all_price_dates[0]} ~ {all_price_dates[-1]}")
            return pd.DataFrame()
        else:
            logger.info(f"实际回测起始日期: {actual_start_date} (第一个交易日)")

    # 回测记录
    backtest_results = []
    current_capital = initial_capital
    current_positions = {}  # {code: amount}
    last_rebalance_idx = start_idx

    # 遍历日期进行回测
    for i in range(start_idx, len(all_price_dates)):
        current_date = all_price_dates[i]

        # 检查是否需要调仓
        days_since_rebalance = i - last_rebalance_idx
        if days_since_rebalance >= rebalance_days:
            # 获取当前日期的预测数据
            day_predictions = predictions_df[predictions_df["date"] == current_date]

            if not day_predictions.empty:
                # 选择预测收益最高的前N个ETF
                top_etfs = day_predictions.nlargest(top_n, 'predicted_return')
                selected_codes = top_etfs['code'].tolist()

                # 计算当前持仓价值
                portfolio_value = current_capital
                for code, amount in current_positions.items():
                    if code in price_data:
                        price = price_data[code][price_data[code]['date'] == current_date]['close'].values
                        if len(price) > 0:
                            portfolio_value += amount * price[0]

                # 卖出当前持仓（计算费用）
                for code in list(current_positions.keys()):
                    if code in price_data:
                        price = price_data[code][price_data[code]['date'] == current_date]['close'].values
                        if len(price) > 0:
                            sell_value = current_positions[code] * price[0]
                            transaction_cost = sell_value * transaction_rate
                            current_capital += sell_value - transaction_cost
                    del current_positions[code]

                # 买入新选中的ETF（等权，计算费用）
                position_size = portfolio_value / len(selected_codes)
                for code in selected_codes:
                    if code in price_data:
                        price = price_data[code][price_data[code]['date'] == current_date]['close'].values
                        if len(price) > 0:
                            buy_value = position_size
                            transaction_cost = buy_value * transaction_rate
                            shares = (buy_value - transaction_cost) / price[0]
                            current_positions[code] = shares
                            current_capital -= buy_value

                last_rebalance_idx = i
                logger.info(f"{current_date}: 调仓 - 选中ETF: {selected_codes}")

        # 计算当日净值
        daily_value = current_capital
        for code, amount in current_positions.items():
            if code in price_data:
                price = price_data[code][price_data[code]['date'] == current_date]['close'].values
                if len(price) > 0:
                    daily_value += amount * price[0]

        backtest_results.append({
            'strategy_name': strategy_name,
            'date': current_date,
            'portfolio_value': daily_value,
            'capital': current_capital,
            'positions': list(current_positions.keys()),
            'daily_return': (daily_value / initial_capital - 1) * 100
        })

    return pd.DataFrame(backtest_results)


def run_classification_backtest(predictions_df: pd.DataFrame,
                            db_path: str,
                            backtest_start_date: str = '2025-01-01',
                            backtest_end_date: str = None,
                            top_n: int = 5,
                            rebalance_days: int = 5,
                            initial_capital: float = 100000,
                            strategy_name: str = 'CLASSIFICATION_TOP5',
                            transaction_rate: float = 0.0003) -> pd.DataFrame:
    """
    运行分类模型回测策略

    策略说明：
    - 从backtest_start_date开始回测
    - 到backtest_end_date结束（如果指定）
    - 每rebalance_days个交易日调仓一次
    - 按大幅上涨概率降序排序
    - 筛选概率>=10%的ETF
    - 排除预测类别为0（大幅下跌）的ETF
    - 最多选择top_n个ETF
    - 等权分配资金
    - 扣除交易费用（买入和卖出均收取）

    Args:
        predictions_df: 预测数据DataFrame (必须包含: date, code, actual_return, predicted_class, class_4_prob)
        注意: predicted_class值为0-3，对应：0=大幅下跌, 1=小幅下跌, 2=小幅上涨, 3=大幅上涨
        db_path: 数据库路径
        backtest_start_date: 回测起始日期
        top_n: 选择ETF的数量
        rebalance_days: 调仓周期（交易日数）
        initial_capital: 初始资金
        strategy_name: 策略名称
        transaction_rate: 交易费率（如0.0003表示0.03%）

    Returns:
        回测结果DataFrame，包含每日净值、收益、持仓等信息
    """
    # 转换日期格式
    predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.strftime('%Y-%m-%d')

    # 获取所有预测日期（去重排序）
    all_dates = sorted(predictions_df['date'].unique())

    if not all_dates:
        logger.warning("没有找到预测数据")
        return pd.DataFrame()

    # 从指定的回测起始日期开始
    logger.info(f"回测起始日期: {backtest_start_date}")

    # 加载所有ETF的价格数据
    logger.info("加载价格数据...")
    price_data = {}
    for code in predictions_df['code'].unique():
        price_df = get_etf_price(code, start_date='2015-01-01')
        if not price_df.empty:
            price_df['date'] = pd.to_datetime(price_df['date']).dt.strftime('%Y-%m-%d')
            price_data[code] = price_df
            logger.info(f"  {code}: {price_df['date'].min()} ~ {price_df['date'].max()}")
        else:
            logger.info(f"  {code}: 无价格数据")

    if not price_data:
        logger.error("错误: 没有可用的价格数据")
        return pd.DataFrame()

    # 找到价格数据的最新日期范围
    all_price_dates = set()
    for df in price_data.values():
        all_price_dates.update(df['date'].unique())
    all_price_dates = sorted(list(all_price_dates))

    # 找到起始日期或之后的第一个交易日的位置
    try:
        start_idx = all_price_dates.index(backtest_start_date)
        actual_start_date = backtest_start_date
    except ValueError:
        actual_start_date = None
        for i, date in enumerate(all_price_dates):
            if date >= backtest_start_date:
                start_idx = i
                actual_start_date = date
                break

        if actual_start_date is None:
            logger.error(f"错误: 找不到 {backtest_start_date} 之后的交易日")
            logger.error(f"可用日期范围: {all_price_dates[0]} ~ {all_price_dates[-1]}")
            return pd.DataFrame()
        else:
            logger.info(f"实际回测起始日期: {actual_start_date} (第一个交易日)")

    # 回测记录
    backtest_results = []
    current_capital = initial_capital
    current_positions = {}  # {code: amount}
    last_rebalance_idx = start_idx

    # 遍历日期进行回测
    for i in range(start_idx, len(all_price_dates)):
        current_date = all_price_dates[i]

        # 检查是否需要调仓
        days_since_rebalance = i - last_rebalance_idx
        if days_since_rebalance >= rebalance_days:
            # 获取当前日期的预测数据
            day_predictions = predictions_df[predictions_df["date"] == current_date]

            # 如果指定了结束日期且当前日期超过结束日期，停止调仓
            if backtest_end_date and current_date > backtest_end_date:
                break

            if not day_predictions.empty:
                # ETF选择逻辑：按大幅上涨概率降序排序，筛选概率>=10%的ETF，排除类别0
                min_prob_threshold = 0.1  # 10%概率阈值

                # 首先排除预测类别为0（大幅下跌）的ETF
                filtered_etfs = day_predictions[day_predictions['predicted_class'] != 0]
                excluded_count = len(day_predictions) - len(filtered_etfs)
                if excluded_count > 0:
                    logger.info(f"  - 排除预测类别0（大幅下跌）的ETF: {excluded_count}个")

                # 筛选概率>=10%的ETF
                high_prob_etfs = filtered_etfs[filtered_etfs['class_4_prob'] >= min_prob_threshold]

                # 按概率降序排序
                high_prob_etfs_sorted = high_prob_etfs.nlargest(len(high_prob_etfs), 'class_4_prob')

                # 选择：如果满足条件的ETF>=5个，选前5个；否则选全部满足条件的
                if len(high_prob_etfs_sorted) >= top_n:
                    selected_codes = high_prob_etfs_sorted.head(top_n)['code'].tolist()
                    top_etfs = high_prob_etfs_sorted.head(top_n)
                    logger.info(f"  - 概率>=10%的ETF: {len(high_prob_etfs_sorted)}个，选前{top_n}个")
                elif len(high_prob_etfs_sorted) > 0:
                    selected_codes = high_prob_etfs_sorted['code'].tolist()
                    top_etfs = high_prob_etfs_sorted
                    logger.info(f"  - 概率>=10%的ETF仅{len(high_prob_etfs_sorted)}个，全部入选")
                else:
                    # 没有满足条件的ETF，保持空仓
                    selected_codes = []
                    top_etfs = pd.DataFrame()
                    logger.info(f"  - 无概率>=10%的ETF，保持空仓")

                # 计算当前持仓价值
                portfolio_value = current_capital
                for code, amount in current_positions.items():
                    if code in price_data:
                        price = price_data[code][price_data[code]['date'] == current_date]['close'].values
                        if len(price) > 0:
                            portfolio_value += amount * price[0]

                # 卖出当前持仓（计算费用）
                for code in list(current_positions.keys()):
                    if code in price_data:
                        price = price_data[code][price_data[code]['date'] == current_date]['close'].values
                        if len(price) > 0:
                            sell_value = current_positions[code] * price[0]
                            transaction_cost = sell_value * transaction_rate
                            current_capital += sell_value - transaction_cost
                    del current_positions[code]

                # 如果没有满足条件的ETF，跳过买入操作，保持空仓
                if len(selected_codes) == 0:
                    last_rebalance_idx = i
                    continue  # 跳过后续买入逻辑，保持空仓

                # 买入新选中的ETF（等权，计算费用）
                position_size = portfolio_value / len(selected_codes)
                for code in selected_codes:
                    if code in price_data:
                        price = price_data[code][price_data[code]['date'] == current_date]['close'].values
                        if len(price) > 0:
                            buy_value = position_size
                            transaction_cost = buy_value * transaction_rate
                            shares = (buy_value - transaction_cost) / price[0]
                            current_positions[code] = shares
                            current_capital -= buy_value

                last_rebalance_idx = i
                logger.info(f"{current_date}: 调仓 - 选中ETF: {selected_codes}")

        # 如果指定了结束日期且当前日期超过结束日期，停止计算净值
        if backtest_end_date and current_date > backtest_end_date:
            break

        # 计算当日净值
        daily_value = current_capital
        for code, amount in current_positions.items():
            if code in price_data:
                price = price_data[code][price_data[code]['date'] == current_date]['close'].values
                if len(price) > 0:
                    daily_value += amount * price[0]

        backtest_results.append({
            'strategy_name': strategy_name,
            'date': current_date,
            'portfolio_value': daily_value,
            'capital': current_capital,
            'positions': list(current_positions.keys()),
            'daily_return': (daily_value / initial_capital - 1) * 100
        })

    return pd.DataFrame(backtest_results)


def run_classification_backtest_with_cash_hedge(
    predictions_df: pd.DataFrame,
    db_path: str,
    backtest_start_date: str = '2025-01-01',
    backtest_end_date: str = None,
    top_n: int = 5,
    rebalance_days: int = 5,
    initial_capital: float = 100000,
    strategy_name: str = 'CLASSIFICATION_TOP5_CASH_HEDGE',
    transaction_rate: float = 0.0003,
    hedge_etf_code: str = '510500'
) -> pd.DataFrame:
    """
    运行带现金对冲的分类模型回测策略

    策略说明：
    - 从backtest_start_date开始回测
    - 到backtest_end_date结束（如果指定）
    - 每rebalance_days个交易日调仓一次
    - 检查中证500ETF的预测类别，如果预测为类别0（大幅下跌）则保持空仓
    - 如果未触发对冲，优先选择预测为类别3（大幅上涨）的ETF
    - 如果类别3数量不足top_n，按类别3概率降序补充
    - 等权分配资金
    - 扣除交易费用（买入和卖出均收取）

    Args:
        predictions_df: 预测数据DataFrame (必须包含: date, code, actual_return, predicted_class, class_4_prob)
        注意: predicted_class值为0-3，对应：0=大幅下跌, 1=小幅下跌, 2=小幅上涨, 3=大幅上涨
        db_path: 数据库路径
        backtest_start_date: 回测起始日期
        backtest_end_date: 回测结束日期（可选）
        top_n: 选择ETF的数量
        rebalance_days: 调仓周期（交易日数）
        initial_capital: 初始资金
        strategy_name: 策略名称
        transaction_rate: 交易费率（如0.0003表示0.03%）
        hedge_etf_code: 用于触发对冲的ETF代码（默认510500中证500）

    Returns:
        回测结果DataFrame，包含每日净值、收益、持仓等信息
    """
    # 转换日期格式
    predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.strftime('%Y-%m-%d')

    # 获取所有预测日期（去重排序）
    all_dates = sorted(predictions_df['date'].unique())

    if not all_dates:
        logger.warning("没有找到预测数据")
        return pd.DataFrame()

    # 从指定的回测起始日期开始
    logger.info(f"回测起始日期: {backtest_start_date}")
    logger.info(f"现金对冲ETF: {hedge_etf_code}")
    logger.info(f"对冲条件: 预测类别0 (大幅下跌 <-5%)")

    # 加载所有ETF的价格数据
    logger.info("加载价格数据...")
    price_data = {}
    for code in predictions_df['code'].unique():
        price_df = get_etf_price(code, start_date='2015-01-01')
        if not price_df.empty:
            price_df['date'] = pd.to_datetime(price_df['date']).dt.strftime('%Y-%m-%d')
            price_data[code] = price_df
            logger.info(f"  {code}: {price_df['date'].min()} ~ {price_df['date'].max()}")
        else:
            logger.info(f"  {code}: 无价格数据")

    if not price_data:
        logger.error("错误: 没有可用的价格数据")
        return pd.DataFrame()

    # 找到价格数据的最新日期范围
    all_price_dates = set()
    for df in price_data.values():
        all_price_dates.update(df['date'].unique())
    all_price_dates = sorted(list(all_price_dates))

    # 找到起始日期或之后的第一个交易日的位置
    try:
        start_idx = all_price_dates.index(backtest_start_date)
        actual_start_date = backtest_start_date
    except ValueError:
        actual_start_date = None
        for i, date in enumerate(all_price_dates):
            if date >= backtest_start_date:
                start_idx = i
                actual_start_date = date
                break

        if actual_start_date is None:
            logger.error(f"错误: 找不到 {backtest_start_date} 之后的交易日")
            logger.error(f"可用日期范围: {all_price_dates[0]} ~ {all_price_dates[-1]}")
            return pd.DataFrame()
        else:
            logger.info(f"实际回测起始日期: {actual_start_date} (第一个交易日)")

    # 回测记录
    backtest_results = []
    current_capital = initial_capital
    current_positions = {}  # {code: amount}
    last_rebalance_idx = start_idx

    # 遍历日期进行回测
    for i in range(start_idx, len(all_price_dates)):
        current_date = all_price_dates[i]

        # 检查是否需要调仓
        days_since_rebalance = i - last_rebalance_idx
        if days_since_rebalance >= rebalance_days:
            # 获取当前日期的预测数据
            day_predictions = predictions_df[predictions_df["date"] == current_date]

            # 如果指定了结束日期且当前日期超过结束日期，停止调仓
            if backtest_end_date and current_date > backtest_end_date:
                break

            if not day_predictions.empty:
                # 检查中证500是否预测大幅下跌
                day_predictions_hedge = day_predictions[day_predictions["code"] == hedge_etf_code]

                if not day_predictions_hedge.empty and day_predictions_hedge.iloc[0]['predicted_class'] == 0:
                    # 触发现金对冲：清仓所有持仓，保持空仓
                    logger.info(f"{current_date}: 触发现金对冲 ({hedge_etf_code}预测大幅下跌)，清仓持仓")

                    # 卖出所有持仓（计算费用）
                    for code in list(current_positions.keys()):
                        if code in price_data:
                            price = price_data[code][price_data[code]['date'] == current_date]['close'].values
                            if len(price) > 0:
                                sell_value = current_positions[code] * price[0]
                                transaction_cost = sell_value * transaction_rate
                                current_capital += sell_value - transaction_cost
                        del current_positions[code]

                    last_rebalance_idx = i
                    continue  # 跳过后续买入逻辑，保持空仓

                # 未触发对冲，使用新的选择逻辑
                # ETF选择逻辑：按大幅上涨概率降序排序，筛选概率>=10%的ETF
                min_prob_threshold = 0.1  # 10%概率阈值

                # 筛选概率>=10%的ETF
                high_prob_etfs = day_predictions[day_predictions['class_4_prob'] >= min_prob_threshold]

                # 按概率降序排序
                high_prob_etfs_sorted = high_prob_etfs.nlargest(len(high_prob_etfs), 'class_4_prob')

                # 选择：如果满足条件的ETF>=5个，选前5个；否则选全部满足条件的
                if len(high_prob_etfs_sorted) >= top_n:
                    selected_codes = high_prob_etfs_sorted.head(top_n)['code'].tolist()
                    top_etfs = high_prob_etfs_sorted.head(top_n)
                    logger.info(f"  - 概率>=10%的ETF: {len(high_prob_etfs_sorted)}个，选前{top_n}个")
                elif len(high_prob_etfs_sorted) > 0:
                    selected_codes = high_prob_etfs_sorted['code'].tolist()
                    top_etfs = high_prob_etfs_sorted
                    logger.info(f"  - 概率>=10%的ETF仅{len(high_prob_etfs_sorted)}个，全部入选")
                else:
                    # 没有满足条件的ETF，保持空仓
                    selected_codes = []
                    top_etfs = pd.DataFrame()
                    logger.info(f"  - 无概率>=10%的ETF，保持空仓")

                # 计算当前持仓价值
                portfolio_value = current_capital
                for code, amount in current_positions.items():
                    if code in price_data:
                        price = price_data[code][price_data[code]['date'] == current_date]['close'].values
                        if len(price) > 0:
                            portfolio_value += amount * price[0]

                # 卖出当前持仓（计算费用）
                for code in list(current_positions.keys()):
                    if code in price_data:
                        price = price_data[code][price_data[code]['date'] == current_date]['close'].values
                        if len(price) > 0:
                            sell_value = current_positions[code] * price[0]
                            transaction_cost = sell_value * transaction_rate
                            current_capital += sell_value - transaction_cost
                    del current_positions[code]

                # 如果没有满足条件的ETF，跳过买入操作，保持空仓
                if len(selected_codes) == 0:
                    last_rebalance_idx = i
                    continue  # 跳过后续买入逻辑，保持空仓

                # 买入新选中的ETF（等权，计算费用）
                position_size = portfolio_value / len(selected_codes)
                for code in selected_codes:
                    if code in price_data:
                        price = price_data[code][price_data[code]['date'] == current_date]['close'].values
                        if len(price) > 0:
                            buy_value = position_size
                            transaction_cost = buy_value * transaction_rate
                            shares = (buy_value - transaction_cost) / price[0]
                            current_positions[code] = shares
                            current_capital -= buy_value

                last_rebalance_idx = i
                logger.info(f"{current_date}: 调仓 - 选中ETF: {selected_codes}")

        # 如果指定了结束日期且当前日期超过结束日期，停止计算净值
        if backtest_end_date and current_date > backtest_end_date:
            break

        # 计算当日净值
        daily_value = current_capital
        for code, amount in current_positions.items():
            if code in price_data:
                price = price_data[code][price_data[code]['date'] == current_date]['close'].values
                if len(price) > 0:
                    daily_value += amount * price[0]

        backtest_results.append({
            'strategy_name': strategy_name,
            'date': current_date,
            'portfolio_value': daily_value,
            'capital': current_capital,
            'positions': list(current_positions.keys()),
            'daily_return': (daily_value / initial_capital - 1) * 100
        })

    return pd.DataFrame(backtest_results)


# 指标计算

def calculate_metrics(results_df: pd.DataFrame, initial_capital: float) -> dict:
    """
    计算回测性能指标

    Returns:
        包含各种指标的字典
    """
    if len(results_df) == 0:
        return {}

    final_value = results_df['portfolio_value'].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100

    # 计算每日收益率
    results_df['daily_return_pct'] = results_df['portfolio_value'].pct_change() * 100
    daily_returns = results_df['daily_return_pct'].dropna()

    # 计算最大回撤
    cumulative = (1 + results_df['daily_return_pct'] / 100).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    max_drawdown = drawdown.min()

    # 计算年化收益率（假设252个交易日）
    num_days = len(results_df)
    annualized_return = (final_value / initial_capital) ** (252 / num_days) - 1 if num_days > 0 else 0

    # 计算夏普比率（假设无风险利率为3%）
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe_ratio = (annualized_return - 0.03) / (daily_returns.std() / 100 * np.sqrt(252))
    else:
        sharpe_ratio = 0

    # 计算胜率（每日收益为正的比例）
    win_rate = (daily_returns > 0).sum() / len(daily_returns) * 100 if len(daily_returns) > 0 else 0

    # 计算盈亏比
    avg_win = daily_returns[daily_returns > 0].mean() if (daily_returns > 0).sum() > 0 else 0
    avg_loss = daily_returns[daily_returns < 0].mean() if (daily_returns < 0).sum() > 0 else 0
    profit_loss_ratio = -avg_win / avg_loss if avg_loss < 0 else 0

    # 调仓次数
    rebalance_count = 0
    prev_positions = None
    for _, row in results_df.iterrows():
        curr_positions = tuple(sorted(row['positions']))
        if prev_positions is None or curr_positions != prev_positions:
            if curr_positions:  # 有持仓才算一次调仓
                rebalance_count += 1
            prev_positions = curr_positions

    return {
        'total_return': round(total_return, 2),
        'final_value': round(final_value, 2),
        'max_drawdown': round(max_drawdown, 2),
        'annualized_return': round(annualized_return * 100, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'win_rate': round(win_rate, 2),
        'profit_loss_ratio': round(profit_loss_ratio, 2),
        'total_trades': rebalance_count,
        'trading_days': num_days,
        'start_date': results_df['date'].iloc[0],
        'end_date': results_df['date'].iloc[-1]
    }


# 输出生成

def generate_backtest_csv(results_df: pd.DataFrame, output_path: str) -> None:
    """生成回测结果CSV"""
    results_df.to_csv(output_path, index=False)
    logger.info(f"回测结果已保存: {output_path}")


def generate_strategy_comparison_summary(all_results: list, output_path: str) -> None:
    """生成策略对比摘要CSV"""
    summary_rows = []
    for result in all_results:
        summary_rows.append({
            'Strategy': result['strategy_name'],
            'Model Type': result['metrics'].get('model_type', ''),
            'Total Return (%)': result['metrics']['total_return'],
            'Final Value': result['metrics']['final_value'],
            'Max Drawdown (%)': result['metrics']['max_drawdown'],
            'Annualized Return (%)': result['metrics']['annualized_return'],
            'Sharpe Ratio': result['metrics']['sharpe_ratio'],
            'Win Rate (%)': result['metrics']['win_rate'],
            'Rebalance Count': result['metrics']['total_trades']
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path, index=False)
    logger.info(f"策略对比摘要已保存: {output_path}")


def generate_backtest_html(results_df: pd.DataFrame, metrics: dict,
                        output_path: str, etf_names: Dict[str, str],
                        model_type: str = '') -> None:
    """生成包含净值曲线的回测HTML报告"""
    # 准备净值曲线数据
    initial_value = 100000
    nav_data = []
    return_data = []
    benchmark_return_data = []

    # 加载沪深300基准数据
    benchmark_code = '510300'
    try:
        import sqlite3
        from ..config import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        benchmark_df = pd.read_sql(f"SELECT date, close FROM etf_{benchmark_code} WHERE date >= \"{results_df['date'].min()}\" AND date <= \"{results_df['date'].max()}\" ORDER BY date", conn)
        conn.close()

        # 计算基准净值和收益率
        benchmark_value = initial_value
        benchmark_initial_close = benchmark_df['close'].iloc[0]

        for _, row in results_df.iterrows():
            nav_val = row['portfolio_value']
            cum_return = (nav_val / initial_value - 1) * 100
            nav_data.append({
                'date': row['date'],
                'value': nav_val,
                'return': cum_return
            })
            return_data.append({
                'date': row['date'],
                'value': nav_val,
                'return': cum_return
            })

            # 获取当天基准价格
            benchmark_row = benchmark_df[benchmark_df['date'] == row['date']]
            if not benchmark_row.empty:
                benchmark_close = benchmark_row['close'].iloc[0]
                benchmark_cum_return = (benchmark_close / benchmark_initial_close - 1) * 100
            else:
                # 如果当天没有数据，使用前一天的值
                benchmark_cum_return = benchmark_return_data[-1]['benchmark_return'] if benchmark_return_data else 0

            benchmark_return_data.append({
                'date': row['date'],
                'strategy_return': cum_return,
                'benchmark_return': benchmark_cum_return
            })
    except Exception as e:
        logger.warning(f"无法加载基准数据，使用模拟数据: {e}")
        # 降级到模拟基准
        benchmark_value = initial_value
        benchmark_cum_return = 0
        for i, row in results_df.iterrows():
            nav_val = row['portfolio_value']
            cum_return = (nav_val / initial_value - 1) * 100
            nav_data.append({
                'date': row['date'],
                'value': nav_val,
                'return': cum_return
            })
            return_data.append({
                'date': row['date'],
                'value': nav_val,
                'return': cum_return
            })

            # 模拟基准收益率（基于市场日均0.05%的收益率）
            if i > 0:
                daily_benchmark_return = 0.05 + (cum_return * 0.002)
                benchmark_value = benchmark_value * (1 + daily_benchmark_return / 100)
                benchmark_cum_return = (benchmark_value / initial_value - 1) * 100
            benchmark_return_data.append({
                'date': row['date'],
                'strategy_return': cum_return,
                'benchmark_return': benchmark_cum_return
            })

    # 生成CSV数据（嵌入到HTML中）
    nav_csv = '\n'.join([f"{d['date']},{d['value']},{d['return']}" for d in nav_data])
    combined_csv = '\n'.join([f"{d['date']},{d['value']},{d['return']}" for d in nav_data])

    # 累计收益率对比数据（策略 vs 沪深300）
    return_csv = '\n'.join([f"{d['date']},{d['strategy_return']:.6f},{d['benchmark_return']:.6f}" for d in benchmark_return_data])

    # 累计超额收益数据（仅对于separate模型）
    excess_return_data = []
    if model_type == 'separate':
        for d in benchmark_return_data:
            excess_return_data.append({
                'date': d['date'],
                'excess_return': d['strategy_return'] - d['benchmark_return']
            })
        excess_return_csv = '\n'.join([f"{d['date']},{d['excess_return']:.6f}" for d in excess_return_data])
    else:
        excess_return_csv = ''

    # 最终持仓
    final_positions = results_df.iloc[-1]['positions']
    if isinstance(final_positions, str):
        final_positions = eval(final_positions)
    positions_html = ""
    for code in final_positions:
        name = etf_names.get(code, code)
        positions_html += f'<div class="position-badge">{code} - {name}</div>'

    # 准备调仓记录表格
    rebalance_records = []
    prev_positions = None
    for i, row in results_df.iterrows():
        curr_positions = row['positions']
        if isinstance(curr_positions, str):
            curr_positions = eval(curr_positions)
        if i == 0 or curr_positions != prev_positions:
            rebalance_records.append(row)

        prev_positions = curr_positions

    # 只显示前20条记录
    table_html = ""
    for i, row in enumerate(rebalance_records[:20]):
        table_html += f"<tr><td>{row['date']}</td>"
        table_html += f"<td>{row['positions']}</td>"
        table_html += f"<td>CNY {row['portfolio_value']:,.2f}</td></tr>\n"

    if len(rebalance_records) > 20:
        table_html += f'<tr><td colspan="3" style="text-align: center; color: #666;">... 还有 {len(rebalance_records) - 20} 次调仓记录</td></tr>\n'

    # 格式化指标值
    total_return = metrics['total_return']
    max_drawdown = metrics['max_drawdown']
    annualized_return = metrics['annualized_return']
    sharpe_ratio = metrics['sharpe_ratio']
    win_rate = metrics['win_rate']
    profit_loss_ratio = metrics['profit_loss_ratio']

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETF预测策略回测报告</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 30px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.2em;
            margin-bottom: 10px;
            font-weight: 700;
        }}

        .header p {{
            font-size: 1em;
            opacity: 0.95;
        }}

        .params-section {{
            padding: 20px 30px;
            background: #f8f9fa;
            border-bottom: 1px solid #eee;
        }}

        .params-section h3 {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .params-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            color: #555;
            font-size: 0.95em;
        }}

        .param-item {{
            display: flex;
            align-items: center;
        }}

        .param-item::before {{
            content: "•";
            color: #667eea;
            margin-right: 8px;
            font-weight: bold;
        }}

        .summary-section {{
            padding: 30px;
        }}

        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px 20px;
            border-radius: 10px;
            text-align: center;
            transition: transform 0.3s ease;
        }}

        .card:hover {{
            transform: translateY(-5px);
        }}

        .card .number {{
            font-size: 2.2em;
            font-weight: bold;
            color: #667eea;
        }}

        .card .label {{
            color: #666;
            margin-top: 8px;
            font-size: 0.95em;
        }}

        .card.positive .number {{
            color: #2e7d32;
        }}

        .card.negative .number {{
            color: #c62828;
        }}

        .chart-section {{
            padding: 30px;
            background: #f8f9fa;
        }}

        .chart-section h2 {{
            color: #333;
            margin-bottom: 25px;
            text-align: center;
            font-size: 1.5em;
        }}

        .chart-container {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }}

        canvas {{
            width: 100%;
            height: 400px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }}

        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}

        td {{
            padding: 14px 15px;
            border-bottom: 1px solid #eee;
        }}

        tr:hover {{
            background-color: #f8f9fa;
        }}

        .table-section {{
            padding: 30px;
        }}

        .table-section h2 {{
            color: #333;
            margin-bottom: 20px;
            text-align: center;
            font-size: 1.5em;
        }}

        .footer {{
            background: #2c3e50;
            color: white;
            padding: 25px;
            text-align: center;
        }}

        .update-time {{
            font-size: 0.9em;
        }}

        .footer-note {{
            font-size: 0.85em;
            color: #95a5a6;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ETF预测策略回测报告</h1>
            <p>模型: {model_type} ({'绝对收益预测' if model_type == 'separate' else ('超额收益预测' if model_type == 'unified' else '分类预测')}) | 回测周期: {metrics.get('start_date', '')} ~ {metrics.get('end_date', '')}</p>
        </div>

        <div class="params-section">
            <h3>回测参数</h3>
            <div class="params-list">
                <div class="param-item">初始资金: CNY 100,000</div>
                <div class="param-item">调仓周期: 每5个交易日</div>
                <div class="param-item">选股数量: 5个</div>
                <div class="param-item">交易费率: 0.030%</div>
                <div class="param-item">回测期间: {metrics.get('start_date', '')} ~ {metrics.get('end_date', '')}</div>
                <div class="param-item">交易天数: {metrics['trading_days']}天</div>
            </div>
        </div>

        <div class="summary-section">
            <h2 style="text-align: center; margin-bottom: 20px; color: #333;">关键指标</h2>
            <div class="summary-cards">
                <div class="card {'positive' if total_return > 0 else 'negative'}">
                    <div class="number">{total_return:.2f}%</div>
                    <div class="label">总收益率</div>
                </div>
                <div class="card negative">
                    <div class="number">{max_drawdown:.2f}%</div>
                    <div class="label">最大回撤</div>
                </div>
                <div class="card">
                    <div class="number">CNY {metrics['final_value']:,.0f}</div>
                    <div class="label">最终净值</div>
                </div>
                <div class="card {'positive' if annualized_return > 0 else 'negative'}">
                    <div class="number">{annualized_return:.2f}%</div>
                    <div class="label">年化收益率</div>
                </div>
                <div class="card">
                    <div class="number">{sharpe_ratio:.2f}</div>
                    <div class="label">夏普比率</div>
                </div>
                <div class="card">
                    <div class="number">{win_rate:.2f}%</div>
                    <div class="label">胜率</div>
                </div>
                <div class="card">
                    <div class="number">{profit_loss_ratio:.2f}</div>
                    <div class="label">盈亏比</div>
                </div>
                <div class="card">
                    <div class="number">{metrics['total_trades']}</div>
                    <div class="label">调仓次数</div>
                </div>
            </div>
        </div>

        <div class="chart-section">
            <h2>净值与累计收益率（双Y轴）</h2>
            <div class="chart-container">
                <canvas id="combinedChart"></canvas>
            </div>
            <div style="text-align: center; margin-top: 15px; color: #666; font-size: 0.9em;">
                <span style="display: inline-block; width: 12px; height: 12px; background: rgb(102, 126, 234); margin-right: 5px;"></span>净值（左轴）
                <span style="display: inline-block; width: 12px; height: 12px; background: rgb(75, 192, 192); margin-left: 20px; margin-right: 5px;"></span>累计收益率（右轴）
            </div>
        </div>

        <div class="chart-section">
            <h2>净值曲线</h2>
            <div class="chart-container">
                <canvas id="navChart"></canvas>
            </div>
        </div>

        <div class="chart-section">
            <h2>累计收益率对比 (策略 vs 沪深300)</h2>
            <div class="chart-container">
                <canvas id="returnChart"></canvas>
            </div>
            <div style="text-align: center; margin-top: 15px; color: #666; font-size: 0.9em;">
                <span style="display: inline-block; width: 12px; height: 12px; background: rgb(102, 126, 234); margin-right: 5px;"></span>策略收益
                <span style="display: inline-block; width: 12px; height: 12px; background: rgb(255, 99, 132); margin-left: 20px; margin-right: 5px;"></span>沪深300
            </div>
        </div>

        <!-- 累计超额收益图表（仅separate模型） -->
        {f'''
        <div class="chart-section">
            <h2>累计超额收益 (策略 - 沪深300)</h2>
            <div class="chart-container">
                <canvas id="excessReturnChart"></canvas>
            </div>
            <div style="text-align: center; margin-top: 15px; color: #666; font-size: 0.9em;">
                <span style="display: inline-block; width: 12px; height: 12px; background: rgb(75, 192, 192); margin-right: 5px;"></span>超额收益（正值为跑赢，负值为跑输）
            </div>
        </div>
        ''' if model_type == 'separate' else ''}

        <div class="table-section">
            <h2>调仓记录</h2>
            <table>
                <thead>
                    <tr>
                        <th>日期</th>
                        <th>持仓ETF</th>
                        <th>净值</th>
                    </tr>
                </thead>
                <tbody>
                    {table_html}
                </tbody>
            </table>
        </div>

        <div class="footer">
            <p class="update-time">报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p class="footer-note">注: 回测基于历史数据，不代表未来表现</p>
        </div>
    </div>

    <script>
        // 净值数据
        const navCSV = `{nav_csv}`;
        // 双轴数据
        const combinedCSV = `{combined_csv}`;
        // 收益率对比数据
        const returnCSV = `{return_csv}`;
        // 超额收益数据（仅separate模型）
        {f"const excessReturnCSV = `{excess_return_csv}`;" if model_type == 'separate' else '// 超额收益数据仅separate模型提供'}

        // 解析CSV
        function parseCSV(text) {{
            const lines = text.trim().split('\\n');
            const data = [];
            for (let i = 1; i < lines.length; i++) {{
                const values = lines[i].split(',');
                data.push({{
                    date: values[0],
                    value: parseFloat(values[1]),
                    return: parseFloat(values[2])
                }});
            }}
            return data;
        }}

        // 解析收益率对比CSV
        function parseReturnCSV(text) {{
            const lines = text.trim().split('\\n');
            const data = [];
            for (let i = 0; i < lines.length; i++) {{
                const values = lines[i].split(',');
                data.push({{
                    date: values[0],
                    strategy_return: parseFloat(values[1]),
                    benchmark_return: parseFloat(values[2])
                }});
            }}
            return data;
        }}

        // 解析超额收益CSV
        function parseExcessReturnCSV(text) {{
            const lines = text.trim().split('\\n');
            const data = [];
            for (let i = 0; i < lines.length; i++) {{
                const values = lines[i].split(',');
                data.push({{
                    date: values[0],
                    excess_return: parseFloat(values[1])
                }});
            }}
            return data;
        }}

        // 绘制超额收益图表
        function drawExcessReturnChart(canvasId, data) {{
            const canvas = document.getElementById(canvasId);
            if (!canvas || !data || data.length === 0) return;
            const ctx = canvas.getContext('2d');
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width * 2;
            canvas.height = 800;
            ctx.scale(2, 2);
            const width = rect.width;
            const height = 400;

            ctx.clearRect(0, 0, width, height);

            const excessReturns = data.map(d => d.excess_return);
            const minRet = Math.min(...excessReturns);
            const maxRet = Math.max(...excessReturns);
            const returnRange = maxRet - minRet || 1;

            const padding = {{ top: 20, right: 50, bottom: 40, left: 60 }};
            const chartWidth = width - padding.left - padding.right;
            const chartHeight = height - padding.top - padding.bottom;

            // 绘制零线
            const zeroY = padding.top + chartHeight - ((0 - minRet) / returnRange) * chartHeight;
            ctx.strokeStyle = 'rgba(200, 200, 200, 0.5)';
            ctx.lineWidth = 1;
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(padding.left, zeroY);
            ctx.lineTo(width - padding.right, zeroY);
            ctx.stroke();
            ctx.setLineDash([]);

            // 网格线
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.05)';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 5; i++) {{
                const y = padding.top + (chartHeight / 5) * i;
                ctx.beginPath();
                ctx.moveTo(padding.left, y);
                ctx.lineTo(width - padding.right, y);
                ctx.stroke();

                const ret = maxRet - (returnRange / 5) * i;
                ctx.fillStyle = '#666';
                ctx.font = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", Tahoma, Geneva, Verdana, sans-serif';
                ctx.textAlign = 'right';
                ctx.fillText(ret.toFixed(1) + '%', padding.left - 8, y + 4);
            }};

            // 绘制超额收益曲线（青色）
            ctx.strokeStyle = 'rgb(75, 192, 192)';
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            data.forEach((d, i) => {{
                const x = padding.left + (chartWidth / (data.length - 1)) * i;
                const y = padding.top + chartHeight - ((d.excess_return - minRet) / returnRange) * chartHeight;
                if (i === 0) {{
                    ctx.moveTo(x, y);
                }} else {{
                    ctx.lineTo(x, y);
                }}
            }});
            ctx.stroke();

            // X轴标签
            ctx.fillStyle = '#666';
            ctx.textAlign = 'center';
            const step = Math.ceil(data.length / 10);
            data.forEach((d, i) => {{
                if (i % step === 0 || i === data.length - 1) {{
                    const x = padding.left + (chartWidth / (data.length - 1)) * i;
                    const date = d.date.slice(5);
                    ctx.fillText(date, x, height - padding.bottom + 15);
                }}
            }});
        }}

        // 绘制双Y轴图表（净值 + 累计收益率）
        function drawCombinedChart(canvasId, data) {{
            const canvas = document.getElementById(canvasId);
            if (!canvas || !data || data.length === 0) return;
            const ctx = canvas.getContext('2d');
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width * 2;
            canvas.height = 800;
            ctx.scale(2, 2);
            const width = rect.width;
            const height = 400;

            ctx.clearRect(0, 0, width, height);

            const values = data.map(d => d.value);
            const returns = data.map(d => d.return);
            const minVal = Math.min(...values);
            const maxVal = Math.max(...values);
            const minRet = Math.min(...returns);
            const maxRet = Math.max(...returns);
            const valueRange = maxVal - minVal || 1;
            const returnRange = maxRet - minRet || 1;

            const padding = {{ top: 20, right: 80, bottom: 40, left: 80 }};
            const chartWidth = width - padding.left - padding.right;
            const chartHeight = height - padding.top - padding.bottom;

            // 网格线
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.05)';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 5; i++) {{
                const y = padding.top + (chartHeight / 5) * i;
                ctx.beginPath();
                ctx.moveTo(padding.left, y);
                ctx.lineTo(width - padding.right, y);
                ctx.stroke();
            }}

            // 左Y轴标签（净值）
            ctx.fillStyle = '#666';
            ctx.font = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", Tahoma, Geneva, Verdana, sans-serif';
            ctx.textAlign = 'right';
            for (let i = 0; i <= 5; i++) {{
                const y = padding.top + (chartHeight / 5) * i;
                const value = maxVal - (valueRange / 5) * i;
                ctx.fillText('CNY ' + value.toFixed(0), padding.left - 8, y + 4);
            }}

            // 右Y轴标签（收益率）
            ctx.textAlign = 'left';
            for (let i = 0; i <= 5; i++) {{
                const y = padding.top + (chartHeight / 5) * i;
                const ret = minRet + (returnRange / 5) * (5 - i);
                ctx.fillText(ret.toFixed(1) + '%', width - padding.right + 8, y + 4);
            }}

            // 绘制净值曲线（蓝色）
            ctx.strokeStyle = 'rgb(102, 126, 234)';
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            data.forEach((d, i) => {{
                const x = padding.left + (chartWidth / (data.length - 1)) * i;
                const y = padding.top + chartHeight - ((d.value - minVal) / valueRange) * chartHeight;
                if (i === 0) {{
                    ctx.moveTo(x, y);
                }} else {{
                    ctx.lineTo(x, y);
                }}
            }});
            ctx.stroke();

            // 绘制收益率曲线（青色）
            ctx.strokeStyle = 'rgb(75, 192, 192)';
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            data.forEach((d, i) => {{
                const x = padding.left + (chartWidth / (data.length - 1)) * i;
                const y = padding.top + chartHeight - ((d.return - minRet) / returnRange) * chartHeight;
                if (i === 0) {{
                    ctx.moveTo(x, y);
                }} else {{
                    ctx.lineTo(x, y);
                }}
            }});
            ctx.stroke();

            // X轴标签
            ctx.fillStyle = '#666';
            ctx.textAlign = 'center';
            const step = Math.ceil(data.length / 10);
            data.forEach((d, i) => {{
                if (i % step === 0 || i === data.length - 1) {{
                    const x = padding.left + (chartWidth / (data.length - 1)) * i;
                    const date = d.date.slice(5);
                    ctx.fillText(date, x, height - padding.bottom + 15);
                }}
            }});
        }}

        // 绘制收益率对比图表
        function drawReturnChart(canvasId, data) {{
            const canvas = document.getElementById(canvasId);
            if (!canvas || !data || data.length === 0) return;
            const ctx = canvas.getContext('2d');
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width * 2;
            canvas.height = 800;
            ctx.scale(2, 2);
            const width = rect.width;
            const height = 400;

            ctx.clearRect(0, 0, width, height);

            const allReturns = [...data.map(d => d.strategy_return), ...data.map(d => d.benchmark_return)];
            const minRet = Math.min(...allReturns);
            const maxRet = Math.max(...allReturns);
            const returnRange = maxRet - minRet || 1;

            const padding = {{ top: 20, right: 50, bottom: 40, left: 60 }};
            const chartWidth = width - padding.left - padding.right;
            const chartHeight = height - padding.top - padding.bottom;

            // 网格线
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.05)';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 5; i++) {{
                const y = padding.top + (chartHeight / 5) * i;
                ctx.beginPath();
                ctx.moveTo(padding.left, y);
                ctx.lineTo(width - padding.right, y);
                ctx.stroke();

                const ret = maxRet - (returnRange / 5) * i;
                ctx.fillStyle = '#666';
                ctx.font = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", Tahoma, Geneva, Verdana, sans-serif';
                ctx.textAlign = 'right';
                ctx.fillText(ret.toFixed(1) + '%', padding.left - 8, y + 4);
            }};

            // 绘制策略收益曲线（蓝色）
            ctx.strokeStyle = 'rgb(102, 126, 234)';
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            data.forEach((d, i) => {{
                const x = padding.left + (chartWidth / (data.length - 1)) * i;
                const y = padding.top + chartHeight - ((d.strategy_return - minRet) / returnRange) * chartHeight;
                if (i === 0) {{
                    ctx.moveTo(x, y);
                }} else {{
                    ctx.lineTo(x, y);
                }}
            }});
            ctx.stroke();

            // 绘制基准收益曲线（红色）
            ctx.strokeStyle = 'rgb(255, 99, 132)';
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            data.forEach((d, i) => {{
                const x = padding.left + (chartWidth / (data.length - 1)) * i;
                const y = padding.top + chartHeight - ((d.benchmark_return - minRet) / returnRange) * chartHeight;
                if (i === 0) {{
                    ctx.moveTo(x, y);
                }} else {{
                    ctx.lineTo(x, y);
                }}
            }});
            ctx.stroke();

            // 零线
            const zeroY = padding.top + chartHeight - ((0 - minRet) / returnRange) * chartHeight;
            ctx.strokeStyle = 'rgba(200, 200, 200, 0.5)';
            ctx.lineWidth = 1;
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(padding.left, zeroY);
            ctx.lineTo(width - padding.right, zeroY);
            ctx.stroke();
            ctx.setLineDash([]);

            // X轴标签
            ctx.fillStyle = '#666';
            ctx.textAlign = 'center';
            const step = Math.ceil(data.length / 10);
            data.forEach((d, i) => {{
                if (i % step === 0 || i === data.length - 1) {{
                    const x = padding.left + (chartWidth / (data.length - 1)) * i;
                    const date = d.date.slice(5);
                    ctx.fillText(date, x, height - padding.bottom + 15);
                }}
            }});
        }}

        // 绘制净值曲线
        function drawNavChart(canvasId, data) {{
            const canvas = document.getElementById(canvasId);
            if (!canvas || !data || data.length === 0) return;
            const ctx = canvas.getContext('2d');
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width * 2;
            canvas.height = 800;
            ctx.scale(2, 2);
            const width = rect.width;
            const height = 400;

            // 清除画布
            ctx.clearRect(0, 0, width, height);

            // 计算数据范围
            const values = data.map(d => d.value);
            const minValue = Math.min(...values);
            const maxValue = Math.max(...values);
            const valueRange = maxValue - minValue || 1;

            // 边距
            const padding = {{ top: 20, right: 50, bottom: 40, left: 80 }};
            const chartWidth = width - padding.left - padding.right;
            const chartHeight = height - padding.top - padding.bottom;

            // 绘制网格线
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.05)';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 5; i++) {{
                const y = padding.top + (chartHeight / 5) * i;
                ctx.beginPath();
                ctx.moveTo(padding.left, y);
                ctx.lineTo(width - padding.right, y);
                ctx.stroke();

                // Y轴标签
                const value = maxValue - (valueRange / 5) * i;
                ctx.fillStyle = '#666';
                ctx.font = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", Tahoma, Geneva, Verdana, sans-serif';
                ctx.textAlign = 'right';
                ctx.fillText('CNY ' + value.toFixed(0), padding.left - 8, y + 4);
            }}

            // 绘制基准线（初始资金）
            const initialY = padding.top + chartHeight - ((100000 - minValue) / valueRange) * chartHeight;
            ctx.strokeStyle = 'rgba(200, 200, 200, 0.5)';
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(padding.left, initialY);
            ctx.lineTo(width - padding.right, initialY);
            ctx.stroke();
            ctx.setLineDash([]);

            // 绘制净值曲线
            ctx.strokeStyle = 'rgb(102, 126, 234)';
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            data.forEach((d, i) => {{
                const x = padding.left + (chartWidth / (data.length - 1)) * i;
                const y = padding.top + chartHeight - ((d.value - minValue) / valueRange) * chartHeight;
                if (i === 0) {{
                    ctx.moveTo(x, y);
                }} else {{
                    ctx.lineTo(x, y);
                }}
            }});
            ctx.stroke();

            // 绘制填充区域
            ctx.fillStyle = 'rgba(102, 126, 234, 0.1)';
            ctx.beginPath();
            data.forEach((d, i) => {{
                const x = padding.left + (chartWidth / (data.length - 1)) * i;
                const y = padding.top + chartHeight - ((d.value - minValue) / valueRange) * chartHeight;
                if (i === 0) {{
                    ctx.moveTo(x, y);
                }} else {{
                    ctx.lineTo(x, y);
                }}
            }});
            ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
            ctx.lineTo(padding.left, padding.top + chartHeight);
            ctx.closePath();
            ctx.fill();

            // 绘制X轴标签（最多10个）
            ctx.fillStyle = '#666';
            ctx.textAlign = 'center';
            const step = Math.ceil(data.length / 10);
            data.forEach((d, i) => {{
                if (i % step === 0 || i === data.length - 1) {{
                    const x = padding.left + (chartWidth / (data.length - 1)) * i;
                    const date = d.date.slice(5);
                    ctx.fillText(date, x, height - padding.bottom + 15);
                }}
            }});

            // 绘制最终值标签
            const lastValue = data[data.length - 1].value;
            const lastX = padding.left + chartWidth;
            const lastY = padding.top + chartHeight - ((lastValue - minValue) / valueRange) * chartHeight;
            ctx.fillStyle = '#667eea';
            ctx.font = 'bold 12px -apple-system, BlinkMacSystemFont, "Segoe UI", Tahoma, Geneva, Verdana, sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText('CNY ' + lastValue.toFixed(0), lastX + 5, lastY + 4);
        }}

        // 页面加载完成后初始化图表
        function initCharts() {{
            console.log('Parsing nav data...');
            const navData = parseCSV(navCSV);
            console.log('Nav data loaded:', navData.length, 'points');
            drawNavChart('navChart', navData);
            console.log('Nav chart initialized!');

            console.log('Parsing combined data...');
            const combinedData = parseCSV(combinedCSV);
            console.log('Combined data loaded:', combinedData.length, 'points');
            drawCombinedChart('combinedChart', combinedData);
            console.log('Combined chart initialized!');

            console.log('Parsing return comparison data...');
            const returnData = parseReturnCSV(returnCSV);
            console.log('Return comparison data loaded:', returnData.length, 'points');
            drawReturnChart('returnChart', returnData);
            console.log('Return comparison chart initialized!');

            {f'''console.log('Parsing excess return data...');
            if (excessReturnCSV && excessReturnCSV.trim().length > 0) {{
                const excessReturnData = parseExcessReturnCSV(excessReturnCSV);
                console.log('Excess return data loaded:', excessReturnData.length, 'points');
                drawExcessReturnChart('excessReturnChart', excessReturnData);
                console.log('Excess return chart initialized!');
            }}''' if model_type == 'separate' else ''}
        }}

        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initCharts);
        }} else {{
            initCharts();
        }}

        // 窗口大小改变时重绘图表
        window.addEventListener('resize', initCharts);
    </script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    logger.info(f"回测报告已保存: {output_path}")


def check_data_completeness(predictions_df: pd.DataFrame,
                           min_completion_ratio: float = 0.8) -> str:
    """
    检查数据完整性，确定合适的回测结束日期

    Args:
        predictions_df: 预测数据DataFrame
        min_completion_ratio: 最小完成比例（默认80%）

    Returns:
        实际应该使用的回测结束日期
    """
    # 获取每个ETF的最新预测日期
    latest_dates = predictions_df.groupby('code')['date'].max().reset_index()
    latest_dates.columns = ['code', 'latest_date']

    # 获取预测数据的最新日期（作为字符串比较）
    overall_latest_str = predictions_df['date'].max()

    # 确保日期格式一致（字符串比较）
    total_count = len(latest_dates)

    # 获取所有预测日期（从最新往前排序）
    all_dates = sorted(predictions_df['date'].unique(), reverse=True)

    # 从最后一天往前检查，直到找到满足80%条件的日期
    for i, check_date in enumerate(all_dates):
        # 计算到该日期为止，有多少ETF的数据（字符串比较）
        df_until_date = predictions_df[predictions_df["date"] <= check_date]
        updated_count = len(df_until_date['code'].unique())

        completion_ratio = updated_count / total_count
        logger.info(f"  检查日期 {check_date}: {updated_count}/{total_count} ETF ({completion_ratio:.1%}) 有数据")

        if completion_ratio >= min_completion_ratio:
            # 找到满足条件的日期
            logger.info(f"  数据完整性检查: 找到满足{min_completion_ratio:.0%}条件的日期: {check_date}")
            return check_date
        elif i > 0:
            # 继续往前检查
            continue

    # 如果遍历完所有日期都没找到，使用第一个日期
    if all_dates:
        logger.warning(f"  无法找到满足{min_completion_ratio:.0%}条件的日期，使用第一个预测日期: {all_dates[-1]}")
        return all_dates[-1]
    else:
        # 没有预测数据
        logger.warning(f"  没有预测数据，使用: {overall_latest_str}")
        print("DEBUG: check_data_completeness returning:", overall_latest_str)
    return overall_latest_str

    print("DEBUG: check_data_completeness returning:", overall_latest_str)
    return overall_latest_str


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='ETF预测策略回测')
    # 分类参数和回归参数互斥
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--model-type', choices=['separate', 'unified', 'both'],
                        help='回测的回归模型类型')
    group.add_argument('--classification', action='store_true',
                        help='使用分类模型进行回测')
    parser.add_argument('--with-cash-hedge', action='store_true',
                        help='使用分类模型进行带现金对冲的回测')
    parser.add_argument('--use-validation', action='store_true',
                        help='使用验证集模型进行分类回测（仅当--classification时有效）')
    args = parser.parse_args()

    # 确定回测哪些模型
    if args.classification:
        if args.with_cash_hedge:
            # 使用验证集模型（如果指定）
            model_type_str = 'classification_validation' if args.use_validation else 'classification'
            model_types = [f'{model_type_str}_cash_hedge']
            print("=" * 80)
            print(f"ETF分类模型策略回测（带现金对冲）- {'使用验证集模型' if args.use_validation else '使用普通模型'}")
            print("=" * 80)
            print(f"现金对冲ETF: 510500 (中证500)")
            print(f"对冲条件: 预测类别0 (大幅下跌 <-5%)")
        else:
            # 使用验证集模型（如果指定）
            model_type_str = 'classification_validation' if args.use_validation else 'classification'
            model_types = [model_type_str]
            print("=" * 80)
            print(f"ETF分类模型策略回测 - {'使用验证集模型' if args.use_validation else '使用普通模型'}")
            print("=" * 80)
    else:
        if args.model_type == 'both':
            model_types = ['separate', 'unified']
        else:
            model_types = [args.model_type]

        print("=" * 80)
        print("ETF预测策略回测")
        print("=" * 80)
        print(f"回测模型类型: {args.model_type}")

    # 加载ETF名称映射
    etf_names_path = os.path.join(REPORTS_DIR, 'etf_names.json')
    etf_names = load_etf_names(etf_names_path)

    # 确保输出目录存在
    os.makedirs(REPORTS_DIR, exist_ok=True)

    from ..config import DB_PATH

    backtest_start = BACKTEST_CONFIG['BACKTEST_START_DATE']

    all_results = []

    for model_type in model_types:
        print("\n" + "=" * 80)
        print(f"回测模型: {model_type}")
        print("=" * 80)

        # 加载预测数据
        print("\n加载预测数据...")
        print(f"  回测起始日期: {backtest_start}")
        # classification_cash_hedge 使用 classification 或 classification_validation 的数据
        if model_type.startswith('classification'):
            data_model_type = 'classification_validation' if args.use_validation else 'classification'
        else:
            data_model_type = model_type
        predictions_df = load_latest_predictions(DATA_DIR, backtest_start, data_model_type, use_validation=args.use_validation)
        print(f"  共加载 {len(predictions_df)} 条预测记录")
        print(f"  ETF数量: {predictions_df['code'].nunique()}")

        # 确定回测结束日期
        if len(predictions_df) > 0:
            if 'classification' in model_type:
                # 分类模型（包括validation）：使用上周最后一个交易日（周五）作为结束日期
                actual_end_date = get_last_trading_day_last_week()
                print(f" 分类模型回测，使用上周最后一个交易日: {actual_end_date}")
            else:
                # 回归模型：使用数据完整性检查确定结束日期
                actual_end_date = check_data_completeness(predictions_df, min_completion_ratio=0.8)
                print(f" 回归模型回测，数据完整性检查确定结束日期: {actual_end_date}")

            # 过滤出回测起始日期到实际结束日期之间的预测
            predictions_df = predictions_df[(predictions_df['date'] >= backtest_start) &
                                                  (predictions_df['date'] <= actual_end_date)].copy()
            print(f" 过滤后: 共 {len(predictions_df)} 条预测记录（{backtest_start} ~ {actual_end_date}）")

        if len(predictions_df) == 0:
            print(f"\n没有找到{model_type}预测数据，跳过")
            continue

        # 运行回测
        print("\n运行回测...")

        if model_type == 'classification' or model_type == 'classification_validation':
            # 标准分类模型回测（排除类别0）
            strategy_name = f'CLASSIFICATION_TOP5_NO_CLASS_0_{"VALIDATION" if model_type == "classification_validation" else ""}'
            results_df = run_classification_backtest(
                predictions_df=predictions_df,
                db_path=DB_PATH,
                backtest_start_date=backtest_start,
                backtest_end_date=actual_end_date,
                top_n=5,  # 固定TOP5
                rebalance_days=BACKTEST_CONFIG['REBALANCE_DAYS'],
                initial_capital=BACKTEST_CONFIG['INITIAL_CAPITAL'],
                strategy_name=strategy_name,
                transaction_rate=BACKTEST_CONFIG['TRANSACTION_RATE']
            )
        elif model_type == 'classification_cash_hedge' or model_type == 'classification_validation_cash_hedge':
            # 带现金对冲的分类模型回测
            strategy_name = 'CLASSIFICATION_TOP5_CASH_HEDGE'
            results_df = run_classification_backtest_with_cash_hedge(
                predictions_df=predictions_df,
                db_path=DB_PATH,
                backtest_start_date=backtest_start,
                backtest_end_date=actual_end_date,
                top_n=5,  # 固定TOP5
                rebalance_days=BACKTEST_CONFIG['REBALANCE_DAYS'],
                initial_capital=BACKTEST_CONFIG['INITIAL_CAPITAL'],
                strategy_name=strategy_name,
                transaction_rate=BACKTEST_CONFIG['TRANSACTION_RATE'],
                hedge_etf_code='510500'
            )
        else:
            # 回归模型回测
            strategy = STRATEGY_CONFIGS[-1]  # TOP5
            strategy_name = f"{strategy['name']}_{model_type}"
            results_df = run_backtest(
                predictions_df=predictions_df,
                db_path=DB_PATH,
                backtest_start_date=backtest_start,
                top_n=strategy['top_n'],
                rebalance_days=BACKTEST_CONFIG['REBALANCE_DAYS'],
                initial_capital=BACKTEST_CONFIG['INITIAL_CAPITAL'],
                strategy_name=strategy_name,
                transaction_rate=BACKTEST_CONFIG['TRANSACTION_RATE']
            )

        if not results_df.empty:
            metrics = calculate_metrics(results_df, BACKTEST_CONFIG['INITIAL_CAPITAL'])
            metrics['model_type'] = model_type

            print(f"\n策略 {strategy_name} 回测结果:")
            print(f"  起始日期: {metrics['start_date']}")
            print(f"  结束日期: {metrics['end_date']}")
            print(f"  交易天数: {metrics['trading_days']}天")
            print(f"  总收益率: {metrics['total_return']}%")
            print(f"  最终净值: CNY {metrics['final_value']:,.2f}")
            print(f"  最大回撤: {metrics['max_drawdown']}%")
            print(f"  年化收益率: {metrics['annualized_return']}%")
            print(f"  夏普比率: {metrics['sharpe_ratio']}")
            print(f"  胜率: {metrics['win_rate']}%")
            print(f"  盈亏比: {metrics['profit_loss_ratio']}")
            print(f"  调仓次数: {metrics['total_trades']}")

            # 保存结果
            csv_path = os.path.join(REPORTS_DIR, f"backtest_results_{strategy_name}.csv")
            html_path = os.path.join(REPORTS_DIR, f"backtest_report_{strategy_name}.html")
            generate_backtest_csv(results_df, csv_path)
            generate_backtest_html(results_df, metrics, html_path, etf_names, model_type=model_type)

            print(f"\n  详细结果CSV: {csv_path}")
            print(f"  HTML报告: {html_path}")

            all_results.append({
                'strategy_name': strategy_name,
                'metrics': metrics,
                'results_df': results_df
            })

    # 生成对比摘要
    if len(all_results) > 0:
        print("\n" + "=" * 80)
        print("策略对比")
        print("=" * 80)
        generate_strategy_comparison_summary(all_results, os.path.join(REPORTS_DIR, "backtest_comparison.csv"))

    print("\n完成！")


if __name__ == '__main__':
    main()
