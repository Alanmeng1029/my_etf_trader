# -*- coding: utf-8 -*-
"""
目标工程模块
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
        df = read_etf_data(benchmark_code, min_date=min_date, use_advanced=False)
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

    # 计算ETF绝对收益率
    abs_return = (df_out[close_col].shift(-horizon) - df_out[close_col]) / df_out[close_col] * 100

    # 获取基准收益率
    if benchmark_returns is None:
        benchmark_returns = get_benchmark_returns(benchmark_code, horizon=horizon)

    if benchmark_returns.empty:
        logger.warning(f"无法获取基准 {benchmark_code} 收益率，使用绝对收益率")
        df_out['excess_return'] = abs_return
    elif 'date' in df_out.columns or '日期' in df_out.columns:
        # 转换日期为datetime用于对齐
        date_col = 'date' if 'date' in df_out.columns else '日期'

        # 简单对齐方法 - 按位置对齐
        if len(df_out) == len(benchmark_returns):
            # 长度相同，直接相减
            df_out['excess_return'] = abs_return.values - benchmark_returns.values
        else:
            # 长度不同，需要对齐
            # 使用简单的reindex方法
            try:
                df_dates = pd.to_datetime(df_out[date_col])
                aligned_returns = benchmark_returns.reindex(df_dates.index, method='ffill')
                df_out['excess_return'] = abs_return.values - aligned_returns.values
            except Exception as e:
                logger.warning(f"对齐失败: {e}，使用绝对收益率")
                df_out['excess_return'] = abs_return.values
    else:
        logger.warning(f"DataFrame中未找到日期列，无法对齐基准收益率")
        df_out['excess_return'] = abs_return.values

    # 移除最后horizon天的数据（无法计算目标）
    df_out = df_out.iloc[:-horizon]

    return df_out


# ============================================================================
# 方向分类目标
# ============================================================================

def create_direction_target(df: pd.DataFrame,
                           target_col: str = 'week_return',
                           threshold: float = 0.0) -> pd.DataFrame:
    """
    创建方向分类目标（涨/跌）

    Args:
        df: 包含目标变量的DataFrame
        target_col: 目标变量列名
        threshold: 阈值，大于阈值为1，否则为0

    Returns:
        添加了direction列的DataFrame
    """
    df_out = df.copy()

    if target_col not in df_out.columns:
        raise ValueError(f"目标列 {target_col} 不存在")

    df_out['direction'] = (df_out[target_col] > threshold).astype(int)

    return df_out


def create_multi_direction_target(df: pd.DataFrame,
                                 target_col: str = 'week_return',
                                 thresholds: Tuple[float, float] = (-1.0, 1.0)) -> pd.DataFrame:
    """
    创建多分类方向目标（强跌/跌/涨/强涨）

    Args:
        df: 包含目标变量的DataFrame
        target_col: 目标变量列名
        thresholds: 分类阈值 (下界, 上界)

    Returns:
        添加了multi_direction列的DataFrame（0=强跌, 1=跌, 2=涨, 3=强涨）
    """
    df_out = df.copy()

    if target_col not in df_out.columns:
        raise ValueError(f"目标列 {target_col} 不存在")

    lower, upper = thresholds

    def classify(x):
        if x < lower:
            return 0  # 强跌
        elif x < upper:
            return 1 if x >= 0 else 0  # 跌（接近阈值以下）
        elif x < upper * 1.5:
            return 2  # 涨
        else:
            return 3  # 强涨

    df_out['multi_direction'] = (df_out[target_col] > 0).astype(int)

    return df_out


# ============================================================================
# 目标对比和可视化
# ============================================================================

def compare_targets(df_abs: pd.DataFrame, df_excess: pd.DataFrame) -> pd.DataFrame:
    """
    对比绝对收益和超额收益的统计特性

    Args:
        df_abs: 包含绝对收益的DataFrame
        df_excess: 包含超额收益的DataFrame

    Returns:
        统计对比结果DataFrame
    """
    abs_col = 'week_return'
    excess_col = 'excess_return'

    stats = {
        'metric': ['均值', '标准差', '最小值', '最大值', '25%分位', '50%分位', '75%分位'],
        'absolute': [
            df_abs[abs_col].mean(),
            df_abs[abs_col].std(),
            df_abs[abs_col].min(),
            df_abs[abs_col].max(),
            df_abs[abs_col].quantile(0.25),
            df_abs[abs_col].quantile(0.5),
            df_abs[abs_col].quantile(0.75),
        ],
        'excess': [
            df_excess[excess_col].mean(),
            df_excess[excess_col].std(),
            df_excess[excess_col].min(),
            df_excess[excess_col].max(),
            df_excess[excess_col].quantile(0.25),
            df_excess[excess_col].quantile(0.5),
            df_excess[excess_col].quantile(0.75),
        ],
    }

    return pd.DataFrame(stats)


def plot_target_comparison(df_abs: pd.DataFrame,
                          df_excess: pd.DataFrame,
                          save_path: Optional[str] = None) -> None:
    """
    绘制绝对收益和超额收益的对比图

    Args:
        df_abs: 包含绝对收益的DataFrame
        df_excess: 包含超额收益的DataFrame
        save_path: 保存路径（如果为None则不保存）

    Returns:
        None
    """
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 分布图
        axes[0].hist(df_abs['week_return'], bins=50, alpha=0.5, label='绝对收益', density=True)
        axes[0].hist(df_excess['excess_return'], bins=50, alpha=0.5, label='超额收益', density=True)
        axes[0].set_xlabel('收益率 (%)')
        axes[0].set_ylabel('密度')
        axes[0].set_title('收益率分布对比')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 累积分布
        abs_sorted = np.sort(df_abs['week_return'])
        excess_sorted = np.sort(df_excess['excess_return'])
        axes[1].plot(abs_sorted, np.arange(1, len(abs_sorted) + 1) / len(abs_sorted),
                    label='绝对收益')
        axes[1].plot(excess_sorted, np.arange(1, len(excess_sorted) + 1) / len(excess_sorted),
                    label='超额收益')
        axes[1].set_xlabel('收益率 (%)')
        axes[1].set_ylabel('累积概率')
        axes[1].set_title('累积分布对比')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Q-Q图
        axes[2].scatter(np.sort(df_abs['week_return']),
                       np.sort(df_excess['excess_return']),
                       alpha=0.5, s=5)
        axes[2].plot([df_abs['week_return'].min(), df_abs['week_return'].max()],
                    [df_abs['week_return'].min(), df_abs['week_return'].max()],
                    'r--', label='y=x')
        axes[2].set_xlabel('绝对收益 (%)')
        axes[2].set_ylabel('超额收益 (%)')
        axes[2].set_title('Q-Q图')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"对比图已保存: {save_path}")

        plt.show()

    except ImportError:
        logger.warning("matplotlib未安装，跳过绘图")


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    import argparse
    from ..config import get_etf_codes
    from ..utils.constants import DB_MIN_DATE

    parser = argparse.ArgumentParser(
        description="目标工程 - 计算绝对收益和超额收益",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 对比绝对收益和超额收益
  python %(prog)s --code 510050 --compare

  # 计算所有ETF的超额收益
  python %(prog)s --all

  # 使用自定义基准ETF
  python %(prog)s --code 510050 --benchmark 159915
        """
    )

    # ETF选择参数
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', help='处理所有ETF')
    group.add_argument('--code', type=str, help='处理指定ETF代码')

    # 其他参数
    parser.add_argument('--benchmark', type=str, default=BENCHMARK_ETF,
                        help=f'基准ETF代码（默认: {BENCHMARK_ETF}）')
    parser.add_argument('--compare', action='store_true',
                        help='对比绝对收益和超额收益')

    args = parser.parse_args()

    # 确定要处理的ETF列表
    if args.all:
        codes = get_etf_codes()
    else:
        codes = [args.code]

    # 获取基准收益率
    logger.info(f"获取基准 {args.benchmark} 收益率...")
    benchmark_returns = get_benchmark_returns(args.benchmark,
                                             min_date=DB_MIN_DATE)

    if benchmark_returns.empty:
        logger.error(f"无法获取基准 {args.benchmark} 的收益率")
        return

    # 处理每个ETF
    for code in codes:
        logger.info(f"\n处理ETF: {code}")

        # 读取数据
        df = read_etf_data(code, min_date=DB_MIN_DATE, use_advanced=True)
        logger.info(f" 读取数据: {len(df)} 条")

        if len(df) < 100:
            logger.warning(f"  数据不足，跳过")
            continue

        # 计算绝对收益
        df_abs = create_absolute_return_target(df)

        # 计算超额收益
        df_excess = create_excess_return_target(df,
                                               benchmark_returns=benchmark_returns)

        # 对比
        if args.compare:
            stats = compare_targets(df_abs, df_excess)
            print("\n" + "="*60)
            print(f"目标对比: {code}")
            print("="*60)
            print(stats.to_string(index=False))
            print("="*60)


if __name__ == '__main__':
    main()
