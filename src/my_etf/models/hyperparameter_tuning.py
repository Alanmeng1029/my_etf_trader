# -*- coding: utf-8 -*-
"""
超参数优化模块
使用Optuna进行贝叶斯优化，支持时间序列交叉验证
"""
import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from ..config import FEATURE_COLS
from ..utils.logger import setup_logger

logger = setup_logger("hyperparameter_tuning", "hyperparameter_tuning.log")


# ============================================================================
# 时间序列交叉验证
# ============================================================================

class WalkForwardCV:
    """
    Walk-Forward交叉验证（适用于时间序列）

    在每个fold中，训练集在前，测试集在后，避免数据泄露
    """

    def __init__(self, n_splits: int = 5, test_size: int = 50):
        """
        Args:
            n_splits: 交叉验证折数
            test_size: 每个fold的测试集大小
        """
        self.n_splits = n_splits
        self.test_size = test_size

    def split(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        生成训练/测试索引

        Args:
            X: 特征DataFrame
            y: 目标变量Series

        Yields:
            (train_indices, test_indices) 元组
        """
        n_samples = len(X)
        min_train_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_start = 0
            train_end = min_train_size + i * self.test_size
            test_start = train_end
            test_end = test_start + self.test_size

            if test_end > n_samples:
                break

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices

    def get_n_splits(self) -> int:
        return self.n_splits


# ============================================================================
# 评估指标
# ============================================================================

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    评估模型性能

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        包含各种评估指标的字典
    """
    # 移除NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        return {
            'r2': -np.inf,
            'mae': np.inf,
            'rmse': np.inf,
            'mape': np.inf
        }

    metrics = {
        'r2': r2_score(y_true_clean, y_pred_clean),
        'mae': mean_absolute_error(y_true_clean, y_pred_clean),
        'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
        'mape': np.mean(np.abs((y_true_clean - y_pred_clean) / np.abs(y_true_clean + 1e-8))) * 100
    }

    return metrics


# ============================================================================
# Optuna优化
# ============================================================================

def xgboost_objective(trial: optuna.Trial,
                      X: pd.DataFrame,
                      y: pd.Series,
                      cv: WalkForwardCV,
                      scaler: Optional[StandardScaler] = None,
                      optimize_metric: str = 'r2') -> float:
    """
    XGBoost超参数优化的目标函数

    Args:
        trial: Optuna trial对象
        X: 特征DataFrame
        y: 目标变量Series
        cv: 交叉验证对象
        scaler: 特征标准化器
        optimize_metric: 优化的指标 ('r2', 'mae', 'rmse', 'mape')

    Returns:
        平均验证分数（如果是R2则最大化，其他则最小化）
    """
    # 定义超参数搜索空间
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
    }

    scores = []

    # 时间序列交叉验证
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 特征标准化
        if scaler is not None:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled, X_test_scaled = X_train, X_test

        # 训练模型
        model = xgb.XGBRegressor(
            **params,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train_scaled, y_train)

        # 预测
        y_pred = model.predict(X_test_scaled)

        # 评估
        metrics = evaluate_model(y_test.values, y_pred)
        scores.append(metrics[optimize_metric])

    # 计算平均分数
    mean_score = np.mean(scores)

    # 如果是R2则最大化，其他则最小化
    if optimize_metric == 'r2':
        return mean_score
    else:
        return -mean_score  # 取负值以便最小化


def optimize_hyperparameters(X: pd.DataFrame,
                            y: pd.Series,
                            n_trials: int = 100,
                            timeout: Optional[int] = None,
                            optimize_metric: str = 'r2',
                            use_scaling: bool = True,
                            cv_splits: int = 5,
                            cv_test_size: int = 50) -> Tuple[Dict, optuna.Study]:
    """
    使用Optuna优化XGBoost超参数

    Args:
        X: 特征DataFrame
        y: 目标变量Series
        n_trials: 优化试验次数
        timeout: 超时时间（秒），None表示不限制
        optimize_metric: 优化的指标
        use_scaling: 是否使用特征标准化
        cv_splits: 交叉验证折数
        cv_test_size: 每个fold的测试集大小

    Returns:
        (最佳参数, Optuna study对象)
    """
    logger.info("="*60)
    logger.info("开始超参数优化")
    logger.info("="*60)
    logger.info(f"数据集大小: {len(X)}")
    logger.info(f"特征数: {len(X.columns)}")
    logger.info(f"试验次数: {n_trials}")
    logger.info(f"优化指标: {optimize_metric}")
    logger.info(f"使用标准化: {use_scaling}")
    logger.info("="*60)

    # 创建标准化器
    scaler = StandardScaler() if use_scaling else None

    # 创建交叉验证对象
    cv = WalkForwardCV(n_splits=cv_splits, test_size=cv_test_size)

    # 创建study
    direction = 'maximize' if optimize_metric == 'r2' else 'minimize'
    study = optuna.create_study(direction=direction)

    # 优化
    study.optimize(
        lambda trial: xgboost_objective(
            trial, X, y, cv, scaler, optimize_metric
        ),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    # 获取最佳参数
    best_params = study.best_params
    best_score = study.best_value

    logger.info("\n" + "="*60)
    logger.info("优化完成")
    logger.info("="*60)
    logger.info(f"最佳分数 ({optimize_metric}): {best_score:.4f}")
    logger.info("最佳参数:")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")
    logger.info("="*60)

    return best_params, study


# ============================================================================
# 结果分析
# ============================================================================

def plot_optimization_history(study: optuna.Study,
                              save_path: Optional[str] = None) -> None:
    """
    绘制优化历史

    Args:
        study: Optuna study对象
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt

        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.title("优化历史")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"优化历史图已保存: {save_path}")

        plt.show()

    except ImportError:
        logger.warning("matplotlib未安装，跳过绘图")


def plot_param_importances(study: optuna.Study,
                           save_path: Optional[str] = None) -> None:
    """
    绘制参数重要性

    Args:
        study: Optuna study对象
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt

        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.title("参数重要性")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"参数重要性图已保存: {save_path}")

        plt.show()

    except ImportError:
        logger.warning("matplotlib未安装，跳过绘图")


def plot_parallel_coordinate(study: optuna.Study,
                            save_path: Optional[str] = None) -> None:
    """
    绘制平行坐标图

    Args:
        study: Optuna study对象
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt

        fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        plt.title("平行坐标图")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"平行坐标图已保存: {save_path}")

        plt.show()

    except ImportError:
        logger.warning("matplotlib未安装，跳过绘图")


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    import argparse
    from ..config import DB_MIN_DATE, BENCHMARK_ETF
    from ..utils.database import read_etf_data
    from ..models.target_engineering import create_excess_return_target, get_benchmark_returns

    parser = argparse.ArgumentParser(
        description="XGBoost超参数优化 - 使用Optuna进行贝叶斯优化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 优化单个ETF的超参数
  python %(prog)s --code 510050 --trials 50

  # 指定优化指标和试验次数
  python %(prog)s --code 510050 --trials 100 --metric mae

  # 保存优化结果
  python %(prog)s --code 510050 --output params.json --plot-dir plots/
        """
    )

    parser.add_argument('--code', type=str, required=True, help='ETF代码')
    parser.add_argument('--trials', type=int, default=100, help='优化试验次数（默认: 100）')
    parser.add_argument('--metric', type=str, default='r2',
                        choices=['r2', 'mae', 'rmse', 'mape'],
                        help='优化指标（默认: r2）')
    parser.add_argument('--no-scaling', action='store_true',
                        help='不使用特征标准化')
    parser.add_argument('--cv-splits', type=int, default=5,
                        help='交叉验证折数（默认: 5）')
    parser.add_argument('--cv-test-size', type=int, default=50,
                        help='每个fold的测试集大小（默认: 50）')
    parser.add_argument('--timeout', type=int, default=None,
                        help='超时时间（秒），None表示不限制')
    parser.add_argument('--output', type=str, default=None,
                        help='保存最佳参数的JSON文件路径')
    parser.add_argument('--plot-dir', type=str, default=None,
                        help='保存图表的目录路径')
    parser.add_argument('--study-name', type=str, default=None,
                        help='study名称（用于加载已有的优化）')

    args = parser.parse_args()

    # 读取数据
    logger.info(f"\n读取ETF数据: {args.code}")
    df = read_etf_data(args.code, min_date=DB_MIN_DATE)

    if len(df) < 100:
        logger.error("数据不足")
        return

    # 计算超额收益目标
    logger.info("计算超额收益目标...")
    benchmark_returns = get_benchmark_returns(BENCHMARK_ETF, min_date=DB_MIN_DATE)
    df = create_excess_return_target(df, benchmark_returns=benchmark_returns)

    # 移除NaN
    df = df.dropna()

    # 获取特征和目标
    X = df[FEATURE_COLS]
    y = df['excess_return']

    logger.info(f"特征数: {len(X.columns)}")
    logger.info(f"样本数: {len(X)}")

    # 超参数优化
    best_params, study = optimize_hyperparameters(
        X, y,
        n_trials=args.trials,
        timeout=args.timeout,
        optimize_metric=args.metric,
        use_scaling=not args.no_scaling,
        cv_splits=args.cv_splits,
        cv_test_size=args.cv_test_size
    )

    # 保存参数
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=2)
        logger.info(f"\n最佳参数已保存: {args.output}")

    # 绘制图表
    if args.plot_dir:
        import os
        os.makedirs(args.plot_dir, exist_ok=True)

        plot_optimization_history(study, save_path=os.path.join(args.plot_dir, 'optimization_history.png'))
        plot_param_importances(study, save_path=os.path.join(args.plot_dir, 'param_importances.png'))
        plot_parallel_coordinate(study, save_path=os.path.join(args.plot_dir, 'parallel_coordinate.png'))


if __name__ == '__main__':
    main()
