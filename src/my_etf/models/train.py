"""
ETF模型训练模块
支持：
1. 绝对收益和超额收益目标
2. 多种模型架构：separate（每ETF单独）、unified（统一模型）、two-stage（两阶段模型）
3. 高级特征支持
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, log_loss
from sklearn.preprocessing import StandardScaler

# 强制重新导入database模块（避免缓存问题）
import importlib
import sys
if 'my_etf.utils' in sys.modules:
    del sys.modules['my_etf.utils']
    import my_etf.utils

from ..config import (
    MODELS_DIR, DATA_DIR, MODEL_PARAMS, FEATURE_COLS,
    ALL_FEATURE_COLS, BENCHMARK_ETF, DB_MIN_DATE, MODEL_ARCHITECTURE,
    CLASSIFICATION_TARGET, CLASSIFICATION_CONFIG, CLASSIFICATION_MODEL_PARAMS,
    BACKTEST_CONFIG
)
from ..utils.database import read_etf_data
from ..utils.logger import setup_logger

logger = setup_logger("etf_train", "train.log")

# 目标列
TARGET_ABSOLUTE_RETURN = 'week_return'
TARGET_EXCESS_RETURN = 'excess_return'


# ============================================================================
# 目标工程
# ============================================================================

def create_target(df: pd.DataFrame,
                 days: int = 5,
                 target_type: str = 'absolute',
                 benchmark_returns: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    创建预测目标

    Args:
        df: 包含收盘价的DataFrame
        days: 预测天数
        target_type: 目标类型 ('absolute' 或 'excess')
        benchmark_returns: 基准收益率（excess目标需要），必须带DatetimeIndex

    Returns:
        添加了目标列的DataFrame
    """
    df = df.copy()

    # 确定收盘价列
    if 'close' in df.columns:
        close_col = 'close'
    elif '收盘' in df.columns:
        close_col = '收盘'
    else:
        logger.error("DataFrame中未找到收盘价列")
        return df

    # 确定日期列
    if 'date' in df.columns:
        date_col = 'date'
    elif '日期' in df.columns:
        date_col = '日期'
    else:
        date_col = None

    # 确保close_col是Series而不是DataFrame
    close_prices = df[close_col]
    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.iloc[:, 0]

    # 计算绝对收益率
    abs_return = (close_prices.shift(-days) - close_prices) / close_prices * 100
    # 确保abs_return是Series
    if isinstance(abs_return, pd.DataFrame):
        abs_return = abs_return.iloc[:, 0]

    df[TARGET_ABSOLUTE_RETURN] = abs_return

    # 计算超额收益率
    if target_type == 'excess' and benchmark_returns is not None:
        if date_col is not None:
            # 使用pd.merge按日期对齐
            # 创建临时DataFrame用于对齐
            df_temp = pd.DataFrame({
                'date': df[date_col],
                'etf_return': abs_return.values
            })
            df_temp['date'] = pd.to_datetime(df_temp['date'])

            # 创建基准收益率DataFrame
            benchmark_df = pd.DataFrame({
                'date': benchmark_returns.index,
                'benchmark_return': benchmark_returns.values
            })

            # 按日期合并
            merged = pd.merge(df_temp, benchmark_df, on='date', how='left')

            # 前向填充缺失的基准收益
            merged['benchmark_return'] = merged['benchmark_return'].ffill()

            # 计算超额收益
            df[TARGET_EXCESS_RETURN] = merged['etf_return'] - merged['benchmark_return'].values
        else:
            logger.warning("DataFrame中未找到日期列，使用绝对收益率")
            df[TARGET_EXCESS_RETURN] = abs_return
    elif target_type == 'excess':
        logger.warning("未提供基准收益率，使用绝对收益率")
        df[TARGET_EXCESS_RETURN] = abs_return
    else:
        df[TARGET_EXCESS_RETURN] = abs_return

    # 移除最后days天的数据（无法计算目标）
    df = df.iloc[:-days]

    return df


def create_classification_target(df: pd.DataFrame, days: int = 5) -> pd.DataFrame:
    """
    创建分类目标

    将下周收益率分为4个类别：
    - 类别0: < -5% (大幅下跌)
    - 类别1: -5% ~ 0% (小幅下跌)
    - 类别2: 0% ~ 5% (小幅上涨)
    - 类别3: > 5% (大幅上涨)

    Args:
        df: 包含收盘价的DataFrame
        days: 预测天数

    Returns:
        添加了分类目标列的DataFrame
    """
    df = df.copy()

    # 复用现有的 create_target 函数计算 week_return
    df = create_target(df, days=days, target_type='absolute')

    # 使用 pd.cut 根据 bins 分箱
    df[CLASSIFICATION_TARGET] = pd.cut(
        df[TARGET_ABSOLUTE_RETURN],
        bins=CLASSIFICATION_CONFIG['bins'],
        labels=CLASSIFICATION_CONFIG['labels']
    )

    # 转换为整数类型
    df[CLASSIFICATION_TARGET] = df[CLASSIFICATION_TARGET].astype(int)

    return df


def get_benchmark_returns(benchmark_code: str = BENCHMARK_ETF,
                         min_date: str = DB_MIN_DATE,
                         horizon: int = 5) -> Optional[pd.Series]:
    """获取基准ETF的未来收益率序列"""
    try:
        df = read_etf_data(benchmark_code, min_date=min_date, use_advanced=False)
        close_col = 'close' if 'close' in df.columns else '收盘'
        benchmark_returns = (df[close_col].shift(-horizon) - df[close_col]) / df[close_col] * 100

        # 设置日期索引
        if 'date' in df.columns:
            benchmark_returns.index = pd.to_datetime(df['date'])

        return benchmark_returns
    except Exception as e:
        logger.error(f"获取基准 {benchmark_code} 收益率失败: {e}")
        return None


# ============================================================================
# 数据处理
# ============================================================================

def clean_data(df: pd.DataFrame, min_history_days: int = 60) -> pd.DataFrame:
    """清洗数据：移除NaN值和早期数据"""
    df = df.copy()
    df = df.iloc[min_history_days:]
    df = df.dropna()
    return df


def split_data(df: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按时间顺序划分训练和测试集"""
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def create_walk_forward_splits(
    df: pd.DataFrame,
    train_days: Optional[int] = None,
    validation_days: Optional[int] = None,
    step_days: Optional[int] = None,
    embargo_days: Optional[int] = None,
) -> List[Dict]:
    """Create rolling train/validation windows with an embargo gap."""
    train_days = train_days or BACKTEST_CONFIG['WALK_FORWARD_TRAIN_DAYS']
    validation_days = validation_days or BACKTEST_CONFIG['WALK_FORWARD_VALIDATION_DAYS']
    step_days = step_days or BACKTEST_CONFIG['WALK_FORWARD_STEP_DAYS']
    embargo_days = embargo_days or BACKTEST_CONFIG['EMBARGO_DAYS']

    ordered = df.sort_values('date').reset_index(drop=True).copy() if 'date' in df.columns else df.reset_index(drop=True).copy()
    splits = []
    start = 0
    fold = 1
    while True:
        train_start = start
        train_end = train_start + train_days
        validation_start = train_end + embargo_days
        validation_end = validation_start + validation_days
        if validation_end > len(ordered):
            break

        train_df = ordered.iloc[train_start:train_end].copy()
        validation_df = ordered.iloc[validation_start:validation_end].copy()
        splits.append({
            'fold': fold,
            'train_start_date': train_df['date'].iloc[0] if 'date' in train_df.columns else train_start,
            'train_end_date': train_df['date'].iloc[-1] if 'date' in train_df.columns else train_end - 1,
            'validation_start_date': validation_df['date'].iloc[0] if 'date' in validation_df.columns else validation_start,
            'validation_end_date': validation_df['date'].iloc[-1] if 'date' in validation_df.columns else validation_end - 1,
            'embargo_days': embargo_days,
            'train_df': train_df,
            'validation_df': validation_df,
        })
        fold += 1
        start += step_days

    return splits


def evaluate_classification_calibration(
    y_true: pd.Series,
    probabilities: np.ndarray,
    labels: Tuple[int, ...] = (0, 1, 2, 3),
    bins: int = 5,
) -> Dict:
    """Evaluate classification probability calibration and class balance."""
    y = pd.Series(y_true).astype(int).reset_index(drop=True)
    proba = np.asarray(probabilities)
    if proba.ndim != 2 or proba.shape[0] != len(y):
        return {}

    one_hot = np.zeros((len(y), len(labels)))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    for row_idx, label in enumerate(y):
        if label in label_to_idx:
            one_hot[row_idx, label_to_idx[label]] = 1

    clipped = np.clip(proba, 1e-12, 1 - 1e-12)
    metrics = {
        'logloss': float(log_loss(y, clipped, labels=list(labels))),
        'brier_score': float(np.mean(np.sum((clipped - one_hot) ** 2, axis=1))),
        'class_distribution': {str(label): int((y == label).sum()) for label in labels},
        'probability_bins': [],
    }

    class3_idx = label_to_idx.get(3, len(labels) - 1)
    class3_proba = clipped[:, class3_idx]
    edges = np.linspace(0, 1, bins + 1)
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (class3_proba >= left) & (class3_proba < right if right < 1 else class3_proba <= right)
        count = int(mask.sum())
        hit_rate = float((y[mask] == 3).mean()) if count else None
        metrics['probability_bins'].append({
            'left': round(float(left), 4),
            'right': round(float(right), 4),
            'count': count,
            'class3_hit_rate': hit_rate,
        })

    return metrics


# ============================================================================
# 模型架构：每ETF单独训练
# ============================================================================

def train_separate_models(codes: List[str],
                          target_type: str = 'absolute',
                          feature_cols: List[str] = ALL_FEATURE_COLS,
                          use_scaling: bool = True,
                          min_date: Optional[str] = None) -> List[Dict]:
    """
    每个ETF单独训练模型

    Args:
        codes: ETF代码列表
        target_type: 目标类型 ('absolute' 或 'excess')
                      - separate 模型默认使用 'absolute'（绝对收益）
        feature_cols: 特征列列表
        use_scaling: 是否使用标准化
        min_date: 最小日期（None表示使用DB_MIN_DATE）

    Returns:
        训练结果列表
    """
    logger.info("="*60)
    logger.info("每ETF单独训练模型")
    logger.info(f"目标类型: {target_type}")
    if target_type == 'absolute':
        logger.info("使用绝对收益目标，每个ETF使用自己的历史数据范围")
    else:
        logger.info("使用超额收益目标，相对于基准ETF")
    logger.info("="*60)

    # 确定使用的数据起始日期
    if min_date is None:
        min_date = DB_MIN_DATE

    # 获取基准收益率（仅在excess目标时需要）
    benchmark_returns = None
    if target_type == 'excess':
        logger.info(f"获取基准收益率 ({BENCHMARK_ETF})...")
        benchmark_returns = get_benchmark_returns(min_date=min_date)
        if benchmark_returns is None or benchmark_returns.empty:
            logger.error("无法获取基准收益率")
            return []

    all_results = []

    for i, code in enumerate(codes, 1):
        logger.info(f"\n[{i}/{len(codes)}] 处理ETF: {code}")

        try:
            # 读取数据 - 单独模型使用每个ETF自己的历史范围
            df = read_etf_data(code, min_date=min_date, use_advanced=True)
            logger.info(f"  原始数据: {len(df)} 条")

            if len(df) < 100:
                logger.warning(f"  数据不足，跳过")
                continue

            # 创建目标
            df = create_target(df, target_type=target_type, benchmark_returns=benchmark_returns)
            logger.info(f"  计算目标后: {len(df)} 条")

            # 清洗数据
            df = clean_data(df, min_history_days=60)
            logger.info(f"  清洗后: {len(df)} 条")

            if len(df) < 50:
                logger.warning(f"  清洗后数据不足，跳过")
                continue

            # 划分训练/测试集
            train_df, test_df = split_data(df, train_ratio=0.7)
            logger.info(f"  训练集: {len(train_df)} 条，测试集: {len(test_df)} 条")

            # 训练模型
            result = train_single_model(
                code, train_df, test_df,
                target_col=TARGET_EXCESS_RETURN if target_type == 'excess' else TARGET_ABSOLUTE_RETURN,
                feature_cols=feature_cols,
                use_scaling=use_scaling
            )

            all_results.append(result)

        except Exception as e:
            logger.error(f"  ETF {code} 处理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    return all_results


def train_single_model(code: str,
                      train_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      target_col: str,
                      feature_cols: List[str],
                      use_scaling: bool = True) -> Dict:
    """
    训练单个ETF的模型

    Returns:
        训练结果字典
    """
    # 准备数据
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # 移除NaN
    mask = ~X_train.isnull().any(axis=1) & ~y_train.isnull()
    X_train = X_train[mask]
    y_train = y_train[mask]

    mask = ~X_test.isnull().any(axis=1) & ~y_test.isnull()
    X_test = X_test[mask]
    y_test = y_test[mask]

    # 标准化
    scaler = None
    if use_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values

    # 创建模型
    model = xgb.XGBRegressor(**MODEL_PARAMS)
    model.fit(X_train_scaled, y_train)

    # 评估
    y_pred = model.predict(X_test_scaled)
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mape': np.mean(np.abs((y_test - y_pred) / np.abs(y_test + 1e-8))) * 100
    }

    logger.info(f"  模型评估: R2={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}%")

    # 保存模型
    model_version = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_model(code, model, scaler, model_version, target_col, feature_cols, model_type='regression')

    return {
        'code': code,
        'model_version': model_version,
        'target': target_col,
        'r2': metrics['r2'],
        'mae': metrics['mae'],
        'rmse': metrics['rmse'],
        'mape': metrics['mape'],
        'train_size': len(X_train),
        'test_size': len(X_test),
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols
    }


def save_model(code: str, model, scaler, model_version: str,
              target_col: str, feature_cols: List[str], model_type: str = 'regression',
              extra_metadata: Optional[Dict] = None):
    """
    保存模型、标准化器和元数据

    Args:
        code: ETF代码
        model: 模型对象
        scaler: 标准化器
        model_version: 模型版本号
        target_col: 目标列名
        feature_cols: 特征列列表
        model_type: 模型类型 ('regression' 或 'classification')
    """
    model_dir = os.path.join(MODELS_DIR, code)
    os.makedirs(model_dir, exist_ok=True)

    # 根据模型类型添加后缀
    suffix = '_classification' if model_type == 'classification' else ''

    # 保存模型
    model_path = os.path.join(model_dir, f"{code}_{model_version}{suffix}.pkl")
    joblib.dump(model, model_path)

    # 保存标准化器
    if scaler is not None:
        scaler_path = os.path.join(model_dir, f"{code}_{model_version}{suffix}_scaler.pkl")
        joblib.dump(scaler, scaler_path)

    # 保存元数据
    metadata = {
        'code': code,
        'model_version': model_version,
        'model_type': model_type,
        'target': target_col,
        'features': feature_cols,
        'feature_count': len(feature_cols),
        'params': MODEL_PARAMS if model_type == 'regression' else CLASSIFICATION_MODEL_PARAMS,
        'train_date': datetime.now().isoformat(),
        'train_cutoff_date': extra_metadata.get('train_cutoff_date') if extra_metadata else None,
        'validation_periods': extra_metadata.get('validation_periods', []) if extra_metadata else [],
        'etf_universe': extra_metadata.get('etf_universe', [code]) if extra_metadata else [code],
        'data_version': extra_metadata.get('data_version', datetime.now().strftime('%Y%m%d')) if extra_metadata else datetime.now().strftime('%Y%m%d'),
        'indicator_version': extra_metadata.get('indicator_version', 'basic+advanced') if extra_metadata else 'basic+advanced',
        'python_version': sys.version,
        'has_scaler': scaler is not None
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


# ============================================================================
# 分类模型训练
# ============================================================================

def train_single_classification_model(code: str,
                                     train_df: pd.DataFrame,
                                     test_df: pd.DataFrame,
                                     feature_cols: List[str],
                                     use_scaling: bool = True) -> Dict:
    """
    训练单个ETF的分类模型

    Returns:
        训练结果字典
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # 准备数据
    X_train = train_df[feature_cols]
    y_train = train_df[CLASSIFICATION_TARGET]

    X_test = test_df[feature_cols]
    y_test = test_df[CLASSIFICATION_TARGET]

    # 移除NaN
    mask = ~X_train.isnull().any(axis=1) & ~y_train.isnull()
    X_train = X_train[mask]
    y_train = y_train[mask]

    mask = ~X_test.isnull().any(axis=1) & ~y_test.isnull()
    X_test = X_test[mask]
    y_test = y_test[mask]

    # 标准化
    scaler = None
    if use_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values

    # 创建分类模型
    model = xgb.XGBClassifier(**CLASSIFICATION_MODEL_PARAMS)
    model.fit(X_train_scaled, y_train)

    # 评估
    y_pred = model.predict(X_test_scaled)
    raw_proba = model.predict_proba(X_test_scaled)
    aligned_proba = np.zeros((len(y_test), 4))
    for idx, cls in enumerate(getattr(model, 'classes_', [])):
        if int(cls) in (0, 1, 2, 3):
            aligned_proba[:, int(cls)] = raw_proba[:, idx]
    calibration_metrics = evaluate_classification_calibration(y_test, aligned_proba)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'logloss': calibration_metrics.get('logloss'),
        'brier_score': calibration_metrics.get('brier_score'),
        'class_distribution': calibration_metrics.get('class_distribution', {}),
        'probability_bins': calibration_metrics.get('probability_bins', []),
    }

    logger.info(f"  分类模型评估: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")

    # 保存模型
    model_version = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_model(
        code,
        model,
        scaler,
        model_version,
        CLASSIFICATION_TARGET,
        feature_cols,
        model_type='classification',
        extra_metadata={
            'train_cutoff_date': train_df['date'].max() if 'date' in train_df.columns else None,
            'validation_periods': [{
                'start': test_df['date'].min() if 'date' in test_df.columns else None,
                'end': test_df['date'].max() if 'date' in test_df.columns else None,
            }],
            'classification_calibration': calibration_metrics,
        },
    )

    return {
        'code': code,
        'model_version': model_version,
        'model_type': 'classification',
        'target': CLASSIFICATION_TARGET,
        'accuracy': metrics['accuracy'],
        'precision_macro': metrics['precision_macro'],
        'recall_macro': metrics['recall_macro'],
        'f1_macro': metrics['f1_macro'],
        'logloss': metrics['logloss'],
        'brier_score': metrics['brier_score'],
        'class_distribution': metrics['class_distribution'],
        'train_size': len(X_train),
        'test_size': len(X_test),
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols
    }


def train_separate_classification_models(codes: List[str],
                                        feature_cols: List[str] = ALL_FEATURE_COLS,
                                        use_scaling: bool = True,
                                        min_date: Optional[str] = None,
                                        max_date: Optional[str] = None) -> List[Dict]:
    """
    训练所有ETF的分类模型（separate模式）

    Args:
        codes: ETF代码列表
        feature_cols: 特征列列表
        use_scaling: 是否使用标准化
        min_date: 最小日期（None表示使用DB_MIN_DATE）
        max_date: 最大训练日期（None表示使用所有数据）

    Returns:
        训练结果列表
    """
    logger.info("="*60)
    logger.info("分类模型训练 - separate模式")
    logger.info("="*60)

    # 确定使用的数据起始日期
    if min_date is None:
        min_date = DB_MIN_DATE

    # 确定训练数据结束日期
    if max_date:
        logger.info(f"训练数据截止日期: {max_date}")

    all_results = []

    for i, code in enumerate(codes, 1):
        logger.info(f"\n[{i}/{len(codes)}] 处理ETF: {code}")

        try:
            # 读取数据
            df = read_etf_data(code, min_date=min_date, use_advanced=True)
            logger.info(f"  原始数据: {len(df)} 条")

            if len(df) < 100:
                logger.warning(f"  数据不足，跳过")
                continue

            # 创建分类目标
            df = create_classification_target(df, days=5)
            logger.info(f"  计算分类目标后: {len(df)} 条")

            # 清洗数据
            df = clean_data(df, min_history_days=60)
            logger.info(f"  清洗后: {len(df)} 条")

            if len(df) < 50:
                logger.warning(f"  清洗后数据不足，跳过")
                continue

            # 如果指定了max_date，只使用该日期之前的数据进行训练
            if max_date:
                df_before_max = df[df['date'] <= max_date].copy()
                df_after_max = df[df['date'] > max_date].copy()

                if len(df_before_max) < 50:
                    logger.warning(f"  max_date之前的数据不足，跳过")
                    continue

                # 使用max_date之前的数据训练，之后的数据测试
                train_df = df_before_max
                test_df = df_after_max if len(df_after_max) > 0 else pd.DataFrame()
                logger.info(f"  训练集: {len(train_df)} 条（截止{max_date}），测试集: {len(test_df)} 条（{max_date}之后）")
            else:
                # 划分训练/测试集（默认按时间顺序70/30划分）
                train_df, test_df = split_data(df, train_ratio=0.7)
                logger.info(f"  训练集: {len(train_df)} 条，测试集: {len(test_df)} 条")

            # 训练分类模型
            result = train_single_classification_model(
                code, train_df, test_df,
                feature_cols=feature_cols,
                use_scaling=use_scaling
            )

            all_results.append(result)

        except Exception as e:
            logger.error(f"  ETF {code} 处理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    return all_results


def evaluate_classification_results(all_results: List[Dict]) -> Dict:
    """
    汇总分类评估结果

    Args:
        all_results: 分类模型训练结果列表

    Returns:
        统计字典
    """
    accuracy_values = [r['accuracy'] for r in all_results if 'accuracy' in r]
    f1_values = [r['f1_macro'] for r in all_results if 'f1_macro' in r]

    return {
        'count': len(all_results),
        'avg_accuracy': np.mean(accuracy_values) if accuracy_values else 0,
        'avg_f1_macro': np.mean(f1_values) if f1_values else 0,
        'accuracy_range': (np.min(accuracy_values), np.max(accuracy_values)) if accuracy_values else (0, 0)
    }


# ============================================================================
# 模型架构：统一模型（所有ETF共享）
# ============================================================================

def train_unified_model(codes: List[str],
                        target_type: str = 'excess',
                        feature_cols: List[str] = ALL_FEATURE_COLS,
                        use_scaling: bool = True,
                        common_start_date: str = '2016-01-01') -> Dict:
    """
    统一模型（所有ETF共享一个模型）

    Args:
        codes: ETF代码列表
        target_type: 目标类型 ('absolute' 或 'excess')
                      - unified 模型默认使用 'excess'（超额收益）
        feature_cols: 特征列列表
        use_scaling: 是否使用标准化
        common_start_date: 统一数据起始日期（默认2016-01-01）

    Returns:
        训练结果字典
    """
    logger.info("="*60)
    logger.info("统一模型训练（所有ETF共享）")
    logger.info(f"目标类型: {target_type}")
    if target_type == 'excess':
        logger.info(f"使用超额收益目标，相对于基准ETF ({BENCHMARK_ETF})")
        logger.info(f"统一数据起始日期: {common_start_date}")
    else:
        logger.info("使用绝对收益目标")
    logger.info("="*60)

    # 获取基准收益率（在统一日期范围内）
    benchmark_returns = None
    if target_type == 'excess':
        logger.info(f"获取基准收益率 ({BENCHMARK_ETF}) 从 {common_start_date}...")
        benchmark_returns = get_benchmark_returns(min_date=common_start_date)
        if benchmark_returns is None or benchmark_returns.empty:
            logger.error("无法获取基准收益率")
            return {}

    # 合并所有ETF数据（使用统一起始日期）
    all_dfs = []
    for code in codes:
        df = read_etf_data(code, min_date=common_start_date, use_advanced=True)
        if len(df) < 100:
            logger.warning(f"  ETF {code} 数据不足，跳过")
            continue

        df = create_target(df, target_type=target_type, benchmark_returns=benchmark_returns)
        df = clean_data(df, min_history_days=60)

        # 添加ETF编码（作为特征）
        df['etf_code'] = code

        all_dfs.append(df)

    if not all_dfs:
        logger.error("没有足够的数据进行训练")
        return {}

    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"合并后数据: {len(combined_df)} 条")

    # 对ETF编码进行Label Encoding (在划分之前)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(combined_df['etf_code'])  # 在所有数据上拟合
    combined_df['etf_code_encoded'] = le.transform(combined_df['etf_code'])

    # 划分训练/测试集
    train_df, test_df = split_data(combined_df, train_ratio=0.7)

    # 特征列（包含ETF编码）
    extended_features = feature_cols + ['etf_code_encoded']

    # 训练
    result = train_single_model(
        'unified', train_df, test_df,
        target_col=TARGET_EXCESS_RETURN if target_type == 'excess' else TARGET_ABSOLUTE_RETURN,
        feature_cols=extended_features,
        use_scaling=use_scaling
    )

    result['label_encoder'] = le
    result['codes'] = codes
    result['common_start_date'] = common_start_date

    return result


# ============================================================================
# 模型架构：两阶段模型（预测所有ETF，然后排序选择）
# ============================================================================

def train_two_stage_model(codes: List[str],
                         target_type: str = 'excess',
                         feature_cols: List[str] = ALL_FEATURE_COLS,
                         use_scaling: bool = True,
                         common_start_date: str = '2016-01-01') -> Dict:
    """
    两阶段模型：预测所有ETF的收益率，然后排序选择

    Args:
        codes: ETF代码列表
        target_type: 目标类型 ('absolute' 或 'excess')
        feature_cols: 特征列列表
        use_scaling: 是否使用标准化
        common_start_date: 统一数据起始日期（默认2016-01-01）

    Returns:
        训练结果字典
    """
    logger.info("="*60)
    logger.info("两阶段模型训练")
    logger.info("阶段1: 预测所有ETF的收益率")
    logger.info("阶段2: 排序并选择TOP ETF")
    logger.info("="*60)

    # 阶段1：训练预测模型（类似unified）
    result = train_unified_model(codes, target_type, feature_cols, use_scaling, common_start_date)
    result['model_type'] = 'two-stage'

    # 两阶段模型的特殊性：
    # - 需要为所有ETF同时预测
    # - 预测后按收益率排序选择TOP N

    return result


# ============================================================================
# 评估和报告
# ============================================================================

def evaluate_results(all_results: List[Dict]) -> Dict:
    """汇总评估所有结果"""
    r2_values = [r['r2'] for r in all_results if 'r2' in r]
    mae_values = [r['mae'] for r in all_results if 'mae' in r]
    rmse_values = [r['rmse'] for r in all_results if 'rmse' in r]

    return {
        'count': len(all_results),
        'avg_r2': np.mean(r2_values) if r2_values else 0,
        'avg_mae': np.mean(mae_values) if mae_values else 0,
        'avg_rmse': np.mean(rmse_values) if rmse_values else 0,
        'r2_range': (np.min(r2_values), np.max(r2_values)) if r2_values else (0, 0)
    }


def create_summary_csv(results, output_path: str, etf_names: Dict[str, str], mode: str = 'normal'):
    """
    创建汇总CSV

    Args:
        results: 训练结果（list of dict for separate, dict for unified/two-stage, or mixed list for hybrid）
        output_path: 输出路径
        etf_names: ETF名称映射
        mode: 'normal'（单一模式）或 'hybrid'（混合模式）
    """
    summary_rows = []

    if mode == 'hybrid':
        # 混合模式：分别处理单独模型和统一模型
        for result in results:
            if isinstance(result, dict):
                if result.get('code') == 'unified':
                    # 统一模型
                    row = {
                        'code': result['code'],
                        'name': etf_names.get(result['code'], result['code']),
                        'model_version': result['model_version'],
                        'model_type': result.get('model_type', 'unified'),
                        'target': result['target'],
                        'r2': round(result['r2'], 4) if 'r2' in result else None,
                        'mae': round(result['mae'], 4) if 'mae' in result else None,
                        'rmse': round(result['rmse'], 4) if 'rmse' in result else None,
                        'mape': round(result['mape'], 4) if 'mape' in result else None,
                        'accuracy': round(result['accuracy'], 4) if 'accuracy' in result else None,
                        'precision_macro': round(result['precision_macro'], 4) if 'precision_macro' in result else None,
                        'recall_macro': round(result['recall_macro'], 4) if 'recall_macro' in result else None,
                        'f1_macro': round(result['f1_macro'], 4) if 'f1_macro' in result else None,
                        'train_size': result['train_size'],
                        'test_size': result['test_size'],
                        'prediction_date': datetime.now().isoformat()
                    }
                else:
                    # 单独模型
                    model_type = result.get('model_type', 'separate')
                    row = {
                        'code': result['code'],
                        'name': etf_names.get(result['code'], result['code']),
                        'model_version': result['model_version'],
                        'model_type': model_type,
                        'target': result['target'],
                        'r2': round(result['r2'], 4) if 'r2' in result else None,
                        'mae': round(result['mae'], 4) if 'mae' in result else None,
                        'rmse': round(result['rmse'], 4) if 'rmse' in result else None,
                        'mape': round(result['mape'], 4) if 'mape' in result else None,
                        'accuracy': round(result['accuracy'], 4) if 'accuracy' in result else None,
                        'precision_macro': round(result['precision_macro'], 4) if 'precision_macro' in result else None,
                        'recall_macro': round(result['recall_macro'], 4) if 'recall_macro' in result else None,
                        'f1_macro': round(result['f1_macro'], 4) if 'f1_macro' in result else None,
                        'train_size': result['train_size'],
                        'test_size': result['test_size'],
                        'prediction_date': datetime.now().isoformat()
                    }
                summary_rows.append(row)
    else:
        # 普通模式
        if isinstance(results, dict) and 'model_type' in results:
            # 统一模型/两阶段模型
            row = {
                'code': results['code'],
                'name': etf_names.get(results['code'], results['code']),
                'model_version': results['model_version'],
                'model_type': results.get('model_type', 'unified'),
                'target': results['target'],
                'r2': round(results['r2'], 4) if 'r2' in results else None,
                'mae': round(results['mae'], 4) if 'mae' in results else None,
                'rmse': round(results['rmse'], 4) if 'rmse' in results else None,
                'mape': round(results['mape'], 4) if 'mape' in results else None,
                'accuracy': round(results['accuracy'], 4) if 'accuracy' in results else None,
                'precision_macro': round(results['precision_macro'], 4) if 'precision_macro' in results else None,
                'recall_macro': round(results['recall_macro'], 4) if 'recall_macro' in results else None,
                'f1_macro': round(results['f1_macro'], 4) if 'f1_macro' in results else None,
                'train_size': results['train_size'],
                'test_size': results['test_size'],
                'prediction_date': datetime.now().isoformat()
            }
            summary_rows.append(row)
        else:
            # 单独模型
            for result in results:
                model_type = result.get('model_type', 'separate')
                row = {
                    'code': result['code'],
                    'name': etf_names.get(result['code'], result['code']),
                    'model_version': result['model_version'],
                    'model_type': model_type,
                    'target': result['target'],
                    'r2': round(result['r2'], 4) if 'r2' in result else None,
                    'mae': round(result['mae'], 4) if 'mae' in result else None,
                    'rmse': round(result['rmse'], 4) if 'rmse' in result else None,
                    'mape': round(result['mape'], 4) if 'mape' in result else None,
                    'accuracy': round(result['accuracy'], 4) if 'accuracy' in result else None,
                    'precision_macro': round(result['precision_macro'], 4) if 'precision_macro' in result else None,
                    'recall_macro': round(result['recall_macro'], 4) if 'recall_macro' in result else None,
                    'f1_macro': round(result['f1_macro'], 4) if 'f1_macro' in result else None,
                    'train_size': result['train_size'],
                    'test_size': result['test_size'],
                    'prediction_date': datetime.now().isoformat()
                }
                summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path, index=False)


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ETF模型训练 - 支持多种架构和目标类型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 混合策略：单独模型(绝对收益) + 统一模型(超额收益)
  python -m my_etf.models.train --hybrid

  # 每ETF单独训练（默认绝对收益）
  python -m my_etf.models.train --model-type separate

  # 统一模型（默认超额收益 + 2016-01-01起）
  python -m my_etf.models.train --model-type unified

  # 两阶段模型（默认超额收益）
  python -m my_etf.models.train

  # 自定义参数
  python -m my_etf.models.train --model-type unified --common-start-date 2017-01-01
  python -m my_etf.models.train --model-type separate --target-type absolute --min-date 2015-01-01
        """
    )

    # 模型架构选择
    parser.add_argument('--model-type', type=str, default='two-stage',
                        choices=['separate', 'unified', 'two-stage'],
                        help='模型架构类型（默认: two-stage）')

    # 混合策略
    parser.add_argument('--hybrid', action='store_true',
                        help='混合策略：单独模型(绝对收益) + 统一模型(超额收益)')

    # 分类模式
    parser.add_argument('--classification', action='store_true',
                        help='使用分类模式训练（将收益率分为4个类别）')

    # 目标类型
    parser.add_argument('--target-type', type=str, default=None,
                        choices=['absolute', 'excess'],
                        help='目标类型（--hybrid模式下不适用）')
    parser.add_argument('--separate-target', type=str, default='absolute',
                        choices=['absolute', 'excess'],
                        help='单独模型的目标类型（默认: absolute）')
    parser.add_argument('--unified-target', type=str, default='excess',
                        choices=['absolute', 'excess'],
                        help='统一模型的目标类型（默认: excess）')

    # 特征选项
    parser.add_argument('--no-advanced', action='store_true',
                        help='不使用高级特征（仅使用基础特征）')
    parser.add_argument('--no-scaling', action='store_true',
                        help='不使用特征标准化')

    # 日期选项
    parser.add_argument('--min-date', type=str, default=None,
                        help='单独模型的最小日期（默认: DB_MIN_DATE）')
    parser.add_argument('--max-date', type=str, default=None,
                        help='训练数据截止日期（用于时间序列分割，训练使用<=该日期的数据，测试使用>该日期的数据）')
    parser.add_argument('--common-start-date', type=str, default='2016-01-01',
                        help='统一模型的统一起始日期（默认: 2016-01-01）')

    parser.add_argument('--benchmark', type=str, default=BENCHMARK_ETF,
                        help=f'基准ETF代码（默认: {BENCHMARK_ETF}）')

    args = parser.parse_args()

    # 配置
    feature_cols = FEATURE_COLS if args.no_advanced else ALL_FEATURE_COLS
    from ..config import get_etf_codes
    codes = get_etf_codes()

    # 混合策略模式
    if args.hybrid:
        print("="*80)
        print("ETF模型训练 - 混合策略")
        print("="*80)
        print(f"ETF数量: {len(codes)}")
        print(f"特征数量: {len(feature_cols)}")
        print(f"使用标准化: {not args.no_scaling}")
        print("\n策略1: 单独模型")
        print(f"  目标类型: {args.separate_target}")
        print(f"  数据范围: 每个ETF自己的历史数据")
        if args.min_date:
            print(f"  起始日期: {args.min_date}")
        print("\n策略2: 统一模型")
        print(f"  目标类型: {args.unified_target}")
        print(f"  统一起始日期: {args.common_start_date}")
        print("="*80)
    elif args.classification:
        # 分类模式
        print("="*80)
        print("ETF分类模型训练 - separate模式")
        print("="*80)
        print(f"ETF数量: {len(codes)}")
        print(f"特征数量: {len(feature_cols)}")
        print(f"使用标准化: {not args.no_scaling}")
        print(f"分类配置: 4个类别")
        print("  类别0: < -5% (大幅下跌)")
        print("  类别1: -5% ~ 0% (小幅下跌)")
        print("  类别2: 0% ~ 5% (小幅上涨)")
        print("  类别3: > 5% (大幅上涨)")
        print("="*80)
    else:
        # 单一模型模式
        target_type = args.target_type if args.target_type else (
            'absolute' if args.model_type == 'separate' else 'excess'
        )
        print("="*80)
        print("ETF模型训练")
        print("="*80)
        print(f"ETF数量: {len(codes)}")
        print(f"模型架构: {args.model_type}")
        print(f"目标类型: {target_type}")
        print(f"特征数量: {len(feature_cols)}")
        print(f"使用标准化: {not args.no_scaling}")
        if args.model_type in ['unified', 'two-stage']:
            print(f"统一起始日期: {args.common_start_date}")
        print("="*80)

    # 加载ETF名称
    etf_names_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  '..', '..', 'reports', 'etf_names.json')
    try:
        with open(etf_names_path, 'r', encoding='utf-8') as f:
            etf_names = json.load(f)
    except FileNotFoundError:
        etf_names = {}

    # 创建输出目录
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    SUMMARY_DIR = os.path.join(DATA_DIR, 'summary')
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    all_results = []

    # 分类模式：训练分类模型
    if args.classification:
        print(f"\n{'='*80}")
        print("训练分类模型")
        print(f"{'='*80}")
        results = train_separate_classification_models(
            codes, feature_cols=feature_cols, use_scaling=not args.no_scaling,
            min_date=args.min_date, max_date=args.max_date
        )
        all_results = results

    # 混合策略：训练单独模型 + 统一模型
    elif args.hybrid:
        # 策略1：单独模型（绝对收益）
        print(f"\n{'='*80}")
        print("训练策略1: 单独模型（绝对收益）")
        print(f"{'='*80}")
        separate_results = train_separate_models(
            codes, target_type=args.separate_target,
            feature_cols=feature_cols, use_scaling=not args.no_scaling,
            min_date=args.min_date
        )
        all_results.extend(separate_results)

        # 策略2：统一模型（超额收益）
        print(f"\n{'='*80}")
        print("训练策略2: 统一模型（超额收益）")
        print(f"{'='*80}")
        unified_result = train_unified_model(
            codes, target_type=args.unified_target,
            feature_cols=feature_cols, use_scaling=not args.no_scaling,
            common_start_date=args.common_start_date
        )
        all_results.append(unified_result)
    else:
        # 单一模型模式
        target_type = args.target_type if args.target_type else (
            'absolute' if args.model_type == 'separate' else 'excess'
        )

        if args.model_type == 'separate':
            results = train_separate_models(
                codes, target_type=target_type,
                feature_cols=feature_cols, use_scaling=not args.no_scaling,
                min_date=args.min_date
            )
            all_results = results
        elif args.model_type == 'unified':
            result = train_unified_model(
                codes, target_type=target_type,
                feature_cols=feature_cols, use_scaling=not args.no_scaling,
                common_start_date=args.common_start_date
            )
            all_results = [result]
        else:  # two-stage
            result = train_two_stage_model(
                codes, target_type=target_type,
                feature_cols=feature_cols, use_scaling=not args.no_scaling,
                common_start_date=args.common_start_date
            )
            all_results = [result]

    # 评估结果
    if args.classification:
        # 分类模式评估
        if isinstance(all_results, list) and len(all_results) > 1:
            summary = evaluate_classification_results(all_results)
            print(f"\n{'='*80}")
            print("分类模型整体统计")
            print(f"{'='*80}")
            print(f"处理数量: {summary['count']}")
            print(f"平均 Accuracy: {summary['avg_accuracy']:.4f}")
            print(f"平均 F1-Macro: {summary['avg_f1_macro']:.4f}")
            print(f"Accuracy 范围: {summary['accuracy_range'][0]:.4f} ~ {summary['accuracy_range'][1]:.4f}")
        elif len(all_results) == 1:
            result = all_results[0]
            print(f"\n{'='*80}")
            print("分类模型评估")
            print(f"{'='*80}")
            print(f"Accuracy: {result['accuracy']:.4f}")
            print(f"Precision (Macro): {result['precision_macro']:.4f}")
            print(f"Recall (Macro): {result['recall_macro']:.4f}")
            print(f"F1 (Macro): {result['f1_macro']:.4f}")
    elif args.hybrid:
        # 混合策略评估
        separate_results = [r for r in all_results if isinstance(r, dict) and r.get('code') != 'unified']
        unified_results = [r for r in all_results if isinstance(r, dict) and r.get('code') == 'unified']

        if separate_results:
            summary = evaluate_results(separate_results)
            print(f"\n{'='*80}")
            print("策略1评估：单独模型")
            print(f"{'='*80}")
            print(f"处理数量: {summary['count']}")
            print(f"平均 R2: {summary['avg_r2']:.4f}")
            print(f"平均 MAE: {summary['avg_mae']:.4f}%")
            print(f"平均 RMSE: {summary['avg_rmse']:.4f}%")

        if unified_results:
            print(f"\n{'='*80}")
            print("策略2评估：统一模型")
            print(f"{'='*80}")
            print(f"R2: {unified_results[0]['r2']:.4f}")
            print(f"MAE: {unified_results[0]['mae']:.4f}%")
            print(f"RMSE: {unified_results[0]['rmse']:.4f}%")
    elif isinstance(all_results, list) and len(all_results) > 1:
        # 多个单独模型
        summary = evaluate_results(all_results)
        print(f"\n{'='*80}")
        print("整体统计")
        print(f"{'='*80}")
        print(f"处理数量: {summary['count']}")
        print(f"平均 R2: {summary['avg_r2']:.4f}")
        print(f"平均 MAE: {summary['avg_mae']:.4f}%")
        print(f"平均 RMSE: {summary['avg_rmse']:.4f}%")
        print(f"R2 范围: {summary['r2_range'][0]:.4f} ~ {summary['r2_range'][1]:.4f}")
    else:
        # 单个模型
        result = all_results[0]
        print(f"\n{'='*80}")
        print("模型评估")
        print(f"{'='*80}")
        print(f"R2: {result['r2']:.4f}")
        print(f"MAE: {result['mae']:.4f}%")
        print(f"RMSE: {result['rmse']:.4f}%")

    # 保存汇总
    summary_path = os.path.join(SUMMARY_DIR, 'training_summary.csv')
    mode = 'hybrid' if args.hybrid else 'normal'
    create_summary_csv(all_results, summary_path, etf_names, mode=mode)
    print(f"\n汇总CSV: {summary_path}")

    print(f"\n{'='*80}")
    print("完成！")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
