"""
ETF分类模型训练模块（带验证集的超参数调优）
支持：
1. 从训练集分出验证集进行参数调优
2. 网格搜索优化模型参数
3. 使用调优后的参数在全部训练数据上重新训练
4. 在测试集上评估最终模型
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from itertools import product

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# 强制重新导入database模块（避免缓存问题）
import importlib
import sys
if 'my_etf.utils' in sys.modules:
    del sys.modules['my_etf.utils']
    import my_etf.utils

from ..config import (
    MODELS_DIR, DATA_DIR, ALL_FEATURE_COLS,
    CLASSIFICATION_TARGET, CLASSIFICATION_CONFIG, CLASSIFICATION_MODEL_PARAMS
)
from ..utils.database import read_etf_data
from ..utils.logger import setup_logger
from .train import create_classification_target, clean_data

logger = setup_logger("etf_train_validation", "train_validation.log")


# ============================================================================
# 数据划分
# ============================================================================

def split_data_with_validation(df: pd.DataFrame,
                             max_date: Optional[str] = None,
                             val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    将数据按日期和比例划分为：训练集 + 验证集 + 测试集

    Args:
        df: 输入数据（必须包含 date 列）
        max_date: 训练集和验证集的截止日期（格式：YYYYMMDD，None则使用train_ratio按比例划分）
        val_ratio: 验证集占训练+验证数据的比例（默认0.2，即20%）

    Returns:
        (train_df, val_df, test_df) 元组
    """
    if max_date is not None:
        # 使用日期划分：max_date之前为训练+验证，之后为测试
        # 确保日期列为字符串格式
        df = df.copy()
        df['date_str'] = df['date'].astype(str).str.replace('-', '')
        max_date_str = max_date.replace('-', '')

        train_val_df = df[df['date_str'] <= max_date_str].copy()
        test_df = df[df['date_str'] > max_date_str].copy()

        # 删除临时列
        train_val_df = train_val_df.drop(columns=['date_str'])
        test_df = test_df.drop(columns=['date_str'])

        logger.info(f"使用日期划分：训练+验证集截止日期 {max_date}")
    else:
        # 使用比例划分（保留原逻辑）
        total_len = len(df)
        split_idx = int(total_len * 0.7)
        train_val_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        logger.info(f"使用比例划分：训练+验证集占总数据的 70%")

    # 从 train+val 中按时间顺序分出 val
    train_val_len = len(train_val_df)
    train_split_idx = int(train_val_len * (1 - val_ratio))

    train_df = train_val_df.iloc[:train_split_idx].copy()
    val_df = train_val_df.iloc[train_split_idx:].copy()

    total_len = len(df)
    logger.info(f"数据划分: {len(train_df)} 训练 + {len(val_df)} 验证 + {len(test_df)} 测试 "
                f"({len(train_df)/total_len:.1%} + {len(val_df)/total_len:.1%} + {len(test_df)/total_len:.1%})")

    return train_df, val_df, test_df


def split_data_mixed(df: pd.DataFrame,
                    max_date: str,
                    val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    混合划分：按日期和比例结合的方式划分数据

    步骤：
        1. 按日期划分：max_date之前为 train+val，之后为 test
        2. 从 train+val 中按时间顺序分出验证集

    Args:
        df: 输入数据（必须包含 date 列）
        max_date: 训练集和验证集的截止日期（格式：YYYYMMDD）
        val_ratio: 验证集占训练+验证数据的比例（默认 0.2）

    Returns:
        (train_df, val_df, test_df) 元组
    """
    df = df.copy()
    df['date_str'] = df['date'].astype(str).str.replace('-', '')
    max_date_str = max_date.replace('-', '')

    # 步骤 1：按日期划分
    train_val_df = df[df['date_str'] <= max_date_str].copy()
    test_df = df[df['date_str'] > max_date_str].copy()

    # 删除临时列
    train_val_df = train_val_df.drop(columns=['date_str'])
    test_df = test_df.drop(columns=['date_str'])

    logger.info(f"混合划分 - 步骤 1：按日期划分")
    logger.info(f"  训练+验证集截止日期: {max_date}")
    logger.info(f"  训练+验证数据量: {len(train_val_df)}, 测试数据量: {len(test_df)}")

    # 步骤 2：从 train+val 中按时间顺序分出验证集
    train_val_len = len(train_val_df)
    train_split_idx = int(train_val_len * (1 - val_ratio))

    train_df = train_val_df.iloc[:train_split_idx].copy()
    val_df = train_val_df.iloc[train_split_idx:].copy()

    total_len = len(df)
    logger.info(f"混合划分 - 步骤 2：按比例划分验证集")
    logger.info(f"数据划分: {len(train_df)} 训练 + {len(val_df)} 验证 + {len(test_df)} 测试 "
                f"({len(train_df)/total_len:.1%} + {len(val_df)/total_len:.1%} + {len(test_df)/total_len:.1%})")

    return train_df, val_df, test_df


# ============================================================================
# 超参数调优
# ============================================================================

def get_default_param_grid() -> Dict[str, List[Any]]:
    """
    获取默认参数网格（约54个组合）

    Returns:
        参数网格字典
    """
    return {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'min_child_weight': [1, 3, 5],
    }


def get_conservative_param_grid() -> Dict[str, List[Any]]:
    """
    获取保守参数网格（96个组合）

    Returns:
        参数网格字典
    """
    return {
        'n_estimators': [100, 200],
        'max_depth': [3, 4, 5],              # 从 [4,6,8] 改为 [3,4,5]
        'learning_rate': [0.05, 0.1],          # 从 [0.05,0.1,0.2] 改为 [0.05,0.1]
        'min_child_weight': [3, 5],              # 从 [1,3,5] 改为 [3,5]
        'subsample': [0.8, 0.9],               # 新增（正则化）
        'colsample_bytree': [0.8, 0.9],         # 新增（正则化）
    }


def simple_grid_search(X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_val: np.ndarray,
                    y_val: np.ndarray,
                    param_grid: Optional[Dict[str, List[Any]]] = None,
                    scaler: Optional[StandardScaler] = None) -> Tuple[Dict[str, Any], List[Dict]]:
    """
    使用简单的网格搜索进行参数调优

    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征
        y_val: 验证标签
        param_grid: 参数网格字典（None使用默认）
        scaler: 标准化器（用于训练时缩放）

    Returns:
        (best_params, all_results) 元组
        - best_params: 最佳参数字典
        - all_results: 所有尝试的参数及其结果列表
    """
    if param_grid is None:
        param_grid = get_default_param_grid()

    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))

    logger.info(f"开始网格搜索，共 {len(all_combinations)} 个参数组合...")

    all_results = []
    best_accuracy = 0
    best_params = None

    # 基础参数（不参与调优的部分）
    base_params = CLASSIFICATION_MODEL_PARAMS.copy()
    # 移除参与调优的参数
    for param_name in param_names:
        base_params.pop(param_name, None)

    for i, combination in enumerate(all_combinations, 1):
        # 构建当前参数组合
        params = base_params.copy()
        params.update(dict(zip(param_names, combination)))

        # 训练模型
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        # 在验证集上评估
        y_val_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred)

        result = {
            'params': params.copy(),
            'accuracy': accuracy,
            'rank': i
        }
        all_results.append(result)

        logger.debug(f"  组合 {i}/{len(all_combinations)}: {params} -> accuracy={accuracy:.4f}")

        # 更新最佳参数
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params.copy()

    # 按accuracy降序排序
    all_results = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)
    # 重新编号rank
    for idx, result in enumerate(all_results, 1):
        result['rank'] = idx

    logger.info(f"网格搜索完成！最佳参数: {best_params}, 验证集准确率: {best_accuracy:.4f}")

    return best_params, all_results


# ============================================================================
# 模型训练（带验证集）
# ============================================================================

def train_single_classification_with_validation(
    code: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    use_scaling: bool = True,
    enable_tuning: bool = True,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    model_suffix: str = '_validation'
) -> Dict:
    """
    训练单个ETF的分类模型（带验证集）

    Args:
        code: ETF代码
        train_df: 训练集
        val_df: 验证集
        test_df: 测试集
        feature_cols: 特征列列表
        use_scaling: 是否使用标准化
        enable_tuning: 是否启用超参数调优
        param_grid: 自定义参数网格（None使用默认）
        model_suffix: 模型文件后缀（默认 '_validation'）

    Returns:
        训练结果字典
    """
    # 准备数据
    X_train = train_df[feature_cols]
    y_train = train_df[CLASSIFICATION_TARGET]

    X_val = val_df[feature_cols]
    y_val = val_df[CLASSIFICATION_TARGET]

    X_test = test_df[feature_cols]
    y_test = test_df[CLASSIFICATION_TARGET]

    # 移除NaN
    mask_train = ~X_train.isnull().any(axis=1) & ~y_train.isnull()
    X_train = X_train[mask_train]
    y_train = y_train[mask_train]

    mask_val = ~X_val.isnull().any(axis=1) & ~y_val.isnull()
    X_val = X_val[mask_val]
    y_val = y_val[mask_val]

    mask_test = ~X_test.isnull().any(axis=1) & ~y_test.isnull()
    X_test = X_test[mask_test]
    y_test = y_test[mask_test]

    # 标准化
    scaler = None
    if use_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train.values
        X_val_scaled = X_val.values
        X_test_scaled = X_test.values

    # 超参数调优（如果启用）
    tuning_results = None
    best_params = None

    if enable_tuning:
        logger.info(f"ETF {code}: 开始超参数调优...")
        best_params, tuning_results = simple_grid_search(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            param_grid=param_grid,
            scaler=scaler
        )
    else:
        best_params = CLASSIFICATION_MODEL_PARAMS.copy()
        logger.info(f"ETF {code}: 使用默认参数，不调优")

    # 合并训练集和验证集，使用全部训练数据重新训练
    X_train_all = np.vstack([X_train_scaled, X_val_scaled])
    y_train_all = np.concatenate([y_train.values, y_val.values])

    logger.info(f"ETF {code}: 使用最佳参数在 {len(X_train_all)} 条数据上重新训练...")

    # 创建最终模型
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train_all, y_train_all)

    # 在测试集上评估
    y_test_pred = final_model.predict(X_test_scaled)
    y_test_proba = final_model.predict_proba(X_test_scaled)

    metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision_macro': precision_score(y_test, y_test_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_test_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_test_pred, average='macro', zero_division=0),
    }

    logger.info(f"ETF {code} 测试集评估: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")

    # 保存模型
    model_version = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_model_with_validation(
        code, final_model, scaler, model_version,
        feature_cols, metrics, best_params, tuning_results,
        model_suffix=model_suffix
    )

    return {
        'code': code,
        'model_version': model_version,
        'model_type': 'classification_with_validation',
        'target': CLASSIFICATION_TARGET,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'best_params': best_params,
        'enable_tuning': enable_tuning,
        'metrics': metrics,
        'tuning_results': tuning_results,
        'model': final_model,
        'scaler': scaler,
        'feature_cols': feature_cols
    }


def train_all_classification_with_validation(
    codes: List[str],
    feature_cols: List[str] = ALL_FEATURE_COLS,
    use_scaling: bool = True,
    enable_tuning: bool = True,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    max_date: Optional[str] = None,
    val_ratio: float = 0.1,
    min_date: Optional[str] = None,
    use_mixed_split: bool = False,
    use_conservative_grid: bool = False
) -> List[Dict]:
    """
    训练所有ETF的分类模型（带验证集）

    Args:
        codes: ETF代码列表
        feature_cols: 特征列列表
        use_scaling: 是否使用标准化
        enable_tuning: 是否启用超参数调优
        param_grid: 自定义参数网格（None使用默认）
        max_date: 训练+验证集的截止日期（格式：YYYYMMDD，None则使用train_ratio按比例划分）
        val_ratio: 验证集占训练+验证数据的比例（默认0.2，即20%）
        min_date: 最小日期（None表示使用DB_MIN_DATE）
        use_mixed_split: 是否使用混合划分（日期 + 比例结合）
        use_conservative_grid: 是否使用保守参数网格（96个组合）

    Returns:
        训练结果列表
    """
    from ..config import DB_MIN_DATE

    logger.info("="*60)
    logger.info("分类模型训练 - 带验证集的超参数调优")
    logger.info("="*60)
    logger.info(f"ETF数量: {len(codes)}")
    logger.info(f"特征数量: {len(feature_cols)}")
    if use_mixed_split:
        logger.info(f"数据划分: 混合划分（日期 + 比例）")
        logger.info(f"  训练+验证集截止日期: {max_date}")
        logger.info(f"  验证集占训练+验证的 {val_ratio*100:.0f}%")
    elif max_date is not None:
        logger.info(f"数据划分: 使用日期划分，训练+验证集截止日期 {max_date}")
        logger.info(f"验证集占训练+验证的 {val_ratio*100:.0f}%")
    else:
        logger.info(f"数据划分: 使用比例划分，训练+验证集占总数据的 70%")
        logger.info(f"验证集占训练+验证的 {val_ratio*100:.0f}%")
    logger.info(f"超参数调优: {'启用' if enable_tuning else '禁用'}")
    logger.info(f"参数网格: {'保守' if use_conservative_grid else '默认'}")
    logger.info("="*60)

    # 确定使用的数据起始日期
    if min_date is None:
        min_date = DB_MIN_DATE

    # 确定参数网格
    if use_conservative_grid:
        param_grid = get_conservative_param_grid()
        logger.info(f"使用保守参数网格（96个组合）")
    elif param_grid is None:
        param_grid = get_default_param_grid()
        logger.info(f"使用默认参数网格（约54个组合）")

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

            # 划分训练/验证/测试集
            if use_mixed_split:
                if max_date is None:
                    logger.warning("  use_mixed_split=True 但未提供 max_date，将回退到日期划分")
                    train_df, val_df, test_df = split_data_with_validation(
                        df, max_date=max_date, val_ratio=val_ratio
                    )
                else:
                    train_df, val_df, test_df = split_data_mixed(
                        df, max_date=max_date, val_ratio=val_ratio
                    )
            else:
                train_df, val_df, test_df = split_data_with_validation(
                    df, max_date=max_date, val_ratio=val_ratio
                )

            if len(train_df) < 30 or len(val_df) < 10 or len(test_df) < 10:
                logger.warning(f"  数据量不足以进行训练/验证/测试，跳过")
                continue

            # 训练分类模型
            # 确定模型后缀
            model_suffix = '_mixed' if use_mixed_split else '_validation'

            result = train_single_classification_with_validation(
                code, train_df, val_df, test_df,
                feature_cols=feature_cols,
                use_scaling=use_scaling,
                enable_tuning=enable_tuning,
                param_grid=param_grid,
                model_suffix=model_suffix
            )

            all_results.append(result)

        except Exception as e:
            logger.error(f"  ETF {code} 处理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    return all_results


# ============================================================================
# 模型保存
# ============================================================================

def save_model_with_validation(
    code: str,
    model: xgb.XGBClassifier,
    scaler: Optional[StandardScaler],
    model_version: str,
    feature_cols: List[str],
    metrics: Dict[str, float],
    best_params: Dict[str, Any],
    tuning_results: Optional[List[Dict]] = None
) -> None:
    """
    保存模型、标准化器、调优结果和元数据

    Args:
        code: ETF代码
        model: 模型对象
        scaler: 标准化器
        model_version: 模型版本号
        feature_cols: 特征列列表
        metrics: 测试集评估指标
        best_params: 最佳参数
        tuning_results: 调优结果（如果启用了调优）
    """
    model_dir = os.path.join(MODELS_DIR, code)
    os.makedirs(model_dir, exist_ok=True)

def save_model_with_validation(
    code: str,
    model: xgb.XGBClassifier,
    scaler: Optional[StandardScaler],
    model_version: str,
    feature_cols: List[str],
    metrics: Dict[str, float],
    best_params: Dict[str, Any],
    tuning_results: Optional[List[Dict]] = None,
    model_suffix: str = '_validation'
) -> None:
    """
    保存模型、标准化器、调优结果和元数据

    Args:
        code: ETF代码
        model: 模型对象
        scaler: 标准化器
        model_version: 模型版本号
        feature_cols: 特征列列表
        metrics: 测试集评估指标
        best_params: 最佳参数
        tuning_results: 调优结果（如果启用了调优）
        model_suffix: 模型文件后缀（默认 '_validation'）
    """
    model_dir = os.path.join(MODELS_DIR, code)
    os.makedirs(model_dir, exist_ok=True)

    # 保存模型（带指定后缀）
    model_path = os.path.join(model_dir, f"{code}_{model_version}{model_suffix}.pkl")
    joblib.dump(model, model_path)
    logger.info(f"  模型已保存: {model_path}")

    # 保存标准化器
    if scaler is not None:
        scaler_path = os.path.join(model_dir, f"{code}_{model_version}{model_suffix}_scaler.pkl")
        joblib.dump(scaler, scaler_path)
        logger.info(f"  标准化器已保存: {scaler_path}")

    # 保存调优结果
    if tuning_results is not None:
        tuning_dir = os.path.join(DATA_DIR, 'tuning_results')
        os.makedirs(tuning_dir, exist_ok=True)
        tuning_path = os.path.join(tuning_dir, f"{code}_{model_version}_tuning.json")

        # 准备保存的数据（移除不可序列化的对象）
        tuning_data = []
        for result in tuning_results:
            tuning_data.append({
                'rank': result['rank'],
                'params': result['params'],
                'accuracy': result['accuracy']
            })

        with open(tuning_path, 'w', encoding='utf-8') as f:
            json.dump(tuning_data, f, indent=2)
        logger.info(f"  调优结果已保存: {tuning_path}")

    # 保存元数据
    metadata = {
        'code': code,
        'model_version': model_version,
        'model_type': 'classification_with_validation',
        'target': CLASSIFICATION_TARGET,
        'features': feature_cols,
        'best_params': best_params,
        'test_metrics': metrics,
        'train_date': datetime.now().isoformat(),
        'has_scaler': scaler is not None,
        'has_tuning_results': tuning_results is not None
    }
    metadata_path = os.path.join(model_dir, 'metadata_validation.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"  元数据已保存: {metadata_path}")


# ============================================================================
# 评估和报告
# ============================================================================

def evaluate_classification_results_with_validation(all_results: List[Dict]) -> Dict:
    """
    汇总分类评估结果（带验证集）

    Args:
        all_results: 分类模型训练结果列表

    Returns:
        统计字典
    """
    accuracy_values = [r['metrics']['accuracy'] for r in all_results if 'metrics' in r]
    f1_values = [r['metrics']['f1_macro'] for r in all_results if 'metrics' in r]

    # 统计参数调优情况
    tuned_count = sum(1 for r in all_results if r.get('enable_tuning', False))

    return {
        'count': len(all_results),
        'avg_accuracy': np.mean(accuracy_values) if accuracy_values else 0,
        'avg_f1_macro': np.mean(f1_values) if f1_values else 0,
        'accuracy_range': (np.min(accuracy_values), np.max(accuracy_values)) if accuracy_values else (0, 0),
        'tuned_count': tuned_count,
        'default_params_count': len(all_results) - tuned_count
    }


def create_summary_csv_with_validation(results: List[Dict],
                                   output_path: str,
                                   etf_names: Dict[str, str]) -> None:
    """
    创建汇总CSV（带验证集版本）

    Args:
        results: 训练结果列表
        output_path: 输出路径
        etf_names: ETF名称映射
    """
    summary_rows = []

    for result in results:
        row = {
            'code': result['code'],
            'name': etf_names.get(result['code'], result['code']),
            'model_version': result['model_version'],
            'model_type': result['model_type'],
            'target': result['target'],
            'accuracy': round(result['metrics']['accuracy'], 4),
            'precision_macro': round(result['metrics']['precision_macro'], 4),
            'recall_macro': round(result['metrics']['recall_macro'], 4),
            'f1_macro': round(result['metrics']['f1_macro'], 4),
            'train_size': result['train_size'],
            'val_size': result['val_size'],
            'test_size': result['test_size'],
            'enable_tuning': result['enable_tuning'],
            'best_n_estimators': result['best_params'].get('n_estimators'),
            'best_max_depth': result['best_params'].get('max_depth'),
            'best_learning_rate': result['best_params'].get('learning_rate'),
            'train_date': datetime.now().isoformat()
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ETF分类模型训练 - 带验证集的超参数调优",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 训练所有分类模型（带调优，使用日期划分）
  python -m my_etf.models.train_classification_with_validation --max-date 20250101

  # 使用混合划分 + 保守参数
  python -m my_etf.models.train_classification_with_validation --use-mixed-split --use-conservative-grid --max-date 20250101

  # 使用混合划分 + 保守参数 + 自定义验证集比例
  python -m my_etf.models.train_classification_with_validation --use-mixed-split --use-conservative-grid --max-date 20250101 --val-ratio 0.15

  # 训练单个ETF
  python -m my_etf.models.train_classification_with_validation --use-mixed-split --use-conservative-grid --max-date 20250101 --code 510050

  # 混合划分 + 默认参数（不使用保守网格）
  python -m my_etf.models.train_classification_with_validation --use-mixed-split --max-date 20250101

  # 原始方法（向后兼容）
  python -m my_etf.models.train_classification_with_validation --max-date 20250101

  # 不调优，使用默认参数
  python -m my_etf.models.train_classification_with_validation --no-tune --max-date 20250101

  # 自定义验证集比例
  python -m my_etf.models.train_classification_with_validation --max-date 20250101 --val-ratio 0.15

  # 自定义参数网格（JSON格式）
  python -m my_etf.models.train_classification_with_validation --max-date 20250101 --param-grid '{"n_estimators": [100, 200], "max_depth": [4, 6]}'

  # 不使用高级特征
  python -m my_etf.models.train_classification_with_validation --max-date 20250101 --no-advanced

  # 不使用特征标准化
  python -m my_etf.models.train_classification_with_validation --max-date 20250101 --no-scaling

  # 使用比例划分（旧方式，不推荐）
  python -m my_etf.models.train_classification_with_validation --train-ratio 0.7
        """
    )

    # 调优选项
    parser.add_argument('--no-tune', action='store_true',
                        help='不进行超参数调优，使用默认参数')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='验证集占训练+验证数据的比例（默认: 0.2）')
    parser.add_argument('--param-grid', type=str, default=None,
                        help='自定义参数网格（JSON格式，例如: \'{"n_estimators": [100, 200], "max_depth": [4, 6]}\'）')

    # ETF选择
    parser.add_argument('--code', type=str, default=None,
                        help='仅训练指定的ETF代码')

    # 特征选项
    parser.add_argument('--no-advanced', action='store_true',
                        help='不使用高级特征（仅使用基础特征）')
    parser.add_argument('--no-scaling', action='store_true',
                        help='不使用特征标准化')

    # 日期选项
    parser.add_argument('--min-date', type=str, default=None,
                        help='最小日期（默认: DB_MIN_DATE）')

    # 数据划分选项
    parser.add_argument('--use-mixed-split', action='store_true',
                        help='使用混合划分（日期 + 比例结合）')
    parser.add_argument('--max-date', type=str, default=None,
                        help='训练+验证集的截止日期（格式: YYYYMMDD，例如: 20250101）')
    parser.add_argument('--use-conservative-grid', action='store_true',
                        help='使用保守参数网格（96个组合 vs 默认约 54 个）')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='训练+验证集占总数据的比例（当--max-date未指定且--use-mixed-split=False时使用，默认：0.7）')

    args = parser.parse_args()

    # 配置
    from ..config import get_etf_codes
    feature_cols = ALL_FEATURE_COLS if not args.no_advanced else FEATURE_COLS

    # 确定要训练的ETF
    if args.code:
        codes = [args.code]
    else:
        codes = get_etf_codes()

    # 解析自定义参数网格
    param_grid = None
    if args.param_grid:
        try:
            param_grid = json.loads(args.param_grid)
            logger.info(f"使用自定义参数网格: {param_grid}")
        except json.JSONDecodeError as e:
            logger.error(f"参数网格JSON解析失败: {e}")
            logger.info("将使用默认参数网格")
            param_grid = None

    # 打印配置信息
    print("="*80)
    print("ETF分类模型训练 - 带验证集的超参数调优")
    print("="*80)
    print(f"ETF数量: {len(codes)}")
    print(f"特征数量: {len(feature_cols)}")
    print(f"使用标准化: {not args.no_scaling}")
    if args.use_mixed_split:
        print(f"数据划分: 混合划分（日期 + 比例）")
        print(f"  训练+验证集截止日期: {args.max_date}")
        print(f"  验证集比例: {args.val_ratio*100:.0f}% (占训练+验证数据)")
    elif args.max_date is not None:
        print(f"数据划分: 使用日期划分，训练+验证集截止日期 {args.max_date}")
        print(f"验证集比例: {args.val_ratio*100:.0f}% (占训练+验证数据)")
    else:
        print(f"数据划分: {args.train_ratio*100:.0f}% (训练+验证) + {(1-args.train_ratio)*100:.0f}% 测试")
        print(f"验证集比例: {args.val_ratio*100:.0f}% (占训练+验证)")
    print(f"超参数调优: {'禁用' if args.no_tune else '启用'}")
    print(f"参数网格: {'保守（96个组合）' if args.use_conservative_grid else '默认（约54个组合）'}")
    print(f"分类配置: 4个类别")
    print("  类别0: < -5% (大幅下跌)")
    print("  类别1: -5% ~ 0% (小幅下跌)")
    print("  类别2: 0% ~ 5% (小幅上涨)")
    print("  类别3: > 5% (大幅上涨)")
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

    # 训练模型
    results = train_all_classification_with_validation(
        codes,
        feature_cols=feature_cols,
        use_scaling=not args.no_scaling,
        enable_tuning=not args.no_tune,
        param_grid=param_grid,
        max_date=args.max_date,
        val_ratio=args.val_ratio,
        min_date=args.min_date,
        use_mixed_split=args.use_mixed_split,
        use_conservative_grid=args.use_conservative_grid
    )

    # 评估结果
    if results:
        summary = evaluate_classification_results_with_validation(results)
        print(f"\n{'='*80}")
        print("分类模型整体统计")
        print(f"{'='*80}")
        print(f"处理数量: {summary['count']}")
        print(f"平均 Accuracy: {summary['avg_accuracy']:.4f}")
        print(f"平均 F1-Macro: {summary['avg_f1_macro']:.4f}")
        print(f"Accuracy 范围: {summary['accuracy_range'][0]:.4f} ~ {summary['accuracy_range'][1]:.4f}")
        print(f"超参数调优: {summary['tuned_count']} 个模型，默认参数: {summary['default_params_count']} 个模型")
    else:
        print("\n没有成功训练的模型")

    # 保存汇总
    summary_suffix = '_mixed_classification.csv' if args.use_mixed_split else '_validation_classification.csv'
    summary_path = os.path.join(SUMMARY_DIR, f'training_summary{summary_suffix}')
    create_summary_csv_with_validation(results, summary_path, etf_names)
    print(f"\n汇总CSV: {summary_path}")

    print(f"\n{'='*80}")
    print("完成！")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
