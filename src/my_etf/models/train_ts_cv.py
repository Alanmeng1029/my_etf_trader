"""
ETF分类模型训练模块 - 时间序列交叉验证（Walk-forward CV）
支持：
1. 使用Walk-forward滑动窗口进行多fold交叉验证
2. 保守参数网格（96个组合）
3. 在所有fold上平均评估参数
4. 使用所有训练数据重新训练最终模型
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
    CLASSIFICATION_TARGET, CLASSIFICATION_CONFIG
)
from ..utils.database import read_etf_data
from ..utils.logger import setup_logger
from .train import create_classification_target, clean_data

logger = setup_logger("etf_train_ts_cv", "train_ts_cv.log")


# ============================================================================
# 保守参数网格
# ============================================================================

def get_conservative_param_grid() -> Dict[str, List[Any]]:
    """
    保守参数网格（96个组合）

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


# ============================================================================
# 时间序列交叉验证（Walk-forward）
# ============================================================================

def walk_forward_split(df: pd.DataFrame,
                     train_size: int = 1200,
                     val_size: int = 250,
                     num_folds: int = 3) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Walk-forward滑动窗口划分数据

    数据结构：
    Fold 1: Train [0:1200], Val [1200:1450]
    Fold 2: Train [0:1450], Val [1450:1700]
    Fold 3: Train [0:1700], Val [1700:1950]

    Args:
        df: 输入数据
        train_size: 初始训练集大小（默认1200，约5年）
        val_size: 每个fold的验证集大小（默认250，约1年）
        num_folds: fold数量（默认3）

    Returns:
        List of (train_df, val_df) 元组
    """
    splits = []
    total_required = train_size + num_folds * val_size

    if len(df) < total_required:
        raise ValueError(f"数据不足：需要 {total_required} 条，现有 {len(df)} 条")

    for fold in range(num_folds):
        train_end = train_size + fold * val_size
        val_start = train_end
        val_end = val_start + val_size

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[val_start:val_end].copy()

        splits.append((train_df, val_df))
        logger.info(f"Fold {fold+1}/{num_folds}: train={len(train_df)}, val={len(val_df)}")

    return splits


def ts_cv_grid_search(
    X_train_list: List[np.ndarray],
    y_train_list: List[np.ndarray],
    X_val_list: List[np.ndarray],
    y_val_list: List[np.ndarray],
    param_grid: Dict[str, List],
    scaler: Optional[StandardScaler] = None
) -> Tuple[Dict, Dict]:
    """
    在多个fold上评估每个参数组合

    Args:
        X_train_list: 训练特征列表
        y_train_list: 训练标签列表
        X_val_list: 验证特征列表
        y_val_list: 验证标签列表
        param_grid: 参数网格字典
        scaler: 标准化器

    Returns:
        (best_params, all_results) 元组
        - best_params: 最佳参数
        - all_results: 所有组合的评估结果
    """
    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))

    logger.info(f"参数组合总数: {len(all_combinations)}")
    logger.info(f"Fold数量: {len(X_train_list)}")

    all_results = {}
    best_score = -np.inf
    best_params = None

    for combo in all_combinations:
        params = dict(zip(param_names, combo))
        param_str = "_".join([f"{k}={v}" for k, v in params.items()])

        fold_metrics = {
            'accuracy': [],
            'precision_macro': [],
            'recall_macro': [],
            'f1_macro': []
        }

        # 在所有fold上评估
        for fold_idx in range(len(X_train_list)):
            X_train, y_train = X_train_list[fold_idx], y_train_list[fold_idx]
            X_val, y_val = X_val_list[fold_idx], y_val_list[fold_idx]

            # 标准化
            if scaler is not None:
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
            else:
                X_train_scaled, X_val_scaled = X_train, X_val

            # 训练XGBoost
            model = xgb.XGBClassifier(
                **params,
                objective='multi:softprob',
                num_class=4,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)

            # 预测和评估
            y_pred = model.predict(X_val_scaled)

            fold_metrics['accuracy'].append(accuracy_score(y_val, y_pred))
            fold_metrics['precision_macro'].append(
                precision_score(y_val, y_pred, average='macro', zero_division=0)
            )
            fold_metrics['recall_macro'].append(
                recall_score(y_val, y_pred, average='macro', zero_division=0)
            )
            fold_metrics['f1_macro'].append(
                f1_score(y_val, y_pred, average='macro', zero_division=0)
            )

        # 计算平均指标
        mean_f1_macro = np.mean(fold_metrics['f1_macro'])
        std_f1_macro = np.std(fold_metrics['f1_macro'])

        all_results[param_str] = {
            'params': params,
            'fold_metrics': fold_metrics,
            'mean_accuracy': np.mean(fold_metrics['accuracy']),
            'mean_precision_macro': np.mean(fold_metrics['precision_macro']),
            'mean_recall_macro': np.mean(fold_metrics['recall_macro']),
            'mean_f1_macro': mean_f1_macro,
            'std_f1_macro': std_f1_macro
        }

        # 选择最佳参数（使用F1-macro）
        # 如果多个参数F1相同，选择max_depth更小的（更简单）
        if mean_f1_macro > best_score or (
            mean_f1_macro == best_score and
            params.get('max_depth', 999) < best_params.get('max_depth', 999)
        ):
            best_score = mean_f1_macro
            best_params = params

    logger.info(f"最佳F1-macro: {best_score:.4f}")
    logger.info(f"最佳参数: {best_params}")

    return best_params, all_results


# ============================================================================
# 模型训练（时间序列CV）
# ============================================================================

def train_single_classification_ts_cv(
    code: str,
    train_dfs: List[pd.DataFrame],
    val_dfs: List[pd.DataFrame],
    test_df: pd.DataFrame,
    feature_cols: List[str],
    param_grid: Dict[str, List],
    use_scaling: bool = True
) -> Dict:
    """
    使用walk-forward CV训练单个ETF分类模型

    流程：
        1. 准备fold数据
        2. 在所有fold上运行网格搜索
        3. 用所有train+val数据重新训练最终模型
        4. 在测试集上评估
        5. 保存模型和CV结果

    Args:
        code: ETF代码
        train_dfs: 训练集列表（每fold一个）
        val_dfs: 验证集列表（每fold一个）
        test_df: 测试集
        feature_cols: 特征列列表
        param_grid: 参数网格
        use_scaling: 是否使用标准化

    Returns:
        训练结果字典
    """
    # 1. 准备fold数据
    X_train_list = [df[feature_cols].values for df in train_dfs]
    y_train_list = [df['week_return_class'].values for df in train_dfs]
    X_val_list = [df[feature_cols].values for df in val_dfs]
    y_val_list = [df['week_return_class'].values for df in val_dfs]

    # 2. 初始化scaler
    scaler = StandardScaler() if use_scaling else None

    # 3. 网格搜索
    logger.info(f"ETF {code}: 开始时间序列交叉验证网格搜索...")
    best_params, cv_results = ts_cv_grid_search(
        X_train_list, y_train_list, X_val_list, y_val_list,
        param_grid, scaler
    )

    # 4. 准备最终训练数据（所有train+val）
    all_train_df = pd.concat(train_dfs + val_dfs, ignore_index=True)
    X_final_train = all_train_df[feature_cols].values
    y_final_train = all_train_df['week_return_class'].values

    # 移除NaN
    mask = ~np.isnan(X_final_train).any(axis=1) & ~np.isnan(y_final_train)
    X_final_train = X_final_train[mask]
    y_final_train = y_final_train[mask]

    logger.info(f"ETF {code}: 最终训练数据: {len(X_final_train)} 条")

    # 标准化最终数据
    if scaler is not None:
        scaler.fit(X_final_train)
        X_final_train_scaled = scaler.transform(X_final_train)
    else:
        X_final_train_scaled = X_final_train

    # 训练最终模型
    logger.info(f"ETF {code}: 使用最佳参数重新训练...")
    final_model = xgb.XGBClassifier(
        **best_params,
        objective='multi:softprob',
        num_class=4,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_final_train_scaled, y_final_train)

    # 5. 在测试集上评估
    X_test = test_df[feature_cols].values
    y_test = test_df['week_return_class'].values

    mask = ~np.isnan(X_test).any(axis=1) & ~np.isnan(y_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test

    y_test_pred = final_model.predict(X_test_scaled)
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision_macro': precision_score(y_test, y_test_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_test_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_test_pred, average='macro', zero_division=0)
    }

    logger.info(f"ETF {code} 测试集评估: Accuracy={test_metrics['accuracy']:.4f}, F1={test_metrics['f1_macro']:.4f}")

    # 6. 保存模型和结果
    model_version = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_ts_cv_model(
        code, final_model, scaler, model_version,
        best_params, cv_results, test_metrics, feature_cols
    )

    return {
        'code': code,
        'model_version': model_version,
        'model_type': 'classification_ts_cv',
        'best_params': best_params,
        'cv_results': cv_results,
        'test_metrics': test_metrics,
        'train_size': len(X_final_train),
        'test_size': len(X_test),
        'model': final_model,
        'scaler': scaler,
        'feature_cols': feature_cols
    }


def train_all_classification_ts_cv(
    codes: List[str],
    num_folds: int = 3,
    train_size: int = 1200,
    val_size: int = 250,
    use_scaling: bool = True,
    param_grid: Optional[Dict[str, List]] = None,
    min_date: Optional[str] = None
) -> List[Dict]:
    """
    批量训练所有ETF的分类模型（时间序列CV）

    Args:
        codes: ETF代码列表
        num_folds: fold数量（默认3）
        train_size: 初始训练集大小（默认1200，约5年）
        val_size: 每个fold的验证集大小（默认250，约1年）
        use_scaling: 是否使用标准化
        param_grid: 自定义参数网格
        min_date: 最小日期

    Returns:
        训练结果列表
    """
    from ..config import DB_MIN_DATE

    logger.info("="*80)
    logger.info("分类模型训练 - 时间序列交叉验证（Walk-forward CV）")
    logger.info("="*80)
    logger.info(f"ETF数量: {len(codes)}")
    logger.info(f"Folds: {num_folds}")
    logger.info(f"训练集大小: {train_size} (~{train_size/240:.1f} 年)")
    logger.info(f"验证集大小: {val_size} (~{val_size/240:.1f} 年)")
    logger.info("="*80)

    if param_grid is None:
        param_grid = get_conservative_param_grid()

    if min_date is None:
        min_date = DB_MIN_DATE

    all_results = []

    for i, code in enumerate(codes, 1):
        try:
            # 读取数据
            df = read_etf_data(code, min_date=min_date, use_advanced=True)
            logger.info(f"\n[{i}/{len(codes)}] 处理ETF: {code}")
            logger.info(f"  原始数据: {len(df)} 条")

            # 创建分类目标
            df = create_classification_target(df, days=5)
            df = clean_data(df, min_history_days=60)
            logger.info(f"  清洗后数据: {len(df)} 条")

            # 检查数据量
            total_required = train_size + num_folds * val_size + 50  # +50 for test
            if len(df) < total_required:
                logger.warning(f"  数据不足（需要{total_required}条），跳过")
                continue

            # Walk-forward划分
            splits = walk_forward_split(df, train_size, val_size, num_folds)
            train_dfs = [train_df for train_df, _ in splits]
            val_dfs = [val_df for _, val_df in splits]

            # 测试集
            test_start = train_size + num_folds * val_size
            test_df = df.iloc[test_start:].copy()
            logger.info(f"  测试集: {len(test_df)} 条")

            # 训练
            result = train_single_classification_ts_cv(
                code, train_dfs, val_dfs, test_df,
                ALL_FEATURE_COLS, param_grid, use_scaling
            )

            all_results.append(result)

        except Exception as e:
            logger.error(f"  ETF {code} 训练失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # 汇总统计
    if all_results:
        summarize_ts_cv_results(all_results)

    return all_results


# ============================================================================
# 模型保存
# ============================================================================

def save_ts_cv_model(
    code: str,
    model: xgb.XGBClassifier,
    scaler: Optional[StandardScaler],
    model_version: str,
    best_params: Dict[str, Any],
    cv_results: Dict,
    test_metrics: Dict[str, float],
    feature_cols: List[str]
) -> None:
    """
    保存时间序列CV训练的模型、标准化器和结果

    Args:
        code: ETF代码
        model: 模型对象
        scaler: 标准化器
        model_version: 模型版本号
        best_params: 最佳参数
        cv_results: CV结果
        test_metrics: 测试集指标
        feature_cols: 特征列列表
    """
    model_dir = os.path.join(MODELS_DIR, code)
    os.makedirs(model_dir, exist_ok=True)

    # 保存模型（带_ts_cv后缀）
    model_path = os.path.join(model_dir, f"{code}_{model_version}_classification_ts_cv.pkl")
    joblib.dump(model, model_path)
    logger.info(f"  模型已保存: {model_path}")

    # 保存标准化器
    if scaler is not None:
        scaler_path = os.path.join(model_dir, f"{code}_{model_version}_classification_ts_cv_scaler.pkl")
        joblib.dump(scaler, scaler_path)
        logger.info(f"  标准化器已保存: {scaler_path}")

    # 保存CV结果
    cv_dir = os.path.join(DATA_DIR, 'tuning_results')
    os.makedirs(cv_dir, exist_ok=True)
    cv_path = os.path.join(cv_dir, f"{code}_{model_version}_ts_cv_results.json")

    # 准备保存的数据（只保存必要的部分）
    cv_data = {
        'model_version': model_version,
        'best_params': best_params,
        'test_metrics': test_metrics,
        'num_params_evaluated': len(cv_results),
        'top_params': sorted(
            cv_results.items(),
            key=lambda x: x[1]['mean_f1_macro'],
            reverse=True
        )[:10]  # 保存top 10参数
    }

    with open(cv_path, 'w', encoding='utf-8') as f:
        json.dump(cv_data, f, indent=2, ensure_ascii=False)
    logger.info(f"  CV结果已保存: {cv_path}")

    # 保存元数据
    metadata = {
        'code': code,
        'model_version': model_version,
        'model_type': 'classification_ts_cv',
        'target': CLASSIFICATION_TARGET,
        'features': feature_cols,
        'best_params': best_params,
        'test_metrics': test_metrics,
        'train_date': datetime.now().isoformat(),
        'has_scaler': scaler is not None,
        'num_params_evaluated': len(cv_results)
    }
    metadata_path = os.path.join(model_dir, 'metadata_ts_cv.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"  元数据已保存: {metadata_path}")


# ============================================================================
# 评估和报告
# ============================================================================

def summarize_ts_cv_results(all_results: List[Dict]) -> None:
    """
    汇总时间序列CV的训练结果

    Args:
        all_results: 训练结果列表
    """
    accuracy_values = [r['test_metrics']['accuracy'] for r in all_results]
    f1_values = [r['test_metrics']['f1_macro'] for r in all_results]

    print(f"\n{'='*80}")
    print("时间序列CV训练汇总")
    print(f"{'='*80}")
    print(f"处理数量: {len(all_results)}")
    print(f"平均 Accuracy: {np.mean(accuracy_values):.4f}")
    print(f"平均 F1-Macro: {np.mean(f1_values):.4f}")
    print(f"Accuracy 范围: {np.min(accuracy_values):.4f} ~ {np.max(accuracy_values):.4f}")
    print(f"F1-Macro 范围: {np.min(f1_values):.4f} ~ {np.max(f1_values):.4f}")
    print(f"{'='*80}")


def create_summary_csv_ts_cv(
    results: List[Dict],
    output_path: str,
    etf_names: Dict[str, str]
) -> None:
    """
    创建时间序列CV的汇总CSV

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
            'accuracy': round(result['test_metrics']['accuracy'], 4),
            'precision_macro': round(result['test_metrics']['precision_macro'], 4),
            'recall_macro': round(result['test_metrics']['recall_macro'], 4),
            'f1_macro': round(result['test_metrics']['f1_macro'], 4),
            'train_size': result['train_size'],
            'test_size': result['test_size'],
            'best_n_estimators': result['best_params'].get('n_estimators'),
            'best_max_depth': result['best_params'].get('max_depth'),
            'best_learning_rate': result['best_params'].get('learning_rate'),
            'best_subsample': result['best_params'].get('subsample'),
            'best_colsample_bytree': result['best_params'].get('colsample_bytree'),
            'num_params_evaluated': len(result['cv_results']),
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
        description="ETF分类模型训练 - 时间序列交叉验证（Walk-forward CV）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 训练所有分类模型（时间序列CV）
  python -m my_etf.models.train_ts_cv

  # 自定义fold数和窗口大小
  python -m my_etf.models.train_ts_cv --num-folds 5 --train-size 1500 --val-size 200

  # 训练单个ETF
  python -m my_etf.models.train_ts_cv --code 510050

  # 不使用标准化
  python -m my_etf.models.train_ts_cv --no-scaling

  # 自定义参数网格（JSON格式）
  python -m my_etf.models.train_ts_cv --param-grid '{"n_estimators": [100, 200], "max_depth": [3, 4, 5]}'
        """
    )

    # CV参数
    parser.add_argument('--num-folds', type=int, default=3,
                        help='Fold数量（默认: 3）')
    parser.add_argument('--train-size', type=int, default=1200,
                        help='初始训练集大小（默认: 1200，约5年）')
    parser.add_argument('--val-size', type=int, default=250,
                        help='每个fold的验证集大小（默认: 250，约1年）')

    # ETF选择
    parser.add_argument('--code', type=str, default=None,
                        help='仅训练指定的ETF代码')

    # 特征选项
    parser.add_argument('--no-scaling', action='store_true',
                        help='不使用特征标准化')

    # 日期选项
    parser.add_argument('--min-date', type=str, default=None,
                        help='最小日期（默认: DB_MIN_DATE）')

    # 参数网格
    parser.add_argument('--param-grid', type=str, default=None,
                        help='自定义参数网格（JSON格式，例如: \'{"n_estimators": [100, 200], "max_depth": [3, 4, 5]}\'）')

    args = parser.parse_args()

    # 配置
    from ..config import get_etf_codes

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
    print("ETF分类模型训练 - 时间序列交叉验证（Walk-forward CV）")
    print("="*80)
    print(f"ETF数量: {len(codes)}")
    print(f"特征数量: {len(ALL_FEATURE_COLS)}")
    print(f"使用标准化: {not args.no_scaling}")
    print(f"Folds: {args.num_folds}")
    print(f"训练集大小: {args.train_size} (~{args.train_size/240:.1f} 年)")
    print(f"验证集大小: {args.val_size} (~{args.val_size/240:.1f} 年)")
    print(f"参数网格: {'自定义' if param_grid else '默认（96个组合）'}")
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
    results = train_all_classification_ts_cv(
        codes,
        num_folds=args.num_folds,
        train_size=args.train_size,
        val_size=args.val_size,
        use_scaling=not args.no_scaling,
        param_grid=param_grid,
        min_date=args.min_date
    )

    # 保存汇总
    if results:
        summary_path = os.path.join(SUMMARY_DIR, 'training_summary_ts_cv_classification.csv')
        create_summary_csv_ts_cv(results, summary_path, etf_names)
        print(f"\n汇总CSV: {summary_path}")
    else:
        print("\n没有成功训练的模型")

    print(f"\n{'='*80}")
    print("完成！")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
