"""
生成回测预测数据
使用训练好的模型生成完整的预测历史数据，用于回测
"""
import argparse
import joblib
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..config import (MODELS_DIR, DATA_DIR, FEATURE_COLS, ALL_FEATURE_COLS,
                     BENCHMARK_ETF, DB_MIN_DATE, get_etf_codes)
from ..utils.database import read_etf_data, get_etf_price
from ..utils.logger import setup_logger

logger = setup_logger("backtest_predictions", "backtest_predictions.log")


def find_regression_model_files(model_dir: str) -> List[str]:
    """查找普通回归模型，排除分类、验证和调参模型文件。"""
    non_regression_markers = (
        '_classification',
        '_validation',
        '_mixed',
        '_ts_cv',
    )
    return sorted([
        f for f in os.listdir(model_dir)
        if f.endswith('.pkl')
        and '_scaler.pkl' not in f
        and not any(marker in f for marker in non_regression_markers)
    ])


def create_target(df: pd.DataFrame, days: int = 5) -> pd.DataFrame:
    """创建预测目标（绝对收益）"""
    df = df.copy()
    close_col = 'close' if 'close' in df.columns else '收盘'

    # 计算绝对收益率
    abs_return = (df[close_col].shift(-days) - df[close_col]) / df[close_col] * 100
    df['week_return'] = abs_return

    # 移除最后days天的数据（无法计算目标）
    df = df.iloc[:-days]

    return df


def generate_separate_predictions(codes: List[str],
                                   feature_cols: List[str],
                                   min_date: str = '2015-01-01',
                                   train_ratio: float = 0.7) -> Dict[str, pd.DataFrame]:
    """
    生成单独模型的预测数据

    Args:
        codes: ETF代码列表
        feature_cols: 特征列
        min_date: 起始日期
        train_ratio: 训练集比例

    Returns:
        {code: DataFrame with predictions}
    """
    logger.info("="*60)
    logger.info("生成单独模型预测（绝对收益）")
    logger.info(f"特征数: {len(feature_cols)}")
    logger.info("="*60)

    predictions_dict = {}

    for code in codes:
        logger.info(f"\n处理ETF: {code}")

        # 读取数据
        df = read_etf_data(code, min_date=min_date, use_advanced=True)
        logger.info(f"  原始数据: {len(df)} 条")

        if len(df) < 100:
            logger.warning(f"  数据不足，跳过")
            continue

        # 创建目标
        df = create_target(df, days=5)
        logger.info(f"  计算目标后: {len(df)} 条")

        # 检查是否有所有必需的特征列
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"  缺失特征列: {len(missing_cols)}个，跳过该ETF")
            continue

        # 移除早期数据（确保有足够历史计算指标）
        df = df.iloc[60:]

        # 移除包含NaN的行
        df = df.dropna(subset=feature_cols + ['week_return'])

        if len(df) < 50:
            logger.warning(f"  清洗后数据不足，跳过")
            continue

        # 划分训练/测试集
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        logger.info(f"  训练集: {len(train_df)} 条，测试集: {len(test_df)} 条")

        # 加载模型和标准化器
        model_dir = os.path.join(MODELS_DIR, code)
        if not os.path.exists(model_dir):
            logger.warning(f"  未找到模型目录，跳过")
            continue

        model_files = find_regression_model_files(model_dir)
        if not model_files:
            logger.warning(f"  未找到模型文件，跳过")
            continue

        latest_model_file = model_files[-1]
        model_version = latest_model_file.replace('.pkl', '')
        scaler_file = f"{model_version}_scaler.pkl"

        model_path = os.path.join(model_dir, latest_model_file)
        scaler_path = os.path.join(model_dir, scaler_file)

        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            # 对测试集进行预测
            X_test = test_df[feature_cols]
            y_test = test_df['week_return'].values

            # 标准化
            X_test_scaled = scaler.transform(X_test)

            # 预测
            y_pred = model.predict(X_test_scaled)

            # 创建预测DataFrame
            predictions_df = pd.DataFrame({
                'date': test_df['date'],
                'code': code,
                'actual_return': y_test,
                'predicted_return': y_pred
            })

            predictions_dict[code] = predictions_df
            logger.info(f"  生成预测: {len(predictions_df)} 条")

            # 保存到文件
            output_dir = os.path.join(DATA_DIR, code)
            os.makedirs(output_dir, exist_ok=True)

            output_file = os.path.join(output_dir, f"{code}_{model_version}_separate.csv")
            predictions_df.to_csv(output_file, index=False)
            logger.info(f"  保存: {output_file}")

        except Exception as e:
            logger.error(f"  预测失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    return predictions_dict


def get_benchmark_returns(min_date: str = '2016-01-01') -> pd.Series:
    """获取基准ETF的未来收益率序列"""
    try:
        df = read_etf_data(BENCHMARK_ETF, min_date=min_date, use_advanced=False)
        close_col = 'close' if 'close' in df.columns else '收盘'
        benchmark_returns = (df[close_col].shift(-5) - df[close_col]) / df[close_col] * 100

        # 设置日期索引
        if 'date' in df.columns:
            benchmark_returns.index = pd.to_datetime(df['date'])

        return benchmark_returns
    except Exception as e:
        logger.error(f"获取基准 {BENCHMARK_ETF} 收益率失败: {e}")
        return pd.Series(dtype=np.float64)


def create_excess_return_target(df: pd.DataFrame, benchmark_returns: pd.Series) -> pd.DataFrame:
    """创建超额收益目标"""
    df = df.copy()
    close_col = 'close' if 'close' in df.columns else '收盘'
    date_col = 'date' if 'date' in df.columns else '日期'

    # 计算绝对收益率
    abs_return = (df[close_col].shift(-5) - df[close_col]) / df[close_col] * 100

    # 使用pd.merge按日期对齐
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
    df['excess_return'] = merged['etf_return'] - merged['benchmark_return'].values

    # 移除最后5天的数据（无法计算目标）
    df = df.iloc[:-5]

    return df


def generate_unified_predictions(codes: List[str],
                                  feature_cols: List[str],
                                  common_start_date: str = '2016-01-01',
                                  train_ratio: float = 0.7) -> Dict[str, pd.DataFrame]:
    """
    生成统一模型的预测数据

    Args:
        codes: ETF代码列表
        feature_cols: 特征列
        common_start_date: 统一起始日期
        train_ratio: 训练集比例

    Returns:
        {code: DataFrame with predictions}
    """
    logger.info("="*60)
    logger.info("生成统一模型预测（超额收益）")
    logger.info(f"特征数: {len(feature_cols)}")
    logger.info(f"统一起始日期: {common_start_date}")
    logger.info("="*60)

    # 获取基准收益率
    benchmark_returns = get_benchmark_returns(min_date=common_start_date)
    if benchmark_returns.empty:
        logger.error("无法获取基准收益率")
        return {}

    # 合并所有ETF数据
    all_dfs = []
    for code in codes:
        df = read_etf_data(code, min_date=common_start_date, use_advanced=True)
        if len(df) < 100:
            logger.warning(f"  ETF {code} 数据不足，跳过")
            continue

        # 计算超额收益目标
        df = create_excess_return_target(df, benchmark_returns)

        # 移除早期数据
        df = df.iloc[60:]

        # 添加ETF编码（作为特征）
        df['etf_code'] = code

        all_dfs.append(df)

    if not all_dfs:
        logger.error("没有足够的数据进行训练")
        return {}

    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"合并后数据: {len(combined_df)} 条")

    # 对ETF编码进行Label Encoding
    le = LabelEncoder()
    le.fit(combined_df['etf_code'])
    combined_df['etf_code_encoded'] = le.transform(combined_df['etf_code'])

    # 移除包含NaN的行
    combined_df = combined_df.dropna(subset=feature_cols + ['etf_code_encoded', 'excess_return'])

    # 划分训练/测试集
    split_idx = int(len(combined_df) * train_ratio)
    train_df = combined_df.iloc[:split_idx]
    test_df = combined_df.iloc[split_idx:]

    logger.info(f"训练集: {len(train_df)} 条，测试集: {len(test_df)} 条")

    # 加载统一模型
    model_dir = os.path.join(MODELS_DIR, 'unified')
    if not os.path.exists(model_dir):
        logger.error("未找到统一模型目录")
        return {}

    model_files = find_regression_model_files(model_dir)
    if not model_files:
        logger.error("未找到统一模型文件")
        return {}

    latest_model_file = model_files[-1]
    model_version = latest_model_file.replace('.pkl', '')
    scaler_file = f"{model_version}_scaler.pkl"

    model_path = os.path.join(model_dir, latest_model_file)
    scaler_path = os.path.join(model_dir, scaler_file)

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # 特征列（包含ETF编码）
        extended_features = feature_cols + ['etf_code_encoded']

        # 对测试集进行预测
        X_test = test_df[extended_features]
        y_test = test_df['excess_return'].values

        # 标准化
        X_test_scaled = scaler.transform(X_test)

        # 预测
        y_pred = model.predict(X_test_scaled)

        # 按ETF分组创建预测DataFrame
        predictions_dict = {}
        for code in codes:
            code_mask = test_df['etf_code'] == code
            if code_mask.sum() > 0:
                code_test = test_df[code_mask]
                code_pred = y_pred[code_mask]
                code_actual = y_test[code_mask]

                predictions_df = pd.DataFrame({
                    'date': code_test['date'],
                    'code': code,
                    'actual_return': code_actual,
                    'predicted_return': code_pred
                })

                predictions_dict[code] = predictions_df
                logger.info(f"  {code}: {len(predictions_df)} 条预测")

                # 保存到文件
                output_dir = os.path.join(DATA_DIR, code)
                os.makedirs(output_dir, exist_ok=True)

                output_file = os.path.join(output_dir, f"{code}_{model_version}_unified.csv")
                predictions_df.to_csv(output_file, index=False)
                logger.info(f"  保存: {output_file}")

    except Exception as e:
        logger.error(f"统一模型预测失败: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return predictions_dict


def generate_classification_predictions(codes: List[str],
                                      feature_cols: List[str],
                                      min_date: str = '2015-01-01',
                                      train_ratio: float = 0.7) -> Dict[str, pd.DataFrame]:
    """
    生成分类模型的预测数据

    Args:
        codes: ETF代码列表
        feature_cols: 特征列
        min_date: 起始日期
        train_ratio: 训练集比例

    Returns:
        {code: DataFrame with predictions}
    """
    logger.info("="*60)
    logger.info("生成分类模型预测（4类收益）")
    logger.info(f"特征数: {len(feature_cols)}")
    logger.info("="*60)

    predictions_dict = {}

    for code in codes:
        logger.info(f"\n处理ETF: {code}")

        # 读取数据
        df = read_etf_data(code, min_date=min_date, use_advanced=True)
        logger.info(f"  原始数据: {len(df)} 条")

        if len(df) < 100:
            logger.warning(f"  数据不足，跳过")
            continue

        # 创建目标（使用同样的week_return）
        df = create_target(df, days=5)
        logger.info(f"  计算目标后: {len(df)} 条")

        # 检查是否有所有必需的特征列
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"  缺失特征列: {len(missing_cols)}个，跳过该ETF")
            logger.warning(f"  缺失列: {missing_cols[:5]}...")
            continue

        # 移除早期数据（确保有足够历史计算指标）
        df = df.iloc[60:]

        # 移除包含NaN的行
        df = df.dropna(subset=feature_cols + ['week_return'])

        if len(df) < 50:
            logger.warning(f"  清洗后数据不足，跳过")
            continue

        # 划分训练/测试集
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        logger.info(f"  训练集: {len(train_df)} 条，测试集: {len(test_df)} 条")

        # 加载分类模型和标准化器
        model_dir = os.path.join(MODELS_DIR, code)
        if not os.path.exists(model_dir):
            logger.warning(f"  未找到模型目录，跳过")
            continue

        model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('_classification.pkl')])
        if not model_files:
            logger.warning(f"  未找到分类模型文件，跳过")
            continue

        latest_model_file = model_files[-1]
        model_version = latest_model_file.replace('_classification.pkl', '')
        scaler_file = f"{model_version}_classification_scaler.pkl"

        model_path = os.path.join(model_dir, latest_model_file)
        scaler_path = os.path.join(model_dir, scaler_file)

        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            # 对测试集进行预测
            X_test = test_df[feature_cols]
            y_test = test_df['week_return'].values

            # 标准化
            X_test_scaled = scaler.transform(X_test)

            # 预测类别
            y_pred = model.predict(X_test_scaled)

            # 获取预测概率（特别是类别3的概率，即大幅上涨）
            y_proba = model.predict_proba(X_test_scaled)
            # 类别索引 0-3 对应：0=大幅下跌, 1=小幅下跌, 2=小幅上涨, 3=大幅上涨
            class_4_prob = y_proba[:, 3] if y_proba.shape[1] > 3 else y_proba[:, -1]

            # 创建预测DataFrame
            predictions_df = pd.DataFrame({
                'date': test_df['date'],
                'code': code,
                'actual_return': y_test,
                'predicted_class': y_pred,  # 类别0-3
                'class_4_prob': class_4_prob
            })

            predictions_dict[code] = predictions_df
            logger.info(f"  生成预测: {len(predictions_df)} 条")

            # 保存到文件
            output_dir = os.path.join(DATA_DIR, code)
            os.makedirs(output_dir, exist_ok=True)

            output_file = os.path.join(output_dir, f"{code}_{model_version}_classification.csv")
            predictions_df.to_csv(output_file, index=False)
            logger.info(f"  保存: {output_file}")

        except Exception as e:
            logger.error(f"  预测失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    return predictions_dict


def main():
    parser = argparse.ArgumentParser(
        description="生成回测预测数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 生成单独模型预测（全部特征）
  python -m my_etf.models.generate_backtest_predictions --separate

  # 生成统一模型预测（基础特征）
  python -m my_etf.models.generate_backtest_predictions --unified --no-advanced

  # 生成两种模型的预测
  python -m my_etf.models.generate_backtest_predictions --separate --unified

  # 生成分类模型预测（全部特征）
  python -m my_etf.models.generate_backtest_predictions --classification
        """
    )

    # 模型类型选择
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--separate', action='store_true', help='单独模型（回归）')
    group.add_argument('--unified', action='store_true', help='统一模型（回归）')
    group.add_argument('--hybrid', action='store_true', help='混合（两种都生成）')
    group.add_argument('--classification', action='store_true', help='分类模型')

    # 特征选项（分类模型固定使用全部特征）
    parser.add_argument('--all-features', action='store_true',
                        help='使用全部特征（48个）')
    parser.add_argument('--no-advanced', action='store_true',
                        help='仅使用基础特征（21个）')

    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='训练集比例（默认: 0.7）')

    args = parser.parse_args()

    # 确定特征列
    if args.classification:
        # 分类模型固定使用全部特征
        feature_cols = ALL_FEATURE_COLS
    elif args.all_features:
        feature_cols = ALL_FEATURE_COLS
    elif args.no_advanced:
        feature_cols = FEATURE_COLS
    else:
        # 默认：separate使用全部特征，unified使用基础特征
        if args.separate or args.hybrid:
            feature_cols = ALL_FEATURE_COLS
        else:  # unified
            feature_cols = FEATURE_COLS

    codes = get_etf_codes()

    # 生成预测
    if args.classification:
        # 分类模式
        generate_classification_predictions(codes, feature_cols=feature_cols,
                                       train_ratio=args.train_ratio)
    elif args.hybrid:
        # 混合模式：生成两种预测
        generate_separate_predictions(codes, feature_cols=ALL_FEATURE_COLS,
                                       train_ratio=args.train_ratio)
        generate_unified_predictions(codes, feature_cols=FEATURE_COLS,
                                     train_ratio=args.train_ratio)
    elif args.separate:
        generate_separate_predictions(codes, feature_cols=feature_cols,
                                       train_ratio=args.train_ratio)
    else:  # unified
        generate_unified_predictions(codes, feature_cols=feature_cols,
                                     train_ratio=args.train_ratio)

    logger.info("\n" + "="*60)
    logger.info("完成！")
    logger.info("="*60)


if __name__ == '__main__':
    main()
