"""
生成回测预测数据（使用验证集模型）
使用带验证集的超参数调优后的模型生成完整的预测历史数据，用于回测
"""
import argparse
import joblib
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import (MODELS_DIR, DATA_DIR, ALL_FEATURE_COLS,
                     DB_MIN_DATE, get_etf_codes)
from ..utils.database import read_etf_data, get_etf_price
from ..utils.logger import setup_logger

logger = setup_logger("backtest_predictions_validation", "backtest_predictions_validation.log")


def create_classification_target(df: pd.DataFrame, days: int = 5) -> pd.DataFrame:
    """
    创建分类目标（4类收益）

    类别：
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
    close_col = 'close' if 'close' in df.columns else '收盘'

    # 计算绝对收益率
    abs_return = (df[close_col].shift(-days) - df[close_col]) / df[close_col] * 100
    df['week_return'] = abs_return

    # 分箱
    bins = [float('-inf'), -5, 0, 5, float('inf')]
    labels = [0, 1, 2, 3]  # XGBoost expects classes starting from 0

    # 首先移除NaN值（无法分类）
    df = df.dropna(subset=['week_return'])

    # 分箱
    df['week_return_class'] = pd.cut(
        df['week_return'],
        bins=bins,
        labels=labels
    )

    # 转换为整数类型
    df['week_return_class'] = df['week_return_class'].astype(int)

    # 移除最后days天的数据（无法计算目标）
    df = df.iloc[:-days]

    return df


def generate_classification_predictions_with_validation(
    codes: List[str],
    feature_cols: List[str],
    min_date: str = '2015-01-01',
    train_ratio: float = 0.7,
    val_ratio: float = 0.1
) -> Dict[str, pd.DataFrame]:
    """
    生成分类模型的预测数据（使用验证集模型）

    Args:
        codes: ETF代码列表
        feature_cols: 特征列
        min_date: 起始日期
        train_ratio: 训练+验证集占总数据的比例
        val_ratio: 验证集占train+val的比例

    Returns:
        {code: DataFrame with predictions}
    """
    logger.info("="*60)
    logger.info("生成分类模型预测（使用验证集模型）")
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

        # 创建分类目标
        df = create_classification_target(df, days=5)
        logger.info(f"  计算分类目标后: {len(df)} 条")

        # 检查是否有所有必需的特征列
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"  缺失特征列: {len(missing_cols)}个，跳过该ETF")
            continue

        # 移除早期数据（确保有足够历史计算指标）
        df = df.iloc[60:]

        # 移除包含NaN的行
        df = df.dropna(subset=feature_cols + ['week_return', 'week_return_class'])

        if len(df) < 50:
            logger.warning(f"  清洗后数据不足，跳过")
            continue

        # 划分训练+验证集和测试集（与训练时相同的方式）
        total_len = len(df)
        split_idx = int(total_len * train_ratio)
        train_val_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        # 从train+val中划分val
        train_val_len = len(train_val_df)
        val_split_idx = int(train_val_len * (val_ratio / train_ratio))

        train_df = train_val_df.iloc[:val_split_idx].copy()
        val_df = train_val_df.iloc[val_split_idx:].copy()

        logger.info(f"  训练集: {len(train_df)} 条，验证集: {len(val_df)} 条，测试集: {len(test_df)} 条")

        # 加载分类模型和标准化器（优先使用验证集模型）
        model_dir = os.path.join(MODELS_DIR, code)
        if not os.path.exists(model_dir):
            logger.warning(f"  未找到模型目录，跳过")
            continue

        # 优先查找验证集模型
        model_files = sorted([
            f for f in os.listdir(model_dir)
            if f.endswith('_validation.pkl') and '_scaler.pkl' not in f
        ])

        # 如果没有验证集模型，回退到普通分类模型
        if not model_files:
            logger.warning(f"  未找到验证集模型，尝试使用普通分类模型")
            model_files = sorted([
                f for f in os.listdir(model_dir)
                if f.endswith('_classification.pkl') and '_validation.pkl' not in f and '_scaler.pkl' not in f
            ])

        if not model_files:
            logger.warning(f"  未找到任何分类模型文件，跳过")
            continue

        latest_model_file = model_files[-1]
        # 提取model_version（去掉validation后缀）
        if '_validation.pkl' in latest_model_file:
            model_version = latest_model_file.replace('_validation.pkl', '')
            scaler_file = f"{model_version}_validation_scaler.pkl"
            is_validation_model = True
        else:
            model_version = latest_model_file.replace('_classification.pkl', '')
            scaler_file = f"{model_version}_classification_scaler.pkl"
            is_validation_model = False

        model_path = os.path.join(model_dir, latest_model_file)
        scaler_path = os.path.join(model_dir, scaler_file)

        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            model_type = "validation集模型" if is_validation_model else "普通模型"
            logger.info(f"  使用{model_type}: {latest_model_file}")

            # 对测试集进行预测
            X_test = test_df[feature_cols].values
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

            # 使用validation后缀区分预测文件
            output_file = os.path.join(output_dir, f"{code}_{model_version}_validation_classification.csv")
            predictions_df.to_csv(output_file, index=False)
            logger.info(f"  保存: {output_file}")

        except Exception as e:
            logger.error(f"  预测失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    return predictions_dict


def main():
    parser = argparse.ArgumentParser(
        description="生成回测预测数据（使用验证集模型）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 生成所有分类模型的预测（使用验证集模型）
  python -m my_etf.models.generate_backtest_predictions_validation

  # 指定训练集比例（默认0.7）
  python -m my_etf.models.generate_backtest_predictions_validation --train-ratio 0.6

  # 指定验证集比例（默认0.1）
  python -m my_etf.models.generate_backtest_predictions_validation --val-ratio 0.15
        """
    )

    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='训练+验证集占总数据的比例（默认: 0.7）')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='验证集占train+val的比例（默认: 0.1）')
    parser.add_argument('--min-date', type=str, default=None,
                        help='最小日期（默认: DB_MIN_DATE）')

    args = parser.parse_args()

    # 分类模型固定使用全部特征
    feature_cols = ALL_FEATURE_COLS

    # 确定数据起始日期
    min_date = args.min_date if args.min_date else DB_MIN_DATE

    codes = get_etf_codes()

    # 生成预测
    predictions_dict = generate_classification_predictions_with_validation(
        codes,
        feature_cols=feature_cols,
        min_date=min_date,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )

    logger.info("\n" + "="*60)
    logger.info(f"完成！共为 {len(predictions_dict)} 个ETF生成预测数据")
    logger.info("="*60)


if __name__ == '__main__':
    main()
