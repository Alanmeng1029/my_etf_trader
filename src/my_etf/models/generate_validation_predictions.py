"""
生成验证集模型的回测预测数据
使用带验证集调优的分类模型生成预测数据
"""
import argparse
import joblib
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import (MODELS_DIR, DATA_DIR, ALL_FEATURE_COLS,
                     get_etf_codes)
from ..utils.database import read_etf_data
from ..utils.logger import setup_logger

logger = setup_logger("validation_predictions", "validation_predictions.log")


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


def generate_validation_predictions(codes: List[str],
                                   feature_cols: List[str],
                                   min_date: str = '2015-01-01',
                                   max_date: str = '20250101') -> Dict[str, pd.DataFrame]:
    """
    生成验证集模型的预测数据

    Args:
        codes: ETF代码列表
        feature_cols: 特征列
        min_date: 起始日期
        max_date: 训练集截止日期（用于确定测试集起始日期）

    Returns:
        {code: DataFrame with predictions}
    """
    logger.info("="*60)
    logger.info("生成验证集模型预测（分类）")
    logger.info(f"特征数: {len(feature_cols)}")
    logger.info(f"训练集截止日期: {max_date}")
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

        # 使用日期划分：max_date之后为测试集
        df['date_str'] = df['date'].astype(str).str.replace('-', '')
        max_date_str = max_date.replace('-', '')
        test_df = df[df['date_str'] > max_date_str].copy()
        train_val_df = df[df['date_str'] <= max_date_str].copy()

        logger.info(f"  训练+验证集: {len(train_val_df)} 条，测试集: {len(test_df)} 条")

        if len(test_df) == 0:
            logger.warning(f"  测试集为空，跳过")
            continue

        # 加载验证模型和标准化器
        model_dir = os.path.join(MODELS_DIR, code)
        if not os.path.exists(model_dir):
            logger.warning(f"  未找到模型目录，跳过")
            continue

        # 查找validation模型
        model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('_validation.pkl')])
        if not model_files:
            logger.warning(f"  未找到验证模型文件，跳过")
            continue

        latest_model_file = model_files[-1]
        model_version = latest_model_file.replace('_validation.pkl', '')
        scaler_file = f"{model_version}_validation_scaler.pkl"

        model_path = os.path.join(model_dir, latest_model_file)
        scaler_path = os.path.join(model_dir, scaler_file)

        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            # 对测试集进行预测
            X_test = test_df[feature_cols].values
            y_test = test_df['week_return'].values

            # 标准化
            X_test_scaled = scaler.transform(X_test)

            # 预测类别
            y_pred = model.predict(X_test_scaled)

            # 获取预测概率
            y_proba = model.predict_proba(X_test_scaled)
            # 类别索引 0-3 对应：0=大幅下跌, 1=小幅下跌, 2=小幅上涨, 3=大幅上涨
            class_4_prob = y_proba[:, 3] if y_proba.shape[1] > 3 else y_proba[:, -1]

            # 创建预测DataFrame
            predictions_df = pd.DataFrame({
                'date': test_df['date'],
                'code': code,
                'actual_return': y_test,
                'predicted_class': y_pred,  # 类别0-3
                'class_4_prob': class_4_prob  # 大幅上涨概率
            })

            predictions_dict[code] = predictions_df
            logger.info(f"  生成预测: {len(predictions_df)} 条")

            # 保存到文件
            output_dir = os.path.join(DATA_DIR, code)
            os.makedirs(output_dir, exist_ok=True)

            output_file = os.path.join(output_dir, f"{code}_{model_version}_validation.csv")
            predictions_df.to_csv(output_file, index=False)
            logger.info(f"  保存: {output_file}")

        except Exception as e:
            logger.error(f"  预测失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    return predictions_dict


def main():
    parser = argparse.ArgumentParser(
        description="生成验证集模型回测预测数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 生成所有验证集模型的预测
  python -m my_etf.models.generate_validation_predictions

  # 指定训练集截止日期
  python -m my_etf.models.generate_validation_predictions --max-date 20250101
        """
    )

    parser.add_argument('--max-date', type=str, default='20250101',
                        help='训练集截止日期（格式: YYYYMMDD，默认: 20250101）')

    args = parser.parse_args()

    # 使用全部特征
    feature_cols = ALL_FEATURE_COLS

    codes = get_etf_codes()

    # 生成预测
    predictions_dict = generate_validation_predictions(
        codes, feature_cols=feature_cols, max_date=args.max_date
    )

    logger.info("\n" + "="*60)
    logger.info(f"完成！共生成 {len(predictions_dict)} 个ETF的预测数据")
    logger.info("="*60)


if __name__ == '__main__':
    main()
