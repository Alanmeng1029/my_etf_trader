"""
ETF模型预测模块
使用训练好的XGBoost模型生成预测
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, List

from ..config import (
    MODELS_DIR, FEATURE_COLS, ALL_FEATURE_COLS, DB_MIN_DATE,
    CLASSIFICATION_CONFIG
)
from ..utils.database import read_etf_data
from ..utils.logger import setup_logger

logger = setup_logger("etf_predict", "predict.log")


def load_latest_model_and_scaler(code: str, model_type: str = 'regression'):
    """
    加载指定ETF的最新模型和标准化器

    Args:
        code: ETF代码
        model_type: 模型类型 ('regression' 或 'classification')

    Returns:
        tuple: (model, scaler, model_version) or (None, None, None) if not found
    """
    import os
    model_dir = os.path.join(MODELS_DIR, code)

    if not os.path.exists(model_dir):
        return None, None, None

    # 查找指定类型的最新模型文件
    suffix = '_classification' if model_type == 'classification' else ''
    model_files = sorted([
        f for f in os.listdir(model_dir)
        if f.endswith('.pkl')
        and '_scaler.pkl' not in f
        and f.endswith(f'{suffix}.pkl')
    ])

    if not model_files:
        return None, None, None

    latest_model_file = model_files[-1]
    model_version = latest_model_file.replace('.pkl', '').replace(suffix, '')
    scaler_file = f"{model_version}{suffix}_scaler.pkl"

    model_path = os.path.join(model_dir, latest_model_file)
    scaler_path = os.path.join(model_dir, scaler_file)

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler, model_version
    except Exception as e:
        logger.error(f"  警告: 加载模型失败 {code}: {e}")
        return None, None, None


def get_class_label_and_name(class_id: int):
    """
    获取类别名称

    Args:
        class_id: 类别ID (1, 2, 3, 4)

    Returns:
        tuple: (class_id, class_name)
    """
    class_name = CLASSIFICATION_CONFIG['label_names'].get(class_id, f"未知类别({class_id})")
    return (class_id, class_name)


def predict_classification(code: str, df: pd.DataFrame) -> Dict:
    """
    使用分类模型预测ETF类别

    Args:
        code: ETF代码
        df: 包含特征数据的DataFrame

    Returns:
        预测结果字典，包含类别、名称、概率、日期、版本
    """
    # 加载分类模型
    model, scaler, model_version = load_latest_model_and_scaler(code, model_type='classification')
    if model is None:
        return None

    # 使用最新的一条数据生成预测
    latest_row = df.iloc[-1][ALL_FEATURE_COLS]

    # 检查是否有NaN值
    if latest_row.isna().any():
        return None

    # 标准化并预测
    X_latest = latest_row.values.reshape(1, -1)
    X_scaled = scaler.transform(X_latest)

    # 预测类别
    class_id = int(model.predict(X_scaled)[0])

    # 获取概率分布
    proba = model.predict_proba(X_scaled)[0]
    confidence = float(proba[class_id - 1])  # class_id 是 1-4，数组索引是 0-3

    # 获取类别名称
    _, class_name = get_class_label_and_name(class_id)

    return {
        'code': code,
        'class_id': class_id,
        'class_name': class_name,
        'confidence': confidence,
        'proba': proba.tolist(),
        'date': df.iloc[-1]['date'],
        'model_version': model_version,
        'model_type': 'classification'
    }


def generate_daily_predictions(model_type: str = 'regression'):
    """
    使用现有模型生成所有ETF的最新预测

    Args:
        model_type: 模型类型 ('regression' 或 'classification')

    Returns:
        dict: 预测结果字典
            - 回归模型: {code: {'predicted_return': float, 'date': str, 'model_version': str}}
            - 分类模型: {code: {'class_id': int, 'class_name': str, 'confidence': float, 'date': str, 'model_version': str}}
    """
    from ..config import get_etf_codes
    codes = get_etf_codes()
    predictions = {}

    logger.info(f"使用现有{model_type}模型生成预测...")

    for code in codes:
        logger.info(f"\n处理 {code}...")

        # 加载数据（包含高级指标）
        df = read_etf_data(code, min_date=DB_MIN_DATE, use_advanced=True)
        if df.empty or len(df) < 60:
            logger.info(f"  跳过: 数据不足")
            continue

        if model_type == 'classification':
            # 使用分类模型预测
            result = predict_classification(code, df)
            if result is None:
                logger.info(f"  跳过: 未找到模型或数据有问题")
                continue

            predictions[code] = result
            logger.info(f"  预测类别: {result['class_name']} (置信度: {result['confidence']:.4f})")
        else:
            # 使用回归模型预测
            model, scaler, model_version = load_latest_model_and_scaler(code, model_type='regression')
            if model is None:
                logger.info(f"  跳过: 未找到模型")
                continue

            # 使用最新的一条数据生成预测
            latest_row = df.iloc[-1][ALL_FEATURE_COLS]

            # 检查是否有NaN值
            if latest_row.isna().any():
                logger.info(f"  跳过: 最新数据包含NaN值")
                continue

            # 标准化并预测
            X_latest = latest_row.values.reshape(1, -1)
            X_scaled = scaler.transform(X_latest)
            predicted_return = model.predict(X_scaled)[0]

            predictions[code] = {
                'predicted_return': predicted_return,
                'date': df.iloc[-1]['date'],
                'model_version': model_version,
                'model_type': 'regression'
            }

            logger.info(f"  预测收益率: {predicted_return:.2f}%")

    return predictions


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(
        description="ETF模型预测 - 支持回归和分类模型"
    )
    parser.add_argument('--model-type', type=str, default='regression',
                        choices=['regression', 'classification'],
                        help='模型类型（默认: regression）')

    args = parser.parse_args()

    predictions = generate_daily_predictions(model_type=args.model_type)

    logger.info("\n" + "=" * 80)
    logger.info("预测完成")
    logger.info("=" * 80)

    if args.model_type == 'classification':
        for code, pred in predictions.items():
            logger.info(f"{code}: {pred['class_name']} (置信度: {pred['confidence']:.4f}, 日期: {pred['date']})")
    else:
        for code, pred in predictions.items():
            logger.info(f"{code}: {pred['predicted_return']:.2f}% (日期: {pred['date']})")


if __name__ == '__main__':
    main()
