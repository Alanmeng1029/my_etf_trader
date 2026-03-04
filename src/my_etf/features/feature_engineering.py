# -*- coding: utf-8 -*-
"""
特征工程模块
提供特征重要性分析、递归特征消除、相关性分析等功能
"""
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler

from ..utils.logger import setup_logger

logger = setup_logger("feature_engineering", "feature_engineering.log")


# ============================================================================
# 特征重要性分析
# ============================================================================

def analyze_feature_importance(model, feature_names: List[str],
                              top_n: int = 20) -> pd.DataFrame:
    """
    分析模型特征重要性

    Args:
        model: 训练好的模型（需要有feature_importances_属性）
        feature_names: 特征名称列表
        top_n: 显示前N个特征

    Returns:
        按重要性排序的DataFrame
    """
    if not hasattr(model, 'feature_importances_'):
        logger.error("模型不支持特征重要性分析（缺少feature_importances_属性）")
        return pd.DataFrame()

    importances = model.feature_importances_

    # 创建DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })

    # 按重要性排序
    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_df['rank'] = range(1, len(importance_df) + 1)
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum()

    # 显示前top_n个特征
    logger.info(f"\n特征重要性排名 (前{min(top_n, len(importance_df))}个):")
    print(importance_df.head(top_n).to_string(index=False))

    # 累积重要性
    top_features = importance_df.head(top_n)
    cumulative = top_features['cumulative_importance'].iloc[-1]
    logger.info(f"\n前{top_n}个特征累积重要性: {cumulative:.2%}")

    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame,
                            top_n: int = 20,
                            save_path: Optional[str] = None) -> None:
    """
    绘制特征重要性条形图

    Args:
        importance_df: 特征重要性DataFrame
        top_n: 显示前N个特征
        save_path: 保存路径（如果为None则不保存）
    """
    try:
        import matplotlib.pyplot as plt

        top_features = importance_df.head(top_n)

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('重要性')
        plt.ylabel('特征')
        plt.title(f'特征重要性 (前{top_n}个)')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"特征重要性图已保存: {save_path}")

        plt.show()

    except ImportError:
        logger.warning("matplotlib未安装，跳过绘图")


# ============================================================================
# 相关性分析
# ============================================================================

def analyze_correlation(df: pd.DataFrame,
                       features: Optional[List[str]] = None,
                       threshold: float = 0.8) -> pd.DataFrame:
    """
    分析特征相关性

    Args:
        df: 包含特征的DataFrame
        features: 要分析的特征列表（如果为None则使用所有数值列）
        threshold: 高相关性的阈值

    Returns:
        相关系数矩阵DataFrame
    """
    if features is None:
        # 自动选择数值列
        features = df.select_dtypes(include=[np.number]).columns.tolist()

    # 计算相关系数
    corr_matrix = df[features].corr()

    # 找出高相关性特征对
    high_corr_pairs = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            corr_value = abs(corr_matrix.iloc[i, j])
            if corr_value >= threshold:
                high_corr_pairs.append({
                    'feature1': features[i],
                    'feature2': features[j],
                    'correlation': corr_matrix.iloc[i, j]
                })

    if high_corr_pairs:
        logger.info(f"\n高相关性特征对 (阈值: {threshold}):")
        high_corr_df = pd.DataFrame(high_corr_pairs)
        high_corr_df = high_corr_df.sort_values('correlation', ascending=False)
        print(high_corr_df.to_string(index=False))
    else:
        logger.info(f"\n没有找到高相关性特征对 (阈值: {threshold})")

    return corr_matrix


def plot_correlation_heatmap(corr_matrix: pd.DataFrame,
                            save_path: Optional[str] = None) -> None:
    """
    绘制相关性热力图

    Args:
        corr_matrix: 相关系数矩阵
        save_path: 保存路径（如果为None则不保存）
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix,
                   annot=True,
                   fmt='.2f',
                   cmap='coolwarm',
                   center=0,
                   square=True)
        plt.title('特征相关性热力图')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"相关性热力图已保存: {save_path}")

        plt.show()

    except ImportError:
        logger.warning("matplotlib或seaborn未安装，跳过绘图")


def remove_highly_correlated_features(df: pd.DataFrame,
                                      features: Optional[List[str]] = None,
                                      threshold: float = 0.9) -> Tuple[List[str], List[str]]:
    """
    移除高相关性特征

    Args:
        df: 包含特征的DataFrame
        features: 要分析的特征列表
        threshold: 高相关性阈值

    Returns:
        (保留的特征列表, 移除的特征列表)
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()

    corr_matrix = df[features].corr()

    # 找出要移除的特征
    to_remove = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                # 保留重要性更高的特征（需要feature_importance）
                # 这里简单保留第一个，移除第二个
                if features[j] not in to_remove:
                    to_remove.append(features[j])

    to_keep = [f for f in features if f not in to_remove]

    logger.info(f"\n移除高相关性特征 (阈值: {threshold}):")
    logger.info(f"移除 {len(to_remove)} 个特征: {to_remove}")
    logger.info(f"保留 {len(to_keep)} 个特征")

    return to_keep, to_remove


# ============================================================================
# 递归特征消除 (RFE)
# ============================================================================

def recursive_feature_elimination(estimator,
                                   X: pd.DataFrame,
                                   y: pd.Series,
                                   n_features_to_select: int = 20) -> RFE:
    """
    递归特征消除

    Args:
        estimator: 估计器（需要有coef_或feature_importances_属性）
        X: 特征DataFrame
        y: 目标变量Series
        n_features_to_select: 要选择的特征数量

    Returns:
        训练好的RFE对象
    """
    logger.info(f"\n执行递归特征消除 (RFE)...")
    logger.info(f"原始特征数: {len(X.columns)}")
    logger.info(f"目标特征数: {n_features_to_select}")

    # 创建RFE
    rfe = RFE(estimator=estimator,
             n_features_to_select=n_features_to_select,
             step=1)

    # 训练RFE
    rfe.fit(X, y)

    # 获取选中的特征
    selected_features = X.columns[rfe.support_].tolist()
    logger.info(f"\n选中的特征 ({len(selected_features)}个):")
    print(selected_features)

    # 特征排名
    feature_ranking = pd.DataFrame({
        'feature': X.columns,
        'rank': rfe.ranking_
    }).sort_values('rank')

    logger.info(f"\n特征排名 (前20个):")
    print(feature_ranking.head(20).to_string(index=False))

    return rfe


# ============================================================================
# 特征缩放和标准化
# ============================================================================

def scale_features(X_train: pd.DataFrame,
                  X_test: pd.DataFrame,
                  method: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    """
    特征缩放

    Args:
        X_train: 训练集特征DataFrame
        X_test: 测试集特征DataFrame
        method: 缩放方法 ('standard', 'minmax', 'robust')

    Returns:
        (缩放后的X_train, 缩放后的X_test, scaler对象)
    """
    if method == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        raise ValueError(f"未知的缩放方法: {method}")

    # 在训练集上拟合scaler
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    # 在测试集上使用相同的scaler
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    logger.info(f"\n使用 {method} 缩放特征")

    return X_train_scaled, X_test_scaled, scaler


def rolling_standardization(df: pd.DataFrame,
                            features: List[str],
                            window: int = 60) -> pd.DataFrame:
    """
    滚动窗口标准化（避免数据泄露）

    Args:
        df: DataFrame
        features: 要标准化的特征列表
        window: 滚动窗口大小

    Returns:
        标准化后的DataFrame
    """
    df_out = df.copy()

    for feature in features:
        if feature in df_out.columns:
            rolling_mean = df_out[feature].rolling(window=window, min_periods=1).mean()
            rolling_std = df_out[feature].rolling(window=window, min_periods=1).std()

            # 避免除零
            rolling_std = rolling_std.replace(0, 1)

            df_out[f'{feature}_scaled'] = (df_out[feature] - rolling_mean) / rolling_std

    logger.info(f"\n使用滚动窗口 (window={window}) 标准化 {len(features)} 个特征")

    return df_out


# ============================================================================
# 特征选择管道
# ============================================================================

def feature_selection_pipeline(X: pd.DataFrame,
                               y: pd.Series,
                               estimator,
                               corr_threshold: float = 0.9,
                               n_features: int = 20) -> Tuple[List[str], object]:
    """
    完整的特征选择管道

    1. 移除高相关性特征
    2. 递归特征消除

    Args:
        X: 特征DataFrame
        y: 目标变量Series
        estimator: 估计器
        corr_threshold: 相关性阈值
        n_features: 最终选择的特征数量

    Returns:
        (选中的特征列表, RFE对象)
    """
    logger.info("="*60)
    logger.info("特征选择管道")
    logger.info("="*60)

    original_features = X.columns.tolist()
    logger.info(f"原始特征数: {len(original_features)}")

    # 1. 移除高相关性特征
    selected, removed = remove_highly_correlated_features(
        X, features=original_features, threshold=corr_threshold
    )

    X_filtered = X[selected]

    # 2. 递归特征消除
    if len(selected) > n_features:
        rfe = recursive_feature_elimination(
            estimator, X_filtered, y, n_features_to_select=n_features
        )
        final_features = X_filtered.columns[rfe.support_].tolist()
    else:
        logger.info(f"特征数量 ({len(selected)}) 已小于目标数 ({n_features})，跳过RFE")
        rfe = None
        final_features = selected

    logger.info("\n" + "="*60)
    logger.info("特征选择完成")
    logger.info("="*60)
    logger.info(f"原始特征: {len(original_features)}")
    logger.info(f"相关性过滤后: {len(selected)}")
    logger.info(f"最终选择: {len(final_features)}")
    logger.info("="*60)

    return final_features, rfe


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    import argparse
    import xgboost as xgb
    from ..config import FEATURE_COLS, ALL_FEATURE_COLS, BENCHMARK_ETF, DB_MIN_DATE
    from ..utils.database import read_etf_data
    from ..models.target_engineering import create_excess_return_target, get_benchmark_returns

    parser = argparse.ArgumentParser(
        description="特征工程 - 特征重要性分析、相关性分析、特征选择",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 分析特征重要性
  python %(prog)s --code 510050 --importance

  # 分析特征相关性
  python %(prog)s --code 510050 --correlation

  # 完整特征选择管道
  python %(prog)s --code 510050 --pipeline --n-features 20
        """
    )

    parser.add_argument('--code', type=str, required=True, help='ETF代码')
    parser.add_argument('--importance', action='store_true', help='分析特征重要性')
    parser.add_argument('--correlation', action='store_true', help='分析特征相关性')
    parser.add_argument('--pipeline', action='store_true', help='执行特征选择管道')
    parser.add_argument('--n-features', type=int, default=20, help='选择的特征数量')
    parser.add_argument('--corr-threshold', type=float, default=0.9, help='相关性阈值')
    parser.add_argument('--top-n', type=int, default=20, help='显示前N个特征')
    parser.add_argument('--plot', type=str, default=None, help='保存图表路径')

    args = parser.parse_args()

    # 读取数据
    logger.info(f"\n读取ETF数据: {args.code}")
    df = read_etf_data(args.code, min_date=DB_MIN_DATE, use_advanced=True)

    if len(df) < 100:
        logger.error("数据不足")
        return

    # 计算超额收益目标
    benchmark_returns = get_benchmark_returns(BENCHMARK_ETF, min_date=DB_MIN_DATE)
    df = create_excess_return_target(df, benchmark_returns=benchmark_returns)

    # 移除早期数据（确保有足够历史计算指标）
    df = df.iloc[60:]  # 移除前60天，确保滚动窗口计算完整

    # 获取特征和目标（使用ALL_FEATURE_COLS包含高级特征）
    X = df[ALL_FEATURE_COLS]
    y = df['excess_return']

    # 只移除包含NaN的行（针对特征）
    valid_rows = X.notna().all(axis=1) & y.notna()
    X = X[valid_rows]
    y = y[valid_rows]

    logger.info(f"特征数: {len(X.columns)}")
    logger.info(f"样本数: {len(X)}")

    # 分析特征重要性
    if args.importance:
        logger.info("\n分析特征重要性...")

        # 训练一个XGBoost模型
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )

        from sklearn.model_selection import train_test_split
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, shuffle=False)

        model.fit(X_train, y_train)

        # 分析重要性
        importance_df = analyze_feature_importance(model, X.columns, top_n=args.top_n)

        # 绘图
        if args.plot:
            plot_feature_importance(importance_df, top_n=args.top_n, save_path=args.plot)

    # 分析相关性
    if args.correlation:
        logger.info("\n分析特征相关性...")

        corr_matrix = analyze_correlation(X, threshold=args.corr_threshold)

        if args.plot:
            plot_correlation_heatmap(corr_matrix, save_path=args.plot)

    # 特征选择管道
    if args.pipeline:
        logger.info("\n执行特征选择管道...")

        estimator = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=4,
            random_state=42
        )

        selected_features, _ = feature_selection_pipeline(
            X, y, estimator,
            corr_threshold=args.corr_threshold,
            n_features=args.n_features
        )

        logger.info(f"\n最终选择的特征 ({len(selected_features)}):")
        print(selected_features)


if __name__ == '__main__':
    main()
