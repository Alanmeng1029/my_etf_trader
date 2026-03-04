# -*- coding: utf-8 -*-
"""
特征工程模块
提供特征重要性分析、相关性分析、特征选择等功能
"""

from .feature_engineering import (
    analyze_feature_importance,
    plot_feature_importance,
    analyze_correlation,
    plot_correlation_heatmap,
    remove_highly_correlated_features,
    recursive_feature_elimination,
    scale_features,
    rolling_standardization,
    feature_selection_pipeline,
)

__all__ = [
    'analyze_feature_importance',
    'plot_feature_importance',
    'analyze_correlation',
    'plot_correlation_heatmap',
    'remove_highly_correlated_features',
    'recursive_feature_elimination',
    'scale_features',
    'rolling_standardization',
    'feature_selection_pipeline',
]
