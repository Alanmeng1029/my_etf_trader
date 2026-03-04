"""模型训练和预测模块"""

from .train import (
    create_target,
    get_benchmark_returns,
    clean_data,
    split_data,
    train_separate_models,
    train_single_model,
    save_model,
    train_unified_model,
    train_two_stage_model,
    evaluate_results,
    create_summary_csv,
    main as train_main,
)
from .predict import (
    load_latest_model_and_scaler,
    generate_daily_predictions,
)

__all__ = [
    'create_target',
    'get_benchmark_returns',
    'clean_data',
    'split_data',
    'train_separate_models',
    'train_single_model',
    'save_model',
    'train_unified_model',
    'train_two_stage_model',
    'evaluate_results',
    'create_summary_csv',
    'train_main',
    'load_latest_model_and_scaler',
    'generate_daily_predictions',
]
