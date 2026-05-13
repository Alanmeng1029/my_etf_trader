"""Model training and prediction public API."""

_EXPORTS = {
    'create_target': ('.train', 'create_target'),
    'get_benchmark_returns': ('.train', 'get_benchmark_returns'),
    'clean_data': ('.train', 'clean_data'),
    'split_data': ('.train', 'split_data'),
    'train_separate_models': ('.train', 'train_separate_models'),
    'train_single_model': ('.train', 'train_single_model'),
    'save_model': ('.train', 'save_model'),
    'train_unified_model': ('.train', 'train_unified_model'),
    'train_two_stage_model': ('.train', 'train_two_stage_model'),
    'evaluate_results': ('.train', 'evaluate_results'),
    'create_summary_csv': ('.train', 'create_summary_csv'),
    'train_main': ('.train', 'main'),
    'load_latest_model_and_scaler': ('.predict', 'load_latest_model_and_scaler'),
    'generate_daily_predictions': ('.predict', 'generate_daily_predictions'),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
