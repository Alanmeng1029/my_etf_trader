"""测试配置模块"""
import pytest


def test_import_config():
    """测试导入配置"""
    from src.my_etf.config import (
        FEATURE_COLS,
        MODEL_PARAMS,
        BACKTEST_CONFIG,
        STRATEGY_CONFIGS,
    )
    assert isinstance(FEATURE_COLS, list)
    assert isinstance(MODEL_PARAMS, dict)
    assert isinstance(BACKTEST_CONFIG, dict)
    assert isinstance(STRATEGY_CONFIGS, list)


def test_feature_cols():
    """测试特征列配置"""
    from src.my_etf.config import FEATURE_COLS

    assert isinstance(FEATURE_COLS, list)
    assert 'close' in FEATURE_COLS
    assert 'MA5' in FEATURE_COLS
    assert len(FEATURE_COLS) == 20


def test_model_params():
    """测试模型参数配置"""
    from src.my_etf.config import MODEL_PARAMS

    assert 'n_estimators' in MODEL_PARAMS
    assert 'max_depth' in MODEL_PARAMS
    assert MODEL_PARAMS['n_estimators'] > 0
    assert MODEL_PARAMS['max_depth'] > 0


def test_backtest_config():
    """测试回测配置"""
    from src.my_etf.config import BACKTEST_CONFIG

    assert 'INITIAL_CAPITAL' in BACKTEST_CONFIG
    assert 'REBALANCE_DAYS' in BACKTEST_CONFIG
    assert BACKTEST_CONFIG['INITIAL_CAPITAL'] > 0
    assert BACKTEST_CONFIG['REBALANCE_DAYS'] > 0


def test_strategy_configs():
    """测试策略配置"""
    from src.my_etf.config import STRATEGY_CONFIGS

    assert len(STRATEGY_CONFIGS) > 0
    for config in STRATEGY_CONFIGS:
        assert 'name' in config
        assert 'top_n' in config
        assert config['top_n'] > 0
