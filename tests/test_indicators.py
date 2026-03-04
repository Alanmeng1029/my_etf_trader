"""测试指标计算"""
import pandas as pd
import numpy as np
import pytest


def create_sample_data():
    """创建示例数据用于测试"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        '日期': dates,
        '开盘': 100 + np.random.randn(100).cumsum(),
        '收盘': 100 + np.random.randn(100).cumsum(),
        '最高': 100 + np.random.randn(100).cumsum() + 1,
        '最低': 100 + np.random.randn(100).cumsum() - 1,
        '成交量': np.random.randint(1000000, 10000000, 100),
        '成交额': np.random.randint(1000000000, 10000000000, 100),
    })
    return data


def test_calculate_ma():
    """测试移动平均线计算"""
    from src.my_etf.indicators.calculator import calculate_ma

    df = create_sample_data()
    result = calculate_ma(df)

    assert 'MA5' in result.columns
    assert 'MA10' in result.columns
    assert 'MA20' in result.columns
    assert 'MA60' in result.columns
    assert len(result) == len(df)


def test_calculate_macd():
    """测试MACD指标计算"""
    from src.my_etf.indicators.calculator import calculate_macd

    df = create_sample_data()
    result = calculate_macd(df)

    assert 'dif' in result.columns
    assert 'dea' in result.columns
    assert 'macd' in result.columns
    assert len(result) == len(df)


def test_calculate_kdj():
    """测试KDJ指标计算"""
    from src.my_etf.indicators.calculator import calculate_kdj

    df = create_sample_data()
    result = calculate_kdj(df)

    assert 'k' in result.columns
    assert 'd' in result.columns
    assert 'j' in result.columns
    assert len(result) == len(df)


def test_calculate_rsi():
    """测试RSI指标计算"""
    from src.my_etf.indicators.calculator import calculate_rsi

    df = create_sample_data()
    result = calculate_rsi(df)

    assert 'rsi' in result.columns
    assert len(result) == len(df)


def test_calculate_boll():
    """测试布林带计算"""
    from src.my_etf.indicators.calculator import calculate_boll

    df = create_sample_data()
    result = calculate_boll(df)

    assert 'boll_upper' in result.columns
    assert 'boll_middle' in result.columns
    assert 'boll_lower' in result.columns
    assert len(result) == len(df)


def test_calculate_all_indicators():
    """测试计算所有指标"""
    from src.my_etf.indicators.calculator import calculate_all_indicators

    df = create_sample_data()
    result = calculate_all_indicators(df)

    expected_cols = ['MA5', 'MA10', 'MA20', 'MA60', 'dif', 'dea', 'macd', 'k', 'd', 'j', 'rsi', 'boll_upper', 'boll_middle', 'boll_lower']
    for col in expected_cols:
        assert col in result.columns
