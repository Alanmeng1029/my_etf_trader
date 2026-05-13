import pandas as pd

from my_etf.backtest.engine import run_portfolio_backtest


def _prices(code, opens):
    dates = pd.date_range("2026-01-01", periods=len(opens), freq="D")
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": opens,
        "close": [price + 1 for price in opens],
    })


def test_backtest_executes_signal_on_next_day_open():
    predictions = pd.DataFrame({
        "date": ["2026-01-01"],
        "code": ["510300"],
        "predicted_return": [2.0],
    })

    results = run_portfolio_backtest(
        predictions,
        strategy_kind="regression",
        backtest_start_date="2026-01-01",
        top_n=1,
        rebalance_days=1,
        initial_capital=100000,
        commission_rate=0,
        slippage_bps=0,
        min_trade_amount=1000,
        max_position_weight=0.3,
        price_data={"510300": _prices("510300", [10, 20, 30])},
    )

    executed = results[results["rebalance"]]
    assert executed.iloc[0]["date"] == "2026-01-02"
    assert executed.iloc[0]["signal_date"] == "2026-01-01"
    assert executed.iloc[0]["execution_price_type"] == "open"
    assert executed.iloc[0]["capital"] == 70000


def test_classification_backtest_excludes_predicted_class_zero():
    predictions = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-01"],
        "code": ["510300", "510500"],
        "predicted_class": [0, 3],
        "class_4_prob": [0.99, 0.2],
    })

    results = run_portfolio_backtest(
        predictions,
        strategy_kind="classification",
        backtest_start_date="2026-01-01",
        top_n=5,
        rebalance_days=1,
        initial_capital=100000,
        commission_rate=0,
        slippage_bps=0,
        min_trade_amount=1000,
        max_position_weight=0.3,
        classification_prob_threshold=0.1,
        price_data={
            "510300": _prices("510300", [10, 11, 12]),
            "510500": _prices("510500", [20, 21, 22]),
        },
    )

    executed = results[results["rebalance"]].iloc[0]
    assert executed["positions"] == ["510500"]


def test_backtest_skips_trades_below_min_trade_amount():
    predictions = pd.DataFrame({
        "date": ["2026-01-01"],
        "code": ["510300"],
        "predicted_return": [2.0],
    })

    results = run_portfolio_backtest(
        predictions,
        strategy_kind="regression",
        backtest_start_date="2026-01-01",
        top_n=1,
        rebalance_days=1,
        initial_capital=100000,
        commission_rate=0,
        slippage_bps=0,
        min_trade_amount=50000,
        max_position_weight=0.3,
        price_data={"510300": _prices("510300", [10, 20, 30])},
    )

    executed = results[results["rebalance"]].iloc[0]
    assert executed["positions"] == []
    assert executed["turnover"] == 0
