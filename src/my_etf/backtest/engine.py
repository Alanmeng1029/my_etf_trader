"""Unified portfolio backtest engine with explicit execution assumptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import BACKTEST_CONFIG, BENCHMARK_ETF
from ..utils.database import read_etf_data


@dataclass
class RebalanceOutcome:
    positions: Dict[str, float]
    cash: float
    transaction_cost: float
    turnover: float
    execution_price_type: str
    fallback_count: int


def _normalise_price_df(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")
    for col in ["open", "close"]:
        if col not in result.columns:
            result[col] = np.nan
    return result[["date", "open", "close"]].sort_values("date").reset_index(drop=True)


def _load_price_data(codes: List[str], start_date: str) -> Dict[str, pd.DataFrame]:
    data = {}
    for code in codes:
        try:
            df = read_etf_data(code, min_date=start_date, use_advanced=False)
            if df is not None and not df.empty and {"date", "close"}.issubset(df.columns):
                data[code] = _normalise_price_df(df)
        except Exception:
            continue
    return data


def _calendar_from_prices(price_data: Dict[str, pd.DataFrame], start_date: str, end_date: Optional[str]) -> List[str]:
    dates = set()
    for df in price_data.values():
        dates.update(df["date"].tolist())
    calendar = sorted(date for date in dates if date >= start_date)
    if end_date:
        calendar = [date for date in calendar if date <= end_date]
    return calendar


def _price_lookup(price_data: Dict[str, pd.DataFrame], date: str, field: str) -> Dict[str, float]:
    prices = {}
    for code, df in price_data.items():
        rows = df[df["date"] == date]
        if rows.empty:
            continue
        value = rows.iloc[0].get(field)
        if pd.notna(value) and value > 0:
            prices[code] = float(value)
    return prices


def _execution_prices(price_data: Dict[str, pd.DataFrame], date: str) -> tuple[Dict[str, float], str, int]:
    open_prices = _price_lookup(price_data, date, "open")
    close_prices = _price_lookup(price_data, date, "close")
    prices = {}
    fallback_count = 0
    for code in price_data:
        if code in open_prices:
            prices[code] = open_prices[code]
        elif code in close_prices:
            prices[code] = close_prices[code]
            fallback_count += 1
    price_type = "open" if fallback_count == 0 else "mixed_open_close_fallback"
    if not open_prices and close_prices:
        price_type = "close_fallback"
    return prices, price_type, fallback_count


def _portfolio_value(cash: float, positions: Dict[str, float], prices: Dict[str, float]) -> float:
    value = cash
    for code, shares in positions.items():
        price = prices.get(code)
        if price is not None:
            value += shares * price
    return float(value)


def _target_weights(
    day_predictions: pd.DataFrame,
    strategy_kind: str,
    top_n: int,
    max_position_weight: float,
    classification_prob_threshold: float,
    hedge_etf_code: str,
) -> Dict[str, float]:
    if day_predictions.empty:
        return {}

    if strategy_kind == "classification_cash_hedge":
        hedge_rows = day_predictions[day_predictions["code"] == hedge_etf_code]
        if not hedge_rows.empty and int(hedge_rows.iloc[0].get("predicted_class", -1)) == 0:
            return {}
        strategy_kind = "classification"

    if strategy_kind == "classification":
        eligible = day_predictions[
            (day_predictions.get("predicted_class") != 0)
            & (day_predictions.get("class_4_prob", 0) >= classification_prob_threshold)
        ].copy()
        if eligible.empty:
            return {}
        selected = eligible.sort_values("class_4_prob", ascending=False).head(top_n)
    else:
        if "predicted_return" not in day_predictions.columns:
            return {}
        selected = day_predictions.sort_values("predicted_return", ascending=False).head(top_n)

    if selected.empty:
        return {}

    raw_weight = 1.0 / len(selected)
    weight = min(raw_weight, max_position_weight)
    return {str(code): weight for code in selected["code"].tolist()}


def _rebalance(
    cash: float,
    positions: Dict[str, float],
    target_weights: Dict[str, float],
    prices: Dict[str, float],
    commission_rate: float,
    slippage_bps: float,
    min_trade_amount: float,
    execution_price_type: str,
    fallback_count: int,
) -> RebalanceOutcome:
    cost_rate = commission_rate + slippage_bps / 10000.0
    portfolio_value = _portfolio_value(cash, positions, prices)
    new_positions = dict(positions)
    transaction_cost = 0.0
    traded_value = 0.0

    all_codes = set(new_positions) | set(target_weights)

    for code in list(all_codes):
        price = prices.get(code)
        if price is None:
            continue
        current_value = new_positions.get(code, 0.0) * price
        target_value = portfolio_value * target_weights.get(code, 0.0)
        delta = target_value - current_value
        if delta >= 0 or abs(delta) < min_trade_amount:
            continue
        sell_value = min(abs(delta), current_value)
        shares_to_sell = sell_value / price
        new_positions[code] = max(new_positions.get(code, 0.0) - shares_to_sell, 0.0)
        cash += sell_value * (1 - cost_rate)
        transaction_cost += sell_value * cost_rate
        traded_value += sell_value
        if new_positions[code] <= 1e-10:
            new_positions.pop(code, None)

    for code, weight in target_weights.items():
        price = prices.get(code)
        if price is None:
            continue
        current_value = new_positions.get(code, 0.0) * price
        target_value = portfolio_value * weight
        delta = target_value - current_value
        if delta <= 0 or delta < min_trade_amount:
            continue
        buy_value = min(delta, cash)
        if buy_value < min_trade_amount:
            continue
        shares_to_buy = buy_value * (1 - cost_rate) / price
        new_positions[code] = new_positions.get(code, 0.0) + shares_to_buy
        cash -= buy_value
        transaction_cost += buy_value * cost_rate
        traded_value += buy_value

    turnover = traded_value / portfolio_value if portfolio_value > 0 else 0.0
    return RebalanceOutcome(
        positions=new_positions,
        cash=float(cash),
        transaction_cost=float(transaction_cost),
        turnover=float(turnover),
        execution_price_type=execution_price_type,
        fallback_count=fallback_count,
    )


def run_portfolio_backtest(
    predictions_df: pd.DataFrame,
    strategy_kind: str = "regression",
    backtest_start_date: str = "2025-01-01",
    backtest_end_date: Optional[str] = None,
    top_n: int = 5,
    rebalance_days: int = 5,
    initial_capital: float = 100000,
    strategy_name: str = "TOP5",
    commission_rate: Optional[float] = None,
    slippage_bps: Optional[float] = None,
    min_trade_amount: Optional[float] = None,
    max_position_weight: Optional[float] = None,
    classification_prob_threshold: Optional[float] = None,
    hedge_etf_code: str = "510500",
    price_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Run a T-close signal, T+1 execution portfolio backtest."""
    if predictions_df is None or predictions_df.empty:
        return pd.DataFrame()

    commission_rate = BACKTEST_CONFIG["COMMISSION_RATE"] if commission_rate is None else commission_rate
    slippage_bps = BACKTEST_CONFIG["SLIPPAGE_BPS"] if slippage_bps is None else slippage_bps
    min_trade_amount = BACKTEST_CONFIG["MIN_TRADE_AMOUNT"] if min_trade_amount is None else min_trade_amount
    max_position_weight = BACKTEST_CONFIG["MAX_POSITION_WEIGHT"] if max_position_weight is None else max_position_weight
    classification_prob_threshold = (
        BACKTEST_CONFIG["CLASSIFICATION_PROB_THRESHOLD"]
        if classification_prob_threshold is None else classification_prob_threshold
    )

    predictions = predictions_df.copy()
    predictions["date"] = pd.to_datetime(predictions["date"]).dt.strftime("%Y-%m-%d")
    predictions["code"] = predictions["code"].astype(str)

    codes = sorted(predictions["code"].unique().tolist())
    if price_data is None:
        price_data = _load_price_data(codes, backtest_start_date)
    else:
        price_data = {code: _normalise_price_df(df) for code, df in price_data.items()}

    calendar = _calendar_from_prices(price_data, backtest_start_date, backtest_end_date)
    if len(calendar) < 2:
        return pd.DataFrame()

    prediction_dates = set(predictions["date"].unique().tolist())
    cash = float(initial_capital)
    positions: Dict[str, float] = {}
    rows = []
    last_portfolio_value = float(initial_capital)
    last_rebalance_idx = -rebalance_days
    pending = None

    for idx, current_date in enumerate(calendar):
        rebalance_flag = False
        transaction_cost = 0.0
        turnover = 0.0
        signal_date = None
        execution_price_type = ""
        fallback_count = 0

        if pending and pending["execution_date"] == current_date:
            prices, execution_price_type, fallback_count = _execution_prices(price_data, current_date)
            outcome = _rebalance(
                cash,
                positions,
                pending["target_weights"],
                prices,
                commission_rate,
                slippage_bps,
                min_trade_amount,
                execution_price_type,
                fallback_count,
            )
            cash = outcome.cash
            positions = outcome.positions
            transaction_cost = outcome.transaction_cost
            turnover = outcome.turnover
            signal_date = pending["signal_date"]
            rebalance_flag = True
            pending = None

        if (
            current_date in prediction_dates
            and pending is None
            and idx + 1 < len(calendar)
            and idx - last_rebalance_idx >= rebalance_days
        ):
            day_predictions = predictions[predictions["date"] == current_date]
            target_weights = _target_weights(
                day_predictions,
                strategy_kind,
                top_n,
                max_position_weight,
                classification_prob_threshold,
                hedge_etf_code,
            )
            pending = {
                "signal_date": current_date,
                "execution_date": calendar[idx + 1],
                "target_weights": target_weights,
            }
            last_rebalance_idx = idx

        close_prices = _price_lookup(price_data, current_date, "close")
        value = _portfolio_value(cash, positions, close_prices)
        daily_return = (value / last_portfolio_value - 1) if rows and last_portfolio_value else 0.0
        position_value = value - cash
        cash_weight = cash / value if value > 0 else 1.0

        rows.append({
            "strategy_name": strategy_name,
            "date": current_date,
            "portfolio_value": value,
            "capital": cash,
            "positions": sorted(positions.keys()),
            "daily_return": daily_return * 100,
            "transaction_cost": transaction_cost,
            "turnover": turnover,
            "rebalance": rebalance_flag,
            "signal_date": signal_date,
            "execution_date": current_date if rebalance_flag else None,
            "execution_price_type": execution_price_type,
            "execution_price_fallback_count": fallback_count,
            "cash_weight": cash_weight,
            "position_weight": position_value / value if value > 0 else 0.0,
            "position_count": len(positions),
        })
        last_portfolio_value = value

    return pd.DataFrame(rows)


def calculate_enhanced_metrics(results_df: pd.DataFrame, initial_capital: float) -> dict:
    """Calculate reporting metrics for the unified backtest output."""
    if results_df is None or results_df.empty:
        return {}

    df = results_df.copy()
    final_value = float(df["portfolio_value"].iloc[-1])
    total_return = (final_value / initial_capital - 1) * 100
    returns = df["portfolio_value"].pct_change().dropna()
    num_days = len(df)

    cumulative = df["portfolio_value"] / initial_capital
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_drawdown = float(drawdown.min() * 100)

    annualized_return = (final_value / initial_capital) ** (252 / num_days) - 1 if num_days else 0.0
    vol = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
    sharpe = (annualized_return - 0.03) / vol if vol > 0 else 0.0
    downside = returns[returns < 0].std() * np.sqrt(252) if (returns < 0).sum() > 1 else 0.0
    sortino = (annualized_return - 0.03) / downside if downside > 0 else 0.0
    calmar = annualized_return / abs(max_drawdown / 100) if max_drawdown < 0 else 0.0

    win_rate = (returns > 0).mean() * 100 if len(returns) else 0.0
    avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0.0
    avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0.0
    profit_loss_ratio = -avg_win / avg_loss if avg_loss < 0 else 0.0
    total_cost = float(df.get("transaction_cost", pd.Series(dtype=float)).sum())
    gross_profit = max(final_value - initial_capital, 0.0)
    cost_to_profit = total_cost / gross_profit * 100 if gross_profit > 0 else 0.0

    return {
        "total_return": round(total_return, 2),
        "final_value": round(final_value, 2),
        "max_drawdown": round(max_drawdown, 2),
        "annualized_return": round(annualized_return * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "calmar_ratio": round(calmar, 2),
        "win_rate": round(win_rate, 2),
        "profit_loss_ratio": round(profit_loss_ratio, 2),
        "total_trades": int(df.get("rebalance", pd.Series(dtype=bool)).sum()),
        "trading_days": num_days,
        "turnover": round(float(df.get("turnover", pd.Series(dtype=float)).sum()), 4),
        "total_transaction_cost": round(total_cost, 2),
        "cost_to_profit_pct": round(cost_to_profit, 2),
        "average_position_count": round(float(df.get("position_count", pd.Series(dtype=float)).mean()), 2),
        "cash_days": int((df.get("position_count", pd.Series(dtype=float)) == 0).sum()),
        "execution_fallback_count": int(df.get("execution_price_fallback_count", pd.Series(dtype=float)).sum()),
        "start_date": df["date"].iloc[0],
        "end_date": df["date"].iloc[-1],
        "benchmark": BENCHMARK_ETF,
    }
