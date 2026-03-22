"""
Trading strategy — the file the agent modifies.
Usage: uv run strategy.py
"""

import time
import numpy as np
import pandas as pd
import ta

from prepare import (
    TICKERS, TIME_BUDGET, INITIAL_CAPITAL,
    get_train_data, get_val_data, backtest,
)


def generate_signals(data):
    """
    Bi-directional SPY regime.
    - Bullish (50-SMA > 200-SMA): equal-weight long all stocks
    - Bearish (50-SMA < 200-SMA): short SPY with 30% allocation
    - Hysteresis to prevent whipsaws
    """
    tickers = list(data.keys())
    stock_tickers = [t for t in tickers if t != 'SPY']
    n_stocks = len(stock_tickers)
    weight = 1.0 / n_stocks

    ref_index = next(iter(data.values())).index
    positions = pd.DataFrame(0.0, index=ref_index, columns=tickers)

    spy_close = data['SPY']['Close'].reindex(ref_index, method='ffill')
    spy_fast = spy_close.rolling(50).mean()
    spy_slow = spy_close.rolling(200).mean()

    # Regime: 1 = bullish, -1 = bearish, 0 = warmup
    regime = pd.Series(0, index=ref_index, dtype=int)
    state = 0
    for i in range(len(regime)):
        f = spy_fast.iloc[i]
        s = spy_slow.iloc[i]
        if pd.isna(f) or pd.isna(s):
            continue
        if state <= 0 and f > s * 1.005:
            state = 1
        elif state >= 0 and f < s * 0.995:
            state = -1
        regime.iloc[i] = state

    # Bullish: long individual stocks
    for ticker in stock_tickers:
        positions[ticker] = (regime == 1).astype(float) * weight

    # Bearish: short SPY with smaller allocation (hedge)
    positions['SPY'] = (regime == -1).astype(float) * (-0.30)

    return positions


if __name__ == "__main__":
    start_time = time.time()

    train_data = get_train_data()
    train_positions = generate_signals(train_data)
    train_results = backtest(train_positions, train_data)

    val_data = get_val_data()
    val_positions = generate_signals(val_data)
    val_results = backtest(val_positions, val_data)

    elapsed = time.time() - start_time

    print("---")
    print(f"val_sharpe:       {val_results['sharpe']:.6f}")
    print(f"val_return:       {val_results['total_return']:.4f}")
    print(f"val_max_drawdown: {val_results['max_drawdown']:.4f}")
    print(f"val_win_rate:     {val_results['win_rate']:.4f}")
    print(f"val_num_trades:   {val_results['num_trades']}")
    print(f"val_profit_factor:{val_results['profit_factor']:.2f}")
    print(f"val_final_equity: {val_results['final_equity']:.2f}")
    print(f"train_sharpe:     {train_results['sharpe']:.6f}")
    print(f"train_return:     {train_results['total_return']:.4f}")
    print(f"total_seconds:    {elapsed:.1f}")
