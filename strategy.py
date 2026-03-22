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
    SPY 50/200 regime + trailing stop.
    - Bull: equal-weight long all stocks
    - Bear: short SPY 100%
    - Trailing stop: if SPY drops 3% from its high since regime entry, exit to cash
    - Re-enter bull only when SMA crossover re-confirms
    """
    tickers = list(data.keys())
    stock_tickers = [t for t in tickers if t != 'SPY']
    n_stocks = len(stock_tickers)
    weight = 1.0 / n_stocks

    ref_index = next(iter(data.values())).index
    positions = pd.DataFrame(0.0, index=ref_index, columns=tickers)

    spy_close = data['SPY']['Close'].reindex(ref_index, method='ffill')
    spy_fast = spy_close.rolling(40).mean()
    spy_slow = spy_close.rolling(120).mean()

    # State machine: 0=warmup, 1=bull, -1=bear, 2=stopped_out (waiting for re-entry)
    state = 0
    spy_peak = 0.0  # track SPY high since bull entry

    for i in range(len(ref_index)):
        f = spy_fast.iloc[i]
        s = spy_slow.iloc[i]
        price = spy_close.iloc[i]

        if pd.isna(f) or pd.isna(s):
            continue

        if state == 1:  # In bull
            spy_peak = max(spy_peak, price)
            # Trailing stop: 3% drawdown from peak
            if price < spy_peak * 0.97:
                state = 2  # stopped out → cash
                spy_peak = 0.0
            elif f < s * 0.995:
                state = -1  # regime turns bear
                spy_peak = 0.0
        elif state == -1:  # In bear
            if f > s * 1.005:
                state = 1
                spy_peak = price
        elif state == 2:  # Stopped out, waiting
            # Re-enter only when SMA confirms bull
            if f > s * 1.01:  # higher threshold to re-enter after stop
                state = 1
                spy_peak = price
            elif f < s * 0.995:
                state = -1
        elif state == 0:  # Warmup
            if f > s * 1.005:
                state = 1
                spy_peak = price
            elif f < s * 0.995:
                state = -1

        if state == 1:
            for t in stock_tickers:
                positions.loc[ref_index[i], t] = weight
        elif state == -1:
            positions.loc[ref_index[i], 'SPY'] = -1.00
        elif state == 2:
            # Stopped out → go short SPY (smaller position than full bear)
            positions.loc[ref_index[i], 'SPY'] = -1.00

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
