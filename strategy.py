"""
Trading strategy — the file the agent modifies.
Usage:
    uv run strategy.py                  # Quick: train/val on hourly data
    uv run strategy.py --walk-forward   # Robust: walk-forward on 10y daily data
"""

import time
import numpy as np
import pandas as pd
import ta

from prepare import (
    TICKERS, TIME_BUDGET, INITIAL_CAPITAL,
    get_train_data, get_val_data, backtest, walk_forward,
)


# Strategy parameters in TRADING DAYS (frequency-independent)
FAST_SMA_DAYS = 7       # ~1.5 weeks
SLOW_SMA_DAYS = 19      # ~1 month
TRAILING_STOP = 0.05    # 5% from peak
HYSTERESIS_UP = 1.015   # 1.5% buffer to enter bull
HYSTERESIS_DN = 0.985   # 1.5% buffer to enter bear
REENTRY_BAR = 1.02      # 2% buffer to re-enter after stop
SHORT_ALLOC = 1.00      # 100% short SPY in bear


def _bars_per_day(data):
    """Detect bars per trading day from data frequency."""
    idx = next(iter(data.values())).index
    if len(idx) < 10:
        return 1
    # Count bars per calendar date
    if hasattr(idx, 'date'):
        bars_per_date = pd.Series(idx).groupby(idx.date).count()
        median_bpd = bars_per_date.median()
        return max(1, int(round(median_bpd)))
    return 1  # daily data


def generate_signals(data):
    """
    SPY regime switch + trailing stop.
    Frequency-adaptive: works on both hourly and daily data.
    - Bull (fast SMA > slow SMA): equal-weight long all stocks
    - Bear (fast SMA < slow SMA): short SPY
    - 3% trailing stop in bull → flip to short
    """
    tickers = list(data.keys())
    stock_tickers = [t for t in tickers if t != 'SPY']
    n_stocks = len(stock_tickers)
    weight = 1.0 / n_stocks

    ref_index = next(iter(data.values())).index
    positions = pd.DataFrame(0.0, index=ref_index, columns=tickers)

    # Convert day-based parameters to bar counts
    bpd = _bars_per_day(data)
    fast_bars = max(2, int(FAST_SMA_DAYS * bpd))
    slow_bars = max(5, int(SLOW_SMA_DAYS * bpd))

    spy_close = data['SPY']['Close'].reindex(ref_index, method='ffill')
    spy_fast = spy_close.rolling(fast_bars).mean()
    spy_slow = spy_close.rolling(slow_bars).mean()

    # State machine: 0=warmup, 1=bull, -1=bear, 2=stopped_out
    state = 0
    spy_peak = 0.0

    for i in range(len(ref_index)):
        f = spy_fast.iloc[i]
        s = spy_slow.iloc[i]
        price = spy_close.iloc[i]

        if pd.isna(f) or pd.isna(s):
            continue

        if state == 1:  # In bull
            spy_peak = max(spy_peak, price)
            if price < spy_peak * (1 - TRAILING_STOP):
                state = 2
                spy_peak = 0.0
            elif f < s * HYSTERESIS_DN:
                state = -1
                spy_peak = 0.0
        elif state == -1:  # In bear
            if f > s * HYSTERESIS_UP:
                state = 1
                spy_peak = price
        elif state == 2:  # Stopped out
            if f > s * REENTRY_BAR:
                state = 1
                spy_peak = price
            elif f < s * HYSTERESIS_DN:
                state = -1
        elif state == 0:  # Warmup
            if f > s * HYSTERESIS_UP:
                state = 1
                spy_peak = price
            elif f < s * HYSTERESIS_DN:
                state = -1

        if state == 1:
            for t in stock_tickers:
                positions.loc[ref_index[i], t] = weight
        elif state in (-1, 2):
            positions.loc[ref_index[i], 'SPY'] = -SHORT_ALLOC

    return positions


if __name__ == "__main__":
    import sys

    start_time = time.time()

    if "--walk-forward" in sys.argv:
        print("Running walk-forward analysis (10y daily data)...")
        print("=" * 60)
        wf = walk_forward(generate_signals)
        elapsed = time.time() - start_time

        print(f"Windows:              {wf['n_windows']}")
        print(f"Avg Sharpe:           {wf['avg_sharpe']:.4f}")
        print(f"Median Sharpe:        {wf['median_sharpe']:.4f}")
        print(f"Std Sharpe:           {wf['std_sharpe']:.4f}")
        print(f"Min/Max Sharpe:       {wf['min_sharpe']:.4f} / {wf['max_sharpe']:.4f}")
        print(f"% Positive Sharpe:    {wf['pct_positive_sharpe']:.1%}")
        print(f"Avg Return:           {wf['avg_return']:.4f}")
        print(f"Avg Max Drawdown:     {wf['avg_max_drawdown']:.4f}")
        print(f"Worst Drawdown:       {wf['worst_drawdown']:.4f}")
        print(f"Avg Win Rate:         {wf['avg_win_rate']:.4f}")
        print(f"Total seconds:        {elapsed:.1f}")
        print()
        print("Per-window results:")
        print(f"{'Test Period':<26} {'Sharpe':>8} {'Return':>8} {'MaxDD':>8} {'Trades':>7}")
        print("-" * 60)
        for w in wf['windows']:
            period = f"{w['test_start']} → {w['test_end']}"
            print(f"{period:<26} {w['sharpe']:>8.3f} {w['total_return']:>8.3%} "
                  f"{w['max_drawdown']:>8.3%} {w['num_trades']:>7}")
    else:
        # Quick iteration: train/val on hourly data
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
