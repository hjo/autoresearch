"""
Trading strategy — the file the agent modifies.
Everything is fair game: indicators, signals, position sizing, risk management.

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


# ---------------------------------------------------------------------------
# Strategy: generate target positions for each ticker
# ---------------------------------------------------------------------------

def generate_signals(data):
    """
    Generate target positions for each ticker.

    Args:
        data: dict mapping ticker -> DataFrame with OHLCV columns
              (Open, High, Low, Close, Volume)

    Returns:
        DataFrame with tickers as columns, timestamps as index.
        Values: target allocation fraction per ticker.
        e.g., 1/N for equal-weight long, 0 for flat, negative for short.
    """
    n_tickers = len(data)
    weight = 1.0 / n_tickers  # equal weight per ticker

    # Use first ticker's index as reference
    ref_index = next(iter(data.values())).index
    positions = pd.DataFrame(0.0, index=ref_index, columns=list(data.keys()))

    for ticker, df in data.items():
        close = df['Close']

        # Simple moving average crossover
        fast_sma = close.rolling(window=10).mean()
        slow_sma = close.rolling(window=30).mean()

        # Go long when fast > slow, flat otherwise
        signal = pd.Series(0.0, index=df.index)
        signal[fast_sma > slow_sma] = weight

        positions[ticker] = signal

    return positions


# ---------------------------------------------------------------------------
# Main: run strategy and report results
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    start_time = time.time()

    # --- Training set ---
    train_data = get_train_data()
    train_positions = generate_signals(train_data)
    train_results = backtest(train_positions, train_data)

    # --- Validation set ---
    val_data = get_val_data()
    val_positions = generate_signals(val_data)
    val_results = backtest(val_positions, val_data)

    elapsed = time.time() - start_time

    # --- Print summary (matches autoresearch output format) ---
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
