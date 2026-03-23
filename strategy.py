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


FAST_SMA_DAYS = 7
SLOW_SMA_DAYS = 19
TRAILING_STOP = 0.05
HYSTERESIS_UP = 1.015
HYSTERESIS_DN = 0.985
REENTRY_BAR = 1.02
TOP_N = 3

# Safe havens to rotate into during bear regime
SAFE_HAVENS = ["GLD"]
SAFE_HAVEN_ALLOC = 0.50  # 50% into gold during bear (rest cash)


def _bars_per_day(data):
    idx = next(iter(data.values())).index
    if len(idx) < 10:
        return 1
    if hasattr(idx, 'date'):
        bars_per_date = pd.Series(idx).groupby(idx.date).count()
        return max(1, int(round(bars_per_date.median())))
    return 1


def generate_signals(data):
    """
    All-weather regime strategy:
    - Bull: top-3 momentum stocks
    - Bear/stopped: rotate into safe havens (TLT + GLD)
    - 5% trailing stop (tightens to 3% after 5% gain)
    """
    tickers = list(data.keys())
    stock_tickers = [t for t in tickers if t not in ('SPY',) + tuple(SAFE_HAVENS)]
    available_havens = [h for h in SAFE_HAVENS if h in tickers]

    ref_index = next(iter(data.values())).index
    positions = pd.DataFrame(0.0, index=ref_index, columns=tickers)

    bpd = _bars_per_day(data)
    fast_bars = max(2, int(FAST_SMA_DAYS * bpd))
    slow_bars = max(5, int(SLOW_SMA_DAYS * bpd))
    mom_bars = max(5, int(15 * bpd))

    spy_close = data['SPY']['Close'].reindex(ref_index, method='ffill')
    spy_high = data['SPY']['High'].reindex(ref_index, method='ffill') if 'High' in data['SPY'].columns else spy_close
    spy_low = data['SPY']['Low'].reindex(ref_index, method='ffill') if 'Low' in data['SPY'].columns else spy_close
    spy_fast = spy_close.rolling(fast_bars).mean()
    spy_slow = spy_close.rolling(slow_bars).mean()

    # RSI for exposure scaling
    rsi_period = max(5, int(14 * bpd))
    spy_rsi = ta.momentum.RSIIndicator(spy_close, window=min(rsi_period, 14)).rsi()

    close_matrix = pd.DataFrame(
        {t: data[t]['Close'].reindex(ref_index, method='ffill') for t in stock_tickers}
    )
    momentum = close_matrix.pct_change(mom_bars)

    # Gold trend for conditional safe haven
    gld_close = data['GLD']['Close'].reindex(ref_index, method='ffill') if 'GLD' in data else None
    gld_sma = gld_close.rolling(slow_bars).mean() if gld_close is not None else None

    state = 0
    spy_peak = 0.0
    spy_entry = 0.0
    selected = []
    weight = 0.0
    exposure_mult = 1.0  # RSI-based exposure multiplier

    for i in range(len(ref_index)):
        f = spy_fast.iloc[i]
        s = spy_slow.iloc[i]
        price = spy_close.iloc[i]

        if pd.isna(f) or pd.isna(s):
            continue

        prev_state = state

        if state == 1:
            spy_peak = max(spy_peak, price)
            gain = (spy_peak / spy_entry - 1) if spy_entry > 0 else 0
            stop_pct = 0.03 if gain > 0.05 else TRAILING_STOP
            if price < spy_peak * (1 - stop_pct):
                state = 2
                spy_peak = 0.0
            elif f < s * HYSTERESIS_DN:
                state = -1
                spy_peak = 0.0
        elif state == -1:
            if f > s * HYSTERESIS_UP:
                state = 1
                spy_peak = price
                spy_entry = price
        elif state == 2:
            if f > s * REENTRY_BAR:
                state = 1
                spy_peak = price
                spy_entry = price
            elif f < s * HYSTERESIS_DN:
                state = -1
        elif state == 0:
            if f > s * HYSTERESIS_UP:
                state = 1
                spy_peak = price
                spy_entry = price
            elif f < s * HYSTERESIS_DN:
                state = -1

        if state == 1:
            # Bull: top-N momentum stocks
            if prev_state != 1:
                mom = momentum.iloc[i]
                valid = mom.dropna()
                if len(valid) >= TOP_N:
                    selected = valid.nlargest(TOP_N).index.tolist()
                else:
                    selected = stock_tickers[:TOP_N]
                weight = 1.0 / len(selected)
                exposure_mult = 1.0  # reset on entry

            # RSI exposure scaling (discrete jumps at extremes)
            rsi_val = spy_rsi.iloc[i] if not pd.isna(spy_rsi.iloc[i]) else 50
            if rsi_val < 30:
                exposure_mult = 1.3   # buy the dip
            elif rsi_val > 70:
                exposure_mult = 0.5   # take profit
            elif 40 < rsi_val < 60:
                exposure_mult = 1.0   # normal

            for t in selected:
                positions.loc[ref_index[i], t] = weight * exposure_mult

        elif state in (-1, 2):
            # Bear/stopped: rotate into gold ONLY if gold is trending up
            if gld_close is not None and gld_sma is not None:
                g = gld_close.iloc[i]
                gs = gld_sma.iloc[i]
                if not pd.isna(gs) and g > gs:
                    positions.loc[ref_index[i], 'GLD'] = SAFE_HAVEN_ALLOC
            # Otherwise stay in cash

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
