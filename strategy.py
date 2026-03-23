"""
Multi-strategy portfolio: combine uncorrelated strategies for higher Sharpe.
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


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _bars_per_day(data):
    idx = next(iter(data.values())).index
    if len(idx) < 10:
        return 1
    if hasattr(idx, 'date'):
        bars_per_date = pd.Series(idx).groupby(idx.date).count()
        return max(1, int(round(bars_per_date.median())))
    return 1


def _get_ref_index(data):
    return next(iter(data.values())).index


def _get_tickers(data, exclude=()):
    return [t for t in data.keys() if t not in exclude]


# ---------------------------------------------------------------------------
# Strategy 1: TREND (our proven regime strategy)
# Allocation: 40% of capital
# ---------------------------------------------------------------------------

def strategy_trend(data, alloc=0.40):
    """SPY regime + top-3 momentum + adaptive trailing stop + conditional gold."""
    tickers = list(data.keys())
    stock_tickers = _get_tickers(data, exclude=('SPY', 'GLD', 'TLT', '^VIX'))
    ref_index = _get_ref_index(data)
    positions = pd.DataFrame(0.0, index=ref_index, columns=tickers)

    bpd = _bars_per_day(data)
    fast_bars = max(2, int(7 * bpd))
    slow_bars = max(5, int(19 * bpd))
    mom_bars = max(5, int(15 * bpd))

    spy_close = data['SPY']['Close'].reindex(ref_index, method='ffill')
    spy_fast = spy_close.rolling(fast_bars).mean()
    spy_slow = spy_close.rolling(slow_bars).mean()
    spy_rsi = ta.momentum.RSIIndicator(spy_close, window=14).rsi()

    close_matrix = pd.DataFrame(
        {t: data[t]['Close'].reindex(ref_index, method='ffill') for t in stock_tickers}
    )
    momentum = close_matrix.pct_change(mom_bars)

    gld_close = data['GLD']['Close'].reindex(ref_index, method='ffill') if 'GLD' in data else None
    gld_sma = gld_close.rolling(slow_bars).mean() if gld_close is not None else None

    tlt_close = data['TLT']['Close'].reindex(ref_index, method='ffill') if 'TLT' in data else None
    tlt_sma = tlt_close.rolling(slow_bars).mean() if tlt_close is not None else None


    state = 0
    spy_peak = 0.0
    spy_entry = 0.0
    selected = []
    weight = 0.0
    exposure_mult = 1.0
    cooldown = 0  # bars remaining before re-entry allowed after trailing stop
    COOLDOWN_BARS = max(3, int(10 * bpd))
    TOP_N = 3

    for i in range(len(ref_index)):
        f = spy_fast.iloc[i]
        s = spy_slow.iloc[i]
        price = spy_close.iloc[i]
        if pd.isna(f) or pd.isna(s):
            continue

        prev_state = state

        if cooldown > 0:
            cooldown -= 1

        if state == 1:
            spy_peak = max(spy_peak, price)
            gain = (spy_peak / spy_entry - 1) if spy_entry > 0 else 0
            stop_pct = 0.03 if gain > 0.03 else 0.05
            if price < spy_peak * (1 - stop_pct):
                state = 2
                spy_peak = 0.0
                cooldown = COOLDOWN_BARS  # start cooldown after trailing stop
            elif f < s * 0.985:
                state = -1
                spy_peak = 0.0
        elif state == -1:
            if f > s * 1.015:
                state = 1
                spy_peak = price
                spy_entry = price
        elif state == 2:
            if cooldown == 0 and f > s * 1.02:
                state = 1
                spy_peak = price
                spy_entry = price
            elif f < s * 0.985:
                state = -1
        elif state == 0:
            if f > s * 1.015:
                state = 1
                spy_peak = price
                spy_entry = price
            elif f < s * 0.985:
                state = -1

        if state == 1:
            if prev_state != 1:
                mom = momentum.iloc[i]
                valid = mom.dropna()
                if len(valid) >= TOP_N:
                    selected = valid.nlargest(TOP_N).index.tolist()
                else:
                    selected = stock_tickers[:TOP_N]
                weight = alloc / len(selected)
                exposure_mult = 1.0

            rsi_val = spy_rsi.iloc[i] if not pd.isna(spy_rsi.iloc[i]) else 50
            if rsi_val > 70:
                exposure_mult = 0.5
            elif 40 < rsi_val < 60:
                exposure_mult = 1.0

            for t in selected:
                positions.loc[ref_index[i], t] = weight * exposure_mult

        elif state in (-1, 2):
            if gld_close is not None and gld_sma is not None:
                g = gld_close.iloc[i]
                gs = gld_sma.iloc[i]
                if not pd.isna(gs) and g > gs:
                    positions.loc[ref_index[i], 'GLD'] = alloc * 0.5

            # TLT defensive rotation: allocate to treasuries during bear when TLT trending up
            if tlt_close is not None and tlt_sma is not None:
                t_price = tlt_close.iloc[i]
                t_sma = tlt_sma.iloc[i]
                if not pd.isna(t_sma) and t_price > t_sma:
                    positions.loc[ref_index[i], 'TLT'] = alloc * 0.3

    return positions


# ---------------------------------------------------------------------------
# Strategy 2: MEAN REVERSION (Bollinger Band bounce)
# Allocation: 25% of capital
# Uncorrelated with trend: profits during choppy/range-bound markets
# ---------------------------------------------------------------------------

def strategy_mean_reversion(data, alloc=0.25):
    """
    Buy individual stocks at Bollinger Band extremes (oversold).
    Hold until price reverts to the mean. No shorting.
    Uses state tracking per stock to avoid continuous rebalancing.
    """
    tickers = list(data.keys())
    stock_tickers = _get_tickers(data, exclude=('SPY', 'GLD', 'TLT', '^VIX'))
    ref_index = _get_ref_index(data)
    positions = pd.DataFrame(0.0, index=ref_index, columns=tickers)

    bpd = _bars_per_day(data)
    bb_window = max(10, int(20 * bpd))
    max_positions = 3  # max 3 simultaneous mean reversion trades
    per_stock_weight = alloc / max_positions

    # Per-stock state: 0=flat, positive=bars held since entry
    stock_bars_held = {t: 0 for t in stock_tickers}
    max_hold = max(5, int(15 * bpd))  # time stop: max 15 days in a trade

    for t in stock_tickers:
        close = data[t]['Close'].reindex(ref_index, method='ffill')
        bb = ta.volatility.BollingerBands(close, window=min(bb_window, 20), window_dev=2)
        bb_lower = bb.bollinger_lband()
        bb_mid = bb.bollinger_mavg()
        bb_upper = bb.bollinger_hband()

        for i in range(len(ref_index)):
            price_val = close.iloc[i]
            lower = bb_lower.iloc[i]
            mid = bb_mid.iloc[i]
            upper = bb_upper.iloc[i]

            if pd.isna(lower) or pd.isna(mid):
                continue

            if stock_bars_held[t] == 0:
                # Entry: price touches lower band
                if price_val <= lower:
                    active = sum(1 for v in stock_bars_held.values() if v > 0)
                    if active < max_positions:
                        stock_bars_held[t] = 1
            elif stock_bars_held[t] > 0:
                stock_bars_held[t] += 1
                # Exit: price reaches upper band (let winners run) OR time stop
                if price_val >= upper or stock_bars_held[t] > max_hold:
                    stock_bars_held[t] = 0

            if stock_bars_held[t] > 0:
                positions.loc[ref_index[i], t] = per_stock_weight

    return positions


# ---------------------------------------------------------------------------
# Strategy 3: CROSS-SECTIONAL RELATIVE VALUE (market neutral)
# Allocation: 20% of capital
# Long recent losers, short recent winners (short-term reversal)
# Rebalance weekly to minimize turnover
# ---------------------------------------------------------------------------

def strategy_relative_value(data, alloc=0.20):
    """
    Short-term reversal: long recent losers, short recent winners.
    Market-neutral (equal long/short). Rebalance every 5 days.
    """
    tickers = list(data.keys())
    stock_tickers = _get_tickers(data, exclude=('SPY', 'GLD', 'TLT', '^VIX'))
    ref_index = _get_ref_index(data)
    positions = pd.DataFrame(0.0, index=ref_index, columns=tickers)

    bpd = _bars_per_day(data)
    lookback = max(3, int(5 * bpd))  # 5-day return for reversal signal
    rebal_period = max(1, int(5 * bpd))  # rebalance every 5 days
    n_long = 3
    n_short = 3
    long_weight = (alloc * 0.5) / n_long
    short_weight = (alloc * 0.5) / n_short

    close_matrix = pd.DataFrame(
        {t: data[t]['Close'].reindex(ref_index, method='ffill') for t in stock_tickers}
    )
    returns = close_matrix.pct_change(lookback)

    current_longs = []
    current_shorts = []

    for i in range(lookback, len(ref_index)):
        if i % rebal_period == 0:
            ret = returns.iloc[i]
            valid = ret.dropna()
            if len(valid) >= n_long + n_short:
                # Reversal: long the losers, short the winners
                current_longs = valid.nsmallest(n_long).index.tolist()
                current_shorts = valid.nlargest(n_short).index.tolist()

        for t in current_longs:
            positions.loc[ref_index[i], t] += long_weight
        for t in current_shorts:
            positions.loc[ref_index[i], t] -= short_weight

    return positions


# ---------------------------------------------------------------------------
# Strategy 4: GOLD MOMENTUM (independent asset class)
# Allocation: 15% of capital
# Simple trend following on gold — uncorrelated with equity strategies
# ---------------------------------------------------------------------------

def strategy_gold_momentum(data, alloc=0.15):
    """
    Simple gold trend following.
    Long GLD when gold is above its 15-day SMA. Cash otherwise.
    """
    tickers = list(data.keys())
    ref_index = _get_ref_index(data)
    positions = pd.DataFrame(0.0, index=ref_index, columns=tickers)

    if 'GLD' not in data:
        return positions

    bpd = _bars_per_day(data)
    sma_bars = max(5, int(15 * bpd))

    gld_close = data['GLD']['Close'].reindex(ref_index, method='ffill')
    gld_sma = gld_close.rolling(sma_bars).mean()

    for i in range(len(ref_index)):
        price = gld_close.iloc[i]
        sma = gld_sma.iloc[i]
        if pd.isna(sma):
            continue
        if price > sma * 1.005:  # small hysteresis
            positions.loc[ref_index[i], 'GLD'] = alloc

    return positions


# ---------------------------------------------------------------------------
# Combined: merge all strategies
# ---------------------------------------------------------------------------

def generate_signals(data):
    """
    Multi-strategy portfolio: dynamic vol-regime allocation + vol targeting overlay.
    - Low vol: tilt heavily toward trend (momentum thrives in calm markets)
    - High vol: reduce both, slight MR tilt (mean reversion in choppy markets)
    - Very high vol: near-zero allocation (crash protection)
    """
    # Generate both strategies at full allocation for flexible weighting
    pos_trend = strategy_trend(data, alloc=1.0)
    pos_mr = strategy_mean_reversion(data, alloc=1.0)

    # Compute vol regime
    ref_index = _get_ref_index(data)
    bpd = _bars_per_day(data)
    spy_close = data['SPY']['Close'].reindex(ref_index, method='ffill')
    vol_lookback = max(5, int(20 * bpd))
    realized_vol = spy_close.pct_change().rolling(vol_lookback).std()
    baseline_vol = realized_vol.rolling(
        max(20, int(90 * bpd)), min_periods=vol_lookback
    ).median()
    vol_ratio = (realized_vol / baseline_vol).fillna(1.0)

    # Dynamic allocation: shift between trend and MR based on vol regime
    trend_w = pd.Series(0.60, index=ref_index)
    mr_w = pd.Series(0.40, index=ref_index)
    # Low vol: concentrate in trend (momentum works best)
    trend_w[vol_ratio < 0.8] = 0.95
    mr_w[vol_ratio < 0.8] = 0.05
    # High vol: balanced but reduced total
    trend_w[vol_ratio > 1.5] = 0.40
    mr_w[vol_ratio > 1.5] = 0.40
    # Very high vol: near-zero (crash protection)
    trend_w[vol_ratio > 2.5] = 0.05
    mr_w[vol_ratio > 2.5] = 0.05

    positions = pos_trend.mul(trend_w, axis=0).add(
        pos_mr.mul(mr_w, axis=0), fill_value=0
    )

    # Vol targeting overlay: additional scale-down during elevated vol
    vol_scale = pd.Series(1.0, index=ref_index)
    vol_scale[vol_ratio > 1.5] = 0.85
    vol_scale[vol_ratio > 2.5] = 0.50
    positions = positions.mul(vol_scale, axis=0)

    return positions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    start_time = time.time()

    if "--walk-forward" in sys.argv:
        print("Running walk-forward analysis (10y daily data)...")
        print("=" * 60)

        # Run combined
        wf = walk_forward(generate_signals)

        # Also run individual strategies for comparison
        wf_trend = walk_forward(lambda d: strategy_trend(d, 1.0))
        wf_mr = walk_forward(lambda d: strategy_mean_reversion(d, 1.0))

        elapsed = time.time() - start_time

        print(f"{'Strategy':<20} {'Avg Sharpe':>10} {'Median':>8} {'%Pos':>6} {'AvgRet':>8}")
        print("-" * 55)
        for name, w in [("TREND", wf_trend), ("MEAN REVERT", wf_mr),
                         ("** COMBINED **", wf)]:
            print(f"{name:<20} {w['avg_sharpe']:>10.4f} {w['median_sharpe']:>8.4f} "
                  f"{w['pct_positive_sharpe']:>5.0%} {w['avg_return']:>8.4f}")

        print()
        print(f"COMBINED walk-forward details:")
        print(f"  Avg Sharpe:         {wf['avg_sharpe']:.4f}")
        print(f"  Median Sharpe:      {wf['median_sharpe']:.4f}")
        print(f"  Std Sharpe:         {wf['std_sharpe']:.4f}")
        print(f"  Min/Max Sharpe:     {wf['min_sharpe']:.4f} / {wf['max_sharpe']:.4f}")
        print(f"  % Positive:         {wf['pct_positive_sharpe']:.1%}")
        print(f"  Avg Return:         {wf['avg_return']:.4f}")
        print(f"  Worst Drawdown:     {wf['worst_drawdown']:.4f}")
        print(f"  Total seconds:      {elapsed:.1f}")
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
