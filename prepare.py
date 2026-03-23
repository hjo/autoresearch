"""
Data preparation and backtesting infrastructure for autotrader.
This file is READ-ONLY for the agent. Do not modify.

Usage:
    uv run prepare.py              # Download and cache all data
    uv run prepare.py --refresh    # Force re-download

Data is stored in ~/.cache/autotrader/.
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

# Diversified universe: tech + healthcare + financials + energy + consumer + industrial + bonds
TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "NVDA",
    # Healthcare
    "UNH", "JNJ",
    # Financials
    "JPM", "V",
    # Energy
    "XOM",
    # Consumer
    "PG", "COST",
    # Industrial
    "CAT",
    # Market index
    "SPY",
]
INITIAL_CAPITAL = 100_000
COMMISSION_BPS = 10          # 10 basis points per trade (0.1%)
TRAIN_RATIO = 0.7            # 70% train, 30% validation
TIME_BUDGET = 120            # max seconds for a single strategy run

# Hourly data (2 years — yfinance limit)
INTERVAL_HOURLY = "1h"
PERIOD_HOURLY = "2y"

# Daily data (10 years — for walk-forward analysis)
INTERVAL_DAILY = "1d"
PERIOD_DAILY = "10y"

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autotrader")

# Walk-forward defaults (in trading days)
WF_TRAIN_DAYS = 756          # ~3 years
WF_TEST_DAYS = 126           # ~6 months
WF_STEP_DAYS = 126           # step forward 6 months

# ---------------------------------------------------------------------------
# Data download and caching
# ---------------------------------------------------------------------------

def _download(tickers, interval, period, suffix, refresh=False):
    """Generic download helper. Saves as {ticker}_{suffix}.parquet."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    for ticker in tickers:
        filepath = os.path.join(CACHE_DIR, f"{ticker}_{suffix}.parquet")
        if os.path.exists(filepath) and not refresh:
            print(f"  {ticker} ({suffix}): cached")
            continue

        print(f"  {ticker} ({suffix}): downloading...")
        try:
            data = yf.download(
                ticker, period=period, interval=interval,
                progress=False, auto_adjust=True
            )
        except Exception as e:
            print(f"  WARNING: Failed to download {ticker}: {e}")
            continue

        if data.empty:
            print(f"  WARNING: No data for {ticker}")
            continue

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.to_parquet(filepath)
        print(f"  {ticker} ({suffix}): {len(data)} bars saved")


def download_data(refresh=False):
    """Download hourly data (2y) for quick iteration."""
    _download(TICKERS, INTERVAL_HOURLY, PERIOD_HOURLY, "1h", refresh)


def download_daily_data(refresh=False):
    """Download daily data (10y) for walk-forward analysis."""
    _download(TICKERS, INTERVAL_DAILY, PERIOD_DAILY, "1d", refresh)


def _load(suffix):
    """Load cached data for all tickers with given suffix."""
    data = {}
    for ticker in TICKERS:
        filepath = os.path.join(CACHE_DIR, f"{ticker}_{suffix}.parquet")
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"No cached {suffix} data for {ticker}. Run: uv run prepare.py"
            )
        df = pd.read_parquet(filepath)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[ticker] = df
    return data


def load_data():
    """Load cached hourly data. Returns dict of ticker -> DataFrame."""
    return _load("1h")


def load_daily_data():
    """Load cached daily data (10y). Returns dict of ticker -> DataFrame."""
    return _load("1d")


def _split_data(data, split):
    """Split data dict into train or val portion."""
    result = {}
    for ticker, df in data.items():
        n = len(df)
        split_idx = int(n * TRAIN_RATIO)
        if split == "train":
            result[ticker] = df.iloc[:split_idx].copy()
        else:
            result[ticker] = df.iloc[split_idx:].copy()
    return result


def get_train_data():
    """Get training split of hourly data (first 70%)."""
    return _split_data(load_data(), "train")


def get_val_data():
    """Get validation split of hourly data (last 30%)."""
    return _split_data(load_data(), "val")


# ---------------------------------------------------------------------------
# Backtesting engine
# ---------------------------------------------------------------------------

def backtest(positions, data, initial_capital=INITIAL_CAPITAL):
    """
    Vectorized backtest engine.

    Args:
        positions: DataFrame with tickers as columns, timestamps as index.
                   Values represent target allocation as fraction of portfolio.
        data: dict mapping ticker -> DataFrame with OHLCV columns
        initial_capital: starting capital

    Returns:
        dict with performance metrics
    """
    # Build aligned close price matrix
    prices = pd.DataFrame({
        ticker: df['Close'] for ticker, df in data.items()
    })

    # Align on common timestamps
    common_idx = positions.index.intersection(prices.index)
    if len(common_idx) < 2:
        return _empty_results()

    positions = positions.loc[common_idx].fillna(0)
    prices = prices.loc[common_idx]

    # Bar-level returns
    returns = prices.pct_change(fill_method=None).fillna(0)

    # Commission on position changes
    pos_changes = positions.diff().fillna(positions.iloc[0:1])
    commission = pos_changes.abs().sum(axis=1) * (COMMISSION_BPS / 10_000)

    # Portfolio return per bar: sum of (prev_position * return) - commission
    prev_pos = positions.shift(1).fillna(0)
    portfolio_returns = (prev_pos * returns).sum(axis=1) - commission

    # Equity curve
    equity = initial_capital * (1 + portfolio_returns).cumprod()

    # --- Metrics ---

    # Total return
    total_return = (equity.iloc[-1] / initial_capital) - 1

    # Annualized Sharpe ratio (computed on daily aggregated returns for stability)
    if hasattr(portfolio_returns.index, 'date'):
        daily_returns = portfolio_returns.groupby(portfolio_returns.index.date).sum()
    else:
        daily_returns = portfolio_returns
    daily_mean = daily_returns.mean()
    daily_std = daily_returns.std()
    sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0

    # Max drawdown
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_drawdown = drawdown.min()

    # Number of trades
    num_trades = int((pos_changes.abs() > 0.001).sum().sum())

    # Win rate
    positioned_mask = prev_pos.abs().sum(axis=1) > 0.001
    positioned_returns = portfolio_returns[positioned_mask]
    win_rate = float((positioned_returns > 0).mean()) if len(positioned_returns) > 0 else 0.0

    # Profit factor
    gross_profit = positioned_returns[positioned_returns > 0].sum()
    gross_loss = abs(positioned_returns[positioned_returns < 0].sum())
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float('inf')

    return {
        "sharpe": float(sharpe),
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "num_trades": num_trades,
        "profit_factor": float(min(profit_factor, 999.9)),
        "final_equity": float(equity.iloc[-1]),
        "equity_curve": equity,
    }


def _empty_results():
    """Return zeroed-out results for failed/empty backtests."""
    return {
        "sharpe": 0.0,
        "total_return": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "num_trades": 0,
        "profit_factor": 0.0,
        "final_equity": 0.0,
        "equity_curve": pd.Series(dtype=float),
    }


def evaluate_sharpe(positions, split="val"):
    """
    Ground truth evaluation function for quick iteration.
    Returns annualized Sharpe ratio on the specified hourly data split.
    """
    data = get_train_data() if split == "train" else get_val_data()
    results = backtest(positions, data)
    return results["sharpe"]


# ---------------------------------------------------------------------------
# Walk-forward analysis
# ---------------------------------------------------------------------------

def walk_forward(signal_fn, train_days=WF_TRAIN_DAYS, test_days=WF_TEST_DAYS,
                 step_days=WF_STEP_DAYS):
    """
    Walk-forward analysis on daily data.

    Splits 10 years of daily data into rolling train/test windows:
      [train_window] [test_window]
                 [train_window] [test_window]
                            ...

    Args:
        signal_fn: callable(data_dict) -> positions DataFrame
                   The strategy's generate_signals function.
        train_days: training window length in trading days
        test_days:  test window length in trading days
        step_days:  how far to step forward between windows

    Returns:
        dict with aggregate and per-window results
    """
    daily_data = load_daily_data()

    # Find common date range across all tickers
    all_dates = None
    for ticker, df in daily_data.items():
        dates = df.index
        if all_dates is None:
            all_dates = dates
        else:
            all_dates = all_dates.intersection(dates)
    all_dates = all_dates.sort_values()
    n_dates = len(all_dates)

    windows = []
    start = 0
    while start + train_days + test_days <= n_dates:
        train_end = start + train_days
        test_end = train_end + test_days

        # Slice data for this window
        train_dates = all_dates[start:train_end]
        test_dates = all_dates[train_end:test_end]

        train_data = {}
        test_data = {}
        for ticker, df in daily_data.items():
            train_data[ticker] = df.loc[df.index.isin(train_dates)].copy()
            test_data[ticker] = df.loc[df.index.isin(test_dates)].copy()

        # Generate signals on test data
        # (strategy should be self-contained — it computes its own indicators)
        try:
            test_positions = signal_fn(test_data)
            test_results = backtest(test_positions, test_data)
        except Exception as e:
            test_results = _empty_results()
            test_results["error"] = str(e)

        window_info = {
            "train_start": str(train_dates[0].date()),
            "train_end": str(train_dates[-1].date()),
            "test_start": str(test_dates[0].date()),
            "test_end": str(test_dates[-1].date()),
            **{k: v for k, v in test_results.items() if k != "equity_curve"},
        }
        windows.append(window_info)

        start += step_days

    if not windows:
        return {"error": "Not enough data for walk-forward analysis"}

    # Aggregate statistics
    sharpes = [w["sharpe"] for w in windows]
    returns = [w["total_return"] for w in windows]
    drawdowns = [w["max_drawdown"] for w in windows]
    win_rates = [w["win_rate"] for w in windows]

    return {
        "n_windows": len(windows),
        "avg_sharpe": float(np.mean(sharpes)),
        "median_sharpe": float(np.median(sharpes)),
        "std_sharpe": float(np.std(sharpes)),
        "min_sharpe": float(np.min(sharpes)),
        "max_sharpe": float(np.max(sharpes)),
        "pct_positive_sharpe": float(np.mean([s > 0 for s in sharpes])),
        "avg_return": float(np.mean(returns)),
        "avg_max_drawdown": float(np.mean(drawdowns)),
        "worst_drawdown": float(np.min(drawdowns)),
        "avg_win_rate": float(np.mean(win_rates)),
        "windows": windows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare trading data")
    parser.add_argument("--refresh", action="store_true", help="Force re-download")
    args = parser.parse_args()

    print("Downloading hourly data (2y)...")
    download_data(refresh=args.refresh)

    print("\nDownloading daily data (10y)...")
    download_daily_data(refresh=args.refresh)

    # Verify hourly
    print("\n--- Hourly Data ---")
    data = load_data()
    for ticker, df in data.items():
        print(f"  {ticker}: {len(df)} bars, {df.index[0]} to {df.index[-1]}")

    train = get_train_data()
    val = get_val_data()
    sample = TICKERS[0]
    t, v = train[sample], val[sample]
    print(f"\nTrain/Val split ({sample}):")
    print(f"  Train: {len(t)} bars ({t.index[0]} to {t.index[-1]})")
    print(f"  Val:   {len(v)} bars ({v.index[0]} to {v.index[-1]})")

    # Verify daily
    print("\n--- Daily Data (10y) ---")
    daily = load_daily_data()
    for ticker, df in daily.items():
        print(f"  {ticker}: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")

    # Walk-forward window count
    sample_n = len(daily[TICKERS[0]])
    n_windows = (sample_n - WF_TRAIN_DAYS - WF_TEST_DAYS) // WF_STEP_DAYS + 1
    print(f"\nWalk-forward: {n_windows} windows "
          f"(train={WF_TRAIN_DAYS}d, test={WF_TEST_DAYS}d, step={WF_STEP_DAYS}d)")

    print(f"\nTickers: {', '.join(TICKERS)}")
    print(f"Initial capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"Commission: {COMMISSION_BPS} bps")
    print("\nData ready!")
