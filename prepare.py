"""
Data preparation and backtesting infrastructure for autotrader.
This file is READ-ONLY for the agent. Do not modify.

Usage:
    uv run prepare.py              # Download and cache data
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

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "SPY"]
INTERVAL = "1h"
PERIOD = "2y"
INITIAL_CAPITAL = 100_000
COMMISSION_BPS = 10          # 10 basis points per trade (0.1%)
TRAIN_RATIO = 0.7            # 70% train, 30% validation
TIME_BUDGET = 120            # max seconds for a single strategy run

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autotrader")

# Annualization factor for Sharpe ratio (hourly bars)
# ~6.5 trading hours/day, 252 trading days/year
HOURS_PER_YEAR = 252 * 6.5
ANNUAL_FACTOR = np.sqrt(HOURS_PER_YEAR)

# ---------------------------------------------------------------------------
# Data download and caching
# ---------------------------------------------------------------------------

def download_data(refresh=False):
    """Download historical data from yfinance and cache as parquet."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    for ticker in TICKERS:
        filepath = os.path.join(CACHE_DIR, f"{ticker}.parquet")
        if os.path.exists(filepath) and not refresh:
            print(f"  {ticker}: cached")
            continue

        print(f"  {ticker}: downloading...")
        try:
            data = yf.download(
                ticker, period=PERIOD, interval=INTERVAL,
                progress=False, auto_adjust=True
            )
        except Exception as e:
            print(f"  WARNING: Failed to download {ticker}: {e}")
            continue

        if data.empty:
            print(f"  WARNING: No data for {ticker}")
            continue

        # Flatten multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.to_parquet(filepath)
        print(f"  {ticker}: {len(data)} bars saved")


def load_data():
    """Load cached data for all tickers. Returns dict of ticker -> DataFrame."""
    data = {}
    for ticker in TICKERS:
        filepath = os.path.join(CACHE_DIR, f"{ticker}.parquet")
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"No cached data for {ticker}. Run: uv run prepare.py"
            )
        df = pd.read_parquet(filepath)
        # Ensure clean column names
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[ticker] = df
    return data


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
    """Get training split of data (first 70%)."""
    return _split_data(load_data(), "train")


def get_val_data():
    """Get validation split of data (last 30%)."""
    return _split_data(load_data(), "val")


# ---------------------------------------------------------------------------
# Backtesting engine
# ---------------------------------------------------------------------------

def backtest(positions, data, initial_capital=INITIAL_CAPITAL):
    """
    Vectorized backtest engine.

    Args:
        positions: DataFrame with tickers as columns, timestamps as index.
                   Values represent target allocation as fraction of portfolio:
                   - 1.0 = fully long this ticker
                   - 0.0 = flat (no position)
                   - -1.0 = fully short
                   Fractional values for partial positions.
                   Sum of absolute allocations should ideally be <= 1.0.

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

    # Number of trades (position changes > threshold)
    num_trades = int((pos_changes.abs() > 0.001).sum().sum())

    # Win rate on bars where we had a position
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
    Ground truth evaluation function.
    Returns annualized Sharpe ratio on the specified data split.

    This is the metric the agent optimizes. Higher is better.
    """
    data = get_train_data() if split == "train" else get_val_data()
    results = backtest(positions, data)
    return results["sharpe"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare trading data")
    parser.add_argument("--refresh", action="store_true", help="Force re-download")
    args = parser.parse_args()

    print("Downloading data...")
    download_data(refresh=args.refresh)

    # Verify
    print("\nData summary:")
    data = load_data()
    for ticker, df in data.items():
        print(f"  {ticker}: {len(df)} bars, {df.index[0]} to {df.index[-1]}")

    # Show train/val split
    train = get_train_data()
    val = get_val_data()
    sample = TICKERS[0]
    t, v = train[sample], val[sample]
    print(f"\nTrain/Val split ({sample}):")
    print(f"  Train: {len(t)} bars ({t.index[0]} to {t.index[-1]})")
    print(f"  Val:   {len(v)} bars ({v.index[0]} to {v.index[-1]})")
    print(f"\nTickers: {', '.join(TICKERS)}")
    print(f"Interval: {INTERVAL}")
    print(f"Initial capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"Commission: {COMMISSION_BPS} bps")
    print("\nData ready!")
