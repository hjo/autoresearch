# autotrader

Dual-strategy equity portfolio (Trend + Mean Reversion) with dynamic volatility-regime allocation and automated IBKR execution.

## Performance (walk-forward, 10 years, 13 six-month windows)

| Metric | Value |
|--------|-------|
| Avg Sharpe | 2.30 |
| Median Sharpe | 2.50 |
| % Positive Windows | 100% |
| Avg 6-Month Return | 15.4% |
| Worst Drawdown | -8.8% |

Robustness: parameters stable within +-20% perturbation, edge survives 3x commissions, bootstrap 90% CI [1.66, 2.91].

## How it works

**Two strategies, dynamically weighted by volatility regime:**

- **Trend** — SPY regime detection (SMA cross), top-3 momentum stock selection, adaptive trailing stop, GLD/TLT defensive rotation during bear markets
- **Mean Reversion** — Bollinger Band bounce on individual stocks, hold until upper band or time stop

**Portfolio construction:**

| Vol Regime | Trend Weight | MR Weight | Vol Scale |
|------------|-------------|-----------|-----------|
| Low (<0.8x median) | 95% | 5% | 100% |
| Normal | 60% | 40% | 100% |
| High (>1.5x) | 40% | 40% | 85% |
| Crisis (>2.5x) | 5% | 5% | 50% |

## Universe

12 stocks (AAPL, MSFT, GOOGL, NVDA, UNH, JNJ, JPM, V, XOM, PG, COST, CAT) + safe havens (TLT, GLD) + SPY as regime indicator.

## Quick start

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Download and cache market data
uv run prepare.py

# Run walk-forward analysis (10y daily, ~1 second)
uv run strategy.py --walk-forward

# Quick train/val on hourly data
uv run strategy.py
```

## Live execution

Requires IBKR account with IB Gateway running.

```bash
# Dry run — shows target portfolio, no connection needed
uv run execute.py

# Paper trading (port 4002)
uv run execute.py --paper

# Live trading (port 4001, asks confirmation)
uv run execute.py --live

# Live trading, automated (no interactive prompt, for cron/scheduler)
uv run execute.py --live --confirm-live --capital 10000
```

Market hours are enforced by default (9:30 AM - 4:00 PM ET, weekdays). Use `--force` to override.

All trades are logged to `trades.log`.

## Automated daily execution

Run once daily near market close (3:45 PM ET). Example cron entry:

```cron
45 15 * * 1-5 cd /path/to/autoresearch && uv run execute.py --paper --capital 10000 >> cron.log 2>&1
```

For fully automated live trading:

```cron
45 15 * * 1-5 cd /path/to/autoresearch && uv run execute.py --live --confirm-live --capital 10000 >> cron.log 2>&1
```

**Prerequisites for automation:**
- IB Gateway running and logged in (use [IBC](https://github.com/IbcAlpha/IBC) for auto-login)
- Market data: delayed data (free) works fine, or subscribe to live data ($4.50/mo per exchange)

## Project structure

```
prepare.py      — data download, caching, backtest engine, walk-forward framework
strategy.py     — trend + mean reversion strategies, dynamic allocation, signal generation
execute.py      — IBKR execution engine (dry run / paper / live / automated)
trades.log      — execution log (appended each run)
pyproject.toml  — dependencies
```

## Minimum capital

$5,000+ recommended. At $1,000, IBKR minimum commissions ($1/trade) eat ~15% of capital annually. At $10,000+, commission drag drops below 2%.

## License

MIT
