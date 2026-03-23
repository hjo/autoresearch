# autotrader

This is an experiment to have an LLM autonomously research trading strategies.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar23`). The branch `autotrader/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autotrader/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, backtesting engine, evaluation. Do not modify.
   - `strategy.py` — the file you modify. Signal generation, indicators, position sizing.
4. **Verify data exists**: Check that `~/.cache/autotrader/` contains parquet files. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Context

**What you're optimizing**: Intraday trading strategies on equities.

**Universe**: AAPL, MSFT, GOOGL, AMZN, NVDA, META, SPY (1-hour bars, ~2 years history).

**Data split**: 70% training / 30% validation. The agent optimizes val_sharpe (validation Sharpe ratio). This prevents overfitting to historical data.

**Available tools** (already in pyproject.toml — do NOT install new packages):
- `pandas` — data manipulation
- `numpy` — numerical computing
- `ta` — technical analysis indicators (RSI, MACD, Bollinger Bands, ATR, Stochastic, ADX, OBV, and many more). Import as `import ta`.
- `matplotlib` — for any analysis plots

**The `ta` library** provides a rich set of indicators. Examples:
```python
import ta
# Trend indicators
ta.trend.SMAIndicator(close, window=20).sma_indicator()
ta.trend.EMAIndicator(close, window=20).ema_indicator()
ta.trend.MACD(close).macd()
ta.trend.ADXIndicator(high, low, close).adx()
# Momentum indicators
ta.momentum.RSIIndicator(close, window=14).rsi()
ta.momentum.StochasticOscillator(high, low, close).stoch()
# Volatility indicators
ta.volatility.BollingerBands(close).bollinger_hband()
ta.volatility.AverageTrueRange(high, low, close).average_true_range()
# Volume indicators
ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
ta.volume.VolumeWeightedAveragePrice(high, low, close, volume).volume_weighted_average_price()
```

## Experimentation

Each experiment is a backtest run. The strategy script runs against ~2 years of hourly data and completes in seconds. You launch it as: `uv run strategy.py`.

**What you CAN do:**
- Modify `strategy.py` — this is the only file you edit. Everything is fair game: indicators, signal logic, position sizing, risk management, regime detection, multi-factor models, mean reversion, momentum, statistical arbitrage, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, backtesting engine, data loading, and constants.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the backtesting/evaluation logic. The `backtest()` and `evaluate_sharpe()` functions in `prepare.py` are the ground truth.

**The goal: get the highest val_sharpe (validation Sharpe ratio).**

Higher Sharpe = better risk-adjusted returns. A Sharpe > 1.0 is good, > 2.0 is excellent.

Also pay attention to:
- **val_max_drawdown**: Should not be catastrophic (worse than -0.30)
- **val_num_trades**: Too few trades = not meaningful, too many = commission drag
- **train vs val gap**: If train_sharpe >> val_sharpe, you're overfitting

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the Sharpe improvement.

**The first run**: Your very first run should always be to establish the baseline with the default strategy, so you will run the script as is.

## Output format

The script prints a summary like this:

```
---
val_sharpe:       0.850000
val_return:       0.0420
val_max_drawdown: -0.0850
val_win_rate:     0.5200
val_num_trades:   342
val_profit_factor:1.35
val_final_equity: 104200.00
train_sharpe:     0.920000
train_return:     0.0680
total_seconds:    2.1
```

Extract the key metric:
```
grep "^val_sharpe:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 6 columns:

```
commit	val_sharpe	val_return	max_dd	status	description
```

1. git commit hash (short, 7 chars)
2. val_sharpe achieved (e.g. 0.850000) — use 0.000000 for crashes
3. val_return (e.g. 0.0420) — use 0.0000 for crashes
4. max drawdown (e.g. -0.0850) — use 0.0000 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	val_sharpe	val_return	max_dd	status	description
a1b2c3d	0.850000	0.0420	-0.0850	keep	baseline SMA crossover
b2c3d4e	1.120000	0.0680	-0.0620	keep	add RSI filter
c3d4e5f	0.780000	0.0310	-0.1200	discard	MACD only (worse Sharpe)
d4e5f6g	0.000000	0.0000	0.0000	crash	division by zero in signal
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autotrader/mar23`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Modify `strategy.py` with an experimental idea by directly editing the code.
3. git commit
4. Run the experiment: `uv run strategy.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_sharpe:\|^val_return:\|^val_max_drawdown:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix. If you can't fix after a few attempts, give up on this idea.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_sharpe improved (higher), you "advance" the branch, keeping the git commit
9. If val_sharpe is equal or worse, you git reset back to where you started

**Strategy ideas to explore** (in roughly increasing sophistication):
- Different indicator combinations (RSI, MACD, Bollinger Bands, ADX, ATR, OBV)
- Multiple timeframe analysis (compute indicators at different lookback windows)
- Regime detection (trending vs mean-reverting markets using ADX or volatility)
- Adaptive position sizing based on conviction/volatility
- Risk management: per-trade stops, max portfolio exposure limits
- Mean reversion strategies (Bollinger Band bounce, RSI oversold/overbought)
- Momentum strategies (breakout, trend following with trailing stops)
- Cross-asset signals (use SPY as market regime indicator for individual stocks)
- Volume-based signals (volume spikes, OBV divergence)
- Ensemble/voting systems (combine multiple weak signals)
- Statistical features (z-scores, rolling correlations, volatility ratios)

**Timeout**: Each backtest should complete in under 2 minutes. If a run exceeds that, kill it and treat as failure.

**Crashes**: If a run crashes, use judgment: typos and easy fixes → fix and re-run. Fundamentally broken idea → skip it, log "crash", move on.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human if you should continue. The human might be asleep or away. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical approaches, re-read the ta library docs for new indicators. The loop runs until the human interrupts you.
