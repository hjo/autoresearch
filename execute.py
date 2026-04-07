"""
Live execution engine for the autotrader strategy.

Connects to IBKR via ib_async, computes target positions from strategy.py,
and submits orders to rebalance.

Usage:
    uv run execute.py              # Dry run (print orders, don't submit)
    uv run execute.py --paper      # Execute on paper trading account
    uv run execute.py --live       # Execute on live account (asks confirmation)

Prerequisites:
    - IB Gateway or TWS running locally
    - Paper: port 4002 (Gateway) or 7497 (TWS)
    - Live:  port 4001 (Gateway) or 7496 (TWS)
"""

import argparse
import asyncio
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from strategy import generate_signals
from prepare import TICKERS, COMMISSION_BPS


# ---------------------------------------------------------------------------
# Data: fetch recent daily data for signal generation
# ---------------------------------------------------------------------------

def fetch_recent_data(tickers, lookback_days=250):
    """Fetch recent daily data from yfinance for signal computation."""
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(
                ticker, period=f"{lookback_days}d", interval="1d",
                progress=False, auto_adjust=True,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                data[ticker] = df
        except Exception as e:
            print(f"  WARNING: Failed to fetch {ticker}: {e}")
    return data


# ---------------------------------------------------------------------------
# Position computation
# ---------------------------------------------------------------------------

def compute_target_positions(capital, data):
    """
    Run the strategy on recent data and return target dollar positions.

    Returns:
        dict of ticker -> target_dollar_amount (positive=long, negative=short)
    """
    positions = generate_signals(data)

    if positions.empty:
        return {}

    # Use the last row (most recent signal)
    latest = positions.iloc[-1]

    targets = {}
    for ticker in latest.index:
        alloc = latest[ticker]
        if abs(alloc) > 0.001:
            targets[ticker] = round(alloc * capital, 2)

    return targets


def compute_orders(current_positions, target_positions, prices):
    """
    Diff current vs target positions to compute required orders.

    Args:
        current_positions: dict of ticker -> current_dollar_value
        target_positions:  dict of ticker -> target_dollar_value
        prices:            dict of ticker -> current_price

    Returns:
        list of dicts with keys: ticker, action, shares, dollar_amount, price
    """
    all_tickers = set(list(current_positions.keys()) + list(target_positions.keys()))
    orders = []

    for ticker in sorted(all_tickers):
        current = current_positions.get(ticker, 0.0)
        target = target_positions.get(ticker, 0.0)
        diff = target - current
        price = prices.get(ticker)

        if price is None or price <= 0:
            continue

        shares = diff / price

        # Skip tiny orders (less than 0.5 shares or $10)
        if abs(shares) < 0.5 and abs(diff) < 10:
            continue

        # Round to nearest share (no fractional for simplicity)
        shares = int(round(shares))
        if shares == 0:
            continue

        orders.append({
            "ticker": ticker,
            "action": "BUY" if shares > 0 else "SELL",
            "shares": abs(shares),
            "dollar_amount": round(abs(shares * price), 2),
            "price": price,
        })

    return orders


# ---------------------------------------------------------------------------
# IBKR connection and execution
# ---------------------------------------------------------------------------

def get_ibkr_positions(ib):
    """Fetch current positions from IBKR."""
    positions = {}
    for pos in ib.positions():
        ticker = pos.contract.symbol
        positions[ticker] = {
            "shares": float(pos.position),
            "avg_cost": float(pos.avgCost),
        }
    return positions


def get_ibkr_account_value(ib):
    """Fetch net liquidation value from IBKR."""
    from ib_async import util
    account = ib.managedAccounts()[0]
    ib.reqAccountUpdates(account=account)
    util.sleep(3)  # wait for data to arrive
    for av in ib.accountValues():
        if av.tag == "NetLiquidation" and av.currency == "USD":
            return float(av.value)
    return None


def get_ibkr_prices(ib, tickers, yf_data=None):
    """Fetch current prices. Try IBKR delayed data, fall back to yfinance."""
    from ib_async import Stock, util
    prices = {}

    # Request delayed market data (free, no subscription needed)
    ib.reqMarketDataType(3)  # 3 = delayed

    tradeable = [t for t in tickers if not t.startswith("^")]
    for ticker in tradeable:
        try:
            contract = Stock(ticker, "SMART", "USD")
            ib.qualifyContracts(contract)
            ticker_data = ib.reqMktData(contract, "", False, False)
            util.sleep(1)
            price = (ticker_data.last if ticker_data.last and ticker_data.last > 0
                     else ticker_data.close if ticker_data.close and ticker_data.close > 0
                     else ticker_data.marketPrice() if ticker_data.marketPrice() > 0
                     else None)
            if price and price > 0:
                prices[ticker] = price
            ib.cancelMktData(contract)
        except Exception:
            pass

    # Fall back to yfinance for any missing prices
    if yf_data:
        for ticker in tradeable:
            if ticker not in prices and ticker in yf_data and not yf_data[ticker].empty:
                prices[ticker] = float(yf_data[ticker]["Close"].iloc[-1])

    return prices


def submit_ibkr_orders(ib, orders):
    """Submit orders to IBKR. Returns list of trade objects."""
    from ib_async import Stock, MarketOrder, util
    trades = []
    for order in orders:
        contract = Stock(order["ticker"], "SMART", "USD")
        ib.qualifyContracts(contract)
        mkt_order = MarketOrder(order["action"], order["shares"])
        trade = ib.placeOrder(contract, mkt_order)
        trades.append(trade)
        print(f"  SUBMITTED: {order['action']} {order['shares']} {order['ticker']}"
              f" (~${order['dollar_amount']:.0f})")

    # Wait for fills (up to 30 seconds)
    deadline = time.time() + 30
    while time.time() < deadline:
        ib.sleep(1)
        if all(t.isDone() for t in trades):
            break

    for trade in trades:
        status = trade.orderStatus.status
        filled = trade.orderStatus.filled
        print(f"  {trade.contract.symbol}: {status} (filled={filled})")

    return trades


# ---------------------------------------------------------------------------
# Main execution modes
# ---------------------------------------------------------------------------

def run_live(port, dry_run=False, capital=10_000):
    """Full execution: connect to IBKR, compute signals, submit orders."""
    from ib_async import IB, util
    util.startLoop()  # enable sync API usage

    print(f"Connecting to IBKR on port {port}...")
    ib = IB()
    try:
        import random
        client_id = random.randint(100, 999)
        ib.connect("127.0.0.1", port, clientId=client_id, timeout=15)
    except Exception as e:
        print(f"ERROR: Could not connect to IBKR on port {port}")
        print(f"  {e}")
        print(f"  Make sure IB Gateway or TWS is running.")
        return

    try:
        accounts = ib.managedAccounts()
        print(f"Connected. Account(s): {accounts}")
        print(f"Using capital: ${capital:,.2f}")

        # Fetch current positions
        ibkr_positions = get_ibkr_positions(ib)
        print(f"\nCurrent positions ({len(ibkr_positions)}):")
        for ticker, info in sorted(ibkr_positions.items()):
            print(f"  {ticker}: {info['shares']:.0f} shares @ ${info['avg_cost']:.2f}")

        # Fetch recent market data for strategy signals (also used as price fallback)
        print("\nFetching recent daily data for strategy signals...")
        data = fetch_recent_data(TICKERS)

        # Fetch prices (IBKR delayed, fallback to yfinance)
        all_tickers = list(set(TICKERS) | set(ibkr_positions.keys()))
        print(f"Fetching prices for {len(all_tickers)} tickers...")
        prices = get_ibkr_prices(ib, all_tickers, yf_data=data)

        # Current dollar positions
        current_dollar = {}
        for ticker, info in ibkr_positions.items():
            price = prices.get(ticker)
            if price:
                current_dollar[ticker] = info["shares"] * price
        print(f"  Loaded {len(data)} tickers, "
              f"{len(next(iter(data.values())))} bars each")

        target_positions = compute_target_positions(capital, data)

        print(f"\nTarget positions:")
        for ticker, amount in sorted(target_positions.items(),
                                     key=lambda x: -abs(x[1])):
            pct = amount / capital * 100
            print(f"  {ticker}: ${amount:>8,.0f} ({pct:>5.1f}%)")

        # Compute orders
        orders = compute_orders(current_dollar, target_positions, prices)

        if not orders:
            print("\nNo orders needed — portfolio is on target.")
            return

        total_traded = sum(o["dollar_amount"] for o in orders)
        est_commission = total_traded * COMMISSION_BPS / 10_000

        print(f"\nOrders to execute ({len(orders)}):")
        print(f"{'Action':<6} {'Shares':>6} {'Ticker':<6} {'Amount':>10} {'Price':>10}")
        print("-" * 42)
        for o in orders:
            print(f"{o['action']:<6} {o['shares']:>6} {o['ticker']:<6} "
                  f"${o['dollar_amount']:>9,.2f} ${o['price']:>9.2f}")
        print(f"\nTotal turnover:  ${total_traded:>10,.2f}")
        print(f"Est. commission: ${est_commission:>10,.2f}")

        if dry_run:
            print("\n** DRY RUN — no orders submitted **")
            return

        # Confirm
        print(f"\nSubmitting {len(orders)} orders...")
        submit_ibkr_orders(ib, orders)
        print("\nDone.")

    finally:
        ib.disconnect()


def run_dry():
    """Dry run: compute signals and print target orders without IBKR."""
    capital = 10_000  # default for dry run

    print(f"=== DRY RUN (no IBKR connection) ===")
    print(f"Assumed capital: ${capital:,.0f}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    print(f"\nFetching recent daily data...")
    data = fetch_recent_data(TICKERS)
    print(f"  Loaded {len(data)} tickers, "
          f"{len(next(iter(data.values())))} bars each")

    target_positions = compute_target_positions(capital, data)

    if not target_positions:
        print("\nStrategy says: all cash (no positions)")
        return

    # Get current prices from yfinance
    prices = {}
    for ticker in TICKERS:
        if ticker in data and not data[ticker].empty:
            prices[ticker] = float(data[ticker]["Close"].iloc[-1])

    print(f"\nTarget portfolio:")
    total_alloc = 0
    print(f"{'Ticker':<8} {'Allocation':>10} {'$Amount':>10} {'Shares':>8} {'Price':>10}")
    print("-" * 50)
    for ticker, amount in sorted(target_positions.items(),
                                 key=lambda x: -abs(x[1])):
        pct = amount / capital * 100
        total_alloc += pct
        price = prices.get(ticker, 0)
        shares = int(round(amount / price)) if price > 0 else 0
        print(f"{ticker:<8} {pct:>9.1f}% ${amount:>9,.0f} {shares:>8} ${price:>9.2f}")
    print(f"{'Total':<8} {total_alloc:>9.1f}% ${sum(target_positions.values()):>9,.0f}")
    print(f"{'Cash':<8} {100-total_alloc:>9.1f}% ${capital - sum(target_positions.values()):>9,.0f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Execute autotrader strategy via IBKR"
    )
    parser.add_argument(
        "--paper", action="store_true",
        help="Execute on paper trading account (port 4002)"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Execute on live account (port 4001) — asks confirmation"
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Override IBKR port (default: 4002 paper, 4001 live)"
    )
    parser.add_argument(
        "--capital", type=float, default=10_000,
        help="Capital for dry run (default: $10,000)"
    )
    args = parser.parse_args()

    if args.live:
        port = args.port or 4001
        print("=" * 50)
        print("  WARNING: LIVE TRADING MODE")
        print(f"  Connecting to port {port}")
        print("=" * 50)
        confirm = input("Type 'yes' to proceed: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return
        run_live(port, dry_run=False, capital=args.capital)

    elif args.paper:
        port = args.port or 4002
        print(f"Paper trading mode (port {port})")
        run_live(port, dry_run=False, capital=args.capital)

    else:
        run_dry()


if __name__ == "__main__":
    main()
