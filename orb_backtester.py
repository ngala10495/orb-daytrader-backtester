"""
Opening Range Breakout (ORB) Day Trading Backtester — Gold
--------------------------------------------------------------
A genuine intraday day-trading strategy: no overnight risk, every position
opened and closed within the same trading session. This is deliberately
different from the SMC/CRT/MA-Crossover/Turtle scripts in this project —
those are swing/position systems holding for hours to months. This one
never holds past the session close.

Logic:
    1. Define the opening range = high/low of the first ORB_MINUTES minutes
       after the NY session opens.
    2. Wait for a breakout: price closes beyond the opening range.
    3. Enter in the breakout direction. Stop = opposite side of the range.
    4. Target = a measured move — the opening range's own height, projected
       from the breakout point. This keeps R:R anchored to that specific
       day's actual volatility rather than an arbitrary fixed number.
    5. Force-flatten at session close if neither stop nor target was hit —
       zero overnight exposure, one trade per day maximum.

Usage:
    pip install yfinance pandas numpy --break-system-packages
    python3 orb_backtester.py

Requires internet access (yfinance) to pull real OHLC data.
NOTE: 15m data is capped at 60 days by Yahoo Finance — same limitation as
the SMC/CRT scripts, for the same reason (need intraday granularity).
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import time

# ----------------------------- CONFIG ----------------------------------

TICKER = "GC=F"
PERIOD = "60d"
INTERVAL = "15m"

# NY session opening range (UTC). 13:30 UTC = 9:30am ET (NY equities/futures
# open) — this is when Gold typically sees its heaviest day-session volume.
ORB_SESSION_START = time(13, 30)
ORB_MINUTES = 30                     # opening range = first 30 minutes

SESSION_CLOSE = time(20, 0)          # force-flatten by this time, no exceptions
MIN_RANGE_PCT = 0.0015                # skip days where the opening range is too
                                       # tight to produce a realistic stop distance


# --------------------------- CORE LOGIC ---------------------------------

def run_orb_strategy(ticker: str) -> dict:
    df = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False)
    if df.empty:
        return {"error": "no data returned"}
    df = df.droplevel(1, axis=1) if isinstance(df.columns, pd.MultiIndex) else df
    df["date"] = df.index.date
    df["t"] = df.index.time

    trades = []

    for d, day_df in df.groupby("date"):
        orb_end_minutes = ORB_SESSION_START.hour * 60 + ORB_SESSION_START.minute + ORB_MINUTES
        orb_end = time(orb_end_minutes // 60, orb_end_minutes % 60)

        orb_bars = day_df[(day_df["t"] >= ORB_SESSION_START) & (day_df["t"] < orb_end)]
        if orb_bars.empty:
            continue
        orb_high = orb_bars["High"].max()
        orb_low = orb_bars["Low"].min()
        if orb_low <= 0:
            continue
        range_pct = (orb_high - orb_low) / orb_low
        if range_pct < MIN_RANGE_PCT:
            continue   # too quiet a morning to produce a realistic stop

        range_height = orb_high - orb_low

        after_orb = day_df[(day_df["t"] >= orb_end) & (day_df["t"] <= SESSION_CLOSE)]
        if after_orb.empty:
            continue

        position = None
        already_traded_today = False
        for ts, bar in after_orb.iterrows():
            if position is None:
                if already_traded_today:
                    continue   # enforce one trade per day, as documented
                if bar["Close"] > orb_high:
                    position = {"direction": "long", "entry": bar["Close"], "entry_time": ts,
                                 "stop": orb_low, "target": bar["Close"] + range_height}
                    already_traded_today = True
                elif bar["Close"] < orb_low:
                    position = {"direction": "short", "entry": bar["Close"], "entry_time": ts,
                                 "stop": orb_high, "target": bar["Close"] - range_height}
                    already_traded_today = True
                continue

            direction = position["direction"]
            if direction == "long":
                if bar["Low"] <= position["stop"]:
                    trades.append({**position, "exit": position["stop"], "exit_time": ts,
                                    "outcome": "loss", "reason": "stop"})
                    position = None
                elif bar["High"] >= position["target"]:
                    trades.append({**position, "exit": position["target"], "exit_time": ts,
                                    "outcome": "win", "reason": "target"})
                    position = None
            else:
                if bar["High"] >= position["stop"]:
                    trades.append({**position, "exit": position["stop"], "exit_time": ts,
                                    "outcome": "loss", "reason": "stop"})
                    position = None
                elif bar["Low"] <= position["target"]:
                    trades.append({**position, "exit": position["target"], "exit_time": ts,
                                    "outcome": "win", "reason": "target"})
                    position = None

        # Force-flatten if still open at session close
        if position is not None:
            last_bar = after_orb.iloc[-1]
            exit_price = last_bar["Close"]
            pnl_direction = 1 if position["direction"] == "long" else -1
            outcome = "win" if (exit_price - position["entry"]) * pnl_direction > 0 else "loss"
            trades.append({**position, "exit": exit_price, "exit_time": after_orb.index[-1],
                            "outcome": outcome, "reason": "eod_flatten"})

    if not trades:
        return {"trades": 0, "win_rate": None, "total_R": 0, "trade_log": []}

    processed = []
    for t in trades:
        risk = abs(t["entry"] - t["stop"])
        pnl = (t["exit"] - t["entry"]) if t["direction"] == "long" else (t["entry"] - t["exit"])
        r_multiple = pnl / risk if risk > 0 else 0
        processed.append({
            "date": t["entry_time"].date(),
            "direction": t["direction"],
            "entry_time": t["entry_time"],
            "exit_time": t["exit_time"],
            "outcome": t["outcome"],
            "reason": t["reason"],
            "rr": round(r_multiple, 2),
        })

    wins = sum(1 for t in processed if t["outcome"] == "win")
    total_R = round(sum(t["rr"] for t in processed), 2)
    reason_counts = pd.Series([t["reason"] for t in processed]).value_counts().to_dict()

    return {
        "trades": len(processed),
        "wins": wins,
        "losses": len(processed) - wins,
        "win_rate": round(100 * wins / len(processed), 1),
        "total_R": total_R,
        "exit_reason_breakdown": reason_counts,
        "trade_log": processed,
    }


if __name__ == "__main__":
    print(f"Running ORB day-trading system on {TICKER} ({PERIOD} of {INTERVAL} data)...")
    result = run_orb_strategy(TICKER)

    print("\n=== ORB DAY TRADING BACKTEST RESULTS ===")
    summary = {k: v for k, v in result.items() if k != "trade_log"}
    print(summary)

    trades = result.get("trade_log", [])
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv("orb_trade_log.csv", index=False)
        print(f"\nSaved {len(trades_df)} trades to orb_trade_log.csv")
    else:
        print("\nNo trades to export.")
