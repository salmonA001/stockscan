#!/usr/bin/env python3
"""
AI Stock Screener — terminal edition.

Pulls daily price history, computes technical signals (momentum, moving
averages, volatility, volume trends), fits a small gradient-boosted model to
estimate short-term bullish probability, and suggests take-profit targets
based on volatility (ATR).

Quick start:
    python screener.py
    python screener.py --tickers AAPL,MSFT,NVDA,AMD --top 5
    python screener.py --similar NVDA
    python screener.py --watch --interval 60

Run `python screener.py --help` for all options.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

import pandas as pd

from data import fetch_ohlcv
from indicators import build_feature_frame
from model import build_training_set, train_model, score_universe, similar_stocks, ScanResult

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box

    RICH = True
except ImportError:  # pragma: no cover
    RICH = False

DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "AMD", "AVGO", "NFLX", "CRM", "JPM", "COST", "SHOP", "PLTR",
]


def _console():
    return Console() if RICH else None


def _fmt_pct(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x * 100:.2f}%"


def _verdict(prob: float) -> str:
    if prob >= 0.70:
        return "STRONG BULLISH"
    if prob >= 0.55:
        return "BULLISH"
    if prob >= 0.45:
        return "NEUTRAL"
    return "WEAK"


def load_universe(tickers: list[str], period: str, verbose: bool = True):
    frames, sources = {}, {}
    for t in tickers:
        result = fetch_ohlcv(t, period=period, verbose=verbose)
        feat = build_feature_frame(result.df)
        if len(feat.dropna()) < 30:
            continue
        frames[t] = feat
        sources[t] = result.source
    return frames, sources


def run_scan(args) -> list[ScanResult]:
    console = _console()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    if console:
        console.print(f"[bold cyan]Fetching data[/bold cyan] for {len(tickers)} tickers ({args.period})...")
    else:
        print(f"Fetching data for {len(tickers)} tickers ({args.period})...")

    frames, sources = load_universe(tickers, args.period, verbose=args.verbose)
    if len(frames) < 3:
        print("Not enough valid tickers with sufficient history to train a model. Try more tickers.")
        return []

    X, y = build_training_set(frames, horizon=args.horizon, target_return=args.target_return)
    model, auc = train_model(X, y)

    if console:
        auc_txt = f"{auc:.3f}" if auc is not None else "n/a (small sample)"
        console.print(f"[bold cyan]Model trained[/bold cyan] on {len(X)} historical samples · holdout AUC: {auc_txt}")
    else:
        print(f"Model trained on {len(X)} samples, holdout AUC: {auc}")

    results = score_universe(frames, sources, model, tp_multiplier=args.tp_multiplier)
    top = results[: args.top]
    print_results(top, console)

    if args.similar:
        ticker = args.similar.upper()
        sim = similar_stocks(ticker, frames, top_n=5)
        print_similar(ticker, sim, console)

    return top


def print_results(results: list[ScanResult], console) -> None:
    if not results:
        print("No results.")
        return

    if console and RICH:
        table = Table(title="AI Stock Screener — Bullish Setups", box=box.ROUNDED, header_style="bold cyan")
        table.add_column("Ticker", style="bold white")
        table.add_column("Price", justify="right")
        table.add_column("Bullish %", justify="right")
        table.add_column("Verdict")
        table.add_column("RSI14", justify="right")
        table.add_column("5D / 20D", justify="right")
        table.add_column("Volatility", justify="right")
        table.add_column("Vol Trend", justify="right")
        table.add_column("Take Profit 1", justify="right")
        table.add_column("Take Profit 2", justify="right")
        table.add_column("Stop", justify="right")
        table.add_column("Src")

        for r in results:
            verdict = _verdict(r.bullish_probability)
            verdict_style = {
                "STRONG BULLISH": "bold green",
                "BULLISH": "green",
                "NEUTRAL": "yellow",
                "WEAK": "red",
            }[verdict]
            table.add_row(
                r.ticker,
                f"${r.last_close:,.2f}",
                f"{r.bullish_probability * 100:.1f}%",
                f"[{verdict_style}]{verdict}[/{verdict_style}]",
                f"{r.rsi_14:.0f}",
                f"{_fmt_pct(r.return_5d)} / {_fmt_pct(r.return_20d)}",
                f"{r.volatility_20d * 100:.1f}%",
                f"{r.volume_trend:.2f}x",
                f"${r.take_profit_1:,.2f} ({_fmt_pct(r.take_profit_pct_1)})",
                f"${r.take_profit_2:,.2f} ({_fmt_pct(r.take_profit_pct_2)})",
                f"${r.stop_loss:,.2f}",
                r.source,
            )
        console.print(table)
        if any(r.source == "synthetic" for r in results):
            console.print(
                "[dim]Note: some tickers used synthetic demo data (live fetch unavailable in this "
                "environment) — labeled 'synthetic' in the Src column.[/dim]"
            )
    else:
        print(f"\n{'TICKER':<8}{'PRICE':>10}{'BULLISH%':>10}{'VERDICT':>16}{'RSI':>6}{'TP1':>12}{'TP2':>12}{'SRC':>10}")
        for r in results:
            print(
                f"{r.ticker:<8}{r.last_close:>10.2f}{r.bullish_probability*100:>9.1f}%"
                f"{_verdict(r.bullish_probability):>16}{r.rsi_14:>6.0f}"
                f"{r.take_profit_1:>12.2f}{r.take_profit_2:>12.2f}{r.source:>10}"
            )


def print_similar(ticker: str, sim: list[tuple[str, float]], console) -> None:
    if not sim:
        return
    if console and RICH:
        text = "\n".join(f"  {t:<8} correlation {c:+.2f}" for t, c in sim)
        console.print(Panel(text, title=f"Stocks that move like {ticker}", border_style="magenta"))
    else:
        print(f"\nStocks that move like {ticker}:")
        for t, c in sim:
            print(f"  {t:<8} correlation {c:+.2f}")


def watch_mode(args) -> None:
    console = _console()
    try:
        while True:
            if console:
                console.rule(f"[bold]Scan @ {datetime.now().strftime('%H:%M:%S')}[/bold]")
            else:
                print(f"\n=== Scan @ {datetime.now().strftime('%H:%M:%S')} ===")
            run_scan(args)
            time.sleep(max(args.interval, 15))
    except KeyboardInterrupt:
        print("\nStopped.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI Stock Screener (terminal edition)")
    p.add_argument("--tickers", default=",".join(DEFAULT_WATCHLIST), help="Comma-separated tickers to scan")
    p.add_argument("--period", default="1y", choices=["3mo", "6mo", "1y", "2y", "5y"], help="History window")
    p.add_argument("--top", type=int, default=10, help="How many top results to display")
    p.add_argument("--horizon", type=int, default=10, help="Forward-looking days used to label training data")
    p.add_argument("--target-return", type=float, default=0.03, help="Return threshold that counts as 'bullish' when training (e.g. 0.03 = 3%%)")
    p.add_argument("--tp-multiplier", type=float, default=1.0, help="Scale take-profit targets up/down")
    p.add_argument("--similar", default=None, help="Show tickers most correlated with this ticker")
    p.add_argument("--watch", action="store_true", help="Keep scanning on a loop, terminal-scanner style")
    p.add_argument("--interval", type=int, default=60, help="Seconds between scans in --watch mode")
    p.add_argument("--verbose", action="store_true", help="Print data-source diagnostics")
    return p


def main():
    args = build_arg_parser().parse_args()
    if args.watch:
        watch_mode(args)
    else:
        run_scan(args)


if __name__ == "__main__":
    main()
