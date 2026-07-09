"""
data.py — fetches daily OHLCV bars for a ticker.

Primary source: Yahoo Finance via the `yfinance` package (free, no API key).
If that fails for any reason (no network, ticker not found, rate limited,
running in a sandboxed/offline environment, etc.) we fall back to a seeded
synthetic price generator so the tool still runs end-to-end and always has
something sensible to show. Every synthetic bar is clearly labeled so it's
never confused with real market data.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FetchResult:
    df: pd.DataFrame
    source: str  # "yfinance" or "synthetic"


def _synthetic_ohlcv(ticker: str, period_days: int = 400, seed: int | None = None) -> pd.DataFrame:
    """Deterministic-per-ticker random walk that looks like a real price series."""
    rng = np.random.default_rng(seed if seed is not None else abs(hash(ticker)) % (2**32))
    start_price = 20 + (abs(hash(ticker)) % 400)
    # mild drift + regime-switching volatility so it isn't a pure random walk
    n = period_days
    drift = rng.normal(0.0003, 0.0002)
    vol_regimes = rng.choice([0.010, 0.018, 0.028], size=n, p=[0.6, 0.3, 0.1])
    daily_returns = rng.normal(drift, vol_regimes)
    # add a few momentum "trend" pockets so technical signals have something to catch
    for _ in range(3):
        start = rng.integers(0, n - 30)
        length = rng.integers(10, 30)
        daily_returns[start : start + length] += rng.normal(0.004, 0.002)

    close = start_price * np.exp(np.cumsum(daily_returns))
    high = close * (1 + np.abs(rng.normal(0.004, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0.004, 0.003, n)))
    open_ = low + (high - low) * rng.random(n)
    base_volume = rng.integers(1_000_000, 20_000_000)
    volume = np.maximum(
        1000, (base_volume * (1 + rng.normal(0, 0.35, n)) * (1 + np.abs(daily_returns) * 8)).astype(int)
    )

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates
    )
    return df


def fetch_ohlcv(ticker: str, period: str = "1y", verbose: bool = True) -> FetchResult:
    """Try real data first, then fall back to synthetic data."""
    try:
        import yfinance as yf

        raw = yf.download(
            ticker, period=period, interval="1d", progress=False, auto_adjust=True, threads=False
        )
        if raw is None or raw.empty or len(raw) < 60:
            raise ValueError("empty or too-short response")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]
        df = raw[["open", "high", "low", "close", "volume"]].dropna()
        return FetchResult(df=df, source="yfinance")
    except Exception as exc:  # noqa: BLE001 - intentionally broad, this is a UX fallback
        if verbose:
            print(
                f"  [data] live fetch for {ticker} unavailable ({exc.__class__.__name__}: {exc}); "
                f"using synthetic demo data instead",
                file=sys.stderr,
            )
        days = {"3mo": 90, "6mo": 130, "1y": 260, "2y": 520, "5y": 1300}.get(period, 260)
        return FetchResult(df=_synthetic_ohlcv(ticker, period_days=max(days, 200)), source="synthetic")
