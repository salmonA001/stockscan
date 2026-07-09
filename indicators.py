"""
indicators.py — technical signal calculations shared by the CLI screener.

Every function takes/returns plain pandas Series so they're easy to unit test
and easy to port (the web app reimplements the same formulas in JavaScript so
both versions agree on what "bullish" means).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return upper, mid, lower, pct_b.fillna(0.5)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def realized_volatility(close: pd.Series, window: int = 20, trading_days: int = 252) -> pd.Series:
    """Annualized volatility of daily log returns."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window=window, min_periods=window).std() * np.sqrt(trading_days)


def volume_trend(volume: pd.Series, fast: int = 5, slow: int = 20) -> pd.Series:
    """Ratio of short vs long volume averages. >1 means volume is picking up."""
    return sma(volume, fast) / sma(volume, slow).replace(0, np.nan)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given an OHLCV frame (columns: open, high, low, close, volume), compute the
    full technical feature set used both for display and for the ML model.
    """
    out = pd.DataFrame(index=df.index)
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    out["close"] = close
    out["return_1d"] = close.pct_change(1)
    out["return_5d"] = close.pct_change(5)
    out["return_20d"] = close.pct_change(20)
    out["sma_20"] = sma(close, 20)
    out["sma_50"] = sma(close, 50)
    out["ema_12"] = ema(close, 12)
    out["ema_26"] = ema(close, 26)
    out["price_vs_sma20"] = close / out["sma_20"] - 1
    out["price_vs_sma50"] = close / out["sma_50"] - 1
    out["sma20_vs_sma50"] = out["sma_20"] / out["sma_50"] - 1
    out["rsi_14"] = rsi(close, 14)
    macd_line, signal_line, hist = macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist
    upper, mid, lower, pct_b = bollinger_bands(close)
    out["bb_pct_b"] = pct_b
    out["atr_14"] = atr(high, low, close, 14)
    out["atr_pct"] = out["atr_14"] / close
    out["volatility_20d"] = realized_volatility(close, 20)
    out["volume_trend"] = volume_trend(volume)
    out["volume"] = volume
    return out
