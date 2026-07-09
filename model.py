"""
model.py — a deliberately *small* machine learning model.

We're not trying to predict the market with a giant model; we're scoring
short-term "bullish setup quality" from a handful of well-understood
technical features, the same way a discretionary trader would eyeball a
chart, but calibrated on historical outcomes.

Pipeline:
1. For every ticker in the universe, build the technical feature frame.
2. Label each historical row: did price rise by more than `target_return`
   within the next `horizon` trading days? (1 = yes / bullish outcome)
3. Pool rows from all tickers, train a small GradientBoostingClassifier.
4. Score the most recent bar per ticker -> "bullish probability".
5. Suggest take-profit targets from ATR (volatility) scaled by confidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from indicators import build_feature_frame

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "price_vs_sma20",
    "price_vs_sma50",
    "sma20_vs_sma50",
    "rsi_14",
    "macd_hist",
    "bb_pct_b",
    "atr_pct",
    "volatility_20d",
    "volume_trend",
]


@dataclass
class ScanResult:
    ticker: str
    source: str
    last_close: float
    bullish_probability: float
    rsi_14: float
    volatility_20d: float
    volume_trend: float
    return_5d: float
    return_20d: float
    atr_14: float
    take_profit_1: float
    take_profit_2: float
    take_profit_pct_1: float
    take_profit_pct_2: float
    stop_loss: float
    feature_frame: pd.DataFrame


def _label_forward_returns(df: pd.DataFrame, horizon: int, target_return: float) -> pd.Series:
    fwd_return = df["close"].shift(-horizon) / df["close"] - 1
    return (fwd_return > target_return).astype(int)


def build_training_set(
    frames: dict[str, pd.DataFrame], horizon: int = 10, target_return: float = 0.03
) -> tuple[pd.DataFrame, pd.Series]:
    rows, labels = [], []
    for ticker, feat in frames.items():
        y = _label_forward_returns(feat, horizon, target_return)
        valid = feat[FEATURE_COLUMNS].notna().all(axis=1) & y.notna()
        valid.iloc[-horizon:] = False  # can't know the label for the tail yet
        rows.append(feat.loc[valid, FEATURE_COLUMNS])
        labels.append(y.loc[valid])
    X = pd.concat(rows, axis=0)
    y = pd.concat(labels, axis=0)
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series, seed: int = 42):
    """Fit a small gradient-boosted classifier. Returns (model, holdout_auc)."""
    model = GradientBoostingClassifier(
        n_estimators=60, max_depth=2, learning_rate=0.08, subsample=0.8, random_state=seed
    )
    auc = None
    if y.nunique() > 1 and len(y) > 40:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y
        )
        model.fit(X_train, y_train)
        try:
            preds = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, preds)
        except Exception:  # noqa: BLE001
            auc = None
        # refit on everything for the final scoring model
        model.fit(X, y)
    else:
        model.fit(X, y)
    return model, auc


def score_universe(
    frames: dict[str, pd.DataFrame],
    sources: dict[str, str],
    model,
    tp_multiplier: float = 1.0,
) -> list[ScanResult]:
    results = []
    for ticker, feat in frames.items():
        latest = feat.iloc[-1]
        if latest[FEATURE_COLUMNS].isna().any():
            continue
        x = latest[FEATURE_COLUMNS].to_frame().T
        prob = float(model.predict_proba(x)[:, 1][0])

        close = float(latest["close"])
        atr_val = float(latest["atr_14"])
        # confidence scales the target a little: more conviction -> reach a bit further
        confidence_scale = 0.75 + prob * 0.75
        tp1 = close + atr_val * 1.5 * tp_multiplier * confidence_scale
        tp2 = close + atr_val * 3.0 * tp_multiplier * confidence_scale
        stop = close - atr_val * 1.2

        results.append(
            ScanResult(
                ticker=ticker,
                source=sources.get(ticker, "unknown"),
                last_close=close,
                bullish_probability=prob,
                rsi_14=float(latest["rsi_14"]),
                volatility_20d=float(latest["volatility_20d"]),
                volume_trend=float(latest["volume_trend"]) if not np.isnan(latest["volume_trend"]) else 1.0,
                return_5d=float(latest["return_5d"]),
                return_20d=float(latest["return_20d"]),
                atr_14=atr_val,
                take_profit_1=tp1,
                take_profit_2=tp2,
                take_profit_pct_1=tp1 / close - 1,
                take_profit_pct_2=tp2 / close - 1,
                stop_loss=stop,
                feature_frame=feat,
            )
        )
    results.sort(key=lambda r: r.bullish_probability, reverse=True)
    return results


def similar_stocks(target: str, frames: dict[str, pd.DataFrame], top_n: int = 5) -> list[tuple[str, float]]:
    """Rank other tickers by correlation of daily returns with the target ticker."""
    if target not in frames:
        return []
    target_ret = frames[target]["return_1d"].dropna()
    scores = []
    for ticker, feat in frames.items():
        if ticker == target:
            continue
        other_ret = feat["return_1d"].dropna()
        joined = pd.concat([target_ret, other_ret], axis=1, join="inner").dropna()
        if len(joined) < 30:
            continue
        corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
        if corr is not None and not np.isnan(corr):
            scores.append((ticker, float(corr)))
    scores.sort(key=lambda t: t[1], reverse=True)
    return scores[:top_n]
