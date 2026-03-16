"""Resampling utilities for OHLCV time-series."""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def normalize_timeframe(timeframe: str) -> Tuple[int, str]:
    """
    Convert a timeframe string to pandas OHLC resample frequency.
    Accepted examples: "1", "1m", "1min", "15", "15min".
    """
    if timeframe is None:
        raise ValueError("timeframe is required")
    key = str(timeframe).strip().lower().replace(" ", "")
    if key.endswith("minutes"):
        key = key[:-7]
    elif key.endswith("minute"):
        key = key[:-6]
    elif key.endswith("min"):
        key = key[:-3]
    elif key.endswith("m"):
        key = key[:-1]

    if not key.isdigit():
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    minutes = int(key)
    if minutes <= 0:
        raise ValueError(f"timeframe must be positive minutes: {timeframe}")
    return minutes, f"{minutes}min"


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV dataframe to a target timeframe."""
    minutes, rule = normalize_timeframe(timeframe)
    if df.empty:
        return df.copy()
    if not {"open", "high", "low", "close"}.issubset(df.columns):
        raise ValueError("Input df must contain open, high, low, close")

    ohlcv = df[["open", "high", "low", "close"]].resample(rule, label="right", closed="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
    )
    ohlcv = ohlcv.dropna(subset=["open", "high", "low", "close"])
    if "volume" in df.columns:
        vol = (
            df["volume"]
            .resample(rule, label="right", closed="right")
            .sum(min_count=1)
            .rename("volume")
        )
        ohlcv = pd.concat([ohlcv, vol], axis=1)
    ohlcv = ohlcv.dropna(subset=["open", "high", "low", "close"]).sort_index()
    return ohlcv


def timeframe_to_minutes(timeframe: str) -> int:
    return normalize_timeframe(timeframe)[0]
