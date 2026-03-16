"""Data loading utilities for OHLCV CSV inputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, cast

import pandas as pd

LOGGER = logging.getLogger(__name__)


DATETIME_CANDIDATES = [
    "datetime",
    "time",
    "timestamp",
    "date",
    "datetime_utc",
    "trade_time",
    "trade_datetime",
    "約定日時",
    "日時",
]

OPEN_CANDIDATES = ["open", "O", "始値", "open_price", "Open"]
HIGH_CANDIDATES = ["high", "H", "高値", "high_price", "High"]
LOW_CANDIDATES = ["low", "L", "安値", "low_price", "Low"]
CLOSE_CANDIDATES = ["close", "C", "終値", "close_price", "Close"]
VOLUME_CANDIDATES = ["volume", "v", "出来高", "取引量", "Volume", "tick_volume"]
SYMBOL_CANDIDATES = ["symbol", "ticker", "pair", "currency_pair", "通貨ペア", "銘柄"]

ENCODINGS_TO_TRY = ["utf-8-sig", "utf-8", "cp932", "shift_jis", "shift_jisx0213", "euc_jp"]


def _pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for col in candidates:
        key = str(col).lower().strip()
        if key in lower_map:
            return lower_map[key]
    return None


def detect_file_encoding(path: Path) -> str:
    """Try a safe encoding list for Japanese/English CSV."""
    last_err: Optional[Exception] = None
    for enc in ENCODINGS_TO_TRY:
        try:
            pd.read_csv(path, encoding=enc, nrows=5)
            LOGGER.debug("Detected encoding %s for %s", enc, path)
            return enc
        except Exception as exc:  # pragma: no cover - intentional defensive branch
            last_err = exc
            continue
    if last_err is not None:
        raise ValueError(f"Failed to read CSV with supported encodings: {path}") from last_err
    raise ValueError(f"Failed to read CSV: {path}")


def _coerce_datetime(series: pd.Series, timezone: Optional[str]) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if timezone:
        if ts.dt.tz is None:
            return ts.dt.tz_localize(timezone)
        return ts.dt.tz_convert(timezone)
    return ts


def _coerce_float(series: pd.Series, name: str) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if out.isna().all():
        raise ValueError(f"Column {name} is invalid or all missing.")
    return out


def load_ohlcv_csv(
    path: str | Path,
    symbol: Optional[str] = None,
    timezone: Optional[str] = None,
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load OHLCV CSV and normalize to:
    datetime index, columns: open/high/low/close/volume(optional).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    enc = encoding or detect_file_encoding(path)
    raw = pd.read_csv(path, encoding=enc)
    if raw.empty:
        raise ValueError(f"Input CSV is empty: {path}")

    dcol = _pick_column(raw, DATETIME_CANDIDATES)
    ocol = _pick_column(raw, OPEN_CANDIDATES)
    hcol = _pick_column(raw, HIGH_CANDIDATES)
    lcol = _pick_column(raw, LOW_CANDIDATES)
    ccol = _pick_column(raw, CLOSE_CANDIDATES)
    vcol = _pick_column(raw, VOLUME_CANDIDATES)
    scol = _pick_column(raw, SYMBOL_CANDIDATES)

    if dcol is None or ocol is None or hcol is None or lcol is None or ccol is None:
        raise ValueError(
            "CSV must contain datetime/open/high/low/close columns. "
            f"Found columns: {list(raw.columns)}"
        )

    if symbol is not None:
        if scol is None:
            raise ValueError("symbol argument was given, but symbol column not found.")
        raw = raw[raw[scol].astype(str).str.strip() == str(symbol).strip()].copy()
        if raw.empty:
            raise ValueError(f"No rows for symbol={symbol}")

    ohlcv = pd.DataFrame(
        {
            "open": _coerce_float(raw[ocol], "open"),
            "high": _coerce_float(raw[hcol], "high"),
            "low": _coerce_float(raw[lcol], "low"),
            "close": _coerce_float(raw[ccol], "close"),
        },
        index=_coerce_datetime(raw[dcol], timezone),
    )

    if vcol is not None:
        ohlcv["volume"] = _coerce_float(raw[vcol], "volume")

    ohlcv = ohlcv.dropna(subset=["open", "high", "low", "close"]).sort_index()
    if ohlcv.empty:
        raise ValueError(f"No valid OHLC rows after cleaning: {path}")

    if ohlcv.index.tz is None:
        tz_msg = "naive"
    else:
        tz_msg = str(ohlcv.index.tz)
    LOGGER.info("Loaded rows=%s tz=%s columns=%s", len(ohlcv), tz_msg, list(ohlcv.columns))
    return ohlcv


def validate_timezone(ts_index: pd.Index, timezone: Optional[str]) -> pd.Index:
    """Return normalized index if timezone argument is given."""
    if timezone is None:
        return ts_index
    if ts_index.tz is None:
        return ts_index.tz_localize(timezone)
    return ts_index.tz_convert(timezone)
