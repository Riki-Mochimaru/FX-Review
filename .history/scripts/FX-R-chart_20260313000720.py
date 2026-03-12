#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_trade_windows.py

Example:
python scripts/FX-R-chart.py \
  --trades data/execution_list_20260313000219.csv \
  --before-minutes 30 \
  --after-minutes 30 \
  --candle-minutes 5 \
  --trades-per-page 4 \
  --outdir out/
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import yfinance as yf


# ===== Matsui execution-list columns (raw CSV support) =====
COL_SYMBOL = "通貨ペア"
COL_SIDE = "売買"
COL_OPEN_CLOSE = "取引区分"
COL_QTY = "数量"
COL_PRICE = "約定価格"
COL_TIME = "約定日時"
COL_PNL_PRIMARY = "受渡金額"
COL_PNL_FALLBACK = "建玉損益(円)"
COL_SWAP = "スワップ"


TRADE_COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "entry_time": ["entry_time", "EntryTime", "open_time", "entry_datetime"],
    "exit_time": ["exit_time", "ExitTime", "close_time", "exit_datetime"],
    "direction": ["direction", "side", "buy_sell", "position_side"],
    "entry_price": ["entry_price", "EntryPrice", "open_price"],
    "exit_price": ["exit_price", "ExitPrice", "close_price"],
    "pnl": ["pnl", "PnL", "profit", "損益", "損益(円)", "建玉損益(円)", "受渡金額"],
    "trade_id": ["trade_id", "id", "ticket", "チケット番号"],
    "symbol": ["symbol", "銘柄", "通貨ペア"],
}

TICK_TIME_CANDIDATES = ["time", "timestamp", "datetime", "date", "_t", "日時", "約定日時"]
TICK_PRICE_CANDIDATES = ["price", "mid", "close", "last", "_p", "bid", "ask", "約定価格"]
BB_WINDOW = 20
BB_SIGMA = 3.0


def pick_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Return the first existing column name from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_float(x) -> float:
    """Parse numeric value from noisy CSV cell."""
    if pd.isna(x):
        return float("nan")
    s = str(x).strip().replace(",", "")
    if s == "":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


@dataclass
class OpenLot:
    """Open position fragment for FIFO reconstruction."""

    symbol: str
    direction: str  # LONG or SHORT
    qty: float
    entry_time: pd.Timestamp
    entry_price: float


def reconstruct_trades_from_matsui(csv_path: Path, encoding: str = "UTF-8") -> pd.DataFrame:
    """Reconstruct trade pairs (entry/exit) from Matsui execution list using FIFO."""
    df = pd.read_csv(csv_path, encoding=encoding)
    required = [COL_SYMBOL, COL_SIDE, COL_OPEN_CLOSE, COL_QTY, COL_PRICE, COL_TIME]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in execution list: {missing}")

    df = df.copy()
    df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")
    df["_qty"] = df[COL_QTY].map(to_float)
    df["_price"] = df[COL_PRICE].map(to_float)

    if COL_PNL_PRIMARY in df.columns:
        df["_pnl_row"] = df[COL_PNL_PRIMARY].map(to_float)
    else:
        df["_pnl_row"] = float("nan")

    if df["_pnl_row"].isna().all():
        if COL_PNL_FALLBACK in df.columns and COL_SWAP in df.columns:
            df["_pnl_row"] = (
                df[COL_PNL_FALLBACK].map(to_float).fillna(0.0)
                + df[COL_SWAP].map(to_float).fillna(0.0)
            )
        elif COL_PNL_FALLBACK in df.columns:
            df["_pnl_row"] = df[COL_PNL_FALLBACK].map(to_float)
        else:
            raise ValueError("No usable P&L columns found in execution list.")

    df = df.sort_values(COL_TIME).reset_index(drop=True)

    opens: Dict[Tuple[str, str], List[OpenLot]] = {}
    rows = []

    for _, r in df.iterrows():
        symbol = str(r[COL_SYMBOL]).strip()
        side = str(r[COL_SIDE]).strip()
        oc = str(r[COL_OPEN_CLOSE]).strip()
        t = r[COL_TIME]
        qty = float(r["_qty"]) if not pd.isna(r["_qty"]) else 0.0
        price = float(r["_price"]) if not pd.isna(r["_price"]) else float("nan")

        if pd.isna(t) or qty <= 0:
            continue

        if oc == "新規":
            direction = "LONG" if side == "買" else "SHORT"
            opens.setdefault((symbol, direction), []).append(OpenLot(symbol, direction, qty, t, price))
            continue

        if oc != "決済":
            continue

        close_dir = "LONG" if side == "売" else "SHORT"
        fifo = opens.get((symbol, close_dir), [])

        remaining = qty
        pnl_row_total = float(r["_pnl_row"]) if not pd.isna(r["_pnl_row"]) else 0.0
        close_qty_total = qty

        while remaining > 1e-12 and fifo:
            lot = fifo[0]
            matched = min(remaining, lot.qty)
            remaining -= matched
            lot.qty -= matched

            alloc_pnl = pnl_row_total * (matched / close_qty_total) if close_qty_total > 0 else 0.0
            hold_sec = (t - lot.entry_time).total_seconds()

            rows.append(
                {
                    "trade_id": len(rows) + 1,
                    "symbol": symbol,
                    "direction": close_dir,
                    "qty": matched,
                    "entry_time": lot.entry_time,
                    "exit_time": t,
                    "entry_price": lot.entry_price,
                    "exit_price": price,
                    "hold_seconds": hold_sec,
                    "hold_minutes": hold_sec / 60.0,
                    "pnl": alloc_pnl,
                }
            )

            if lot.qty <= 1e-12:
                fifo.pop(0)

        opens[(symbol, close_dir)] = fifo

    trades = pd.DataFrame(rows)
    if trades.empty:
        raise ValueError("Could not reconstruct trades from execution list.")

    return trades.sort_values("exit_time").reset_index(drop=True)


def _load_fx_r_matsui_module():
    """Load FX-R-matsui.py dynamically (hyphenated filename)."""
    here = Path(__file__).resolve()
    candidates = [
        here.parent / "FX-R-matsui.py",        # scripts/FX-R-matsui.py
        here.parents[1] / "FX-R-matsui.py",    # repo_root/FX-R-matsui.py
    ]
    for module_path in candidates:
        if not module_path.exists():
            continue
        spec = importlib.util.spec_from_file_location("fx_r_matsui_shared", str(module_path))
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    return None


def _bind_shared_functions() -> None:
    """Reuse helpers from FX-R-matsui.py when available."""
    mod = _load_fx_r_matsui_module()
    if mod is None:
        print("[INFO] FX-R-matsui.py not found. Using local helpers.")
        return

    shared_names = ["pick_col", "to_float", "reconstruct_trades_from_matsui"]
    for name in shared_names:
        if hasattr(mod, name):
            globals()[name] = getattr(mod, name)
    print("[INFO] Reusing helper functions from FX-R-matsui.py")


def normalize_direction(x: str) -> str:
    """Normalize side/direction into BUY or SELL."""
    s = str(x).strip().upper()
    if s in {"LONG", "BUY", "B", "買"}:
        return "BUY"
    if s in {"SHORT", "SELL", "S", "売"}:
        return "SELL"
    return s


def load_trades_csv(path: Path, encoding: str = "utf-8") -> pd.DataFrame:
    """Load trades CSV.

    Supports both:
    - reconstructed trades CSV with entry_time/exit_time
    - Matsui raw execution list (auto reconstructed)
    """
    used_encoding = encoding
    try:
        df = pd.read_csv(path, encoding=used_encoding)
    except UnicodeDecodeError:
        used_encoding = "cp932"
        df = pd.read_csv(path, encoding=used_encoding)

    # If already reconstructed format, map columns.
    if "entry_time" in df.columns and "exit_time" in df.columns:
        out = df.copy()
    else:
        # Try candidate mapping first.
        mapped = {}
        for std, candidates in TRADE_COLUMN_CANDIDATES.items():
            c = pick_col(df, candidates)
            if c is not None:
                mapped[std] = c

        if "entry_time" in mapped and "exit_time" in mapped:
            out = pd.DataFrame(
                {
                    "trade_id": df[mapped["trade_id"]] if "trade_id" in mapped else np.arange(len(df)) + 1,
                    "symbol": df[mapped["symbol"]] if "symbol" in mapped else "",
                    "direction": df[mapped["direction"]] if "direction" in mapped else "",
                    "entry_time": df[mapped["entry_time"]],
                    "exit_time": df[mapped["exit_time"]],
                    "entry_price": pd.to_numeric(df[mapped["entry_price"]], errors="coerce") if "entry_price" in mapped else np.nan,
                    "exit_price": pd.to_numeric(df[mapped["exit_price"]], errors="coerce") if "exit_price" in mapped else np.nan,
                    "pnl": pd.to_numeric(df[mapped["pnl"]], errors="coerce") if "pnl" in mapped else np.nan,
                }
            )
        else:
            # Finally, try Matsui raw format.
            try:
                out = reconstruct_trades_from_matsui(path, encoding=used_encoding)
            except UnicodeDecodeError:
                fallback = "cp932" if used_encoding.lower() != "cp932" else "utf-8"
                out = reconstruct_trades_from_matsui(path, encoding=fallback)

    if "trade_id" not in out.columns:
        out["trade_id"] = np.arange(len(out)) + 1
    if "direction" not in out.columns:
        out["direction"] = ""
    if "symbol" not in out.columns:
        out["symbol"] = ""
    if "pnl" not in out.columns:
        out["pnl"] = np.nan

    out = out.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce")
    out["exit_time"] = pd.to_datetime(out["exit_time"], errors="coerce")
    out["entry_price"] = pd.to_numeric(out["entry_price"], errors="coerce")
    out["exit_price"] = pd.to_numeric(out["exit_price"], errors="coerce")
    out["pnl"] = pd.to_numeric(out["pnl"], errors="coerce")
    out["direction"] = out["direction"].map(normalize_direction)

    out = out.dropna(subset=["entry_time", "exit_time"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid trade rows found after parsing.")
    return out


def load_ticks_csv(path: Path, encoding: str = "utf-8") -> pd.DataFrame:
    """Load tick CSV and normalize into columns: _t, _p."""
    try:
        df = pd.read_csv(path, encoding=encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp932")

    tcol = pick_col(df, TICK_TIME_CANDIDATES)
    if tcol is None:
        raise ValueError(f"Tick CSV missing time column. candidates={TICK_TIME_CANDIDATES}")

    pcol = pick_col(df, TICK_PRICE_CANDIDATES)
    if pcol is None and ("bid" in df.columns and "ask" in df.columns):
        df["_mid"] = (pd.to_numeric(df["bid"], errors="coerce") + pd.to_numeric(df["ask"], errors="coerce")) / 2.0
        pcol = "_mid"

    if pcol is None:
        raise ValueError(f"Tick CSV missing price column. candidates={TICK_PRICE_CANDIDATES}")

    out = pd.DataFrame({"_t": pd.to_datetime(df[tcol], errors="coerce")})
    out["_p"] = pd.to_numeric(df[pcol], errors="coerce")
    out = out.dropna(subset=["_t", "_p"]).sort_values("_t").reset_index(drop=True)
    if out.empty:
        raise ValueError("Tick CSV has no valid rows.")
    return out


def maybe_localize_or_convert(ts: pd.Series, tz: Optional[str]) -> pd.Series:
    """Apply timezone consistently: localize naive, convert aware."""
    if tz is None:
        return ts
    if getattr(ts.dt, "tz", None) is None:
        return ts.dt.tz_localize(tz)
    return ts.dt.tz_convert(tz)


def fetch_yfinance_ohlc(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    tz: Optional[str] = None,
    interval_preferred: str = "1m",
    source_tz: str = "UTC",
) -> pd.DataFrame:
    """Fetch OHLC from yfinance with 1m->5m fallback."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if tz:
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize(tz)
        else:
            start_ts = start_ts.tz_convert(tz)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize(tz)
        else:
            end_ts = end_ts.tz_convert(tz)

    # yfinance expects UTC-style datetimes for stable intraday range queries.
    if start_ts.tzinfo is None:
        fetch_start = (start_ts - pd.Timedelta(minutes=5)).to_pydatetime()
        fetch_end = (end_ts + pd.Timedelta(minutes=5)).to_pydatetime()
    else:
        fetch_start = (start_ts.tz_convert("UTC") - pd.Timedelta(minutes=5)).tz_localize(None).to_pydatetime()
        fetch_end = (end_ts.tz_convert("UTC") + pd.Timedelta(minutes=5)).tz_localize(None).to_pydatetime()

    interval = interval_preferred
    try:
        raw = yf.download(
            tickers=ticker,
            start=fetch_start,
            end=fetch_end,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if raw.empty:
            raise ValueError("empty 1m data")
    except Exception:
        interval = "5m"
        raw = yf.download(
            tickers=ticker,
            start=fetch_start,
            end=fetch_end,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if raw.empty:
            raise ValueError(f"Failed to fetch yfinance OHLC for {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    col_map = {"Open": "open", "High": "high", "Low": "low", "Close": "close"}
    ohlc = raw.rename(columns=col_map)[["open", "high", "low", "close"]].copy()
    ohlc.index = pd.to_datetime(ohlc.index, errors="coerce")
    ohlc = ohlc.dropna(subset=["open", "high", "low", "close"]).sort_index()
    if tz:
        if ohlc.index.tz is None:
            # yfinance intraday timestamps are effectively UTC when tz-naive.
            ohlc.index = ohlc.index.tz_localize(source_tz).tz_convert(tz)
        else:
            ohlc.index = ohlc.index.tz_convert(tz)
    return ohlc


def build_ohlc_from_ticks(ticks: pd.DataFrame, candle_minutes: int) -> pd.DataFrame:
    """Build m-minute OHLC from tick price series."""
    if candle_minutes <= 0:
        raise ValueError("--candle-minutes must be > 0")
    s = ticks.set_index("_t")["_p"].sort_index()
    ohlc = s.resample(f"{candle_minutes}min").ohlc().dropna()
    return ohlc


def build_ohlc_from_ohlc(ohlc: pd.DataFrame, candle_minutes: int) -> pd.DataFrame:
    """Resample OHLC to target candle minutes."""
    if candle_minutes <= 0:
        raise ValueError("--candle-minutes must be > 0")
    out = (
        ohlc.sort_index()
        .resample(f"{candle_minutes}min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    return out


def slice_trade_window(
    ticks: pd.DataFrame,
    ohlc: pd.DataFrame,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    before_minutes: int,
    after_minutes: int,
    candle_minutes: int = 5,
    bb_window_bars: int = BB_WINDOW,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Slice tick and ohlc by trade window.

    For Bollinger Bands stability, ohlc includes hidden lookback bars.
    """
    base_start = min(entry_time, exit_time)
    base_end = max(entry_time, exit_time)
    start = base_start - pd.Timedelta(minutes=before_minutes)
    end = base_end + pd.Timedelta(minutes=after_minutes)
    warmup_start = start - pd.Timedelta(minutes=max(0, bb_window_bars) * max(1, candle_minutes))

    ticks_w = ticks[(ticks["_t"] >= start) & (ticks["_t"] <= end)].copy()
    ohlc_w = ohlc[(ohlc.index >= warmup_start) & (ohlc.index <= end)].copy()
    return ticks_w, ohlc_w, start, end


def _draw_candles(ax: plt.Axes, ohlc_w: pd.DataFrame) -> None:
    """Draw candlesticks using matplotlib primitives."""
    if ohlc_w.empty:
        return

    x = mdates.date2num(ohlc_w.index.to_pydatetime())
    o = ohlc_w["open"].to_numpy()
    h = ohlc_w["high"].to_numpy()
    l = ohlc_w["low"].to_numpy()
    c = ohlc_w["close"].to_numpy()

    if len(x) > 1:
        w = float(np.median(np.diff(x))) * 0.7
    else:
        w = 0.0006

    for xi, oi, hi, li, ci in zip(x, o, h, l, c):
        up = ci >= oi
        color = "#2ca02c" if up else "#d62728"
        ax.vlines(xi, li, hi, color=color, linewidth=0.8, alpha=0.9)
        y0 = min(oi, ci)
        h_body = max(abs(ci - oi), 1e-8)
        rect = plt.Rectangle((xi - w / 2.0, y0), w, h_body, edgecolor=color, facecolor=color, alpha=0.6)
        ax.add_patch(rect)


def _bollinger_bands(close: pd.Series, window: int = 20, sigma: float = 3.0) -> pd.DataFrame:
    """Return Bollinger Bands and %B from close series."""
    ma = close.rolling(window=window, min_periods=window).mean()
    sd = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = ma + sigma * sd
    lower = ma - sigma * sd
    den = (upper - lower).replace(0.0, np.nan)
    pb = (close - lower) / den
    return pd.DataFrame({"bb_mid": ma, "bb_u": upper, "bb_l": lower, "bb_pb": pb}, index=close.index)


def _event_sides(direction: str) -> Tuple[str, str]:
    """Return (entry_side, exit_side) by trade direction."""
    d = normalize_direction(direction)
    if d == "BUY":
        return "BUY", "SELL"
    if d == "SELL":
        return "SELL", "BUY"
    return "ENTRY", "EXIT"


def _side_marker(side: str) -> str:
    """Buy=up arrow, Sell=down arrow."""
    s = str(side).upper()
    if s == "BUY":
        return "^"
    if s == "SELL":
        return "v"
    return "o"


def _percent_b_at(ts: pd.Timestamp, bb: pd.DataFrame) -> float:
    """Get nearest %B value to timestamp."""
    if bb.empty:
        return float("nan")
    idx = bb.index
    i = idx.searchsorted(ts)
    if i <= 0:
        j = 0
    elif i >= len(idx):
        j = len(idx) - 1
    else:
        prev = idx[i - 1]
        nxt = idx[i]
        j = (i - 1) if abs((ts - prev).total_seconds()) <= abs((nxt - ts).total_seconds()) else i
    return float(bb["bb_pb"].iloc[j]) if "bb_pb" in bb.columns else float("nan")


def plot_single_trade_column(
    ax_candle: plt.Axes,
    ax_tick: plt.Axes,
    trade: pd.Series,
    ticks_w: pd.DataFrame,
    ohlc_w: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> bool:
    """Plot one trade column (candle upper, tick lower). Returns True if plotted."""
    entry_t = pd.to_datetime(trade["entry_time"])
    exit_t = pd.to_datetime(trade["exit_time"])
    entry_p = float(trade.get("entry_price", np.nan))
    exit_p = float(trade.get("exit_price", np.nan))
    direction = str(trade.get("direction", ""))
    entry_side, exit_side = _event_sides(direction)
    entry_marker = _side_marker(entry_side)
    exit_marker = _side_marker(exit_side)

    if ticks_w.empty and ohlc_w.empty:
        msg = "No tick/ohlc in this window"
        ax_candle.text(0.5, 0.5, msg, ha="center", va="center", transform=ax_candle.transAxes, fontsize=9)
        ax_tick.text(0.5, 0.5, msg, ha="center", va="center", transform=ax_tick.transAxes, fontsize=9)
        for ax in (ax_candle, ax_tick):
            ax.set_xlim(start, end)
            ax.grid(True, alpha=0.2)
        return False

    if not ohlc_w.empty:
        _draw_candles(ax_candle, ohlc_w)
        bb = _bollinger_bands(ohlc_w["close"], window=BB_WINDOW, sigma=BB_SIGMA)
        ax_candle.plot(bb.index, bb["bb_u"], color="tab:blue", linestyle="--", linewidth=0.9, alpha=0.8, label="BB +3σ")
        ax_candle.plot(bb.index, bb["bb_mid"], color="gray", linestyle="-", linewidth=0.7, alpha=0.7, label="BB MA20")
        ax_candle.plot(bb.index, bb["bb_l"], color="tab:blue", linestyle="--", linewidth=0.9, alpha=0.8, label="BB -3σ")
    else:
        bb = pd.DataFrame()

    if not ticks_w.empty:
        ax_tick.plot(ticks_w["_t"], ticks_w["_p"], color="tab:blue", linewidth=0.9)

    # Buy/Sell markers by side: BUY=up, SELL=down.
    ax_candle.scatter([entry_t], [entry_p], s=38, c="red", marker=entry_marker, zorder=5)
    ax_tick.scatter([entry_t], [entry_p], s=38, c="red", marker=entry_marker, zorder=5, label=f"Entry {entry_side}")

    ax_candle.scatter([exit_t], [exit_p], s=40, c="tab:orange", marker=exit_marker, zorder=6)
    ax_tick.scatter([exit_t], [exit_p], s=40, c="tab:orange", marker=exit_marker, zorder=6)

    entry_pb = _percent_b_at(entry_t, bb)
    exit_pb = _percent_b_at(exit_t, bb)
    entry_txt = f"ENTRY {entry_side} {entry_t.strftime('%H:%M:%S')} {entry_p:.3f}  %B={entry_pb:.3f}"
    exit_txt = f"EXIT {exit_side} {exit_t.strftime('%H:%M:%S')} {exit_p:.3f}  %B={exit_pb:.3f}"
    ax_candle.annotate(
        f"%B={entry_pb:.3f}",
        xy=(entry_t, entry_p),
        xytext=(4, 10),
        textcoords="offset points",
        fontsize=7,
        color="red",
    )
    ax_candle.annotate(
        f"%B={exit_pb:.3f}",
        xy=(exit_t, exit_p),
        xytext=(4, 10),
        textcoords="offset points",
        fontsize=7,
        color="tab:orange",
    )
    ax_tick.annotate(
        entry_txt,
        xy=(entry_t, entry_p),
        xytext=(4, 8),
        textcoords="offset points",
        fontsize=7,
        color="red",
    )
    ax_tick.annotate(
        exit_txt,
        xy=(exit_t, exit_p),
        xytext=(4, 8),
        textcoords="offset points",
        fontsize=7,
        color="tab:orange",
    )

    # x-range shared per trade column.
    for ax in (ax_candle, ax_tick):
        ax.set_xlim(start, end)
        ax.grid(True, alpha=0.25)

    # Light y-margin for readability.
    for ax in (ax_candle, ax_tick):
        ymin, ymax = ax.get_ylim()
        if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
            pad = (ymax - ymin) * 0.08
            ax.set_ylim(ymin - pad, ymax + pad)

    tid = trade.get("trade_id", "")
    pnl = trade.get("pnl", np.nan)
    title = (
        f"Trade {tid} | {direction}\n"
        f"IN {entry_t.strftime('%m-%d %H:%M:%S')} @ {entry_p:.3f}\n"
        f"OUT {exit_t.strftime('%m-%d %H:%M:%S')} @ {exit_p:.3f} | PnL {pnl:.1f}"
        if pd.notna(pnl)
        else f"Trade {tid} | {direction}\nIN {entry_t.strftime('%m-%d %H:%M:%S')} @ {entry_p:.3f}\nOUT {exit_t.strftime('%m-%d %H:%M:%S')} @ {exit_p:.3f}"
    )
    ax_candle.set_title(title, fontsize=8)
    if not ohlc_w.empty:
        ax_candle.legend(loc="upper left", fontsize=6, frameon=False)
    ax_candle.tick_params(axis="x", labelbottom=False)
    ax_tick.tick_params(axis="x", labelrotation=30)
    ax_candle.set_ylabel("Candle")
    ax_tick.set_ylabel("Tick")

    return True


def paginate_trades(trades: pd.DataFrame, trades_per_page: int) -> List[pd.DataFrame]:
    """Split trades into pages."""
    if trades_per_page <= 0:
        raise ValueError("--trades-per-page must be > 0")
    pages = []
    for i in range(0, len(trades), trades_per_page):
        pages.append(trades.iloc[i : i + trades_per_page].reset_index(drop=True))
    return pages


def save_trade_window_pages(
    trades: pd.DataFrame,
    ticks: pd.DataFrame,
    candle_minutes: int,
    before_minutes: int,
    after_minutes: int,
    trades_per_page: int,
    outdir: Path,
    prefix: str = "trade_windows",
    dpi: int = 160,
    ohlc_source: Optional[pd.DataFrame] = None,
) -> Tuple[Path, int, int, int]:
    """Render paginated trade-window figures and save PDF only."""
    if ohlc_source is None:
        ohlc = build_ohlc_from_ticks(ticks, candle_minutes)
    else:
        ohlc = build_ohlc_from_ohlc(ohlc_source, candle_minutes)
    pages = paginate_trades(trades, trades_per_page)
    ticks_min = ticks["_t"].min() if not ticks.empty else pd.NaT
    ticks_max = ticks["_t"].max() if not ticks.empty else pd.NaT

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{prefix}_{run_ts}"

    rendered = 0
    skipped = 0
    num_pages = 0
    pdf_path = outdir / f"{base}.pdf"

    with PdfPages(pdf_path) as pdf:
        for page_idx, page_trades in enumerate(pages, start=1):
            n = len(page_trades)
            fig_w = max(4.2 * n, 10.0)
            fig_h = 8.2
            fig, axs = plt.subplots(2, n, figsize=(fig_w, fig_h), sharex="col", squeeze=False)
            fig.suptitle(
                f"Trade Windows p{page_idx}/{len(pages)} | before={before_minutes}m after={after_minutes}m candle={candle_minutes}m",
                fontsize=12,
            )

            for col in range(n):
                trade = page_trades.iloc[col]
                ticks_w, ohlc_w, start, end = slice_trade_window(
                    ticks,
                    ohlc,
                    pd.to_datetime(trade["entry_time"]),
                    pd.to_datetime(trade["exit_time"]),
                    before_minutes,
                    after_minutes,
                    candle_minutes=candle_minutes,
                    bb_window_bars=BB_WINDOW,
                )

                ok = plot_single_trade_column(axs[0, col], axs[1, col], trade, ticks_w, ohlc_w, start, end)
                if ok:
                    rendered += 1
                else:
                    skipped += 1
                    print(
                        "[WARN] no data in trade window: "
                        f"trade_id={trade.get('trade_id', col + 1)} "
                        f"window=[{start} .. {end}] "
                        f"ticks=[{ticks_min} .. {ticks_max}]"
                    )

            for ax in axs[1, :]:
                ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

            fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.95])
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)
            num_pages += 1

    return pdf_path, num_pages, rendered, skipped


def main() -> None:
    """CLI entrypoint."""
    _bind_shared_functions()

    ap = argparse.ArgumentParser(description="Plot per-trade before/after windows with candle+tick rows.")
    ap.add_argument("--trades", type=str, required=True, help="Trades CSV path")
    ap.add_argument("--ticks", type=str, default=None, help="Tick CSV path (optional when using yfinance)")
    ap.add_argument("--ticker", type=str, default="JPY=X", help="yfinance ticker (used when --ticks is omitted)")
    ap.add_argument("--before-minutes", type=int, default=30)
    ap.add_argument("--after-minutes", type=int, default=30)
    ap.add_argument("--window-minutes", type=int, default=None, help="If set, overrides before/after minutes")
    ap.add_argument("--candle-minutes", type=int, default=5)
    ap.add_argument("--trades-per-page", type=int, default=4)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--outdir", type=str, default="out/trade_windows")
    ap.add_argument("--prefix", type=str, default="trade_windows")
    ap.add_argument("--trades-encoding", type=str, default="utf-8")
    ap.add_argument("--ticks-encoding", type=str, default="utf-8")
    ap.add_argument("--tz", type=str, default=None)
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    before_minutes = args.before_minutes
    after_minutes = args.after_minutes
    if args.window_minutes is not None:
        before_minutes = args.window_minutes
        after_minutes = args.window_minutes

    if before_minutes < 0 or after_minutes < 0:
        raise ValueError("before/after minutes must be >= 0")

    trades = load_trades_csv(Path(args.trades), encoding=args.trades_encoding)
    if args.tz:
        trades = trades.copy()
        trades["entry_time"] = maybe_localize_or_convert(pd.to_datetime(trades["entry_time"], errors="coerce"), args.tz)
        trades["exit_time"] = maybe_localize_or_convert(pd.to_datetime(trades["exit_time"], errors="coerce"), args.tz)

    trades_min_t = min(pd.to_datetime(trades["entry_time"]).min(), pd.to_datetime(trades["exit_time"]).min())
    trades_max_t = max(pd.to_datetime(trades["entry_time"]).max(), pd.to_datetime(trades["exit_time"]).max())
    global_start = trades_min_t - pd.Timedelta(minutes=before_minutes)
    global_end = trades_max_t + pd.Timedelta(minutes=after_minutes)

    ohlc_source: Optional[pd.DataFrame] = None
    if args.ticks:
        ticks = load_ticks_csv(Path(args.ticks), encoding=args.ticks_encoding)
    else:
        ohlc_source = fetch_yfinance_ohlc(
            ticker=args.ticker,
            start=global_start,
            end=global_end,
            tz=args.tz,
            interval_preferred="1m",
        )
        idx_name = ohlc_source.index.name or "_t"
        ticks = (
            ohlc_source.reset_index()
            .rename(columns={idx_name: "_t", "close": "_p"})[["_t", "_p"]]
            .copy()
        )
        print(f"[INFO] yfinance range requested: {global_start} .. {global_end}")

    if args.tz:
        ticks = ticks.copy()
        ticks["_t"] = maybe_localize_or_convert(pd.to_datetime(ticks["_t"], errors="coerce"), args.tz)
        if ohlc_source is not None:
            if ohlc_source.index.tz is None:
                ohlc_source.index = ohlc_source.index.tz_localize(args.tz)
            else:
                ohlc_source.index = ohlc_source.index.tz_convert(args.tz)
    print(f"[INFO] ticks range loaded: {ticks['_t'].min()} .. {ticks['_t'].max()}")

    if args.start:
        start = pd.to_datetime(args.start, errors="coerce")
        if pd.isna(start):
            raise ValueError("Invalid --start")
        trades = trades[(trades["entry_time"] >= start) | (trades["exit_time"] >= start)].copy()
    if args.end:
        end = pd.to_datetime(args.end, errors="coerce")
        if pd.isna(end):
            raise ValueError("Invalid --end")
        trades = trades[(trades["entry_time"] <= end) | (trades["exit_time"] <= end)].copy()

    trades = trades.sort_values("entry_time").reset_index(drop=True)
    if args.limit is not None and args.limit > 0:
        trades = trades.iloc[: args.limit].copy()

    if trades.empty:
        raise ValueError("No trades to plot after filtering.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pdf_path, num_pages, rendered, skipped = save_trade_window_pages(
        trades=trades,
        ticks=ticks,
        candle_minutes=args.candle_minutes,
        before_minutes=before_minutes,
        after_minutes=after_minutes,
        trades_per_page=args.trades_per_page,
        outdir=outdir,
        prefix=args.prefix,
        dpi=args.dpi,
        ohlc_source=ohlc_source,
    )

    print(f"[INFO] trades input={len(trades)} rendered={rendered} skipped={skipped}")
    print(f"[INFO] pages={num_pages}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
