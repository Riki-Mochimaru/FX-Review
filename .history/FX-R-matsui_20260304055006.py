#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader


# ===== Matsui execution list (Japanese columns; never rendered as text) =====
COL_SYMBOL = "通貨ペア"
COL_SIDE = "売買"
COL_OPEN_CLOSE = "取引区分"
COL_QTY = "数量"
COL_PRICE = "約定価格"
COL_TIME = "約定日時"
COL_PNL_PRIMARY = "受渡金額"
COL_PNL_FALLBACK = "建玉損益(円)"
COL_SWAP = "スワップ"


@dataclass
class OpenLot:
    symbol: str
    direction: str  # "LONG" or "SHORT"
    qty: float
    entry_time: pd.Timestamp
    entry_price: float


def to_float(x) -> float:
    if pd.isna(x):
        return float("nan")
    s = str(x).strip().replace(",", "")
    if s == "":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")

def apply_timezone_to_trades(trades: pd.DataFrame, tz: str) -> pd.DataFrame:
    trades = trades.copy()
    for col in ["entry_time", "exit_time"]:
        t = pd.to_datetime(trades[col], errors="coerce")
        # if naive -> localize, if aware -> convert
        if getattr(t.dt, "tz", None) is None:
            t = t.dt.tz_localize(tz)
        else:
            t = t.dt.tz_convert(tz)
        trades[col] = t
    return trades

def to_time(x) -> pd.Timestamp:
    return pd.to_datetime(x, errors="coerce")


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ===== Trade reconstruction (FIFO) =====
def reconstruct_trades_from_matsui(csv_path: Path, encoding: str = "UTF-8") -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding=encoding)
    required = [COL_SYMBOL, COL_SIDE, COL_OPEN_CLOSE, COL_QTY, COL_PRICE, COL_TIME]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in execution list: {missing}")

    df = df.copy()
    df[COL_TIME] = df[COL_TIME].map(to_time)
    df["_qty"] = df[COL_QTY].map(to_float)
    df["_price"] = df[COL_PRICE].map(to_float)

    # Row-level P&L source
    if COL_PNL_PRIMARY in df.columns:
        df["_pnl_row"] = df[COL_PNL_PRIMARY].map(to_float)
    else:
        df["_pnl_row"] = float("nan")

    if df["_pnl_row"].isna().all():
        if COL_PNL_FALLBACK in df.columns and COL_SWAP in df.columns:
            df["_pnl_row"] = df[COL_PNL_FALLBACK].map(to_float).fillna(0.0) + df[COL_SWAP].map(to_float).fillna(0.0)
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

        # side == "売" closes LONG, side == "買" closes SHORT
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

            rows.append({
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
            })

            if lot.qty <= 1e-12:
                fifo.pop(0)

        opens[(symbol, close_dir)] = fifo

    trades = pd.DataFrame(rows)
    if trades.empty:
        raise ValueError("Could not reconstruct trades (no OPEN/CLOSE matches found).")

    trades["is_win"] = trades["pnl"] > 0
    trades = trades.sort_values("exit_time").reset_index(drop=True)
    return trades


# ===== Metrics =====
def _fit_quantile_regression_irls(
    x: np.ndarray,
    y: np.ndarray,
    q: float,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if len(x) < 2:
        return float("nan"), float("nan")

    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    for _ in range(max_iter):
        resid = y - X @ beta
        w = np.where(resid >= 0, q, 1.0 - q) / np.maximum(np.abs(resid), 1e-6)
        w = np.clip(w, 1e-6, 1e6)
        sw = np.sqrt(w)
        Xw = X * sw[:, None]
        yw = y * sw
        beta_new = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    return float(beta[0]), float(beta[1])


def compute_metrics(
    trades: pd.DataFrame,
    qreg_hold_min: Optional[float] = None,
    qreg_hold_max: Optional[float] = None,
) -> dict:
    pnl = trades["pnl"].astype(float)
    hold_arr = trades["hold_minutes"].astype(float).to_numpy()
    pnl_arr = pnl.to_numpy()
    wins = trades[trades["is_win"]]
    losses = trades[~trades["is_win"]]

    win_rate = float((pnl > 0).mean())
    avg_hold_win = float(wins["hold_minutes"].mean()) if len(wins) else float("nan")
    avg_hold_loss = float(losses["hold_minutes"].mean()) if len(losses) else float("nan")
    avg_win = float(wins["pnl"].mean()) if len(wins) else float("nan")
    avg_loss = float(losses["pnl"].mean()) if len(losses) else float("nan")
    corr_hold_pnl = trades[["hold_minutes", "pnl"]].corr().iloc[0, 1] if len(trades) > 1 else float("nan")
    expectancy = float(pnl.mean())
    median = float(pnl.median())
    std = float(pnl.std(ddof=1)) if len(trades) > 1 else float("nan")

    sum_win = float(wins["pnl"].sum())
    sum_loss = float(losses["pnl"].sum())
    profit_factor = (sum_win / abs(sum_loss)) if sum_loss < 0 else (float("inf") if sum_win > 0 else float("nan"))

    equity = pnl.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    max_dd = float(dd.min())

    seq = trades["is_win"].astype(int).replace({0: -1}).to_numpy()
    max_win_streak = 0
    max_loss_streak = 0
    cur_w = 0
    cur_l = 0
    for v in seq:
        if v == 1:
            cur_w += 1
            cur_l = 0
        else:
            cur_l += 1
            cur_w = 0
        max_win_streak = max(max_win_streak, cur_w)
        max_loss_streak = max(max_loss_streak, cur_l)

    t = pd.to_datetime(trades["exit_time"])
    days = max(1, (t.max().normalize() - t.min().normalize()).days + 1)
    trades_per_day = float(len(trades) / days)
    qreg_mask = np.isfinite(hold_arr) & np.isfinite(pnl_arr)
    if qreg_hold_min is not None:
        qreg_mask &= hold_arr >= float(qreg_hold_min)
    if qreg_hold_max is not None:
        qreg_mask &= hold_arr <= float(qreg_hold_max)
    hold_qreg = hold_arr[qreg_mask]
    pnl_qreg = pnl_arr[qreg_mask]
    hold_median_min = float(np.nanmedian(hold_qreg)) if len(hold_qreg) else float("nan")

    qreg = {}
    for q in (0.10, 0.50, 0.90):
        b0, b1 = _fit_quantile_regression_irls(hold_qreg, pnl_qreg, q)
        qk = f"{int(q * 100):02d}"
        qreg[f"qreg_pnl_hold_q{qk}_intercept"] = b0
        qreg[f"qreg_pnl_hold_q{qk}_slope"] = b1
        qreg[f"qreg_pnl_hold_q{qk}_at_median_hold"] = (b0 + b1 * hold_median_min) if np.isfinite(hold_median_min) else float("nan")

    return {
        "num_trades": int(len(trades)),
        "win_rate": win_rate,
        "avg_hold_win_min": avg_hold_win,
        "avg_hold_loss_min": avg_hold_loss,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "median": median,
        "std": std,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "best_trade": float(pnl.max()),
        "worst_trade": float(pnl.min()),
        "max_win_streak": int(max_win_streak),
        "max_loss_streak": int(max_loss_streak),
        "trades_per_day": trades_per_day,
        "corr_hold_pnl": corr_hold_pnl,
        "qreg_hold_median_min": hold_median_min,
        "qreg_hold_min": float(qreg_hold_min) if qreg_hold_min is not None else float("nan"),
        "qreg_hold_max": float(qreg_hold_max) if qreg_hold_max is not None else float("nan"),
        "qreg_n": int(len(hold_qreg)),
        **qreg,
    }


# ===== Price data loading =====
def load_minute_ohlc(path: Path, tz: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = pick_col(df, ["time", "timestamp", "datetime", "date"])
    if tcol is None:
        raise ValueError("Minute OHLC: missing time column (time/timestamp/datetime).")

    rename_map = {}
    for k in ["open", "high", "low", "close"]:
        if k in df.columns:
            continue
        alt = pick_col(df, [k.upper(), k.capitalize()])
        if alt is not None:
            rename_map[alt] = k

    df = df.rename(columns=rename_map).copy()
    for k in ["open", "high", "low", "close"]:
        if k not in df.columns:
            raise ValueError(f"Minute OHLC: missing column '{k}'.")

    df["_t"] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=["_t"]).sort_values("_t")

    if tz:
        if df["_t"].dt.tz is None:
            df["_t"] = df["_t"].dt.tz_localize(tz)
        else:
            df["_t"] = df["_t"].dt.tz_convert(tz)

    for k in ["open", "high", "low", "close"]:
        df[k] = pd.to_numeric(df[k], errors="coerce")

    return df[["_t", "open", "high", "low", "close"]].reset_index(drop=True)


def load_ticks_optional(path: Optional[Path], tz: Optional[str], minute_ohlc: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    If tick CSV is provided, use it.
    Else, build pseudo-ticks from minute close.
    Returns: (ticks_df with columns [_t, _p], label)
    """
    if path is None:
        ticks = minute_ohlc[["_t", "close"]].rename(columns={"close": "_p"}).copy()
        ticks["_p"] = pd.to_numeric(ticks["_p"], errors="coerce")
        ticks = ticks.dropna(subset=["_t", "_p"]).reset_index(drop=True)
        return ticks, "Pseudo ticks from minute close (no tick file)"

    df = pd.read_csv(path)
    tcol = pick_col(df, ["time", "timestamp", "datetime", "date"])
    if tcol is None:
        raise ValueError("Tick: missing time column (time/timestamp/datetime).")

    pcol = pick_col(df, ["price", "mid", "last", "bid", "ask"])
    if pcol is None:
        raise ValueError("Tick: missing price column (price/mid/last/bid/ask).")

    df["_t"] = pd.to_datetime(df[tcol], errors="coerce")
    df["_p"] = pd.to_numeric(df[pcol], errors="coerce")
    df = df.dropna(subset=["_t", "_p"]).sort_values("_t")

    if tz:
        if df["_t"].dt.tz is None:
            df["_t"] = df["_t"].dt.tz_localize(tz)
        else:
            df["_t"] = df["_t"].dt.tz_convert(tz)

    return df[["_t", "_p"]].reset_index(drop=True), "Real ticks"


def _downsample_df(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    idx = np.linspace(0, len(df) - 1, max_points).astype(int)
    return df.iloc[idx].reset_index(drop=True)


def _downsample_indexed_df(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    idx = np.linspace(0, len(df) - 1, max_points).astype(int)
    return df.iloc[idx]


def _align_timestamp_to_tz(ts: pd.Timestamp, target_tz) -> pd.Timestamp:
    ts = pd.to_datetime(ts, errors="coerce")
    if pd.isna(ts):
        return ts

    if target_tz is None:
        return ts.tz_localize(None) if ts.tzinfo is not None else ts

    if ts.tzinfo is None:
        return ts.tz_localize(target_tz)
    return ts.tz_convert(target_tz)


def _align_datetime_index_to_tz(idx: pd.DatetimeIndex, tz: Optional[str]) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx, errors="coerce")
    if tz:
        if idx.tz is None:
            return idx.tz_localize(tz)
        return idx.tz_convert(tz)
    if idx.tz is not None:
        return idx.tz_localize(None)
    return idx


def _filter_time_range(df: pd.DataFrame, tcol: str, start, end) -> pd.DataFrame:
    series = df[tcol]
    target_tz = series.dt.tz
    start_aligned = _align_timestamp_to_tz(start, target_tz)
    end_aligned = _align_timestamp_to_tz(end, target_tz)
    return df[(series >= start_aligned) & (series <= end_aligned)].reset_index(drop=True)


def compute_focus_window(trades: pd.DataFrame, pad_minutes: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    t0 = pd.to_datetime(trades["entry_time"]).min()
    t1 = pd.to_datetime(trades["exit_time"]).max()
    return t0 - pd.Timedelta(minutes=pad_minutes), t1 + pd.Timedelta(minutes=pad_minutes)


def fetch_usdjpy_ohlc_yf(
    start: pd.Timestamp,
    end: pd.Timestamp,
    tz: Optional[str],
    interval_preferred: str = "1m",
) -> Tuple[pd.DataFrame, str]:
    start_q = pd.to_datetime(start)
    end_q = pd.to_datetime(end) + pd.Timedelta(minutes=1)

    if start_q.tzinfo is not None:
        start_q = start_q.tz_convert("UTC").tz_localize(None)
    if end_q.tzinfo is not None:
        end_q = end_q.tz_convert("UTC").tz_localize(None)

    intervals = [interval_preferred] if interval_preferred == "5m" else [interval_preferred, "5m"]
    last_err = None
    ticker = yf.Ticker("JPY=X")

    for iv in intervals:
        try:
            hist = ticker.history(start=start_q.to_pydatetime(), end=end_q.to_pydatetime(), interval=iv)
            if hist is None or hist.empty:
                raise ValueError("empty OHLC from yfinance")

            hist = hist.rename(columns=str.lower)
            need = ["open", "high", "low", "close"]
            if any(c not in hist.columns for c in need):
                raise ValueError("yfinance OHLC columns are missing")

            ohlc = hist[["open", "high", "low", "close"]].copy()
            ohlc.index = _align_datetime_index_to_tz(ohlc.index, tz)
            ohlc = ohlc.sort_index()
            print(f"[INFO] yfinance JPY=X fetched interval={iv}, rows={len(ohlc)}")
            return ohlc, f"yfinance JPY=X ({iv})"
        except Exception as e:
            last_err = e
            print(f"[WARN] yfinance fetch failed with interval={iv}: {e}")

    raise RuntimeError(f"yfinance fetch failed for JPY=X (1m/5m): {last_err}")


def _make_candle_ohlc_indexed(minute_ohlc: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, tz: Optional[str], max_bars: int) -> pd.DataFrame:
    ohlc = _filter_time_range(minute_ohlc, "_t", start, end)
    ohlc = _downsample_df(ohlc, max_bars)
    if ohlc.empty:
        raise ValueError("Minute OHLC is empty after focus-window filtering.")
    out = ohlc.copy()
    out["_t"] = pd.to_datetime(out["_t"], errors="coerce")
    out = out.dropna(subset=["_t"]).sort_values("_t")
    out["_t"] = _align_datetime_index_to_tz(pd.DatetimeIndex(out["_t"]), tz)
    out = out.set_index("_t")[["open", "high", "low", "close"]]
    return out


# ===== Plotting =====
def _overlay_markers_on_ohlc_index(ax, ohlc_indexed: pd.DataFrame, trades: pd.DataFrame):
    tt = ohlc_indexed.index
    target_tz = tt.tz

    def nearest_idx(ts: pd.Timestamp) -> int:
        ts = _align_timestamp_to_tz(ts, target_tz)
        i = tt.searchsorted(ts)
        if i <= 0:
            return 0
        if i >= len(tt):
            return len(tt) - 1
        prev = tt[i - 1]
        nxt = tt[i]
        return (i - 1) if abs((ts - prev).total_seconds()) <= abs((nxt - ts).total_seconds()) else i

    for _, r in trades.iterrows():
        ei = nearest_idx(pd.to_datetime(r["entry_time"]))
        xi = nearest_idx(pd.to_datetime(r["exit_time"]))
        ep = r.get("entry_price", np.nan)
        xp = r.get("exit_price", np.nan)
        win = bool(r["is_win"])

        if pd.isna(ep):
            ep = float(ohlc_indexed["close"].iloc[ei])
        if pd.isna(xp):
            xp = float(ohlc_indexed["close"].iloc[xi])

        ax.scatter([tt[ei]], [ep], marker="^", s=2)
        ax.scatter([tt[xi]], [xp], marker="o", s=2, alpha=0.9 if win else 0.35)


def overlay_markers_on_ticks(ax, ticks: pd.DataFrame, trades: pd.DataFrame):
    tt = ticks["_t"]

    def nearest_tick(ts: pd.Timestamp) -> int:
        target_tz = tt.dt.tz
        ts = _align_timestamp_to_tz(ts, target_tz)
        i = tt.searchsorted(ts)
        if i <= 0:
            return 0
        if i >= len(tt):
            return len(tt) - 1
        prev = tt.iloc[i - 1]
        nxt = tt.iloc[i]
        return (i - 1) if abs((ts - prev).total_seconds()) <= abs((nxt - ts).total_seconds()) else i

    for _, r in trades.iterrows():
        ei = nearest_tick(pd.to_datetime(r["entry_time"]))
        xi = nearest_tick(pd.to_datetime(r["exit_time"]))
        win = bool(r["is_win"])

        ax.scatter([ticks["_t"].iloc[ei]], [ticks["_p"].iloc[ei]], marker="^", s=10)
        ax.scatter([ticks["_t"].iloc[xi]], [ticks["_p"].iloc[xi]], marker="o", s=10, alpha=0.9 if win else 0.35)


def _metrics_lines(metrics: dict, price_label: str, tick_label: str) -> List[str]:
    def fmt_pct(x: float) -> str:
        return "NaN" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x*100:.2f}%"

    def fmt_num(x: float) -> str:
        return "NaN" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:,.2f}"

    return [
        f"Trades: {metrics['num_trades']}    Win rate: {fmt_pct(metrics['win_rate'])}    Profit factor: {fmt_num(metrics['profit_factor'])}",
        f"Avg hold (wins): {fmt_num(metrics['avg_hold_win_min'])} min    Avg hold (losses): {fmt_num(metrics['avg_hold_loss_min'])} min",
        f"Avg win: {fmt_num(metrics['avg_win'])} JPY    Avg loss: {fmt_num(metrics['avg_loss'])} JPY    Expectancy: {fmt_num(metrics['expectancy'])} JPY/trade",
        f"Corr(hold,pnl): {fmt_num(metrics['corr_hold_pnl'])}",
        (
            "QReg slope pnl~hold (JPY/min): "
            f"q10={fmt_num(metrics['qreg_pnl_hold_q10_slope'])}  "
            f"q50={fmt_num(metrics['qreg_pnl_hold_q50_slope'])}  "
            f"q90={fmt_num(metrics['qreg_pnl_hold_q90_slope'])}"
        ),
        (
            f"QReg hold range: {fmt_num(metrics['qreg_hold_min'])} to {fmt_num(metrics['qreg_hold_max'])} min"
            f"    samples={metrics['qreg_n']}"
        ),
        (
            f"QReg pnl at median hold ({fmt_num(metrics['qreg_hold_median_min'])} min): "
            f"q10={fmt_num(metrics['qreg_pnl_hold_q10_at_median_hold'])}  "
            f"q50={fmt_num(metrics['qreg_pnl_hold_q50_at_median_hold'])}  "
            f"q90={fmt_num(metrics['qreg_pnl_hold_q90_at_median_hold'])} JPY"
        ),
        f"Max drawdown: {fmt_num(metrics['max_drawdown'])} JPY    Best: {fmt_num(metrics['best_trade'])}    Worst: {fmt_num(metrics['worst_trade'])}",
        f"Max win streak: {metrics['max_win_streak']}    Max loss streak: {metrics['max_loss_streak']}    Trades/day: {fmt_num(metrics['trades_per_day'])}",
        f"Price source: {price_label}",
        f"Tick source: {tick_label}",
        "Markers: entry=triangle, exit=circle (exit opacity indicates win/loss).",
    ]


def _bollinger_bands(close: pd.Series, window: int = 20) -> pd.DataFrame:
    ma = close.rolling(window=window, min_periods=window).mean()
    sd = close.rolling(window=window, min_periods=window).std(ddof=0)
    return pd.DataFrame({
        "bb_u2": ma + 2.0 * sd,
        "bb_l2": ma - 2.0 * sd,
        "bb_u3": ma + 3.0 * sd,
        "bb_l3": ma - 3.0 * sd,
    }, index=close.index)


def make_page1_png(
    metrics: dict,
    trades: pd.DataFrame,
    candle_ohlc: pd.DataFrame,
    ticks: pd.DataFrame,
    price_label: str,
    tick_label: str,
    out_png: Path,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
):
    pnl = trades["pnl"].astype(float).to_numpy()
    equity = np.cumsum(pnl)
    # mplfinance is more reliable with tz-naive DatetimeIndex for candle rendering.
    candle_plot = candle_ohlc.copy()
    if candle_plot.index.tz is not None:
        candle_plot.index = candle_plot.index.tz_localize(None)
    target_tz = candle_plot.index.tz
    exit_time = pd.to_datetime(trades["exit_time"]).map(lambda ts: _align_timestamp_to_tz(ts, target_tz))

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle("Trade Report (Page 1/2)", fontsize=14)

    ax1 = fig.add_axes([0.08, 0.64, 0.84, 0.25])
    ax1.set_title("USDJPY Candles with Trade Markers")
    mpf_df = candle_plot.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    mpf_df = mpf_df.dropna(subset=["Open", "High", "Low", "Close"])
    print(f"[INFO] page1 candle bars={len(mpf_df)}")
    mpf.plot(
        mpf_df,
        type="candle",
        style="yahoo",
        volume=False,
        show_nontrading=True,
        ax=ax1,
    )
    bb = _bollinger_bands(candle_plot["close"], window=20)
    ax1.plot(bb.index, bb["bb_u2"], color="tab:blue", linestyle="--", linewidth=1.0, alpha=0.8, label="BB +2σ")
    ax1.plot(bb.index, bb["bb_l2"], color="tab:blue", linestyle="--", linewidth=1.0, alpha=0.8, label="BB -2σ")
    ax1.plot(bb.index, bb["bb_u3"], color="tab:red", linewidth=0.9, alpha=0.8, label="BB +3σ")
    ax1.plot(bb.index, bb["bb_l3"], color="tab:red", linewidth=0.9, alpha=0.8, label="BB -3σ")
    _overlay_markers_on_ohlc_index(ax1, candle_plot, trades)
    ax1.set_ylabel("Price")
    ax1.tick_params(axis="x", labelbottom=False)
    ax1.legend(loc="upper left", fontsize=8, frameon=False)

    ax2 = fig.add_axes([0.08, 0.36, 0.84, 0.22])
    ax2.set_title("Tick Price with Trade Markers")
    ax2.plot(ticks["_t"], ticks["_p"])
    overlay_markers_on_ticks(ax2, ticks, trades)
    ax2.set_ylabel("Price")
    ax2.tick_params(axis="x", labelbottom=False)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_axes([0.08, 0.09, 0.84, 0.20])
    ax3.set_title("Equity Curve (Cumulative P&L)")
    ax3.plot(exit_time, equity)
    ax3.set_ylabel("JPY")
    ax3.grid(True, alpha=0.3)

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_page2_png(metrics: dict, price_label: str, tick_label: str, trades: pd.DataFrame, out_png: Path):
    pnl = trades["pnl"].astype(float).to_numpy()
    hold = trades["hold_minutes"].astype(float).to_numpy()
    is_win = trades["is_win"].astype(bool).to_numpy()

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle("Trade Report (Page 2/2)", fontsize=14)


    ax0 = fig.add_axes([0.08, 0.76, 0.84, 0.18])
    ax0.axis("off")
    lines = _metrics_lines(metrics, price_label, tick_label)
    y = 0.95
    line_step = 0.90 / max(1, len(lines) - 1)
    for s in lines:
        ax0.text(0.0, y, s, transform=ax0.transAxes, fontsize=9, va="top")
        y -= line_step

    axs = fig.subplots(2, 2)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.70, hspace=0.35, wspace=0.25)

    ax1 = axs[0, 0]
    ax1.set_title("Holding Time vs P&L (All)")
    ax1.scatter(hold, pnl, alpha=0.7)
    x1_min = float(np.nanmin(hold)) if len(hold) else float("nan")
    x1_max = float(np.nanmax(hold)) if len(hold) else float("nan")
    if np.isfinite(x1_min) and np.isfinite(x1_max) and x1_max > x1_min:
        x1g = np.linspace(x1_min, x1_max, 100)
        for q_label, color in [("10", "tab:green"), ("50", "tab:orange"), ("90", "tab:red")]:
            b0 = metrics.get(f"qreg_pnl_hold_q{q_label}_intercept", float("nan"))
            b1 = metrics.get(f"qreg_pnl_hold_q{q_label}_slope", float("nan"))
            if np.isfinite(b0) and np.isfinite(b1):
                ax1.plot(x1g, b0 + b1 * x1g, color=color, linewidth=1.2, label=f"q={int(q_label)/100:.2f}")
        ax1.legend(fontsize=8, frameon=False)
    ax1.set_ylabel("P&L (JPY)")
    ax1.set_xlabel("Hold Time (Minutes)")
    ax1.grid(True, alpha=0.3)

    ax2 = axs[0, 1]
    qreg_hold_min = metrics.get("qreg_hold_min", float("nan"))
    qreg_hold_max = metrics.get("qreg_hold_max", float("nan"))
    mask = np.isfinite(hold) & np.isfinite(pnl)
    if np.isfinite(qreg_hold_min):
        mask &= hold >= qreg_hold_min
    if np.isfinite(qreg_hold_max):
        mask &= hold <= qreg_hold_max
    hold_filtered = hold[mask]
    pnl_filtered = pnl[mask]
    range_parts = []
    if np.isfinite(qreg_hold_min):
        range_parts.append(f">={qreg_hold_min:.2f}")
    if np.isfinite(qreg_hold_max):
        range_parts.append(f"<={qreg_hold_max:.2f}")
    range_label = "all" if not range_parts else " and ".join(range_parts)
    ax2.set_title(f"Holding Time vs P&L (QReg Range: {range_label} min)")
    ax2.scatter(hold_filtered, pnl_filtered, alpha=0.7)
    x_min = float(np.nanmin(hold_filtered)) if len(hold_filtered) else float("nan")
    x_max = float(np.nanmax(hold_filtered)) if len(hold_filtered) else float("nan")
    if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
        xg = np.linspace(x_min, x_max, 100)
        for q_label, color in [("10", "tab:green"), ("50", "tab:orange"), ("90", "tab:red")]:
            b0 = metrics.get(f"qreg_pnl_hold_q{q_label}_intercept", float("nan"))
            b1 = metrics.get(f"qreg_pnl_hold_q{q_label}_slope", float("nan"))
            if np.isfinite(b0) and np.isfinite(b1):
                ax2.plot(xg, b0 + b1 * xg, color=color, linewidth=1.2, label=f"q={int(q_label)/100:.2f}")
        ax2.legend(fontsize=8, frameon=False)
    ax2.set_ylabel("P&L (JPY)")
    ax2.set_xlabel("Hold Time (Minutes)")
    ax2.grid(True, alpha=0.3)

    ax3 = axs[1, 0]
    ax3.set_title("P&L Distribution")
    ax3.hist(pnl, bins=20)
    ax3.set_xlabel("JPY")
    ax3.set_ylabel("Count")
    ax3.grid(True, alpha=0.3)

    ax4 = axs[1, 1]
    ax4.set_title("Holding Time (Minutes)")
    ax4.hist(hold[is_win], bins=20, alpha=0.7, label="Wins")
    ax4.hist(hold[~is_win], bins=20, alpha=0.7, label="Losses")
    ax4.set_xlabel("Minutes")
    ax4.set_ylabel("Count")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_pdf_from_pngs(pngs: List[Path], out_pdf: Path):
    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    w, h = A4
    margin = 8 * mm

    for p in pngs:
        img = ImageReader(str(p))
        img_w = w - 2 * margin
        img_h = h - 2 * margin
        c.drawImage(img, margin, margin, width=img_w, height=img_h, preserveAspectRatio=True, anchor="sw")
        c.showPage()
    c.save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execution_csv", type=str, required=True)
    ap.add_argument("--execution_encoding", type=str, default="CP932")

    ap.add_argument("--tz", type=str, default=None)
    ap.add_argument("--pad_minutes", type=int, default=60)
    ap.add_argument("--max_minute_bars", type=int, default=4000)
    ap.add_argument("--max_ticks", type=int, default=60000)
    ap.add_argument("--qreg_hold_min", type=float, default=None)
    ap.add_argument("--qreg_hold_max", type=float, default=None)

    ap.add_argument("--outdir", type=str, default="out_report")
    ap.add_argument("--prefix", type=str, default="overlay_report")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = f"{args.prefix}_{run_ts}"

    trades = reconstruct_trades_from_matsui(Path(args.execution_csv), encoding=args.execution_encoding)
    if args.tz:
        trades = apply_timezone_to_trades(trades, args.tz)
    qreg_hold_min = args.qreg_hold_min
    qreg_hold_max = args.qreg_hold_max
    if qreg_hold_min is not None and qreg_hold_max is not None and qreg_hold_min > qreg_hold_max:
        qreg_hold_min, qreg_hold_max = qreg_hold_max, qreg_hold_min
    metrics = compute_metrics(trades, qreg_hold_min=qreg_hold_min, qreg_hold_max=qreg_hold_max)
    print("metrics.corr_hold_pnl =", metrics["corr_hold_pnl"])
    # Render traded time window with optional padding
    start, end = compute_focus_window(trades, pad_minutes=args.pad_minutes)

    try:
        candle_ohlc, price_label = fetch_usdjpy_ohlc_yf(start, end, args.tz, interval_preferred="1m")
        candle_ohlc = candle_ohlc[(candle_ohlc.index >= _align_timestamp_to_tz(start, candle_ohlc.index.tz)) &
                                  (candle_ohlc.index <= _align_timestamp_to_tz(end, candle_ohlc.index.tz))]
        if candle_ohlc.empty:
            raise ValueError("yfinance OHLC is empty in focus window")
        candle_ohlc = _downsample_indexed_df(candle_ohlc, args.max_minute_bars)
    except Exception as e:
        print(f"[ERROR] yfinance fetch failed. report generation aborted: {e}")
        return

    if candle_ohlc.empty:
        raise ValueError("Candle OHLC is empty in the focus window.")

    tick_label = "Pseudo ticks from yfinance close"
    idx_name = candle_ohlc.index.name or "_t"
    ticks = (
        candle_ohlc.reset_index()
        .rename(columns={idx_name: "_t", "close": "_p"})[["_t", "_p"]]
        .copy()
    )
    ticks = _downsample_df(ticks, args.max_ticks)

    # Save reconstructed trades
    trades_out = outdir / f"{run_prefix}_trades.csv"
    trades.to_csv(trades_out, index=False, encoding="utf-8")

    # PNG pages define the exact report content
    p1 = outdir / f"{run_prefix}_page1.png"
    p2 = outdir / f"{run_prefix}_page2.png"
    make_page1_png(metrics, trades, candle_ohlc, ticks, price_label, tick_label, p1, start=start, end=end)
    make_page2_png(metrics, price_label, tick_label, trades, p2)

    # PDF == PNG pages
    pdf_out = outdir / f"{run_prefix}.pdf"
    make_pdf_from_pngs([p1, p2], pdf_out)

    print("OK")
    print(f"PDF: {pdf_out}")
    print(f"PNG page1: {p1}")
    print(f"PNG page2: {p2}")
    print(f"Trades CSV: {trades_out}")


if __name__ == "__main__":
    main()
