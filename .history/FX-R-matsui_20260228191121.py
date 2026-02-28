#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
def compute_metrics(trades: pd.DataFrame) -> dict:
    pnl = trades["pnl"].astype(float)
    wins = trades[trades["is_win"]]
    losses = trades[~trades["is_win"]]

    win_rate = float((pnl > 0).mean())
    avg_hold_win = float(wins["hold_minutes"].mean()) if len(wins) else float("nan")
    avg_hold_loss = float(losses["hold_minutes"].mean()) if len(losses) else float("nan")
    avg_win = float(wins["pnl"].mean()) if len(wins) else float("nan")
    avg_loss = float(losses["pnl"].mean()) if len(losses) else float("nan")

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


def _downsample_df(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    idx = np.linspace(0, len(df) - 1, max_points).astype(int)
    return df.iloc[idx].reset_index(drop=True)


def _align_timestamp_to_tz(ts: pd.Timestamp, target_tz) -> pd.Timestamp:
    ts = pd.to_datetime(ts, errors="coerce")
    if pd.isna(ts):
        return ts

    if target_tz is None:
        return ts.tz_localize(None) if ts.tzinfo is not None else ts

    if ts.tzinfo is None:
        return ts.tz_localize(target_tz)
    return ts.tz_convert(target_tz)


def _filter_time_range(df: pd.DataFrame, tcol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    series = df[tcol]
    target_tz = series.dt.tz
    start_aligned = _align_timestamp_to_tz(start, target_tz)
    end_aligned = _align_timestamp_to_tz(end, target_tz)
    return df[(series >= start_aligned) & (series <= end_aligned)].reset_index(drop=True)


def compute_focus_window(trades: pd.DataFrame, pad_minutes: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    t0 = pd.to_datetime(trades["entry_time"]).min()
    t1 = pd.to_datetime(trades["exit_time"]).max()
    return t0 - pd.Timedelta(minutes=pad_minutes), t1 + pd.Timedelta(minutes=pad_minutes)


# ===== Plotting =====
def draw_candles(ax, ohlc: pd.DataFrame, start: int  width: float = 0.6):
    x = np.arange(len(ohlc))
    o = ohlc["open"].to_numpy()
    h = ohlc["high"].to_numpy()
    l = ohlc["low"].to_numpy()
    c = ohlc["close"].to_numpy()
    compute_focus_window
    for i in range(len(ohlc)):
        ax.vlines(x[i], l[i], h[i], linewidth=1)
        y0 = min(o[i], c[i])
        height = abs(c[i] - o[i])
        if height == 0:
            height = 1e-9
        rect = plt.Rectangle((x[i] - width / 2, y0), width, height)
        ax.add_patch(rect)

    ax.set_xlim()
    ticks = np.linspace(0, len(ohlc) - 1, 6).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([ohlc["_t"].iloc[i].strftime("%m-%d %H:%M") for i in ticks])
    ax.grid(True, alpha=0.3)


def overlay_markers_on_minute(ax, ohlc: pd.DataFrame, trades: pd.DataFrame):
    t = ohlc["_t"]
    target_tz = t.dt.tz

    def nearest_idx(ts: pd.Timestamp) -> int:
        ts = _align_timestamp_to_tz(ts, target_tz)
        i = t.searchsorted(ts)
        if i <= 0:
            return 0
        if i >= len(t):
            return len(t) - 1
        prev = t.iloc[i - 1]
        nxt = t.iloc[i]
        return (i - 1) if abs((ts - prev).total_seconds()) <= abs((nxt - ts).total_seconds()) else i

    x = np.arange(len(ohlc))
    for _, r in trades.iterrows():
        ei = nearest_idx(pd.to_datetime(r["entry_time"]))
        xi = nearest_idx(pd.to_datetime(r["exit_time"]))
        ep = r.get("entry_price", np.nan)
        xp = r.get("exit_price", np.nan)
        win = bool(r["is_win"])

        if pd.isna(ep):
            ep = float(ohlc["close"].iloc[ei])
        if pd.isna(xp):
            xp = float(ohlc["close"].iloc[xi])

        ax.scatter([x[ei]], [ep], marker="^", s=28)
        ax.scatter([x[xi]], [xp], marker="o", s=28, alpha=0.9 if win else 0.35)


def _metrics_lines(metrics: dict) -> List[str]:
    def fmt_pct(x: float) -> str:
        return "NaN" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x*100:.2f}%"

    def fmt_num(x: float) -> str:
        return "NaN" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:,.2f}"

    return [
        f"Trades: {metrics['num_trades']}    Win rate: {fmt_pct(metrics['win_rate'])}    Profit factor: {fmt_num(metrics['profit_factor'])}",
        f"Avg hold (wins): {fmt_num(metrics['avg_hold_win_min'])} min    Avg hold (losses): {fmt_num(metrics['avg_hold_loss_min'])} min",
        f"Avg win: {fmt_num(metrics['avg_win'])} JPY    Avg loss: {fmt_num(metrics['avg_loss'])} JPY    Expectancy: {fmt_num(metrics['expectancy'])} JPY/trade",
        f"Max drawdown: {fmt_num(metrics['max_drawdown'])} JPY    Best: {fmt_num(metrics['best_trade'])}    Worst: {fmt_num(metrics['worst_trade'])}",
        f"Max win streak: {metrics['max_win_streak']}    Max loss streak: {metrics['max_loss_streak']}    Trades/day: {fmt_num(metrics['trades_per_day'])}",
        "Price source: minute OHLC",
        "Markers: entry=triangle, exit=circle (exit opacity indicates win/loss).",
    ]


def make_page1_png(metrics: dict, trades: pd.DataFrame, ohlc: pd.DataFrame, out_png: Path):
    pnl = trades["pnl"].astype(float).to_numpy()
    equity = np.cumsum(pnl)

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle("Trade Report (Page 1/2)", fontsize=14)

    ax0 = fig.add_axes([0.08, 0.83, 0.84, 0.13])
    ax0.axis("off")
    y = 0.95
    for s in _metrics_lines(metrics):
        ax0.text(0.0, y, s, transform=ax0.transAxes, fontsize=9, va="top")
        y -= 0.16

    ax1 = fig.add_axes([0.08, 0.46, 0.84, 0.33])
    ax1.set_title("Minute Candles with Trade Markers")
    draw_candles(ax1, ohlc, trades)
    overlay_markers_on_minute(ax1, ohlc, trades)
    ax1.set_ylabel("Price")

    ax2 = fig.add_axes([0.08, 0.10, 0.84, 0.28])
    ax2.set_title("Equity Curve (Cumulative P&L)")
    ax2.plot(pd.to_datetime(trades["exit_time"]), equity)
    ax2.set_ylabel("JPY")
    ax2.grid(True, alpha=0.3)

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_page2_png(metrics: dict, trades: pd.DataFrame, ohlc: pd.DataFrame, out_png: Path):
    pnl = trades["pnl"].astype(float).to_numpy()
    hold = trades["hold_minutes"].astype(float).to_numpy()
    is_win = trades["is_win"].astype(bool).to_numpy()

    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle("Trade Report (Page 2/2)", fontsize=14)

    ax0 = fig.add_axes([0.08, 0.83, 0.84, 0.13])
    ax0.axis("off")
    y = 0.95
    for s in _metrics_lines(metrics):
        ax0.text(0.0, y, s, transform=ax0.transAxes, fontsize=9, va="top")
        y -= 0.16

    ax2 = fig.add_axes([0.08, 0.28, 0.84, 0.14])
    ax2.set_title("Drawdown")
    ax2.plot(pd.to_datetime(trades["exit_time"]), dd)
    ax2.set_ylabel("JPY")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_axes([0.08, 0.08, 0.40, 0.16])
    ax3.set_title("P&L Distribution")
    ax3.hist(pnl, bins=20)
    ax3.set_xlabel("JPY")
    ax3.set_ylabel("Count")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_axes([0.52, 0.08, 0.40, 0.16])
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
    ap.add_argument("--execution_encoding", type=str, default="UTF-8")
    ap.add_argument("--minute_csv", type=str, required=True)

    ap.add_argument("--tz", type=str, default=None)
    ap.add_argument("--pad_minutes", type=int, default=60)
    ap.add_argument("--max_minute_bars", type=int, default=4000)

    ap.add_argument("--outdir", type=str, default="out_report")
    ap.add_argument("--prefix", type=str, default="overlay_report")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    trades = reconstruct_trades_from_matsui(Path(args.execution_csv), encoding=args.execution_encoding)
    metrics = compute_metrics(trades)

    ohlc = load_minute_ohlc(Path(args.minute_csv), tz=args.tz)

    # Focus around trades
    start, end = compute_focus_window(trades, pad_minutes=args.pad_minutes)
    ohlc = _filter_time_range(ohlc, "_t", start, end)

    # Downsample
    ohlc = _downsample_df(ohlc, args.max_minute_bars)

    # Save reconstructed trades
    trades_out = outdir / f"{args.prefix}_trades.csv"
    trades.to_csv(trades_out, index=False, encoding="utf-8")

    # PNG pages define the exact report content
    p1 = outdir / f"{args.prefix}_page1.png"
    p2 = outdir / f"{args.prefix}_page2.png"
    make_page1_png(metrics, trades, ohlc, p1, start, end)
    make_page2_png(metrics, trades, ohlc, p2, start, end)

    # PDF == PNG pages
    pdf_out = outdir / f"{args.prefix}.pdf"
    make_pdf_from_pngs([p1, p2], pdf_out)

    print("OK")
    print(f"PDF: {pdf_out}")
    print(f"PNG page1: {p1}")
    print(f"PNG page2: {p2}")
    print(f"Trades CSV: {trades_out}")


if __name__ == "__main__":
    main()
