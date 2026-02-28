#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
English-only one-page trading report generator for "execution_list_*.csv" (Japanese broker export).

What it outputs (A4, single page):
- Win rate
- Average holding time (wins / losses)
- Average P&L (wins / losses)
Plus useful extra metrics:
- Expectancy (mean P&L per trade)
- Profit factor
- Max drawdown (from cumulative P&L)
- Best / worst trade
- Max win / loss streak
- Trades per day

Input CSV (example columns in Japanese):
- 通貨ペア, 売買, 取引区分(新規/決済), 数量, 約定価格, 約定日時, 受渡金額, 建玉損益(円), スワップ, ...

Trade reconstruction:
- FIFO matching of OPEN lots vs CLOSE lots per (symbol, direction).
- P&L for each CLOSE row is allocated proportionally by matched quantity.
- P&L source priority: 受渡金額 -> (建玉損益 + スワップ) -> 建玉損益

Note:
- Reads Shift_JIS family by default (cp932).
- All report text is English (no Japanese characters in the PDF).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader


# --- Column mapping (Japanese -> internal keys) ---
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


def to_time(x) -> pd.Timestamp:
    return pd.to_datetime(x, errors="coerce")


def reconstruct_trades(csv_path: Path, encoding: str = "cp932") -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding=encoding)
    required = [COL_SYMBOL, COL_SIDE, COL_OPEN_CLOSE, COL_QTY, COL_PRICE, COL_TIME]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    df = df.copy()
    df[COL_TIME] = df[COL_TIME].map(to_time)
    df["_qty"] = df[COL_QTY].map(to_float)
    df["_price"] = df[COL_PRICE].map(to_float)

    # Decide P&L column
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
            # If truly no P&L column exists, we cannot compute it
            raise ValueError("No usable P&L columns found (expected 受渡金額 or 建玉損益(円)(+スワップ)).")

    df = df.sort_values(COL_TIME).reset_index(drop=True)

    # FIFO queues for open lots: key=(symbol, direction)
    opens: dict[tuple[str, str], list[OpenLot]] = {}
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

        # Close semantics:
        # - If side == "売" => closing LONG
        # - If side == "買" => closing SHORT
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
    sum_loss = float(losses["pnl"].sum())  # negative
    profit_factor = (sum_win / abs(sum_loss)) if sum_loss < 0 else (float("inf") if sum_win > 0 else float("nan"))

    # streaks
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

    equity = pnl.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    max_dd = float(dd.min())

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


def make_onepage_chart(trades: pd.DataFrame, out_png: Path) -> None:
    pnl = trades["pnl"].astype(float).to_numpy()
    hold = trades["hold_minutes"].astype(float)
    is_win = trades["is_win"].astype(bool).to_numpy()

    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak

    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait inches
    fig.suptitle("Trading Summary", fontsize=20, fontweight="bold")

    ax1 = fig.add_axes([0.10, 0.73, 0.85, 0.20])
    ax1.plot(pd.to_datetime(trades["exit_time"]), equity)
    ax1.set_title("Equity Curve (Cumulative P&L)")
    ax1.set_ylabel("JPY")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_axes([0.10, 0.51, 0.85, 0.18])
    ax2.plot(pd.to_datetime(trades["exit_time"]), dd)
    ax2.set_title("Drawdown")
    ax2.set_ylabel("JPY")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_axes([0.10, 0.29, 0.85, 0.18])
    ax3.hist(pnl, bins=20)
    ax3.set_title("P&L Distribution (Per Trade)")
    ax3.set_xlabel("JPY")
    ax3.set_ylabel("Count")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_axes([0.10, 0.08, 0.85, 0.17])
    ax4.hist(hold[is_win], bins=20, alpha=0.7, label="Wins")
    ax4.hist(hold[~is_win], bins=20, alpha=0.7, label="Losses")
    ax4.set_title("Holding Time Distribution (Minutes)")
    ax4.set_xlabel("Minutes")
    ax4.set_ylabel("Count")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_pdf(metrics: dict, chart_png: Path, out_pdf: Path, src_name: str) -> None:
    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    w, h = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(20 * mm, h - 20 * mm, "Trading Report (One Page)")
    c.setFont("Helvetica", 10)
    c.drawString(20 * mm, h - 26 * mm, f"Input: {src_name}")

    x0 = 20 * mm
    y0 = h - 35 * mm
    line = 5.2 * mm

    def fmt_pct(x: float) -> str:
        return "NaN" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x*100:.2f}%"

    def fmt_num(x: float) -> str:
        return "NaN" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:,.2f}"

    rows = [
        ("Trades", str(metrics["num_trades"])),
        ("Win rate", fmt_pct(metrics["win_rate"])),
        ("Avg hold (wins) [min]", fmt_num(metrics["avg_hold_win_min"])),
        ("Avg hold (losses) [min]", fmt_num(metrics["avg_hold_loss_min"])),
        ("Avg win [JPY]", fmt_num(metrics["avg_win"])),
        ("Avg loss [JPY]", fmt_num(metrics["avg_loss"])),
        ("Expectancy [JPY/trade]", fmt_num(metrics["expectancy"])),
        ("Median P&L [JPY]", fmt_num(metrics["median"])),
        ("P&L stdev [JPY]", fmt_num(metrics["std"])),
        ("Profit factor", fmt_num(metrics["profit_factor"])),
        ("Max drawdown [JPY]", fmt_num(metrics["max_drawdown"])),
        ("Best trade [JPY]", fmt_num(metrics["best_trade"])),
        ("Worst trade [JPY]", fmt_num(metrics["worst_trade"])),
        ("Max win streak", str(metrics["max_win_streak"])),
        ("Max loss streak", str(metrics["max_loss_streak"])),
        ("Trades per day", fmt_num(metrics["trades_per_day"])),
    ]

    c.setFont("Helvetica-Bold", 12)
    c.drawString(x0, y0, "Key Metrics")
    c.setFont("Helvetica", 10)

    y = y0 - 2 * line
    for k, v in rows:
        c.drawString(x0, y, f"{k}: {v}")
        y -= line

    img = ImageReader(str(chart_png))
    img_w = w - 40 * mm
    img_h = 125 * mm
    c.drawImage(img, 20 * mm, 20 * mm, width=img_w, height=img_h, preserveAspectRatio=True, anchor="sw")

    c.showPage()
    c.save()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to execution_list_*.csv")
    ap.add_argument("--encoding", type=str, default="cp932", help="Default: cp932 (Shift_JIS family)")
    ap.add_argument("--outdir", type=str, default="out_report", help="Output directory")
    ap.add_argument("--prefix", type=str, default="trade_report", help="Output filename prefix")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    trades = reconstruct_trades(csv_path, encoding=args.encoding)
    metrics = compute_metrics(trades)

    # Save reconstructed trades (English columns)
    trades_out = outdir / f"{args.prefix}_trades.csv"
    trades.to_csv(trades_out, index=False, encoding="utf-8")

    chart_png = outdir / f"{args.prefix}_chart.png"
    make_onepage_chart(trades, chart_png)

    out_pdf = outdir / f"{args.prefix}.pdf"
    make_pdf(metrics, chart_png, out_pdf, src_name=str(csv_path))

    print("=== Key Metrics ===")
    print(f"Trades: {metrics['num_trades']}")
    print(f"Win rate: {metrics['win_rate']*100:.2f}%")
    print(f"Avg hold (wins): {metrics['avg_hold_win_min']:.2f} min")
    print(f"Avg hold (losses): {metrics['avg_hold_loss_min']:.2f} min")
    print(f"Avg win: {metrics['avg_win']:.2f} JPY")
    print(f"Avg loss: {metrics['avg_loss']:.2f} JPY")
    print(f"Expectancy: {metrics['expectancy']:.2f} JPY/trade")
    print(f"Profit factor: {metrics['profit_factor']:.2f}")
    print(f"Max drawdown: {metrics['max_drawdown']:.2f} JPY")
    print(f"PDF: {out_pdf}")
    print(f"PNG: {chart_png}")
    print(f"Trades CSV: {trades_out}")


if __name__ == "__main__":
    main()