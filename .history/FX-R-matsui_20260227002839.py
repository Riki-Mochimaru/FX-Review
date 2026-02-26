#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
execution_list_*.csv（約定一覧）から、勝率、平均保持時間（勝ち／負け）、平均損益（勝ち／負け）などを集計し、
A4 1枚PDFレポート（＋PNG、＋トレード明細CSV）を出力する。

このCSVは Shift_JIS（cp932）想定。
列例（今回のファイル）:
- 通貨ペア, 売買, 種類, 取引区分(新規/決済), 数量, 約定価格, 建玉損益(円), スワップ, 受渡金額, 受渡日, 約定日時, 注文日時

建玉の突合はFIFOで行う。
- 新規: 売買=買 → ロング建て、売買=売 → ショート建て
- 決済: 売買=売 → ロング決済、売買=買 → ショート決済
損益は原則「受渡金額」を採用（なければ 建玉損益(円)+スワップ を使用）。
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


@dataclass
class OpenLot:
    symbol: str
    direction: str  # "LONG" or "SHORT"
    qty: float
    entry_time: pd.Timestamp
    entry_price: float


def _to_float(x) -> float:
    if pd.isna(x):
        return float("nan")
    s = str(x).strip().replace(",", "")
    if s == "":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _parse_time(x) -> pd.Timestamp:
    return pd.to_datetime(x, errors="coerce")


def load_and_reconstruct_trades(csv_path: Path, encoding: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding=encoding)

    required = ["通貨ペア", "売買", "取引区分", "数量", "約定価格", "約定日時"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"必要な列が見つかりませんでした: {missing}、検出列={list(df.columns)}")

    df = df.copy()
    df["約定日時"] = df["約定日時"].map(_parse_time)
    df["数量_f"] = df["数量"].map(_to_float)
    df["約定価格_f"] = df["約定価格"].map(_to_float)

    # 損益列の決定
    if "受渡金額" in df.columns:
        df["pnl_f"] = df["受渡金額"].map(_to_float)
    else:
        df["pnl_f"] = float("nan")

    if df["pnl_f"].isna().all():
        # フォールバック: 建玉損益+スワップ
        if "建玉損益(円)" in df.columns and "スワップ" in df.columns:
            df["pnl_f"] = df["建玉損益(円)"].map(_to_float).fillna(0.0) + df["スワップ"].map(_to_float).fillna(0.0)
        elif "建玉損益(円)" in df.columns:
            df["pnl_f"] = df["建玉損益(円)"].map(_to_float)

    df = df.sort_values("約定日時").reset_index(drop=True)

    # open lots: key=(symbol, direction) -> FIFO list
    opens: dict[tuple[str, str], list[OpenLot]] = {}
    trades_rows = []

    for _, r in df.iterrows():
        symbol = str(r["通貨ペア"]).strip()
        side = str(r["売買"]).strip()
        kind = str(r["取引区分"]).strip()
        t = r["約定日時"]
        qty = float(r["数量_f"]) if not pd.isna(r["数量_f"]) else 0.0
        price = float(r["約定価格_f"]) if not pd.isna(r["約定価格_f"]) else float("nan")

        if pd.isna(t) or qty <= 0:
            continue

        if kind == "新規":
            direction = "LONG" if side == "買" else "SHORT"
            key = (symbol, direction)
            opens.setdefault(key, []).append(OpenLot(symbol, direction, qty, t, price))
            continue

        if kind != "決済":
            continue

        # 決済は反対側の建玉を消す
        close_direction = "LONG" if side == "売" else "SHORT"
        open_direction = close_direction  # たとえば売決済はロング建玉を減らす
        key = (symbol, open_direction)
        fifo = opens.get(key, [])

        remaining = qty
        close_pnl_total = float(r["pnl_f"]) if not pd.isna(r["pnl_f"]) else 0.0
        close_qty_total = qty

        # 建玉が足りない場合でも、可能な範囲で処理（データ不整合に強くする）
        while remaining > 1e-12 and fifo:
            lot = fifo[0]
            matched = min(remaining, lot.qty)
            remaining -= matched
            lot.qty -= matched

            # 決済行の損益を数量按分
            alloc_pnl = close_pnl_total * (matched / close_qty_total) if close_qty_total > 0 else 0.0

            hold_sec = (t - lot.entry_time).total_seconds()
            trades_rows.append({
                "symbol": symbol,
                "direction": open_direction,
                "qty": matched,
                "entry_time": lot.entry_time,
                "exit_time": t,
                "hold_seconds": hold_sec,
                "hold_min": hold_sec / 60.0,
                "pnl": alloc_pnl,
            })

            if lot.qty <= 1e-12:
                fifo.pop(0)

        opens[key] = fifo

    trades = pd.DataFrame(trades_rows)
    if trades.empty:
        raise ValueError("新規と決済を突合できず、トレードが構成できませんでした。")

    trades["is_win"] = trades["pnl"] > 0
    trades = trades.sort_values("exit_time").reset_index(drop=True)
    return trades


def compute_metrics(trades: pd.DataFrame) -> dict:
    pnl = trades["pnl"].astype(float)
    wins = trades[trades["is_win"]]
    losses = trades[~trades["is_win"]]

    win_rate = float((pnl > 0).mean())
    avg_hold_win = float(wins["hold_min"].mean()) if len(wins) else float("nan")
    avg_hold_loss = float(losses["hold_min"].mean()) if len(losses) else float("nan")
    avg_win = float(wins["pnl"].mean()) if len(wins) else float("nan")
    avg_loss = float(losses["pnl"].mean()) if len(losses) else float("nan")

    expectancy = float(pnl.mean())
    median = float(pnl.median())
    std = float(pnl.std(ddof=1)) if len(trades) > 1 else float("nan")

    sum_win = float(wins["pnl"].sum())
    sum_loss = float(losses["pnl"].sum())  # 負
    profit_factor = (sum_win / abs(sum_loss)) if sum_loss < 0 else (float("inf") if sum_win > 0 else float("nan"))

    # 連勝連敗
    streak = trades["is_win"].astype(int).replace({0: -1}).to_numpy()
    max_win_streak = 0
    max_loss_streak = 0
    cur_w = 0
    cur_l = 0
    for v in streak:
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
        "trades": int(len(trades)),
        "win_rate": win_rate,
        "avg_hold_win_min": avg_hold_win,
        "avg_hold_loss_min": avg_hold_loss,
        "avg_win_yen": avg_win,
        "avg_loss_yen": avg_loss,
        "expectancy_yen": expectancy,
        "median_yen": median,
        "std_yen": std,
        "profit_factor": profit_factor,
        "max_drawdown_yen": max_dd,
        "best_trade_yen": float(pnl.max()),
        "worst_trade_yen": float(pnl.min()),
        "max_win_streak": int(max_win_streak),
        "max_loss_streak": int(max_loss_streak),
        "trades_per_day": trades_per_day,
    }


def make_figures(trades: pd.DataFrame, out_png: Path) -> None:
    pnl = trades["pnl"].astype(float).to_numpy()
    hold = trades["hold_min"].astype(float)
    is_win = trades["is_win"].astype(bool).to_numpy()

    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak

    fig = plt.figure(figsize=(8.27, 11.69))  # A4縦
    fig.suptitle("取引サマリー", fontsize=16)

    ax1 = fig.add_axes([0.10, 0.73, 0.85, 0.20])
    ax1.plot(pd.to_datetime(trades["exit_time"]), equity)
    ax1.set_title("エクイティカーブ（累積損益）")
    ax1.set_ylabel("円")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_axes([0.10, 0.51, 0.85, 0.18])
    ax2.plot(pd.to_datetime(trades["exit_time"]), dd)
    ax2.set_title("ドローダウン")
    ax2.set_ylabel("円")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_axes([0.10, 0.29, 0.85, 0.18])
    ax3.hist(pnl, bins=20)
    ax3.set_title("損益分布（1トレード）")
    ax3.set_xlabel("円")
    ax3.set_ylabel("回数")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_axes([0.10, 0.08, 0.85, 0.17])
    ax4.hist(hold[is_win], bins=20, alpha=0.7, label="勝ち")
    ax4.hist(hold[~is_win], bins=20, alpha=0.7, label="負け")
    ax4.set_title("保持時間分布（分）")
    ax4.set_xlabel("分")
    ax4.set_ylabel("回数")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_pdf_report(metrics: dict, chart_png: Path, out_pdf: Path, src_name: str) -> None:
    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    w, h = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(20 * mm, h - 20 * mm, "取引レポート（1枚）")
    c.setFont("Helvetica", 10)
    c.drawString(20 * mm, h - 26 * mm, f"入力: {src_name}")

    x0 = 20 * mm
    y0 = h - 35 * mm
    line = 5.2 * mm

    def fmt_pct(x):
        return "NaN" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x*100:.2f}%"

    def fmt_num(x):
        return "NaN" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:,.2f}"

    def fmt_int(x):
        return "NaN" if x is None else f"{int(x)}"

    rows = [
        ("総トレード数", fmt_int(metrics["trades"])),
        ("勝率", fmt_pct(metrics["win_rate"])),
        ("平均保持時間（勝ち）[分]", fmt_num(metrics["avg_hold_win_min"])),
        ("平均保持時間（負け）[分]", fmt_num(metrics["avg_hold_loss_min"])),
        ("平均勝ち額 [円]", fmt_num(metrics["avg_win_yen"])),
        ("平均負け額 [円]", fmt_num(metrics["avg_loss_yen"])),
        ("期待値（平均損益）[円]", fmt_num(metrics["expectancy_yen"])),
        ("損益中央値 [円]", fmt_num(metrics["median_yen"])),
        ("損益標準偏差 [円]", fmt_num(metrics["std_yen"])),
        ("プロフィットファクター", fmt_num(metrics["profit_factor"])),
        ("最大ドローダウン [円]", fmt_num(metrics["max_drawdown_yen"])),
        ("ベストトレード [円]", fmt_num(metrics["best_trade_yen"])),
        ("ワーストトレード [円]", fmt_num(metrics["worst_trade_yen"])),
        ("最大連勝", fmt_int(metrics["max_win_streak"])),
        ("最大連敗", fmt_int(metrics["max_loss_streak"])),
        ("1日あたり平均トレード数", fmt_num(metrics["trades_per_day"])),
    ]

    c.setFont("Helvetica-Bold", 12)
    c.drawString(x0, y0, "数値サマリー")
    c.setFont("Helvetica", 10)

    y = y0 - 2 * line
    for k, v in rows:
        c.drawString(x0, y, f"{k}：{v}")
        y -= line

    img = ImageReader(str(chart_png))
    img_w = w - 40 * mm
    img_h = 125 * mm
    c.drawImage(img, 20 * mm, 20 * mm, width=img_w, height=img_h, preserveAspectRatio=True, anchor="sw")

    c.showPage()
    c.save()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="execution_list_*.csv のパス")
    ap.add_argument("--encoding", type=str, default="cp932", help="Shift_JIS系は cp932 推奨")
    ap.add_argument("--outdir", type=str, default="out_report", help="出力先ディレクトリ")
    ap.add_argument("--prefix", type=str, default="trade_report", help="出力ファイルの接頭辞")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    trades = load_and_reconstruct_trades(csv_path, encoding=args.encoding)
    metrics = compute_metrics(trades)

    trades_out = outdir / f"{args.prefix}_trades.csv"
    trades.to_csv(trades_out, index=False, encoding="utf-8-sig")

    chart_png = outdir / f"{args.prefix}_chart.png"
    make_figures(trades, chart_png)

    out_pdf = outdir / f"{args.prefix}.pdf"
    make_pdf_report(metrics, chart_png, out_pdf, src_name=str(csv_path))

    print("=== 主要指標 ===")
    print(f"総トレード数: {metrics['trades']}")
    print(f"勝率: {metrics['win_rate']*100:.2f}%")
    print(f"平均保持時間（勝ち）: {metrics['avg_hold_win_min']:.2f} 分")
    print(f"平均保持時間（負け）: {metrics['avg_hold_loss_min']:.2f} 分")
    print(f"平均勝ち額: {metrics['avg_win_yen']:.2f} 円")
    print(f"平均負け額: {metrics['avg_loss_yen']:.2f} 円")
    print(f"期待値（平均損益）: {metrics['expectancy_yen']:.2f} 円")
    print(f"プロフィットファクター: {metrics['profit_factor']:.2f}")
    print(f"最大ドローダウン: {metrics['max_drawdown_yen']:.2f} 円")
    print(f"出力PDF: {out_pdf}")
    print(f"出力PNG: {chart_png}")
    print(f"トレード明細CSV: {trades_out}")


if __name__ == "__main__":
    main()