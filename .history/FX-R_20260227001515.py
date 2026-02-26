#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
transactions.csv（取引履歴）から、勝率、平均保持時間（勝ち／負け）、平均損益（勝ち／負け）などを集計し、
1枚で見られるPDFレポート（＋PNG）を出力するスクリプト。

想定フォーマット（例）
- チケット番号, 種別, 日時 (JST), 銘柄, 売買, 数量, 約定価格, …, 取引損益/入出金金額(円)
- 種別に「新規取引」「決済取引」「入金」などが含まれる
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader


JP_COL_CANDIDATES = {
    "ticket": ["チケット番号", "ticket", "Ticket", "注文番号", "id"],
    "type": ["種別", "type", "Type"],
    "time": ["日時 (JST)", "日時", "約定日時", "time", "Time", "timestamp", "Timestamp"],
    "symbol": ["銘柄", "通貨ペア", "symbol", "Symbol", "instrument"],
    "side": ["売買", "side", "Side", "方向", "buy/sell"],
    "qty": ["数量", "取引数量", "units", "qty", "Quantity"],
    "pnl": ["取引損益/入出金金額(円)", "損益(円)", "損益", "pnl", "PnL", "profit", "Profit"],
}

ENTRY_KEYWORDS = ["新規", "OPEN", "ENTRY", "建", "買建", "売建"]
EXIT_KEYWORDS = ["決済", "CLOSE", "EXIT", "落", "手仕舞い"]


@dataclass
class Trade:
    ticket: str
    symbol: str
    side: str
    qty: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    hold_seconds: float
    pnl: float


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _contains_any(s: str, keywords: list[str]) -> bool:
    s_up = s.upper()
    for k in keywords:
        if k.upper() in s_up:
            return True
    return False


def load_and_parse_trades(csv_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)

    col_ticket = _pick_col(df, JP_COL_CANDIDATES["ticket"])
    col_type = _pick_col(df, JP_COL_CANDIDATES["type"])
    col_time = _pick_col(df, JP_COL_CANDIDATES["time"])
    col_symbol = _pick_col(df, JP_COL_CANDIDATES["symbol"])
    col_side = _pick_col(df, JP_COL_CANDIDATES["side"])
    col_qty = _pick_col(df, JP_COL_CANDIDATES["qty"])
    col_pnl = _pick_col(df, JP_COL_CANDIDATES["pnl"])

    missing = [k for k, v in {
        "ticket": col_ticket, "type": col_type, "time": col_time,
        "symbol": col_symbol, "side": col_side, "qty": col_qty, "pnl": col_pnl
    }.items() if v is None]
    if missing:
        raise ValueError(
            f"必要そうな列が見つかりませんでした: {missing}\n"
            f"検出した列: {list(df.columns)}"
        )

    df = df.copy()
    df[col_time] = pd.to_datetime(df[col_time], errors="coerce")
    df[col_ticket] = df[col_ticket].astype(str)

    # 取引系のみ抽出（入出金などを除外）
    df["_type_s"] = df[col_type].map(_normalize_str)
    df["_is_entry"] = df["_type_s"].map(lambda s: _contains_any(s, ENTRY_KEYWORDS))
    df["_is_exit"] = df["_type_s"].map(lambda s: _contains_any(s, EXIT_KEYWORDS))

    tx = df[df["_is_entry"] | df["_is_exit"]].copy()

    # 1チケット内で、最初の新規をentry、最後の決済をexit、pnlは決済行の合計
    trades: list[Trade] = []
    for ticket, g in tx.groupby(col_ticket, sort=False):
        g = g.sort_values(col_time)
        ge = g[g["_is_entry"]]
        gx = g[g["_is_exit"]]
        if ge.empty or gx.empty:
            continue

        entry_row = ge.iloc[0]
        exit_row = gx.iloc[-1]

        entry_time = entry_row[col_time]
        exit_time = exit_row[col_time]
        if pd.isna(entry_time) or pd.isna(exit_time):
            continue

        hold = (exit_time - entry_time).total_seconds()
        # 決済が複数に分かれるケースを想定して合計
        pnl_sum = pd.to_numeric(gx[col_pnl], errors="coerce").fillna(0.0).sum()

        symbol = _normalize_str(entry_row[col_symbol]) or _normalize_str(exit_row[col_symbol])
        side = _normalize_str(entry_row[col_side]) or _normalize_str(exit_row[col_side])
        qty = pd.to_numeric(entry_row[col_qty], errors="coerce")
        qty = float(qty) if not pd.isna(qty) else float(pd.to_numeric(exit_row[col_qty], errors="coerce") or 0.0)

        trades.append(Trade(
            ticket=ticket,
            symbol=symbol,
            side=side,
            qty=qty,
            entry_time=entry_time,
            exit_time=exit_time,
            hold_seconds=float(hold),
            pnl=float(pnl_sum),
        ))

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    if trades_df.empty:
        raise ValueError("新規取引と決済取引のペアが見つからず、トレードを構成できませんでした。")

    trades_df["is_win"] = trades_df["pnl"] > 0
    trades_df["hold_min"] = trades_df["hold_seconds"] / 60.0
    trades_df = trades_df.sort_values("exit_time").reset_index(drop=True)

    return df, trades_df


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
    sum_loss = float(losses["pnl"].sum())  # 負のはず
    profit_factor = (sum_win / abs(sum_loss)) if sum_loss < 0 else float("inf") if sum_win > 0 else float("nan")

    # 連勝連敗
    streak = (trades["is_win"].astype(int).replace({0: -1})).to_numpy()
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

    # エクイティと最大DD
    equity = pnl.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    max_dd = float(dd.min())

    # 取引頻度（日次）
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

    # 1枚に4パネル
    fig = plt.figure(figsize=(8.27, 11.69))  # A4縦（inch）
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
    # 勝ち／負けの保持時間分布を同一図に
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

    # タイトル
    c.setFont("Helvetica-Bold", 16)
    c.drawString(20 * mm, h - 20 * mm, "取引レポート（1枚）")
    c.setFont("Helvetica", 10)
    c.drawString(20 * mm, h - 26 * mm, f"入力: {src_name}")

    # 数値サマリー
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

    # 図（A4に収める）
    img = ImageReader(str(chart_png))
    img_w = w - 40 * mm
    img_h = 125 * mm
    c.drawImage(img, 20 * mm, 20 * mm, width=img_w, height=img_h, preserveAspectRatio=True, anchor='sw')

    c.showPage()
    c.save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="transactions.csv のパス")
    ap.add_argument("--outdir", type=str, default="out_report", help="出力先ディレクトリ")
    ap.add_argument("--prefix", type=str, default="trade_report", help="出力ファイルの接頭辞")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _, trades = load_and_parse_trades(csv_path)
    metrics = compute_metrics(trades)

    # 取引一覧も保存（後で検算しやすい）
    trades_out = outdir / f"{args.prefix}_trades.csv"
    trades.to_csv(trades_out, index=False, encoding="utf-8-sig")

    chart_png = outdir / f"{args.prefix}_chart.png"
    make_figures(trades, chart_png)

    out_pdf = outdir / f"{args.prefix}.pdf"
    make_pdf_report(metrics, chart_png, out_pdf, src_name=str(csv_path))

    # コンソールにも主要指標を出力
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
    print(f"出力: {out_pdf}")
    print(f"図: {chart_png}")
    print(f"トレード一覧: {trades_out}")


if __name__ == "__main__":
    main()
    