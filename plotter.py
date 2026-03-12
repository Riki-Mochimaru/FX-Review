"""Plot helpers for daily session comparison charts."""

from __future__ import annotations

import logging
from datetime import time
from typing import Dict, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def configure_matplotlib_style(style: str = "white") -> None:
    """Set matplotlib style and font settings with JP-friendly defaults."""
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.grid.axis"] = "both"
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = [
        "Hiragino Kaku Gothic ProN",
        "Yu Gothic",
        "Meiryo",
        "Noto Sans CJK JP",
        "IPAexGothic",
        "DejaVu Sans",
        "sans-serif",
    ]
    plt.rcParams["axes.facecolor"] = "white"
    try:
        plt.style.use(style)
    except Exception:
        LOGGER.warning("matplotlib style not found, fallback to default: %s", style)


def _session_minutes(start: time, end: time) -> int:
    start_min = start.hour * 60 + start.minute
    end_min = end.hour * 60 + end.minute
    span = end_min - start_min
    if span <= 0:
        span += 24 * 60
    return span


def _format_session_minute(x: float, pos: object, start: time) -> str:  # noqa: ARG002
    total = int(round(x))
    if total < 0:
        total = 0
    mins = total % (24 * 60)
    base_hour = start.hour * 60 + start.minute
    label_min = (base_hour + mins) % (24 * 60)
    hh = label_min // 60
    mm = label_min % 60
    return f"{hh:02d}:{mm:02d}"


def _format_ohlc_stat(stat: Dict[str, float]) -> str:
    return (
        f"日付: {stat['date']}  始値: {stat['open']:.5f}  "
        f"高値: {stat['high']:.5f}  安値: {stat['low']:.5f}  "
        f"終値: {stat['close']:.5f}  リターン: {stat['ret']:+.2f}%"
    )


def compute_session_stats(df: pd.DataFrame, date_label: str) -> Dict[str, float]:
    if df.empty:
        return {
            "date": date_label,
            "open": float("nan"),
            "high": float("nan"),
            "low": float("nan"),
            "close": float("nan"),
            "ret": float("nan"),
        }
    open_p = float(df["open"].iloc[0])
    close_p = float(df["close"].iloc[-1])
    high_p = float(df["high"].max())
    low_p = float(df["low"].min())
    ret = (close_p / open_p - 1.0) * 100.0 if open_p != 0 else float("nan")
    return {
        "date": date_label,
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "close": close_p,
        "ret": ret,
    }


def draw_candles(ax: plt.Axes, df: pd.DataFrame, bar_minutes: int) -> None:
    if df.empty:
        return
    if "offset_min" not in df.columns:
        raise ValueError("draw_candles requires 'offset_min' column")

    x = df["offset_min"].to_numpy(dtype=float)
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    width = max(bar_minutes * 0.75, 0.2)

    for xi, o, h, l, c in zip(x, opens, highs, lows, closes):
        color = "#d62728" if c >= o else "#2ca02c"
        ax.vlines(xi, l, h, color=color, linewidth=0.8)
        y0 = min(o, c)
        h_body = max(abs(c - o), 1e-9)
        rect = plt.Rectangle((xi - width / 2.0, y0), width, h_body, edgecolor=color, facecolor=color, alpha=0.75)
        ax.add_patch(rect)


def draw_volume(ax: plt.Axes, df: plt.Axes, bar_minutes: int, scale: float = 0.4) -> None:
    if df.empty or "volume" not in df.columns:
        return
    if "offset_min" not in df.columns:
        raise ValueError("draw_volume requires 'offset_min' column")
    vol_ax = ax.twinx()
    x = df["offset_min"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)
    v = np.nan_to_num(v, nan=0.0, nan=0.0)
    width = max(bar_minutes * 0.6, 0.1)
    vol_ax.bar(x, v, width=width, alpha=0.35, color="tab:blue", align="center")
    max_v = float(np.nanmax(v))
    if max_v > 0:
        current = ax.get_ylim()
        vol_ax.set_ylim(0, max_v / scale if scale > 0 else max_v * 1.0)
        if not np.isfinite(current[0]) or not np.isfinite(current[1]):
            pass
    vol_ax.set_yticks([])
    vol_ax.set_ylabel("Volume", fontsize=7)


def apply_shared_x_axis_style(ax: plt.Axes, session_start: time, duration: int, show_labels: bool) -> None:
    ax.set_xlim(0, duration)
    major = max(5, duration // 12)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(major))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: _format_session_minute(v, p, session_start)))
    if show_labels:
        ax.set_xlabel("時刻", fontsize=8)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)


def add_day_panel(
    ax: plt.Axes,
    date_label: str,
    df: pd.DataFrame,
    timeframe_minutes: int,
    session_start: time,
    session_end: time,
    show_volume: bool = False,
    fixed_ylim: Optional[Tuple[float, float]] = None,
    show_xlabels: bool = True,
) -> None:
    duration = _session_minutes(session_start, session_end)

    if df.empty:
        ax.text(0.5, 0.5, "データなし", ha="center", va="center", fontsize=12)
        apply_shared_x_axis_style(ax, session_start, duration, show_xlabels)
        ax.set_title(date_label, fontsize=9)
        return

    x = df["offset_min"]
    bar_min = max(timeframe_minutes, 1)

    if "volume" in df.columns and show_volume:
        draw_volume(ax, df, bar_min)

    draw_candles(ax, df, bar_min)
    stats = compute_session_stats(df, date_label)
    ax.set_title(f"{stats['date']} | リターン {stats['ret']:+.2f}%")
    ax.text(
        0.01,
        0.98,
        _format_ohlc_stat(stats),
        fontsize=7,
        va="top",
        ha="left",
        transform=ax.transAxes,
        bbox={"facecolor": "white", "alpha": 0.7, "pad": 2},
    )

    if fixed_ylim is not None:
        ymin, ymax = fixed_ylim
        if ymin < ymax and np.isfinite(ymin) and np.isfinite(ymax):
            ax.set_ylim(ymin, ymax)

    apply_shared_x_axis_style(ax, session_start, duration, show_xlabels)
