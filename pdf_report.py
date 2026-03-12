"""PDF report generation for daily session comparison."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as date_type
from datetime import datetime, time
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pandas as pd

from plotter import add_day_panel, configure_matplotlib_style
from resample import timeframe_to_minutes

LOGGER = logging.getLogger(__name__)

A4_PORTRAIT = (8.267, 11.693)
A4_LANDSCAPE = (11.693, 8.267)


@dataclass(frozen=True)
class ReportConfig:
    symbol: Optional[str]
    session_start: time
    session_end: time
    rows: int
    cols: int
    show_volume: bool = False
    fixed_ylim: Optional[Tuple[float, float]] = None
    style: str = "default"
    orientation: str = "portrait"
    dpi: int = 180


def _combine_session_datetime(d: date_type, tm: time, tz) -> pd.Timestamp:
    base = datetime.combine(d, tm)
    ts = pd.Timestamp(base)
    if tz is None:
        return ts
    return ts.tz_localize(tz)


def _split_chunks(items: Sequence[date_type], n: int) -> Iterable[Sequence[date_type]]:
    if n <= 0:
        raise ValueError("chunk size must be > 0")
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _session_minutes(start: time, end: time) -> int:
    s = start.hour * 60 + start.minute
    e = end.hour * 60 + end.minute
    span = e - s
    if span <= 0:
        span += 24 * 60
    return span


def _slice_day_session(
    df: pd.DataFrame,
    d: date_type,
    session_start: time,
    session_end: time,
) -> pd.DataFrame:
    tz = df.index.tz
    start = _combine_session_datetime(d, session_start, tz)
    end = _combine_session_datetime(d, session_end, tz)
    if end <= start:
        end = end + pd.Timedelta(days=1)
    out = df.loc[(df.index >= start) & (df.index <= end)].copy()
    if out.empty:
        return out
    minutes = (out.index - start).total_seconds() / 60.0
    if session_end <= session_start:
        minutes = minutes.where(minutes >= 0, minutes + 24 * 60)
    out["offset_min"] = minutes
    return out


def generate_session_report(
    df: pd.DataFrame,
    dates: Sequence[date_type],
    timeframes: Sequence[str],
    output: str | Path,
    config: ReportConfig,
) -> None:
    """
    Generate PDF with session-wise daily comparisons.

    One page is created per timeframe for each block of rows*cols days.
    """
    if df.empty:
        raise ValueError("No OHLCV data for report.")
    if config.rows <= 0 or config.cols <= 0:
        raise ValueError("rows/cols must be positive.")

    configure_matplotlib_style(config.style)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    page_size = A4_LANDSCAPE if config.orientation == "landscape" else A4_PORTRAIT

    with PdfPages(str(output)) as pdf:
        per_page = config.rows * config.cols
        for tf in timeframes:
            tf_minutes = timeframe_to_minutes(tf)
            tf_df = df.copy()
            # Expect pre-resampled df can be passed here; keep function generic.
            if tf != "original":
                # If caller passes raw data, re-resample inside report generation.
                tf_df = (
                    tf_df[["open", "high", "low", "close"]]
                    .resample(f"{tf_minutes}min", label="right", closed="right")
                    .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
                    .dropna(subset=["open", "high", "low", "close"])
                )
                if "volume" in df.columns:
                    tf_df["volume"] = tf_df["volume"].resample(f"{tf_minutes}min", label="right", closed="right").sum(
                        min_count=1
                    )
            if "volume" in tf_df.columns:
                tf_df = tf_df.dropna(subset=["open", "high", "low", "close", "volume"])

            if tf_df.empty:
                LOGGER.warning("Skipping empty timeframe: %s", tf)
                continue

            duration = _session_minutes(config.session_start, config.session_end)
            title_tf = f"Timeframe={tf}分"

            for page_idx, date_block in enumerate(_split_chunks(dates, per_page), start=1):
                fig, axs = plt.subplots(
                    config.rows,
                    config.cols,
                    figsize=page_size,
                    squeeze=False,
                    sharex=True,
                )
                fig.suptitle(
                    f"{config.symbol or 'All'}  {title_tf}  {config.session_start:%H:%M}-{config.session_end:%H:%M}  "
                    f"Page {page_idx}",
                    fontsize=11,
                )

                flat_axes = axs.ravel()
                for slot, axis in enumerate(flat_axes):
                    col = slot % config.cols
                    row = slot // config.cols
                    if slot >= len(date_block):
                        axis.set_visible(False)
                        continue
                    day = date_block[slot]
                    day_df = _slice_day_session(tf_df, day, config.session_start, config.session_end)
                    is_bottom = row == config.rows - 1
                    add_day_panel(
                        ax=axis,
                        date_label=str(day),
                        df=day_df,
                        timeframe_minutes=tf_minutes,
                        session_start=config.session_start,
                        session_end=config.session_end,
                        show_volume=config.show_volume,
                        fixed_ylim=config.fixed_ylim,
                        show_xlabels=is_bottom,
                    )
                    axis.set_xlabel("時間" if is_bottom else "")

                    if not day_df.empty:
                        axis.set_title(
                            f"{day}  (高:{day_df['high'].max():.5f}  安:{day_df['low'].min():.5f}  "
                            f"始:{day_df['open'].iloc[0]:.5f}  終:{day_df['close'].iloc[-1]:.5f})",
                            fontsize=8,
                        )
                plt.tight_layout(rect=[0.0, 0.02, 1.0, 0.96])
                pdf.savefig(fig, dpi=config.dpi)
                plt.close(fig)
                LOGGER.info("Written page: tf=%s date_block=%s", tf, date_block)

    LOGGER.info("Saved PDF: %s", output)
