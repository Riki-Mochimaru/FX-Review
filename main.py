"""CLI entrypoint for session-based FX/CFD OHLCV comparison report."""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, time
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from loader import load_ohlcv_csv
from pdf_report import ReportConfig, generate_session_report


LOGGER = logging.getLogger(__name__)


def _to_time(value: str) -> time:
    try:
        t = datetime.strptime(value.strip(), "%H:%M").time()
    except ValueError as exc:
        raise ValueError(f"Invalid time format: {value}. Expected HH:MM.") from exc
    return t


def _to_date(value: str) -> date:
    try:
        return pd.Timestamp(value).date()
    except Exception as exc:
        raise ValueError(f"Invalid date format: {value}. Use YYYY-MM-DD.") from exc


def resolve_dates(
    start_date: Optional[str],
    end_date: Optional[str],
    days: Optional[int],
    explicit_dates: Optional[Sequence[str]],
) -> List[date]:
    if explicit_dates:
        dates = [_to_date(d) for d in explicit_dates]
        if not dates:
            raise ValueError("--dates was given but empty.")
        uniq = sorted(set(dates))
        return [d for d in uniq]

    if start_date is not None and end_date is not None:
        rng = pd.bdate_range(start=_to_date(start_date), end=_to_date(end_date))
        return [d.to_pydatetime().date() for d in rng]

    if start_date is not None and days is not None:
        start = pd.Timestamp(_to_date(start_date))
        rng = pd.bdate_range(start=start, periods=days)
        return [d.to_pydatetime().date() for d in rng]

    if end_date is not None and days is not None:
        end = pd.Timestamp(_to_date(end_date))
        rng = pd.bdate_range(end=end, periods=days)
        return [d.to_pydatetime().date() for d in rng]

    if end_date is not None:
        return [_to_date(end_date)]

    if start_date is not None:
        return [_to_date(start_date)]

    if days is None:
        days = 20
    end = pd.Timestamp.now().normalize()
    rng = pd.bdate_range(end=end, periods=days)
    return [d.to_pydatetime().date() for d in rng]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate daily session comparison PDF.")
    ap.add_argument("--input", required=True, help="Path of OHLCV CSV")
    ap.add_argument("--symbol", default=None, help="Filter symbol/pair")
    ap.add_argument("--start-date", default=None, help="Range start date YYYY-MM-DD")
    ap.add_argument("--end-date", default=None, help="Range end date YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=None, help="Number of business days to use")
    ap.add_argument("--dates", nargs="+", help="Explicit date list YYYY-MM-DD")
    ap.add_argument("--session-start", default="09:00", help="Session start HH:MM")
    ap.add_argument("--session-end", default="10:30", help="Session end HH:MM")
    ap.add_argument(
        "--timeframes",
        nargs="+",
        default=["1", "5", "15"],
        help="Timeframes in minutes: ex: 1 5 15",
    )
    ap.add_argument("--timezone", default=None, help="Optional timezone for datetime normalization")
    ap.add_argument("--output", default="out/session_report.pdf", help="Output PDF path")
    ap.add_argument("--rows", type=int, default=2, help="Subplot rows per page")
    ap.add_argument("--cols", type=int, default=2, help="Subplot cols per page")
    ap.add_argument("--show-volume", action="store_true", help="Show volume bars")
    ap.add_argument("--style", default="default", help="Matplotlib style name")
    ap.add_argument("--orientation", choices=["portrait", "landscape"], default="portrait", help="A4 page orientation")
    ap.add_argument("--fixed-ymin", type=float, default=None, help="Fixed y-axis min (optional)")
    ap.add_argument("--fixed-ymax", type=float, default=None, help="Fixed y-axis max (optional)")
    ap.add_argument("--dpi", type=int, default=180, help="PDF image dpi")
    ap.add_argument("--log-level", default="INFO", help="Log level")
    return ap.parse_args()


def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    dates = resolve_dates(args.start_date, args.end_date, args.days, args.dates)
    if not dates:
        raise ValueError("No date targets selected.")

    if any(d is None for d in dates):
        raise ValueError("Some selected dates are invalid.")

    df = load_ohlcv_csv(
        args.input,
        symbol=args.symbol,
        timezone=args.timezone,
    )

    if df.empty:
        raise ValueError("Input data is empty after filtering.")

    # Keep original timezone; timezone mismatch handled at data slice step.
    config = ReportConfig(
        symbol=args.symbol,
        session_start=_to_time(args.session_start),
        session_end=_to_time(args.session_end),
        rows=args.rows,
        cols=args.cols,
        show_volume=args.show_volume,
        fixed_ylim=(args.fixed_ymin, args.fixed_ymax) if args.fixed_ymin is not None and args.fixed_ymax is not None else None,
        style=args.style,
        orientation=args.orientation,
        dpi=args.dpi,
    )

    generate_session_report(df, dates, args.timeframes, args.output, config)
    print(f"[INFO] Saved PDF: {Path(args.output).resolve()}")
    return 0


def main() -> None:
    args = parse_args()
    code = _run(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
