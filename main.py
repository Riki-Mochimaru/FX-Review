"""CLI entrypoint for session-based FX/CFD OHLCV comparison report."""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, time
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
import yfinance as yf

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
    ap.add_argument("--input", default=None, help="Path of OHLCV CSV (optional when using yfinance)")
    ap.add_argument("--symbol", default=None, help="Display symbol/pair name")
    ap.add_argument("--ticker", default=None, help="yfinance ticker (ex: JPY=X, EURUSD=X)")
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


def _to_utc_naive(ts: pd.Timestamp) -> datetime:
    if ts.tzinfo is None:
        return ts.to_pydatetime()
    return ts.tz_convert("UTC").tz_localize(None).to_pydatetime()


def _resolve_fetch_window(
    dates: Sequence[date],
    session_start: time,
    session_end: time,
    timezone: Optional[str],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    d0 = min(dates)
    d1 = max(dates)
    start = pd.Timestamp(datetime.combine(d0, session_start))
    end = pd.Timestamp(datetime.combine(d1, session_end))
    if session_end <= session_start:
        end = end + pd.Timedelta(days=1)
    # Small margins to avoid endpoint truncation by provider.
    start = start - pd.Timedelta(minutes=10)
    end = end + pd.Timedelta(minutes=10)
    if timezone:
        start = start.tz_localize(timezone)
        end = end.tz_localize(timezone)
    return start, end


def fetch_ohlcv_yfinance(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timezone: Optional[str],
) -> pd.DataFrame:
    """Fetch OHLCV from yfinance. Try 1m first, fallback to 5m on empty/failure."""
    start_dt = _to_utc_naive(start)
    end_dt = _to_utc_naive(end)
    interval = "1m"
    raw = pd.DataFrame()
    for cand in ("1m", "5m"):
        try:
            raw = yf.download(
                tickers=ticker,
                start=start_dt,
                end=end_dt,
                interval=cand,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if not raw.empty:
                interval = cand
                break
        except Exception:
            continue
    if raw.empty:
        raise ValueError(f"Failed to fetch yfinance data for ticker={ticker} in range {start}..{end}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    col_map = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    out = raw.rename(columns=col_map).copy()
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in out.columns]
    out = out[keep]
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"]).sort_index()

    if timezone:
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC").tz_convert(timezone)
        else:
            out.index = out.index.tz_convert(timezone)

    LOGGER.info(
        "Loaded yfinance rows=%s ticker=%s interval=%s range=%s..%s",
        len(out),
        ticker,
        interval,
        out.index.min(),
        out.index.max(),
    )
    return out


def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    dates = resolve_dates(args.start_date, args.end_date, args.days, args.dates)
    if not dates:
        raise ValueError("No date targets selected.")

    if any(d is None for d in dates):
        raise ValueError("Some selected dates are invalid.")

    session_start = _to_time(args.session_start)
    session_end = _to_time(args.session_end)

    if args.input:
        df = load_ohlcv_csv(
            args.input,
            symbol=args.symbol,
            timezone=args.timezone,
        )
    else:
        ticker = args.ticker or args.symbol or "JPY=X"
        start, end = _resolve_fetch_window(dates, session_start, session_end, args.timezone)
        df = fetch_ohlcv_yfinance(ticker=ticker, start=start, end=end, timezone=args.timezone)
        if args.symbol is None:
            args.symbol = ticker

    if df.empty:
        raise ValueError("Input data is empty after filtering.")

    config = ReportConfig(
        symbol=args.symbol,
        session_start=session_start,
        session_end=session_end,
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
