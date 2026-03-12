# Daily Session OHLCV Comparison Report

Generate a PDF report that compares the same intraday session across multiple days for FX/CFD data.

- Works directly with `yfinance` (CSV is optional)
- Supports multiple timeframes per run (e.g. `1 5 15`)
- Handles missing-session days without crashing
- A4 portrait/landscape output with subplot grid

## Files

- `main.py`: CLI entrypoint
- `loader.py`: CSV loading and encoding detection
- `resample.py`: timeframe conversion helpers
- `plotter.py`: candlestick/session panel drawing
- `pdf_report.py`: page composition and PDF export
- `requirements.txt`: dependencies

## Expected CSV Format (Optional Input Mode)

If you use `--input`, the CSV should include at least:

- `datetime` (or `time`, `timestamp`, `date`; Japanese datetime column aliases are also supported)
- `open`
- `high`
- `low`
- `close`
- `volume` is optional

Encoding is auto-detected (`UTF-8`, `CP932`, `Shift_JIS`, etc.).

## Usage

### 1) yfinance only (no CSV)

```bash
python main.py \
  --ticker JPY=X \
  --days 20 \
  --session-start 09:00 \
  --session-end 10:30 \
  --timeframes 1 5 15 \
  --timezone Asia/Tokyo \
  --rows 2 \
  --cols 2 \
  --show-volume \
  --output out/session_report.pdf
```

### 2) Cross-midnight session (e.g. 21:00 to 01:00)

```bash
python main.py \
  --ticker JPY=X \
  --days 20 \
  --session-start 21:00 \
  --session-end 01:00 \
  --timeframes 1 5 15 \
  --timezone Asia/Tokyo \
  --output out/session_report_night.pdf
```

### 3) CSV input mode

```bash
python main.py \
  --input data/ohlcv.csv \
  --symbol USDJPY \
  --start-date 2026-02-01 \
  --end-date 2026-03-01 \
  --session-start 09:00 \
  --session-end 10:30 \
  --timeframes 1 5 15 \
  --timezone Asia/Tokyo \
  --output out/session_report.pdf
```

### 4) Explicit date list mode

```bash
python main.py \
  --ticker EURUSD=X \
  --dates 2026-03-01 2026-03-05 2026-03-06 \
  --session-start 09:00 \
  --session-end 10:30 \
  --timeframes 1 5 15
```

## Key Arguments

- `--input`: optional CSV path
- `--ticker`: yfinance ticker when `--input` is omitted
- `--symbol`: display symbol name
- `--start-date`, `--end-date`: date range
- `--days`: last N business days
- `--dates`: explicit date list
- `--session-start`, `--session-end`: session time window
- `--timeframes`: list of minute timeframes
- `--timezone`: timezone (e.g. `Asia/Tokyo`)
- `--output`: output PDF path
- `--rows`, `--cols`: subplot layout per page
- `--show-volume`: overlay volume bars
- `--style`: matplotlib style
- `--orientation`: `portrait` or `landscape`
- `--fixed-ymin`, `--fixed-ymax`: fixed y-axis scale

## Notes

- yfinance 1-minute history has period limits; fallback to `5m` is implemented.
- Days with no data in the target session are rendered as `No data`.
- Cross-midnight sessions are supported (e.g. `22:00` to `01:00`).

## Extension Ideas

- Overlay entry/exit points on each day panel
- Add return-distribution summary pages
- Add average intraday path per session
