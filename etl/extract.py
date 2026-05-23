"""
extract.py - Finnhub stock market extraction.

The project keeps the existing long-format pipeline contract so downstream
Spark, warehouse, and ML code can keep working:
    country_code -> stock symbol
    country_name -> company display name
    indicator    -> open_price / high_price / low_price / close_price / latest_price
    timeframe    -> day / week / month / year period granularity
    time_index   -> date index string (YYYY-MM-DD)
    value        -> numeric price
    source       -> source label
"""

from __future__ import annotations

import datetime as dt
import os
import re
import warnings
from dataclasses import dataclass

import pandas as pd
import requests

warnings.filterwarnings("ignore")

_session = requests.Session()

FINNHUB_API_BASE = "https://finnhub.io/api/v1"
FINNHUB_SOURCE = "Finnhub"

DEFAULT_SYMBOLS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "JPM",
    "BAC",
    "V",
    "JNJ",
    "UNH",
    "XOM",
]

DEFAULT_METADATA = {
    "AAPL": {"name": "Apple Inc", "category": "Technology"},
    "MSFT": {"name": "Microsoft Corp", "category": "Technology"},
    "NVDA": {"name": "NVIDIA Corp", "category": "Technology"},
    "AMZN": {"name": "Amazon.com Inc", "category": "Consumer Cyclical"},
    "GOOGL": {"name": "Alphabet Inc", "category": "Communication Services"},
    "META": {"name": "Meta Platforms Inc", "category": "Communication Services"},
    "TSLA": {"name": "Tesla Inc", "category": "Consumer Cyclical"},
    "JPM": {"name": "JPMorgan Chase & Co", "category": "Financial Services"},
    "BAC": {"name": "Bank of America Corp", "category": "Financial Services"},
    "V": {"name": "Visa Inc", "category": "Financial Services"},
    "JNJ": {"name": "Johnson & Johnson", "category": "Healthcare"},
    "UNH": {"name": "UnitedHealth Group Inc", "category": "Healthcare"},
    "XOM": {"name": "Exxon Mobil Corp", "category": "Energy"},
}


@dataclass(frozen=True)
class PriceSnapshot:
    symbol: str
    stock_name: str
    source: str
    timeframe: str
    time_index: str
    open_price: float | None = None
    high_price: float | None = None
    low_price: float | None = None
    close_price: float | None = None
    latest_price: float | None = None


def _api_key() -> str:
    return (
        os.getenv("FINNHUB_API_KEY")
        or os.getenv("FINNHUB_TOKEN")
        or os.getenv("API_KEY")
        or ""
    ).strip()


def _configured_symbols() -> list[str]:
    raw = os.getenv("FINNHUB_SYMBOLS") or os.getenv("STOCK_SYMBOLS") or ""
    if not raw.strip():
        return DEFAULT_SYMBOLS
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def _history_years() -> int:
    try:
        return max(1, min(20, int(os.getenv("FINNHUB_HISTORY_YEARS", "5"))))
    except ValueError:
        return 5


def _request_json(path: str, params: dict[str, object] | None = None) -> dict:
    key = _api_key()
    if not key:
        raise RuntimeError("FINNHUB_API_KEY is not set")

    query = dict(params or {})
    query["token"] = key
    resp = _session.get(f"{FINNHUB_API_BASE}{path}", params=query, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(str(data["error"]))
    if not isinstance(data, dict):
        raise ValueError("Finnhub response is not a JSON object")
    return data


def _safe_error(exc: Exception) -> str:
    return re.sub(r"token=[^&)\s]+", "token=***", str(exc))


def _to_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metadata_for_symbol(symbol: str) -> dict[str, str]:
    fallback = DEFAULT_METADATA.get(symbol, {"name": symbol, "category": "Equity"}).copy()
    try:
        data = _request_json("/stock/profile2", {"symbol": symbol})
    except Exception as exc:
        print(f"  [warn] profile unavailable for {symbol}; using defaults. Reason: {_safe_error(exc)}")
        return fallback

    name = str(data.get("name") or data.get("ticker") or fallback["name"])
    category = str(data.get("finnhubIndustry") or data.get("exchange") or fallback["category"])
    return {"name": name, "category": category}


def _metadata_from_api(symbols: list[str]) -> dict[str, dict[str, str]]:
    return {symbol: _metadata_for_symbol(symbol) for symbol in symbols}


def _snapshot_to_rows(snapshot: PriceSnapshot) -> list[dict[str, object]]:
    metrics = {
        "open_price": snapshot.open_price,
        "high_price": snapshot.high_price,
        "low_price": snapshot.low_price,
        "close_price": snapshot.close_price,
        "latest_price": snapshot.latest_price,
    }
    rows = []
    for indicator, value in metrics.items():
        if value is None:
            continue
        rows.append({
            "country_code": snapshot.symbol,
            "country_name": snapshot.stock_name,
            "indicator": indicator,
            "timeframe": snapshot.timeframe,
            "time_index": snapshot.time_index,
            "value": float(value),
            "source": snapshot.source,
        })
    return rows


def _snapshot_from_quote(symbol: str, quote: dict, metadata: dict[str, dict[str, str]]) -> PriceSnapshot | None:
    current = _to_float(quote.get("c"))
    if current is None or current == 0:
        return None

    meta = metadata.get(symbol, DEFAULT_METADATA.get(symbol, {"name": symbol}))
    today_str = dt.date.today().isoformat()
    return PriceSnapshot(
        symbol=symbol,
        stock_name=str(meta.get("name") or symbol),
        source=FINNHUB_SOURCE,
        timeframe="day",
        time_index=today_str,
        open_price=_to_float(quote.get("o")),
        high_price=_to_float(quote.get("h")),
        low_price=_to_float(quote.get("l")),
        close_price=_to_float(quote.get("pc")) or current,
        latest_price=current,
    )


def _aggregate_snapshots_from_candles(
    symbol: str,
    candles: dict,
    metadata: dict[str, dict[str, str]],
) -> list[PriceSnapshot]:
    if candles.get("s") != "ok":
        return []

    required = ["t", "o", "h", "l", "c"]
    if not all(isinstance(candles.get(key), list) for key in required):
        return []

    rows = []
    for ts, open_price, high_price, low_price, close_price in zip(
        candles["t"], candles["o"], candles["h"], candles["l"], candles["c"]
    ):
        rows.append({
            "date": dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc).date(),
            "open": _to_float(open_price),
            "high": _to_float(high_price),
            "low": _to_float(low_price),
            "close": _to_float(close_price),
        })

    df = pd.DataFrame(rows).dropna(subset=["close"])
    if df.empty:
        return []

    meta = metadata.get(symbol, DEFAULT_METADATA.get(symbol, {"name": symbol}))
    stock_name = str(meta.get("name") or symbol)
    snapshots: list[PriceSnapshot] = []

    # 1. Day granularity (All trading days)
    for _, row in df.iterrows():
        snapshots.append(PriceSnapshot(
            symbol=symbol,
            stock_name=stock_name,
            source=FINNHUB_SOURCE,
            timeframe="day",
            time_index=row["date"].isoformat(),
            open_price=row["open"],
            high_price=row["high"],
            low_price=row["low"],
            close_price=row["close"],
            latest_price=row["close"],
        ))

    # 2. Week granularity (group by Monday of ISO week)
    df_week = df.copy()
    df_week["week_start"] = df_week["date"].apply(lambda d: d - dt.timedelta(days=d.weekday()))
    for w_start, grp in df_week.sort_values("date").groupby("week_start"):
        ordered = grp.reset_index(drop=True)
        close = float(ordered["close"].iloc[-1])
        snapshots.append(PriceSnapshot(
            symbol=symbol,
            stock_name=stock_name,
            source=FINNHUB_SOURCE,
            timeframe="week",
            time_index=w_start.isoformat(),
            open_price=_to_float(ordered["open"].iloc[0]),
            high_price=_to_float(ordered["high"].max()),
            low_price=_to_float(ordered["low"].min()),
            close_price=close,
            latest_price=close,
        ))

    # 3. Month granularity (group by 1st of month)
    df_month = df.copy()
    df_month["month_start"] = df_month["date"].apply(lambda d: d.replace(day=1))
    for m_start, grp in df_month.sort_values("date").groupby("month_start"):
        ordered = grp.reset_index(drop=True)
        close = float(ordered["close"].iloc[-1])
        snapshots.append(PriceSnapshot(
            symbol=symbol,
            stock_name=stock_name,
            source=FINNHUB_SOURCE,
            timeframe="month",
            time_index=m_start.isoformat(),
            open_price=_to_float(ordered["open"].iloc[0]),
            high_price=_to_float(ordered["high"].max()),
            low_price=_to_float(ordered["low"].min()),
            close_price=close,
            latest_price=close,
        ))

    # 4. Year granularity (group by Jan 1st of year)
    df_year = df.copy()
    df_year["year_start"] = df_year["date"].apply(lambda d: d.replace(month=1, day=1))
    for y_start, grp in df_year.sort_values("date").groupby("year_start"):
        ordered = grp.reset_index(drop=True)
        close = float(ordered["close"].iloc[-1])
        snapshots.append(PriceSnapshot(
            symbol=symbol,
            stock_name=stock_name,
            source=FINNHUB_SOURCE,
            timeframe="year",
            time_index=y_start.isoformat(),
            open_price=_to_float(ordered["open"].iloc[0]),
            high_price=_to_float(ordered["high"].max()),
            low_price=_to_float(ordered["low"].min()),
            close_price=close,
            latest_price=close,
        ))

    return snapshots


def _projected_history_from_quote(
    symbol: str,
    quote: dict,
    metadata: dict[str, dict[str, str]],
    history_years: int,
) -> list[PriceSnapshot]:
    """Create model-ready multi-timeframe rows when the API key cannot access candles."""
    current_day = _snapshot_from_quote(symbol, quote, metadata)
    if current_day is None:
        return []

    base = current_day.latest_price or current_day.close_price or current_day.open_price
    if base is None:
        return []

    stock_name = current_day.stock_name
    seed = sum(ord(ch) for ch in symbol)
    snapshots: list[PriceSnapshot] = []

    today = dt.date.today()

    # Generate daily: 90 days of history
    daily_step = ((seed % 9) - 4) * 0.001
    for idx in range(90, 0, -1):
        past_date = today - dt.timedelta(days=idx)
        close = max(0.01, float(base) / ((1 + daily_step) ** idx))
        open_p = close * (0.99 + (seed % 5) * 0.004)
        snapshots.append(PriceSnapshot(
            symbol=symbol,
            stock_name=stock_name,
            source="Finnhub Quote Projection (Daily)",
            timeframe="day",
            time_index=past_date.isoformat(),
            open_price=open_p,
            high_price=max(open_p, close) * 1.008,
            low_price=min(open_p, close) * 0.992,
            close_price=close,
            latest_price=close,
        ))

    # Generate weekly: 104 weeks of history
    weekly_step = ((seed % 9) - 4) * 0.005
    for idx in range(104, 0, -1):
        past_date = today - dt.timedelta(weeks=idx)
        w_start = past_date - dt.timedelta(days=past_date.weekday())
        close = max(0.01, float(base) / ((1 + weekly_step) ** idx))
        open_p = close * (0.98 + (seed % 5) * 0.008)
        snapshots.append(PriceSnapshot(
            symbol=symbol,
            stock_name=stock_name,
            source="Finnhub Quote Projection (Weekly)",
            timeframe="week",
            time_index=w_start.isoformat(),
            open_price=open_p,
            high_price=max(open_p, close) * 1.018,
            low_price=min(open_p, close) * 0.982,
            close_price=close,
            latest_price=close,
        ))

    # Generate monthly: 36 months of history
    monthly_step = ((seed % 9) - 4) * 0.02
    for idx in range(36, 0, -1):
        past_year = today.year - (idx // 12)
        past_month = today.month - (idx % 12)
        if past_month <= 0:
            past_month += 12
            past_year -= 1
        m_start = dt.date(past_year, past_month, 1)
        close = max(0.01, float(base) / ((1 + monthly_step) ** idx))
        open_p = close * (0.97 + (seed % 5) * 0.012)
        snapshots.append(PriceSnapshot(
            symbol=symbol,
            stock_name=stock_name,
            source="Finnhub Quote Projection (Monthly)",
            timeframe="month",
            time_index=m_start.isoformat(),
            open_price=open_p,
            high_price=max(open_p, close) * 1.03,
            low_price=min(open_p, close) * 0.97,
            close_price=close,
            latest_price=close,
        ))

    # Generate yearly: 5 years of history
    yearly_step = ((seed % 9) - 4) * 0.06
    for idx in range(history_years, 0, -1):
        y_start = dt.date(today.year - idx, 1, 1)
        close = max(0.01, float(base) / ((1 + yearly_step) ** idx))
        open_p = close * (0.96 + (seed % 5) * 0.016)
        snapshots.append(PriceSnapshot(
            symbol=symbol,
            stock_name=stock_name,
            source="Finnhub Quote Projection (Yearly)",
            timeframe="year",
            time_index=y_start.isoformat(),
            open_price=open_p,
            high_price=max(open_p, close) * 1.045,
            low_price=min(open_p, close) * 0.955,
            close_price=close,
            latest_price=close,
        ))

    # Add the current day snapshot
    snapshots.append(current_day)
    return snapshots


def _history_window(history_years: int) -> tuple[int, int]:
    today = dt.date.today()
    start = dt.date(today.year - history_years + 1, 1, 1)
    start_ts = int(dt.datetime.combine(start, dt.time.min, tzinfo=dt.timezone.utc).timestamp())
    end_ts = int(dt.datetime.combine(today, dt.time.max, tzinfo=dt.timezone.utc).timestamp())
    return start_ts, end_ts


def extract_latest_commodity_prices(symbols: list[str] | None = None) -> pd.DataFrame:
    """Backward-compatible alias for live Finnhub stock quotes."""
    return extract_latest_stock_prices(symbols)


def extract_latest_stock_prices(symbols: list[str] | None = None) -> pd.DataFrame:
    """Fetch latest stock quotes from Finnhub."""
    symbols = symbols or _configured_symbols()
    metadata = _metadata_from_api(symbols)
    rows: list[dict[str, object]] = []

    for symbol in symbols:
        try:
            quote = _request_json("/quote", {"symbol": symbol})
            snapshot = _snapshot_from_quote(symbol, quote, metadata)
        except Exception as exc:
            print(f"  [error] latest quote {symbol}: {_safe_error(exc)}")
            snapshot = None
        if snapshot:
            rows.extend(_snapshot_to_rows(snapshot))
            print(f"  [ok] latest quote: {symbol}")

    return pd.DataFrame(rows, columns=["country_code", "country_name", "indicator", "timeframe", "time_index", "value", "source"])


def extract_commodity_prices(
    symbols: list[str] | None = None,
    history_years: int | None = None,
) -> pd.DataFrame:
    """Backward-compatible alias for Finnhub stock history."""
    return extract_stock_prices(symbols=symbols, history_years=history_years)


def extract_stock_prices(
    symbols: list[str] | None = None,
    history_years: int | None = None,
) -> pd.DataFrame:
    """Fetch OHLC stock snapshots from Finnhub daily candles across all granularities."""
    symbols = symbols or _configured_symbols()
    history_years = history_years or _history_years()
    metadata = _metadata_from_api(symbols)
    start_ts, end_ts = _history_window(history_years)
    rows: list[dict[str, object]] = []

    for symbol in symbols:
        projected_attempted = False
        try:
            candles = _request_json(
                "/stock/candle",
                {"symbol": symbol, "resolution": "D", "from": start_ts, "to": end_ts},
            )
            snapshots = _aggregate_snapshots_from_candles(symbol, candles, metadata)
        except Exception as exc:
            print(f"  [error] historical candles {symbol}: {_safe_error(exc)}")
            projected_attempted = True
            try:
                quote = _request_json("/quote", {"symbol": symbol})
                snapshots = _projected_history_from_quote(symbol, quote, metadata, history_years)
                if snapshots:
                    print(f"  [ok] quote projection: {symbol} ({len(snapshots)} snapshots)")
            except Exception as quote_exc:
                print(f"  [error] quote projection {symbol}: {_safe_error(quote_exc)}")
                snapshots = []

        if not snapshots and not projected_attempted:
            try:
                quote = _request_json("/quote", {"symbol": symbol})
                snapshots = _projected_history_from_quote(symbol, quote, metadata, history_years)
                if snapshots:
                    print(f"  [ok] quote projection: {symbol} ({len(snapshots)} snapshots)")
            except Exception as quote_exc:
                print(f"  [error] quote projection {symbol}: {_safe_error(quote_exc)}")

        for snapshot in snapshots:
            rows.extend(_snapshot_to_rows(snapshot))
        print(f"  [ok] history rows: {symbol} ({len(snapshots)} snapshots)")

    if not rows:
        print("  [fallback] No Finnhub rows fetched; using offline stock demo data")
        return _demo_stock_prices(symbols, history_years)

    df = pd.DataFrame(rows)
    print(f"Finnhub: {len(df):,} rows")
    return df[["country_code", "country_name", "indicator", "timeframe", "time_index", "value", "source"]]


def _demo_stock_prices(symbols: list[str], history_years: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        meta = DEFAULT_METADATA.get(symbol, {"name": symbol})
        seed = sum(ord(ch) for ch in symbol)
        base = 45 + (seed % 280)
        quote = {"c": base, "o": base * 0.99, "h": base * 1.01, "l": base * 0.98, "pc": base * 0.995}
        metadata = {symbol: {"name": str(meta.get("name", symbol)), "category": str(meta.get("category", "Equity"))}}
        snapshots = _projected_history_from_quote(symbol, quote, metadata, history_years)
        for snap in snapshots:
            rows.extend(_snapshot_to_rows(snap))

    df = pd.DataFrame(rows)
    print(f"  [demo] Offline stock data (All Timeframes): {len(df):,} rows")
    return df[["country_code", "country_name", "indicator", "timeframe", "time_index", "value", "source"]]


def extract_all(
    include_realtime: bool = True,
    include_legacy: bool = False,
    include_api: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Return source DataFrames keyed by source name.
    """
    _ = include_legacy, include_api
    result: dict[str, pd.DataFrame] = {}
    if include_realtime:
        result["finnhub_stocks"] = extract_stock_prices()
    if not result:
        raise ValueError("No sources selected. Pass include_realtime=True.")
    total = sum(len(df) for df in result.values())
    print(f"Total: {total:,} rows across {len(result)} source(s)")
    return result
