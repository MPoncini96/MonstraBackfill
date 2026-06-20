from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, time, timezone
from functools import lru_cache
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from env_loader import load_env

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - handled at runtime
    yf = None


logger = logging.getLogger(__name__)

DEFAULT_DATA_PROVIDER = "alpaca"
DEFAULT_ALPACA_DATA_FEED = "iex"
DEFAULT_ALPACA_TRADING_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_BASE_URL = "https://data.alpaca.markets"

ALPACA_SYMBOL_MAP: dict[str, str] = {
    "MOG-A": "MOG.A",
    "BRK-B": "BRK.B",
    "BF-B": "BF.B",
}

BAR_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Trade Count",
    "VWAP",
]


@dataclass(frozen=True)
class ProviderConfig:
    requested_provider: str
    provider: str
    alpaca_api_key: str | None
    alpaca_secret_key: str | None
    alpaca_data_feed: str
    alpaca_trading_base_url: str


def _clean_env_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().strip('"').strip("'")
    return cleaned or None


def _normalize_monstra_symbol(symbol: Any) -> str | None:
    if symbol is None:
        return None
    value = str(symbol).strip().upper()
    return value or None


def _normalize_symbol_list(symbols: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        normalized = _normalize_monstra_symbol(symbol)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _to_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _to_alpaca_time(value: Any) -> str:
    return _to_timestamp(value).isoformat().replace("+00:00", "Z")


def _to_yfinance_time(value: Any) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.strftime("%Y-%m-%d")


@lru_cache(maxsize=1)
def _cached_config() -> ProviderConfig:
    load_env()

    requested_provider = os.environ.get("DATA_PROVIDER", DEFAULT_DATA_PROVIDER).strip().lower()
    alpaca_api_key = _clean_env_value(os.environ.get("ALPACA_API_KEY"))
    alpaca_secret_key = _clean_env_value(os.environ.get("ALPACA_SECRET_KEY"))
    alpaca_data_feed = os.environ.get("ALPACA_DATA_FEED", DEFAULT_ALPACA_DATA_FEED).strip().lower()
    alpaca_trading_base_url = os.environ.get(
        "ALPACA_TRADING_BASE_URL",
        DEFAULT_ALPACA_TRADING_BASE_URL,
    ).strip()

    provider = requested_provider or DEFAULT_DATA_PROVIDER
    if provider == "alpaca" and (not alpaca_api_key or not alpaca_secret_key):
        logger.warning(
            "ALPACA_API_KEY/ALPACA_SECRET_KEY not set; falling back to yfinance provider."
        )
        provider = "yfinance"

    if provider not in {"alpaca", "yfinance"}:
        logger.warning("Unknown DATA_PROVIDER=%s; falling back to yfinance.", provider)
        provider = "yfinance"

    return ProviderConfig(
        requested_provider=requested_provider,
        provider=provider,
        alpaca_api_key=alpaca_api_key,
        alpaca_secret_key=alpaca_secret_key,
        alpaca_data_feed=alpaca_data_feed or DEFAULT_ALPACA_DATA_FEED,
        alpaca_trading_base_url=alpaca_trading_base_url or DEFAULT_ALPACA_TRADING_BASE_URL,
    )


def _get_config() -> ProviderConfig:
    return _cached_config()


def get_provider_info() -> dict[str, Any]:
    config = _get_config()
    return {
        "requested_provider": config.requested_provider,
        "provider": config.provider,
        "feed": config.alpaca_data_feed,
        "trading_base_url": config.alpaca_trading_base_url,
    }


def _alpaca_headers(config: ProviderConfig) -> dict[str, str]:
    if not config.alpaca_api_key or not config.alpaca_secret_key:
        raise RuntimeError("Alpaca credentials are not configured.")
    return {
        "APCA-API-KEY-ID": config.alpaca_api_key,
        "APCA-API-SECRET-KEY": config.alpaca_secret_key,
    }


def _prepare_alpaca_symbols(symbols: list[str]) -> tuple[list[str], dict[str, str], dict[str, str]]:
    request_symbols: list[str] = []
    original_to_request: dict[str, str] = {}
    request_to_original: dict[str, str] = {}

    for original in symbols:
        request_symbol = ALPACA_SYMBOL_MAP.get(original, original)
        original_to_request[original] = request_symbol
        request_to_original[request_symbol] = original
        request_symbols.append(request_symbol)

    return request_symbols, original_to_request, request_to_original


def _empty_bars_frame(symbols: list[str]) -> pd.DataFrame:
    if len(symbols) <= 1:
        return pd.DataFrame(columns=BAR_COLUMNS)

    columns = pd.MultiIndex.from_product([BAR_COLUMNS, symbols])
    return pd.DataFrame(columns=columns)


def _finalize_bars_frame(frame: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return _empty_bars_frame(symbols)

    if len(symbols) == 1:
        symbol = symbols[0]
        cols = [col for col in BAR_COLUMNS if col in frame.columns]
        result = frame[cols].copy()
        missing_cols = [col for col in BAR_COLUMNS if col not in result.columns]
        for col in missing_cols:
            result[col] = pd.NA
        result = result[BAR_COLUMNS]
        result.index = pd.to_datetime(result.index)
        result.index.name = "Date"
        logger.info(
            "Market data provider=%s feed=%s rows=%s symbols=%s missing_symbols=%s",
            _get_config().provider,
            _get_config().alpaca_data_feed,
            len(result.index),
            len(symbols),
            [] if not result.empty else [symbol],
        )
        return result.sort_index()

    parts: dict[tuple[str, str], pd.Series] = {}
    missing_symbols: list[str] = []

    for symbol in symbols:
        if symbol not in frame.columns.get_level_values(1):
            missing_symbols.append(symbol)
            continue
        for field in BAR_COLUMNS:
            key = (field, symbol)
            if key in frame.columns:
                parts[key] = frame[key]
            else:
                parts[key] = pd.Series(index=frame.index, dtype="float64")

    if not parts:
        logger.warning(
            "Market data provider=%s feed=%s returned no rows for symbols=%s",
            _get_config().provider,
            _get_config().alpaca_data_feed,
            symbols,
        )
        return _empty_bars_frame(symbols)

    result = pd.DataFrame(parts)
    result = result.sort_index(axis=1, level=[0, 1])
    result.index = pd.to_datetime(result.index)
    result.index.name = "Date"

    logger.info(
        "Market data provider=%s feed=%s rows=%s symbols=%s missing_symbols=%s",
        _get_config().provider,
        _get_config().alpaca_data_feed,
        len(result.index),
        len(symbols),
        missing_symbols,
    )
    return result.sort_index()


def _bars_to_price_series(frame: pd.DataFrame, symbols: list[str]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype="float64")

    if len(symbols) == 1:
        close_col = "Close" if "Close" in frame.columns else None
        if close_col is None or frame.empty:
            return pd.Series(dtype="float64")
        return pd.Series({symbols[0]: float(frame[close_col].iloc[-1])})

    prices: dict[str, float] = {}
    if not isinstance(frame.columns, pd.MultiIndex):
        return pd.Series(dtype="float64")

    for symbol in symbols:
        key = ("Close", symbol)
        if key not in frame.columns:
            continue
        series = frame[key].dropna()
        if series.empty:
            continue
        prices[symbol] = float(series.iloc[-1])
    return pd.Series(prices, dtype="float64")


def _build_alpaca_bars_frame(
    requested_symbols: list[str],
    request_to_original: dict[str, str],
    bars_by_symbol: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    per_symbol_frames: dict[str, pd.DataFrame] = {}

    for request_symbol in requested_symbols:
        original_symbol = request_to_original.get(request_symbol, request_symbol)
        bars = bars_by_symbol.get(request_symbol, [])
        if not bars:
            continue

        rows: list[dict[str, Any]] = []
        for bar in bars:
            rows.append(
                {
                    "Date": pd.Timestamp(bar["t"]).tz_convert(None),
                    "Open": bar.get("o"),
                    "High": bar.get("h"),
                    "Low": bar.get("l"),
                    "Close": bar.get("c"),
                    "Adj Close": bar.get("c"),
                    "Volume": bar.get("v"),
                    "Trade Count": bar.get("n"),
                    "VWAP": bar.get("vw"),
                }
            )

        symbol_frame = pd.DataFrame(rows).set_index("Date").sort_index()
        symbol_frame = symbol_frame[~symbol_frame.index.duplicated(keep="last")]
        per_symbol_frames[original_symbol] = symbol_frame

    if not per_symbol_frames:
        return pd.DataFrame()

    if len(requested_symbols) == 1:
        original_symbol = request_to_original.get(requested_symbols[0], requested_symbols[0])
        return per_symbol_frames.get(original_symbol, pd.DataFrame())

    combined = pd.concat(per_symbol_frames, axis=1)
    combined = combined.swaplevel(0, 1, axis=1).sort_index(axis=1, level=[0, 1])
    return combined


def _fetch_alpaca_daily_bars(symbols: list[str], start: Any, end: Any, *, adjusted: bool) -> pd.DataFrame:
    config = _get_config()
    request_symbols, _, request_to_original = _prepare_alpaca_symbols(symbols)
    params = {
        "symbols": ",".join(request_symbols),
        "timeframe": "1Day",
        "start": _to_alpaca_time(start),
        "end": _to_alpaca_time(end),
        "adjustment": "all" if adjusted else "raw",
        "feed": config.alpaca_data_feed,
        "limit": 10000,
        "sort": "asc",
    }

    bars_by_symbol: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in request_symbols}
    next_page_token: str | None = None

    while True:
        page_params = dict(params)
        if next_page_token:
            page_params["page_token"] = next_page_token

        response = requests.get(
            f"{ALPACA_DATA_BASE_URL}/v2/stocks/bars",
            headers=_alpaca_headers(config),
            params=page_params,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()

        page_bars = payload.get("bars") or {}
        for request_symbol, rows in page_bars.items():
            bars_by_symbol.setdefault(request_symbol, []).extend(rows or [])

        next_page_token = payload.get("next_page_token")
        if not next_page_token:
            break

    frame = _build_alpaca_bars_frame(request_symbols, request_to_original, bars_by_symbol)
    return _finalize_bars_frame(frame, symbols)


def _fetch_yfinance_daily_bars(symbols: list[str], start: Any, end: Any, *, adjusted: bool) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed.")

    raw = yf.download(
        tickers=symbols,
        start=_to_yfinance_time(start),
        end=_to_yfinance_time(end),
        interval="1d",
        auto_adjust=adjusted,
        progress=False,
        group_by="column",
        threads=False,
    )

    if raw is None or raw.empty:
        logger.warning("yfinance returned no rows for symbols=%s", symbols)
        return _empty_bars_frame(symbols)

    frame = raw.copy()
    if len(symbols) == 1:
        if adjusted and "Adj Close" not in frame.columns and "Close" in frame.columns:
            frame["Adj Close"] = frame["Close"]
        return _finalize_bars_frame(frame, symbols)

    if isinstance(frame.columns, pd.MultiIndex):
        if adjusted and "Adj Close" not in frame.columns.get_level_values(0):
            close_frame = frame["Close"].copy()
            adj_close = pd.concat({"Adj Close": close_frame}, axis=1)
            frame = pd.concat([frame, adj_close], axis=1)
            frame = frame.sort_index(axis=1, level=[0, 1])
    return _finalize_bars_frame(frame, symbols)


def get_daily_bars(
    symbols: list[str] | tuple[str, ...] | set[str],
    start: Any,
    end: Any,
    *,
    adjusted: bool = True,
) -> pd.DataFrame:
    normalized_symbols = _normalize_symbol_list(symbols)
    if not normalized_symbols:
        return pd.DataFrame()

    config = _get_config()
    if config.provider == "alpaca":
        return _fetch_alpaca_daily_bars(normalized_symbols, start, end, adjusted=adjusted)
    return _fetch_yfinance_daily_bars(normalized_symbols, start, end, adjusted=adjusted)


def get_latest_bars(symbols: list[str] | tuple[str, ...] | set[str]) -> pd.DataFrame:
    normalized_symbols = _normalize_symbol_list(symbols)
    if not normalized_symbols:
        return pd.DataFrame()

    config = _get_config()
    if config.provider == "alpaca":
        request_symbols, _, request_to_original = _prepare_alpaca_symbols(normalized_symbols)
        response = requests.get(
            f"{ALPACA_DATA_BASE_URL}/v2/stocks/bars/latest",
            headers=_alpaca_headers(config),
            params={
                "symbols": ",".join(request_symbols),
                "feed": config.alpaca_data_feed,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        bars_by_symbol = {
            symbol: [bar]
            for symbol, bar in (payload.get("bars") or {}).items()
            if bar
        }
        frame = _build_alpaca_bars_frame(request_symbols, request_to_original, bars_by_symbol)
        return _finalize_bars_frame(frame, normalized_symbols)

    end = pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=10)
    frame = _fetch_yfinance_daily_bars(normalized_symbols, start, end, adjusted=False)
    if frame.empty:
        return frame

    if len(normalized_symbols) == 1:
        return frame.tail(1)

    latest_idx = frame.index.max()
    return frame.loc[[latest_idx]]


def get_latest_prices(symbols: list[str] | tuple[str, ...] | set[str]) -> pd.Series:
    normalized_symbols = _normalize_symbol_list(symbols)
    if not normalized_symbols:
        return pd.Series(dtype="float64")

    bars = get_latest_bars(normalized_symbols)
    prices = _bars_to_price_series(bars, normalized_symbols)
    missing_symbols = [symbol for symbol in normalized_symbols if symbol not in prices.index]
    logger.info(
        "Latest prices provider=%s feed=%s symbols=%s returned=%s missing_symbols=%s",
        _get_config().provider,
        _get_config().alpaca_data_feed,
        len(normalized_symbols),
        len(prices.index),
        missing_symbols,
    )
    return prices


def is_market_open() -> bool:
    config = _get_config()
    if config.provider == "alpaca":
        response = requests.get(
            f"{config.alpaca_trading_base_url.rstrip('/')}/v2/clock",
            headers=_alpaca_headers(config),
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        is_open = bool(payload.get("is_open"))
        logger.info(
            "Market clock provider=%s feed=%s is_open=%s",
            config.provider,
            config.alpaca_data_feed,
            is_open,
        )
        return is_open

    now_et = datetime.now(ZoneInfo("America/New_York"))
    session_open = time(9, 30)
    session_close = time(16, 0)
    is_open = now_et.weekday() < 5 and session_open <= now_et.time() <= session_close
    logger.info(
        "Market clock provider=%s feed=%s is_open=%s",
        config.provider,
        config.alpaca_data_feed,
        is_open,
    )
    return is_open
