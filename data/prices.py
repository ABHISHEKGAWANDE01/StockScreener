from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import (
    CACHE_DIR,
    DOWNLOAD_PERIOD,
    INTRADAY_CACHE_HOURS,
    INTRADAY_PERIOD,
    PRICE_CACHE_HOURS,
    REQUEST_THREADS,
    StyleSpec,
    get_style,
)
from utils import ensure_dir

logger = logging.getLogger(__name__)


def _price_path(ticker: str, interval: str = "1d") -> Path:
    safe = ticker.replace("^", "_").replace("/", "_")
    prefix = "px15" if interval.endswith("m") else "px"
    return ensure_dir(CACHE_DIR) / f"{prefix}_{safe}.parquet"


def _is_fresh(path: Path, hours: float = PRICE_CACHE_HOURS) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < hours * 3600


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    out = df[keep].copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    out.index = idx
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna(how="all")


def download_one(
    ticker: str,
    force: bool = False,
    interval: str = "1d",
    period: str | None = None,
) -> pd.DataFrame:
    path = _price_path(ticker, interval)
    cache_hours = INTRADAY_CACHE_HOURS if interval.endswith("m") else PRICE_CACHE_HOURS
    if not force and _is_fresh(path, cache_hours):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    raw = yf.download(
        ticker,
        period=period or (INTRADAY_PERIOD if interval.endswith("m") else DOWNLOAD_PERIOD),
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    df = _normalize(raw)
    if not df.empty:
        df.to_parquet(path)
    return df


def download_for_style(ticker: str, style: StyleSpec | str, force: bool = False) -> pd.DataFrame:
    spec = get_style(style) if isinstance(style, str) else style
    return download_one(ticker, force=force, interval=spec.interval, period=spec.download_period)


def resample_ohlc(daily: pd.DataFrame, rule: str | None) -> pd.DataFrame:
    if daily.empty:
        return daily
    if not rule:
        return daily.copy()
    out = daily.resample(rule).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    return out.dropna(subset=["Close"])


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    return resample_ohlc(daily, "W-FRI")


def to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    return resample_ohlc(daily, "ME")


def download_many(
    tickers: list[str],
    force: bool = False,
    workers: int = REQUEST_THREADS,
    interval: str = "1d",
    period: str | None = None,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    unique = list(dict.fromkeys(tickers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(download_one, t, force, interval, period): t for t in unique}
        for i, fut in enumerate(as_completed(futs), start=1):
            ticker = futs[fut]
            try:
                df = fut.result()
                if df.empty:
                    logger.debug("No price data for %s", ticker)
                else:
                    out[ticker] = df
            except Exception as exc:
                logger.warning("Download failed for %s: %s", ticker, exc)
            if i % 50 == 0 or i == len(unique):
                logger.info("Downloaded %s/%s tickers", i, len(unique))
    return out
