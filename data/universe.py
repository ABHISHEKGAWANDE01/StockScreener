from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from config import CACHE_DIR, UNIVERSE_CACHE_HOURS, YF_NSE_SUFFIX
from utils import ensure_dir

logger = logging.getLogger(__name__)

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

NIFTY_LIST_URLS = {
    "nifty50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "nifty200": "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
    "nifty500": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
}

WIKI_NIFTY500 = "https://en.wikipedia.org/wiki/NIFTY_500"


def _cache_path(name: str) -> Path:
    return ensure_dir(CACHE_DIR) / f"universe_{name}.json"


def _is_fresh(path: Path, hours: float) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < hours * 3600


def _save(name: str, rows: list[dict]) -> None:
    _cache_path(name).write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _load(name: str) -> list[dict] | None:
    path = _cache_path(name)
    if not _is_fresh(path, UNIVERSE_CACHE_HOURS):
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(NSE_HEADERS)
    try:
        s.get("https://www.nseindia.com", timeout=15)
    except requests.RequestException:
        logger.debug("NSE homepage warmup failed; continuing")
    return s


def _from_nse_csv(universe: str) -> list[dict]:
    url = NIFTY_LIST_URLS[universe]
    session = _session()
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    symbol_col = next(c for c in df.columns if c.lower().replace(" ", "") in {"symbol", "symbols"})
    name_col = next((c for c in df.columns if "company" in c.lower() or c.lower() == "name"), symbol_col)
    industry_col = next((c for c in df.columns if "industry" in c.lower()), None)
    rows = []
    for _, row in df.iterrows():
        symbol = str(row[symbol_col]).strip().upper()
        if not symbol or symbol == "NAN":
            continue
        rows.append(
            {
                "symbol": symbol,
                "yf_ticker": f"{symbol}{YF_NSE_SUFFIX}",
                "name": str(row[name_col]).strip(),
                "industry": str(row[industry_col]).strip() if industry_col else "",
            }
        )
    if len(rows) < 30:
        raise ValueError(f"NSE CSV for {universe} looked incomplete ({len(rows)} rows)")
    return rows


def _from_wikipedia() -> list[dict]:
    tables = pd.read_html(WIKI_NIFTY500)
    df = next(
        t
        for t in tables
        if any("symbol" in str(c).lower() for c in t.columns) and len(t) > 100
    )
    symbol_col = next(c for c in df.columns if "symbol" in str(c).lower())
    name_col = next((c for c in df.columns if "company" in str(c).lower()), symbol_col)
    rows = []
    for _, row in df.iterrows():
        symbol = str(row[symbol_col]).strip().upper().replace(".NS", "")
        if not symbol or symbol == "NAN":
            continue
        rows.append(
            {
                "symbol": symbol,
                "yf_ticker": f"{symbol}{YF_NSE_SUFFIX}",
                "name": str(row[name_col]).strip(),
                "industry": "",
            }
        )
    return rows


FALLBACK_NIFTY50 = [
    "RELIANCE", "TCS", "HDFCBANK", "BHARTIARTL", "ICICIBANK", "SBIN", "INFY",
    "LICI", "ITC", "HINDUNILVR", "LT", "BAJFINANCE", "HCLTECH", "MARUTI",
    "SUNPHARMA", "KOTAKBANK", "AXISBANK", "NTPC", "ONGC", "TATAMOTORS",
    "M&M", "ADANIENT", "ADANIPORTS", "POWERGRID", "ULTRACEMCO", "TITAN",
    "WIPRO", "JSWSTEEL", "TATASTEEL", "COALINDIA", "BAJAJFINSV", "NESTLEIND",
    "GRASIM", "TECHM", "HINDALCO", "CIPLA", "SBILIFE", "HDFCLIFE", "DRREDDY",
    "BPCL", "EICHERMOT", "APOLLOHOSP", "HEROMOTOCO", "INDUSINDBK", "TRENT",
    "SHRIRAMFIN", "BEL", "ASIANPAINT", "TATACONSUM", "BAJAJ-AUTO",
]


def _fallback(universe: str) -> list[dict]:
    logger.warning("Using embedded Nifty 50 fallback for %s", universe)
    return [
        {"symbol": s, "yf_ticker": f"{s}{YF_NSE_SUFFIX}", "name": s, "industry": ""}
        for s in FALLBACK_NIFTY50
    ]


def load_universe(universe: str = "nifty500") -> pd.DataFrame:
    universe = universe.lower()
    if universe not in NIFTY_LIST_URLS:
        raise ValueError(f"Unknown universe {universe}. Choose nifty50, nifty200, or nifty500.")

    cached = _load(universe)
    if cached:
        return pd.DataFrame(cached)

    rows: list[dict] = []
    try:
        rows = _from_nse_csv(universe)
        logger.info("Loaded %s constituents from NSE (%s)", len(rows), universe)
    except Exception as exc:
        logger.warning("NSE list download failed (%s). Trying Wikipedia.", exc)
        if universe == "nifty500":
            try:
                rows = _from_wikipedia()
                logger.info("Loaded %s constituents from Wikipedia", len(rows))
            except Exception as wiki_exc:
                logger.warning("Wikipedia download failed (%s)", wiki_exc)

    if not rows:
        rows = _fallback(universe)

    _save(universe, rows)
    return pd.DataFrame(rows)
