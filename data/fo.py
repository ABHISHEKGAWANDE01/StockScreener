"""NSE F&O membership and option-contract suggestions."""

from __future__ import annotations

import io
import json
import logging
import time
import calendar
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from config import CACHE_DIR, FO_CACHE_HOURS, OPTION_CHAIN_CACHE_MINUTES
from utils import ensure_dir

logger = logging.getLogger(__name__)

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/option-chain",
}

LOT_URLS = [
    "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv",
    "https://archives.nseindia.com/content/fo/fo_mktlots.csv",
]

FALLBACK_FO = {
    "RELIANCE": 250,
    "TCS": 175,
    "HDFCBANK": 550,
    "BHARTIARTL": 475,
    "ICICIBANK": 700,
    "SBIN": 750,
    "INFY": 400,
    "ITC": 1600,
    "HINDUNILVR": 300,
    "LT": 150,
    "BAJFINANCE": 125,
    "HCLTECH": 350,
    "MARUTI": 50,
    "SUNPHARMA": 350,
    "KOTAKBANK": 400,
    "AXISBANK": 625,
    "NTPC": 1500,
    "ONGC": 1925,
    "TATAMOTORS": 550,
    "M&M": 200,
    "ADANIENT": 300,
    "ADANIPORTS": 475,
    "POWERGRID": 1200,
    "ULTRACEMCO": 100,
    "TITAN": 175,
    "WIPRO": 1500,
    "JSWSTEEL": 675,
    "TATASTEEL": 550,
    "COALINDIA": 1350,
    "BAJAJFINSV": 500,
    "NESTLEIND": 40,
    "GRASIM": 125,
    "TECHM": 600,
    "HINDALCO": 1400,
    "CIPLA": 375,
    "SBILIFE": 375,
    "HDFCLIFE": 1100,
    "DRREDDY": 125,
    "BPCL": 1800,
    "EICHERMOT": 175,
    "APOLLOHOSP": 125,
    "HEROMOTOCO": 150,
    "INDUSINDBK": 500,
    "TRENT": 50,
    "SHRIRAMFIN": 300,
    "BEL": 1550,
    "ASIANPAINT": 200,
    "TATACONSUM": 450,
    "BAJAJ-AUTO": 75,
    "NIFTY": 25,
    "BANKNIFTY": 15,
}


def _lots_path() -> Path:
    return ensure_dir(CACHE_DIR) / "fo_lots.json"


def _chain_path(symbol: str) -> Path:
    safe = symbol.replace("&", "_").replace("-", "_")
    return ensure_dir(CACHE_DIR) / f"oc_{safe}.json"


def _fresh(path: Path, hours: float) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < hours * 3600


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(NSE_HEADERS)
    try:
        s.get("https://www.nseindia.com", timeout=12)
        s.get("https://www.nseindia.com/option-chain", timeout=12)
    except requests.RequestException:
        logger.debug("NSE warmup failed")
    return s


def _parse_lots(text: str) -> dict[str, int]:
    df = pd.read_csv(io.StringIO(text))
    cols = {c.lower().strip().replace(" ", ""): c for c in df.columns}
    sym_col = next((cols[k] for k in cols if "symbol" in k or "underlying" in k), None)
    lot_col = next((cols[k] for k in cols if "lot" in k), None)
    if not sym_col or not lot_col:
        raise ValueError(f"Unexpected lot-file columns: {list(df.columns)}")
    out: dict[str, int] = {}
    for _, row in df.iterrows():
        symbol = str(row[sym_col]).strip().upper()
        if not symbol or symbol in {"NAN", "SYMBOL", "UNDERLYING"}:
            continue
        try:
            lot = int(float(str(row[lot_col]).replace(",", "")))
        except ValueError:
            continue
        if lot > 0:
            out[symbol] = lot
    return out


def load_fo_lots(force: bool = False) -> dict[str, int]:
    path = _lots_path()
    if not force and _fresh(path, FO_CACHE_HOURS):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data:
                return {str(k).upper(): int(v) for k, v in data.items()}
        except Exception:
            pass

    lots: dict[str, int] = {}
    session = _session()
    for url in LOT_URLS:
        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            lots = _parse_lots(resp.text)
            if lots:
                logger.info("Loaded %s F&O lots from NSE", len(lots))
                break
        except Exception as exc:
            logger.debug("F&O lot download failed (%s): %s", url, exc)

    if not lots:
        lots = dict(FALLBACK_FO)
        logger.warning("Using embedded F&O list (%s names)", len(lots))

    path.write_text(json.dumps(lots, indent=2), encoding="utf-8")
    return lots


def is_fo(symbol: str, lots: dict[str, int] | None = None) -> bool:
    lots = lots or load_fo_lots()
    return symbol.upper() in lots


def lot_size(symbol: str, lots: dict[str, int] | None = None) -> int | None:
    lots = lots or load_fo_lots()
    return lots.get(symbol.upper())


def last_tuesday_of_month(year: int, month: int) -> date:
    last = calendar.monthrange(year, month)[1]
    d = date(year, month, last)
    return d - timedelta(days=(d.weekday() - 1) % 7)


def nearest_stock_monthly_expiry(from_day: date | None = None) -> date:
    """NSE stock F&O monthlies expire on the last Tuesday of the month (since Aug 2025)."""
    day = from_day or date.today()
    exp = last_tuesday_of_month(day.year, day.month)
    if exp < day:
        year, month = (day.year + 1, 1) if day.month == 12 else (day.year, day.month + 1)
        exp = last_tuesday_of_month(year, month)
    return exp


def next_thursday(from_day: date | None = None) -> date:
    """Deprecated alias — stock options are not weekly Thursday. Kept for imports."""
    return nearest_stock_monthly_expiry(from_day)


def strike_step(price: float) -> float:
    if price < 50:
        return 2.5
    if price < 250:
        return 5.0
    if price < 500:
        return 10.0
    if price < 1000:
        return 20.0
    if price < 2500:
        return 50.0
    if price < 5000:
        return 50.0
    return 100.0


def round_strike(price: float, otm: bool = False) -> float:
    step = strike_step(price)
    atm = round(price / step) * step
    if otm and atm < price:
        atm += step
    return float(atm)


def _parse_expiry(text: str) -> date | None:
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%b-%y", "%d-%m-%Y", "%d-%m-%y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _valid_chain(data: dict | None) -> bool:
    if not data or not isinstance(data, dict):
        return False
    records = data.get("records") or {}
    expiries = records.get("expiryDates") or data.get("expiryDates") or []
    rows = records.get("data") or []
    return bool(expiries or rows)


def fetch_contract_info(symbol: str, session: requests.Session | None = None) -> dict | None:
    try:
        session = session or _session()
        session.headers["Referer"] = "https://www.nseindia.com/option-chain"
        url = f"https://www.nseindia.com/api/option-chain-contract-info?symbol={quote(symbol)}"
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("expiryDates"):
            return data
    except Exception as exc:
        logger.debug("Contract info failed for %s: %s", symbol, exc)
    return None


def fetch_option_chain(symbol: str, expiry_raw: str | None = None) -> dict | None:
    path = _chain_path(symbol)
    if _fresh(path, OPTION_CHAIN_CACHE_MINUTES / 60):
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if _valid_chain(cached):
                return cached
        except Exception:
            pass
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    session = _session()
    info = fetch_contract_info(symbol, session)
    chosen = expiry_raw
    if not chosen and info:
        today = date.today()
        for raw in info.get("expiryDates") or []:
            exp = _parse_expiry(str(raw))
            if exp and exp >= today:
                chosen = str(raw)
                break
        if not chosen and info.get("expiryDates"):
            chosen = str(info["expiryDates"][0])

    try:
        session.headers["Referer"] = "https://www.nseindia.com/option-chain"
        if chosen:
            url = (
                "https://www.nseindia.com/api/option-chain-v3"
                f"?type=equity&symbol={quote(symbol)}&expiry={quote(chosen)}"
            )
        else:
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={quote(symbol)}"
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not _valid_chain(data) and info:
            data = {
                "records": {
                    "expiryDates": info.get("expiryDates") or [],
                    "strikePrices": info.get("strikePrice") or [],
                    "data": [],
                    "underlyingValue": None,
                }
            }
        if _valid_chain(data):
            path.write_text(json.dumps(data), encoding="utf-8")
            return data
        logger.debug("Empty option chain for %s", symbol)
    except Exception as exc:
        logger.debug("Option chain failed for %s: %s", symbol, exc)
        if info:
            return {
                "records": {
                    "expiryDates": info.get("expiryDates") or [],
                    "strikePrices": info.get("strikePrice") or [],
                    "data": [],
                }
            }
    return None


def _listed_strikes(chain: dict) -> list[float]:
    records = chain.get("records") or {}
    raw = records.get("strikePrices") or chain.get("strikePrice") or []
    out = []
    for x in raw:
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            continue
    return out


def _atm_call_from_chain(chain: dict, spot: float) -> dict | None:
    records = chain.get("records") or {}
    expiries = records.get("expiryDates") or []
    today = date.today()
    chosen = None
    for raw in expiries:
        exp = _parse_expiry(str(raw))
        if exp and exp >= today:
            chosen = str(raw)
            break
    if not chosen and expiries:
        chosen = str(expiries[0])
    if not chosen:
        return None

    rows = [r for r in (records.get("data") or []) if r.get("CE")]
    if chosen:
        matched = [
            r
            for r in rows
            if str(r.get("expiryDate") or r.get("expiryDates") or "") in {chosen, chosen.replace(" ", "")}
            or _parse_expiry(str(r.get("expiryDate") or r.get("expiryDates") or "")) == _parse_expiry(chosen)
        ]
        if matched:
            rows = matched

    strike = None
    ce = {}
    if rows:
        best = min(rows, key=lambda r: abs(float(r.get("strikePrice") or (r.get("CE") or {}).get("strikePrice") or 0) - spot))
        ce = best.get("CE") or {}
        strike = float(best.get("strikePrice") or ce.get("strikePrice") or 0)
    else:
        strikes = _listed_strikes(chain)
        if strikes:
            strike = min(strikes, key=lambda x: abs(x - spot))

    if not strike:
        return {
            "expiry_raw": chosen,
            "expiry": _parse_expiry(chosen),
            "strike": None,
            "ltp": None,
            "bid": None,
            "ask": None,
            "iv": None,
        }

    ltp = float(ce.get("lastPrice") or 0)
    bid = float(ce.get("bidprice") or ce.get("buyPrice1") or 0)
    ask = float(ce.get("askPrice") or ce.get("sellPrice1") or 0)
    return {
        "expiry_raw": chosen,
        "expiry": _parse_expiry(chosen),
        "strike": strike,
        "ltp": ltp if ltp > 0 else None,
        "bid": bid if bid > 0 else None,
        "ask": ask if ask > 0 else None,
        "iv": float(ce.get("impliedVolatility") or 0) or None,
    }


def suggest_long_option(symbol: str, spot: float, lots: dict[str, int] | None = None) -> dict:
    """Nearest listed NSE CE if the name is F&O; otherwise spot."""
    lots = lots or load_fo_lots()
    if not is_fo(symbol, lots):
        return {
            "instrument": "spot",
            "fo": False,
            "note": "Not in NSE F&O — trade the cash/spot book.",
        }

    lot = lot_size(symbol, lots)
    chain = fetch_option_chain(symbol)
    live = _atm_call_from_chain(chain, spot) if chain else None
    listed = bool(live and live.get("expiry"))
    expiry = live["expiry"] if listed else nearest_stock_monthly_expiry()
    strike = live["strike"] if live and live.get("strike") else round_strike(spot)
    premium = None
    if live:
        premium = live.get("ask") or live.get("ltp") or live.get("bid")
    if premium is None:
        premium = round(max(spot * 0.006, strike_step(spot) * 0.15), 2)

    contract = f"{symbol} {expiry.strftime('%d-%b-%Y')} {strike:g} CE"
    source = "NSE listed expiry" if listed else "assumed last-Tuesday monthly expiry"
    return {
        "instrument": "option",
        "fo": True,
        "option_type": "CE",
        "strike": strike,
        "expiry": expiry.isoformat(),
        "expiry_label": expiry.strftime("%d %b %Y"),
        "contract": contract,
        "lot_size": lot,
        "premium": round(float(premium), 2) if premium else None,
        "premium_live": bool(live and (live.get("ltp") or live.get("ask") or live.get("bid"))),
        "expiry_source": source,
        "note": (
            f"F&O available. Prefer {contract} ({source})"
            + (f" (LTP/ask ≈ ₹{premium:.2f})" if premium else "")
            + (f", lot {lot}" if lot else "")
            + ". If the option is illiquid, use the spot plan instead."
        ),
    }
