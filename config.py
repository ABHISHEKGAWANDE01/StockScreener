"""Screener thresholds for Intraday vs Swing styles."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_UNIVERSE = "nifty500"
DEFAULT_STYLE = "swing"
YF_NSE_SUFFIX = ".NS"
NIFTY_BENCHMARK = "^NSEI"

DOWNLOAD_PERIOD = "10y"
INTRADAY_PERIOD = "60d"
INTRADAY_INTERVAL = "15m"
REQUEST_THREADS = 12

MIN_PRICE = 30.0
MIN_AVG_DAILY_VALUE_CR = 5.0
RSI_MAX = 78.0
RSI_MIN = 42.0

WEIGHTS = {
    "proximity": 22,
    "squeeze": 16,
    "structure": 16,
    "volume": 12,
    "momentum": 12,
    "relative_strength": 12,
    "breakout_quality": 10,
}

BUY_ZONE_WEIGHTS = {
    "trend": 20,
    "discount": 18,
    "support": 16,
    "reset_momentum": 14,
    "reward_risk": 16,
    "relative_strength": 10,
    "volume": 6,
}
MIN_RR = 1.2
MA_SUPPORT_BAND_PCT = 4.0

CACHE_DIR = "data/cache"
UNIVERSE_CACHE_HOURS = 24
PRICE_CACHE_HOURS = 12
INTRADAY_CACHE_HOURS = 1
FO_CACHE_HOURS = 24
OPTION_CHAIN_CACHE_MINUTES = 15

AI_TECHNICAL_WEIGHT = 0.6
AI_MODEL_WEIGHT = 0.4
AI_REVIEW_WORKERS = 4

NSE_SQUARE_OFF = "15:15 IST"


@dataclass(frozen=True)
class StyleSpec:
    key: str
    label: str
    interval: str
    download_period: str
    resample: str | None
    use_ema: bool
    min_bars: int
    donchian_short: int
    donchian_long: int
    ma_fast: int
    ma_slow: int
    squeeze_bars: int
    volume_avg_bars: int
    rs_lookback: int
    pullback_bars: int
    pullback_min_pct: float
    pullback_max_pct: float
    max_dist_high_pct: float
    min_dist_high_pct: float
    max_dist_high_swing: float
    rsi_buy_min: float
    rsi_buy_max: float
    date_fmt: str
    hold_hint: str
    hold_business_days: int
    chart_bars: int
    tail_bars: int
    high_label: str
    fast_ma_label: str
    slow_ma_label: str
    t1_r: float
    t2_r: float
    min_atr_mult: float
    max_atr_mult: float


STYLES: dict[str, StyleSpec] = {
    "intraday": StyleSpec(
        key="intraday",
        label="Intraday",
        interval="15m",
        download_period=INTRADAY_PERIOD,
        resample=None,
        use_ema=True,
        min_bars=80,
        donchian_short=20,
        donchian_long=50,
        ma_fast=20,
        ma_slow=50,
        squeeze_bars=8,
        volume_avg_bars=20,
        rs_lookback=20,
        pullback_bars=16,
        pullback_min_pct=0.4,
        pullback_max_pct=2.5,
        max_dist_high_pct=0.8,
        min_dist_high_pct=-0.35,
        max_dist_high_swing=2.2,
        rsi_buy_min=35.0,
        rsi_buy_max=62.0,
        date_fmt="%Y-%m-%d %H:%M",
        hold_hint=f"same session — square off by {NSE_SQUARE_OFF}",
        hold_business_days=0,
        chart_bars=96,
        tail_bars=32,
        high_label="20-bar (15m) high",
        fast_ma_label="20-bar EMA",
        slow_ma_label="50-bar SMA",
        t1_r=1.2,
        t2_r=2.0,
        min_atr_mult=0.45,
        max_atr_mult=1.6,
    ),
    "swing": StyleSpec(
        key="swing",
        label="Swing trade",
        interval="1d",
        download_period=DOWNLOAD_PERIOD,
        resample=None,
        use_ema=True,
        min_bars=120,
        donchian_short=20,
        donchian_long=50,
        ma_fast=20,
        ma_slow=50,
        squeeze_bars=8,
        volume_avg_bars=20,
        rs_lookback=20,
        pullback_bars=20,
        pullback_min_pct=2.0,
        pullback_max_pct=12.0,
        max_dist_high_pct=5.0,
        min_dist_high_pct=-1.2,
        max_dist_high_swing=16.0,
        rsi_buy_min=32.0,
        rsi_buy_max=62.0,
        date_fmt="%Y-%m-%d",
        hold_hint="about 2–4 weeks",
        hold_business_days=15,
        chart_bars=180,
        tail_bars=40,
        high_label="20-day high",
        fast_ma_label="20-day EMA",
        slow_ma_label="50-day SMA",
        t1_r=1.6,
        t2_r=2.5,
        min_atr_mult=0.7,
        max_atr_mult=2.8,
    ),
}


def get_style(name: str) -> StyleSpec:
    key = (name or DEFAULT_STYLE).lower().replace(" ", "_")
    aliases = {"swing_trade": "swing", "intraday_trade": "intraday", "daily": "swing"}
    key = aliases.get(key, key)
    if key not in STYLES:
        raise ValueError(f"Unknown style {name}. Choose intraday or swing.")
    return STYLES[key]


# Back-compat for any leftover imports
DEFAULT_TIMEFRAME = DEFAULT_STYLE
TIMEFRAMES = STYLES
get_timeframe = get_style
TimeframeSpec = StyleSpec
