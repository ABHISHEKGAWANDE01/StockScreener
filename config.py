"""Screener thresholds, including per-timeframe lookbacks."""

from __future__ import annotations

from dataclasses import dataclass

# Universe
DEFAULT_UNIVERSE = "nifty500"
DEFAULT_TIMEFRAME = "monthly"
YF_NSE_SUFFIX = ".NS"
NIFTY_BENCHMARK = "^NSEI"

DOWNLOAD_PERIOD = "10y"
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

AI_TECHNICAL_WEIGHT = 0.6
AI_MODEL_WEIGHT = 0.4
AI_REVIEW_WORKERS = 4


@dataclass(frozen=True)
class TimeframeSpec:
    key: str
    label: str
    resample: str | None
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
    chart_bars: int
    tail_bars: int
    high_label: str
    fast_ma_label: str
    slow_ma_label: str


TIMEFRAMES: dict[str, TimeframeSpec] = {
    "daily": TimeframeSpec(
        key="daily",
        label="Daily",
        resample=None,
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
        hold_hint="days to a few weeks",
        chart_bars=180,
        tail_bars=40,
        high_label="20-day high",
        fast_ma_label="20-day EMA",
        slow_ma_label="50-day SMA",
    ),
    "weekly": TimeframeSpec(
        key="weekly",
        label="Weekly",
        resample="W-FRI",
        min_bars=52,
        donchian_short=13,
        donchian_long=26,
        ma_fast=10,
        ma_slow=20,
        squeeze_bars=6,
        volume_avg_bars=13,
        rs_lookback=13,
        pullback_bars=13,
        pullback_min_pct=3.0,
        pullback_max_pct=15.0,
        max_dist_high_pct=7.0,
        min_dist_high_pct=-1.5,
        max_dist_high_swing=18.0,
        rsi_buy_min=35.0,
        rsi_buy_max=65.0,
        date_fmt="%Y-%m-%d",
        hold_hint="2–8 weeks",
        chart_bars=104,
        tail_bars=30,
        high_label="13-week high",
        fast_ma_label="10-week EMA",
        slow_ma_label="20-week SMA",
    ),
    "monthly": TimeframeSpec(
        key="monthly",
        label="Monthly",
        resample="ME",
        min_bars=36,
        donchian_short=12,
        donchian_long=24,
        ma_fast=10,
        ma_slow=20,
        squeeze_bars=6,
        volume_avg_bars=12,
        rs_lookback=6,
        pullback_bars=6,
        pullback_min_pct=3.0,
        pullback_max_pct=18.0,
        max_dist_high_pct=8.0,
        min_dist_high_pct=-1.5,
        max_dist_high_swing=22.0,
        rsi_buy_min=38.0,
        rsi_buy_max=68.0,
        date_fmt="%Y-%m",
        hold_hint="about 1 month",
        chart_bars=60,
        tail_bars=18,
        high_label="12-month high",
        fast_ma_label="10-month SMA",
        slow_ma_label="20-month SMA",
    ),
}


def get_timeframe(name: str) -> TimeframeSpec:
    key = (name or DEFAULT_TIMEFRAME).lower()
    if key not in TIMEFRAMES:
        raise ValueError(f"Unknown timeframe {name}. Choose daily, weekly, or monthly.")
    return TIMEFRAMES[key]
