from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analysis.indicators import adx, atr, bollinger_bandwidth, donchian_high, ema, linear_slope, macd, rsi, sma
from config import (
    BUY_ZONE_WEIGHTS,
    MA_SUPPORT_BAND_PCT,
    MIN_AVG_DAILY_VALUE_CR,
    MIN_PRICE,
    MIN_RR,
    RSI_MAX,
    RSI_MIN,
    TimeframeSpec,
    WEIGHTS,
    get_timeframe,
)


@dataclass
class BreakoutResult:
    symbol: str
    name: str
    industry: str
    passed: bool
    score: float
    close: float
    reject_reason: str = ""
    setup: str = ""
    timeframe: str = ""
    breakout_score: float = 0.0
    buy_zone_score: float = 0.0
    metrics: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    monthly_tail: list[dict] = field(default_factory=list)
    daily_tail: list[dict] = field(default_factory=list)
    bars_tail: list[dict] = field(default_factory=list)
    ai_score: float | None = None
    ai_verdict: str = ""
    ai_confidence: float | None = None
    ai_rationale: str = ""
    ai_risks: list[str] = field(default_factory=list)
    combined_score: float | None = None
    signal: str = ""
    entry: float | None = None
    sl: float | None = None
    target: float | None = None
    target_2: float | None = None
    entry_type: str = ""


def _clip01(x: float) -> float:
    if np.isnan(x):
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


def _proximity_score(dist_pct: float, tf: TimeframeSpec) -> float:
    if np.isnan(dist_pct):
        return 0.0
    if dist_pct > tf.max_dist_high_pct or dist_pct < tf.min_dist_high_pct:
        return 0.0
    sweet_hi = min(4.0, tf.max_dist_high_pct * 0.55)
    if 0.3 <= dist_pct <= sweet_hi:
        return 1.0
    if dist_pct < 0.3:
        return _clip01((dist_pct - tf.min_dist_high_pct) / (0.3 - tf.min_dist_high_pct))
    return _clip01(1.0 - (dist_pct - sweet_hi) / (tf.max_dist_high_pct - sweet_hi))


def _squeeze_score(range_ratio: float, bb_pctile: float, atr_pctile: float) -> float:
    tightness = _clip01(1.0 - range_ratio)
    coil = 0.5 * (1.0 - _clip01(bb_pctile)) + 0.5 * (1.0 - _clip01(atr_pctile))
    return 0.55 * tightness + 0.45 * coil


def _volume_score(vol_dryup: float, vol_expand: float, obv_slope: float) -> float:
    dry = _clip01((1.0 - vol_dryup) * 1.4)
    expand = _clip01((vol_expand - 0.9) / 0.8)
    accumulation = _clip01((obv_slope + 0.01) / 0.03)
    return 0.4 * dry + 0.35 * expand + 0.25 * accumulation


def _structure_score(
    above_fast: bool,
    stacked: bool,
    higher_lows: bool,
    close_in_range: float,
    adx_val: float,
) -> float:
    parts = [
        1.0 if above_fast else 0.0,
        1.0 if stacked else 0.25,
        1.0 if higher_lows else 0.2,
        _clip01((close_in_range - 0.5) / 0.5),
        _clip01((adx_val - 12) / 20),
    ]
    return float(np.mean(parts))


def _momentum_score(rsi_val: float, macd_hist: float, macd_rising: bool) -> float:
    rsi_quality = 0.0
    if RSI_MIN <= rsi_val <= RSI_MAX:
        if 52 <= rsi_val <= 68:
            rsi_quality = 1.0
        elif rsi_val < 52:
            rsi_quality = _clip01((rsi_val - RSI_MIN) / (52 - RSI_MIN))
        else:
            rsi_quality = _clip01((RSI_MAX - rsi_val) / (RSI_MAX - 68))
    macd_part = 0.7 if macd_hist > 0 else 0.25
    if macd_rising:
        macd_part = min(1.0, macd_part + 0.3)
    return 0.6 * rsi_quality + 0.4 * macd_part


def _rs_score(rs_val: float, rs_slope: float) -> float:
    level = _clip01((rs_val + 5) / 20)
    slope = _clip01((rs_slope + 0.005) / 0.02)
    return 0.65 * level + 0.35 * slope


def _quality_score(upper_wick: float, body_strength: float, near_long_high: bool) -> float:
    wick_pen = _clip01(1.0 - upper_wick / 0.45)
    body = _clip01(body_strength)
    long_high = 1.0 if near_long_high else 0.55
    return 0.4 * wick_pen + 0.35 * body + 0.25 * long_high


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def _ohlc_tail(df: pd.DataFrame, n: int, date_fmt: str) -> list[dict]:
    rows = []
    for idx, row in df.tail(n).iterrows():
        rows.append(
            {
                "date": idx.strftime(date_fmt),
                "open": round(float(row.Open), 2),
                "high": round(float(row.High), 2),
                "low": round(float(row.Low), 2),
                "close": round(float(row.Close), 2),
                "volume": int(row.Volume) if not np.isnan(row.Volume) else 0,
            }
        )
    return rows


def _fail(symbol, name, industry, px, reason, timeframe: str = "") -> BreakoutResult:
    return BreakoutResult(symbol, name, industry, False, 0, px, reason, timeframe=timeframe)


def _r2(x: float) -> float:
    return float(round(x, 2))


def build_trade_plan(
    *,
    px: float,
    setup: str,
    score: float,
    high_s: float,
    high_l: float,
    recent_high: float,
    swing_low: float,
    sma_slow: float,
    atr_val: float,
) -> dict:
    """Entry, stop, and two targets. BUY = take now; BUY STOP = wait for the break; WATCH = levels only."""
    if np.isnan(atr_val) or atr_val <= 0:
        atr_val = px * 0.02

    in_zone = "buy_zone" in setup
    already_broke = px >= high_s * 0.998

    if in_zone and not already_broke:
        signal = "BUY"
        entry_type = "market"
        entry = px
    elif already_broke:
        signal = "BUY"
        entry_type = "market"
        entry = px
    else:
        signal = "BUY STOP"
        entry_type = "buy_stop"
        entry = high_s * 1.002

    struct_sl = min(swing_low, sma_slow if not np.isnan(sma_slow) else swing_low) - 0.3 * atr_val
    sl = struct_sl
    min_gap = max(0.7 * atr_val, entry * 0.008)
    max_gap = max(2.8 * atr_val, entry * 0.08)
    if entry - sl < min_gap:
        sl = entry - min_gap
    if entry - sl > max_gap:
        sl = entry - max_gap
    if sl >= entry:
        sl = entry - min_gap

    risk = entry - sl
    t1 = entry + 1.6 * risk
    if recent_high > entry * 1.006:
        t1 = max(t1, recent_high)
    if already_broke or signal == "BUY STOP":
        stretch = high_l if high_l > entry else entry + 2.2 * risk
        t1 = max(t1, min(stretch, entry + 2.0 * risk))
    t1 = max(t1, entry + 1.2 * risk)

    t2 = max(t1 + 0.8 * risk, entry + 2.5 * risk)
    if high_l > t1:
        t2 = max(t2, high_l)

    rr = (t1 - entry) / risk if risk > 0 else 0.0
    if rr < 1.15 or score < 52:
        signal = "WATCH"

    return {
        "signal": signal,
        "entry_type": entry_type,
        "entry": _r2(entry),
        "sl": _r2(sl),
        "target": _r2(t1),
        "target_2": _r2(t2),
        "rr": round(rr, 2),
        "reward_pct": round((t1 - entry) / entry * 100, 2),
        "risk_pct": round((entry - sl) / entry * 100, 2),
    }


def apply_ai_to_signal(result: BreakoutResult) -> BreakoutResult:
    if result.ai_verdict == "reject":
        result.signal = "AVOID"
    elif result.ai_verdict == "watch" and result.signal == "BUY":
        result.signal = "WATCH"
    result.metrics["signal"] = result.signal
    return result


def analyze_symbol(
    symbol: str,
    name: str,
    industry: str,
    daily: pd.DataFrame,
    bars: pd.DataFrame,
    nifty_bars: pd.DataFrame | None,
    tf: TimeframeSpec | str = "monthly",
) -> BreakoutResult:
    if isinstance(tf, str):
        tf = get_timeframe(tf)
    if len(bars) < tf.min_bars or len(daily) < 60:
        return _fail(symbol, name, industry, 0, "insufficient price history", tf.key)

    m = bars.copy()
    close = m["Close"]
    high = m["High"]
    low = m["Low"]
    vol = m["Volume"]

    m["ema_fast"] = ema(close, tf.ma_fast)
    m["sma_fast"] = sma(close, tf.ma_fast)
    m["sma_slow"] = sma(close, tf.ma_slow)
    m["rsi"] = rsi(close, 14)
    m = m.join(macd(close))
    m["atr"] = atr(m, 14)
    m["adx"] = adx(m, 14)
    m["bb_bw"] = bollinger_bandwidth(close, min(20, max(10, tf.ma_slow)))
    m["donchian_s"] = donchian_high(high.shift(1), tf.donchian_short)
    m["donchian_l"] = donchian_high(high.shift(1), tf.donchian_long)
    m["range"] = high - low
    m["obv"] = _obv(close, vol)

    last = m.iloc[-1]
    prev = m.iloc[-2]
    px = float(last["Close"])
    if px < MIN_PRICE or np.isnan(px):
        return _fail(symbol, name, industry, px, "price too low / illiquid penny", tf.key)

    recent_daily = daily.tail(60)
    adv_cr = float((recent_daily["Close"] * recent_daily["Volume"]).mean() / 1e7)
    if np.isnan(adv_cr) or adv_cr < MIN_AVG_DAILY_VALUE_CR:
        return _fail(symbol, name, industry, px, f"ADV ₹{adv_cr:.1f}cr below liquidity floor", tf.key)

    high_s = float(last["donchian_s"])
    high_l = float(last["donchian_l"]) if not np.isnan(last["donchian_l"]) else high_s
    if np.isnan(high_s) or high_s <= 0:
        return _fail(symbol, name, industry, px, f"no {tf.high_label}", tf.key)

    dist_s = (high_s - px) / high_s * 100
    dist_l = (high_l - px) / high_l * 100
    rsi_val = float(last["rsi"])
    ema_fast = float(last["ema_fast"])
    sma_fast = float(last["sma_fast"])
    sma_slow = float(last["sma_slow"])
    fast_ma = ema_fast if tf.key in {"daily", "weekly"} else sma_fast
    above_fast = px > fast_ma
    stacked = (not np.isnan(sma_slow)) and fast_ma > sma_slow and px > fast_ma
    above_slow = (not np.isnan(sma_slow)) and px > sma_slow

    if np.isnan(rsi_val) or rsi_val > 82 or rsi_val < 32:
        return _fail(symbol, name, industry, px, f"{tf.label} RSI {rsi_val:.0f} is not a buy", tf.key)
    if not above_slow:
        return _fail(symbol, name, industry, px, f"below {tf.slow_ma_label} — trend not buyable", tf.key)

    last_n = m.tail(tf.squeeze_bars)
    prior_n = m.iloc[-(tf.squeeze_bars * 2) : -tf.squeeze_bars]
    range_ratio = float(last_n["range"].mean() / prior_n["range"].mean()) if prior_n["range"].mean() else np.nan
    bb_hist = m["bb_bw"].dropna()
    atr_pct = m["atr"] / close
    bb_pctile = float((bb_hist <= last["bb_bw"]).mean()) if len(bb_hist) else np.nan
    atr_pctile = float((atr_pct.dropna() <= atr_pct.iloc[-1]).mean()) if atr_pct.notna().any() else np.nan

    vol_avg = float(vol.tail(tf.volume_avg_bars).mean())
    base_vol = float(vol.iloc[-4:-1].mean()) if len(vol) >= 4 else float(vol.mean())
    vol_dryup = base_vol / vol_avg if vol_avg else np.nan
    vol_expand = float(last["Volume"]) / vol_avg if vol_avg else np.nan
    obv_slope = linear_slope(m["obv"], min(8, max(4, tf.squeeze_bars)))

    lows = low.tail(6).to_numpy()
    higher_lows = bool(lows[-1] > lows[0] and lows[-1] >= np.median(lows))
    bar_range = float(last["High"] - last["Low"])
    close_in_range = float((px - last["Low"]) / bar_range) if bar_range else 0.5
    upper_wick = float((last["High"] - max(px, last["Open"])) / bar_range) if bar_range else 0
    body_strength = abs(float(last["Close"] - last["Open"])) / bar_range if bar_range else 0
    near_long_high = dist_l <= 6.0

    rs_val = float("nan")
    rs_slope = float("nan")
    if nifty_bars is not None and len(nifty_bars) >= tf.rs_lookback + 1:
        aligned = pd.concat({"stock": close, "nifty": nifty_bars["Close"]}, axis=1).dropna()
        if len(aligned) >= tf.rs_lookback + 1:
            stock_ret = aligned["stock"].iloc[-1] / aligned["stock"].iloc[-1 - tf.rs_lookback] - 1
            nifty_ret = aligned["nifty"].iloc[-1] / aligned["nifty"].iloc[-1 - tf.rs_lookback] - 1
            rs_val = float((stock_ret - nifty_ret) * 100)
            rs_slope = linear_slope(aligned["stock"] / aligned["nifty"], tf.rs_lookback)

    macd_hist = float(last["hist"])
    macd_rising = float(last["hist"]) > float(prev["hist"])

    recent_high = float(high.tail(tf.pullback_bars).max())
    pullback_pct = (recent_high - px) / recent_high * 100 if recent_high else np.nan
    dist_fast = (px - fast_ma) / px * 100
    dist_slow = (px - sma_slow) / px * 100
    near_ma = min(abs(dist_fast), abs(dist_slow))
    holding_ma = (px >= fast_ma * 0.985) or (px >= sma_slow * 0.985)
    atr_val = float(last["atr"]) if not np.isnan(last["atr"]) else px * 0.02
    swing_low = float(low.tail(max(5, tf.squeeze_bars)).min())
    stop = min(swing_low, sma_slow if not np.isnan(sma_slow) else swing_low) * 0.997
    target = recent_high if px < recent_high else px + 2.0 * atr_val
    if dist_s > 0:
        target = max(target, min(high_s, px * 1.12))
    risk_pct = (px - stop) / px * 100
    reward_pct = (target - px) / px * 100
    rr = reward_pct / risk_pct if risk_pct > 0.35 else 0.0
    chase_lookback = min(len(m) - 1, max(5, tf.squeeze_bars * 2))
    recent_ret = float(px / close.iloc[-1 - chase_lookback] - 1) * 100 if len(m) > chase_lookback else 0.0

    pull_vol = float(vol.tail(max(4, tf.squeeze_bars)).mean())
    prior_vol = float(vol.iloc[-(tf.squeeze_bars * 4) : -tf.squeeze_bars].mean()) if len(vol) > tf.squeeze_bars * 4 else pull_vol
    vol_dry = pull_vol / prior_vol if prior_vol else np.nan

    breakout_ok = (
        above_fast
        and RSI_MIN <= rsi_val <= RSI_MAX
        and tf.min_dist_high_pct <= dist_s <= tf.max_dist_high_pct
    )
    breakout_score = 0.0
    if breakout_ok:
        b_scores = {
            "proximity": _proximity_score(dist_s, tf),
            "squeeze": _squeeze_score(range_ratio, bb_pctile, atr_pctile),
            "structure": _structure_score(above_fast, stacked, higher_lows, close_in_range, float(last["adx"])),
            "volume": _volume_score(vol_dryup, vol_expand, obv_slope),
            "momentum": _momentum_score(rsi_val, macd_hist, macd_rising),
            "relative_strength": _rs_score(rs_val, rs_slope),
            "breakout_quality": _quality_score(upper_wick, body_strength, near_long_high),
        }
        breakout_score = sum(b_scores[k] * WEIGHTS[k] for k in WEIGHTS)

    buy_ok = (
        above_slow
        and dist_s <= tf.max_dist_high_swing
        and tf.pullback_min_pct <= pullback_pct <= tf.pullback_max_pct
        and tf.rsi_buy_min <= rsi_val <= tf.rsi_buy_max
        and holding_ma
        and near_ma <= MA_SUPPORT_BAND_PCT + 1.5
        and rr >= MIN_RR
        and recent_ret < 14.0
        and px > stop
    )
    buy_zone_score = 0.0
    if buy_ok:
        mid_lo, mid_hi = 5.0, 12.0
        if mid_lo <= pullback_pct <= mid_hi:
            discount = 1.0
        elif pullback_pct < mid_lo:
            discount = _clip01((pullback_pct - tf.pullback_min_pct) / (mid_lo - tf.pullback_min_pct))
        else:
            discount = _clip01(1.0 - (pullback_pct - mid_hi) / (tf.pullback_max_pct - mid_hi))

        support = _clip01(1.0 - near_ma / MA_SUPPORT_BAND_PCT)
        if dist_fast < -1.5 and dist_slow < -1.5:
            support *= 0.5

        if 40 <= rsi_val <= 55:
            reset = 1.0
        elif rsi_val < 40:
            reset = _clip01((rsi_val - tf.rsi_buy_min) / max(1.0, 40 - tf.rsi_buy_min))
        else:
            reset = _clip01((tf.rsi_buy_max - rsi_val) / max(1.0, tf.rsi_buy_max - 55))
        if macd_rising:
            reset = min(1.0, reset + 0.15)

        rr_part = _clip01((rr - 1.2) / 1.8)
        upside = _clip01((reward_pct - 3) / 8)
        rr_score = 0.6 * rr_part + 0.4 * upside
        trend = 0.5 * (1.0 if stacked else 0.55) + 0.5 * (1.0 if 45 <= rsi_val <= 70 else 0.5)
        vol_part = _clip01((1.05 - vol_dry) / 0.4) if not np.isnan(vol_dry) else 0.4
        z_scores = {
            "trend": trend,
            "discount": discount,
            "support": support,
            "reset_momentum": reset,
            "reward_risk": rr_score,
            "relative_strength": _rs_score(rs_val, rs_slope),
            "volume": vol_part,
        }
        buy_zone_score = sum(z_scores[k] * BUY_ZONE_WEIGHTS[k] for k in BUY_ZONE_WEIGHTS)

    if breakout_score < 1 and buy_zone_score < 1:
        return _fail(
            symbol,
            name,
            industry,
            px,
            f"neither a {tf.label.lower()} breakout nor a buy zone",
            tf.key,
        )

    tags = []
    if breakout_score >= 50:
        tags.append("breakout")
    if buy_zone_score >= 50:
        tags.append("buy_zone")
    if not tags:
        tags.append("breakout" if breakout_score >= buy_zone_score else "buy_zone")
    setup = "+".join(tags)
    score = max(breakout_score, buy_zone_score)
    if breakout_score >= 55 and buy_zone_score >= 55:
        score = min(100.0, score + 5.0)

    reasons = []
    if "breakout" in setup:
        if 0 <= dist_s <= 4:
            reasons.append(f"Breakout: within {dist_s:.1f}% of {tf.high_label}")
        elif dist_s < 0:
            reasons.append(f"Breakout: fresh {abs(dist_s):.1f}% above {tf.high_label}")
        if range_ratio < 0.75:
            reasons.append(f"{tf.label} ranges contracting (squeeze)")
        if vol_expand > 1.15:
            reasons.append("Current bar volume expanding")
    if "buy_zone" in setup:
        reasons.append(
            f"Buy zone: {pullback_pct:.1f}% pullback into {tf.fast_ma_label} / {tf.slow_ma_label}"
        )
        reasons.append(
            f"R:R {rr:.1f} for {tf.hold_hint} (upside {reward_pct:.1f}% / risk {risk_pct:.1f}% to stop ₹{stop:.1f})"
        )
        if macd_rising:
            reasons.append(f"{tf.label} MACD histogram turning up")
    if stacked:
        reasons.append(f"Price > {tf.fast_ma_label} > {tf.slow_ma_label}")
    if higher_lows:
        reasons.append(f"Higher-low base on the {tf.label.lower()} chart")
    if 52 <= rsi_val <= 68:
        reasons.append(f"{tf.label} RSI {rsi_val:.0f} in a constructive zone")
    if not np.isnan(rs_val) and rs_val > 0:
        reasons.append(f"Outperforming Nifty over the last {tf.rs_lookback} {tf.key} bars ({rs_val:.1f}%)")
    if close_in_range >= 0.7 and "breakout" in setup:
        reasons.append("Holding in the upper third of the current bar")

    plan = build_trade_plan(
        px=px,
        setup=setup,
        score=score,
        high_s=high_s,
        high_l=high_l,
        recent_high=recent_high,
        swing_low=swing_low,
        sma_slow=sma_slow,
        atr_val=atr_val,
    )
    if plan["signal"] == "BUY":
        reasons.insert(0, f"Signal BUY around ₹{plan['entry']:.2f} | SL ₹{plan['sl']:.2f} | T1 ₹{plan['target']:.2f} | T2 ₹{plan['target_2']:.2f}")
    elif plan["signal"] == "BUY STOP":
        reasons.insert(
            0,
            f"Signal BUY STOP ₹{plan['entry']:.2f} (break of {tf.high_label}) | SL ₹{plan['sl']:.2f} | T1 ₹{plan['target']:.2f}",
        )
    else:
        reasons.insert(0, f"Signal WATCH — planned SL ₹{plan['sl']:.2f} / T1 ₹{plan['target']:.2f} (R:R {plan['rr']:.1f})")

    bars_tail = _ohlc_tail(m, tf.tail_bars, tf.date_fmt)
    metrics = {
        "timeframe": tf.key,
        "close": round(px, 2),
        "setup": setup,
        "signal": plan["signal"],
        "entry_type": plan["entry_type"],
        "entry": plan["entry"],
        "sl": plan["sl"],
        "target": plan["target"],
        "target_2": plan["target_2"],
        "breakout_score": round(breakout_score, 1),
        "buy_zone_score": round(buy_zone_score, 1),
        "high_lookback": round(high_s, 2),
        "dist_high_pct": round(dist_s, 2),
        "dist_long_high_pct": round(dist_l, 2) if not np.isnan(dist_l) else None,
        "dist_12m_pct": round(dist_s, 2),
        "pullback_pct": round(pullback_pct, 2) if not np.isnan(pullback_pct) else None,
        "pullback_3m_pct": round(pullback_pct, 2) if not np.isnan(pullback_pct) else None,
        "rsi_tf": round(rsi_val, 1),
        "rsi_m": round(rsi_val, 1),
        "adx": round(float(last["adx"]), 1) if not np.isnan(last["adx"]) else None,
        "macd_hist": round(macd_hist, 3),
        "range_ratio": round(range_ratio, 2) if not np.isnan(range_ratio) else None,
        "rs_pct": round(rs_val, 1) if not np.isnan(rs_val) else None,
        "rs_6m_pct": round(rs_val, 1) if not np.isnan(rs_val) else None,
        "adv_cr": round(adv_cr, 1),
        "ma_fast": round(fast_ma, 2),
        "ma_slow": round(sma_slow, 2) if not np.isnan(sma_slow) else None,
        "stop": plan["sl"],
        "target_1m": plan["target"],
        "reward_pct": plan["reward_pct"],
        "risk_pct": plan["risk_pct"],
        "rr": plan["rr"],
        "close_in_range": round(close_in_range, 2),
        "high_label": tf.high_label,
        "hold_hint": tf.hold_hint,
    }

    return BreakoutResult(
        symbol=symbol,
        name=name,
        industry=industry,
        passed=True,
        score=round(score, 1),
        close=px,
        setup=setup,
        timeframe=tf.key,
        breakout_score=round(breakout_score, 1),
        buy_zone_score=round(buy_zone_score, 1),
        metrics=metrics,
        reasons=reasons,
        monthly_tail=bars_tail,
        daily_tail=bars_tail if tf.key == "daily" else _ohlc_tail(daily, 30, "%Y-%m-%d"),
        bars_tail=bars_tail,
        signal=plan["signal"],
        entry=plan["entry"],
        sl=plan["sl"],
        target=plan["target"],
        target_2=plan["target_2"],
        entry_type=plan["entry_type"],
    )
