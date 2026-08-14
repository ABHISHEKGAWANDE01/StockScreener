from __future__ import annotations

import logging

import pandas as pd

from analysis.ai_review import review_shortlist
from analysis.breakout import BreakoutResult, analyze_symbol, apply_vehicle
from config import DEFAULT_STYLE, NIFTY_BENCHMARK, get_style
from data.fo import load_fo_lots, suggest_long_option
from data.prices import download_many, resample_ohlc
from data.universe import load_universe

logger = logging.getLogger(__name__)


def _rows_from_passed(passed: list[BreakoutResult]) -> list[dict]:
    rows = []
    for r in passed:
        rows.append(
            {
                "symbol": r.symbol,
                "name": r.name,
                "industry": r.industry,
                "style": r.style or r.timeframe,
                "timeframe": r.timeframe,
                "setup": r.setup,
                "signal": r.signal,
                "instrument": r.instrument,
                "option_contract": r.option_contract,
                "option_premium": r.option_premium,
                "option_sl": r.option_sl,
                "option_target": r.option_target,
                "lot_size": r.lot_size,
                "hold_until": r.hold_until,
                "validity_note": r.validity_note,
                "entry": r.entry,
                "sl": r.sl,
                "target": r.target,
                "target_2": r.target_2,
                "score": r.score,
                "breakout_score": r.breakout_score,
                "buy_zone_score": r.buy_zone_score,
                "ai_score": r.ai_score,
                "ai_verdict": r.ai_verdict,
                "ai_confidence": r.ai_confidence,
                "combined_score": r.combined_score,
                "close": r.metrics.get("close"),
                "dist_high_pct": r.metrics.get("dist_high_pct"),
                "dist_12m_pct": r.metrics.get("dist_high_pct"),
                "rsi_tf": r.metrics.get("rsi_tf"),
                "rsi_m": r.metrics.get("rsi_tf"),
                "rs_pct": r.metrics.get("rs_pct"),
                "rs_6m_pct": r.metrics.get("rs_pct"),
                "pullback_pct": r.metrics.get("pullback_pct"),
                "pullback_3m_pct": r.metrics.get("pullback_pct"),
                "rr": r.metrics.get("rr"),
                "reward_pct": r.metrics.get("reward_pct"),
                "risk_pct": r.metrics.get("risk_pct"),
                "stop": r.metrics.get("stop"),
                "target_1m": r.metrics.get("target"),
                "high_label": r.metrics.get("high_label"),
                "adv_cr": r.metrics.get("adv_cr"),
                "why": "; ".join(r.reasons),
                "ai_rationale": r.ai_rationale,
                "ai_risks": "; ".join(r.ai_risks),
            }
        )
    return rows


def _attach_options(passed: list[BreakoutResult]) -> None:
    if not passed:
        return
    lots = load_fo_lots()
    for r in passed:
        vehicle = suggest_long_option(r.symbol, r.close or r.entry or 0, lots)
        apply_vehicle(r, vehicle)


def run_screener(
    universe: str = "nifty500",
    min_score: float = 55.0,
    force_download: bool = False,
    ai_verify: bool = False,
    ai_limit: int | None = 25,
    style: str | None = None,
    timeframe: str | None = None,
) -> tuple[pd.DataFrame, list[BreakoutResult]]:
    spec = get_style(style or timeframe or DEFAULT_STYLE)
    uni = load_universe(universe)
    tickers = uni["yf_ticker"].tolist() + [NIFTY_BENCHMARK]

    daily_prices = download_many(tickers, force=force_download, interval="1d")
    if spec.interval == "1d":
        style_prices = daily_prices
    else:
        style_prices = download_many(
            tickers,
            force=force_download,
            interval=spec.interval,
            period=spec.download_period,
        )

    nifty_daily = daily_prices.get(NIFTY_BENCHMARK, pd.DataFrame())
    nifty_bars = style_prices.get(NIFTY_BENCHMARK, pd.DataFrame())
    if nifty_bars.empty and not nifty_daily.empty:
        nifty_bars = resample_ohlc(nifty_daily, spec.resample)

    results: list[BreakoutResult] = []
    for row in uni.itertuples(index=False):
        daily = daily_prices.get(row.yf_ticker)
        bars = style_prices.get(row.yf_ticker)
        if daily is None or daily.empty:
            results.append(
                BreakoutResult(
                    row.symbol,
                    row.name,
                    row.industry,
                    False,
                    0,
                    0,
                    "no price data",
                    timeframe=spec.key,
                    style=spec.key,
                )
            )
            continue
        if spec.interval != "1d" and (bars is None or bars.empty):
            results.append(
                BreakoutResult(
                    row.symbol,
                    row.name,
                    row.industry,
                    False,
                    0,
                    0,
                    "no 15-minute bars",
                    timeframe=spec.key,
                    style=spec.key,
                )
            )
            continue
        if bars is None or bars.empty:
            bars = resample_ohlc(daily, spec.resample)
        results.append(
            analyze_symbol(
                row.symbol,
                row.name,
                row.industry,
                daily,
                bars,
                nifty_bars if not nifty_bars.empty else None,
                spec,
            )
        )

    passed = [r for r in results if r.passed and r.score >= min_score]
    passed.sort(key=lambda r: r.score, reverse=True)
    logger.info(
        "Screened %s names for %s, %s passed min_score=%.0f",
        len(results),
        spec.key,
        len(passed),
        min_score,
    )

    if spec.key == "intraday":
        _attach_options(passed)

    if ai_verify and passed:
        to_review = passed[:ai_limit] if ai_limit else passed
        reviewed = review_shortlist(to_review)
        by_symbol = {r.symbol: r for r in reviewed}
        passed = [by_symbol.get(r.symbol, r) for r in passed]
        passed.sort(
            key=lambda r: (r.combined_score is not None, r.combined_score or r.score),
            reverse=True,
        )
        for r in reviewed:
            for i, original in enumerate(results):
                if original.symbol == r.symbol:
                    results[i] = r
                    break

    df = pd.DataFrame(_rows_from_passed(passed))
    return df, results
