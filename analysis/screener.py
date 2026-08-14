from __future__ import annotations

import logging

import pandas as pd

from analysis.ai_review import review_shortlist
from analysis.breakout import BreakoutResult, analyze_symbol
from config import DEFAULT_TIMEFRAME, NIFTY_BENCHMARK, get_timeframe
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
                "timeframe": r.timeframe,
                "setup": r.setup,
                "signal": r.signal,
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
                "target": r.metrics.get("target"),
                "target_1m": r.metrics.get("target"),
                "high_label": r.metrics.get("high_label"),
                "adv_cr": r.metrics.get("adv_cr"),
                "why": "; ".join(r.reasons),
                "ai_rationale": r.ai_rationale,
                "ai_risks": "; ".join(r.ai_risks),
            }
        )
    return rows


def run_screener(
    universe: str = "nifty500",
    min_score: float = 55.0,
    force_download: bool = False,
    ai_verify: bool = False,
    ai_limit: int | None = 25,
    timeframe: str = DEFAULT_TIMEFRAME,
) -> tuple[pd.DataFrame, list[BreakoutResult]]:
    tf = get_timeframe(timeframe)
    uni = load_universe(universe)
    tickers = uni["yf_ticker"].tolist() + [NIFTY_BENCHMARK]
    prices = download_many(tickers, force=force_download)

    nifty_daily = prices.get(NIFTY_BENCHMARK, pd.DataFrame())
    nifty_bars = resample_ohlc(nifty_daily, tf.resample) if not nifty_daily.empty else pd.DataFrame()

    results: list[BreakoutResult] = []
    for row in uni.itertuples(index=False):
        daily = prices.get(row.yf_ticker)
        if daily is None or daily.empty:
            results.append(
                BreakoutResult(
                    row.symbol, row.name, row.industry, False, 0, 0, "no price data", timeframe=tf.key
                )
            )
            continue
        bars = resample_ohlc(daily, tf.resample)
        results.append(
            analyze_symbol(
                row.symbol,
                row.name,
                row.industry,
                daily,
                bars,
                nifty_bars if not nifty_bars.empty else None,
                tf,
            )
        )

    passed = [r for r in results if r.passed and r.score >= min_score]
    passed.sort(key=lambda r: r.score, reverse=True)
    logger.info(
        "Screened %s names on %s, %s passed min_score=%.0f",
        len(results),
        tf.key,
        len(passed),
        min_score,
    )

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
