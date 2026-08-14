from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from analysis.breakout import BreakoutResult, apply_ai_to_signal
from config import AI_MODEL_WEIGHT, AI_REVIEW_WORKERS, AI_TECHNICAL_WEIGHT

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a disciplined NSE swing trader.
The candidate was scored on a chosen analysis timeframe (daily, weekly, or monthly).
Each name is tagged breakout, buy_zone (pullback in an uptrend), or both.
You receive quantitative metrics, a rules-based score, OHLC on the selected timeframe, and daily bars when provided.

Score 0-100 for long quality on THAT timeframe (asymmetric reward vs nearby support):
- 80-100: trend intact, coiling under the lookback high or sitting on the fast/slow MAs after a clean dip, R:R >= 2, RS positive
- 65-79: valid swing, one or two blemishes
- 50-64: mixed; watchlist only
- below 50: reject — falling knife, exhausted RSI, poor R:R, or broken trend vs the slow MA

Be skeptical of already-extended rips, sloppy ranges, RSI > 75, underperformance vs Nifty, closes below the slow MA, R:R under 1.5 for buy-zone names.

Return JSON only with keys:
ai_score (number 0-100),
verdict (one of confirm, watch, reject),
confidence (number 0-100),
rationale (2-4 sentences focused on reward vs risk on this timeframe),
risks (array of short strings).
The technical plan already includes entry, sl, target, target_2 — comment if those levels look wrong, but do not invent a new trade unless the structure is broken.
"""


def _client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is missing. In the project venv run: pip install openai python-dotenv"
        ) from exc
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing. Put it in a .env file in the project root.")
    return OpenAI(api_key=key)


def _payload(result: BreakoutResult) -> dict:
    return {
        "symbol": result.symbol,
        "name": result.name,
        "industry": result.industry,
        "timeframe": result.timeframe,
        "setup": result.setup,
        "technical_score": result.score,
        "breakout_score": result.breakout_score,
        "buy_zone_score": result.buy_zone_score,
        "metrics": result.metrics,
        "technical_reasons": result.reasons,
        "timeframe_ohlc": result.bars_tail or result.monthly_tail,
        "daily_ohlc": result.daily_tail,
    }


def review_one(result: BreakoutResult, model: str) -> BreakoutResult:
    client = _client()
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Verify this NSE swing candidate on the stated timeframe (breakout and/or buy zone). "
                    "Use only the data given.\n\n"
                    + json.dumps(_payload(result), ensure_ascii=False)
                ),
            },
        ],
    )
    raw = response.choices[0].message.content or "{}"
    data = json.loads(raw)
    ai_score = float(data.get("ai_score", 0))
    result.ai_score = round(ai_score, 1)
    result.ai_verdict = str(data.get("verdict", "watch")).lower()
    conf = data.get("confidence")
    result.ai_confidence = round(float(conf), 1) if conf is not None else None
    result.ai_rationale = str(data.get("rationale", "")).strip()
    risks = data.get("risks") or []
    result.ai_risks = [str(r) for r in risks] if isinstance(risks, list) else []
    result.combined_score = round(
        AI_TECHNICAL_WEIGHT * result.score + AI_MODEL_WEIGHT * result.ai_score, 1
    )
    return apply_ai_to_signal(result)


def review_shortlist(
    results: list[BreakoutResult],
    model: str | None = None,
    workers: int = AI_REVIEW_WORKERS,
) -> list[BreakoutResult]:
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not results:
        return results

    logger.info("OpenAI reviewing %s shortlisted names with %s", len(results), model)
    reviewed: dict[str, BreakoutResult] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(review_one, r, model): r.symbol for r in results}
        for fut in as_completed(futs):
            symbol = futs[fut]
            try:
                reviewed[symbol] = fut.result()
            except Exception as exc:
                logger.warning("AI review failed for %s: %s", symbol, exc)
                original = next(r for r in results if r.symbol == symbol)
                original.ai_rationale = f"AI review failed: {exc}"
                original.ai_verdict = "error"
                reviewed[symbol] = original

    return [reviewed[r.symbol] for r in results]
