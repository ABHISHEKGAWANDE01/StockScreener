from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from analysis.breakout import BreakoutResult, apply_ai_to_signal
from config import AI_MODEL_WEIGHT, AI_REVIEW_WORKERS, AI_TECHNICAL_WEIGHT

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a disciplined NSE trader reviewing a rules-based shortlist.
Style is either intraday (15-minute chart, square off same day) or swing (daily chart, hold days to weeks).
Each name is tagged breakout, buy_zone, or both.
Intraday names may include an NSE F&O call suggestion; if missing, the plan is spot/cash.

Score 0-100 for long quality on THAT style:
- 80-100: trend intact, clean location, R:R usable, RS positive
- 65-79: valid trade, one or two blemishes
- 50-64: mixed; watchlist only
- below 50: reject — falling knife, exhausted RSI, poor R:R, or broken trend

Be skeptical of already-extended rips, sloppy ranges, RSI > 75, underperformance vs Nifty, closes below the slow MA.
For intraday, reject if the move already looks late in the session or if option liquidity would be poor vs spot.
For swing, comment on whether the hold-until date matches the structure.

Return JSON only with keys:
ai_score (number 0-100),
verdict (one of confirm, watch, reject),
confidence (number 0-100),
rationale (2-4 sentences on reward vs risk for this style),
risks (array of short strings).
The technical plan already includes entry, sl, target, target_2, hold_until, and optional option_contract — comment if those look wrong, but do not invent a new trade unless structure is broken.
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
        "style": result.style or result.timeframe,
        "timeframe": result.timeframe,
        "setup": result.setup,
        "instrument": result.instrument,
        "option_contract": result.option_contract,
        "hold_until": result.hold_until,
        "validity_note": result.validity_note,
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
                    "Verify this NSE candidate for the stated style (intraday or swing). "
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
