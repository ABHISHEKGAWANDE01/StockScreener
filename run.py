from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from analysis.screener import run_screener
from config import DEFAULT_STYLE, DEFAULT_UNIVERSE, STYLES
from utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NSE buy screener: Intraday (15m, F&O CE if listed) or Swing (daily, with hold-until)."
    )
    parser.add_argument(
        "--universe",
        default=DEFAULT_UNIVERSE,
        choices=["nifty50", "nifty200", "nifty500"],
    )
    parser.add_argument(
        "--style",
        default=DEFAULT_STYLE,
        choices=list(STYLES.keys()),
        help="intraday or swing",
    )
    parser.add_argument("--min-score", type=float, default=55.0)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--ai-verify", action="store_true", help="Ask OpenAI to score the shortlist")
    parser.add_argument("--ai-limit", type=int, default=25)
    parser.add_argument("--out", default="data/cache/screener.csv")
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    setup_logging(logging.INFO)
    df, _ = run_screener(
        args.universe,
        args.min_score,
        args.force_download,
        ai_verify=args.ai_verify,
        ai_limit=args.ai_limit,
        style=args.style,
    )
    if df.empty:
        print("No stocks passed the filters. Try the other style or lower --min-score.")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    cols = [
        c
        for c in [
            "symbol",
            "style",
            "signal",
            "instrument",
            "option_contract",
            "hold_until",
            "entry",
            "sl",
            "target",
            "target_2",
            "setup",
            "score",
            "ai_score",
            "combined_score",
            "ai_verdict",
            "rr",
            "close",
        ]
        if c in df.columns
    ]
    print(df[cols].head(args.top).to_string(index=False))
    print(f"\nSaved {len(df)} names to {out}")


if __name__ == "__main__":
    main()
