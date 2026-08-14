# NSE 1-Month Buy Screener

Finds liquid **NSE** names that are reasonable longs on the **timeframe you pick**: daily, weekly, or monthly.

Every name must hold the slow MA on that timeframe. Then it can qualify as:

- **breakout** — coiling near the lookback high (20-day / 13-week / 12-month)
- **buy_zone** — pullback into the fast/slow MAs with usable R:R
- **breakout+buy_zone** — both

Buy-zone extras: 200-day SMA, 1-month relative strength vs Nifty, volume dry-up on the dip, suggested stop and 1-month target.

## Setup

```powershell
cd G:\StockScreener
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. Do not commit the key.

## CLI

```powershell
python run.py --universe nifty50 --timeframe weekly --min-score 55
python run.py --universe nifty500 --timeframe monthly --min-score 60 --top 30 --ai-verify
```

CSV is written to `data/cache/monthly_breakout.csv`. Daily prices are cached for 12 hours.

## Dashboard

```powershell
streamlit run app.py
```

Filter the table by setup. Charts have a monthly tab (trend) and a daily tab (20/50-day buy zone). OpenAI reviews the shortlist using metrics plus monthly and daily bars.

**Combined score** = 60% technical + 40% AI.

This is a research tool, not investment advice.
