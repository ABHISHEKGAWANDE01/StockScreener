# NSE Intraday / Swing Screener

Python app that screens **NSE** stocks and proposes a trade plan. There are two styles only (no daily/weekly/monthly picker):

| Style | Chart | Vehicle | Validity |
| --- | --- | --- | --- |
| **Intraday** | 15-minute | Nearest listed NSE **CE** if the name is in F&O; otherwise **spot** | Same session. Square off by **15:15 IST**. Do not carry overnight. |
| **Swing** | Daily | Spot | **Hold until** date on each row (~2–4 weeks). Exit earlier at T1, T2, or SL. |

Every name must hold the slow moving average on that style’s chart. Setups are **breakout**, **buy_zone** (pullback in an uptrend), or both.

This is a research tool, **not investment advice**.

## Prerequisites

- Windows, Python 3.11+ (3.12 is fine)
- Internet access (Yahoo Finance prices, NSE lists / option chain, OpenAI if enabled)
- An OpenAI API key if you want AI confirmation

## Setup

```powershell
cd G:\StockScreener
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy the env template and add your key (never commit `.env`):

```powershell
copy .env.example .env
```

`.env`:

```
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o-mini
```

`OPENAI_MODEL` can be any chat model your account supports. Default is `gpt-4o-mini`.

## Run the dashboard (usual way)

With the venv active:

```powershell
cd G:\StockScreener
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Streamlit opens a browser (typically `http://localhost:8501`).

**Sidebar**

- **Universe:** `nifty50` (fast test), `nifty200`, or `nifty500`
- **Trade style:** Intraday or Swing
- **Minimum score:** technical cutoff (default 55)
- **Refresh prices:** ignore cache and re-download
- **OpenAI confirmation:** independently scores the shortlist
- **AI review top N:** how many names to send to the model (default 25)

Click **Run screener**. First run on Nifty 500 can take several minutes while prices download.

**Table columns (high level)**

- `signal` — BUY, BUY STOP, WATCH, or AVOID (AI reject)
- `instrument` — `option` or `spot`
- `option_contract` — e.g. `RELIANCE 21-Aug-2026 3000 CE`
- `hold_until` — 15:15 today (intraday) or a calendar date (swing)
- Spot **entry / SL / T1 / T2**; option **premium / SL / T1** when F&O is used
- `score` (rules), `ai_score`, `combined_score` (60% technical + 40% AI)

Pick a symbol to see the chart, levels, validity note, and (for intraday F&O) the options plan.

## Run from the command line

```powershell
cd G:\StockScreener
.\.venv\Scripts\Activate.ps1
python run.py --universe nifty50 --style swing --min-score 55
python run.py --universe nifty50 --style intraday --min-score 55 --ai-verify
python run.py --universe nifty500 --style swing --top 30 --ai-verify --ai-limit 25
```

| Flag | Meaning | Default |
| --- | --- | --- |
| `--universe` | `nifty50` / `nifty200` / `nifty500` | `nifty500` |
| `--style` | `intraday` or `swing` | `swing` |
| `--min-score` | Keep names at or above this technical score | `55` |
| `--force-download` | Ignore price cache | off |
| `--ai-verify` | Call OpenAI on the shortlist | off |
| `--ai-limit` | Max names sent to OpenAI | `25` |
| `--out` | CSV path | `data/cache/screener.csv` |
| `--top` | Rows printed to the console | `40` |

## How scoring works

**Breakout:** close near the lookback high, above the fast MA, squeeze (contracting ranges / ATR / Bollinger bandwidth), volume, RSI/MACD, relative strength vs Nifty 50.

**Buy zone:** still above the slow MA, a measured pullback into the fast/slow MAs, RSI reset, R:R at least about 1.2.

**Liquidity:** roughly ₹5 crore average daily value; very cheap names are dropped.

**Signals**

| Signal | Meaning |
| --- | --- |
| **BUY** | Take now at entry (buy-zone or already breaking out) |
| **BUY STOP** | Buy only if price trades through entry (break of the lookback high) |
| **WATCH** | Levels shown; wait for a better trigger or R:R |
| **AVOID** | OpenAI rejected the plan |

Stops sit under the recent swing / slow MA, padded with ATR. T1 is about 1.2R (intraday) or 1.6R (swing). T2 is the next extension.

**Intraday options:** uses the **nearest expiry actually listed on NSE** for that stock (HDFC Life is monthly only — last Tuesday of the month, e.g. 25 Aug 2026). The app no longer invents a weekly Thursday. Premium comes from the NSE chain when it loads. If the option is illiquid, use the spot plan. Option SL/T1 are a ~0.5 delta mapping from the spot plan, not a live option model.

**Swing validity:** hold-until is a business-day count from today (shorter for pure breakouts, longer for buy-zone). Recheck the name after that date.

## Data and cache

| Source | Use |
| --- | --- |
| NSE / Wikipedia / embedded Nifty 50 | Universe |
| Yahoo Finance | Daily and 15-minute OHLC (`*.NS`) |
| NSE F&O lot file + option-chain API | Which names are F&O, ATM CE quote |
| OpenAI | Confirmation score and verdict |

Cached under `data/cache/` (gitignored):

- Universe JSON — 24 hours
- Daily parquet — 12 hours
- 15-minute parquet — 1 hour
- F&O lots — 24 hours
- Option chains — 15 minutes

Use **Refresh prices** or `--force-download` to bypass the price cache.

## Project layout

```
app.py                 Streamlit dashboard
run.py                 CLI
config.py              Style specs, weights, cache TTLs
analysis/breakout.py   Indicators → score → entry/SL/targets → validity
analysis/screener.py   Universe loop, options attach, AI
analysis/ai_review.py  OpenAI JSON review
analysis/indicators.py SMA, EMA, RSI, ATR, MACD, ADX, Donchian, …
data/universe.py       Nifty 50 / 200 / 500
data/prices.py         Yahoo download + resample
data/fo.py             F&O lots and CE suggestion
```

## GitHub

https://github.com/ABHISHEKGAWANDE01/StockScreener

After cloning, copy `.env.example` to `.env` and add `OPENAI_API_KEY`. The key is not stored in the repo.
