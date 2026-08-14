from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analysis.indicators import ema, sma
from analysis.screener import run_screener
from config import DEFAULT_TIMEFRAME, DEFAULT_UNIVERSE, TIMEFRAMES, get_timeframe
from data.prices import download_one, resample_ohlc
from data.universe import load_universe


st.set_page_config(page_title="NSE Timeframe Buy Screener", layout="wide")
st.title("NSE Timeframe Buy Screener")
st.caption(
    "Pick daily, weekly, or monthly. The scan looks for breakouts under the lookback high "
    "and pullbacks into the fast/slow moving averages on that same timeframe. "
    "OpenAI then independently scores the shortlist."
)

with st.sidebar:
    st.header("Scan")
    universe = st.selectbox("Universe", ["nifty50", "nifty200", "nifty500"], index=2)
    timeframe = st.selectbox(
        "Timeframe",
        options=list(TIMEFRAMES.keys()),
        index=list(TIMEFRAMES.keys()).index(DEFAULT_TIMEFRAME),
        format_func=lambda k: TIMEFRAMES[k].label,
    )
    tf_info = get_timeframe(timeframe)
    st.caption(
        f"Breakout vs {tf_info.high_label}. Buy zone vs {tf_info.fast_ma_label} / {tf_info.slow_ma_label}. "
        f"Hold window: {tf_info.hold_hint}."
    )
    min_score = st.slider("Minimum score", 40, 85, 55)
    force = st.checkbox("Refresh prices (ignore cache)", value=False)
    ai_verify = st.checkbox("OpenAI confirmation on shortlist", value=True)
    ai_limit = st.slider("AI review top N", 5, 50, 25)
    run = st.button("Run screener", type="primary")

if "last_df" not in st.session_state:
    st.session_state.last_df = pd.DataFrame()
    st.session_state.last_results = []
    st.session_state.last_timeframe = DEFAULT_TIMEFRAME

if run:
    with st.spinner(f"Scoring {tf_info.label.lower()} breakouts and buy zones…"):
        df, results = run_screener(
            universe or DEFAULT_UNIVERSE,
            float(min_score),
            force,
            ai_verify=ai_verify,
            ai_limit=int(ai_limit),
            timeframe=timeframe,
        )
    st.session_state.last_df = df
    st.session_state.last_results = results
    st.session_state.last_timeframe = timeframe

df = st.session_state.last_df
active_tf = get_timeframe(st.session_state.get("last_timeframe", timeframe))
if df.empty:
    st.info("Choose universe and timeframe, then click **Run screener**. Start with Nifty 50 to test quickly.")
    st.markdown(
        f"""
**Analysis timeframe:** {tf_info.label}

| Setup | What it looks for on this timeframe |
| --- | --- |
| **breakout** | Coil near {tf_info.high_label}, above {tf_info.fast_ma_label}, squeeze + volume + RS |
| **buy_zone** | Uptrend (above {tf_info.slow_ma_label}), dip into the MAs, RSI reset, R:R ≥ 1.2 |
| **BUY** | Take now at **entry** (buy-zone or already breaking out) |
| **BUY STOP** | Buy only if price trades through **entry** (break of the lookback high) |
| **WATCH** | Levels shown, wait for a better trigger or R:R |
| **AVOID** | OpenAI rejected the plan |

**SL** is below the recent swing / slow MA, padded with ATR. **T1** is ~1.6R (or the nearby high). **T2** is the next extension.

Liquidity floor is about ₹5 crore ADV. **Combined score** = 60% technical + 40% AI.
"""
    )
    st.stop()

view = df
f1, f2 = st.columns(2)
with f1:
    if "setup" in df.columns:
        kinds = ["all"] + sorted({s for s in df["setup"].dropna().unique()})
        picked_setup = st.selectbox("Filter by setup", kinds)
        if picked_setup != "all":
            view = view[view["setup"] == picked_setup]
with f2:
    if "signal" in view.columns:
        sigs = ["all"] + sorted({s for s in view["signal"].dropna().unique()})
        picked_sig = st.selectbox("Filter by signal", sigs)
        if picked_sig != "all":
            view = view[view["signal"] == picked_sig]

if view.empty:
    st.warning("No rows match those filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Timeframe", active_tf.label)
c2.metric("Setups found", len(view))
c3.metric("Top technical score", f"{view['score'].max():.0f}" if len(view) else "—")
if "ai_score" in view.columns and view["ai_score"].notna().any():
    c4.metric("Top AI score", f"{view['ai_score'].max():.0f}")
elif "rr" in view.columns and view["rr"].notna().any():
    c4.metric("Median R:R", f"{view['rr'].median():.1f}")
else:
    c4.metric("AI reviews", "off")

display_cols = [
    "symbol",
    "name",
    "signal",
    "entry",
    "sl",
    "target",
    "target_2",
    "rr",
    "setup",
    "timeframe",
    "score",
    "ai_score",
    "combined_score",
    "ai_verdict",
    "close",
    "pullback_pct",
    "dist_high_pct",
    "rsi_tf",
    "ai_rationale",
    "why",
]
st.dataframe(
    view[[c for c in display_cols if c in view.columns]],
    use_container_width=True,
    hide_index=True,
    column_config={
        "signal": st.column_config.TextColumn("signal"),
        "entry": st.column_config.NumberColumn("entry", format="₹%.2f"),
        "sl": st.column_config.NumberColumn("SL", format="₹%.2f"),
        "target": st.column_config.NumberColumn("T1", format="₹%.2f"),
        "target_2": st.column_config.NumberColumn("T2", format="₹%.2f"),
        "score": st.column_config.ProgressColumn("technical", min_value=0, max_value=100, format="%.1f"),
        "ai_score": st.column_config.ProgressColumn("AI score", min_value=0, max_value=100, format="%.1f"),
        "combined_score": st.column_config.ProgressColumn("combined", min_value=0, max_value=100, format="%.1f"),
        "dist_high_pct": st.column_config.NumberColumn(f"% below {active_tf.high_label}", format="%.1f"),
        "ai_rationale": st.column_config.TextColumn("AI rationale", width="large"),
        "why": st.column_config.TextColumn("technical why", width="large"),
    },
)

symbols = view["symbol"].tolist()
picked = st.selectbox("Chart a setup", symbols)
if picked:
    uni = load_universe(universe)
    row = uni[uni["symbol"] == picked].iloc[0]
    daily = download_one(row["yf_ticker"])
    bars = resample_ohlc(daily, active_tf.resample).tail(active_tf.chart_bars)
    fast = ema(bars["Close"], active_tf.ma_fast) if active_tf.key in {"daily", "weekly"} else sma(bars["Close"], active_tf.ma_fast)
    slow = sma(bars["Close"], active_tf.ma_slow)
    don = bars["High"].shift(1).rolling(active_tf.donchian_short).max()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
        subplot_titles=(f"{picked} {active_tf.label.lower()}", "Volume"),
    )
    fig.add_trace(
        go.Candlestick(
            x=bars.index,
            open=bars["Open"],
            high=bars["High"],
            low=bars["Low"],
            close=bars["Close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=bars.index, y=fast, name=active_tf.fast_ma_label, line=dict(width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bars.index, y=slow, name=active_tf.slow_ma_label, line=dict(width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bars.index, y=don, name=f"Prior {active_tf.high_label}", line=dict(dash="dot", width=1)), row=1, col=1)
    match_levels = next((r for r in st.session_state.last_results if r.symbol == picked), None)
    if match_levels and match_levels.entry:
        fig.add_hline(y=match_levels.entry, line_dash="dash", line_color="#2ecc71", annotation_text="Entry", row=1, col=1)
        fig.add_hline(y=match_levels.sl, line_dash="dash", line_color="#e74c3c", annotation_text="SL", row=1, col=1)
        fig.add_hline(y=match_levels.target, line_dash="dash", line_color="#3498db", annotation_text="T1", row=1, col=1)
        if match_levels.target_2:
            fig.add_hline(y=match_levels.target_2, line_dash="dot", line_color="#9b59b6", annotation_text="T2", row=1, col=1)
    fig.add_trace(go.Bar(x=bars.index, y=bars["Volume"], name="Volume", marker_color="#6b7c93"), row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, height=640, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    match = next((r for r in st.session_state.last_results if r.symbol == picked), None)
    if match:
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Signal", match.signal or "—")
        s2.metric("Entry", f"₹{match.entry:.2f}" if match.entry else "—")
        s3.metric("SL", f"₹{match.sl:.2f}" if match.sl else "—")
        s4.metric("T1", f"₹{match.target:.2f}" if match.target else "—")
        s5.metric("T2", f"₹{match.target_2:.2f}" if match.target_2 else "—")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Setup", match.setup or "—")
        m2.metric("Breakout", f"{match.breakout_score:.0f}")
        m3.metric("Buy zone", f"{match.buy_zone_score:.0f}")
        rr = match.metrics.get("rr")
        m4.metric("R:R to T1", f"{rr:.1f}" if rr is not None else "—")
        if match.ai_score is not None:
            st.subheader("OpenAI confirmation")
            a1, a2, a3 = st.columns(3)
            a1.metric("AI score", f"{match.ai_score:.0f}")
            a2.metric("Verdict", match.ai_verdict or "—")
            a3.metric("Combined", f"{match.combined_score:.0f}" if match.combined_score is not None else "—")
            st.write(match.ai_rationale)
            if match.ai_risks:
                st.caption("Risks: " + " · ".join(match.ai_risks))
        st.subheader("Why this name scored")
        for reason in match.reasons:
            st.write(f"- {reason}")
        st.json(match.metrics)

st.caption("Educational screener, not investment advice. Size positions off the suggested stop, not the score.")
