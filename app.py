from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analysis.indicators import ema, sma
from analysis.screener import run_screener
from config import DEFAULT_STYLE, DEFAULT_UNIVERSE, STYLES, get_style
from data.prices import download_for_style, download_one
from data.universe import load_universe


st.set_page_config(page_title="NSE Intraday / Swing Screener", layout="wide")
st.title("NSE Intraday / Swing Screener")
st.caption(
    "Two modes only: Intraday (15-minute chart, F&O call if listed, else spot) "
    "and Swing (daily chart with an explicit hold-until date). "
    "OpenAI independently scores the shortlist."
)

with st.sidebar:
    st.header("Scan")
    universe = st.selectbox("Universe", ["nifty50", "nifty200", "nifty500"], index=2)
    style = st.radio(
        "Trade style",
        options=list(STYLES.keys()),
        index=list(STYLES.keys()).index(DEFAULT_STYLE),
        format_func=lambda k: STYLES[k].label,
    )
    tf_info = get_style(style)
    if style == "intraday":
        st.caption(
            "15-minute breakout / buy-zone. Uses the nearest listed NSE CE when the stock is in F&O; "
            "otherwise the cash/spot plan. Square off by 15:15 IST."
        )
    else:
        st.caption(
            f"Daily swing: breakout vs {tf_info.high_label}, buy zone vs "
            f"{tf_info.fast_ma_label} / {tf_info.slow_ma_label}. Each row gets a hold-until date."
        )
    min_score = st.slider("Minimum score", 40, 85, 55)
    force = st.checkbox("Refresh prices (ignore cache)", value=False)
    ai_verify = st.checkbox("OpenAI confirmation on shortlist", value=True)
    ai_limit = st.slider("AI review top N", 5, 50, 25)
    run = st.button("Run screener", type="primary")

if "last_df" not in st.session_state:
    st.session_state.last_df = pd.DataFrame()
    st.session_state.last_results = []
    st.session_state.last_style = DEFAULT_STYLE

if run:
    with st.spinner(f"Scoring {tf_info.label.lower()} setups…"):
        df, results = run_screener(
            universe or DEFAULT_UNIVERSE,
            float(min_score),
            force,
            ai_verify=ai_verify,
            ai_limit=int(ai_limit),
            style=style,
        )
    st.session_state.last_df = df
    st.session_state.last_results = results
    st.session_state.last_style = style

df = st.session_state.last_df
active = get_style(st.session_state.get("last_style", style))
if df.empty:
    st.info("Choose universe and style, then click **Run screener**. Start with Nifty 50 to test quickly.")
    st.markdown(
        f"""
**Style:** {tf_info.label}

| Setup | What it looks for |
| --- | --- |
| **breakout** | Coil near {tf_info.high_label}, above {tf_info.fast_ma_label} |
| **buy_zone** | Uptrend (above {tf_info.slow_ma_label}), dip into the MAs, usable R:R |
| **BUY** | Take now at **entry** |
| **BUY STOP** | Buy only if price trades through **entry** |
| **WATCH** | Levels shown — wait |
| **AVOID** | OpenAI rejected the plan |

**Intraday:** F&O names get the nearest **listed** NSE **CE**. Others use **spot**. Valid until **15:15 IST** the same session.

**Swing:** Spot plan plus **hold until** (about 2–4 weeks). Exit at T1/T2, SL, or that date.

Liquidity floor is about ₹5 crore ADV. **Combined score** = 60% technical + 40% AI.
"""
    )
    st.stop()

view = df
f1, f2, f3 = st.columns(3)
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
with f3:
    if "instrument" in view.columns:
        insts = ["all"] + sorted({s for s in view["instrument"].dropna().unique()})
        picked_inst = st.selectbox("Filter by instrument", insts)
        if picked_inst != "all":
            view = view[view["instrument"] == picked_inst]

if view.empty:
    st.warning("No rows match those filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Style", active.label)
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
    "instrument",
    "option_contract",
    "hold_until",
    "entry",
    "sl",
    "target",
    "target_2",
    "option_premium",
    "option_sl",
    "option_target",
    "rr",
    "setup",
    "score",
    "ai_score",
    "combined_score",
    "ai_verdict",
    "close",
    "validity_note",
    "ai_rationale",
    "why",
]
st.dataframe(
    view[[c for c in display_cols if c in view.columns]],
    use_container_width=True,
    hide_index=True,
    column_config={
        "signal": st.column_config.TextColumn("signal"),
        "instrument": st.column_config.TextColumn("vehicle"),
        "option_contract": st.column_config.TextColumn("option"),
        "hold_until": st.column_config.TextColumn("valid / hold until"),
        "entry": st.column_config.NumberColumn("spot entry", format="₹%.2f"),
        "sl": st.column_config.NumberColumn("spot SL", format="₹%.2f"),
        "target": st.column_config.NumberColumn("spot T1", format="₹%.2f"),
        "target_2": st.column_config.NumberColumn("spot T2", format="₹%.2f"),
        "option_premium": st.column_config.NumberColumn("opt ~premium", format="₹%.2f"),
        "option_sl": st.column_config.NumberColumn("opt SL", format="₹%.2f"),
        "option_target": st.column_config.NumberColumn("opt T1", format="₹%.2f"),
        "score": st.column_config.ProgressColumn("technical", min_value=0, max_value=100, format="%.1f"),
        "ai_score": st.column_config.ProgressColumn("AI score", min_value=0, max_value=100, format="%.1f"),
        "combined_score": st.column_config.ProgressColumn("combined", min_value=0, max_value=100, format="%.1f"),
        "validity_note": st.column_config.TextColumn("validity", width="large"),
        "ai_rationale": st.column_config.TextColumn("AI rationale", width="large"),
        "why": st.column_config.TextColumn("technical why", width="large"),
    },
)

def _nse_session_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Keep NSE cash-session stamps only (09:15–15:30 IST, Mon–Fri)."""
    if df.empty:
        return df
    idx = pd.DatetimeIndex(df.index)
    minutes = idx.hour * 60 + idx.minute
    mask = (idx.dayofweek < 5) & (minutes >= 9 * 60 + 15) & (minutes <= 15 * 60 + 30)
    return df.loc[mask]


symbols = view["symbol"].tolist()
picked = st.selectbox("Chart a setup", symbols)
if picked:
    uni = load_universe(universe)
    row = uni[uni["symbol"] == picked].iloc[0]
    bars = download_for_style(row["yf_ticker"], active).tail(active.chart_bars)
    if bars.empty:
        bars = download_one(row["yf_ticker"]).tail(active.chart_bars)
    fast = ema(bars["Close"], active.ma_fast) if active.use_ema else sma(bars["Close"], active.ma_fast)
    slow = sma(bars["Close"], active.ma_slow)
    don = bars["High"].shift(1).rolling(active.donchian_short).max()
    if active.interval.endswith("m"):
        bars = _nse_session_bars(bars)
        fast = fast.reindex(bars.index)
        slow = slow.reindex(bars.index)
        don = don.reindex(bars.index)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
        subplot_titles=(f"{picked} · {active.label}", "Volume"),
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
    fig.add_trace(go.Scatter(x=bars.index, y=fast, name=active.fast_ma_label, line=dict(width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bars.index, y=slow, name=active.slow_ma_label, line=dict(width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bars.index, y=don, name=f"Prior {active.high_label}", line=dict(dash="dot", width=1)), row=1, col=1)
    match_levels = next((r for r in st.session_state.last_results if r.symbol == picked), None)
    if match_levels and match_levels.entry:
        fig.add_hline(y=match_levels.entry, line_dash="dash", line_color="#2ecc71", annotation_text="Entry", row=1, col=1)
        fig.add_hline(y=match_levels.sl, line_dash="dash", line_color="#e74c3c", annotation_text="SL", row=1, col=1)
        fig.add_hline(y=match_levels.target, line_dash="dash", line_color="#3498db", annotation_text="T1", row=1, col=1)
        if match_levels.target_2:
            fig.add_hline(y=match_levels.target_2, line_dash="dot", line_color="#9b59b6", annotation_text="T2", row=1, col=1)
    fig.add_trace(go.Bar(x=bars.index, y=bars["Volume"], name="Volume", marker_color="#6b7c93"), row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, height=640, legend=dict(orientation="h"))
    if active.interval.endswith("m"):
        # Drop overnight / weekend / lunch-to-open empty time so candles sit side by side.
        session_breaks = [
            dict(bounds=["sat", "mon"]),
            dict(bounds=[15.5, 9.25], pattern="hour"),
        ]
        fig.update_xaxes(rangebreaks=session_breaks, row=1, col=1)
        fig.update_xaxes(rangebreaks=session_breaks, row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    match = next((r for r in st.session_state.last_results if r.symbol == picked), None)
    if match:
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Signal", match.signal or "—")
        s2.metric("Spot entry", f"₹{match.entry:.2f}" if match.entry else "—")
        s3.metric("Spot SL", f"₹{match.sl:.2f}" if match.sl else "—")
        s4.metric("Spot T1", f"₹{match.target:.2f}" if match.target else "—")
        s5.metric("Spot T2", f"₹{match.target_2:.2f}" if match.target_2 else "—")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Vehicle", match.instrument or "spot")
        v2.metric("Hold / valid until", match.hold_until or "—")
        v3.metric("Setup", match.setup or "—")
        rr = match.metrics.get("rr")
        v4.metric("R:R to T1", f"{rr:.1f}" if rr is not None else "—")
        if match.instrument == "option" and match.option_contract:
            st.info(
                f"**Options plan:** BUY `{match.option_contract}`"
                + (f" @ ~₹{match.option_premium:.2f}" if match.option_premium else "")
                + (f" | option SL ₹{match.option_sl:.2f}" if match.option_sl else "")
                + (f" | option T1 ₹{match.option_target:.2f}" if match.option_target else "")
                + (f" | lot {match.lot_size}" if match.lot_size else "")
                + ". Use the spot plan if the option is illiquid."
            )
        elif match.instrument == "spot":
            st.info("**Spot/cash trade** — this name is not suggested via F&O (unlisted or chain unavailable).")
        st.caption(match.validity_note)
        m1, m2 = st.columns(2)
        m1.metric("Breakout score", f"{match.breakout_score:.0f}")
        m2.metric("Buy zone score", f"{match.buy_zone_score:.0f}")
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
