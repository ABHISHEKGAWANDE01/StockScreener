from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analysis.indicators import ema, sma
from analysis.screener import run_screener
from config import DEFAULT_STYLE, DEFAULT_UNIVERSE, STYLES, get_style
from data.prices import download_for_style, download_one
from data.universe import load_universe

UNIVERSES = ["nifty50", "nifty200", "nifty500"]
FUNKY_CSS = Path(__file__).resolve().parent / "static" / "funky.css"
SIGNAL_BADGE = {
    "BUY": ("green", ":material/check_circle:"),
    "BUY STOP": ("blue", ":material/north:"),
    "WATCH": ("orange", ":material/visibility:"),
    "AVOID": ("red", ":material/block:"),
}

st.set_page_config(
    page_title="NSE intraday / swing screener",
    page_icon=":material/candlestick_chart:",
    layout="wide",
)


def _inject_funky_ui() -> None:
    st.html(FUNKY_CSS)


def _hero(*, live: bool, setups: int) -> None:
    status = "LIVE TAPE" if live else "STANDBY"
    chip = f"{setups} SETUPS LOCKED" if live else "WAITING FOR IGNITION"
    st.html(
        f"""
<div class="fx-hero">
  <div class="fx-orb a"></div>
  <div class="fx-orb b"></div>
  <div class="fx-orb c"></div>
  <p class="fx-kicker">{escape(status)} · NSE</p>
  <h1 class="fx-title">Pulse screener</h1>
  <p class="fx-sub">Intraday heat on the 15-minute tape, swing holds with a date, and an OpenAI second opinion. Loud visuals. Quiet risk.</p>
  <div class="fx-chips">
    <span class="fx-chip live">{escape(chip)}</span>
    <span class="fx-chip hot">Breakouts</span>
    <span class="fx-chip ice">Buy zone</span>
    <span class="fx-chip">F&amp;O + spot</span>
  </div>
</div>
"""
    )


def _ticker(df: pd.DataFrame) -> None:
    if df.empty or "symbol" not in df.columns:
        return
    names = [escape(str(s)) for s in df["symbol"].dropna().astype(str).head(36).tolist()]
    if not names:
        return
    chips = "".join(f"<span><b>▲</b> {s}</span>" for s in names)
    st.html(f'<div class="fx-ticker"><div class="fx-ticker-track">{chips}{chips}</div></div>')


def _init_state() -> None:
    st.session_state.setdefault("last_df", pd.DataFrame())
    st.session_state.setdefault("last_results", [])
    st.session_state.setdefault("last_style", DEFAULT_STYLE)
    st.session_state.setdefault("last_universe", DEFAULT_UNIVERSE)
    st.session_state.setdefault("last_force", False)
    st.session_state.setdefault("has_run", False)


def _universe_label(key: str) -> str:
    return {"nifty50": "Nifty 50", "nifty200": "Nifty 200", "nifty500": "Nifty 500"}.get(key, key)


@st.cache_data(ttl="24h", max_entries=8, show_spinner=False)
def _cached_universe(name: str) -> pd.DataFrame:
    return load_universe(name)


@st.cache_data(ttl="15m", max_entries=128, show_spinner=False)
def _cached_chart_bars(ticker: str, style_key: str) -> pd.DataFrame:
    spec = get_style(style_key)
    bars = download_for_style(ticker, spec)
    if bars.empty:
        bars = download_one(ticker)
    return bars.tail(spec.chart_bars)


def _nse_session_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Keep NSE cash-session stamps only (09:15–15:30 IST, Mon–Fri)."""
    if df.empty:
        return df
    idx = pd.DatetimeIndex(df.index)
    minutes = idx.hour * 60 + idx.minute
    mask = (idx.dayofweek < 5) & (minutes >= 9 * 60 + 15) & (minutes <= 15 * 60 + 30)
    return df.loc[mask]


def _filter_frame(df: pd.DataFrame, column: str, picked: str | None) -> pd.DataFrame:
    if not picked or picked == "all" or column not in df.columns:
        return df
    return df[df[column] == picked]


def _empty_guide(tf_info) -> None:
    with st.container(border=True, key="ready_card"):
        st.markdown("**The tape is quiet. Make it loud.**")
        st.caption("Pick a universe in the sidebar, then Fire scan. Nifty 50 is the fastest first hit.")
        st.badge(tf_info.label, icon=":material/bolt:", color="red")
        with st.expander("How setups work", icon=":material/auto_awesome:"):
            st.markdown(f":red-badge[breakout] Coil near {tf_info.high_label}, above {tf_info.fast_ma_label}")
            st.markdown(
                f":red-badge[buy_zone] Uptrend above {tf_info.slow_ma_label}, dip into the MAs, usable R:R"
            )
            st.markdown(":green-badge[BUY] Take now at **entry**")
            st.markdown(":blue-badge[BUY STOP] Buy only if price trades through **entry**")
            st.markdown(":orange-badge[WATCH] Levels shown — wait")
            st.markdown(":red-badge[AVOID] OpenAI rejected the plan")
            st.markdown(
                "**Intraday:** F&O names get the nearest listed NSE CE. Others use spot. Valid until **15:15 IST**."
            )
            st.markdown(
                "**Swing:** Spot plan plus **hold until** (about 2–4 weeks). Exit at T1/T2, SL, or that date."
            )
            st.caption("Liquidity floor is about ₹5 crore ADV. Combined score = 60% technical + 40% AI.")


def _render_kpis(view: pd.DataFrame, active) -> None:
    top_tech = f"{view['score'].max():.0f}" if len(view) and "score" in view.columns else "—"
    with st.container(horizontal=True, key="kpi_row"):
        st.metric("Style", active.label, border=True)
        st.metric("Setups found", len(view), border=True)
        st.metric("Top technical", top_tech, border=True)
        if "ai_score" in view.columns and view["ai_score"].notna().any():
            st.metric("Top AI score", f"{view['ai_score'].max():.0f}", border=True)
        elif "rr" in view.columns and view["rr"].notna().any():
            st.metric("Median R:R", f"{view['rr'].median():.1f}", border=True)
        else:
            st.metric("AI reviews", "off", border=True)


def _render_table(view: pd.DataFrame) -> None:
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
        hide_index=True,
        width="stretch",
        key="screener_table",
        column_config={
            "signal": st.column_config.TextColumn("Signal"),
            "instrument": st.column_config.TextColumn("Vehicle"),
            "option_contract": st.column_config.TextColumn("Option"),
            "hold_until": st.column_config.TextColumn("Valid / hold until"),
            "entry": st.column_config.NumberColumn("Spot entry", format="₹%.2f"),
            "sl": st.column_config.NumberColumn("Spot SL", format="₹%.2f"),
            "target": st.column_config.NumberColumn("Spot T1", format="₹%.2f"),
            "target_2": st.column_config.NumberColumn("Spot T2", format="₹%.2f"),
            "option_premium": st.column_config.NumberColumn("Opt ~premium", format="₹%.2f"),
            "option_sl": st.column_config.NumberColumn("Opt SL", format="₹%.2f"),
            "option_target": st.column_config.NumberColumn("Opt T1", format="₹%.2f"),
            "score": st.column_config.ProgressColumn("Technical", min_value=0, max_value=100, format="%.1f"),
            "ai_score": st.column_config.ProgressColumn("AI score", min_value=0, max_value=100, format="%.1f"),
            "combined_score": st.column_config.ProgressColumn("Combined", min_value=0, max_value=100, format="%.1f"),
            "validity_note": st.column_config.TextColumn("Validity", width="large"),
            "ai_rationale": st.column_config.MarkdownColumn("AI rationale", width="large"),
            "why": st.column_config.MarkdownColumn("Technical why", width="large"),
        },
    )


@st.fragment
def _chart_setup(view: pd.DataFrame, active, universe: str) -> None:
    symbols = view["symbol"].tolist()
    picked = st.selectbox("Chart a setup", symbols, key="chart_symbol")
    if not picked:
        return

    uni = _cached_universe(universe)
    row = uni[uni["symbol"] == picked].iloc[0]

    with st.skeleton(height=640):
        if st.session_state.last_force:
            spec = active
            bars = download_for_style(row["yf_ticker"], spec, force=True).tail(spec.chart_bars)
            if bars.empty:
                bars = download_one(row["yf_ticker"], force=True).tail(spec.chart_bars)
        else:
            bars = _cached_chart_bars(row["yf_ticker"], active.key)
        if bars.empty:
            st.warning("No price bars for this symbol.", icon=":material/show_chart:")
            return
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
        fig.add_trace(
            go.Scatter(x=bars.index, y=don, name=f"Prior {active.high_label}", line=dict(dash="dot", width=1)),
            row=1,
            col=1,
        )
        match_levels = next((r for r in st.session_state.last_results if r.symbol == picked), None)
        if match_levels and match_levels.entry:
            fig.add_hline(y=match_levels.entry, line_dash="dash", line_color="#bef264", annotation_text="Entry", row=1, col=1)
            fig.add_hline(y=match_levels.sl, line_dash="dash", line_color="#fb7185", annotation_text="SL", row=1, col=1)
            fig.add_hline(y=match_levels.target, line_dash="dash", line_color="#22d3ee", annotation_text="T1", row=1, col=1)
            if match_levels.target_2:
                fig.add_hline(
                    y=match_levels.target_2,
                    line_dash="dot",
                    line_color="#e879f9",
                    annotation_text="T2",
                    row=1,
                    col=1,
                )
        fig.add_trace(go.Bar(x=bars.index, y=bars["Volume"], name="Volume", marker_color="#c084fc"), row=2, col=1)
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=640,
            legend=dict(orientation="h"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(7,1,15,0.35)",
            font=dict(color="#f8fafc"),
        )
        if active.interval.endswith("m"):
            session_breaks = [
                dict(bounds=["sat", "mon"]),
                dict(bounds=[15.5, 9.25], pattern="hour"),
            ]
            fig.update_xaxes(rangebreaks=session_breaks, row=1, col=1)
            fig.update_xaxes(rangebreaks=session_breaks, row=2, col=1)
        st.plotly_chart(fig, width="stretch", theme="streamlit", key="setup_chart")

    match = next((r for r in st.session_state.last_results if r.symbol == picked), None)
    if not match:
        return

    color, icon = SIGNAL_BADGE.get(match.signal or "", ("gray", ":material/help:"))
    st.badge(match.signal or "—", icon=icon, color=color)

    with st.container(horizontal=True):
        st.metric("Spot entry", f"₹{match.entry:.2f}" if match.entry else "—", border=True)
        st.metric("Spot SL", f"₹{match.sl:.2f}" if match.sl else "—", border=True)
        st.metric("Spot T1", f"₹{match.target:.2f}" if match.target else "—", border=True)
        st.metric("Spot T2", f"₹{match.target_2:.2f}" if match.target_2 else "—", border=True)

    with st.container(horizontal=True):
        st.metric("Vehicle", match.instrument or "spot", border=True)
        st.metric("Hold / valid until", match.hold_until or "—", border=True)
        st.metric("Setup", match.setup or "—", border=True)
        rr = match.metrics.get("rr")
        st.metric("R:R to T1", f"{rr:.1f}" if rr is not None else "—", border=True)

    if match.instrument == "option" and match.option_contract:
        st.info(
            f"**Options plan:** BUY `{match.option_contract}`"
            + (f" @ ~₹{match.option_premium:.2f}" if match.option_premium else "")
            + (f" | option SL ₹{match.option_sl:.2f}" if match.option_sl else "")
            + (f" | option T1 ₹{match.option_target:.2f}" if match.option_target else "")
            + (f" | lot {match.lot_size}" if match.lot_size else "")
            + ". Use the spot plan if the option is illiquid.",
            icon=":material/receipt_long:",
        )
    elif match.instrument == "spot":
        st.caption("Spot/cash trade — this name is not suggested via F&O (unlisted or chain unavailable).")
    if match.validity_note:
        st.caption(match.validity_note)

    with st.container(horizontal=True):
        st.metric("Breakout score", f"{match.breakout_score:.0f}", border=True)
        st.metric("Buy zone score", f"{match.buy_zone_score:.0f}", border=True)

    if match.ai_score is not None:
        with st.container(border=True):
            st.markdown("**OpenAI confirmation**")
            with st.container(horizontal=True):
                st.metric("AI score", f"{match.ai_score:.0f}", border=True)
                st.metric("Verdict", match.ai_verdict or "—", border=True)
                st.metric("Combined", f"{match.combined_score:.0f}" if match.combined_score is not None else "—", border=True)
            st.write(match.ai_rationale)
            if match.ai_risks:
                st.caption("Risks: " + " · ".join(match.ai_risks))

    with st.expander("Why this name scored", icon=":material/analytics:"):
        for reason in match.reasons:
            st.write(f"- {reason}")
    with st.expander("Raw metrics", icon=":material/data_object:"):
        st.json(match.metrics)


@st.fragment
def _results_view() -> None:
    df = st.session_state.last_df
    active = get_style(st.session_state.last_style)
    view = df

    with st.container(horizontal=True):
        if "setup" in df.columns:
            kinds = ["all"] + sorted({s for s in df["setup"].dropna().unique()})
            picked_setup = st.pills("Setup", kinds, default="all", key="filter_setup")
            view = _filter_frame(view, "setup", picked_setup)
        if "signal" in df.columns:
            sigs = ["all"] + sorted({s for s in df["signal"].dropna().unique()})
            picked_sig = st.pills("Signal", sigs, default="all", key="filter_signal")
            view = _filter_frame(view, "signal", picked_sig)
        if "instrument" in df.columns:
            insts = ["all"] + sorted({s for s in df["instrument"].dropna().unique()})
            picked_inst = st.pills("Vehicle", insts, default="all", key="filter_instrument")
            view = _filter_frame(view, "instrument", picked_inst)

    if view.empty:
        st.warning("No rows match those filters.", icon=":material/filter_alt_off:")
        return

    _render_kpis(view, active)

    csv = view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download table",
        data=csv,
        file_name="screener.csv",
        mime="text/csv",
        icon=":material/download:",
        type="tertiary",
    )

    with st.container(border=True, key="shortlist_card"):
        st.markdown("**Shortlist** · sort, filter, then chart a name")
        _render_table(view)

    with st.container(border=True, key="chart_card"):
        st.markdown("**Chart** · levels on the tape")
        _chart_setup(view, active, st.session_state.last_universe)


_init_state()
_inject_funky_ui()

with st.sidebar:
    st.markdown("### Scan booth")
    st.caption("Dial the universe. Fire when ready.")
    with st.form("scan", border=True):
        universe = st.radio(
            "Universe",
            UNIVERSES,
            index=UNIVERSES.index(DEFAULT_UNIVERSE),
            format_func=_universe_label,
            key="scan_universe",
        )
        style = st.radio(
            "Trade style",
            list(STYLES.keys()),
            index=list(STYLES.keys()).index(DEFAULT_STYLE),
            format_func=lambda k: STYLES[k].label,
            key="scan_style",
        )
        tf_info = get_style(style or DEFAULT_STYLE)
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
        force = st.toggle("Refresh prices", value=False, help="Ignore the local price cache and re-download.")
        ai_verify = st.toggle("OpenAI confirmation", value=True)
        ai_limit = st.slider("AI review top N", 5, 50, 25)
        run = st.form_submit_button("Fire scan", type="primary", icon=":material/bolt:", width="stretch")

if universe not in UNIVERSES:
    universe = DEFAULT_UNIVERSE
if style not in STYLES:
    style = DEFAULT_STYLE
tf_info = get_style(style)

if run:
    with st.status(f":shimmer[Scoring {tf_info.label.lower()} setups…]", expanded=True) as status:
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
        st.session_state.last_universe = universe
        st.session_state.last_force = bool(force)
        st.session_state.has_run = True
        status.update(label=f"Drop complete · {len(df)} setups on tape", state="complete")
    if df.empty:
        st.toast("Tape is empty. Loosen the score.", icon=":material/info:")
    else:
        st.toast(f"{len(df)} names just hit the board.", icon=":material/celebration:")

df = st.session_state.last_df
_hero(live=st.session_state.has_run and not df.empty, setups=len(df))
if not df.empty:
    _ticker(df)

if df.empty:
    if st.session_state.has_run:
        st.warning(
            "No setups met the score cutoff. Lower the minimum score or try another universe.",
            icon=":material/search_off:",
        )
    else:
        _empty_guide(tf_info)
    st.caption("Educational screener, not investment advice. Size positions off the suggested stop, not the score.")
    st.stop()

_results_view()
st.caption("Educational screener, not investment advice. Size positions off the suggested stop, not the score.")
