"""Hong Kong private housing dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_data import (
    CLASS_LABELS,
    PRICE_INDEX_URL,
    RENTAL_INDEX_URL,
    class_change_table,
    load_market_data,
    series_summary,
)
from policy_events import POLICY_COLORS, POLICY_EVENTS, get_policy_events_df


st.set_page_config(
    page_title="Hong Kong private housing",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap');

    :root {
        --paper: #f3f0e9;
        --surface: #faf8f3;
        --ink: #17242f;
        --muted: #657078;
        --line: #d8d1c4;
        --blue: #244b68;
        --rust: #b35e3d;
        --green: #496c5a;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }

    .stApp { background: var(--paper); }

    .main .block-container {
        max-width: 1180px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    header[data-testid="stHeader"] { background: rgba(243, 240, 233, 0.94); }
    #MainMenu, footer { visibility: hidden; }

    h1, h2, h3 {
        font-family: 'Source Serif 4', serif !important;
        color: var(--ink) !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }

    h1 { font-size: clamp(2.5rem, 6vw, 4.8rem) !important; line-height: 0.98 !important; }
    h2 { font-size: 1.8rem !important; margin-top: 0.5rem !important; }
    h3 { font-size: 1.2rem !important; }

    .eyebrow {
        color: var(--rust);
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.16em;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
    }

    .intro {
        color: #4d5961;
        font-family: 'Source Serif 4', serif;
        font-size: 1.08rem;
        line-height: 1.55;
        max-width: 760px;
        margin: 0.7rem 0 1.1rem;
    }

    .freshness {
        display: inline-block;
        border: 1px solid var(--line);
        background: rgba(250, 248, 243, 0.7);
        color: var(--muted);
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        padding: 0.45rem 0.65rem;
        margin-bottom: 1.8rem;
    }

    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-top: 3px solid var(--ink);
        min-height: 132px;
        padding: 1rem 1.05rem;
    }

    [data-testid="stMetricLabel"] p {
        color: var(--muted) !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.04em;
    }

    [data-testid="stMetricValue"] {
        color: var(--ink) !important;
        font-family: 'Source Serif 4', serif !important;
        font-size: 2rem !important;
    }

    [data-testid="stMetricDelta"] {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem !important;
    }

    [data-baseweb="tab-list"] {
        border-bottom: 1px solid var(--line);
        gap: 1.5rem;
        margin-top: 1.2rem;
    }

    [data-baseweb="tab"] {
        background: transparent;
        color: var(--muted);
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        padding-left: 0;
        padding-right: 0;
    }

    [aria-selected="true"][data-baseweb="tab"] {
        color: var(--ink) !important;
        border-bottom-color: var(--rust) !important;
    }

    [data-testid="stSelectbox"] label p {
        color: var(--muted);
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
    }

    [data-baseweb="select"] > div {
        background: var(--surface);
        border-color: var(--line);
    }

    .section-note {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.55;
        max-width: 760px;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }

    .finding {
        background: var(--surface);
        border-left: 3px solid var(--rust);
        color: #3f4a51;
        line-height: 1.6;
        padding: 0.9rem 1rem;
        margin: 0.5rem 0 1.5rem;
    }

    .policy-card {
        border-bottom: 1px solid var(--line);
        padding: 0.9rem 0 1rem;
    }

    .policy-date, .policy-type {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .policy-date { color: var(--muted); margin-right: 0.7rem; }
    .policy-type { color: var(--rust); }
    .policy-title { color: var(--ink); font-family: 'Source Serif 4', serif; font-size: 1.12rem; margin: 0.35rem 0 0.2rem; }
    .policy-copy { color: var(--muted); font-size: 0.85rem; }

    .source-box {
        background: var(--surface);
        border: 1px solid var(--line);
        min-height: 170px;
        padding: 1rem;
    }

    .source-box p { color: var(--muted); font-size: 0.86rem; line-height: 1.55; }
    .source-box a { color: var(--blue) !important; }

    hr { border-color: var(--line); margin: 2rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


COLORS = {
    "ink": "#17242f",
    "muted": "#657078",
    "line": "#d8d1c4",
    "paper": "#f3f0e9",
    "blue": "#244b68",
    "rust": "#b35e3d",
    "green": "#496c5a",
}


def chart_layout(height: int = 420) -> dict:
    return {
        "height": height,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "DM Sans, sans-serif", "color": COLORS["muted"], "size": 12},
        "hoverlabel": {"bgcolor": "#faf8f3", "bordercolor": COLORS["line"], "font": {"color": COLORS["ink"]}},
        "margin": {"l": 18, "r": 18, "t": 35, "b": 25},
        "legend": {"orientation": "h", "y": 1.08, "x": 0, "title": None},
        "xaxis": {"showgrid": False, "linecolor": COLORS["line"], "tickformat": "%b\n%Y"},
        "yaxis": {"gridcolor": COLORS["line"], "zeroline": False, "title": "Index, 1999 = 100"},
    }


@st.cache_data(ttl=3600, show_spinner=False)
def get_market_data() -> pd.DataFrame:
    return load_market_data()


def filter_period(data: pd.DataFrame, period: str) -> pd.DataFrame:
    years = {"5 years": 5, "10 years": 10}.get(period)
    if years is None:
        return data.copy()
    cutoff = data["date"].max() - pd.DateOffset(years=years)
    return data.loc[data["date"] >= cutoff].copy()


def market_chart(data: pd.DataFrame, class_code: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data[f"price_{class_code}"],
            name="Sale price index",
            mode="lines",
            line={"color": COLORS["blue"], "width": 2.6},
            hovertemplate="%{x|%b %Y}<br>Price %{y:.1f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data[f"rental_{class_code}"],
            name="Rental index",
            mode="lines",
            line={"color": COLORS["rust"], "width": 2.3},
            hovertemplate="%{x|%b %Y}<br>Rent %{y:.1f}<extra></extra>",
        )
    )
    fig.update_layout(**chart_layout())
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def class_change_chart(changes: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=changes["label"],
            y=changes["price_change"],
            name="Sale prices",
            marker_color=COLORS["blue"],
            hovertemplate="%{x}<br>%{y:+.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=changes["label"],
            y=changes["rental_change"],
            name="Rents",
            marker_color=COLORS["rust"],
            hovertemplate="%{x}<br>%{y:+.1f}%<extra></extra>",
        )
    )
    layout = chart_layout(390)
    layout["barmode"] = "group"
    layout["yaxis"] = {"gridcolor": COLORS["line"], "zerolinecolor": COLORS["ink"], "title": "Change over 12 months"}
    fig.update_layout(**layout)
    fig.update_yaxes(ticksuffix="%")
    return fig


def policy_chart() -> go.Figure:
    events = get_policy_events_df()
    labels = {"cooling": "Cooling", "easing": "Easing", "mortgage": "Mortgage rules", "external": "External"}
    fig = go.Figure()
    for event_type, group in events.groupby("type", sort=False):
        fig.add_trace(
            go.Scatter(
                x=group["date"],
                y=[labels[event_type]] * len(group),
                mode="markers",
                name=labels[event_type],
                marker={"size": 13, "color": POLICY_COLORS[event_type], "line": {"color": COLORS["paper"], "width": 2}},
                text=group["name"],
                customdata=group["description"],
                hovertemplate="<b>%{text}</b><br>%{x|%d %b %Y}<br>%{customdata}<extra></extra>",
            )
        )
    layout = chart_layout(310)
    layout["yaxis"] = {"showgrid": False, "title": None}
    layout["xaxis"] = {"showgrid": False, "linecolor": COLORS["line"], "tickformat": "%Y"}
    fig.update_layout(**layout)
    return fig


def render_policy_cards() -> None:
    for event in reversed(POLICY_EVENTS):
        st.markdown(
            f"""
            <div class="policy-card">
                <span class="policy-date">{pd.Timestamp(event['date']).strftime('%d %b %Y')}</span>
                <span class="policy-type">{event['type']}</span>
                <div class="policy-title">{event['name']}</div>
                <div class="policy-copy">{event['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown('<div class="eyebrow">Rating and Valuation Department</div>', unsafe_allow_html=True)
st.title("Hong Kong private housing")
st.markdown(
    '<div class="intro">Monthly sale price and rental indices for private domestic property. '
    'Both series use 1999 = 100, which makes their movement comparable without implying actual prices or yields.</div>',
    unsafe_allow_html=True,
)

try:
    with st.spinner("Reading the latest RVD workbooks..."):
        market = get_market_data()
except Exception as exc:
    st.error("The official RVD workbooks could not be loaded. Please try again shortly.")
    st.caption(str(exc))
    st.stop()

latest_date = market["date"].max()
st.markdown(
    f'<div class="freshness">Official monthly series · through {latest_date.strftime("%b %Y")} · latest figures provisional</div>',
    unsafe_allow_html=True,
)

control_a, control_b, spacer = st.columns([1.25, 1, 2.5])
with control_a:
    selected_label = st.selectbox("Flat size", list(CLASS_LABELS.values()), index=0)
with control_b:
    period = st.selectbox("Chart period", ["5 years", "10 years", "Since 1993"], index=0)

class_code = next(code for code, label in CLASS_LABELS.items() if label == selected_label)
view = filter_period(market, period)
price_stats = series_summary(market, f"price_{class_code}")
rental_stats = series_summary(market, f"rental_{class_code}")

metric_a, metric_b, metric_c, metric_d = st.columns(4)
metric_a.metric("Sale price index", f"{price_stats['latest']:.1f}", f"{price_stats['yoy']:+.1f}% over 12 months")
metric_b.metric("From the price peak", f"{price_stats['drawdown']:.1f}%", f"Peak: {price_stats['peak_date'].strftime('%b %Y')}", delta_color="off")
metric_c.metric("Rental index", f"{rental_stats['latest']:.1f}", f"{rental_stats['yoy']:+.1f}% over 12 months")
metric_d.metric("Latest month", latest_date.strftime("%b %Y"), "Provisional figures", delta_color="off")

overview_tab, classes_tab, policy_tab, sources_tab = st.tabs(["Market view", "By flat size", "Policy record", "Sources"])

with overview_tab:
    st.header("Sale prices and rents")
    st.markdown(
        '<p class="section-note">The two indices share the same 1999 base. The chart compares direction and scale, not cash prices or rental yields.</p>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(market_chart(view, class_code), use_container_width=True, config={"displayModeBar": False})

    price_gap = abs(price_stats["drawdown"])
    rental_peak_text = "at its series high" if abs(rental_stats["drawdown"]) < 0.05 else f"{abs(rental_stats['drawdown']):.1f}% below its peak"
    st.markdown(
        f'<div class="finding">For {selected_label.lower()}, sale prices are {price_gap:.1f}% below their '
        f'{price_stats["peak_date"].strftime("%b %Y")} peak. The rental index is {rental_peak_text}.</div>',
        unsafe_allow_html=True,
    )

with classes_tab:
    st.header("Change by flat size")
    st.markdown(
        '<p class="section-note">RVD classes are based on saleable floor area. The bars show the latest 12-month change for each class.</p>',
        unsafe_allow_html=True,
    )
    changes = class_change_table(market)
    st.plotly_chart(class_change_chart(changes), use_container_width=True, config={"displayModeBar": False})
    display = changes[["label", "price_latest", "price_change", "rental_latest", "rental_change"]].copy()
    display.columns = ["Flat size", "Price index", "Price change", "Rental index", "Rental change"]
    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Price index": st.column_config.NumberColumn(format="%.1f"),
            "Price change": st.column_config.NumberColumn(format="%+.1f%%"),
            "Rental index": st.column_config.NumberColumn(format="%.1f"),
            "Rental change": st.column_config.NumberColumn(format="%+.1f%%"),
        },
    )

with policy_tab:
    st.header("Policy record")
    st.markdown(
        '<p class="section-note">Selected tax and mortgage measures that changed how residential property purchases were financed or taxed.</p>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(policy_chart(), use_container_width=True, config={"displayModeBar": False})
    render_policy_cards()

with sources_tab:
    st.header("Sources and definitions")
    source_a, source_b = st.columns(2)
    with source_a:
        st.markdown(
            f'''<div class="source-box"><h3>Sale price index</h3><p>RVD table 1.4, private domestic price indices by class, territory-wide. Primary sales are excluded from this series.</p><p><a href="{PRICE_INDEX_URL}">Open the official workbook</a></p></div>''',
            unsafe_allow_html=True,
        )
    with source_b:
        st.markdown(
            f'''<div class="source-box"><h3>Rental index</h3><p>RVD table 1.3, private domestic rental indices by class, territory-wide. Both index series use 1999 = 100.</p><p><a href="{RENTAL_INDEX_URL}">Open the official workbook</a></p></div>''',
            unsafe_allow_html=True,
        )

    st.subheader("Flat size classes")
    st.markdown("A: under 40 m² · B: 40 to 69.9 m² · C: 70 to 99.9 m² · D: 100 to 159.9 m² · E: 160 m² or more")
    st.caption("The newest RVD observations are provisional and may be revised. Data is requested when the app starts and cached for one hour.")
    st.download_button(
        "Download the cleaned monthly series",
        data=market.to_csv(index=False).encode("utf-8"),
        file_name="hong_kong_private_housing_indices.csv",
        mime="text/csv",
    )

st.divider()
st.caption("Source: Hong Kong Rating and Valuation Department. Built by Alek Swiderski.")
