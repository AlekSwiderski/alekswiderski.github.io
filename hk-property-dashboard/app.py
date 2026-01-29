"""
Hong Kong Property Market Dashboard
Interactive analysis of HK property prices, affordability, and policy impact
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import io
from pathlib import Path

from policy_events import POLICY_EVENTS, POLICY_COLORS, get_policy_events_df

# Page config
st.set_page_config(
    page_title="HK Property Market",
    page_icon="\u25C8",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — Dark finance editorial theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ===== GLOBAL ===== */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #E6EDF3;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background-color: #0E1117;
        border-bottom: 1px solid #2A2F3A;
    }

    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #2A2F3A;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #8B949E;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
    }

    [data-testid="stSidebar"] .stRadio > label {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.85rem;
        color: #8B949E;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label[data-checked="true"],
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked) {
        color: #C9A84C !important;
        font-weight: 700;
    }

    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {
        color: #E6EDF3;
        background-color: #1C2333;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-family: 'DM Serif Display', serif;
        color: #C9A84C;
        letter-spacing: 0.02em;
    }

    [data-testid="stSidebar"] [data-testid="stAlert"] {
        background-color: #1C2333;
        border: 1px solid #2A2F3A;
        border-left: 3px solid #C9A84C;
        color: #8B949E;
        border-radius: 2px;
    }

    [data-testid="stSidebar"] hr {
        border-color: #2A2F3A;
    }

    /* ===== MAIN HEADER ===== */
    .main-header {
        font-family: 'DM Serif Display', serif !important;
        font-size: 2.8rem !important;
        font-weight: 400 !important;
        color: #E6EDF3 !important;
        margin-bottom: 0.25rem !important;
        letter-spacing: -0.01em;
        line-height: 1.1;
    }

    .main-header::after {
        content: '';
        display: block;
        width: 60px;
        height: 3px;
        background-color: #C9A84C;
        margin-top: 0.75rem;
    }

    .sub-header {
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        color: #6B7280 !important;
        margin-bottom: 2.5rem !important;
        font-weight: 400;
        letter-spacing: 0.04em;
    }

    /* ===== ALL HEADERS ===== */
    h1, h2, h3 {
        font-family: 'DM Serif Display', serif !important;
        color: #E6EDF3 !important;
    }

    h1 {
        font-size: 2rem !important;
        border-bottom: 1px solid #2A2F3A;
        padding-bottom: 0.5rem;
    }

    h2 {
        font-size: 1.5rem !important;
        color: #C9A84C !important;
    }

    h3 {
        font-size: 1.2rem !important;
    }

    /* ===== METRIC CARDS ===== */
    [data-testid="metric-container"] {
        background-color: #161B22;
        border: 1px solid #2A2F3A;
        border-top: 3px solid #C9A84C;
        padding: 1.25rem 1rem;
        border-radius: 0px;
        transition: border-color 0.2s ease, background-color 0.2s ease, transform 0.15s ease;
    }

    [data-testid="metric-container"]:hover {
        border-top-color: #E8C547;
        background-color: #1C2333;
        transform: translateY(-2px);
    }

    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #8B949E !important;
    }

    [data-testid="stMetricLabel"] p {
        font-size: 0.75rem !important;
        color: #8B949E !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.8rem !important;
        font-weight: 500;
        color: #E6EDF3 !important;
    }

    [data-testid="stMetricValue"] div {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.8rem !important;
    }

    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem !important;
    }

    /* ===== HORIZONTAL RULES ===== */
    hr {
        border: none;
        border-top: 1px solid #2A2F3A;
        margin: 2rem 0;
    }

    /* ===== MARKDOWN / BODY TEXT ===== */
    [data-testid="stMarkdownContainer"] {
        font-family: 'Inter', sans-serif;
        line-height: 1.65;
    }

    [data-testid="stMarkdownContainer"] p {
        color: #E6EDF3;
    }

    [data-testid="stMarkdownContainer"] strong {
        color: #C9A84C;
        font-weight: 600;
    }

    [data-testid="stMarkdownContainer"] li {
        color: #8B949E;
        margin-bottom: 0.25rem;
    }

    /* ===== TABLES ===== */
    [data-testid="stMarkdownContainer"] table {
        border-collapse: collapse;
        width: 100%;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
    }

    [data-testid="stMarkdownContainer"] table th {
        background-color: #161B22;
        color: #C9A84C;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        padding: 0.75rem 1rem;
        border-bottom: 2px solid #C9A84C;
        text-align: left;
    }

    [data-testid="stMarkdownContainer"] table td {
        padding: 0.6rem 1rem;
        border-bottom: 1px solid #2A2F3A;
        color: #E6EDF3;
    }

    [data-testid="stMarkdownContainer"] table tr:hover td {
        background-color: #1C2333;
    }

    /* ===== DATAFRAMES ===== */
    [data-testid="stDataFrame"] {
        border: 1px solid #2A2F3A;
        border-radius: 0px;
    }

    /* ===== LINKS ===== */
    a {
        color: #C9A84C !important;
        text-decoration: none;
        border-bottom: 1px solid transparent;
        transition: border-color 0.2s ease;
    }

    a:hover {
        color: #E8C547 !important;
        border-bottom-color: #E8C547;
    }

    /* ===== ALERT BOXES ===== */
    [data-testid="stAlert"] {
        background-color: #161B22;
        border: 1px solid #2A2F3A;
        border-radius: 0px;
        color: #8B949E;
    }

    /* ===== ANIMATIONS ===== */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(12px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes goldPulse {
        0%, 100% { border-top-color: #C9A84C; }
        50% { border-top-color: #E8C547; }
    }

    [data-testid="stVerticalBlock"] > [data-testid="element-container"] {
        animation: fadeInUp 0.4s ease-out both;
    }

    [data-testid="stVerticalBlock"] > [data-testid="element-container"]:nth-child(1) { animation-delay: 0.05s; }
    [data-testid="stVerticalBlock"] > [data-testid="element-container"]:nth-child(2) { animation-delay: 0.10s; }
    [data-testid="stVerticalBlock"] > [data-testid="element-container"]:nth-child(3) { animation-delay: 0.15s; }
    [data-testid="stVerticalBlock"] > [data-testid="element-container"]:nth-child(4) { animation-delay: 0.20s; }
    [data-testid="stVerticalBlock"] > [data-testid="element-container"]:nth-child(5) { animation-delay: 0.25s; }
    [data-testid="stVerticalBlock"] > [data-testid="element-container"]:nth-child(6) { animation-delay: 0.30s; }

    [data-testid="column"]:first-child [data-testid="metric-container"] {
        animation: goldPulse 2s ease-in-out 0.5s 1;
    }

    .main .block-container {
        animation: fadeInUp 0.3s ease-out;
    }

    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0E1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #2A2F3A;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #484F58;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# PLOTLY DARK THEME
# =============================================================================

def get_plotly_layout_defaults():
    """Return dark finance theme layout kwargs for all Plotly charts."""
    return dict(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(
            family="Inter, sans-serif",
            color="#8B949E",
            size=12,
        ),
        title_font=dict(
            family="DM Serif Display, serif",
            color="#E6EDF3",
            size=18,
        ),
        xaxis=dict(
            gridcolor="#2A2F3A",
            gridwidth=1,
            zerolinecolor="#2A2F3A",
            linecolor="#2A2F3A",
            tickfont=dict(family="JetBrains Mono, monospace", size=11, color="#6B7280"),
            title_font=dict(family="Inter, sans-serif", color="#8B949E", size=12),
        ),
        yaxis=dict(
            gridcolor="#2A2F3A",
            gridwidth=1,
            zerolinecolor="#2A2F3A",
            linecolor="#2A2F3A",
            tickfont=dict(family="JetBrains Mono, monospace", size=11, color="#6B7280"),
            title_font=dict(family="Inter, sans-serif", color="#8B949E", size=12),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8B949E", size=11),
            bordercolor="#2A2F3A",
            borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor="#1C2333",
            bordercolor="#C9A84C",
            font=dict(family="JetBrains Mono, monospace", color="#E6EDF3", size=12),
        ),
        margin=dict(l=60, r=30, t=60, b=50),
    )


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(ttl=3600)
def load_rvd_price_indices():
    """Load RVD private domestic price indices."""
    url = "https://www.rvd.gov.hk/doc/en/statistics/his_data_2.xls"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_excel(io.BytesIO(response.content), sheet_name=0, header=None)
        return df
    except Exception as e:
        st.error(f"Error loading price indices: {e}")
        return None


@st.cache_data(ttl=3600)
def load_rvd_rental_indices():
    """Load RVD private domestic rental indices."""
    url = "https://www.rvd.gov.hk/doc/en/statistics/his_data_4.xls"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_excel(io.BytesIO(response.content), sheet_name=0, header=None)
        return df
    except Exception as e:
        st.error(f"Error loading rental indices: {e}")
        return None


@st.cache_data(ttl=3600)
def load_rvd_transactions():
    """Load RVD transaction data."""
    urls = [
        "https://www.rvd.gov.hk/doc/en/statistics/his_data_8.xls",
        "https://www.rvd.gov.hk/doc/en/statistics/hs_data_8.xls",
    ]
    for url in urls:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            df = pd.read_excel(io.BytesIO(response.content), sheet_name=0, header=None)
            return df
        except Exception:
            continue
    return None


def parse_rvd_price_data(raw_df):
    """Parse RVD price indices Excel into clean DataFrame."""
    if raw_df is None:
        return None

    df = raw_df.copy()

    header_row = None
    for i, row in df.iterrows():
        row_str = " ".join([str(x).lower() for x in row.values if pd.notna(x)])
        if "year" in row_str or "class" in row_str:
            header_row = i
            break

    if header_row is None:
        header_row = 3

    df.columns = df.iloc[header_row]
    df = df.iloc[header_row + 1 :].reset_index(drop=True)
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() if pd.notna(c) else f"col_{i}" for i, c in enumerate(df.columns)]

    return df


# =============================================================================
# CHART FUNCTIONS
# =============================================================================

def create_price_trend_chart(df, policy_df):
    """Create price trend chart with policy overlays."""
    fig = go.Figure()

    price_col = None
    for col in df.columns:
        if "all" in str(col).lower() or "overall" in str(col).lower():
            price_col = col
            break

    if price_col is None and len(df.columns) > 1:
        price_col = df.columns[1]

    if price_col:
        year_col = df.columns[0]
        fig.add_trace(
            go.Scatter(
                x=df[year_col],
                y=pd.to_numeric(df[price_col], errors="coerce"),
                mode="lines",
                name="Price Index",
                line=dict(color="#C9A84C", width=2.5),
            )
        )

    for event in POLICY_EVENTS:
        color = POLICY_COLORS.get(event["type"], "#6B7280")
        fig.add_vline(
            x=event["date"],
            line_dash="dash",
            line_color=color,
            opacity=0.7,
            annotation_text=event["name"],
            annotation_position="top",
            annotation_font_size=10,
            annotation_font_color="#8B949E",
        )

    fig.update_layout(
        **get_plotly_layout_defaults(),
        title="Private Domestic Property Price Index",
        xaxis_title="Year",
        yaxis_title="Price Index (1999 = 100)",
        hovermode="x unified",
        height=500,
    )

    return fig


def create_affordability_chart():
    """Create affordability ratio visualization."""
    affordability_data = {
        "Year": [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        "Ratio": [11.4, 12.6, 13.5, 14.9, 17.0, 19.0, 18.1, 19.4, 20.9, 20.8, 20.7, 23.2, 18.8, 18.8, 16.7],
    }
    df = pd.DataFrame(affordability_data)

    fig = go.Figure()

    colors = [
        "#E05252" if r > 15
        else "#F59E0B" if r > 10
        else "#3FB68B"
        for r in df["Ratio"]
    ]

    fig.add_trace(
        go.Bar(
            x=df["Year"],
            y=df["Ratio"],
            marker_color=colors,
            marker_line_color="#0E1117",
            marker_line_width=1,
            text=df["Ratio"],
            textposition="outside",
            textfont=dict(family="JetBrains Mono, monospace", size=11, color="#E6EDF3"),
        )
    )

    fig.add_hline(
        y=5, line_dash="dash", line_color="#3FB68B", line_width=1,
        annotation_text="Affordable (<5)",
        annotation_font=dict(color="#3FB68B", size=10),
    )
    fig.add_hline(
        y=10, line_dash="dash", line_color="#F59E0B", line_width=1,
        annotation_text="Severely Unaffordable (>10)",
        annotation_font=dict(color="#F59E0B", size=10),
    )

    fig.update_layout(
        **get_plotly_layout_defaults(),
        title="Housing Affordability Ratio",
        xaxis_title="Year",
        yaxis_title="Price-to-Income Ratio",
        height=400,
        showlegend=False,
    )

    return fig


def create_policy_timeline():
    """Create visual policy timeline."""
    policy_df = get_policy_events_df()

    fig = go.Figure()

    for i, row in policy_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["date"]],
                y=[row["type"]],
                mode="markers+text",
                marker=dict(
                    size=16,
                    color=row["color"],
                    symbol="diamond",
                    line=dict(color="#0E1117", width=2),
                ),
                text=row["name"],
                textposition="top center",
                textfont=dict(
                    family="Inter, sans-serif",
                    size=10,
                    color="#E6EDF3",
                ),
                name=row["name"],
                hovertemplate=(
                    f"<b>{row['name']}</b><br>"
                    f"{row['date'].strftime('%Y-%m-%d')}<br>"
                    f"{row['description']}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        **get_plotly_layout_defaults(),
        title="Policy Timeline",
        xaxis_title="Date",
        yaxis_title="Policy Type",
        height=300,
        showlegend=False,
    )

    return fig


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown(
        '<p class="main-header">Hong Kong Property Market</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Price Trends / Affordability / Policy Impact / 1979 \u2014 Present</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.markdown(
        '<p style="font-family: DM Serif Display, serif; font-size: 1.3rem; '
        'color: #C9A84C; margin-bottom: 0.25rem; letter-spacing: 0.02em;">HK Property</p>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<p style="font-family: JetBrains Mono, monospace; font-size: 0.7rem; '
        'color: #6B7280; letter-spacing: 0.15em; text-transform: uppercase; '
        'margin-bottom: 1.5rem;">Market Intelligence</p>',
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Price Trends", "Affordability", "Policy Impact", "Data Sources"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<p style="font-family: Inter, sans-serif; font-size: 0.7rem; '
        'color: #6B7280; text-transform: uppercase; letter-spacing: 0.1em; '
        'margin-bottom: 0.5rem;">Data Sources</p>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "Rating & Valuation Dept.  \n"
        "HK Monetary Authority  \n"
        "Census & Statistics Dept.  \n"
        "Demographia International"
    )

    # Load data
    raw_price_df = load_rvd_price_indices()
    raw_rental_df = load_rvd_rental_indices()
    raw_transactions_df = load_rvd_transactions()

    policy_df = get_policy_events_df()

    # Page routing
    if page == "Overview":
        show_overview(raw_price_df, policy_df)
    elif page == "Price Trends":
        show_price_trends(raw_price_df, raw_rental_df, policy_df)
    elif page == "Affordability":
        show_affordability()
    elif page == "Policy Impact":
        show_policy_impact(policy_df)
    elif page == "Data Sources":
        show_data_sources()


def show_overview(raw_price_df, policy_df):
    """Overview page with key metrics and summary chart."""
    st.header("Market Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Latest Affordability Ratio",
            value="16.7x",
            delta="-2.1 from 2023",
            delta_color="normal",
        )

    with col2:
        st.metric(
            label="Price Change (2024)",
            value="-7.5%",
            delta="Continuing decline",
            delta_color="inverse",
        )

    with col3:
        st.metric(
            label="Policy Status",
            value="All Duties Removed",
            delta="Feb 2024",
        )

    with col4:
        st.metric(
            label="Data Coverage",
            value="1979-2024",
            delta="45+ years",
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Affordability Trend")
        fig = create_affordability_chart()
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Recent Policy Changes")
        fig = create_policy_timeline()
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Key Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **Price Trends:**
            - Hong Kong property prices peaked in 2021
            - Prices have declined ~20% from peak
            - 2024 stamp duty removal aimed to stimulate market
            """
        )

    with col2:
        st.markdown(
            """
            **Affordability:**
            - HK remains one of world's least affordable markets
            - Peak ratio of 23.2x in 2021
            - Recent decline to 16.7x still "severely unaffordable"
            """
        )


def show_price_trends(raw_price_df, raw_rental_df, policy_df):
    """Price trends analysis page."""
    st.header("Price & Rental Trends")

    if raw_price_df is not None:
        st.subheader("Raw Price Index Data Preview")
        st.dataframe(raw_price_df.head(20), use_container_width=True)

        st.info(
            "The RVD data requires parsing. The table above shows the raw format. "
            "Price indices are typically in columns by property class (A-E) with years in rows."
        )

        price_df = parse_rvd_price_data(raw_price_df)
        if price_df is not None and len(price_df) > 0:
            st.subheader("Parsed Data")
            st.dataframe(price_df.head(20), use_container_width=True)
    else:
        st.warning("Could not load price index data. Please check your connection.")

    if raw_rental_df is not None:
        st.subheader("Raw Rental Index Data Preview")
        st.dataframe(raw_rental_df.head(20), use_container_width=True)


def show_affordability():
    """Affordability analysis page."""
    st.header("Affordability Analysis")

    st.markdown(
        """
        The **median multiple** (median house price / median household income) is a widely used
        measure of housing affordability. According to Demographia:

        | Rating | Multiple |
        |--------|----------|
        | Affordable | \u2264 3.0 |
        | Moderately Unaffordable | 3.1 - 4.0 |
        | Seriously Unaffordable | 4.1 - 5.0 |
        | Severely Unaffordable | > 5.0 |

        Hong Kong has consistently ranked as the **world's least affordable housing market** since 2010.
        """
    )

    fig = create_affordability_chart()
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Contributing Factors")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **Supply Constraints:**
            - Limited land supply (only ~25% of HK is developed)
            - Strict land use regulations
            - Government controls land release via auctions
            - Competition from mainland capital
            """
        )

    with col2:
        st.markdown(
            """
            **Demand Factors:**
            - Low interest rate environment (until 2022)
            - Safe haven for mainland wealth
            - Limited investment alternatives
            - Speculative activity
            """
        )


def show_policy_impact(policy_df):
    """Policy impact analysis page."""
    st.header("Government Policy Impact")

    st.markdown(
        """
        Hong Kong's government has implemented various **demand-side measures** to cool the property
        market since 2010. These include stamp duties targeting speculators and non-local buyers.
        """
    )

    fig = create_policy_timeline()
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Policy Details")

    for event in POLICY_EVENTS:
        if event["type"] == "cooling":
            badge = '<span style="display:inline-block; padding:2px 8px; background:#E05252; color:#0E1117; font-family:JetBrains Mono,monospace; font-size:0.7rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase;">COOLING</span>'
        elif event["type"] == "easing":
            badge = '<span style="display:inline-block; padding:2px 8px; background:#3FB68B; color:#0E1117; font-family:JetBrains Mono,monospace; font-size:0.7rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase;">EASING</span>'
        else:
            badge = '<span style="display:inline-block; padding:2px 8px; background:#A78BFA; color:#0E1117; font-family:JetBrains Mono,monospace; font-size:0.7rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase;">EXTERNAL</span>'

        st.markdown(
            f"""
            <div style="padding:1rem 0; border-bottom:1px solid #2A2F3A;">
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:0.5rem;">
                    <span style="font-family:JetBrains Mono,monospace; color:#6B7280; font-size:0.8rem;">{event['date']}</span>
                    {badge}
                </div>
                <div style="font-family:DM Serif Display,serif; font-size:1.1rem; color:#E6EDF3; margin-bottom:0.25rem;">{event['name']}</div>
                <div style="font-family:Inter,sans-serif; font-size:0.85rem; color:#8B949E;">{event['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_data_sources():
    """Data sources and methodology page."""
    st.header("Data Sources & Methodology")

    st.markdown(
        """
        ### Data Sources

        | Source | Data | Update Frequency |
        |--------|------|------------------|
        | [Rating & Valuation Department](https://www.rvd.gov.hk/en/property_market_statistics/index.html) | Price indices, rental indices, transactions, completions, stock | Monthly |
        | [HKMA](https://www.hkma.gov.hk/eng/data-publications-and-research/data-and-statistics/) | Mortgage survey, negative equity, interest rates | Monthly/Quarterly |
        | [Census & Statistics](https://www.censtatd.gov.hk/) | Household income, demographics | Annual |
        | [Demographia](http://www.demographia.com/) | International affordability comparisons | Annual |

        ### Methodology

        **Price Indices:**
        - RVD indices use 1999 as base year (1999 = 100)
        - Indices are transaction-based, reflecting actual sales
        - Broken down by property class (size):
          - Class A: < 40 m\u00b2
          - Class B: 40-69.9 m\u00b2
          - Class C: 70-99.9 m\u00b2
          - Class D: 100-159.9 m\u00b2
          - Class E: \u2265 160 m\u00b2

        **Affordability Ratio:**
        - Median dwelling price / Median annual household income
        - International standard for housing affordability comparison

        ### Data Refresh

        Data is cached for 1 hour to reduce load on government servers.
        """
    )

    st.markdown("---")
    st.subheader("Download Raw Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            '<a href="https://www.rvd.gov.hk/doc/en/statistics/his_data_2.xls" '
            'style="display:inline-block; padding:8px 16px; border:1px solid #C9A84C; '
            'color:#C9A84C !important; font-family:JetBrains Mono,monospace; font-size:0.75rem; '
            'font-weight:600; text-decoration:none; letter-spacing:0.05em; text-transform:uppercase;"'
            '>Price Indices XLS</a>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<a href="https://www.rvd.gov.hk/doc/en/statistics/his_data_4.xls" '
            'style="display:inline-block; padding:8px 16px; border:1px solid #C9A84C; '
            'color:#C9A84C !important; font-family:JetBrains Mono,monospace; font-size:0.75rem; '
            'font-weight:600; text-decoration:none; letter-spacing:0.05em; text-transform:uppercase;"'
            '>Rental Indices XLS</a>',
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            '<a href="https://www.rvd.gov.hk/doc/en/statistics/hs_data_8.xls" '
            'style="display:inline-block; padding:8px 16px; border:1px solid #C9A84C; '
            'color:#C9A84C !important; font-family:JetBrains Mono,monospace; font-size:0.75rem; '
            'font-weight:600; text-decoration:none; letter-spacing:0.05em; text-transform:uppercase;"'
            '>Transactions XLS</a>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
