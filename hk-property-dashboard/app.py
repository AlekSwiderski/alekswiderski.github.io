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
    page_title="HK Property Market Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
    }
    .policy-cooling {
        color: #e74c3c;
        font-weight: bold;
    }
    .policy-easing {
        color: #27ae60;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    # Try multiple possible URLs for transaction data
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
    """
    Parse RVD price indices Excel into clean DataFrame.
    RVD files have specific structure with metadata rows.
    """
    if raw_df is None:
        return None

    df = raw_df.copy()

    # Find the header row (usually contains 'Year' or similar)
    header_row = None
    for i, row in df.iterrows():
        row_str = " ".join([str(x).lower() for x in row.values if pd.notna(x)])
        if "year" in row_str or "class" in row_str:
            header_row = i
            break

    if header_row is None:
        # Try to use row 3 as header (common in RVD files)
        header_row = 3

    # Set header and clean
    df.columns = df.iloc[header_row]
    df = df.iloc[header_row + 1 :].reset_index(drop=True)

    # Drop empty columns
    df = df.dropna(axis=1, how="all")

    # Clean column names
    df.columns = [str(c).strip() if pd.notna(c) else f"col_{i}" for i, c in enumerate(df.columns)]

    return df


def create_price_trend_chart(df, policy_df):
    """Create price trend chart with policy overlays."""
    fig = go.Figure()

    # Add price trend line (assuming 'All Classes' column exists)
    # We'll need to adapt this based on actual column names
    price_col = None
    for col in df.columns:
        if "all" in str(col).lower() or "overall" in str(col).lower():
            price_col = col
            break

    if price_col is None and len(df.columns) > 1:
        price_col = df.columns[1]  # Use first data column

    if price_col:
        # Try to create a date/year column
        year_col = df.columns[0]

        fig.add_trace(
            go.Scatter(
                x=df[year_col],
                y=pd.to_numeric(df[price_col], errors="coerce"),
                mode="lines",
                name="Price Index",
                line=dict(color="#3498db", width=2),
            )
        )

    # Add policy event markers
    for event in POLICY_EVENTS:
        color = POLICY_COLORS.get(event["type"], "#95a5a6")
        fig.add_vline(
            x=event["date"],
            line_dash="dash",
            line_color=color,
            opacity=0.7,
            annotation_text=event["name"],
            annotation_position="top",
            annotation_font_size=10,
        )

    fig.update_layout(
        title="Private Domestic Property Price Index with Policy Events",
        xaxis_title="Year",
        yaxis_title="Price Index (1999 = 100)",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )

    return fig


def create_affordability_chart():
    """
    Create affordability ratio visualization.
    Uses historical median price to income ratios.
    """
    # Historical affordability data (price-to-income ratio)
    # Source: Demographia, various years
    affordability_data = {
        "Year": [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        "Ratio": [11.4, 12.6, 13.5, 14.9, 17.0, 19.0, 18.1, 19.4, 20.9, 20.8, 20.7, 23.2, 18.8, 18.8, 16.7],
    }
    df = pd.DataFrame(affordability_data)

    fig = go.Figure()

    # Color bars based on affordability threshold
    colors = ["#e74c3c" if r > 15 else "#f39c12" if r > 10 else "#27ae60" for r in df["Ratio"]]

    fig.add_trace(
        go.Bar(
            x=df["Year"],
            y=df["Ratio"],
            marker_color=colors,
            text=df["Ratio"],
            textposition="outside",
        )
    )

    # Add threshold lines
    fig.add_hline(y=5, line_dash="dash", line_color="green", annotation_text="Affordable (<5)")
    fig.add_hline(y=10, line_dash="dash", line_color="orange", annotation_text="Severely Unaffordable (>10)")

    fig.update_layout(
        title="Hong Kong Housing Affordability (Median Price-to-Income Ratio)",
        xaxis_title="Year",
        yaxis_title="Price-to-Income Ratio",
        template="plotly_white",
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
                marker=dict(size=20, color=row["color"], symbol="diamond"),
                text=row["name"],
                textposition="top center",
                name=row["name"],
                hovertemplate=f"<b>{row['name']}</b><br>{row['date'].strftime('%Y-%m-%d')}<br>{row['description']}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Property Policy Timeline",
        xaxis_title="Date",
        yaxis_title="Policy Type",
        template="plotly_white",
        height=300,
        showlegend=False,
    )

    return fig


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🏠 Hong Kong Property Market Dashboard</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Analyzing price trends, affordability, and policy impact (1979-Present)</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select View",
        ["Overview", "Price Trends", "Affordability", "Policy Impact", "Data Sources"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard analyzes Hong Kong's property market using official data "
        "from the Rating and Valuation Department (RVD) and Hong Kong Monetary Authority (HKMA)."
    )

    # Load data (silently handle errors)
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

    # Key metrics row
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

    # Summary charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Affordability Trend")
        fig = create_affordability_chart()
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Recent Policy Changes")
        fig = create_policy_timeline()
        st.plotly_chart(fig, use_container_width=True)

    # Key insights
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

    # Parse data
    if raw_price_df is not None:
        st.subheader("Raw Price Index Data Preview")
        st.dataframe(raw_price_df.head(20), use_container_width=True)

        st.info(
            "💡 The RVD data requires parsing. The table above shows the raw format. "
            "Price indices are typically in columns by property class (A-E) with years in rows."
        )

        # Try to create chart
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
        | Affordable | ≤ 3.0 |
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

    # Policy timeline
    fig = create_policy_timeline()
    st.plotly_chart(fig, use_container_width=True)

    # Policy details table
    st.subheader("Policy Details")

    for event in POLICY_EVENTS:
        type_class = "policy-cooling" if event["type"] == "cooling" else "policy-easing"
        type_label = "🔴 Cooling" if event["type"] == "cooling" else "🟢 Easing" if event["type"] == "easing" else "🟣 External"

        st.markdown(
            f"""
            **{event['date']} - {event['name']}** {type_label}

            {event['description']}

            ---
            """
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
          - Class A: < 40 m²
          - Class B: 40-69.9 m²
          - Class C: 70-99.9 m²
          - Class D: 100-159.9 m²
          - Class E: ≥ 160 m²

        **Affordability Ratio:**
        - Median dwelling price / Median annual household income
        - International standard for housing affordability comparison

        ### Data Refresh

        Data is cached for 1 hour to reduce load on government servers.
        Last refresh: {current time}
        """
    )

    st.markdown("---")
    st.subheader("Download Raw Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("[📥 Price Indices (XLS)](https://www.rvd.gov.hk/doc/en/statistics/his_data_2.xls)")

    with col2:
        st.markdown("[📥 Rental Indices (XLS)](https://www.rvd.gov.hk/doc/en/statistics/his_data_4.xls)")

    with col3:
        st.markdown("[📥 Transactions (XLS)](https://www.rvd.gov.hk/doc/en/statistics/hs_data_8.xls)")


if __name__ == "__main__":
    main()
