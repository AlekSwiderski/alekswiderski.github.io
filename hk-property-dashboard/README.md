# Hong Kong Property Market Dashboard

Interactive dashboard analyzing Hong Kong's property market through price trends, affordability metrics, and government policy impact.

**[Live Demo](https://hk-property-dashboard.streamlit.app)** *(link active after deployment)*

![Dashboard Preview](preview.png)

## Features

- **Price Trends**: Historical price and rental indices from 1979-present
- **Affordability Analysis**: Price-to-income ratios showing HK as world's least affordable market
- **Policy Impact**: Timeline of government cooling measures and their effects
- **Data Transparency**: Direct links to official government data sources

## Data Sources

| Source | Data | Update |
|--------|------|--------|
| [Rating & Valuation Dept](https://www.rvd.gov.hk/) | Price/rental indices, transactions | Monthly |
| [HKMA](https://www.hkma.gov.hk/) | Mortgage data, negative equity | Monthly |
| [Census & Statistics](https://www.censtatd.gov.hk/) | Income data | Annual |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## Deployment

This app is configured for [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push to GitHub
2. Connect repo to Streamlit Cloud
3. Deploy from `hk-property-dashboard/app.py`

## Project Structure

```
hk-property-dashboard/
├── app.py              # Main Streamlit application
├── data_fetcher.py     # Data ingestion utilities
├── policy_events.py    # Policy timeline data
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

## Key Insights

- Hong Kong property prices peaked in 2021, declining ~20% since
- Affordability ratio peaked at 23.2x income in 2021 (severely unaffordable = >5x)
- Government removed all stamp duties in Feb 2024 to stimulate market
- Prices continue to decline despite policy easing

## Author

[Alek Swiderski](https://alekswiderski.github.io)
