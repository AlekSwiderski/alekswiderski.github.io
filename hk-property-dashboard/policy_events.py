"""Selected Hong Kong residential property policy events."""

POLICY_EVENTS = [
    {
        "date": "2010-11-20",
        "name": "Special Stamp Duty introduced",
        "description": "A new duty applied to homes resold within 24 months of purchase.",
        "type": "cooling",
    },
    {
        "date": "2012-10-27",
        "name": "Buyer's Stamp Duty introduced",
        "description": "A 15% duty was introduced for buyers who were not Hong Kong permanent residents.",
        "type": "cooling",
    },
    {
        "date": "2013-02-23",
        "name": "Ad valorem duty increased",
        "description": "Higher rates applied to most residential purchases other than eligible first-time local buyers.",
        "type": "cooling",
    },
    {
        "date": "2016-11-05",
        "name": "Flat 15% duty introduced",
        "description": "The ad valorem rate rose to 15% for most buyers who already owned a home.",
        "type": "cooling",
    },
    {
        "date": "2017-04-12",
        "name": "Mortgage limits tightened",
        "description": "HKMA lowered loan-to-value limits for borrowers with existing mortgages and for investment property.",
        "type": "cooling",
    },
    {
        "date": "2019-10-16",
        "name": "Mortgage limits relaxed",
        "description": "Mortgage insurance coverage expanded for eligible first-time buyers of homes priced up to HK$8 million.",
        "type": "easing",
    },
    {
        "date": "2020-01-25",
        "name": "First local COVID-19 cases",
        "description": "Hong Kong confirmed its first COVID-19 cases as travel and economic activity began to slow.",
        "type": "external",
    },
    {
        "date": "2023-10-25",
        "name": "Stamp duties reduced",
        "description": "Buyer's Stamp Duty and the new residential duty were halved, while the SSD holding period was shortened.",
        "type": "easing",
    },
    {
        "date": "2024-02-28",
        "name": "Extra stamp duties withdrawn",
        "description": "The government removed SSD, Buyer's Stamp Duty and the higher residential ad valorem rate.",
        "type": "easing",
    },
    {
        "date": "2024-10-16",
        "name": "Mortgage rules standardised",
        "description": "HKMA set the maximum loan-to-value ratio at 70% and the debt servicing ratio limit at 50% for residential property.",
        "type": "mortgage",
    },
]

POLICY_COLORS = {
    "cooling": "#a84f3d",
    "easing": "#496c5a",
    "mortgage": "#244b68",
    "external": "#82725f",
}


def get_policy_events_df():
    """Convert policy events to a pandas DataFrame."""
    import pandas as pd

    df = pd.DataFrame(POLICY_EVENTS)
    df["date"] = pd.to_datetime(df["date"])
    df["color"] = df["type"].map(POLICY_COLORS)
    return df
