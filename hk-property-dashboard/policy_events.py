"""
Hong Kong Property Policy Timeline
Key government interventions and their dates
"""

POLICY_EVENTS = [
    {
        "date": "2010-11-20",
        "name": "SSD Introduced",
        "description": "Special Stamp Duty (SSD) introduced for properties resold within 24 months",
        "type": "cooling",
    },
    {
        "date": "2012-10-27",
        "name": "BSD Introduced",
        "description": "Buyer's Stamp Duty (BSD) of 15% for non-permanent residents",
        "type": "cooling",
    },
    {
        "date": "2013-02-23",
        "name": "DSD Doubled",
        "description": "Double Stamp Duty (DSD) for all property transactions except first-time HK permanent resident buyers",
        "type": "cooling",
    },
    {
        "date": "2016-11-05",
        "name": "15% Flat AVD",
        "description": "Ad Valorem Stamp Duty increased to flat 15% for non-first-time buyers",
        "type": "cooling",
    },
    {
        "date": "2017-04-12",
        "name": "LTV Tightening",
        "description": "HKMA tightened loan-to-value ratios for investment properties",
        "type": "cooling",
    },
    {
        "date": "2019-10-16",
        "name": "LTV Relaxation",
        "description": "HKMA relaxed LTV caps for first-time homebuyers (up to 90% for properties up to HK$8M)",
        "type": "easing",
    },
    {
        "date": "2020-01-25",
        "name": "COVID-19 Outbreak",
        "description": "First COVID-19 cases in Hong Kong, beginning of pandemic impact",
        "type": "external",
    },
    {
        "date": "2023-10-25",
        "name": "Partial Stamp Duty Relaxation",
        "description": "BSD and SSD rates reduced; DSD halved",
        "type": "easing",
    },
    {
        "date": "2024-02-28",
        "name": "Full Stamp Duty Removal",
        "description": "All extra stamp duties (SSD, BSD, DSD) completely removed",
        "type": "easing",
    },
]

# Color coding for policy types
POLICY_COLORS = {
    "cooling": "#e74c3c",  # Red - restrictive measures
    "easing": "#27ae60",   # Green - relaxation measures
    "external": "#9b59b6", # Purple - external events
}


def get_policy_events_df():
    """Convert policy events to a pandas DataFrame."""
    import pandas as pd

    df = pd.DataFrame(POLICY_EVENTS)
    df["date"] = pd.to_datetime(df["date"])
    df["color"] = df["type"].map(POLICY_COLORS)
    return df
