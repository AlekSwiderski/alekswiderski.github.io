"""
Data fetcher for Hong Kong Property Market Dashboard
Fetches data from Rating & Valuation Department and HKMA
"""

import requests
import pandas as pd
from pathlib import Path
import io

# Base paths
DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# RVD Data URLs (Rating and Valuation Department)
RVD_URLS = {
    "price_indices": "https://www.rvd.gov.hk/doc/en/statistics/his_data_2.xls",
    "rental_indices": "https://www.rvd.gov.hk/doc/en/statistics/his_data_4.xls",
    "prices_by_class": "https://www.rvd.gov.hk/doc/en/statistics/his_data_1.xls",
    "rents_by_class": "https://www.rvd.gov.hk/doc/en/statistics/his_data_3.xls",
    "transactions": "https://www.rvd.gov.hk/doc/en/statistics/hs_data_8.xls",
    "completions": "https://www.rvd.gov.hk/doc/en/statistics/his_data_5.xls",
    "stock": "https://www.rvd.gov.hk/doc/en/statistics/his_data_6.xls",
    "vacancy": "https://www.rvd.gov.hk/doc/en/statistics/his_data_7.xls",
}

# HKMA Data (will use their API)
HKMA_API_BASE = "https://api.hkma.gov.hk/public/market-data-and-statistics/monthly-statistical-bulletin"


def fetch_rvd_data(data_key: str, save_raw: bool = True) -> pd.DataFrame | None:
    """
    Fetch data from RVD website.

    Args:
        data_key: Key from RVD_URLS dict
        save_raw: Whether to save the raw Excel file

    Returns:
        DataFrame or None if fetch fails
    """
    if data_key not in RVD_URLS:
        print(f"Unknown data key: {data_key}")
        return None

    url = RVD_URLS[data_key]
    print(f"Fetching {data_key} from {url}...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        if save_raw:
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            raw_path = RAW_DIR / f"{data_key}.xls"
            with open(raw_path, "wb") as f:
                f.write(response.content)
            print(f"Saved raw file to {raw_path}")

        # Read Excel file
        df = pd.read_excel(io.BytesIO(response.content), sheet_name=0)
        return df

    except requests.RequestException as e:
        print(f"Error fetching {data_key}: {e}")
        return None
    except Exception as e:
        print(f"Error parsing {data_key}: {e}")
        return None


def fetch_hkma_mortgage_data() -> pd.DataFrame | None:
    """
    Fetch residential mortgage survey data from HKMA API.

    Returns:
        DataFrame with mortgage statistics
    """
    url = f"{HKMA_API_BASE}/residential-mortgage-survey"

    print(f"Fetching HKMA mortgage data...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "result" in data and "records" in data["result"]:
            df = pd.DataFrame(data["result"]["records"])
            return df
        else:
            print("Unexpected API response structure")
            return None

    except requests.RequestException as e:
        print(f"Error fetching HKMA data: {e}")
        return None


def process_price_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process RVD price indices data into clean format.
    """
    if df is None:
        return None

    # RVD data typically has header rows to skip
    # Find the row with actual data headers
    df_clean = df.copy()

    # Save processed
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    return df_clean


def fetch_all_data():
    """Fetch all data sources and save to disk."""
    print("=" * 50)
    print("Fetching HK Property Market Data")
    print("=" * 50)

    results = {}

    # Fetch RVD data
    for key in ["price_indices", "rental_indices", "transactions"]:
        df = fetch_rvd_data(key)
        if df is not None:
            results[key] = df
            print(f"✓ {key}: {len(df)} rows")
        else:
            print(f"✗ {key}: Failed to fetch")

    # Fetch HKMA data
    mortgage_df = fetch_hkma_mortgage_data()
    if mortgage_df is not None:
        results["mortgage"] = mortgage_df
        print(f"✓ mortgage: {len(mortgage_df)} rows")
    else:
        print(f"✗ mortgage: Failed to fetch")

    print("=" * 50)
    print(f"Successfully fetched {len(results)} datasets")

    return results


if __name__ == "__main__":
    fetch_all_data()
