"""Load and validate the official Hong Kong private housing index series."""

from __future__ import annotations

import io

import pandas as pd
import requests


PRICE_INDEX_URL = "https://www.rvd.gov.hk/doc/en/statistics/his_data_4.xls"
RENTAL_INDEX_URL = "https://www.rvd.gov.hk/doc/en/statistics/his_data_3.xls"

CLASS_LABELS = {
    "all": "All classes",
    "A": "Class A, under 40 m²",
    "B": "Class B, 40 to 69.9 m²",
    "C": "Class C, 70 to 99.9 m²",
    "D": "Class D, 100 to 159.9 m²",
    "E": "Class E, 160 m² or more",
}

RVD_COLUMNS = {
    "A": 8,
    "B": 11,
    "C": 14,
    "D": 17,
    "E": 20,
    "all": 29,
}


def download_workbook(url: str) -> bytes:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content


def parse_monthly_index(workbook: bytes | pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Parse the RVD monthly worksheet at one row per month."""
    if isinstance(workbook, pd.DataFrame):
        raw = workbook.copy()
    else:
        raw = pd.read_excel(io.BytesIO(workbook), sheet_name=0, header=None)

    if raw.shape[1] <= max(RVD_COLUMNS.values()):
        raise ValueError("The RVD workbook has fewer columns than expected.")

    year = pd.to_numeric(raw.iloc[:, 1], errors="coerce").ffill()
    month = pd.to_numeric(raw.iloc[:, 5], errors="coerce")
    parsed = pd.DataFrame({"date": pd.to_datetime(dict(year=year, month=month, day=1), errors="coerce")})

    for class_code, column_index in RVD_COLUMNS.items():
        parsed[f"{prefix}_{class_code}"] = pd.to_numeric(raw.iloc[:, column_index], errors="coerce")

    parsed = parsed.dropna(subset=["date", f"{prefix}_all"]).sort_values("date").reset_index(drop=True)

    if parsed.empty:
        raise ValueError(f"No valid {prefix} observations were found in the RVD workbook.")
    if parsed["date"].duplicated().any():
        raise ValueError(f"Duplicate monthly observations were found in the {prefix} series.")
    if parsed.filter(like=f"{prefix}_").le(0).any().any():
        raise ValueError(f"The {prefix} series contains a non-positive index value.")

    return parsed


def load_market_data() -> pd.DataFrame:
    price = parse_monthly_index(download_workbook(PRICE_INDEX_URL), "price")
    rental = parse_monthly_index(download_workbook(RENTAL_INDEX_URL), "rental")
    merged = price.merge(rental, on="date", how="inner", validate="one_to_one")

    if len(merged) < 300:
        raise ValueError("The merged RVD series is unexpectedly short.")
    if merged["date"].max() < pd.Timestamp.today().normalize() - pd.DateOffset(months=4):
        raise ValueError("The latest RVD observation is more than four months old.")

    return merged


def series_summary(data: pd.DataFrame, column: str) -> dict:
    series = data[["date", column]].dropna().sort_values("date")
    if len(series) < 13:
        raise ValueError(f"At least 13 observations are required for {column}.")

    latest = series.iloc[-1]
    previous = series.iloc[-2]
    year_ago_date = latest["date"] - pd.DateOffset(years=1)
    year_ago = series.loc[series["date"] == year_ago_date]
    if year_ago.empty:
        raise ValueError(f"A 12-month comparison is unavailable for {column}.")

    peak_row = series.loc[series[column].idxmax()]
    return {
        "latest": float(latest[column]),
        "mom": float((latest[column] / previous[column] - 1) * 100),
        "yoy": float((latest[column] / year_ago.iloc[-1][column] - 1) * 100),
        "peak": float(peak_row[column]),
        "peak_date": pd.Timestamp(peak_row["date"]),
        "drawdown": float((latest[column] / peak_row[column] - 1) * 100),
    }


def class_change_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for class_code, label in CLASS_LABELS.items():
        price = series_summary(data, f"price_{class_code}")
        rental = series_summary(data, f"rental_{class_code}")
        rows.append(
            {
                "class_code": class_code,
                "label": label,
                "price_latest": price["latest"],
                "price_change": price["yoy"],
                "rental_latest": rental["latest"],
                "rental_change": rental["yoy"],
            }
        )
    return pd.DataFrame(rows)
