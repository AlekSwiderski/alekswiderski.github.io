import pandas as pd
import pytest

from dashboard_data import class_change_table, parse_monthly_index, series_summary


def sample_raw() -> pd.DataFrame:
    raw = pd.DataFrame(index=range(24), columns=range(31))
    for offset in range(14):
        row = 10 + offset
        date = pd.Timestamp("2024-01-01") + pd.DateOffset(months=offset)
        if date.month == 1:
            raw.iloc[row, 1] = date.year
        raw.iloc[row, 5] = date.month
        for column, base in [(8, 100), (11, 110), (14, 120), (17, 130), (20, 140), (29, 115)]:
            raw.iloc[row, column] = base + offset
    return raw


def test_parse_monthly_index_extracts_dates_and_classes():
    parsed = parse_monthly_index(sample_raw(), "price")
    assert len(parsed) == 14
    assert parsed.iloc[0]["date"] == pd.Timestamp("2024-01-01")
    assert parsed.iloc[-1]["date"] == pd.Timestamp("2025-02-01")
    assert parsed.iloc[-1]["price_all"] == 128


def test_series_summary_uses_a_true_twelve_month_comparison():
    data = parse_monthly_index(sample_raw(), "price")
    summary = series_summary(data, "price_all")
    assert summary["latest"] == 128
    assert summary["yoy"] == pytest.approx((128 / 116 - 1) * 100)
    assert summary["peak_date"] == pd.Timestamp("2025-02-01")
    assert summary["drawdown"] == 0


def test_class_change_table_has_one_row_per_class():
    price = parse_monthly_index(sample_raw(), "price")
    rental = parse_monthly_index(sample_raw(), "rental")
    market = price.merge(rental, on="date", validate="one_to_one")
    changes = class_change_table(market)
    assert changes["class_code"].tolist() == ["all", "A", "B", "C", "D", "E"]
    assert changes[["price_change", "rental_change"]].notna().all().all()
