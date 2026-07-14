# Hong Kong private housing

A Streamlit dashboard for the official monthly sale price and rental indices published by Hong Kong's Rating and Valuation Department.

[Open the live dashboard](https://alekswiderskiappio-6gsea6xfxap4xwqfjxy46p.streamlit.app/)

## What it covers

- Monthly private domestic sale price and rental indices from January 1993
- Latest monthly and 12-month changes
- Distance from each series peak
- Comparisons across RVD flat size classes A to E
- Selected residential property tax and mortgage policy dates

Both index series use 1999 = 100. They can be compared as indices, but they do not represent cash prices, rents or rental yields.

## Official sources

- [Private domestic price indices by class](https://www.rvd.gov.hk/doc/en/statistics/his_data_4.xls)
- [Private domestic rental indices by class](https://www.rvd.gov.hk/doc/en/statistics/his_data_3.xls)

The latest RVD observations are provisional. The app requests the workbooks when it starts and caches the cleaned data for one hour.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Run the parser and metric tests with:

```bash
pytest -q
```
