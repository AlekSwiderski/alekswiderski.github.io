"""Rebuild the portfolio case-study data from public ETF price history.

Run with:
    python portfolio_optimization.py

The script downloads adjusted daily prices from Yahoo Finance, calculates one
set of annualised assumptions, and writes the static data used by the GitHub
Pages case study. The web page does not recalculate or substitute correlations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize


TICKERS = ["SPY", "QQQ", "AGG", "GLD", "VNQ", "VEA", "VWO"]
NAMES = {
    "SPY": "US large cap",
    "QQQ": "Nasdaq 100",
    "AGG": "US bonds",
    "GLD": "Gold",
    "VNQ": "US real estate",
    "VEA": "Developed markets",
    "VWO": "Emerging markets",
}
START = "2018-01-01"
END_EXCLUSIVE = "2025-04-01"
TRADING_DAYS = 252
RISK_FREE_RATE = 0.02
OUTPUT = Path(__file__).with_name("portfolio-optimization-data.js")


def download_prices() -> pd.DataFrame:
    """Download each ticker separately to avoid Yahoo's threaded cache lock."""
    series: list[pd.Series] = []
    for ticker in TICKERS:
        frame = yf.download(
            ticker,
            start=START,
            end=END_EXCLUSIVE,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if frame.empty:
            raise RuntimeError(f"No price history returned for {ticker}")
        close = frame["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.name = ticker
        series.append(close)

    prices = pd.concat(series, axis=1).dropna()
    if list(prices.columns) != TICKERS:
        raise RuntimeError("Downloaded columns do not match the requested universe")
    return prices


def portfolio_metrics(
    weights: np.ndarray, mean_returns: np.ndarray, covariance: np.ndarray
) -> tuple[float, float, float]:
    expected_return = float(weights @ mean_returns)
    volatility = float(np.sqrt(weights @ covariance @ weights))
    sharpe = (expected_return - RISK_FREE_RATE) / volatility
    return expected_return, volatility, sharpe


def solve_portfolios(
    mean_returns: np.ndarray, covariance: np.ndarray
) -> tuple[dict, dict, list[dict]]:
    count = len(mean_returns)
    equal = np.full(count, 1 / count)
    bounds = [(0.0, 1.0)] * count
    fully_invested = {"type": "eq", "fun": lambda weights: weights.sum() - 1}

    min_vol_result = minimize(
        lambda weights: np.sqrt(weights @ covariance @ weights),
        equal,
        method="SLSQP",
        bounds=bounds,
        constraints=[fully_invested],
        options={"ftol": 1e-12, "maxiter": 2_000},
    )
    if not min_vol_result.success:
        raise RuntimeError(f"Minimum volatility optimisation failed: {min_vol_result.message}")

    max_sharpe_result = minimize(
        lambda weights: -portfolio_metrics(weights, mean_returns, covariance)[2],
        equal,
        method="SLSQP",
        bounds=bounds,
        constraints=[fully_invested],
        options={"ftol": 1e-12, "maxiter": 2_000},
    )
    if not max_sharpe_result.success:
        raise RuntimeError(f"Maximum Sharpe optimisation failed: {max_sharpe_result.message}")

    minimum_return = portfolio_metrics(
        min_vol_result.x, mean_returns, covariance
    )[0]
    target_returns = np.linspace(minimum_return, float(mean_returns.max()), 48)
    frontier: list[dict] = []
    starting_weights = min_vol_result.x

    for target in target_returns:
        constraints = [
            fully_invested,
            {
                "type": "eq",
                "fun": lambda weights, target=target: weights @ mean_returns - target,
            },
        ]
        result = minimize(
            lambda weights: np.sqrt(weights @ covariance @ weights),
            starting_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-11, "maxiter": 2_000},
        )
        if result.success:
            expected_return, volatility, sharpe = portfolio_metrics(
                result.x, mean_returns, covariance
            )
            frontier.append(
                {
                    "return": round(expected_return, 8),
                    "volatility": round(volatility, 8),
                    "sharpe": round(sharpe, 8),
                    "weights": [round(float(weight), 8) for weight in result.x],
                }
            )
            starting_weights = result.x

    def describe(weights: np.ndarray) -> dict:
        expected_return, volatility, sharpe = portfolio_metrics(
            weights, mean_returns, covariance
        )
        return {
            "return": round(expected_return, 8),
            "volatility": round(volatility, 8),
            "sharpe": round(sharpe, 8),
            "weights": [round(float(weight), 8) for weight in weights],
        }

    return describe(min_vol_result.x), describe(max_sharpe_result.x), frontier


def build_payload(prices: pd.DataFrame) -> dict:
    returns = prices.pct_change(fill_method=None).dropna()
    annual_returns = returns.mean() * TRADING_DAYS
    annual_covariance = returns.cov() * TRADING_DAYS
    annual_volatility = np.sqrt(np.diag(annual_covariance))

    min_vol, max_sharpe, frontier = solve_portfolios(
        annual_returns.to_numpy(), annual_covariance.to_numpy()
    )
    equal_weights = np.full(len(TICKERS), 1 / len(TICKERS))
    equal_return, equal_volatility, equal_sharpe = portfolio_metrics(
        equal_weights, annual_returns.to_numpy(), annual_covariance.to_numpy()
    )

    return {
        "tickers": TICKERS,
        "names": [NAMES[ticker] for ticker in TICKERS],
        "period": {
            "start": prices.index.min().date().isoformat(),
            "end": prices.index.max().date().isoformat(),
            "observations": int(len(returns)),
        },
        "method": {
            "source": "Yahoo Finance via yfinance",
            "prices": "Adjusted daily close",
            "annualisation": TRADING_DAYS,
            "risk_free_rate": RISK_FREE_RATE,
            "constraints": "Long only; weights sum to 100%",
        },
        "annual_returns": [round(float(value), 8) for value in annual_returns],
        "annual_volatility": [round(float(value), 8) for value in annual_volatility],
        "covariance": [
            [round(float(value), 10) for value in row]
            for row in annual_covariance.to_numpy()
        ],
        "correlation": [
            [round(float(value), 4) for value in row]
            for row in returns.corr().to_numpy()
        ],
        "total_returns": [
            round(float(prices[ticker].iloc[-1] / prices[ticker].iloc[0] - 1), 8)
            for ticker in TICKERS
        ],
        "min_volatility": min_vol,
        "max_sharpe": max_sharpe,
        "equal_weight": {
            "return": round(equal_return, 8),
            "volatility": round(equal_volatility, 8),
            "sharpe": round(equal_sharpe, 8),
            "weights": [round(float(weight), 8) for weight in equal_weights],
        },
        "frontier": frontier,
    }


def main() -> None:
    prices = download_prices()
    payload = build_payload(prices)
    OUTPUT.write_text(
        "window.PORTFOLIO_DATA = "
        + json.dumps(payload, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {OUTPUT.name}: {payload['period']['observations']} daily returns, "
        f"{len(payload['frontier'])} frontier portfolios"
    )


if __name__ == "__main__":
    main()
