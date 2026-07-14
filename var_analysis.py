"""Generate the data used by the Value at Risk case study.

The example is a hypothetical $100,000 portfolio rebalanced to its target
weights each trading day. All reported VaR figures are one-day loss estimates.
Run this file to refresh ``var-analysis-data.js``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from scipy.special import xlogy


@dataclass(frozen=True)
class AnalysisConfig:
    assets: tuple[str, ...] = ("AAPL", "MSFT", "JPM", "GS", "XLF", "TLT", "GLD")
    weights: tuple[float, ...] = (0.15, 0.15, 0.15, 0.15, 0.20, 0.10, 0.10)
    start: str = "2021-01-01"
    end_exclusive: str = "2025-04-01"
    investment: float = 100_000
    rolling_window: int = 252
    simulations: int = 100_000
    seed: int = 42


CONFIG = AnalysisConfig()
CONFIDENCE_LEVELS = (0.90, 0.95, 0.975, 0.99)
OUTPUT = Path(__file__).with_name("var-analysis-data.js")


def download_adjusted_prices(config: AnalysisConfig) -> pd.DataFrame:
    """Download each asset separately and align common trading dates."""
    series: list[pd.Series] = []
    for ticker in config.assets:
        frame = yf.download(
            ticker,
            start=config.start,
            end=config.end_exclusive,
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
    if list(prices.columns) != list(config.assets):
        raise RuntimeError("Downloaded columns do not match the requested portfolio")
    return prices


def portfolio_returns(prices: pd.DataFrame, config: AnalysisConfig) -> pd.Series:
    """Calculate simple returns for a portfolio reset to target weights daily."""
    asset_returns = prices.pct_change(fill_method=None).dropna()
    weights = np.asarray(config.weights)
    returns = asset_returns @ weights
    returns.name = "portfolio_return"
    return returns


def historical_risk(returns: pd.Series, confidence: float) -> dict:
    threshold = float(np.quantile(returns, 1 - confidence))
    tail = returns[returns <= threshold]
    return {
        "var_return": round(-threshold, 8),
        "expected_shortfall_return": round(float(-tail.mean()), 8),
    }


def parametric_risk(returns: pd.Series, confidence: float) -> dict:
    mean = float(returns.mean())
    volatility = float(returns.std(ddof=1))
    alpha = 1 - confidence
    z_score = float(stats.norm.ppf(alpha))
    threshold = mean + z_score * volatility
    tail_mean = mean - volatility * stats.norm.pdf(z_score) / alpha
    return {
        "var_return": round(-threshold, 8),
        "expected_shortfall_return": round(float(-tail_mean), 8),
    }


def monte_carlo_risk(
    returns: pd.Series, confidence: float, config: AnalysisConfig
) -> dict:
    """Sample the fitted normal model with a fixed seed for reproducibility."""
    generator = np.random.default_rng(config.seed)
    simulated = generator.normal(
        float(returns.mean()),
        float(returns.std(ddof=1)),
        config.simulations,
    )
    threshold = float(np.quantile(simulated, 1 - confidence))
    tail = simulated[simulated <= threshold]
    return {
        "var_return": round(-threshold, 8),
        "expected_shortfall_return": round(float(-tail.mean()), 8),
    }


def kupiec_test(breaches: int, observations: int, confidence: float) -> dict:
    """Kupiec unconditional coverage test using stable log likelihoods."""
    expected_probability = 1 - confidence
    observed_probability = breaches / observations
    null_log_likelihood = (
        xlogy(observations - breaches, 1 - expected_probability)
        + xlogy(breaches, expected_probability)
    )
    fitted_log_likelihood = (
        xlogy(observations - breaches, 1 - observed_probability)
        + xlogy(breaches, observed_probability)
    )
    statistic = float(-2 * (null_log_likelihood - fitted_log_likelihood))
    p_value = float(stats.chi2.sf(statistic, 1))
    return {
        "statistic": round(statistic, 6),
        "p_value": round(p_value, 6),
        "passes_at_5pct": p_value > 0.05,
    }


def rolling_backtest(
    returns: pd.Series,
    confidence: float,
    method: str,
    config: AnalysisConfig,
) -> dict:
    values = returns.to_numpy()
    thresholds: list[float] = []
    actual: list[float] = []
    dates: list[str] = []

    for index in range(config.rolling_window, len(values)):
        sample = values[index - config.rolling_window:index]
        if method == "historical":
            threshold = float(np.quantile(sample, 1 - confidence))
        elif method == "parametric":
            threshold = float(
                sample.mean()
                + stats.norm.ppf(1 - confidence) * sample.std(ddof=1)
            )
        else:
            raise ValueError(f"Unknown backtest method: {method}")

        thresholds.append(threshold)
        actual.append(float(values[index]))
        dates.append(returns.index[index].date().isoformat())

    breach_mask = np.asarray(actual) < np.asarray(thresholds)
    breach_count = int(breach_mask.sum())
    observations = len(actual)
    test = kupiec_test(breach_count, observations, confidence)

    return {
        "method": method,
        "confidence": confidence,
        "window": config.rolling_window,
        "observations": observations,
        "breaches": breach_count,
        "breach_rate": round(breach_count / observations, 8),
        "expected_rate": round(1 - confidence, 8),
        "kupiec": test,
        "series": [
            {
                "date": date,
                "loss": round(-return_value, 8),
                "var": round(-threshold, 8),
                "breach": bool(is_breach),
            }
            for date, return_value, threshold, is_breach in zip(
                dates, actual, thresholds, breach_mask
            )
        ],
    }


def drawdown_summary(portfolio_value: pd.Series) -> dict:
    running_peak = portfolio_value.cummax()
    drawdown = portfolio_value / running_peak - 1
    trough_date = drawdown.idxmin()
    peak_date = portfolio_value.loc[:trough_date].idxmax()
    peak_value = portfolio_value.loc[peak_date]
    recovered = portfolio_value.loc[trough_date:]
    recovered = recovered[recovered >= peak_value]
    recovery_date = recovered.index[0] if not recovered.empty else None
    return {
        "max_drawdown": round(float(drawdown.min()), 8),
        "peak_date": peak_date.date().isoformat(),
        "trough_date": trough_date.date().isoformat(),
        "recovery_date": recovery_date.date().isoformat() if recovery_date else None,
    }


def build_payload(prices: pd.DataFrame, config: AnalysisConfig) -> dict:
    returns = portfolio_returns(prices, config)
    portfolio_value = config.investment * (1 + returns).cumprod()
    risk = {}
    for confidence in CONFIDENCE_LEVELS:
        key = str(confidence)
        risk[key] = {
            "historical": historical_risk(returns, confidence),
            "parametric": parametric_risk(returns, confidence),
            "monte_carlo": monte_carlo_risk(returns, confidence, config),
        }

    historical_backtest = rolling_backtest(returns, 0.95, "historical", config)
    parametric_backtest = rolling_backtest(returns, 0.95, "parametric", config)
    worst_days = returns.nsmallest(5)

    return {
        "portfolio": {
            "assets": list(config.assets),
            "weights": list(config.weights),
            "investment": config.investment,
            "rebalancing": "Daily to target weights",
        },
        "period": {
            "start": prices.index.min().date().isoformat(),
            "end": prices.index.max().date().isoformat(),
            "observations": int(len(returns)),
        },
        "source": "Yahoo Finance via yfinance, adjusted daily close",
        "performance": {
            "final_value": round(float(portfolio_value.iloc[-1]), 2),
            "total_return": round(float(portfolio_value.iloc[-1] / config.investment - 1), 8),
            "annual_return": round(float(returns.mean() * 252), 8),
            "annual_volatility": round(float(returns.std(ddof=1) * np.sqrt(252)), 8),
            "drawdown": drawdown_summary(portfolio_value),
        },
        "risk": risk,
        "backtests": {
            "historical": historical_backtest,
            "parametric": parametric_backtest,
        },
        "returns": [round(float(value), 8) for value in returns],
        "value_series": [
            {"date": date.date().isoformat(), "value": round(float(value), 2)}
            for date, value in portfolio_value.items()
        ],
        "worst_days": [
            {"date": date.date().isoformat(), "return": round(float(value), 8)}
            for date, value in worst_days.items()
        ],
        "method": {
            "return_type": "Simple daily returns",
            "rolling_window": config.rolling_window,
            "simulations": config.simulations,
            "seed": config.seed,
            "confidence_levels": list(CONFIDENCE_LEVELS),
        },
    }


def main() -> None:
    prices = download_adjusted_prices(CONFIG)
    payload = build_payload(prices, CONFIG)
    OUTPUT.write_text(
        "window.VAR_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    backtest = payload["backtests"]["historical"]
    print(
        f"Wrote {OUTPUT.name}: {payload['period']['observations']} returns, "
        f"{backtest['breaches']}/{backtest['observations']} historical VaR breaches"
    )


if __name__ == "__main__":
    main()
