"""
Value at Risk (VaR) Analysis Tool

A comprehensive Python implementation for portfolio risk assessment using
multiple VaR calculation methods, backtesting, and stress testing.

Author: Alek Swiderski
"""

import warnings
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')


class VaRAnalysis:
    """
    Value at Risk analysis framework supporting multiple calculation methods,
    backtesting, stress testing, and visualization.
    """

    # Stress test scenarios with historical impact percentages
    STRESS_SCENARIOS = {
        '2008 Financial Crisis': -0.45,
        '2020 COVID Crash': -0.35,
        'Tech Bubble Burst': -0.30,
        'Moderate Recession': -0.25,
        '2018 December Selloff': -0.20,
    }

    def __init__(
        self,
        assets: List[str],
        weights: List[float],
        investment: float = 100000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Initialize the VaR analysis with portfolio parameters.

        Args:
            assets: List of ticker symbols (e.g., ['AAPL', 'MSFT'])
            weights: Portfolio weights for each asset (must sum to 1)
            investment: Initial investment amount in dollars
            start_date: Start date for analysis (YYYY-MM-DD), default 4 years ago
            end_date: End date for analysis (YYYY-MM-DD), default today
        """
        if len(assets) != len(weights):
            raise ValueError("Number of assets must match number of weights")

        if not np.isclose(sum(weights), 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights):.4f}")

        self.assets = assets
        self.weights = np.array(weights)
        self.investment = investment

        # Set date range
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.start_date = start_date or (
            datetime.now() - timedelta(days=4*365)
        ).strftime('%Y-%m-%d')

        # Data containers
        self.prices = None
        self.returns = None
        self.portfolio_returns = None
        self.portfolio_value = None

        # Load data on initialization
        self._load_data()

    def _load_data(self) -> None:
        """Download historical price data and calculate returns."""
        print("=" * 70)
        print("LOADING PORTFOLIO DATA")
        print("=" * 70)

        try:
            print(f"\n1. Downloading data for {len(self.assets)} assets...")
            print(f"   Period: {self.start_date} to {self.end_date}")

            # Download price data
            data = yf.download(
                self.assets,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )

            # Handle yfinance API changes - extract Close prices
            if isinstance(data.columns, pd.MultiIndex):
                # New API: MultiIndex columns like ('Close', 'SPY')
                self.prices = data['Close']
            elif 'Adj Close' in data.columns:
                # Old API with Adj Close
                self.prices = data['Adj Close']
            elif 'Close' in data.columns:
                # Old API with just Close
                self.prices = data['Close']
            else:
                self.prices = data

            # Handle single asset case
            if isinstance(self.prices, pd.Series):
                self.prices = self.prices.to_frame(name=self.assets[0])

            # Reorder columns to match asset order
            self.prices = self.prices[self.assets]

            # Forward fill missing values
            self.prices = self.prices.ffill().dropna()

            print(f"   Downloaded {len(self.prices)} trading days")

            # Calculate daily log returns
            print("\n2. Calculating returns...")
            self.returns = np.log(self.prices / self.prices.shift(1)).dropna()

            # Calculate portfolio returns (weighted sum)
            self.portfolio_returns = (self.returns * self.weights).sum(axis=1)

            # Calculate portfolio value over time
            cumulative_returns = (1 + self.portfolio_returns).cumprod()
            self.portfolio_value = self.investment * cumulative_returns

            print(f"   Calculated {len(self.returns)} daily returns")
            print(f"   Portfolio value: ${self.investment:,.0f} -> ${self.portfolio_value.iloc[-1]:,.0f}")

        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

    def historical_var(
        self,
        confidence_level: float = 0.95
    ) -> Tuple[float, Dict]:
        """
        Calculate VaR using the historical simulation method.

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Tuple of (VaR in dollars, details dict)
        """
        # Calculate percentile of returns
        var_percentile = np.percentile(
            self.portfolio_returns,
            (1 - confidence_level) * 100
        )

        # Convert to dollar amount
        var_dollars = abs(var_percentile * self.investment)

        # Calculate CVaR (Expected Shortfall)
        cvar_returns = self.portfolio_returns[
            self.portfolio_returns <= var_percentile
        ]
        cvar_dollars = abs(cvar_returns.mean() * self.investment)

        details = {
            'method': 'historical',
            'confidence_level': confidence_level,
            'var_return': var_percentile,
            'var_dollars': var_dollars,
            'cvar_dollars': cvar_dollars,
            'observations': len(self.portfolio_returns)
        }

        return var_dollars, details

    def parametric_var(
        self,
        confidence_level: float = 0.95
    ) -> Tuple[float, Dict]:
        """
        Calculate VaR using the parametric (variance-covariance) method.
        Assumes returns follow a normal distribution.

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Tuple of (VaR in dollars, details dict)
        """
        # Calculate portfolio statistics
        mean_return = self.portfolio_returns.mean()
        std_return = self.portfolio_returns.std()

        # Get z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)

        # Calculate VaR
        var_return = mean_return + z_score * std_return
        var_dollars = abs(var_return * self.investment)

        # Calculate CVaR for normal distribution
        cvar_multiplier = stats.norm.pdf(z_score) / (1 - confidence_level)
        cvar_return = mean_return - std_return * cvar_multiplier
        cvar_dollars = abs(cvar_return * self.investment)

        details = {
            'method': 'parametric',
            'confidence_level': confidence_level,
            'mean_return': mean_return,
            'std_return': std_return,
            'z_score': z_score,
            'var_return': var_return,
            'var_dollars': var_dollars,
            'cvar_dollars': cvar_dollars
        }

        return var_dollars, details

    def monte_carlo_var(
        self,
        confidence_level: float = 0.95,
        simulations: int = 10000
    ) -> Tuple[float, Dict]:
        """
        Calculate VaR using Monte Carlo simulation.

        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            simulations: Number of random scenarios to generate

        Returns:
            Tuple of (VaR in dollars, details dict)
        """
        # Get portfolio return statistics
        mean_return = self.portfolio_returns.mean()
        std_return = self.portfolio_returns.std()

        # Generate random returns
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(mean_return, std_return, simulations)

        # Calculate VaR from simulated distribution
        var_return = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        var_dollars = abs(var_return * self.investment)

        # Calculate CVaR
        cvar_returns = simulated_returns[simulated_returns <= var_return]
        cvar_dollars = abs(cvar_returns.mean() * self.investment)

        details = {
            'method': 'monte_carlo',
            'confidence_level': confidence_level,
            'simulations': simulations,
            'mean_return': mean_return,
            'std_return': std_return,
            'var_return': var_return,
            'var_dollars': var_dollars,
            'cvar_dollars': cvar_dollars
        }

        return var_dollars, details

    def backtest(
        self,
        method: str = 'historical',
        confidence_level: float = 0.95,
        window: int = 252
    ) -> Dict:
        """
        Backtest the VaR model against historical data.

        Args:
            method: VaR calculation method ('historical', 'parametric', 'monte_carlo')
            confidence_level: Confidence level for VaR
            window: Rolling window size for VaR calculation

        Returns:
            Dictionary with backtest results
        """
        print(f"\nBacktesting {method} VaR ({confidence_level*100:.0f}% confidence)...")

        breaches = 0
        total_obs = 0
        breach_dates = []

        returns_array = self.portfolio_returns.values

        for i in range(window, len(returns_array)):
            # Calculate VaR using rolling window
            window_returns = returns_array[i-window:i]

            if method == 'historical':
                var_return = np.percentile(window_returns, (1 - confidence_level) * 100)
            elif method == 'parametric':
                mean = np.mean(window_returns)
                std = np.std(window_returns)
                z = stats.norm.ppf(1 - confidence_level)
                var_return = mean + z * std
            else:  # monte_carlo
                mean = np.mean(window_returns)
                std = np.std(window_returns)
                simulated = np.random.normal(mean, std, 1000)
                var_return = np.percentile(simulated, (1 - confidence_level) * 100)

            # Check if actual return breached VaR
            actual_return = returns_array[i]
            if actual_return < var_return:
                breaches += 1
                breach_dates.append(self.portfolio_returns.index[i])

            total_obs += 1

        # Calculate breach rate
        breach_rate = breaches / total_obs if total_obs > 0 else 0
        expected_rate = 1 - confidence_level

        # Kupiec POF test
        kupiec_result = self._kupiec_test(breaches, total_obs, confidence_level)

        results = {
            'method': method,
            'confidence_level': confidence_level,
            'window': window,
            'total_observations': total_obs,
            'breaches': breaches,
            'breach_rate': breach_rate,
            'expected_rate': expected_rate,
            'kupiec_statistic': kupiec_result['statistic'],
            'kupiec_p_value': kupiec_result['p_value'],
            'kupiec_passed': kupiec_result['passed'],
            'breach_dates': breach_dates
        }

        print(f"   Breaches: {breaches}/{total_obs} ({breach_rate*100:.2f}%)")
        print(f"   Expected: {expected_rate*100:.2f}%")
        print(f"   Kupiec test: {'PASSED' if kupiec_result['passed'] else 'FAILED'}")

        return results

    def _kupiec_test(
        self,
        breaches: int,
        total_obs: int,
        confidence_level: float
    ) -> Dict:
        """
        Perform Kupiec Proportion of Failures (POF) test.

        Args:
            breaches: Number of VaR breaches
            total_obs: Total number of observations
            confidence_level: VaR confidence level

        Returns:
            Dictionary with test results
        """
        p = 1 - confidence_level  # Expected failure rate
        n = total_obs
        x = breaches

        # Avoid log(0)
        if x == 0:
            x = 0.5
        if x == n:
            x = n - 0.5

        # Likelihood ratio statistic
        lr = -2 * (
            np.log((1 - p)**(n - x) * p**x) -
            np.log((1 - x/n)**(n - x) * (x/n)**x)
        )

        # p-value from chi-squared distribution with 1 df
        p_value = 1 - stats.chi2.cdf(lr, 1)

        return {
            'statistic': lr,
            'p_value': p_value,
            'passed': p_value > 0.05  # 5% significance level
        }

    def stress_test(self) -> pd.DataFrame:
        """
        Simulate portfolio performance under various stress scenarios.

        Returns:
            DataFrame with stress test results
        """
        results = []

        for scenario, impact in self.STRESS_SCENARIOS.items():
            loss = self.investment * abs(impact)
            remaining = self.investment + (self.investment * impact)

            results.append({
                'Scenario': scenario,
                'Impact (%)': f"{impact*100:.0f}%",
                'Loss ($)': f"${loss:,.0f}",
                'Remaining Value ($)': f"${remaining:,.0f}"
            })

        return pd.DataFrame(results)

    def plot_portfolio_performance(self, save_path: Optional[str] = None) -> None:
        """Plot portfolio value over time."""
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(self.portfolio_value.index, self.portfolio_value.values,
                color='#2c3e50', linewidth=2)
        ax.axhline(y=self.investment, color='#e74c3c', linestyle='--',
                   label=f'Initial Investment: ${self.investment:,.0f}')

        ax.fill_between(self.portfolio_value.index, self.investment,
                        self.portfolio_value.values,
                        where=self.portfolio_value.values >= self.investment,
                        color='#27ae60', alpha=0.3, label='Gain')
        ax.fill_between(self.portfolio_value.index, self.investment,
                        self.portfolio_value.values,
                        where=self.portfolio_value.values < self.investment,
                        color='#e74c3c', alpha=0.3, label='Loss')

        ax.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_var_with_ci(
        self,
        method: str = 'historical',
        confidence_level: float = 0.95,
        save_path: Optional[str] = None
    ) -> None:
        """Plot return distribution with VaR threshold."""
        # Get VaR value
        if method == 'historical':
            var_dollars, details = self.historical_var(confidence_level)
        elif method == 'parametric':
            var_dollars, details = self.parametric_var(confidence_level)
        else:
            var_dollars, details = self.monte_carlo_var(confidence_level)

        var_return = details['var_return']

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot histogram
        n, bins, patches = ax.hist(self.portfolio_returns, bins=50,
                                    density=True, alpha=0.7, color='#3498db',
                                    edgecolor='white')

        # Color bins below VaR threshold
        for i, (patch, left_edge) in enumerate(zip(patches, bins[:-1])):
            if left_edge < var_return:
                patch.set_facecolor('#e74c3c')

        # Add VaR line
        ax.axvline(x=var_return, color='#e74c3c', linestyle='--', linewidth=2,
                   label=f'VaR ({confidence_level*100:.0f}%): ${var_dollars:,.0f}')

        # Add normal distribution overlay for parametric
        if method == 'parametric':
            x = np.linspace(self.portfolio_returns.min(),
                           self.portfolio_returns.max(), 100)
            mean = self.portfolio_returns.mean()
            std = self.portfolio_returns.std()
            ax.plot(x, stats.norm.pdf(x, mean, std), 'k-', linewidth=2,
                   label='Normal Distribution')

        ax.set_title(f'{method.title()} VaR with {confidence_level*100:.0f}% Confidence',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Daily Returns')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_var_over_time(
        self,
        method: str = 'historical',
        confidence_level: float = 0.95,
        window: int = 252,
        save_path: Optional[str] = None
    ) -> None:
        """Plot rolling VaR over time."""
        rolling_var = []
        dates = []

        returns_array = self.portfolio_returns.values

        for i in range(window, len(returns_array)):
            window_returns = returns_array[i-window:i]

            if method == 'historical':
                var_return = np.percentile(window_returns, (1 - confidence_level) * 100)
            elif method == 'parametric':
                mean = np.mean(window_returns)
                std = np.std(window_returns)
                z = stats.norm.ppf(1 - confidence_level)
                var_return = mean + z * std
            else:
                mean = np.mean(window_returns)
                std = np.std(window_returns)
                simulated = np.random.normal(mean, std, 1000)
                var_return = np.percentile(simulated, (1 - confidence_level) * 100)

            var_dollars = abs(var_return * self.investment)
            rolling_var.append(var_dollars)
            dates.append(self.portfolio_returns.index[i])

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(dates, rolling_var, color='#e74c3c', linewidth=2)
        ax.fill_between(dates, 0, rolling_var, color='#e74c3c', alpha=0.3)

        ax.set_title(f'{method.title()} VaR Over Time ({confidence_level*100:.0f}% Confidence)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('VaR ($)')
        ax.grid(True, alpha=0.3)

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_correlation_matrix(self, save_path: Optional[str] = None) -> None:
        """Plot correlation matrix of asset returns."""
        corr_matrix = self.returns.corr()

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                    center=0, vmin=-1, vmax=1, square=True,
                    linewidths=0.5, ax=ax)

        ax.set_title('Asset Correlation Matrix', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def export_to_csv(self, filepath: str) -> None:
        """Export analysis results to CSV."""
        # Calculate all VaR methods
        hist_var, hist_details = self.historical_var()
        param_var, param_details = self.parametric_var()
        mc_var, mc_details = self.monte_carlo_var()

        # Summary statistics
        total_return = (self.portfolio_value.iloc[-1] / self.investment - 1) * 100
        annual_return = self.portfolio_returns.mean() * 252 * 100
        annual_vol = self.portfolio_returns.std() * np.sqrt(252) * 100
        sharpe = (annual_return - 2) / annual_vol  # Assuming 2% risk-free rate

        results = {
            'Metric': [
                'Initial Investment',
                'Final Value',
                'Total Return (%)',
                'Annual Return (%)',
                'Annual Volatility (%)',
                'Sharpe Ratio',
                'Historical VaR (95%)',
                'Parametric VaR (95%)',
                'Monte Carlo VaR (95%)',
                'Historical CVaR (95%)',
                'Parametric CVaR (95%)',
                'Monte Carlo CVaR (95%)'
            ],
            'Value': [
                f'${self.investment:,.0f}',
                f'${self.portfolio_value.iloc[-1]:,.0f}',
                f'{total_return:.2f}%',
                f'{annual_return:.2f}%',
                f'{annual_vol:.2f}%',
                f'{sharpe:.2f}',
                f'${hist_var:,.0f}',
                f'${param_var:,.0f}',
                f'${mc_var:,.0f}',
                f'${hist_details["cvar_dollars"]:,.0f}',
                f'${param_details["cvar_dollars"]:,.0f}',
                f'${mc_details["cvar_dollars"]:,.0f}'
            ]
        }

        df = pd.DataFrame(results)
        df.to_csv(filepath, index=False)
        print(f"\nResults exported to {filepath}")

    def generate_report(self) -> None:
        """Print a formatted summary report."""
        print("\n" + "=" * 70)
        print("VALUE AT RISK ANALYSIS REPORT")
        print("=" * 70)

        # Portfolio composition
        print("\n1. PORTFOLIO COMPOSITION")
        print("-" * 40)
        for asset, weight in zip(self.assets, self.weights):
            allocation = weight * self.investment
            print(f"   {asset:8} {weight*100:6.1f}%  ${allocation:>12,.0f}")
        print(f"   {'TOTAL':8} {'100.0':>6}%  ${self.investment:>12,.0f}")

        # Performance metrics
        print("\n2. PERFORMANCE METRICS")
        print("-" * 40)
        total_return = (self.portfolio_value.iloc[-1] / self.investment - 1) * 100
        annual_return = self.portfolio_returns.mean() * 252 * 100
        annual_vol = self.portfolio_returns.std() * np.sqrt(252) * 100
        sharpe = (annual_return - 2) / annual_vol

        print(f"   Period: {self.start_date} to {self.end_date}")
        print(f"   Final Value:      ${self.portfolio_value.iloc[-1]:>12,.0f}")
        print(f"   Total Return:     {total_return:>12.2f}%")
        print(f"   Annual Return:    {annual_return:>12.2f}%")
        print(f"   Annual Volatility:{annual_vol:>12.2f}%")
        print(f"   Sharpe Ratio:     {sharpe:>12.2f}")

        # VaR results
        print("\n3. VALUE AT RISK (95% Confidence)")
        print("-" * 40)
        hist_var, _ = self.historical_var()
        param_var, _ = self.parametric_var()
        mc_var, _ = self.monte_carlo_var()

        print(f"   Historical VaR:   ${hist_var:>12,.0f}")
        print(f"   Parametric VaR:   ${param_var:>12,.0f}")
        print(f"   Monte Carlo VaR:  ${mc_var:>12,.0f}")

        # Stress test
        print("\n4. STRESS TEST SCENARIOS")
        print("-" * 40)
        for scenario, impact in self.STRESS_SCENARIOS.items():
            loss = self.investment * abs(impact)
            print(f"   {scenario:25} {impact*100:>6.0f}%  -${loss:>10,.0f}")

        print("\n" + "=" * 70)

    def run_comprehensive_analysis(self, show_plots: bool = True) -> None:
        """Run full VaR analysis with all components."""
        # Print report
        self.generate_report()

        # Backtest all methods
        print("\n5. BACKTESTING RESULTS")
        print("-" * 40)
        for method in ['historical', 'parametric', 'monte_carlo']:
            self.backtest(method=method)

        # Show plots
        if show_plots:
            print("\n6. GENERATING VISUALIZATIONS")
            print("-" * 40)
            self.plot_portfolio_performance()
            self.plot_var_with_ci(method='historical')
            self.plot_var_over_time(method='historical')
            self.plot_correlation_matrix()


if __name__ == "__main__":
    # Demo with the example portfolio from the documentation
    print("\n" + "=" * 70)
    print("VaR ANALYSIS TOOL - DEMONSTRATION")
    print("=" * 70)

    # Initialize with example portfolio
    var = VaRAnalysis(
        assets=['AAPL', 'MSFT', 'JPM', 'GS', 'XLF', 'TLT', 'GLD'],
        weights=[0.15, 0.15, 0.15, 0.15, 0.2, 0.1, 0.1],
        investment=100000
    )

    # Run comprehensive analysis
    var.run_comprehensive_analysis()

    # Show stress test table
    print("\nSTRESS TEST DETAILS:")
    print(var.stress_test().to_string(index=False))
