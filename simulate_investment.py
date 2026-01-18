"""Simulate monthly crypto investment strategies with benchmarks.

This script pulls historical daily price data for the top 100 cryptocurrencies
by market cap from CoinGecko, simulates periodic investments with rebalancing,
and compares portfolio performance to Bitcoin, Nifty 50, and NASDAQ indices.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "yfinance is required for benchmark data. Install with `pip install yfinance`."
    ) from exc

import matplotlib.pyplot as plt

COINGECKO_API = "https://api.coingecko.com/api/v3"
CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)


@dataclass
class SimulationConfig:
    days: int = 444
    rebalance: str = "monthly"
    stop_loss: float = 0.20
    take_profit: float = 0.50
    weighting: str = "equal"
    amounts: tuple[int, ...] = (1000, 2000, 3000, 4000, 5000)


def _request_json(url: str, params: dict | None = None) -> dict:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_top_coins(limit: int = 100) -> pd.DataFrame:
    url = f"{COINGECKO_API}/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": limit,
        "page": 1,
        "sparkline": "false",
    }
    data = _request_json(url, params=params)
    return pd.DataFrame(data)


def fetch_coin_history(coin_id: str, days: int) -> pd.Series:
    cache_path = CACHE_DIR / f"{coin_id}_{days}.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        series = cached.iloc[:, 0]
        series.name = coin_id
        return series

    url = f"{COINGECKO_API}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    data = _request_json(url, params=params)
    prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    prices["date"] = pd.to_datetime(prices["timestamp"], unit="ms").dt.date
    series = prices.groupby("date")["price"].last()
    series.index = pd.to_datetime(series.index)
    series.name = coin_id
    series.to_csv(cache_path)
    time.sleep(1.2)
    return series


def fetch_crypto_prices(coin_ids: list[str], days: int) -> pd.DataFrame:
    series_list = [fetch_coin_history(coin_id, days) for coin_id in coin_ids]
    prices = pd.concat(series_list, axis=1).sort_index()
    return prices.ffill().dropna()


def fetch_benchmark_prices(days: int) -> pd.DataFrame:
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=days)
    tickers = {"Bitcoin": "BTC-USD", "Nifty50": "^NSEI", "NASDAQ": "^IXIC"}
    data = yf.download(
        list(tickers.values()),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
    )["Adj Close"]
    data = data.rename(columns={v: k for k, v in tickers.items()})
    return data.ffill().dropna()


def _get_rebalance_dates(index: pd.DatetimeIndex, interval: str) -> pd.DatetimeIndex:
    if interval == "weekly":
        return index[::7]
    if interval == "quarterly":
        return index[::90]
    return index[index.is_month_end]


def compute_weights(
    prices: pd.DataFrame, weighting: str, window: int = 30
) -> pd.Series:
    if weighting == "equal":
        return pd.Series(1.0, index=prices.columns) / prices.shape[1]

    if weighting == "volatility":
        returns = prices.pct_change().tail(window)
        volatility = returns.std().replace(0, np.nan)
        inv_vol = 1 / volatility
        weights = inv_vol / inv_vol.sum()
        return weights.fillna(0)

    if weighting == "momentum":
        perf = prices.pct_change(window).iloc[-1].clip(lower=0)
        if perf.sum() == 0:
            return pd.Series(1.0, index=prices.columns) / prices.shape[1]
        return perf / perf.sum()

    raise ValueError(f"Unknown weighting strategy: {weighting}")


def apply_stop_take_profit(
    returns: pd.Series, stop_loss: float, take_profit: float
) -> pd.Series:
    capped = returns.copy()
    capped = capped.clip(lower=-stop_loss, upper=take_profit)
    return capped


def simulate_portfolio(
    prices: pd.DataFrame,
    amount: int,
    config: SimulationConfig,
) -> pd.Series:
    rebalance_dates = _get_rebalance_dates(prices.index, config.rebalance)
    if rebalance_dates.empty:
        raise ValueError("No rebalance dates found for the given price history.")

    portfolio_value = pd.Series(index=prices.index, dtype=float)
    current_value = 0.0
    last_rebalance = rebalance_dates[0]

    for date in rebalance_dates:
        current_value += amount
        start_prices = prices.loc[last_rebalance]
        end_prices = prices.loc[date]
        returns = (end_prices / start_prices) - 1
        returns = apply_stop_take_profit(returns, config.stop_loss, config.take_profit)
        weights = compute_weights(prices.loc[:date], config.weighting)
        period_return = (weights * returns).sum()
        current_value *= 1 + period_return
        portfolio_value.loc[date] = current_value
        last_rebalance = date

    portfolio_value = portfolio_value.ffill()
    portfolio_value.attrs["contributions"] = len(rebalance_dates)
    return portfolio_value


def summarize_results(
    results: dict[str, dict[int, pd.Series]], amounts: tuple[int, ...]
) -> pd.DataFrame:
    rows = []
    for strategy, amount_map in results.items():
        for amount in amounts:
            series = amount_map[amount]
            contributions = series.attrs.get("contributions", len(series.dropna()))
            total_invested = amount * contributions
            final_value = series.dropna().iloc[-1]
            profit = final_value - total_invested
            rows.append(
                {
                    "Strategy": strategy,
                    "Amount": amount,
                    "Total Invested": total_invested,
                    "Final Value": final_value,
                    "Profit": profit,
                }
            )
    return pd.DataFrame(rows)


def plot_results(
    results: dict[str, dict[int, pd.Series]],
    benchmarks: pd.DataFrame,
    amounts: tuple[int, ...],
) -> None:
    for strategy, amount_map in results.items():
        plt.figure(figsize=(12, 6))
        for amount in amounts:
            series = amount_map[amount]
            plt.plot(series.index, series.values, label=f"{strategy} - {amount} NTD")

        normalized = benchmarks / benchmarks.iloc[0]
        for column in normalized.columns:
            plt.plot(
                normalized.index,
                normalized[column] * amounts[0],
                linestyle="--",
                label=f"{column} (normalized)",
            )

        plt.title(f"Portfolio Value - {strategy} weighting")
        plt.xlabel("Date")
        plt.ylabel("Value (NTD)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"portfolio_{strategy}.png")
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate crypto investment strategies.")
    parser.add_argument("--days", type=int, default=444)
    parser.add_argument(
        "--rebalance",
        choices=["monthly", "weekly", "quarterly"],
        default="monthly",
    )
    parser.add_argument(
        "--weighting",
        choices=["equal", "volatility", "momentum", "all"],
        default="all",
    )
    parser.add_argument("--stop-loss", type=float, default=0.20)
    parser.add_argument("--take-profit", type=float, default=0.50)
    parser.add_argument(
        "--amounts",
        type=int,
        nargs="+",
        default=[1000, 2000, 3000, 4000, 5000],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(
        days=args.days,
        rebalance=args.rebalance,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        weighting=args.weighting if args.weighting != "all" else "equal",
        amounts=tuple(args.amounts),
    )

    top_coins = fetch_top_coins(100)
    coin_ids = top_coins["id"].tolist()
    prices = fetch_crypto_prices(coin_ids, config.days)
    benchmarks = fetch_benchmark_prices(config.days)
    benchmarks = benchmarks.reindex(prices.index).ffill().dropna()

    weightings = (
        ["equal", "volatility", "momentum"] if args.weighting == "all" else [args.weighting]
    )

    results: dict[str, dict[int, pd.Series]] = {}
    for weighting in weightings:
        config.weighting = weighting
        results[weighting] = {}
        for amount in config.amounts:
            results[weighting][amount] = simulate_portfolio(prices, amount, config)

    summary = summarize_results(results, config.amounts)
    summary.to_csv("performance_summary.csv", index=False)
    print(summary)
    plot_results(results, benchmarks, config.amounts)


if __name__ == "__main__":
    main()
