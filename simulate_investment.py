"""Simulate monthly crypto investment strategies with benchmarks.

This script pulls historical daily price data for the top 100 cryptocurrencies
by market cap from CoinGecko, simulates periodic investments with rebalancing,
and compares portfolio performance to Bitcoin, Nifty 50, and NASDAQ indices.
"""

from __future__ import annotations

import argparse
import os
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
# Treat cached data with fewer than 80% of expected rows as invalid.
EXPECTED_COMPLETENESS_RATIO = 0.8


def log(message: str) -> None:
    print(message, flush=True)


def render_progress(current: int, total: int, prefix: str = "") -> None:
    width = 30
    ratio = min(max(current / total, 0), 1)
    filled = int(width * ratio)
    bar = "█" * filled + "-" * (width - filled)
    percent = ratio * 100
    log(f"{prefix}[{bar}] {current}/{total} ({percent:5.1f}%)")


@dataclass
class SimulationConfig:
    days: int = 444
    rebalance: str = "monthly"
    stop_loss: float = 0.20
    take_profit: float = 0.50
    weighting: str = "equal"
    amounts: tuple[int, ...] = (1000, 2000, 3000, 4000, 5000)


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    return None


def atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path)
    tmp_path.replace(path)


def _request_json(
    url: str, params: dict | None = None, headers: dict | None = None
) -> dict:
    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def _request_json_with_retry(
    url: str, params: dict | None = None, headers: dict | None = None
) -> dict:
    for attempt in range(3):
        try:
            return _request_json(url, params=params, headers=headers)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response else None
            if status in {429, 500, 502, 503, 504} and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise


def fetch_top_coins(
    limit: int = 100,
    cache_dir: Path | None = None,
    refresh: bool = False,
    cache_tag: str = "v1",
) -> pd.DataFrame:
    cache_dir = cache_dir or Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"top_coins_{limit}_{cache_tag}.csv"
    if not refresh:
        cached = read_csv_if_exists(cache_path)
        if cached is not None and len(cached) >= int(limit * EXPECTED_COMPLETENESS_RATIO):
            log(f"CACHE HIT: {cache_path}")
            return cached
        if cached is not None:
            log(f"CACHE INVALID: {cache_path} (refetching)")

    url = f"{COINGECKO_API}/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": limit,
        "page": 1,
        "sparkline": "false",
    }
    api_key = os.getenv("COINGECKO_DEMO_API_KEY")
    if not api_key:
        raise SystemExit(
            "COINGECKO_DEMO_API_KEY is required for CoinGecko fetches. "
            "Set it in your environment before running with cache refresh."
        )
    headers = {"x-cg-demo-api-key": api_key}
    log(f"FETCH: {url} -> {cache_path}")
    data = _request_json_with_retry(url, params=params, headers=headers)
    df = pd.DataFrame(data)
    atomic_to_csv(df, cache_path)
    return df


def _validate_cache_length(df: pd.DataFrame, expected_rows: int) -> bool:
    return len(df) >= int(expected_rows * EXPECTED_COMPLETENESS_RATIO)


def fetch_coin_history(
    coin_id: str,
    days: int,
    cache_dir: Path | None = None,
    refresh: bool = False,
    cache_tag: str = "v1",
) -> pd.Series:
    cache_dir = cache_dir or Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"prices_{coin_id}_{days}_{cache_tag}.csv"
    if not refresh:
        cached = read_csv_if_exists(cache_path)
        if cached is not None and _validate_cache_length(cached, days):
            log(f"CACHE HIT: {cache_path}")
            series = cached.iloc[:, 0]
            series.name = coin_id
            return series
        if cached is not None:
            log(f"CACHE INVALID: {cache_path} (refetching)")

    url = f"{COINGECKO_API}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    api_key = os.getenv("COINGECKO_DEMO_API_KEY")
    if not api_key:
        raise SystemExit(
            "COINGECKO_DEMO_API_KEY is required for CoinGecko fetches. "
            "Set it in your environment before running with cache refresh."
        )
    headers = {"x-cg-demo-api-key": api_key}
    log(f"FETCH: {url} -> {cache_path}")
    data = _request_json_with_retry(url, params=params, headers=headers)
    prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    prices["date"] = pd.to_datetime(prices["timestamp"], unit="ms").dt.date
    series = prices.groupby("date")["price"].last()
    series.index = pd.to_datetime(series.index)
    series.name = coin_id
    atomic_to_csv(series.to_frame(), cache_path)
    time.sleep(1.2)
    return series


def fetch_crypto_prices(
    coin_ids: list[str],
    days: int,
    cache_dir: Path,
    refresh: bool,
    cache_tag: str,
) -> pd.DataFrame:
    series_list = []
    total = len(coin_ids)
    log(f"Starting crypto price fetch for {total} coins.")
    for idx, coin_id in enumerate(coin_ids, start=1):
        render_progress(idx, total, prefix="Coins ")
        series = fetch_coin_history(
            coin_id, days, cache_dir=cache_dir, refresh=refresh, cache_tag=cache_tag
        )
        series_list.append(series)
    prices = pd.concat(series_list, axis=1).sort_index()
    return prices.ffill().dropna()


def fetch_benchmark_prices(
    days: int,
    cache_dir: Path | None = None,
    refresh: bool = False,
    cache_tag: str = "v1",
) -> pd.DataFrame:
    cache_dir = cache_dir or Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"benchmarks_{days}_{cache_tag}.csv"
    if not refresh:
        cached = read_csv_if_exists(cache_path)
        if cached is not None and _validate_cache_length(cached, days):
            log(f"CACHE HIT: {cache_path}")
            return cached
        if cached is not None:
            log(f"CACHE INVALID: {cache_path} (refetching)")

    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=days)
    tickers = {"Bitcoin": "BTC-USD", "Nifty50": "^NSEI", "NASDAQ": "^IXIC"}
    log(f"FETCH: yfinance benchmarks -> {cache_path}")
    data = yf.download(
        list(tickers.values()),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
    )["Adj Close"]
    data = data.rename(columns={v: k for k, v in tickers.items()})
    data = data.ffill().dropna()
    atomic_to_csv(data, cache_path)
    return data


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
    parser.add_argument("--cache-dir", type=str, default="data_cache")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--cache-tag", type=str, default="v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log("Starting simulation.")
    render_progress(0, 5, prefix="Setup ")
    config = SimulationConfig(
        days=args.days,
        rebalance=args.rebalance,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        weighting=args.weighting if args.weighting != "all" else "equal",
        amounts=tuple(args.amounts),
    )

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(exist_ok=True)
    render_progress(1, 5, prefix="Setup ")
    top_coins = fetch_top_coins(
        100, cache_dir=cache_dir, refresh=args.refresh_cache, cache_tag=args.cache_tag
    )
    coin_ids = top_coins["id"].tolist()
    render_progress(2, 5, prefix="Setup ")
    prices = fetch_crypto_prices(
        coin_ids,
        config.days,
        cache_dir=cache_dir,
        refresh=args.refresh_cache,
        cache_tag=args.cache_tag,
    )
    render_progress(3, 5, prefix="Setup ")
    benchmarks = fetch_benchmark_prices(
        config.days,
        cache_dir=cache_dir,
        refresh=args.refresh_cache,
        cache_tag=args.cache_tag,
    )
    benchmarks = benchmarks.reindex(prices.index).ffill().dropna()
    render_progress(4, 5, prefix="Setup ")

    weightings = (
        ["equal", "volatility", "momentum"] if args.weighting == "all" else [args.weighting]
    )

    results: dict[str, dict[int, pd.Series]] = {}
    for weighting in weightings:
        log(f"Running simulations for weighting={weighting}.")
        config.weighting = weighting
        results[weighting] = {}
        for amount in config.amounts:
            log(f"Simulating amount {amount} NTD.")
            results[weighting][amount] = simulate_portfolio(prices, amount, config)

    summary = summarize_results(results, config.amounts)
    summary.to_csv("performance_summary.csv", index=False)
    log("Simulation summary:")
    log(summary.to_string(index=False))
    plot_results(results, benchmarks, config.amounts)
    render_progress(5, 5, prefix="Setup ")
    log("Finished. Outputs: performance_summary.csv and portfolio_*.png files.")


if __name__ == "__main__":
    main()
