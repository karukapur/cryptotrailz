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
from urllib.parse import urljoin, urlsplit, urlunsplit

import numpy as np
import pandas as pd
import requests
from requests import RequestException

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "yfinance is required for benchmark data. Install with `pip install yfinance`."
    ) from exc

import matplotlib.pyplot as plt

COINGECKO_API = "https://api.coingecko.com/api/v3"
COINCAP_API = "https://api.coincap.io/v2"
CRYPTOCOMPARE_API = "https://min-api.cryptocompare.com/data/v2"
# Treat cached data with fewer than 80% of expected rows as invalid.
EXPECTED_COMPLETENESS_RATIO = 0.8
MIN_REQUIRED_ASSETS = 80


def log(message: str) -> None:
    print(message, flush=True)


def render_progress(current: int, total: int, prefix: str = "") -> None:
    width = 30
    ratio = min(max(current / total, 0), 1)
    filled = int(width * ratio)
    bar = "█" * filled + "-" * (width - filled)
    percent = ratio * 100
    log(f"{prefix}[{bar}] {current}/{total} ({percent:5.1f}%)")


def build_coingecko_url(path: str) -> str:
    parts = urlsplit(COINGECKO_API)
    if parts.query or parts.fragment:
        raise SystemExit(
            "COINGECKO_API must be a base URL without query parameters or fragments. "
            "Ensure it is set to https://api.coingecko.com/api/v3."
        )
    base = urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/") + "/", "", ""))
    return urljoin(base, path.lstrip("/"))


def build_coincap_url(path: str) -> str:
    base = COINCAP_API.rstrip("/") + "/"
    return urljoin(base, path.lstrip("/"))


def build_cryptocompare_url(path: str) -> str:
    base = CRYPTOCOMPARE_API.rstrip("/") + "/"
    return urljoin(base, path.lstrip("/"))


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
        return pd.read_csv(path, index_col=0, parse_dates=[0])
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
            if status == 401:
                raise SystemExit(
                    "CoinGecko API request returned 401 Unauthorized. "
                    "Verify COINGECKO_DEMO_API_KEY is set and valid."
                ) from exc
            if status in {429, 500, 502, 503, 504} and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise


def _request_json_with_retry_cryptocompare(
    url: str, params: dict | None = None, headers: dict | None = None
) -> dict:
    for attempt in range(3):
        try:
            return _request_json(url, params=params, headers=headers)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response else None
            if status == 401:
                raise SystemExit(
                    "CryptoCompare API request returned 401 Unauthorized. "
                    "Verify CRYPTOCOMPARE_API_KEY is set and valid."
                ) from exc
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
    cache_path = cache_dir / f"coingecko_top_coins_{limit}_{cache_tag}.csv"
    if not refresh:
        cached = read_csv_if_exists(cache_path)
        if cached is not None and len(cached) >= int(limit * EXPECTED_COMPLETENESS_RATIO):
            log(f"CACHE HIT: {cache_path}")
            return cached
        if cached is not None:
            log(f"CACHE INVALID: {cache_path} (refetching)")

    url = build_coingecko_url("/coins/markets")
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
    if "id" not in df.columns:
        raise SystemExit(
            "Unexpected CoinGecko response: missing 'id' column. "
            "Check COINGECKO_API and API key configuration."
        )
    atomic_to_csv(df, cache_path)
    return df


def _validate_cache_length(df: pd.DataFrame, expected_rows: int) -> bool:
    return len(df) >= int(expected_rows * EXPECTED_COMPLETENESS_RATIO)


def _validate_series(series: pd.Series, expected_rows: int) -> bool:
    if len(series) < int(expected_rows * EXPECTED_COMPLETENESS_RATIO):
        return False
    if series.isna().mean() > 0.2:
        return False
    return True


def fetch_coin_history(
    coin_id: str,
    days: int,
    cache_dir: Path | None = None,
    refresh: bool = False,
    cache_tag: str = "v1",
) -> pd.Series:
    cache_dir = cache_dir or Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"coingecko_prices_{coin_id}_{days}_{cache_tag}.csv"
    if not refresh:
        cached = read_csv_if_exists(cache_path)
        if cached is not None:
            series = cached.iloc[:, 0]
            series.name = coin_id
            if _validate_series(series, days):
                log(f"CACHE HIT: {cache_path}")
                return series
            log(f"CACHE INVALID: {cache_path} (refetching)")

    url = build_coingecko_url(f"/coins/{coin_id}/market_chart")
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


def fetch_cryptocompare_history(
    symbol: str,
    days_needed: int,
    cache_dir: Path | None = None,
    refresh: bool = False,
    cache_tag: str = "v1",
    end_ts: int | None = None,
) -> pd.Series:
    if days_needed <= 0:
        raise ValueError("days_needed must be a positive integer.")
    cache_dir = cache_dir or Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"cryptocompare_prices_{symbol}_{days_needed}_{cache_tag}.csv"
    if not refresh:
        cached = read_csv_if_exists(cache_path)
        if cached is not None:
            series = cached.iloc[:, 0]
            series.name = symbol
            if _validate_series(series, days_needed):
                log(f"CACHE HIT: {cache_path}")
                return series
            log(f"CACHE INVALID: {cache_path} (refetching)")

    url = build_cryptocompare_url("/histoday")
    params = {
        "fsym": symbol.upper(),
        "tsym": "USD",
        "limit": max(days_needed - 1, 0),
    }
    if end_ts is None:
        end_ts = int(pd.Timestamp.utcnow().normalize().timestamp())
    params["toTs"] = int(end_ts)
    headers = {}
    api_key = os.getenv("CRYPTOCOMPARE_API_KEY")
    if api_key:
        headers["authorization"] = f"Apikey {api_key}"
    log(f"FETCH: CryptoCompare {url} -> {cache_path}")
    data = _request_json_with_retry_cryptocompare(url, params=params, headers=headers)
    history = pd.DataFrame(data.get("Data", {}).get("Data", []))
    if history.empty:
        raise SystemExit(f"CryptoCompare returned no history for {symbol}.")
    history["date"] = pd.to_datetime(history["time"], unit="s").dt.date
    history["close"] = pd.to_numeric(history["close"], errors="coerce")
    series = history.groupby("date")["close"].last()
    series.index = pd.to_datetime(series.index)
    series.name = symbol
    atomic_to_csv(series.to_frame(), cache_path)
    return series


def fetch_hybrid_history(
    coin_id: str,
    symbol: str,
    days: int,
    cache_dir: Path,
    refresh: bool,
    cache_tag: str,
) -> pd.Series:
    if days <= 365:
        return fetch_coin_history(
            coin_id, days, cache_dir=cache_dir, refresh=refresh, cache_tag=cache_tag
        )
    cache_path = cache_dir / f"stitched_prices_{coin_id}_{days}_{cache_tag}.csv"
    if not refresh:
        cached = read_csv_if_exists(cache_path)
        if cached is not None:
            series = cached.iloc[:, 0]
            series.name = coin_id
            if _validate_series(series, days):
                log(f"CACHE HIT: {cache_path}")
                return series
            log(f"CACHE INVALID: {cache_path} (refetching)")

    recent_series = fetch_coin_history(
        coin_id, 365, cache_dir=cache_dir, refresh=refresh, cache_tag=cache_tag
    )
    cutoff_date = recent_series.index.min()
    older_days = days - 365
    end_ts = int((cutoff_date - pd.Timedelta(days=1)).timestamp())
    older_series = fetch_cryptocompare_history(
        symbol,
        older_days,
        cache_dir=cache_dir,
        refresh=refresh,
        cache_tag=cache_tag,
        end_ts=end_ts,
    )
    older_series.name = coin_id
    combined = pd.concat([older_series, recent_series], axis=0)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.name = coin_id
    if not _validate_series(combined, days):
        raise ValueError(f"Hybrid series for {coin_id} is incomplete.")
    atomic_to_csv(combined.to_frame(), cache_path)
    return combined


def fetch_coincap_top_assets(
    limit: int,
    cache_dir: Path,
    refresh: bool,
    cache_tag: str,
) -> pd.DataFrame:
    cache_path = cache_dir / f"coincap_top_assets_{limit}_{cache_tag}.csv"
    if not refresh:
        cached = read_csv_if_exists(cache_path)
        if cached is not None and len(cached) >= int(limit * EXPECTED_COMPLETENESS_RATIO):
            log(f"CACHE HIT: {cache_path}")
            return cached
        if cached is not None:
            log(f"CACHE INVALID: {cache_path} (refetching)")

    url = build_coincap_url("/assets")
    params = {"limit": limit}
    log(f"FETCH: CoinCap {url} -> {cache_path}")
    try:
        data = _request_json_with_retry(url, params=params)
    except RequestException as exc:
        raise SystemExit(
            "CoinCap request failed. Check network connectivity or try again later."
        ) from exc
    assets = pd.DataFrame(data.get("data", []))
    if assets.empty:
        raise SystemExit("CoinCap returned no asset data.")
    if "rank" in assets.columns:
        assets["rank"] = pd.to_numeric(assets["rank"], errors="coerce")
        assets = assets.sort_values("rank")
    elif "marketCapUsd" in assets.columns:
        assets["marketCapUsd"] = pd.to_numeric(assets["marketCapUsd"], errors="coerce")
        assets = assets.sort_values("marketCapUsd", ascending=False)
    assets = assets.head(limit)
    atomic_to_csv(assets, cache_path)
    return assets


def fetch_coincap_history(
    asset_id: str,
    days: int,
    cache_dir: Path,
    refresh: bool,
    cache_tag: str,
) -> pd.Series:
    cache_path = cache_dir / f"coincap_prices_{asset_id}_{days}_{cache_tag}.csv"
    if not refresh:
        cached = read_csv_if_exists(cache_path)
        if cached is not None:
            series = cached.iloc[:, 0]
            series.name = asset_id
            if _validate_series(series, days):
                log(f"CACHE HIT: {cache_path}")
                return series
            log(f"CACHE INVALID: {cache_path} (refetching)")

    url = build_coincap_url(f"/assets/{asset_id}/history")
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=days)
    params = {
        "interval": "d1",
        "start": int(start.timestamp() * 1000),
        "end": int(end.timestamp() * 1000),
    }
    log(f"FETCH: CoinCap {url} -> {cache_path}")
    try:
        data = _request_json_with_retry(url, params=params)
    except RequestException as exc:
        raise SystemExit(
            "CoinCap request failed. Check network connectivity or try again later."
        ) from exc
    history = pd.DataFrame(data.get("data", []))
    if history.empty:
        raise SystemExit(f"CoinCap returned no history for {asset_id}.")
    history["date"] = pd.to_datetime(history["date"]).dt.date
    history["priceUsd"] = pd.to_numeric(history["priceUsd"], errors="coerce")
    series = history.groupby("date")["priceUsd"].last()
    series.index = pd.to_datetime(series.index)
    series.name = asset_id
    atomic_to_csv(series.to_frame(), cache_path)
    return series


def fetch_coingecko_prices(
    coin_ids: list[str],
    days: int,
    cache_dir: Path,
    refresh: bool,
    cache_tag: str,
) -> pd.DataFrame:
    series_list = []
    total = len(coin_ids)
    log(f"Starting CoinGecko price fetch for {total} coins.")
    for idx, coin_id in enumerate(coin_ids, start=1):
        render_progress(idx, total, prefix="Coins ")
        series = fetch_coin_history(
            coin_id, days, cache_dir=cache_dir, refresh=refresh, cache_tag=cache_tag
        )
        series_list.append(series)
    prices = pd.concat(series_list, axis=1).sort_index()
    return prices.ffill().dropna()


def fetch_cryptocompare_prices(
    assets: list[dict[str, str]],
    days: int,
    cache_dir: Path,
    refresh: bool,
    cache_tag: str,
    provider_label: str,
) -> pd.DataFrame:
    series_list: list[pd.Series] = []
    total = len(assets)
    log(f"Starting {provider_label} price fetch for {total} assets.")
    for idx, asset in enumerate(assets, start=1):
        render_progress(idx, total, prefix="Coins ")
        try:
            series = fetch_cryptocompare_history(
                asset["symbol"],
                days,
                cache_dir=cache_dir,
                refresh=refresh,
                cache_tag=cache_tag,
            )
            series.name = asset["id"]
            series_list.append(series)
        except (RequestException, SystemExit, ValueError) as exc:
            log(f"PROVIDER FAIL: CryptoCompare {asset['symbol']} {exc}")
            continue
    if len(series_list) < MIN_REQUIRED_ASSETS:
        raise SystemExit(
            "CryptoCompare returned too few assets to build a portfolio. "
            "Set CRYPTOCOMPARE_API_KEY or reduce the universe size."
        )
    prices = pd.concat(series_list, axis=1).sort_index()
    return prices.ffill().dropna()


def fetch_hybrid_prices(
    assets: list[dict[str, str]],
    days: int,
    cache_dir: Path,
    refresh: bool,
    cache_tag: str,
) -> pd.DataFrame:
    series_list: list[pd.Series] = []
    total = len(assets)
    log(f"Starting hybrid price fetch for {total} assets.")
    for idx, asset in enumerate(assets, start=1):
        render_progress(idx, total, prefix="Coins ")
        try:
            series = fetch_hybrid_history(
                asset["id"],
                asset["symbol"],
                days,
                cache_dir=cache_dir,
                refresh=refresh,
                cache_tag=cache_tag,
            )
            series_list.append(series)
        except (RequestException, SystemExit, ValueError) as exc:
            log(f"PROVIDER FAIL: Hybrid {asset['symbol']} {exc}")
            continue
    if len(series_list) < MIN_REQUIRED_ASSETS:
        raise SystemExit(
            "Hybrid history returned too few assets to build a portfolio. "
            "Set CRYPTOCOMPARE_API_KEY or reduce the universe size."
        )
    prices = pd.concat(series_list, axis=1).sort_index()
    return prices.ffill().dropna()


def fetch_crypto_prices(
    days: int,
    cache_dir: Path,
    refresh: bool,
    cache_tag: str,
    limit: int,
    provider: str,
) -> pd.DataFrame:
    top_coins = fetch_top_coins(
        limit, cache_dir=cache_dir, refresh=refresh, cache_tag=cache_tag
    )
    if "id" not in top_coins.columns or "symbol" not in top_coins.columns:
        raise SystemExit("CoinGecko top coins data is missing required fields.")
    assets = [
        {"id": coin_id, "symbol": symbol}
        for coin_id, symbol in zip(top_coins["id"], top_coins["symbol"], strict=False)
        if isinstance(symbol, str) and symbol
    ]
    if provider == "coingecko":
        if days > 365:
            raise SystemExit(
                "CoinGecko demo/public plans only support 365 days of history. "
                "Use --provider hybrid or reduce --days."
            )
        coin_ids = [asset["id"] for asset in assets]
        return fetch_coingecko_prices(
            coin_ids, days, cache_dir=cache_dir, refresh=refresh, cache_tag=cache_tag
        )
    if provider == "cryptocompare":
        return fetch_cryptocompare_prices(
            assets,
            days,
            cache_dir=cache_dir,
            refresh=refresh,
            cache_tag=cache_tag,
            provider_label="CryptoCompare",
        )
    if provider == "hybrid":
        if days <= 365:
            coin_ids = [asset["id"] for asset in assets]
            return fetch_coingecko_prices(
                coin_ids, days, cache_dir=cache_dir, refresh=refresh, cache_tag=cache_tag
            )
        return fetch_hybrid_prices(
            assets, days, cache_dir=cache_dir, refresh=refresh, cache_tag=cache_tag
        )
    raise ValueError(f"Unknown provider: {provider}")


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
    parser.add_argument(
        "--provider",
        choices=["coingecko", "cryptocompare", "hybrid"],
        default="hybrid",
    )
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
    render_progress(2, 5, prefix="Setup ")
    prices = fetch_crypto_prices(
        config.days,
        cache_dir=cache_dir,
        refresh=args.refresh_cache,
        cache_tag=args.cache_tag,
        limit=100,
        provider=args.provider,
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
