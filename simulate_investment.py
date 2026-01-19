"""Simulate monthly crypto investment strategies with benchmarks.

This script builds a top-N crypto universe based on exchange liquidity
(quote volume) using CCXT, simulates periodic investments with rebalancing,
and compares portfolio performance to Bitcoin, Nifty 50, and NASDAQ indices.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "ccxt is required for crypto exchange data. Install with `pip install ccxt`."
    ) from exc

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "yfinance is required for benchmark data. Install with `pip install yfinance`."
    ) from exc

import matplotlib.pyplot as plt

# Treat cached data with fewer than 80% of expected rows as invalid.
EXPECTED_COMPLETENESS_RATIO = 0.8
MIN_REQUIRED_ASSETS = 80
MS_IN_DAY = 24 * 60 * 60 * 1000

LEVERAGED_TOKEN_HINTS = ("3L", "3S", "5L", "5S", "UP", "DOWN", "BULL", "BEAR")
STABLE_BASES = {
    "USDT",
    "USDC",
    "DAI",
    "TUSD",
    "BUSD",
    "FDUSD",
    "USDP",
    "PAX",
    "USDD",
    "GUSD",
    "USDJ",
    "FRAX",
    "LUSD",
    "UST",
    "USTC",
    "EURS",
}


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


@dataclass(frozen=True)
class UniverseSettings:
    exchange: str
    quote: str
    top_n: int
    days: int
    cache_tag: str
    exclude_stable_bases: bool
    exclude_leveraged: bool
    min_quote_volume: float


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=[0])
    return None


def atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path)
    tmp_path.replace(path)


def _validate_cache_length(df: pd.DataFrame, expected_rows: int) -> bool:
    return len(df) >= int(expected_rows * EXPECTED_COMPLETENESS_RATIO)


def _validate_series(series: pd.Series, expected_rows: int) -> bool:
    if len(series) < int(expected_rows * EXPECTED_COMPLETENESS_RATIO):
        return False
    if series.isna().mean() > 0.2:
        return False
    return True


def _sanitize_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_").replace(" ", "_")


def _is_leveraged_token(base: str) -> bool:
    upper = base.upper()
    if any(upper.endswith(suffix) for suffix in ("3L", "3S", "5L", "5S")):
        return True
    return any(hint in upper for hint in LEVERAGED_TOKEN_HINTS)


def _is_stable_base(base: str) -> bool:
    return base.upper() in STABLE_BASES


def _is_spot_market(market: dict[str, Any]) -> bool:
    if market.get("spot") is True:
        return True
    return market.get("type") == "spot"


def _extract_quote_volume(ticker: dict[str, Any]) -> float | None:
    quote_volume = ticker.get("quoteVolume")
    if quote_volume is None:
        base_volume = ticker.get("baseVolume")
        last = ticker.get("last")
        if base_volume is not None and last is not None:
            try:
                return float(base_volume) * float(last)
            except (TypeError, ValueError):
                return None
        return None
    try:
        return float(quote_volume)
    except (TypeError, ValueError):
        return None


def _ccxt_call_with_retry(func, *args, **kwargs):
    for attempt in range(3):
        try:
            return func(*args, **kwargs)
        except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
            if attempt >= 2:
                raise
            log(f"CCXT retry {attempt + 1}/3 after error: {exc}")
            time.sleep(2 ** attempt)


def build_ccxt_universe(
    exchange_id: str,
    quote: str,
    top_n: int,
    cache_dir: Path,
    refresh: bool,
    cache_tag: str,
    exclude_stable_bases: bool,
    exclude_leveraged: bool,
    min_quote_volume: float,
) -> pd.DataFrame:
    cache_dir.mkdir(exist_ok=True)
    cache_path = (
        cache_dir
        / f"ccxt_universe_{exchange_id}_{quote}_{top_n}_{cache_tag}.csv"
    )
    if not refresh:
        cached = read_csv_if_exists(cache_path)
        if cached is not None and len(cached) >= int(top_n * EXPECTED_COMPLETENESS_RATIO):
            log(f"CACHE HIT: {cache_path}")
            return cached
        if cached is not None:
            log(f"CACHE INVALID: {cache_path} (refetching)")

    try:
        exchange_class = getattr(ccxt, exchange_id)
    except AttributeError as exc:
        raise SystemExit(f"Unknown CCXT exchange '{exchange_id}'.") from exc

    exchange = exchange_class({"enableRateLimit": True})
    log(f"Loading markets for exchange={exchange_id}.")
    _ccxt_call_with_retry(exchange.load_markets)

    quote_upper = quote.upper()
    spot_symbols = [
        symbol
        for symbol, market in exchange.markets.items()
        if _is_spot_market(market) and market.get("quote", "").upper() == quote_upper
    ]
    candidate_count = len(spot_symbols)
    log(f"Universe build (liquidity-ranked, not market-cap): {candidate_count} candidates.")

    tickers: dict[str, Any] = {}
    if exchange.has.get("fetchTickers"):
        log("Fetching tickers via fetch_tickers().")
        tickers = _ccxt_call_with_retry(exchange.fetch_tickers)
    else:
        log("Fetching tickers via per-symbol fetch_ticker().")
        for symbol in spot_symbols:
            tickers[symbol] = _ccxt_call_with_retry(exchange.fetch_ticker, symbol)
            time.sleep(exchange.rateLimit / 1000)

    records: list[dict[str, Any]] = []
    after_filter = 0
    for symbol in spot_symbols:
        market = exchange.markets.get(symbol, {})
        base = market.get("base") or symbol.split("/")[0]
        quote_sym = market.get("quote") or quote_upper
        ticker = tickers.get(symbol) or {}
        quote_volume = _extract_quote_volume(ticker)
        if quote_volume is None:
            continue
        if min_quote_volume and quote_volume < min_quote_volume:
            continue
        if exclude_leveraged and _is_leveraged_token(base):
            continue
        if exclude_stable_bases and _is_stable_base(base):
            continue
        after_filter += 1
        records.append(
            {
                "symbol": symbol,
                "base": base,
                "quote": quote_sym,
                "quoteVolume": quote_volume,
                "last": ticker.get("last"),
                "timestamp": pd.Timestamp.utcnow().isoformat(),
            }
        )

    log(
        "Universe filters applied: "
        f"{candidate_count} -> {after_filter} -> top {top_n}."
    )

    if not records:
        raise SystemExit("No markets available after applying universe filters.")

    universe = (
        pd.DataFrame(records)
        .sort_values("quoteVolume", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    atomic_to_csv(universe, cache_path)
    return universe


def fetch_ccxt_ohlcv_series(
    exchange: ccxt.Exchange,
    symbol: str,
    days: int,
    cache_dir: Path,
    refresh: bool,
    cache_tag: str,
) -> tuple[pd.Series | None, bool]:
    cache_dir.mkdir(exist_ok=True)
    sanitized = _sanitize_symbol(symbol)
    cache_path = (
        cache_dir / f"ccxt_prices_{exchange.id}_{sanitized}_{days}_{cache_tag}.csv"
    )
    if not refresh:
        cached = read_csv_if_exists(cache_path)
        if cached is not None:
            series = cached.iloc[:, 0]
            series.name = symbol
            if _validate_series(series, days):
                log(f"CACHE HIT: {cache_path}")
                return series, True
            log(f"CACHE INVALID: {cache_path} (refetching)")

    log(f"FETCH: OHLCV {symbol} -> {cache_path}")
    since = int((pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=days)).timestamp() * 1000)
    all_rows: list[list[Any]] = []

    while True:
        batch = _ccxt_call_with_retry(
            exchange.fetch_ohlcv, symbol, timeframe="1d", since=since, limit=1000
        )
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= since:
            break
        since = last_ts + MS_IN_DAY
        time.sleep(exchange.rateLimit / 1000)

    if not all_rows:
        return None, False

    history = pd.DataFrame(
        all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    history["date"] = pd.to_datetime(history["timestamp"], unit="ms", utc=True).dt.normalize()
    history["close"] = pd.to_numeric(history["close"], errors="coerce")
    series = history.groupby("date")["close"].last().sort_index()
    series = series.tail(days)
    series.name = symbol
    if not _validate_series(series, days):
        return None, False
    atomic_to_csv(series.to_frame(), cache_path)
    return series, False


def build_ccxt_prices(
    universe: pd.DataFrame,
    days: int,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    exchange_id = universe.attrs.get("exchange_id")
    if not exchange_id:
        raise SystemExit("Universe is missing exchange metadata.")

    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    _ccxt_call_with_retry(exchange.load_markets)

    series_list: list[pd.Series] = []
    cache_hits = 0
    fetches = 0
    skipped = 0
    total = len(universe)

    log(f"Fetching OHLCV for {total} markets on {exchange_id}.")
    for idx, symbol in enumerate(universe["symbol"], start=1):
        render_progress(idx, total, prefix="Markets ")
        series, from_cache = fetch_ccxt_ohlcv_series(
            exchange,
            symbol,
            days=days,
            cache_dir=cache_dir,
            refresh=refresh,
            cache_tag=universe.attrs.get("cache_tag", "v1"),
        )
        if series is None:
            log(f"SKIP: {symbol} missing or insufficient OHLCV history.")
            skipped += 1
            continue
        if from_cache:
            cache_hits += 1
        else:
            fetches += 1
        series_list.append(series)

    log(f"Cache hits: {cache_hits}, Fetches: {fetches}, Skipped: {skipped}.")

    if len(series_list) < MIN_REQUIRED_ASSETS:
        raise SystemExit(
            "Too few markets with valid history to build a portfolio. "
            "Try lowering --top-n, switching exchange/quote, or reducing --days."
        )

    prices = pd.concat(series_list, axis=1).sort_index()
    prices = prices.ffill().dropna()
    return prices


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def _build_candidate_settings(args: argparse.Namespace) -> list[UniverseSettings]:
    exchanges = _unique_preserve_order(
        [args.exchange, "binance", "kraken", "coinbase", "bitstamp"]
    )
    quotes = _unique_preserve_order([args.quote, "USDT", "USD", "USDC"])
    top_ns = [args.top_n, max(args.top_n - 20, 60), max(args.top_n - 40, 40), 30]
    days_list = [args.days, max(int(args.days * 0.85), 365), max(int(args.days * 0.7), 250)]

    candidates: list[UniverseSettings] = []
    for exchange in exchanges:
        for quote in quotes:
            for days in days_list:
                for top_n in top_ns:
                    candidates.append(
                        UniverseSettings(
                            exchange=exchange,
                            quote=quote,
                            top_n=top_n,
                            days=days,
                            cache_tag=args.cache_tag,
                            exclude_stable_bases=args.exclude_stable_bases,
                            exclude_leveraged=args.exclude_leveraged,
                            min_quote_volume=args.min_quote_volume,
                        )
                    )
    return candidates


def _attempt_universe_build(
    settings: UniverseSettings,
    cache_dir: Path,
    refresh: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    log(
        "Attempting settings: "
        f"exchange={settings.exchange}, quote={settings.quote}, "
        f"top_n={settings.top_n}, days={settings.days}"
    )
    universe = build_ccxt_universe(
        exchange_id=settings.exchange,
        quote=settings.quote,
        top_n=settings.top_n,
        cache_dir=cache_dir,
        refresh=refresh,
        cache_tag=settings.cache_tag,
        exclude_stable_bases=settings.exclude_stable_bases,
        exclude_leveraged=settings.exclude_leveraged,
        min_quote_volume=settings.min_quote_volume,
    )
    universe.attrs["exchange_id"] = settings.exchange
    universe.attrs["cache_tag"] = settings.cache_tag

    prices = build_ccxt_prices(
        universe,
        days=settings.days,
        cache_dir=cache_dir,
        refresh=refresh,
    )
    return universe, prices


def _recursive_autotune(
    candidates: list[UniverseSettings],
    cache_dir: Path,
    refresh: bool,
    index: int = 0,
) -> tuple[UniverseSettings, pd.DataFrame, pd.DataFrame]:
    if index >= len(candidates):
        raise SystemExit(
            "Auto-tuning failed to find a viable universe. "
            "Try adjusting --exchange, --quote, --top-n, or --days."
        )
    settings = candidates[index]
    log(f"Auto-tune attempt {index + 1}/{len(candidates)}.")
    try:
        universe, prices = _attempt_universe_build(settings, cache_dir, refresh)
        log(
            "Auto-tune success with "
            f"exchange={settings.exchange}, quote={settings.quote}, "
            f"top_n={settings.top_n}, days={settings.days}."
        )
        return settings, universe, prices
    except SystemExit as exc:
        log(f"Auto-tune attempt failed: {exc}")
        return _recursive_autotune(candidates, cache_dir, refresh, index + 1)


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
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--quote", type=str, default="USDT")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument(
        "--exclude-stable-bases",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--exclude-leveraged",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min-quote-volume", type=float, default=0.0)
    parser.add_argument("--cache-dir", type=str, default="data_cache")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--cache-tag", type=str, default="v1")
    parser.add_argument(
        "--auto-tune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically retry with lower top-n/days or alternate exchanges/quotes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log("Starting simulation.")
    render_progress(0, 5, prefix="Setup ")
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(exist_ok=True)
    render_progress(1, 5, prefix="Setup ")
    log(f"Exchange: {args.exchange}, Quote: {args.quote}")

    render_progress(2, 5, prefix="Setup ")
    if args.auto_tune:
        log("Auto-tune enabled: searching for viable universe settings.")
        candidates = _build_candidate_settings(args)
        chosen_settings, universe, prices = _recursive_autotune(
            candidates, cache_dir, args.refresh_cache
        )
        config_days = chosen_settings.days
    else:
        chosen_settings = UniverseSettings(
            exchange=args.exchange,
            quote=args.quote,
            top_n=args.top_n,
            days=args.days,
            cache_tag=args.cache_tag,
            exclude_stable_bases=args.exclude_stable_bases,
            exclude_leveraged=args.exclude_leveraged,
            min_quote_volume=args.min_quote_volume,
        )
        universe, prices = _attempt_universe_build(
            chosen_settings, cache_dir, args.refresh_cache
        )
        config_days = chosen_settings.days

    config = SimulationConfig(
        days=config_days,
        rebalance=args.rebalance,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        weighting=args.weighting if args.weighting != "all" else "equal",
        amounts=tuple(args.amounts),
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
