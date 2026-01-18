# CryptoTrailz Investment Simulator

This project provides a Python script that simulates periodic investments in the top 100 cryptocurrencies by market cap over the past 444 days. It supports multiple investment amounts, different rebalancing schedules, dynamic weighting schemes, and risk controls (stop-loss/take-profit), while comparing results against Bitcoin, Nifty 50, and NASDAQ benchmarks.

## Features

- **Top 100 crypto universe** sourced from CoinGecko (market cap ranked).
- **Investment amounts** from 1,000–5,000 NTD (configurable).
- **Rebalancing intervals**: monthly (default), weekly, quarterly.
- **Weighting strategies**: equal weight, volatility-weighted, momentum-weighted.
- **Risk controls**: stop-loss (default 20%), take-profit (default 50%).
- **Benchmark comparison**: Bitcoin, Nifty 50, NASDAQ (via `yfinance`).
- **CSV caching** of all external data pulls (CoinGecko + benchmarks).
- **Progress logs** so long fetches and simulations show status.

## Requirements

- Python 3.10+
- Packages: `pandas`, `numpy`, `requests`, `matplotlib`, `yfinance`

Install dependencies:

```bash
pip install pandas numpy requests matplotlib yfinance
```

## CoinGecko Demo API Key

The script uses the CoinGecko Demo API and requires an API key **only when a fetch is needed**. Cached runs do **not** require a key.

Set the key in your environment before a fresh fetch:

```bash
export COINGECKO_DEMO_API_KEY="your_demo_key_here"
```

## Usage

Basic run (defaults to 444 days, monthly rebalancing, all weighting strategies, cached data reuse):

```bash
python simulate_investment.py
```

Force a full refresh of cached data:

```bash
python simulate_investment.py --refresh-cache
```

Change cache directory and tag (versioned cache filenames):

```bash
python simulate_investment.py --cache-dir my_cache --cache-tag v2
```

Select a single weighting strategy:

```bash
python simulate_investment.py --weighting equal
```

Change rebalancing interval:

```bash
python simulate_investment.py --rebalance weekly
```

Use custom investment amounts:

```bash
python simulate_investment.py --amounts 1000 2500 5000
```

## Outputs

After running, the script generates:

- `performance_summary.csv` — profit/loss summary by strategy and investment amount.
- `portfolio_<strategy>.png` — line charts of portfolio value vs. benchmarks.
- Cache files under `data_cache/` (or your specified cache directory).

## Caching Details

The script uses a **read-first, fetch-if-missing** pattern and writes cache files atomically:

- Top coins: `top_coins_{limit}_{cache_tag}.csv`
- Per-coin prices: `prices_{coin_id}_{days}_{cache_tag}.csv`
- Benchmarks: `benchmarks_{days}_{cache_tag}.csv`

If cached files have fewer than 80% of expected rows, they are treated as invalid and refetched.

## Notes

- CoinGecko rate limits may still apply. The script implements basic retry/backoff for 429/5xx errors.
- The first run can take time because it pulls and caches data for up to 100 coins.
