"""pytest -q"""

import numpy as np
import pandas as pd
import pytest

from simulate_investment import (
    SimulationConfig,
    _get_rebalance_dates,
    _build_cashflows,
    assert_not_identical_series,
    compute_xirr,
    simulate_fixed_deposit,
    simulate_portfolio_daily,
)


def _make_prices(start: str, end: str, tz: str | None = None) -> pd.DataFrame:
    index = pd.date_range(start=start, end=end, freq="D", tz=tz)
    data = {
        "AssetA": np.linspace(100, 130, len(index)),
        "AssetB": np.linspace(100, 110, len(index)),
    }
    return pd.DataFrame(data, index=index)


def test_rebalance_membership_timezone_normalization() -> None:
    prices = _make_prices("2024-01-01", "2024-03-15", tz="UTC")
    rebalance_dates = _get_rebalance_dates(prices.index, "month_end", 30)
    config = SimulationConfig(weighting="equal", deposit_timing="start")
    series = simulate_portfolio_daily(prices, 100, config, rebalance_dates)

    assert series.attrs["deposits_applied"] == len(rebalance_dates)
    assert series.attrs["rebalances_executed"] == len(rebalance_dates)
    assert series.iloc[-1] > 0


def test_zero_portfolio_regression_constant_prices() -> None:
    index = pd.date_range("2024-01-01", "2024-03-01", freq="D")
    prices = pd.DataFrame({"AssetA": 100.0, "AssetB": 100.0}, index=index)
    rebalance_dates = _get_rebalance_dates(prices.index, "month_end", 30)
    config = SimulationConfig(weighting="equal", deposit_timing="start")
    series = simulate_portfolio_daily(prices, 200, config, rebalance_dates)

    total_invested = 200 * len(rebalance_dates)
    assert series.iloc[-1] > 0
    assert np.isclose(series.iloc[-1], total_invested, atol=1e-6)


def test_crypto_vs_fixed_deposit_aliasing_guard() -> None:
    prices = _make_prices("2024-01-01", "2024-02-15")
    rebalance_dates = _get_rebalance_dates(prices.index, "month_end", 30)
    config = SimulationConfig(weighting="equal", deposit_timing="start")
    crypto_series = simulate_portfolio_daily(prices, 150, config, rebalance_dates)
    fd_series = simulate_fixed_deposit(
        150, rebalance_dates, prices.index, annual_rate=0.05, deposit_timing="start"
    )

    assert_not_identical_series(crypto_series, fd_series, "crypto", "fd")

    with pytest.raises(ValueError):
        assert_not_identical_series(fd_series, fd_series.copy(), "fd", "fd-copy")


def test_strategy_differentiation_momentum_vs_equal() -> None:
    index = pd.date_range("2024-01-01", "2024-04-15", freq="D")
    prices = pd.DataFrame(
        {
            "AssetA": np.linspace(100, 160, len(index)),
            "AssetB": np.linspace(100, 100, len(index)),
        },
        index=index,
    )
    rebalance_dates = _get_rebalance_dates(prices.index, "month_end", 30)
    equal_cfg = SimulationConfig(weighting="equal", deposit_timing="start")
    momentum_cfg = SimulationConfig(weighting="momentum", deposit_timing="start")

    equal_series = simulate_portfolio_daily(prices, 100, equal_cfg, rebalance_dates)
    momentum_series = simulate_portfolio_daily(prices, 100, momentum_cfg, rebalance_dates)

    assert not np.allclose(
        equal_series.values, momentum_series.values, rtol=1e-6, atol=1e-6
    )


def test_xirr_sanity() -> None:
    index = pd.date_range("2024-01-01", "2024-03-01", freq="D")
    prices = pd.DataFrame({"AssetA": 100.0, "AssetB": 100.0}, index=index)
    rebalance_dates = _get_rebalance_dates(prices.index, "month_end", 30)
    config = SimulationConfig(weighting="equal", deposit_timing="start")
    series = simulate_portfolio_daily(prices, 100, config, rebalance_dates)

    cashflows = _build_cashflows(series, 100, rebalance_dates)
    xirr = compute_xirr(cashflows)
    assert abs(xirr) < 1e-4

    rising_prices = _make_prices("2024-01-01", "2024-03-01")
    rising_series = simulate_portfolio_daily(rising_prices, 100, config, rebalance_dates)
    rising_xirr = compute_xirr(_build_cashflows(rising_series, 100, rebalance_dates))
    assert rising_xirr > 0
