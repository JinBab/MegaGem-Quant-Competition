"""Volatility scanner for the predefined SYMBOLS universe.

Computes realized volatility from Binance OHLCV data using log returns.

Defaults:
- Interval: 1h
- Lookback: 7 days
- Annualized using sqrt(periods_per_year) based on interval

Outputs a pandas DataFrame with columns:
  Coin, Volatility, Samples

CLI usage (quick): run the module to print top 10 by annualized vol.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List, Optional, Tuple

import math
import pandas as pd

# Local imports
from market_scanner import SYMBOLS
from BinanceAPI import fetch_data


# ------------------------------
# Helpers
# ------------------------------

def _annualization_factor(interval: str) -> float:
    """Return sqrt(periods_per_year) for the given bar interval.

    Assumes crypto trades 24/7.
    """
    interval = interval.lower()
    if interval.endswith('m'):
        minutes = int(interval[:-1] or 1)
        periods_per_day = (24 * 60) / minutes
        periods_per_year = 365 * periods_per_day
    elif interval.endswith('h'):
        hours = int(interval[:-1] or 1)
        periods_per_day = 24 / hours
        periods_per_year = 365 * periods_per_day
    elif interval.endswith('d'):
        days = int(interval[:-1] or 1)
        periods_per_year = 365 / days
    elif interval.endswith('w'):
        weeks = int(interval[:-1] or 1)
        periods_per_year = 52 / weeks
    elif interval.endswith('m'.upper()):  # '1M' monthly bars (Binance style)
        months = int(interval[:-1] or 1)
        periods_per_year = 12 / months
    else:
        # Fallback: assume hourly
        periods_per_year = 365 * 24
    return math.sqrt(periods_per_year)


def _time_fmt(dt: datetime) -> str:
    """Format datetime for BinanceAPI.fetch_data."""
    return dt.strftime("%d %b %Y %H:%M:%S")


def _symbol_to_coin(symbol: str) -> str:
    return symbol.replace('USDT', '')


def _realized_vol_from_prices(prices: pd.Series, annualize: bool, interval: str) -> Tuple[Optional[float], int]:
    """Compute realized volatility from a price series.

    Returns (volatility, sample_count). If insufficient data, volatility is None.
    """
    try:
        p = pd.to_numeric(prices, errors='coerce').dropna()
        if len(p) < 3:
            return None, int(len(p))
        rets = (p / p.shift(1)).apply(lambda x: math.log(x)).dropna()
        std = rets.std(ddof=1)
        if pd.isna(std) or std == 0:
            return None, int(len(rets))
        if annualize:
            af = _annualization_factor(interval)
            vol = float(std * af)
        else:
            vol = float(std)
        return vol, int(len(rets))
    except Exception:
        return None, 0


@dataclass
class VolConfig:
    interval: str = '1h'
    lookback_days: int = 7
    annualize: bool = True
    max_workers: int = 8


def get_volatility_df(
    interval: str = '1h',
    lookback_days: int = 7,
    symbols: Optional[Iterable[str]] = None,
    annualize: bool = True,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Compute realized volatility for the given symbols.

    - interval: Binance kline interval (e.g., '1m','5m','1h','4h','1d')
    - lookback_days: number of days to fetch
    - symbols: iterable of Binance symbols; defaults to SYMBOLS from market_scanner
    - annualize: multiply by sqrt(periods/year) to annualize
    - max_workers: concurrency for fetching klines
    """
    symbols = list(symbols) if symbols is not None else list(SYMBOLS)

    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=lookback_days)
    start_str = _time_fmt(start_dt)
    end_str = _time_fmt(end_dt)

    results: List[Tuple[str, Optional[float], int]] = []

    def _work(sym: str) -> Tuple[str, Optional[float], int]:
        try:
            df = fetch_data(sym, interval, start_str, end_str)
            if df is None or df.empty or 'Close' not in df.columns:
                return sym, None, 0
            vol, n = _realized_vol_from_prices(df['Close'], annualize, interval)
            return sym, vol, n
        except Exception:
            return sym, None, 0

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_work, s): s for s in symbols}
        for fut in as_completed(futures):
            sym, vol, n = fut.result()
            results.append((sym, vol, n))

    rows = []
    for sym, vol, n in results:
        if vol is None:
            continue
        rows.append({
            'Coin': _symbol_to_coin(sym),
            'Volatility': vol,
            'Samples': n,
            'Symbol': sym,
            'Interval': interval,
            'LookbackDays': lookback_days,
        })

    if not rows:
        return pd.DataFrame(columns=['Coin','Volatility','Samples','Symbol','Interval','LookbackDays'])

    out = pd.DataFrame(rows)
    out = out.sort_values('Volatility', ascending=False).reset_index(drop=True)
    return out


if __name__ == '__main__':
    cfg = VolConfig()
    df = get_volatility_df(
        interval=cfg.interval,
        lookback_days=cfg.lookback_days,
        annualize=cfg.annualize,
        max_workers=cfg.max_workers,
    )
    print(f"Top 10 volatility ({cfg.interval}, {cfg.lookback_days}d, annualized={cfg.annualize})")
    with pd.option_context('display.float_format', '{:,.4f}'.format):
        print(df.head(10).to_string(index=False))
