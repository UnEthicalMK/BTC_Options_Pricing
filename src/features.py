import pandas as pd
import numpy as np
from src import config
from src.baseline import calculate_greeks


def engineer_features(df: pd.DataFrame, rolling_window: int = 30) -> pd.DataFrame:
    """
    Computes rolling historical volatility and analytical Greeks.

    Look-ahead bias prevention:
    - rolling().std() is purely backward-looking (trailing window).
    - fillna(0.40) is used as a cold-start prior; bfill() has been removed
      because it propagates a future observation backward into the first
      (rolling_window - 1) dates, introducing look-ahead bias.
    - Greeks are computed per-row using only information available at
      each timestamp.
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='us')
    df['date'] = df['datetime'].dt.date

    daily_spot = df.groupby('date')['underlying_price'].last().sort_index()
    daily_log_returns = np.log(daily_spot / daily_spot.shift(1))

    rolling_vol = daily_log_returns.rolling(window=rolling_window).std() * np.sqrt(365.25)
    # LOOK-AHEAD BIAS FIX: removed bfill() which propagated future vol backward.
    # Use a fixed prior (40% annualised vol) for the cold-start period.
    rolling_vol = rolling_vol.fillna(0.40)

    vol_map = rolling_vol.to_dict()
    df['historical_volity'] = df['date'].map(vol_map).astype(float)

    # Compute Greeks for calls and puts separately to avoid manual sign patching.
    call_mask = df['option_type'].str.lower() == 'call'
    put_mask  = ~call_mask

    df['delta'] = np.nan
    df['gamma'] = np.nan
    df['vega']  = np.nan

    for mask, otype in [(call_mask, 'call'), (put_mask, 'put')]:
        if not mask.any():
            continue
        sub = df[mask]
        d, g, v = calculate_greeks(
            S=sub['underlying_price'].values,
            K=sub['strike'].values,
            T=sub['T'].values,
            r=config.RISK_FREE_RATE,
            sigma=sub['historical_volity'].values,
            option_type=otype,
        )
        df.loc[mask, 'delta'] = d
        df.loc[mask, 'gamma'] = g
        df.loc[mask, 'vega']  = v

    df = df.dropna(subset=['historical_volity', 'delta', 'gamma', 'vega']).reset_index(drop=True)
    return df