import os
import pandas as pd
import numpy as np
from src import config


def clean_deribit_options_data(file_name: str) -> pd.DataFrame:
    """
    Loads raw historical data, cleans microstructure artifacts,
    and returns a DataFrame with fractional-year time-to-expiry.

    Cleaning steps applied in order:
    1. Column rename to canonical names.
    2. Required-column assertion.
    3. Drop NaN across ALL required columns (not just 3).
    4. Positive price guard: underlying_price > 0, strike > 0.
    5. Liquidity filter: best_bid > 0 and best_ask > 0.
    6. Crossed-market filter: best_ask > best_bid.
       (Crossed markets produce negative bid_ask_spread — a corrupted training feature.)
    7. Timestamp conversion to fractional years (T).
    8. Near-expiry filter: T > 2 hours.
    9. Derived feature columns.
    """
    raw_path = os.path.join(config.RAW_DATA_DIR, file_name)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at: {raw_path}")

    # pandas auto-decompresses .gz by extension
    df = pd.read_csv(raw_path)

    # ── 1. Column rename ───────────────────────────────────────────────────
    rename_dict = {
        'type': 'option_type',
        'strike_price': 'strike',
        'bid_price': 'best_bid',
        'ask_price': 'best_ask',
    }

    df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})
    print(df.columns.tolist())
    # ── 2. Required-column check ───────────────────────────────────────────
    required_cols = [
        'timestamp',
        'expiration',
        'strike',
        'option_type',
        'underlying_price',
        'best_bid',
        'best_ask',
    ]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Input file missing required column: '{col}'. "
                           f"Available: {df.columns.tolist()}")

    # ── 3. Drop NaN across ALL required columns (was only 3 before) ────────
    # NaN in strike or timestamps would cause division-by-zero / NaT errors.
    before = len(df)
    df = df.dropna(subset=required_cols)
    dropped = before - len(df)
    if dropped:
        print(f"  [data_loader] Dropped {dropped:,} rows with NaN in required columns.")

    # ── 4. Positive-value guard ────────────────────────────────────────────
    # zero/negative underlying_price or strike → log(S/K) = -inf in BS formula
    df = df[(df['underlying_price'] > 0) & (df['strike'] > 0)]

    # ── 5. Liquidity filter ────────────────────────────────────────────────
    df = df[(df['best_bid'] > 0) & (df['best_ask'] > 0)]

    # ── 6. Crossed-market filter ───────────────────────────────────────────
    # Without this, bid_ask_spread = ask - bid can be negative (a crossed/locked
    # market), corrupting the spread feature used directly in model training.
    crossed = (df['best_ask'] <= df['best_bid']).sum()
    if crossed:
        print(f"  [data_loader] Removed {crossed:,} crossed/locked market rows "
              f"(best_ask <= best_bid).")
    df = df[df['best_ask'] > df['best_bid']]

    # Convert BTC premium to USD premium
    df['market_mid'] = ((df['best_bid'] + df['best_ask']) / 2.0) * df['underlying_price']

    # Deribit IVs are typically quoted in %
    df['implied_vol'] = df['mark_iv'] / 100.0

    # Safety filter
    df = df[
        (df['implied_vol'] > 0.01) &
        (df['implied_vol'] < 5.0)
    ]

    # Timestamps are in microseconds (Tardis options_chain schema)
    time_to_exp_seconds = (
        df['expiration'] - df['timestamp']
    ) / 1e6
    df['T'] = time_to_exp_seconds / (365.25 * 24 * 60 * 60)

    # ── 8. Near-expiry filter (< 2 h) ─────────────────────────────────────
    df = df[df['T'] > (2.0 / (24.0 * 365.25))]

    # ── 9. Derived columns ─────────────────────────────────────────────────
    df['moneyness']      = df['underlying_price'] / df['strike']
    df['bid_ask_spread'] = (df['best_ask'] - df['best_bid']) * df['underlying_price']

    df = df.sort_values(by='timestamp').reset_index(drop=True)
    return df


def save_processed_data(df: pd.DataFrame, file_name: str) -> None:
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(config.PROCESSED_DATA_DIR, file_name)
    df.to_csv(output_path, index=False)