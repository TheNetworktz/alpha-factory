# src/features/build_features.py
"""
A library of functions for engineering features for the Cascade-AI-Trader model.
Each function takes a DataFrame and returns it with new feature columns added.
"""
import pandas as pd
import numpy as np
from typing import List
from pathlib import Path

def calculate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates time-based features from the DataFrame's index.
    """
    print("    - Calculating time features...")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")

    df_copy = df.copy()
    local_tz = "America/New_York"
    df_copy.index = df_copy.index.tz_convert(local_tz)

    london_open = pd.to_datetime("03:00").time()
    ny_am_open = pd.to_datetime("09:30").time()
    ny_pm_open = pd.to_datetime("13:30").time()
    market_close = pd.to_datetime("16:00").time()

    london_killzone_start = pd.to_datetime("02:00").time()
    london_killzone_end = pd.to_datetime("05:00").time()
    ny_killzone_start = pd.to_datetime("08:30").time()
    ny_killzone_end = pd.to_datetime("11:00").time()

    time_of_day = df_copy.index.time

    conditions = [
        (time_of_day >= london_open) & (time_of_day < ny_am_open),
        (time_of_day >= ny_am_open) & (time_of_day < ny_pm_open),
        (time_of_day >= ny_pm_open) & (time_of_day < market_close)
    ]
    choices = [1, 2, 3]
    df_copy['f_time_session_cat'] = np.select(conditions, choices, default=0)

    is_london_kz = (time_of_day >= london_killzone_start) & (time_of_day < london_killzone_end)
    is_ny_kz = (time_of_day >= ny_killzone_start) & (time_of_day < ny_killzone_end)
    df_copy['f_time_is_killzone'] = np.where(is_london_kz | is_ny_kz, 1, 0)

    market_open_dt = df_copy.index.normalize() + pd.Timedelta(hours=9, minutes=30)
    time_diff = (df_copy.index - market_open_dt).total_seconds() / 60
    df_copy['f_time_since_open'] = np.where(time_diff >= 0, time_diff, -1).astype(int)

    return df.join(df_copy[['f_time_session_cat', 'f_time_is_killzone', 'f_time_since_open']])


def calculate_time_based_liquidity(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Calculates time-based liquidity levels like Previous Day High/Low.
    """
    print(f"    - Calculating time-based liquidity for {timeframe}...")
    
    if timeframe.lower() != "d1":
        print(f"    ! Warning: Liquidity for timeframe '{timeframe}' not yet implemented. Skipping.")
        return df

    daily_data_path = Path("data/raw/SPY_1d.csv")
    if not daily_data_path.exists():
        raise FileNotFoundError(f"Daily data file not found at {daily_data_path}. Cannot calculate PDH/PDL.")
        
    # --- THIS IS THE FIX ---
    # Load the file, telling pandas to use the first two rows as a multi-level header.
    df_daily = pd.read_csv(daily_data_path, header=[0, 1], index_col=0)
    
    # The debug output showed the OHLC names are on the *second* level (index 1) of the header.
    df_daily.columns = df_daily.columns.get_level_values(1)
    
    # Standardize all column names to be lowercase to prevent case-sensitivity errors.
    df_daily.columns = [col.lower() for col in df_daily.columns]
    
    # Now that the DataFrame is clean, process the index.
    df_daily.index = pd.to_datetime(df_daily.index, utc=True)
    df_daily.index.name = 'Datetime'
    
    # Now we can safely access the lowercase 'high' and 'low' columns.
    df_daily['PDH'] = df_daily['high'].shift(1)
    df_daily['PDL'] = df_daily['low'].shift(1)

    df_with_date = df.copy()
    df_with_date['Date'] = df_with_date.index.normalize()
    
    df_daily_to_merge = df_daily[['PDH', 'PDL']].copy()
    df_daily_to_merge['Date'] = df_daily_to_merge.index.normalize()

    merged_df = pd.merge(df_with_date, df_daily_to_merge, on='Date', how='left')
    merged_df.set_index(df_with_date.index, inplace=True)

    df_final = df.copy()
    df_final[f'f_{timeframe}_pdh'] = merged_df['PDH']
    df_final[f'f_{timeframe}_pdl'] = merged_df['PDL']

    df_final.fillna(0, inplace=True)

    return df_final


def find_htf_pois(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Identifies Higher-Timeframe Points of Interest (FVGs, Order Blocks).
    """
    print(f"    - Finding POIs on {timeframe}...")
    df[f'_{timeframe}_in_poi'] = 0
    return df

def calculate_smt_divergence(df_main: pd.DataFrame, df_correlated: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates SMT divergence between the main asset and a correlated asset.
    """
    print("    - Calculating SMT divergence...")
    df_main['f_m15_has_smt'] = 0
    return df_main

def calculate_execution_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates features related to the 15M execution trigger.
    """
    print("    - Calculating execution features (CISD, etc.)...")
    df['f_m15_is_sweep'] = 0
    df['f_m15_cisd_strength'] = 0.0
    df['f_m15_cisd_volume'] = 0.0
    return df

def calculate_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates risk/reward based features.
    """
    print("    - Calculating risk features (RR ratio, etc.)...")
    df['f_risk_rr_ratio'] = 0.0
    df['f_risk_path_clear'] = 0
    return df
