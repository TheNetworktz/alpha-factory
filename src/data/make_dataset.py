# src/data/make_dataset.py
"""
This script downloads raw market data, processes it, engineers features,
and saves the final master dataset.
"""
import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import List, Dict

# Use an absolute import from the project's 'src' directory.
# This works because we installed the project in editable mode with 'pip install -e .'
from src.features import build_features

# --- Configuration ---
ASSETS = ["SPY", "QQQ"]
TIMEFRAMES = ["15m", "1h", "4h", "1d", "1wk"]
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"

def download_data(assets: List[str], timeframes: List[str], start: str, end: str):
    """
    Downloads historical market data using yfinance for specified assets and timeframes.
    Respects the data limitations of the yfinance API.
    """
    print("--- Running download_data function... ---")
    
    raw_data_path = Path("data/raw")
    raw_data_path.mkdir(parents=True, exist_ok=True)

    for asset in assets:
        for tf in timeframes:
            print(f"  - Downloading {asset} {tf} data...")
            
            start_date = start
            end_date = end
            period = None

            if tf == "15m":
                # 15m data is only available for the last 60 days.
                start_date = None
                end_date = None
                period = "60d"
            elif tf == "1h":
                # 1h data is available for the last 730 days.
                start_date = None
                end_date = None
                period = "730d"
            elif tf == "4h":
                # yfinance does not support '4h'.
                print(f"    ! Warning: yfinance does not support '4h' interval. Skipping.")
                continue
            
            data = yf.download(
                tickers=asset,
                start=start_date,
                end=end_date,
                period=period,
                interval=tf,
                auto_adjust=True,
                progress=False,
            )
            
            if data.empty:
                print(f"    ! Warning: No data downloaded for {asset} {tf}. Skipping.")
                continue

            output_file = raw_data_path / f"{asset}_{tf}.csv"
            data.to_csv(output_file)
            print(f"    -> Saved to {output_file}")

def synthesize_features():
    """
    Main function to orchestrate the feature engineering process.
    It loads raw data, applies feature engineering functions, and saves the result.
    """
    print("\n--- Running synthesize_features function... ---")
    
    print("  - Loading base 15m data for SPY...")
    base_df_path = Path("data/raw/SPY_15m.csv")
    if not base_df_path.exists():
        raise FileNotFoundError(f"Base data file not found at {base_df_path}. Run download_data first.")
    
    # --- THIS IS THE FIX ---
    # Use index_col=0 to specify that the first column in the CSV is the index.
    # yfinance saves the Datetime index as an unnamed first column.
    df = pd.read_csv(base_df_path, index_col=0, parse_dates=True)
    
    # Let's give the index a name for clarity
    df.index.name = 'Datetime'

    print("  - Starting feature engineering cascade...")
    
    df = build_features.calculate_time_features(df)
    df = build_features.calculate_time_based_liquidity(df, timeframe="d1")
    df = build_features.calculate_execution_features(df)
    df = build_features.calculate_risk_features(df)
    
    print("  - Feature engineering cascade complete.")

    output_path = Path("data/processed/master_m15_features.csv")
    df.to_csv(output_path)
    print(f"  -> Master feature dataset saved to {output_path}")


def main():
    """Main function to run the data processing pipeline."""
    download_data(assets=ASSETS, timeframes=TIMEFRAMES, start=START_DATE, end=END_DATE)
    synthesize_features()


if __name__ == "__main__":
    main()
