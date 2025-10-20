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
from src.features import build_features

# --- Configuration ---
ASSETS = ["SPY", "QQQ"]
TIMEFRAMES = ["15m", "1h", "1d", "1wk"]
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"

def download_data(assets: List[str], timeframes: List[str], start: str, end: str):
    """
    Downloads historical market data using yfinance for specified assets and timeframes.
    """
    print("--- Running download_data function... ---")
    
    raw_data_path = Path("data/raw")
    raw_data_path.mkdir(parents=True, exist_ok=True)

    for asset in assets:
        for tf in timeframes:
            print(f"\n[DEBUG] --- Processing: {asset} {tf} ---")
            
            params = {
                "tickers": asset,
                "interval": tf,
                "auto_adjust": True,
                "progress": False,
                "group_by": 'ticker'
            }

            if tf == "15m":
                params["period"] = "60d"
            elif tf == "1h":
                params["period"] = "730d"
            elif tf == "4h":
                print(f"    ! Warning: yfinance does not support '4h' interval. Skipping.")
                continue
            else: # For '1d' and '1wk'
                params["start"] = start
                params["end"] = end
            
            print(f"[DEBUG] yfinance download parameters: {params}")
            
            try:
                data = yf.download(**params)
                
                print(f"[DEBUG] Download result for {asset} {tf}:")
                if data.empty:
                    print("[DEBUG] -> DataFrame is EMPTY.")
                else:
                    print(f"[DEBUG] -> DataFrame is NOT empty. Shape: {data.shape}")
                    print("[DEBUG] -> First 5 rows of data head:\n", data.head())

                if data.empty:
                    print(f"    ! Warning: No data downloaded for {asset} {tf}. Skipping.")
                    continue

                output_file = raw_data_path / f"{asset}_{tf}.csv"
                print(f"[DEBUG] Attempting to save to: {output_file.resolve()}")
                data.to_csv(output_file)
                print(f"    -> Saved to {output_file}")
                
                # VERIFY that the file was created
                if output_file.exists():
                    print(f"[DEBUG] VERIFIED: File exists at {output_file}")
                else:
                    print(f"[DEBUG] ERROR: FILE WAS NOT CREATED AT {output_file}")

            except Exception as e:
                print(f"[DEBUG] An exception occurred during download for {asset} {tf}: {e}")
                continue


def synthesize_features():
    """
    Main function to orchestrate the feature engineering process.
    """
    print("\n--- Running synthesize_features function... ---")
    
    print("  - Loading base 15m data for SPY...")
    base_df_path = Path("data/raw/SPY_15m.csv")
    if not base_df_path.exists():
        raise FileNotFoundError(f"Base data file not found at {base_df_path}. Run download_data first.")
    
    df = pd.read_csv(base_df_path, header=[0, 1], index_col=0)
    df.columns = df.columns.get_level_values(1)
    df.columns = [col.lower() for col in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = 'Datetime'

    print("  - Starting feature engineering cascade...")
    
    df = build_features.calculate_time_features(df)
    df = build_features.calculate_time_based_liquidity(df, timeframe="d1")
    
    df = build_features.find_htf_pois(df, timeframe="h1")
    df = build_features.find_htf_pois(df, timeframe="d1")
    
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

