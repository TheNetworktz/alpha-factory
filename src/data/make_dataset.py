# src/data/make_dataset.py
"""
This script downloads raw market data, processes it, engineers features,
and saves the final master dataset.
"""
import yfinance as yf
import pandas as pd
from pathlib import Path
from src.features import build_features

# Use a 2-year period for all timeframes to ensure data overlap
DATA_PERIOD = "2y" 
ASSETS = ['SPY', 'QQQ']
TIMEFRAMES = ['15m', '1h', '1d', '1wk']


def download_data(assets: list, timeframes: list, period: str):
    """
    Downloads historical market data for given assets and timeframes.
    """
    print("--- Running download_data function... ---")
    raw_data_dir = Path("data/raw")
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    for asset in assets:
        for tf in timeframes:
            print(f"  - Downloading {asset} {tf} data...")
            
            # Use specific periods for intervals with API limits
            current_period = period
            if tf == '15m':
                current_period = '60d'
            
            try:
                # Use group_by='ticker' to get a consistent multi-level header
                data = yf.download(
                    tickers=asset,
                    interval=tf,
                    auto_adjust=True,
                    progress=False,
                    group_by='ticker',
                    period=current_period
                )
                
                if data.empty:
                    print(f"    ! Warning: No data downloaded for {asset} {tf}. Skipping.")
                    continue

                filepath = raw_data_dir / f"{asset}_{tf}.csv"
                data.to_csv(filepath)
                print(f"    -> Saved to {filepath}")

            except Exception as e:
                print(f"    ! ERROR downloading {asset} {tf}: {e}")

def synthesize_features():
    """
    Loads the base 15m data and cascades through the feature engineering functions.
    """
    print("\n--- Running synthesize_features function... ---")
    base_asset = ASSETS[0]
    base_df_path = Path(f"data/raw/{base_asset}_15m.csv")
    
    if not base_df_path.exists():
        raise FileNotFoundError(f"Base data file not found at {base_df_path}. Run download_data first.")

    print(f"  - Loading base 15m data for {base_asset}...")
    
    # --- THIS IS THE CRITICAL REGRESSION FIX ---
    # Use the robust loading logic we discovered earlier
    df = pd.read_csv(base_df_path, header=[0, 1], index_col=0)
    df.columns = df.columns.get_level_values(1)
    df.columns = [col.lower() for col in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = 'Datetime'
    # ---
    
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
    download_data(assets=ASSETS, timeframes=TIMEFRAMES, period=DATA_PERIOD)
    synthesize_features()


if __name__ == "__main__":
    main()
