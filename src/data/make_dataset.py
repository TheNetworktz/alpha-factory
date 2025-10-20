# src/data/make_dataset.py
import time
from pathlib import Path

def main():
    """
    Dummy script to simulate the data synthesis process.
    It creates a placeholder file to signal completion.
    """
    print("--- Running make_dataset.py: Simulating data synthesis... ---")
    
    # Define the output path for the dummy processed data
    output_path = Path("data/processed/dummy_data.csv")
    
    # Ensure the parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the dummy file
    output_path.touch()
    
    time.sleep(1) # Simulate work
    print(f"--- Success: Created dummy data file at {output_path} ---")
    print("-" * 60)


if __name__ == "__main__":
    main()

