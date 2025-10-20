# Snakefile - The Final, Correct Version

# --- "All" Rule ---
rule all:
    input:
        "reports/dummy_backtest_report.html"

# --- Rule 1: Synthesize the dataset ---
# Using 'python -m' tells Python to run the module as a script,
# which correctly resolves package-level imports.
rule synthesize_dataset:
    output:
        "data/processed/master_m15_features.csv"
    shell:
        "python -m src.data.make_dataset"

# --- Rule 2: Train the model ---
rule train_model:
    input:
        "data/processed/master_m15_features.csv"
    output:
        "models/dummy_model.joblib"
    shell:
        "python -m src.models.train_model"

# --- Rule 3: Run the backtest ---
rule run_backtest:
    input:
        "models/dummy_model.joblib"
    output:
        "reports/dummy_backtest_report.html"
    shell:
        "python -m src.models.predict_model"
