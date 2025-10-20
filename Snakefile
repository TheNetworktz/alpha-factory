# Snakefile

# --- "All" Rule ---
# This is the default target that Snakemake will build.
# By placing it first, we tell Snakemake that our goal is to produce the final report.
# Snakemake will then work backwards to figure out all the steps needed to get there.
rule all:
    input:
        "reports/dummy_backtest_report.html"

# --- Rule 1: Synthesize the dataset ---
# This rule runs the make_dataset.py script.
# Its output is the dummy data file.
rule synthesize_dataset:
    output:
        "data/processed/dummy_data.csv"
    shell:
        "python src/data/make_dataset.py"

# --- Rule 2: Train the model ---
# This rule runs the train_model.py script.
# It depends on the output of the synthesize_dataset rule.
# Its output is the dummy model file.
rule train_model:
    input:
        "data/processed/dummy_data.csv"
    output:
        "models/dummy_model.joblib"
    shell:
        "python src/models/train_model.py"

# --- Rule 3: Run the backtest ---
# This rule runs the predict_model.py script.
# It depends on the output of the train_model rule.
# Its output is the final report file.
rule run_backtest:
    input:
        "models/dummy_model.joblib"
    output:
        "reports/dummy_backtest_report.html"
    shell:
        "python src/models/predict_model.py"
