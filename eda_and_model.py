# eda_and_model.py

from data_tools.load_data import load_fintech_csv, clean_and_split_id_target
from data_tools.eda import basic_overview, numeric_summary
from data_tools.modeling import train_baseline_logreg

CSV_PATH = "data/fintech.csv"


def main():
    # 1. Load
    df = load_fintech_csv(CSV_PATH)

    # 2. EDA
    overview = basic_overview(df)
    print("=== Overview ===")
    print(overview)

    print("\n=== Numeric Summary ===")
    print(numeric_summary(df))

    # 3. Split features/target
    X, y = clean_and_split_id_target(df, target_col="Exited")

    # 4. Train baseline model
    clf, metrics = train_baseline_logreg(X, y)

    print("\n=== Model Metrics ===")
    print(metrics)


if __name__ == "__main__":
    main()
