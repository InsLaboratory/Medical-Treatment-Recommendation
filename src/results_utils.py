import pandas as pd


def build_results_table(results: list[dict]) -> pd.DataFrame:
    """
    Build a formatted DataFrame from a list of result dicts.

    Each dict must have at least the keys: model_name, accuracy, macro_f1.

    Parameters
    ----------
    results : list of dicts produced by train_and_evaluate / evaluate_resnet18
              / train_and_evaluate_bc

    Returns
    -------
    pd.DataFrame with columns Model, Accuracy, Macro F1 (rounded to 4 d.p.)
    """
    rows = [
        {
            "Model":    r["model_name"],
            "Accuracy": round(r["accuracy"], 4),
            "Macro F1": round(r["macro_f1"], 4),
        }
        for r in results
    ]
    return pd.DataFrame(rows)


def print_results_table(df: pd.DataFrame) -> None:
    """Pretty-print *df* to stdout without the row index."""
    print("\n" + "=" * 60)
    print("Baseline Results Summary")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60 + "\n")


def save_results_csv(df: pd.DataFrame, path: str = "baseline_results.csv") -> None:
    """Save *df* to a CSV file."""
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")