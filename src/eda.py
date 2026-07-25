from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .data_loader import load_config, load_data, select_feature_columns


OUTPUT_DIR = Path("outputs/eda")


def save_plot(fig, name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def run_eda():
    config = load_config()
    df = load_data(config)
    target_column = config.get("target_column", "target")

    if target_column not in df.columns:
        raise ValueError(f"Kolom target tidak ditemukan: {target_column}")

    numeric_features, categorical_features = select_feature_columns(
        df,
        target_column,
        config.get("numeric_features"),
        config.get("categorical_features"),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.describe(include="all").transpose().to_csv(OUTPUT_DIR / "summary.csv")
    df.isna().sum().sort_values(ascending=False).to_csv(OUTPUT_DIR / "missing_values.csv")
    pd.DataFrame({"duplicate_count": [df.duplicated().sum()]}).to_csv(OUTPUT_DIR / "duplicates.csv")

    print(f"Summary saved to {OUTPUT_DIR / 'summary.csv'}")
    print(f"Missing values saved to {OUTPUT_DIR / 'missing_values.csv'}")
    print(f"Duplicate count saved to {OUTPUT_DIR / 'duplicates.csv'}")

    if target_column in df.columns and pd.api.types.is_numeric_dtype(df[target_column]):
        fig = plt.figure(figsize=(8, 4))
        df[target_column].hist(bins=30)
        plt.title(f"Distribusi target: {target_column}")
        plt.xlabel(target_column)
        plt.ylabel("Count")
        save_plot(fig, "target_distribution.png")

    for column in numeric_features[:3]:
        fig = plt.figure(figsize=(8, 4))
        df[column].hist(bins=30)
        plt.title(f"Histogram: {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        save_plot(fig, f"hist_{column}.png")

    for column in categorical_features[:3]:
        fig = plt.figure(figsize=(8, 4))
        df[column].value_counts(dropna=False).head(20).plot(kind="bar")
        plt.title(f"Category counts: {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        save_plot(fig, f"cat_{column}.png")

    if numeric_features and target_column in df.columns and pd.api.types.is_numeric_dtype(df[target_column]):
        feature_for_scatter = numeric_features[0]
        fig = plt.figure(figsize=(8, 4))
        plt.scatter(df[feature_for_scatter], df[target_column], alpha=0.6)
        plt.title(f"{feature_for_scatter} vs {target_column}")
        plt.xlabel(feature_for_scatter)
        plt.ylabel(target_column)
        save_plot(fig, "scatter_target.png")

    if len(numeric_features) > 1:
        corr = df[numeric_features].corr()
        fig = plt.figure(figsize=(10, 8))
        plt.imshow(corr, cmap="coolwarm", aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(corr)), corr.columns, rotation=90)
        plt.yticks(range(len(corr)), corr.index)
        plt.title("Correlation matrix")
        save_plot(fig, "correlation_matrix.png")

    print("EDA selesai. Cek folder outputs/eda untuk hasil.")


if __name__ == "__main__":
    run_eda()
