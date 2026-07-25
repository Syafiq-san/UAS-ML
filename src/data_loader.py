import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(config: Dict[str, Any]) -> pd.DataFrame:
    path = config.get("dataset_path", "data/raw/cars.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv"]:
        return pd.read_csv(path)
    if ext in [".xls", ".xlsx"]:
        return pd.read_excel(path)
    raise ValueError("Format file tidak didukung. Gunakan CSV atau Excel.")


def infer_problem_type(
    df: pd.DataFrame,
    target_column: str,
    override: Optional[str] = None,
) -> str:
    if override and override.lower() in {"classification", "regression"}:
        return override.lower()

    target = df[target_column]
    if pd.api.types.is_numeric_dtype(target):
        return "regression"

    if target.dtype.kind in "O" or target.dtype.name == "category":
        return "classification"

    unique_values = target.nunique(dropna=True)
    if unique_values <= 20:
        return "classification"
    return "regression"


def select_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    if (numeric_features is not None and len(numeric_features) > 0) or (
        categorical_features is not None and len(categorical_features) > 0
    ):
        return numeric_features or [], categorical_features or []

    features = [c for c in df.columns if c != target_column]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    return numeric, categorical
