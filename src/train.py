from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_loader import load_config, load_data, infer_problem_type, select_feature_columns


ARTIFACT_PATH = Path("artifacts")
EVAL_PATH = Path("outputs/eval")


def build_pipeline(numeric_features, categorical_features, problem_type: str):
    transformers = []
    if numeric_features:
        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_features))

    if not transformers:
        raise ValueError("Tidak ada fitur yang tersedia untuk training.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = (
        RandomForestClassifier(random_state=42)
        if problem_type == "classification"
        else RandomForestRegressor(random_state=42)
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def main():
    config = load_config()
    df = load_data(config)
    target_column = config.get("target_column", "target")

    if target_column not in df.columns:
        raise ValueError(f"Kolom target tidak ditemukan: {target_column}")

    problem_type = infer_problem_type(df, target_column, config.get("problem_type"))
    numeric_features, categorical_features = select_feature_columns(
        df,
        target_column,
        config.get("numeric_features"),
        config.get("categorical_features"),
    )

    print(f"Problem type: {problem_type}")
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    X = df[numeric_features + categorical_features]
    y = df[target_column]

    if problem_type == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=float(config.get("test_size", 0.2)),
            random_state=int(config.get("random_state", 42)),
            stratify=y,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=float(config.get("test_size", 0.2)),
            random_state=int(config.get("random_state", 42)),
        )

    pipeline = build_pipeline(numeric_features, categorical_features, problem_type)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\nEvaluasi model:")
    if problem_type == "classification":
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2_score(y_test, y_pred):.4f}")

    ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
    EVAL_PATH.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ARTIFACT_PATH / "pipeline.joblib")
    with open(EVAL_PATH / "metrics.txt", "w", encoding="utf-8") as f:
        if problem_type == "classification":
            f.write(f"accuracy={accuracy_score(y_test, y_pred):.4f}\n")
        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = mse ** 0.5
            f.write(f"mae={mean_absolute_error(y_test, y_pred):.4f}\n")
            f.write(f"mse={mse:.4f}\n")
            f.write(f"rmse={rmse:.4f}\n")
            f.write(f"r2={r2_score(y_test, y_pred):.4f}\n")
    print(f"Pipeline disimpan di {ARTIFACT_PATH / 'pipeline.joblib'}")
    print(f"Metadata evaluasi disimpan di {EVAL_PATH}")


if __name__ == "__main__":
    main()
