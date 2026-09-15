from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DB_PATH = Path("db/yield_forecasting.duckdb")
OUTDIR = Path("data/processed/strawberry")
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET = "next_fresh_matter_g"
TRAIN_YEAR = 2022
TEST_YEAR = 2023
RANDOM_STATE = 42


MODEL_DEFINITIONS = {
    "baseline_persistence": {
        "label": "Persistence baseline",
        "type": "baseline",
        "note": "Predicts next fresh matter using current fresh matter from the same treatment.",
    },
    "baseline_treatment_mean": {
        "label": "Treatment mean baseline",
        "type": "baseline",
        "note": "Predicts using the previous season mean target value for each nitrogen treatment.",
    },
    "weather_gdd_ridge": {
        "label": "Weather + GDD linear model",
        "type": "ridge",
        "features": [
            "treatment_n_level",
            "forecast_horizon_days",
            "day_of_year",
            "days_since_weather_start",
            "tmean_c",
            "rhmean_pct",
            "solar_radiation",
            "gdd_base_10",
            "cumulative_gdd_base_10",
            "cumulative_solar_radiation",
            "tmean_7d",
            "solar_radiation_7d",
            "rhmean_7d",
        ],
        "note": "Uses treatment, weather and thermal-time features only.",
    },
    "crop_development_ridge": {
        "label": "Crop development linear model",
        "type": "ridge",
        "features": [
            "treatment_n_level",
            "forecast_horizon_days",
            "day_of_year",
            "days_since_weather_start",
            "cumulative_gdd_base_10",
            "cumulative_solar_radiation",
            "tmean_7d",
            "solar_radiation_7d",
            "current_fruit_number",
            "current_fresh_matter_g",
            "current_dry_matter_g",
            "previous_fruit_number",
            "previous_fresh_matter_g",
            "previous_dry_matter_g",
            "change_fruit_number",
            "change_fresh_matter_g",
            "rolling_mean_2_fruit_number",
            "rolling_mean_2_fresh_matter_g",
            "rolling_mean_3_fruit_number",
            "rolling_mean_3_fresh_matter_g",
            "mean_fruit_diameter_mm",
            "mean_individual_fruit_fresh_weight_g",
            "mean_tagged_fruit_diameter_mm",
            "mean_tagged_fruit_length_mm",
        ],
        "note": "Adds crop-status and recent development features available at the forecast origin date.",
    },
    "crop_development_rf": {
        "label": "Crop development random forest",
        "type": "random_forest",
        "features": [
            "treatment_n_level",
            "forecast_horizon_days",
            "day_of_year",
            "days_since_weather_start",
            "tmax_c",
            "tmin_c",
            "tmean_c",
            "rhmean_pct",
            "solar_radiation",
            "gdd_base_10",
            "cumulative_gdd_base_10",
            "cumulative_solar_radiation",
            "tmean_7d",
            "solar_radiation_7d",
            "rhmean_7d",
            "current_fruit_number",
            "current_fresh_matter_g",
            "current_dry_matter_g",
            "previous_fruit_number",
            "previous_fresh_matter_g",
            "previous_dry_matter_g",
            "change_fruit_number",
            "change_fresh_matter_g",
            "rolling_mean_2_fruit_number",
            "rolling_mean_2_fresh_matter_g",
            "rolling_mean_3_fruit_number",
            "rolling_mean_3_fresh_matter_g",
            "mean_fruit_diameter_mm",
            "mean_individual_fruit_fresh_weight_g",
            "mean_tagged_fruit_diameter_mm",
            "mean_tagged_fruit_length_mm",
        ],
        "note": "Non-linear model using weather, GDD and crop-development features.",
    },
}


def load_forecast_dataset() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT * FROM strawberry_forecast_features").fetchdf()
    con.close()

    df["current_observation_date"] = pd.to_datetime(df["current_observation_date"])
    df["target_date"] = pd.to_datetime(df["target_date"])

    return df


def split_train_test(df: pd.DataFrame):
    train_df = df.loc[df["year"] == TRAIN_YEAR].copy()
    test_df = df.loc[df["year"] == TEST_YEAR].copy()

    train_df = train_df.dropna(subset=[TARGET])
    test_df = test_df.dropna(subset=[TARGET])

    return train_df, test_df


def calculate_metrics(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_observed = np.mean(y_true)

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "r2": r2_score(y_true, y_pred),
        "bias": float(np.mean(y_pred - y_true)),
        "mean_observed": mean_observed,
        "mae_pct_of_mean_observed": mean_absolute_error(y_true, y_pred)
        / mean_observed
        * 100,
    }


def build_prediction_output(test_df, model_name, model_label, y_pred):
    output = test_df[
        [
            "year",
            "nitrogen_treatment",
            "current_observation_date",
            "target_date",
            "forecast_horizon_days",
            "current_fruit_number",
            "current_fresh_matter_g",
            TARGET,
        ]
    ].copy()

    output["model"] = model_name
    output["model_label"] = model_label
    output["observed"] = output[TARGET]
    output["predicted"] = y_pred
    output["residual"] = output["observed"] - output["predicted"]

    return output


def train_persistence_baseline(train_df, test_df, model_name, model_def):
    prediction_source = test_df["current_fresh_matter_g"].fillna(
        train_df[TARGET].mean()
    )

    y_pred = prediction_source.values
    y_true = test_df[TARGET].values

    metrics = calculate_metrics(y_true, y_pred)

    predictions = build_prediction_output(
        test_df=test_df,
        model_name=model_name,
        model_label=model_def["label"],
        y_pred=y_pred,
    )

    return metrics, predictions, pd.DataFrame()


def train_treatment_mean_baseline(train_df, test_df, model_name, model_def):
    treatment_means = train_df.groupby("nitrogen_treatment")[TARGET].mean()
    global_mean = train_df[TARGET].mean()

    y_pred = (
        test_df["nitrogen_treatment"]
        .map(treatment_means)
        .fillna(global_mean)
        .values
    )

    y_true = test_df[TARGET].values

    metrics = calculate_metrics(y_true, y_pred)

    predictions = build_prediction_output(
        test_df=test_df,
        model_name=model_name,
        model_label=model_def["label"],
        y_pred=y_pred,
    )

    return metrics, predictions, pd.DataFrame()


def build_model_pipeline(model_type: str, feature_cols: list[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    if model_type == "ridge":
        model = Ridge(alpha=1.0)

    elif model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def extract_feature_effects(pipeline, model_type, feature_cols, model_name, model_label):
    model = pipeline.named_steps["model"]

    if model_type == "ridge":
        values = model.coef_
        value_name = "coefficient"

    elif model_type == "random_forest":
        values = model.feature_importances_
        value_name = "importance"

    else:
        return pd.DataFrame()

    rows = []

    for feature, value in zip(feature_cols, values):
        rows.append(
            {
                "model": model_name,
                "model_label": model_label,
                "feature": feature,
                "effect_type": value_name,
                "effect_value": value,
                "abs_effect_value": abs(value),
            }
        )

    return pd.DataFrame(rows).sort_values("abs_effect_value", ascending=False)


def train_sklearn_model(train_df, test_df, model_name, model_def):
    feature_cols = [
        col for col in model_def["features"]
        if col in train_df.columns
    ]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]

    X_test = test_df[feature_cols]
    y_test = test_df[TARGET]

    pipeline = build_model_pipeline(model_def["type"], feature_cols)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = calculate_metrics(y_test.values, y_pred)

    predictions = build_prediction_output(
        test_df=test_df,
        model_name=model_name,
        model_label=model_def["label"],
        y_pred=y_pred,
    )

    feature_effects = extract_feature_effects(
        pipeline=pipeline,
        model_type=model_def["type"],
        feature_cols=feature_cols,
        model_name=model_name,
        model_label=model_def["label"],
    )

    return metrics, predictions, feature_effects


def main():
    df = load_forecast_dataset()
    train_df, test_df = split_train_test(df)

    print(f"Train year: {TRAIN_YEAR}, rows: {len(train_df)}")
    print(f"Test year: {TEST_YEAR}, rows: {len(test_df)}")

    metrics_rows = []
    prediction_frames = []
    effect_frames = []

    for model_name, model_def in MODEL_DEFINITIONS.items():
        print(f"\nTraining/evaluating: {model_name}")

        if model_name == "baseline_persistence":
            metrics, predictions, effects = train_persistence_baseline(
                train_df, test_df, model_name, model_def
            )

        elif model_name == "baseline_treatment_mean":
            metrics, predictions, effects = train_treatment_mean_baseline(
                train_df, test_df, model_name, model_def
            )

        else:
            metrics, predictions, effects = train_sklearn_model(
                train_df, test_df, model_name, model_def
            )

        metrics.update(
            {
                "model": model_name,
                "model_label": model_def["label"],
                "model_type": model_def["type"],
                "train_year": TRAIN_YEAR,
                "test_year": TEST_YEAR,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "n_features": len(model_def.get("features", [])),
                "note": model_def["note"],
            }
        )

        metrics_rows.append(metrics)
        prediction_frames.append(predictions)

        if not effects.empty:
            effect_frames.append(effects)

        print(
            f"{model_def['label']}: "
            f"MAE={metrics['mae']:.2f}, "
            f"RMSE={metrics['rmse']:.2f}, "
            f"R2={metrics['r2']:.3f}, "
            f"Bias={metrics['bias']:.2f}"
        )

    metrics_df = pd.DataFrame(metrics_rows)

    persistence_mae = metrics_df.loc[
        metrics_df["model"] == "baseline_persistence",
        "mae",
    ].iloc[0]

    metrics_df["mae_improvement_vs_persistence"] = (
        persistence_mae - metrics_df["mae"]
    )

    metrics_df["mae_improvement_pct_vs_persistence"] = (
        metrics_df["mae_improvement_vs_persistence"] / persistence_mae * 100
    )

    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    if effect_frames:
        effects_df = pd.concat(effect_frames, ignore_index=True)
    else:
        effects_df = pd.DataFrame()

    metrics_path = OUTDIR / "strawberry_forecast_model_metrics.csv"
    predictions_path = OUTDIR / "strawberry_forecast_predictions.csv"
    effects_path = OUTDIR / "strawberry_forecast_feature_effects.csv"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    effects_df.to_csv(effects_path, index=False)

    con = duckdb.connect(str(DB_PATH))

    tables = {
        "strawberry_forecast_model_metrics": metrics_df,
        "strawberry_forecast_predictions": predictions_df,
        "strawberry_forecast_feature_effects": effects_df,
    }

    for table_name, table_df in tables.items():
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.register("tmp_df", table_df)
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")

    con.close()

    print("\nSaved:")
    print(metrics_path)
    print(predictions_path)
    print(effects_path)

    print("\nModel metrics:")
    print(
        metrics_df[
            [
                "model_label",
                "mae",
                "rmse",
                "r2",
                "bias",
                "mae_improvement_pct_vs_persistence",
            ]
        ]
    )


if __name__ == "__main__":
    main()
