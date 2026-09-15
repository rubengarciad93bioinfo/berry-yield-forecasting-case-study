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
TEST_YEAR = 2023
RANDOM_STATE = 42


MODEL_DEFINITIONS = {
    "rolling_persistence": {
        "label": "Rolling persistence baseline",
        "type": "baseline_persistence",
        "features": [],
        "note": "Predicts next fresh matter using current fresh matter from the same treatment.",
    },
    "rolling_treatment_mean": {
        "label": "Rolling treatment mean baseline",
        "type": "baseline_treatment_mean",
        "features": [],
        "note": "Predicts from historical mean fresh matter for the same nitrogen treatment.",
    },
    "rolling_weather_gdd_ridge": {
        "label": "Rolling weather + GDD linear model",
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
        "note": "Rolling model using treatment, weather and thermal-time features.",
    },
    "rolling_crop_development_ridge": {
        "label": "Rolling crop development linear model",
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
        "note": "Rolling model using current crop status and recent development features.",
    },
    "rolling_crop_development_rf": {
        "label": "Rolling crop development random forest",
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
        "note": "Rolling non-linear model using weather, GDD and crop-development features.",
    },
}


def load_forecast_dataset() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT * FROM strawberry_forecast_features").fetchdf()
    con.close()

    df["current_observation_date"] = pd.to_datetime(df["current_observation_date"])
    df["target_date"] = pd.to_datetime(df["target_date"])

    return df


def safe_r2(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan

    return r2_score(y_true, y_pred)


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_observed = np.mean(y_true)

    return {
        "n_predictions": len(y_true),
        "mae": mae,
        "rmse": rmse,
        "r2": safe_r2(y_true, y_pred),
        "bias": float(np.mean(y_pred - y_true)),
        "mean_observed": mean_observed,
        "mae_pct_of_mean_observed": mae / mean_observed * 100,
    }


def build_pipeline(model_type: str, feature_cols: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_cols,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    if model_type == "ridge":
        model = Ridge(alpha=1.0)

    elif model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
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


def predict_baseline(history_df, forecast_rows, model_type):
    if model_type == "baseline_persistence":
        fallback = history_df[TARGET].mean()
        return forecast_rows["current_fresh_matter_g"].fillna(fallback).values

    if model_type == "baseline_treatment_mean":
        treatment_means = history_df.groupby("nitrogen_treatment")[TARGET].mean()
        global_mean = history_df[TARGET].mean()

        return (
            forecast_rows["nitrogen_treatment"]
            .map(treatment_means)
            .fillna(global_mean)
            .values
        )

    raise ValueError(f"Unsupported baseline type: {model_type}")


def extract_feature_effects(
    pipeline,
    model_type,
    feature_cols,
    model_name,
    model_label,
    origin_date,
    n_training_rows,
    n_current_year_training_rows,
):
    model = pipeline.named_steps["model"]

    if model_type == "ridge":
        values = model.coef_
        effect_type = "standardized_coefficient"

    elif model_type == "random_forest":
        values = model.feature_importances_
        effect_type = "importance"

    else:
        return pd.DataFrame()

    rows = []

    for feature, value in zip(feature_cols, values):
        rows.append(
            {
                "model": model_name,
                "model_label": model_label,
                "forecast_origin_date": origin_date,
                "feature": feature,
                "effect_type": effect_type,
                "effect_value": value,
                "abs_effect_value": abs(value),
                "n_training_rows": n_training_rows,
                "n_current_year_training_rows": n_current_year_training_rows,
            }
        )

    return pd.DataFrame(rows)


def build_prediction_rows(
    forecast_rows,
    model_name,
    model_label,
    model_note,
    y_pred,
    n_training_rows,
    n_previous_year_training_rows,
    n_current_year_training_rows,
):
    output = forecast_rows[
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
    output["model_note"] = model_note
    output["observed"] = output[TARGET]
    output["predicted"] = y_pred
    output["residual"] = output["observed"] - output["predicted"]
    output["absolute_error"] = output["residual"].abs()
    output["n_training_rows"] = n_training_rows
    output["n_previous_year_training_rows"] = n_previous_year_training_rows
    output["n_current_year_training_rows"] = n_current_year_training_rows

    return output


def run_rolling_forecast(df: pd.DataFrame):
    test_df = df.loc[df["year"] == TEST_YEAR].dropna(subset=[TARGET]).copy()
    origin_dates = sorted(test_df["current_observation_date"].dropna().unique())

    prediction_frames = []
    effect_frames = []
    training_rows = []

    for origin_date in origin_dates:
        forecast_rows = test_df.loc[
            test_df["current_observation_date"] == origin_date
        ].copy()

        history_df = df.loc[
            (df["year"] < TEST_YEAR)
            | (
                (df["year"] == TEST_YEAR)
                & (df["target_date"] <= origin_date)
            )
        ].dropna(subset=[TARGET]).copy()

        if history_df.empty:
            continue

        n_previous_year_training_rows = int((history_df["year"] < TEST_YEAR).sum())
        n_current_year_training_rows = int((history_df["year"] == TEST_YEAR).sum())

        training_rows.append(
            {
                "forecast_origin_date": origin_date,
                "n_training_rows": len(history_df),
                "n_previous_year_training_rows": n_previous_year_training_rows,
                "n_current_year_training_rows": n_current_year_training_rows,
                "n_forecast_rows": len(forecast_rows),
            }
        )

        for model_name, model_def in MODEL_DEFINITIONS.items():
            model_type = model_def["type"]

            if model_type.startswith("baseline"):
                y_pred = predict_baseline(
                    history_df=history_df,
                    forecast_rows=forecast_rows,
                    model_type=model_type,
                )

                effects = pd.DataFrame()

            else:
                feature_cols = [
                    col for col in model_def["features"]
                    if col in history_df.columns
                ]

                pipeline = build_pipeline(model_type, feature_cols)

                X_train = history_df[feature_cols]
                y_train = history_df[TARGET]

                X_forecast = forecast_rows[feature_cols]

                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_forecast)

                effects = extract_feature_effects(
                    pipeline=pipeline,
                    model_type=model_type,
                    feature_cols=feature_cols,
                    model_name=model_name,
                    model_label=model_def["label"],
                    origin_date=origin_date,
                    n_training_rows=len(history_df),
                    n_current_year_training_rows=n_current_year_training_rows,
                )

            predictions = build_prediction_rows(
                forecast_rows=forecast_rows,
                model_name=model_name,
                model_label=model_def["label"],
                model_note=model_def["note"],
                y_pred=y_pred,
                n_training_rows=len(history_df),
                n_previous_year_training_rows=n_previous_year_training_rows,
                n_current_year_training_rows=n_current_year_training_rows,
            )

            prediction_frames.append(predictions)

            if not effects.empty:
                effect_frames.append(effects)

    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    training_history_df = pd.DataFrame(training_rows)

    if effect_frames:
        effects_df = pd.concat(effect_frames, ignore_index=True)
    else:
        effects_df = pd.DataFrame()

    return predictions_df, effects_df, training_history_df


def build_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (model, model_label), group in predictions_df.groupby(["model", "model_label"]):
        metrics = calculate_metrics(group["observed"], group["predicted"])
        metrics.update(
            {
                "model": model,
                "model_label": model_label,
                "forecast_mode": "rolling_in_season",
                "test_year": TEST_YEAR,
            }
        )
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows)

    persistence_mae = metrics_df.loc[
        metrics_df["model"] == "rolling_persistence",
        "mae",
    ].iloc[0]

    metrics_df["mae_improvement_vs_persistence"] = (
        persistence_mae - metrics_df["mae"]
    )

    metrics_df["mae_improvement_pct_vs_persistence"] = (
        metrics_df["mae_improvement_vs_persistence"] / persistence_mae * 100
    )

    return metrics_df.sort_values("mae")


def build_metrics_by_origin(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = ["model", "model_label", "current_observation_date"]

    for keys, group in predictions_df.groupby(group_cols):
        model, model_label, origin_date = keys
        metrics = calculate_metrics(group["observed"], group["predicted"])

        metrics.update(
            {
                "model": model,
                "model_label": model_label,
                "current_observation_date": origin_date,
                "n_current_year_training_rows": group[
                    "n_current_year_training_rows"
                ].iloc[0],
            }
        )

        rows.append(metrics)

    return pd.DataFrame(rows).sort_values(
        ["model", "current_observation_date"]
    )


def main():
    df = load_forecast_dataset()

    predictions_df, effects_df, training_history_df = run_rolling_forecast(df)
    metrics_df = build_metrics(predictions_df)
    metrics_by_origin_df = build_metrics_by_origin(predictions_df)

    paths = {
        "strawberry_rolling_forecast_predictions": OUTDIR
        / "strawberry_rolling_forecast_predictions.csv",
        "strawberry_rolling_forecast_feature_effects": OUTDIR
        / "strawberry_rolling_forecast_feature_effects.csv",
        "strawberry_rolling_forecast_training_history": OUTDIR
        / "strawberry_rolling_forecast_training_history.csv",
        "strawberry_rolling_forecast_metrics": OUTDIR
        / "strawberry_rolling_forecast_metrics.csv",
        "strawberry_rolling_forecast_metrics_by_origin": OUTDIR
        / "strawberry_rolling_forecast_metrics_by_origin.csv",
    }

    outputs = {
        "strawberry_rolling_forecast_predictions": predictions_df,
        "strawberry_rolling_forecast_feature_effects": effects_df,
        "strawberry_rolling_forecast_training_history": training_history_df,
        "strawberry_rolling_forecast_metrics": metrics_df,
        "strawberry_rolling_forecast_metrics_by_origin": metrics_by_origin_df,
    }

    con = duckdb.connect(str(DB_PATH))

    for table_name, table_df in outputs.items():
        table_df.to_csv(paths[table_name], index=False)

        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.register("tmp_df", table_df)
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")

        print(f"Saved: {paths[table_name]}")

    con.close()

    print("\nRolling forecast metrics:")
    print(
        metrics_df[
            [
                "model_label",
                "n_predictions",
                "mae",
                "rmse",
                "r2",
                "bias",
                "mae_improvement_pct_vs_persistence",
            ]
        ]
    )

    print("\nTraining history:")
    print(training_history_df.head(10))


if __name__ == "__main__":
    main()
