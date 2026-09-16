#!/usr/bin/env python3
"""
Stress-test short-horizon strawberry fresh-matter forecasts with and without autoregressive fresh-matter history.

Purpose
-------
The preferred rolling model uses recent fresh-matter history, which is useful for short-horizon forecasts
but may lag behind rapid production flushes. This script tests that concern explicitly by comparing:

1. Rolling persistence baseline
2. Rolling crop/weather RF (full base feature set; includes current/recent fresh matter)
3. Rolling leading-indicator RF (excludes direct fresh-matter history)
4. Rolling leading-indicator + tagged-fruit RF (adds tagged fruit diameter/length context)
5. Rolling crop/weather + tagged-fruit RF (full base + tagged-fruit context)

The goal is not to blindly pick the most complex model. It is to check whether leading indicators can
anticipate flushes better than an autoregressive model that mostly follows recent production.

Inputs expected
---------------
data/processed/strawberry/strawberry_forecast_features.csv
    Created by scripts/02_build_forecast_features.py

data/processed/strawberry/strawberry_tagged_fruit_daily_features.csv
    Created by scripts/07_build_fruit_level_features.py

Outputs
-------
data/processed/strawberry/strawberry_rolling_leading_indicator_predictions.csv
    Row-level rolling predictions for all stress-test models.

data/processed/strawberry/strawberry_rolling_leading_indicator_model_metrics.csv
    Standard metrics by model.

data/processed/strawberry/strawberry_rolling_leading_indicator_error_audit.csv
    Error audit focused on underprediction and actual production flushes.

data/processed/strawberry/strawberry_rolling_leading_indicator_operational_board.csv
    Target-window aggregate forecasts with aligned target_date values, previous-window comparisons,
    error bands, and operational flags. This is designed to feed the Streamlit operational board.

data/processed/strawberry/strawberry_rolling_leading_indicator_feature_importance.csv
    Mean RF feature importance aggregated across rolling refits.

data/processed/strawberry/strawberry_rolling_leading_indicator_block_importance.csv
    RF importance grouped into interpretable feature blocks.

The outputs are also written to DuckDB tables with the same base names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed" / "strawberry"
DB_PATH = ROOT / "db" / "yield_forecasting.duckdb"

FORECAST_FEATURES_PATH = PROCESSED_DIR / "strawberry_forecast_features.csv"
TAGGED_FEATURES_PATH = PROCESSED_DIR / "strawberry_tagged_fruit_daily_features.csv"

TARGET_COL = "next_fresh_matter_g"
TEST_YEAR = 2023
RANDOM_STATE = 42
OPERATIONAL_CHANGE_THRESHOLD_PCT = 20.0


@dataclass(frozen=True)
class ModelSpec:
    model: str
    label: str
    feature_set: str
    features: list[str]
    model_type: str = "random_forest"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse mixed date formats robustly across pandas versions."""
    clean = series.astype("string").str.strip()
    try:
        return pd.to_datetime(clean, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(clean, errors="coerce")



def require_file(path: Path, how_to_create: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}\n{how_to_create}")



def make_one_hot_encoder() -> OneHotEncoder:
    """Handle scikit-learn versions before/after sparse_output."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)



def safe_pct_change(new_value: pd.Series | float, old_value: pd.Series | float) -> pd.Series:
    old = pd.to_numeric(old_value, errors="coerce")
    new = pd.to_numeric(new_value, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (new - old) / old.abs() * 100.0
    if isinstance(out, pd.Series):
        return out.replace([np.inf, -np.inf], np.nan)
    return pd.Series([out]).replace([np.inf, -np.inf], np.nan)



def coerce_numeric_frame(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out



def is_problematic_tagged_input(col: str) -> bool:
    """Exclude sparse/destructive tagged-fruit inputs from all feature sets."""
    lower = col.lower()
    return (
        "tagged_fruit_fresh_matter" in lower
        or "tagged_fresh_matter" in lower
        or "tagged_fruit_lifespan" in lower
        or "tagged_lifespan" in lower
    )



def is_direct_fresh_matter_history(col: str) -> bool:
    """Identify direct autoregressive target-history features to exclude in the leading-indicator model.

    We exclude total fresh-matter history at the treatment/date level because this is very close to the
    target and may make the model follow recent production rather than anticipate production flushes.

    We intentionally keep individual fruit weight/size sample columns, because those are fruit-development
    signals rather than the treatment-level target itself.
    """
    lower = col.lower()

    # Keep fruit-level size/weight proxies; they are possible leading indicators.
    if "individual" in lower or "fruit_weight" in lower or "fresh_weight" in lower:
        return False

    direct_names = {
        "fresh_matter_g",
        "current_fresh_matter_g",
        "previous_fresh_matter_g",
        "change_fresh_matter_g",
        "rolling_mean_2_fresh_matter_g",
        "rolling_mean_3_fresh_matter_g",
        "rolling_std_2_fresh_matter_g",
        "rolling_std_3_fresh_matter_g",
    }
    if lower in direct_names:
        return True

    return "fresh_matter_g" in lower and (
        lower.startswith("current_")
        or lower.startswith("previous_")
        or lower.startswith("change_")
        or lower.startswith("rolling_")
    )


# ---------------------------------------------------------------------------
# Data loading / feature construction
# ---------------------------------------------------------------------------


def load_base_forecast_features() -> pd.DataFrame:
    require_file(
        FORECAST_FEATURES_PATH,
        "Run: python scripts/02_build_forecast_features.py",
    )

    df = pd.read_csv(FORECAST_FEATURES_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in {FORECAST_FEATURES_PATH}")

    if "current_observation_date" not in df.columns:
        if "date" in df.columns:
            df = df.rename(columns={"date": "current_observation_date"})
        else:
            raise ValueError("Could not find 'current_observation_date' or 'date' in forecast features.")

    df["current_observation_date"] = parse_dates(df["current_observation_date"])
    if "target_date" in df.columns:
        df["target_date"] = parse_dates(df["target_date"])

    if "year" not in df.columns:
        df["year"] = df["current_observation_date"].dt.year

    # Safe calendar features known at the forecast origin date.
    df["origin_day_of_year"] = df["current_observation_date"].dt.dayofyear
    df["origin_month"] = df["current_observation_date"].dt.month
    df["origin_week_of_year"] = df["current_observation_date"].dt.isocalendar().week.astype(int)

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    return df.dropna(subset=["current_observation_date", "year", TARGET_COL]).copy()



def load_tagged_fruit_features() -> pd.DataFrame:
    require_file(
        TAGGED_FEATURES_PATH,
        "Run: python scripts/07_build_fruit_level_features.py",
    )

    tagged = pd.read_csv(TAGGED_FEATURES_PATH)
    if tagged.empty:
        raise ValueError(f"Tagged fruit feature file is empty: {TAGGED_FEATURES_PATH}")

    if "date" not in tagged.columns:
        raise ValueError(f"Expected a 'date' column in {TAGGED_FEATURES_PATH}")

    tagged = tagged.copy()
    tagged["date"] = parse_dates(tagged["date"])
    if "year" not in tagged.columns:
        tagged["year"] = tagged["date"].dt.year

    safe_tagged_cols = [
        c
        for c in tagged.columns
        if (c.startswith("tagged_diameter_mm_") or c.startswith("tagged_length_mm_"))
        and not is_problematic_tagged_input(c)
    ]

    keep = ["year", "date"] + safe_tagged_cols
    tagged = tagged[keep].dropna(subset=["date", "year"]).copy()
    tagged = tagged.rename(columns={c: f"fruit_{c}" for c in safe_tagged_cols})
    return tagged



def join_tagged_features(forecast: pd.DataFrame, tagged: pd.DataFrame) -> pd.DataFrame:
    df = forecast.copy()
    tagged = tagged.copy()

    df["join_date"] = df["current_observation_date"].dt.normalize()
    tagged["join_date"] = tagged["date"].dt.normalize()

    merged = df.merge(
        tagged.drop(columns=["date"]),
        on=["year", "join_date"],
        how="left",
        validate="many_to_one",
    )
    return merged.drop(columns=["join_date"])


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------


def is_forbidden_feature(col: str) -> bool:
    """Avoid target leakage and non-feature identifiers."""
    lower = col.lower()

    exact_forbidden = {
        TARGET_COL,
        "target_date",
        "current_observation_date",
        "date",
        "join_date",
        "source_file",
        "model",
        "model_label",
        "prediction",
        "predicted_next_fresh_matter_g",
        "observed_next_fresh_matter_g",
        "error_g",
        "absolute_error_g",
        "relative_error_pct",
    }
    if col in exact_forbidden:
        return True

    if lower == "forecast_horizon_days":
        return False

    if is_problematic_tagged_input(col):
        return True

    forbidden_patterns = [
        "next_",          # future target/state columns
        "target_",        # target metadata except forecast_horizon
        "observed_",
        "predicted_",
        "error",
        "mae",
        "rmse",
        "r2",
        "bias",
    ]
    return any(pattern in lower for pattern in forbidden_patterns)



def get_numeric_and_categorical_features(df: pd.DataFrame, candidate_cols: list[str]) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for col in candidate_cols:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            numeric.append(col)
        else:
            n_unique = df[col].nunique(dropna=True)
            if n_unique <= 20:
                categorical.append(col)
    return numeric, categorical



def select_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    tagged_cols = [c for c in df.columns if c.startswith("fruit_tagged_")]

    base_candidates = [
        c
        for c in df.columns
        if not is_forbidden_feature(c)
        and not c.startswith("fruit_tagged_")
        and c not in {"year"}
    ]
    if "nitrogen_treatment" in df.columns and "nitrogen_treatment" not in base_candidates:
        base_candidates.append("nitrogen_treatment")

    leading_candidates = [
        c for c in base_candidates if not is_direct_fresh_matter_history(c)
    ]

    return {
        "base_crop_weather": sorted(dict.fromkeys(base_candidates)),
        "leading_indicators": sorted(dict.fromkeys(leading_candidates)),
        "leading_plus_tagged": sorted(dict.fromkeys(leading_candidates + tagged_cols)),
        "base_plus_tagged": sorted(dict.fromkeys(base_candidates + tagged_cols)),
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def build_rf_pipeline(train_df: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
    numeric_cols, categorical_cols = get_numeric_and_categorical_features(train_df, feature_cols)

    transformers = []
    if numeric_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        raise ValueError("No usable features were selected for the RF pipeline.")

    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")
    model = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])



def get_transformed_feature_names(pipeline: Pipeline) -> list[str]:
    preprocess: ColumnTransformer = pipeline.named_steps["preprocess"]
    try:
        return [str(x) for x in preprocess.get_feature_names_out()]
    except Exception:
        names: list[str] = []
        for name, transformer, cols in preprocess.transformers_:
            if name == "remainder":
                continue
            if name == "num":
                names.extend([f"num__{c}" for c in cols])
            elif name == "cat":
                try:
                    onehot = transformer.named_steps["onehot"]
                    encoded = onehot.get_feature_names_out(cols)
                    names.extend([f"cat__{x}" for x in encoded])
                except Exception:
                    names.extend([f"cat__{c}" for c in cols])
        return names



def persistence_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    current_col = next((c for c in ["current_fresh_matter_g", "fresh_matter_g"] if c in test_df.columns), None)
    if current_col is not None:
        pred = pd.to_numeric(test_df[current_col], errors="coerce")
    else:
        pred = pd.Series(np.nan, index=test_df.index, dtype="float64")

    global_mean = float(pd.to_numeric(train_df[TARGET_COL], errors="coerce").mean())
    if "nitrogen_treatment" in train_df.columns and "nitrogen_treatment" in test_df.columns:
        treatment_means = train_df.groupby("nitrogen_treatment")[TARGET_COL].mean()
        fallback = test_df["nitrogen_treatment"].map(treatment_means).fillna(global_mean)
    else:
        fallback = pd.Series(global_mean, index=test_df.index)

    return pred.fillna(fallback).to_numpy(dtype=float)



def treatment_mean_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    global_mean = float(pd.to_numeric(train_df[TARGET_COL], errors="coerce").mean())
    if "nitrogen_treatment" in train_df.columns and "nitrogen_treatment" in test_df.columns:
        treatment_means = train_df.groupby("nitrogen_treatment")[TARGET_COL].mean()
        pred = test_df["nitrogen_treatment"].map(treatment_means).fillna(global_mean)
    else:
        pred = pd.Series(global_mean, index=test_df.index)
    return pred.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Rolling validation
# ---------------------------------------------------------------------------


def run_rolling_validation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_sets = select_feature_sets(df)

    model_specs = [
        ModelSpec(
            model="rolling_base_crop_weather_rf",
            label="Rolling crop/weather RF",
            feature_set="base_crop_weather",
            features=feature_sets["base_crop_weather"],
        ),
        ModelSpec(
            model="rolling_leading_indicator_rf",
            label="Rolling leading-indicator RF",
            feature_set="leading_indicators",
            features=feature_sets["leading_indicators"],
        ),
        ModelSpec(
            model="rolling_leading_indicator_tagged_rf",
            label="Rolling leading-indicator + tagged-fruit RF",
            feature_set="leading_plus_tagged",
            features=feature_sets["leading_plus_tagged"],
        ),
        ModelSpec(
            model="rolling_base_plus_tagged_fruit_rf",
            label="Rolling crop/weather + tagged-fruit RF",
            feature_set="base_plus_tagged",
            features=feature_sets["base_plus_tagged"],
        ),
    ]

    predictions: list[pd.DataFrame] = []
    importance_rows: list[dict[str, object]] = []

    test_origins = sorted(
        df.loc[df["year"].astype(int) == TEST_YEAR, "current_observation_date"].dropna().unique()
    )
    if not test_origins:
        raise ValueError(f"No forecast origins found for test year {TEST_YEAR}.")

    for origin in test_origins:
        origin = pd.Timestamp(origin)
        train_mask = (df["year"].astype(int) < TEST_YEAR) | (
            (df["year"].astype(int) == TEST_YEAR)
            & (df["current_observation_date"] < origin)
        )
        test_mask = (df["year"].astype(int) == TEST_YEAR) & (
            df["current_observation_date"] == origin
        )

        train_df = df.loc[train_mask].dropna(subset=[TARGET_COL]).copy()
        test_df = df.loc[test_mask].dropna(subset=[TARGET_COL]).copy()
        if train_df.empty or test_df.empty:
            continue

        n_prev_year = int((train_df["year"].astype(int) < TEST_YEAR).sum())
        n_current_year = int((train_df["year"].astype(int) == TEST_YEAR).sum())

        baseline_specs = [
            (
                "rolling_persistence",
                "Rolling persistence baseline",
                "baseline",
                persistence_predictions(train_df, test_df),
                0,
                0,
            ),
            (
                "rolling_treatment_mean",
                "Rolling treatment mean baseline",
                "baseline",
                treatment_mean_predictions(train_df, test_df),
                0,
                0,
            ),
        ]

        for model, label, feature_set, pred, n_features, n_tagged_features in baseline_specs:
            predictions.append(
                build_prediction_frame(
                    test_df=test_df,
                    pred=pred,
                    model=model,
                    label=label,
                    feature_set=feature_set,
                    n_features=n_features,
                    n_training_rows=len(train_df),
                    n_previous_year_training_rows=n_prev_year,
                    n_current_year_training_rows=n_current_year,
                    n_tagged_features=n_tagged_features,
                )
            )

        for spec in model_specs:
            features = [f for f in spec.features if f in train_df.columns]
            if not features:
                continue

            train_model_df = train_df[features + [TARGET_COL]].copy()
            test_model_df = test_df[features + [TARGET_COL]].copy()

            pipeline = build_rf_pipeline(train_model_df, features)
            pipeline.fit(train_model_df[features], train_model_df[TARGET_COL])
            pred = pipeline.predict(test_model_df[features])

            n_tagged_features = sum(1 for f in features if f.startswith("fruit_tagged_"))
            predictions.append(
                build_prediction_frame(
                    test_df=test_df,
                    pred=pred,
                    model=spec.model,
                    label=spec.label,
                    feature_set=spec.feature_set,
                    n_features=len(features),
                    n_training_rows=len(train_df),
                    n_previous_year_training_rows=n_prev_year,
                    n_current_year_training_rows=n_current_year,
                    n_tagged_features=n_tagged_features,
                )
            )

            transformed_names = get_transformed_feature_names(pipeline)
            importances = pipeline.named_steps["model"].feature_importances_
            if len(transformed_names) == len(importances):
                for feature_name, importance in zip(transformed_names, importances):
                    importance_rows.append(
                        {
                            "model": spec.model,
                            "model_label": spec.label,
                            "feature_set": spec.feature_set,
                            "forecast_origin_date": origin.date().isoformat(),
                            "feature": feature_name,
                            "feature_block": infer_feature_block(feature_name),
                            "importance": float(importance),
                            "n_training_rows": len(train_df),
                        }
                    )

    if not predictions:
        raise RuntimeError("No rolling predictions were generated.")

    predictions_df = pd.concat(predictions, ignore_index=True)
    metrics_df = calculate_metrics(predictions_df)
    importance_df = pd.DataFrame(importance_rows)
    return predictions_df, metrics_df, importance_df



def build_prediction_frame(
    test_df: pd.DataFrame,
    pred: np.ndarray,
    model: str,
    label: str,
    feature_set: str,
    n_features: int,
    n_training_rows: int,
    n_previous_year_training_rows: int,
    n_current_year_training_rows: int,
    n_tagged_features: int,
) -> pd.DataFrame:
    out_cols = [
        c
        for c in [
            "year",
            "nitrogen_treatment",
            "current_observation_date",
            "target_date",
            "forecast_horizon_days",
            "current_fresh_matter_g",
            TARGET_COL,
        ]
        if c in test_df.columns
    ]
    out = test_df[out_cols].copy()
    out = out.rename(columns={TARGET_COL: "observed_next_fresh_matter_g"})
    out["predicted_next_fresh_matter_g"] = pred.astype(float)
    out["model"] = model
    out["model_label"] = label
    out["feature_set"] = feature_set
    out["model_type"] = "baseline" if feature_set == "baseline" else "random_forest"
    out["n_features"] = n_features
    out["n_tagged_features"] = n_tagged_features
    out["n_training_rows"] = n_training_rows
    out["n_previous_year_training_rows"] = n_previous_year_training_rows
    out["n_current_year_training_rows"] = n_current_year_training_rows

    out["error_g"] = out["predicted_next_fresh_matter_g"] - out["observed_next_fresh_matter_g"]
    out["absolute_error_g"] = out["error_g"].abs()
    out["relative_error_pct"] = safe_pct_change(
        out["predicted_next_fresh_matter_g"],
        out["observed_next_fresh_matter_g"],
    ).to_numpy()

    if "current_fresh_matter_g" in out.columns:
        out["predicted_change_vs_current_g"] = out["predicted_next_fresh_matter_g"] - out["current_fresh_matter_g"]
        out["predicted_change_vs_current_pct"] = safe_pct_change(
            out["predicted_next_fresh_matter_g"],
            out["current_fresh_matter_g"],
        ).to_numpy()
        out["actual_change_vs_current_g"] = out["observed_next_fresh_matter_g"] - out["current_fresh_matter_g"]
        out["actual_change_vs_current_pct"] = safe_pct_change(
            out["observed_next_fresh_matter_g"],
            out["current_fresh_matter_g"],
        ).to_numpy()
        out["operational_flag"] = np.select(
            [
                out["predicted_change_vs_current_pct"] >= OPERATIONAL_CHANGE_THRESHOLD_PCT,
                out["predicted_change_vs_current_pct"] <= -OPERATIONAL_CHANGE_THRESHOLD_PCT,
            ],
            ["Production flush risk", "Supply drop risk"],
            default="Stable window",
        )
        out["actual_flush_vs_current"] = out["actual_change_vs_current_pct"] >= OPERATIONAL_CHANGE_THRESHOLD_PCT
    else:
        out["operational_flag"] = "Not assessed"
        out["actual_flush_vs_current"] = False

    return out



def calculate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["model", "model_label", "feature_set", "model_type"]
    for (model, label, feature_set, model_type), part in predictions.groupby(group_cols, dropna=False):
        y_true = pd.to_numeric(part["observed_next_fresh_matter_g"], errors="coerce")
        y_pred = pd.to_numeric(part["predicted_next_fresh_matter_g"], errors="coerce")
        valid = y_true.notna() & y_pred.notna()
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        if y_true.empty:
            continue

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
        bias = float((y_pred - y_true).mean())
        mean_observed = float(y_true.mean())
        mae_pct = float(mae / mean_observed * 100) if mean_observed else np.nan

        rows.append(
            {
                "model": model,
                "model_label": label,
                "feature_set": feature_set,
                "model_type": model_type,
                "forecast_mode": "rolling_leading_indicator_stress_test",
                "test_year": TEST_YEAR,
                "n_predictions": int(len(y_true)),
                "n_unique_origin_dates": int(part["current_observation_date"].nunique()),
                "n_features_median": float(part["n_features"].median()),
                "n_tagged_features_median": float(part["n_tagged_features"].median()),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "bias": bias,
                "mean_observed": mean_observed,
                "mae_pct_of_mean_observed": mae_pct,
            }
        )

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics

    baseline = metrics.loc[metrics["model"] == "rolling_persistence", "mae"]
    if not baseline.empty:
        baseline_mae = float(baseline.iloc[0])
        metrics["mae_improvement_vs_persistence"] = baseline_mae - metrics["mae"]
        metrics["mae_improvement_pct_vs_persistence"] = (baseline_mae - metrics["mae"]) / baseline_mae * 100.0
    else:
        metrics["mae_improvement_vs_persistence"] = np.nan
        metrics["mae_improvement_pct_vs_persistence"] = np.nan

    return metrics.sort_values("mae", ascending=True).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Error and operational audits
# ---------------------------------------------------------------------------


def add_model_mae(predictions: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    cols = ["model", "mae", "rmse", "r2", "bias", "mae_improvement_pct_vs_persistence"]
    return predictions.merge(metrics[cols], on="model", how="left", validate="many_to_one")



def build_error_audit(predictions: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    pred = add_model_mae(predictions, metrics)
    pred["large_underprediction"] = pred["error_g"] < -pred["mae"]
    pred["large_overprediction"] = pred["error_g"] > pred["mae"]
    pred["underpredicted_actual_flush"] = pred["actual_flush_vs_current"] & pred["large_underprediction"]

    rows = []
    for (model, label), part in pred.groupby(["model", "model_label"], dropna=False):
        n = len(part)
        rows.append(
            {
                "model": model,
                "model_label": label,
                "n_predictions": int(n),
                "mae": float(part["mae"].iloc[0]) if n else np.nan,
                "n_large_underpredictions": int(part["large_underprediction"].sum()),
                "n_large_overpredictions": int(part["large_overprediction"].sum()),
                "n_actual_flush_rows": int(part["actual_flush_vs_current"].sum()),
                "n_underpredicted_actual_flush_rows": int(part["underpredicted_actual_flush"].sum()),
                "share_underpredicted_actual_flush_rows_pct": float(part["underpredicted_actual_flush"].mean() * 100.0) if n else np.nan,
                "mean_error_on_actual_flush_rows_g": float(part.loc[part["actual_flush_vs_current"], "error_g"].mean()) if part["actual_flush_vs_current"].any() else np.nan,
                "mean_abs_error_on_actual_flush_rows_g": float(part.loc[part["actual_flush_vs_current"], "absolute_error_g"].mean()) if part["actual_flush_vs_current"].any() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("mae", ascending=True).reset_index(drop=True)



def build_operational_board(predictions: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    pred = add_model_mae(predictions, metrics).copy()
    pred["target_date"] = parse_dates(pred["target_date"])
    pred = pred.dropna(subset=["target_date"])

    grouped = (
        pred.groupby(["model", "model_label", "target_date"], dropna=False)
        .agg(
            forecast_fresh_matter_g=("predicted_next_fresh_matter_g", "sum"),
            observed_fresh_matter_g=("observed_next_fresh_matter_g", "sum"),
            n_treatments=("nitrogen_treatment", "nunique"),
            avg_forecast_horizon_days=("forecast_horizon_days", "mean"),
            row_level_mae_g=("mae", "first"),
            row_level_rmse_g=("rmse", "first"),
            row_level_r2=("r2", "first"),
            row_level_bias_g=("bias", "first"),
            row_level_improvement_pct_vs_persistence=("mae_improvement_pct_vs_persistence", "first"),
        )
        .reset_index()
        .sort_values(["model", "target_date"])
    )

    # Compute aggregate-level errors and aggregate MAE by model. This is better for the operational
    # board than using row-level MAE around summed treatment-level forecasts.
    grouped["aggregate_error_g"] = grouped["forecast_fresh_matter_g"] - grouped["observed_fresh_matter_g"]
    grouped["aggregate_absolute_error_g"] = grouped["aggregate_error_g"].abs()
    aggregate_mae = (
        grouped.groupby("model", dropna=False)["aggregate_absolute_error_g"]
        .mean()
        .rename("aggregate_mae_g")
        .reset_index()
    )
    grouped = grouped.merge(aggregate_mae, on="model", how="left", validate="many_to_one")
    grouped["forecast_lower_g"] = (grouped["forecast_fresh_matter_g"] - grouped["aggregate_mae_g"]).clip(lower=0)
    grouped["forecast_upper_g"] = grouped["forecast_fresh_matter_g"] + grouped["aggregate_mae_g"]

    board_parts = []
    for model, part in grouped.groupby("model", dropna=False):
        part = part.sort_values("target_date").copy()
        part["previous_observed_fresh_matter_g"] = part["observed_fresh_matter_g"].shift(1)
        part["forecast_change_vs_previous_observed_g"] = part["forecast_fresh_matter_g"] - part["previous_observed_fresh_matter_g"]
        part["observed_change_vs_previous_observed_g"] = part["observed_fresh_matter_g"] - part["previous_observed_fresh_matter_g"]
        part["forecast_change_vs_previous_observed_pct"] = safe_pct_change(
            part["forecast_fresh_matter_g"], part["previous_observed_fresh_matter_g"]
        ).to_numpy()
        part["observed_change_vs_previous_observed_pct"] = safe_pct_change(
            part["observed_fresh_matter_g"], part["previous_observed_fresh_matter_g"]
        ).to_numpy()
        part["operational_flag"] = np.select(
            [
                part["forecast_change_vs_previous_observed_pct"] >= OPERATIONAL_CHANGE_THRESHOLD_PCT,
                part["forecast_change_vs_previous_observed_pct"] <= -OPERATIONAL_CHANGE_THRESHOLD_PCT,
            ],
            ["Production flush risk", "Supply drop risk"],
            default="Stable window",
        )
        part["actual_flush_window"] = part["observed_change_vs_previous_observed_pct"] >= OPERATIONAL_CHANGE_THRESHOLD_PCT
        part["outside_aggregate_mae_band"] = ~part["observed_fresh_matter_g"].between(part["forecast_lower_g"], part["forecast_upper_g"])
        part["underpredicted_flush_window"] = part["actual_flush_window"] & (part["aggregate_error_g"] < -part["aggregate_mae_g"])
        board_parts.append(part)

    board = pd.concat(board_parts, ignore_index=True)
    board["target_date"] = pd.to_datetime(board["target_date"]).dt.date.astype(str)
    return board.sort_values(["model", "target_date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Importance summaries
# ---------------------------------------------------------------------------


def infer_feature_block(feature_name: str) -> str:
    lower = feature_name.lower().replace("num__", "").replace("cat__", "")
    if "fruit_tagged" in lower or "tagged_diameter" in lower or "tagged_length" in lower:
        return "tagged fruit development"
    if "nitrogen_treatment" in lower or "treatment" in lower:
        return "nitrogen treatment"
    if any(x in lower for x in ["gdd", "radiation", "solar", "tmean", "tmax", "tmin", "rh", "weather"]):
        return "weather / thermal time"
    if any(x in lower for x in ["horizon", "day_of_year", "month", "week"]):
        return "calendar / forecast horizon"
    if any(x in lower for x in ["rolling", "previous", "lag", "change"]):
        return "recent crop trajectory"
    if any(x in lower for x in ["fruit_number", "dry_matter", "biomass"]):
        return "current crop state"
    if any(x in lower for x in ["diameter", "length", "weight", "condition"]):
        return "fruit size sample"
    if "fresh_matter" in lower:
        return "fresh matter history"
    return "other"



def aggregate_feature_importance(importance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if importance.empty:
        return importance, pd.DataFrame()

    feature_summary = (
        importance.groupby(["model", "model_label", "feature_set", "feature", "feature_block"], dropna=False)
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            n_rolling_fits=("forecast_origin_date", "nunique"),
        )
        .reset_index()
        .sort_values(["model", "mean_importance"], ascending=[True, False])
    )

    block_summary = (
        feature_summary.groupby(["model", "model_label", "feature_set", "feature_block"], dropna=False)
        .agg(
            total_mean_importance=("mean_importance", "sum"),
            n_features_in_block=("feature", "nunique"),
            n_rolling_fits=("n_rolling_fits", "max"),
        )
        .reset_index()
    )
    totals = block_summary.groupby("model")["total_mean_importance"].transform("sum")
    block_summary["share_of_model_importance_pct"] = np.where(
        totals > 0,
        block_summary["total_mean_importance"] / totals * 100.0,
        np.nan,
    )
    return block_summary.sort_values(["model", "share_of_model_importance_pct"], ascending=[True, False]), feature_summary


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_outputs(outputs: dict[str, pd.DataFrame]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for filename, df in outputs.items():
        df.to_csv(PROCESSED_DIR / filename, index=False)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    try:
        for filename, df in outputs.items():
            table_name = filename.replace(".csv", "")
            con.register("tmp_df", df)
            con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM tmp_df")
            con.unregister("tmp_df")
    finally:
        con.close()



def print_summary(
    metrics: pd.DataFrame,
    error_audit: pd.DataFrame,
    block_importance: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> None:
    print("Leading-indicator stress test complete.")
    print(f"Forecast features: {FORECAST_FEATURES_PATH}")
    print(f"Tagged-fruit features: {TAGGED_FEATURES_PATH}")

    print("\nFeature set sizes:")
    for name, cols in feature_sets.items():
        n_direct_fresh = sum(is_direct_fresh_matter_history(c) for c in cols)
        n_tagged = sum(c.startswith("fruit_tagged_") for c in cols)
        print(f"  {name}: {len(cols)} features | direct fresh-matter history={n_direct_fresh} | tagged={n_tagged}")

    if not metrics.empty:
        cols = [
            "model_label",
            "n_predictions",
            "n_features_median",
            "n_tagged_features_median",
            "mae",
            "rmse",
            "r2",
            "bias",
            "mae_improvement_pct_vs_persistence",
        ]
        print("\nModel comparison:")
        print(metrics[cols].to_string(index=False))

    if not error_audit.empty:
        cols = [
            "model_label",
            "n_large_underpredictions",
            "n_actual_flush_rows",
            "n_underpredicted_actual_flush_rows",
            "mean_abs_error_on_actual_flush_rows_g",
        ]
        print("\nFlush / underprediction audit:")
        print(error_audit[cols].to_string(index=False))

    if not block_importance.empty:
        print("\nTop feature blocks by model:")
        preview_cols = [
            "model_label",
            "feature_block",
            "share_of_model_importance_pct",
            "n_features_in_block",
        ]
        print(block_importance[preview_cols].head(30).to_string(index=False))

    print("\nOutputs written to:")
    for filename in [
        "strawberry_rolling_leading_indicator_predictions.csv",
        "strawberry_rolling_leading_indicator_model_metrics.csv",
        "strawberry_rolling_leading_indicator_error_audit.csv",
        "strawberry_rolling_leading_indicator_operational_board.csv",
        "strawberry_rolling_leading_indicator_feature_importance.csv",
        "strawberry_rolling_leading_indicator_block_importance.csv",
    ]:
        print(f"- {PROCESSED_DIR / filename}")



def main() -> None:
    forecast = load_base_forecast_features()
    tagged = load_tagged_fruit_features()
    data = join_tagged_features(forecast, tagged)
    tagged_cols = [c for c in data.columns if c.startswith("fruit_tagged_")]
    data = coerce_numeric_frame(data, tagged_cols)

    feature_sets = select_feature_sets(data)
    predictions, metrics, raw_importance = run_rolling_validation(data)
    error_audit = build_error_audit(predictions, metrics)
    operational_board = build_operational_board(predictions, metrics)
    block_importance, feature_importance = aggregate_feature_importance(raw_importance)

    outputs = {
        "strawberry_rolling_leading_indicator_predictions.csv": predictions,
        "strawberry_rolling_leading_indicator_model_metrics.csv": metrics,
        "strawberry_rolling_leading_indicator_error_audit.csv": error_audit,
        "strawberry_rolling_leading_indicator_operational_board.csv": operational_board,
        "strawberry_rolling_leading_indicator_feature_importance.csv": feature_importance,
        "strawberry_rolling_leading_indicator_block_importance.csv": block_importance,
    }
    write_outputs(outputs)
    print_summary(metrics, error_audit, block_importance, feature_sets)


if __name__ == "__main__":
    main()

