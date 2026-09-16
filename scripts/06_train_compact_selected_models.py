#!/usr/bin/env python3
"""
Train sequential compact strawberry forecasting models from audited feature sets.

Run order:
    python scripts/10_feature_audit_and_selection.py
    python scripts/11_train_compact_selected_models.py

This script trains rolling in-season models for 2023:
- Each forecast date trains on all 2022 rows plus 2023 rows strictly before that date.
- The target is next_fresh_matter_g.
- Models are defined by the audited feature sets produced by script 10.
- The final recommendation favours the simplest competitive model, not simply the
  most complex model with the lowest MAE.
"""

from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import duckdb
except Exception:  # pragma: no cover
    duckdb = None

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed" / "strawberry"
DB_PATH = ROOT / "db" / "yield_forecasting.duckdb"

FORECAST_FEATURES_PATH = PROCESSED_DIR / "strawberry_forecast_features.csv"
TAGGED_FEATURES_PATH = PROCESSED_DIR / "strawberry_tagged_fruit_daily_features.csv"
TREATMENT_FRUIT_FEATURES_PATH = PROCESSED_DIR / "strawberry_fruit_size_treatment_daily_features.csv"
SELECTED_MODEL_FEATURES_PATH = PROCESSED_DIR / "strawberry_selected_model_feature_sets.csv"
SELECTED_COMPACT_FEATURES_PATH = PROCESSED_DIR / "strawberry_selected_compact_feature_set.csv"

TARGET_COL = "next_fresh_matter_g"
TEST_YEAR = 2023
RANDOM_STATE = 42
PREFERRED_COMPETITIVE_MARGIN_G = 0.50
PREFERRED_COMPETITIVE_MARGIN_PCT_OF_BEST = 0.02


# ---------------------------------------------------------------------------
# Shared loading / feature engineering from script 10
# ---------------------------------------------------------------------------


def require_file(path: Path, how_to_create: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}\n{how_to_create}")



def parse_dates(series: pd.Series) -> pd.Series:
    clean = series.astype("string").str.strip()
    try:
        return pd.to_datetime(clean, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(clean, errors="coerce")



def load_forecast_features() -> pd.DataFrame:
    require_file(FORECAST_FEATURES_PATH, "Run: python scripts/02_build_forecast_features.py")
    df = pd.read_csv(FORECAST_FEATURES_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in {FORECAST_FEATURES_PATH}")
    if "current_observation_date" not in df.columns:
        if "date" in df.columns:
            df = df.rename(columns={"date": "current_observation_date"})
        else:
            raise ValueError("Expected 'current_observation_date' or 'date' in forecast features.")
    df["current_observation_date"] = parse_dates(df["current_observation_date"])
    if "target_date" in df.columns:
        df["target_date"] = parse_dates(df["target_date"])
    if "year" not in df.columns:
        df["year"] = df["current_observation_date"].dt.year
    df["origin_day_of_year"] = df["current_observation_date"].dt.dayofyear
    df["origin_month"] = df["current_observation_date"].dt.month
    df["origin_week_of_year"] = df["current_observation_date"].dt.isocalendar().week.astype("Int64")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    return df.dropna(subset=["current_observation_date", "year", TARGET_COL]).copy()



def load_treatment_fruit_features_optional() -> pd.DataFrame:
    if not TREATMENT_FRUIT_FEATURES_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(TREATMENT_FRUIT_FEATURES_PATH)
    if df.empty or not {"year", "date", "nitrogen_treatment"}.issubset(df.columns):
        return pd.DataFrame()
    df = df.copy()
    df["date"] = parse_dates(df["date"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    drop_cols = [c for c in ["source_file", "measurement_type"] if c in df.columns]
    return df.drop(columns=drop_cols).dropna(subset=["year", "date", "nitrogen_treatment"])



def join_treatment_fruit_features(forecast: pd.DataFrame, treatment: pd.DataFrame) -> pd.DataFrame:
    if treatment.empty:
        return forecast.copy()
    df = forecast.copy()
    df["join_date"] = df["current_observation_date"].dt.normalize()
    tmp = treatment.copy()
    tmp["join_date"] = tmp["date"].dt.normalize()
    tmp = tmp.drop(columns=["date"])
    overlapping = [c for c in tmp.columns if c in df.columns and c not in {"year", "join_date", "nitrogen_treatment"}]
    tmp = tmp.rename(columns={c: f"script07_{c}" for c in overlapping})
    merged = df.merge(tmp, on=["year", "join_date", "nitrogen_treatment"], how="left", validate="many_to_one")
    return merged.drop(columns=["join_date"])



def load_tagged_features_optional() -> pd.DataFrame:
    if not TAGGED_FEATURES_PATH.exists():
        return pd.DataFrame()
    tagged = pd.read_csv(TAGGED_FEATURES_PATH)
    if tagged.empty or "date" not in tagged.columns:
        return pd.DataFrame()
    tagged = tagged.copy()
    tagged["date"] = parse_dates(tagged["date"])
    if "year" not in tagged.columns:
        tagged["year"] = tagged["date"].dt.year
    safe_cols = []
    for col in tagged.columns:
        lower = col.lower()
        if (lower.startswith("tagged_diameter_mm_") or lower.startswith("tagged_length_mm_")) and "fresh_matter" not in lower and "lifespan" not in lower:
            safe_cols.append(col)
    keep = ["year", "date"] + safe_cols
    tagged = tagged[keep].dropna(subset=["year", "date"]).copy()
    tagged = tagged.rename(columns={c: f"fruit_{c}" for c in safe_cols})
    return tagged



def join_tagged_features(forecast: pd.DataFrame, tagged: pd.DataFrame) -> pd.DataFrame:
    if tagged.empty:
        return forecast.copy()
    df = forecast.copy()
    df["join_date"] = df["current_observation_date"].dt.normalize()
    tmp = tagged.copy()
    tmp["join_date"] = tmp["date"].dt.normalize()
    tmp = tmp.drop(columns=["date"])
    merged = df.merge(tmp, on=["year", "join_date"], how="left", validate="many_to_one")
    return merged.drop(columns=["join_date"])



def numeric_col(df: pd.DataFrame, name: str) -> pd.Series | None:
    if name not in df.columns:
        return None
    return pd.to_numeric(df[name], errors="coerce")



def first_existing(columns: Iterable[str], df: pd.DataFrame) -> str | None:
    for col in columns:
        if col in df.columns:
            return col
    return None



def add_compact_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    current_fresh = numeric_col(out, "current_fresh_matter_g")
    previous_fresh = numeric_col(out, "previous_fresh_matter_g")
    change_fresh = numeric_col(out, "change_fresh_matter_g")
    if current_fresh is not None and previous_fresh is not None:
        out["fresh_matter_trend_g"] = current_fresh - previous_fresh
    elif change_fresh is not None:
        out["fresh_matter_trend_g"] = change_fresh
    for src in ["rolling_mean_3_fresh_matter_g", "rolling_mean_2_fresh_matter_g"]:
        if src in out.columns:
            out["fresh_matter_recent_mean_g"] = pd.to_numeric(out[src], errors="coerce")
            break

    current_fruit = numeric_col(out, "current_fruit_number")
    previous_fruit = numeric_col(out, "previous_fruit_number")
    change_fruit = numeric_col(out, "change_fruit_number")
    if current_fruit is not None and previous_fruit is not None:
        out["fruit_number_trend"] = current_fruit - previous_fruit
    elif change_fruit is not None:
        out["fruit_number_trend"] = change_fruit
    for src in ["rolling_mean_3_fruit_number", "rolling_mean_2_fruit_number"]:
        if src in out.columns:
            out["fruit_number_recent_mean"] = pd.to_numeric(out[src], errors="coerce")
            break

    current_dry = numeric_col(out, "current_dry_matter_g")
    previous_dry = numeric_col(out, "previous_dry_matter_g")
    change_dry = numeric_col(out, "change_dry_matter_g")
    if current_dry is not None and previous_dry is not None:
        out["dry_matter_trend_g"] = current_dry - previous_dry
    elif change_dry is not None:
        out["dry_matter_trend_g"] = change_dry
    for src in ["rolling_mean_3_dry_matter_g", "rolling_mean_2_dry_matter_g"]:
        if src in out.columns:
            out["dry_matter_recent_mean_g"] = pd.to_numeric(out[src], errors="coerce")
            break

    diameter_current = first_existing(
        ["fruit_tagged_diameter_mm_median", "fruit_tagged_diameter_mm_mean", "fruit_tagged_diameter_median", "fruit_tagged_diameter_mean"],
        out,
    )
    length_current = first_existing(
        ["fruit_tagged_length_mm_median", "fruit_tagged_length_mm_mean", "fruit_tagged_length_median", "fruit_tagged_length_mean"],
        out,
    )
    if diameter_current:
        out["tagged_fruit_diameter_current_mm"] = pd.to_numeric(out[diameter_current], errors="coerce")
    if length_current:
        out["tagged_fruit_length_current_mm"] = pd.to_numeric(out[length_current], errors="coerce")

    diameter_trend = first_existing([c for c in out.columns if c.startswith("fruit_tagged_diameter") and ("change" in c or "trend" in c or "diff" in c)], out)
    length_trend = first_existing([c for c in out.columns if c.startswith("fruit_tagged_length") and ("change" in c or "trend" in c or "diff" in c)], out)
    if diameter_trend:
        out["tagged_fruit_diameter_trend_mm"] = pd.to_numeric(out[diameter_trend], errors="coerce")
    if length_trend:
        out["tagged_fruit_length_trend_mm"] = pd.to_numeric(out[length_trend], errors="coerce")

    tagged_n_obs = first_existing(
        [c for c in out.columns if c.startswith("fruit_tagged_") and ("n_non_missing" in c or "n_observed" in c or "n_observ" in c or "observations" in c) and ("diameter" in c or "length" in c)],
        out,
    )
    if tagged_n_obs:
        out["tagged_fruit_n_observed"] = pd.to_numeric(out[tagged_n_obs], errors="coerce")
    return out


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def infer_feature_family(feature: str) -> str:
    lower = feature.lower()
    if feature == "nitrogen_treatment" or lower.startswith("cat__nitrogen") or feature == "treatment_n_level":
        return "nitrogen treatment"
    if "horizon" in lower or "day_of_year" in lower or "month" in lower or "week" in lower:
        return "calendar / lead time"
    if lower.startswith("fruit_tagged_") or lower.startswith("tagged_fruit_"):
        return "tagged fruit development"
    if any(term in lower for term in ["gdd", "radiation", "solar", "rad", "tmean", "tmax", "tmin", "rh", "humidity", "temperature"]):
        return "weather / thermal time"
    if "fruit_number" in lower or "fruit_count" in lower:
        return "fruit load"
    if any(term in lower for term in ["diameter", "length", "individual_fruit", "fresh_weight", "fruit_weight", "fruit_size"]):
        return "fruit size sample"
    if "fresh_matter" in lower:
        if any(term in lower for term in ["previous", "rolling", "trend", "change", "recent"]):
            return "fresh matter trajectory"
        return "current fresh matter state"
    if "dry_matter" in lower:
        if any(term in lower for term in ["previous", "rolling", "trend", "change", "recent"]):
            return "dry matter trajectory"
        return "current dry matter state"
    return "other"



def humanize_feature(feature: str) -> str:
    manual = {
        "nitrogen_treatment": "Nitrogen treatment category",
        "forecast_horizon_days": "Forecast lead time",
        "origin_day_of_year": "Forecast day of year",
        "current_fresh_matter_g": "Current fresh matter",
        "fresh_matter_trend_g": "Fresh-matter trend",
        "current_fruit_number": "Current fruit count",
        "fruit_number_trend": "Fruit-count trend",
        "current_dry_matter_g": "Current dry matter",
        "dry_matter_trend_g": "Dry-matter trend",
        "mean_fruit_length_mm": "Mean fruit length",
        "median_fruit_length_mm": "Median fruit length",
        "mean_fruit_diameter_mm": "Mean fruit diameter",
        "median_fruit_diameter_mm": "Median fruit diameter",
        "mean_individual_fruit_fresh_weight_g": "Mean individual fruit weight",
        "median_individual_fruit_fresh_weight_g": "Median individual fruit weight",
        "sample_diameter_mm_mean": "Mean sample fruit diameter",
        "sample_diameter_mm_median": "Median sample fruit diameter",
        "sample_length_mm_mean": "Mean sample fruit length",
        "sample_length_mm_median": "Median sample fruit length",
        "sample_individual_fruit_fresh_weight_g_mean": "Mean sample individual fruit weight",
        "sample_individual_fruit_fresh_weight_g_median": "Median sample individual fruit weight",
        "tagged_fruit_diameter_current_mm": "Tagged-fruit diameter",
        "tagged_fruit_length_current_mm": "Tagged-fruit length",
        "tagged_fruit_diameter_trend_mm": "Tagged-fruit diameter trend",
        "tagged_fruit_length_trend_mm": "Tagged-fruit length trend",
        "tagged_fruit_n_observed": "Tagged fruits observed",
    }
    if feature in manual:
        return manual[feature]
    label = re.sub(r"^fruit_", "", feature)
    label = re.sub(r"_", " ", label)
    return label[:1].upper() + label[1:]



def strip_transformer_prefix(name: str) -> str:
    # num__foo -> foo; cat__nitrogen_treatment_0N -> nitrogen_treatment
    if name.startswith("num__"):
        return name[5:]
    if name.startswith("cat__"):
        raw = name[5:]
        # We only expect one categorical feature in selected sets: nitrogen_treatment.
        if raw.startswith("nitrogen_treatment"):
            return "nitrogen_treatment"
        return raw.split("_")[0]
    return name


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)



def build_pipeline(feature_df: pd.DataFrame, features: list[str]) -> tuple[Pipeline, list[str], list[str]]:
    existing = [f for f in features if f in feature_df.columns]
    if not existing:
        raise ValueError("No existing features supplied to pipeline.")
    categorical = [f for f in existing if not pd.api.types.is_numeric_dtype(feature_df[f])]
    numeric = [f for f in existing if f not in categorical]

    transformers = []
    if numeric:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_one_hot_encoder())]), categorical))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)]), numeric, categorical



def load_model_feature_sets() -> pd.DataFrame:
    require_file(SELECTED_MODEL_FEATURES_PATH, "Run: python scripts/10_feature_audit_and_selection.py")
    model_features = pd.read_csv(SELECTED_MODEL_FEATURES_PATH)
    required = {"model_id", "model_label", "model_type", "feature"}
    if not required.issubset(model_features.columns):
        raise ValueError(f"{SELECTED_MODEL_FEATURES_PATH} is missing required columns: {required}")
    return model_features



def model_definitions(model_features: pd.DataFrame) -> list[dict[str, object]]:
    defs = []
    for (model_id, model_label, model_type), part in model_features.groupby(["model_id", "model_label", "model_type"], sort=False):
        features = [str(f) for f in part["feature"].dropna().tolist() if str(f)]
        note = str(part["note"].dropna().iloc[0]) if "note" in part and part["note"].notna().any() else ""
        defs.append({"model_id": model_id, "model_label": model_label, "model_type": model_type, "features": features, "note": note})
    return defs



def add_prediction_metadata(test: pd.DataFrame, pred: np.ndarray, model_def: dict[str, object], n_features: int, n_train: int) -> pd.DataFrame:
    out = test[["year", "current_observation_date", "target_date", "nitrogen_treatment", TARGET_COL]].copy()
    out = out.rename(columns={TARGET_COL: "observed_next_fresh_matter_g"})
    out["predicted_next_fresh_matter_g"] = pred
    out["error_g"] = out["predicted_next_fresh_matter_g"] - out["observed_next_fresh_matter_g"]
    out["absolute_error_g"] = out["error_g"].abs()
    out["model_id"] = model_def["model_id"]
    out["model_label"] = model_def["model_label"]
    out["model_type"] = model_def["model_type"]
    out["n_model_features"] = n_features
    out["n_training_rows"] = n_train
    return out



def predict_persistence(test: pd.DataFrame) -> np.ndarray:
    if "current_fresh_matter_g" not in test.columns:
        return np.full(len(test), np.nan)
    return pd.to_numeric(test["current_fresh_matter_g"], errors="coerce").to_numpy(dtype=float)



def predict_treatment_mean(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    treatment_means = train.groupby("nitrogen_treatment")[TARGET_COL].mean()
    global_mean = float(train[TARGET_COL].mean())
    return test["nitrogen_treatment"].map(treatment_means).fillna(global_mean).to_numpy(dtype=float)



def extract_feature_importance(pipeline: Pipeline, model_def: dict[str, object], origin_date: pd.Timestamp) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        transformed_names = list(preprocessor.get_feature_names_out())
    except Exception:
        transformed_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
    importances = model.feature_importances_
    rows = []
    for transformed, importance in zip(transformed_names, importances):
        original = strip_transformer_prefix(str(transformed))
        rows.append(
            {
                "model_id": model_def["model_id"],
                "model_label": model_def["model_label"],
                "forecast_origin_date": origin_date,
                "transformed_feature": transformed,
                "feature": original,
                "human_name": humanize_feature(original),
                "feature_family": infer_feature_family(original),
                "importance": float(importance),
            }
        )
    return pd.DataFrame(rows)



def run_rolling_models(df: pd.DataFrame, model_defs: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df = df.dropna(subset=[TARGET_COL, "current_observation_date"]).sort_values(["current_observation_date", "nitrogen_treatment"])
    origins = sorted(df.loc[df["year"] == TEST_YEAR, "current_observation_date"].dropna().unique())

    predictions = []
    importances = []
    history = []

    for origin in origins:
        origin_ts = pd.Timestamp(origin)
        train = df[(df["year"] < TEST_YEAR) | ((df["year"] == TEST_YEAR) & (df["current_observation_date"] < origin_ts))].copy()
        test = df[(df["year"] == TEST_YEAR) & (df["current_observation_date"] == origin_ts)].copy()
        train = train.dropna(subset=[TARGET_COL])
        test = test.dropna(subset=[TARGET_COL])
        if train.empty or test.empty:
            continue

        for model_def in model_defs:
            model_type = str(model_def["model_type"])
            model_id = str(model_def["model_id"])
            features = [f for f in model_def.get("features", []) if f in df.columns]

            if model_type == "baseline_rule" and model_id == "M0":
                pred = predict_persistence(test)
                n_features = 1 if "current_fresh_matter_g" in test.columns else 0
                predictions.append(add_prediction_metadata(test, pred, model_def, n_features, len(train)))
                continue

            if model_type == "baseline_rule" and model_id == "M0b":
                pred = predict_treatment_mean(train, test)
                predictions.append(add_prediction_metadata(test, pred, model_def, 1, len(train)))
                continue

            if not features or len(train) < 12:
                continue
            pipeline, numeric, categorical = build_pipeline(train, features)
            X_train = train[features]
            y_train = train[TARGET_COL]
            X_test = test[features]
            try:
                pipeline.fit(X_train, y_train)
                pred = pipeline.predict(X_test)
            except Exception as exc:
                print(f"Skipped {model_id} at {origin_ts.date()} because fit/predict failed: {exc}")
                continue

            predictions.append(add_prediction_metadata(test, pred, model_def, len(features), len(train)))
            importances.append(extract_feature_importance(pipeline, model_def, origin_ts))

            history.append(
                {
                    "forecast_origin_date": origin_ts,
                    "model_id": model_id,
                    "model_label": model_def["model_label"],
                    "n_training_rows": int(len(train)),
                    "n_previous_year_training_rows": int((train["year"] < TEST_YEAR).sum()),
                    "n_current_year_training_rows": int(((train["year"] == TEST_YEAR) & (train["current_observation_date"] < origin_ts)).sum()),
                    "n_forecast_rows": int(len(test)),
                    "n_model_features": int(len(features)),
                    "numeric_features": ", ".join(numeric),
                    "categorical_features": ", ".join(categorical),
                }
            )

    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    imp_df = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    hist_df = pd.DataFrame(history)
    return pred_df, imp_df, hist_df


# ---------------------------------------------------------------------------
# Metrics / selection / operations
# ---------------------------------------------------------------------------


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))



def build_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if predictions.empty:
        return pd.DataFrame()

    # Baseline rules such as persistence can legitimately be unavailable on
    # rows where the required current measurement is missing. sklearn metrics
    # do not accept NaN predictions, so evaluate each model on finite
    # prediction/target pairs and report how many rows were dropped.
    persistence_mae = np.nan
    for (model_id, model_label, model_type), part_raw in predictions.groupby(["model_id", "model_label", "model_type"], sort=False):
        part = part_raw.copy()
        part["observed_next_fresh_matter_g"] = pd.to_numeric(part["observed_next_fresh_matter_g"], errors="coerce")
        part["predicted_next_fresh_matter_g"] = pd.to_numeric(part["predicted_next_fresh_matter_g"], errors="coerce")
        valid_mask = np.isfinite(part["observed_next_fresh_matter_g"]) & np.isfinite(part["predicted_next_fresh_matter_g"])
        valid = part.loc[valid_mask].copy()
        n_dropped = int((~valid_mask).sum())
        if valid.empty:
            continue

        y_true = valid["observed_next_fresh_matter_g"]
        y_pred = valid["predicted_next_fresh_matter_g"]
        mae = float((y_pred - y_true).abs().mean())
        if model_id == "M0":
            persistence_mae = mae
        median_training_rows = float(valid["n_training_rows"].median())
        n_model_features = int(valid["n_model_features"].max())
        rows.append(
            {
                "model_id": model_id,
                "model_label": model_label,
                "model_type": model_type,
                "n_predictions": int(len(valid)),
                "n_predictions_raw": int(len(part_raw)),
                "n_predictions_dropped_nan": n_dropped,
                "n_model_features": n_model_features,
                "median_training_rows": median_training_rows,
                "feature_to_training_row_ratio": float(n_model_features / median_training_rows) if median_training_rows else np.nan,
                "mae": mae,
                "rmse": rmse(y_true, y_pred),
                "r2": float(r2_score(y_true, y_pred)) if len(valid) > 1 else np.nan,
                "bias": float((y_pred - y_true).mean()),
                "mean_observed": float(y_true.mean()),
                "mae_pct_of_mean_observed": float(100 * mae / y_true.mean()) if y_true.mean() else np.nan,
            }
        )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics
    if not np.isnan(persistence_mae) and persistence_mae != 0:
        metrics["mae_improvement_vs_persistence"] = persistence_mae - metrics["mae"]
        metrics["mae_improvement_pct_vs_persistence"] = 100 * (persistence_mae - metrics["mae"]) / persistence_mae
    else:
        metrics["mae_improvement_vs_persistence"] = np.nan
        metrics["mae_improvement_pct_vs_persistence"] = np.nan
    return metrics.sort_values(["mae", "n_model_features"]).reset_index(drop=True)



def summarize_importance(importances: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if importances.empty:
        return pd.DataFrame(), pd.DataFrame()
    feature_imp = (
        importances.groupby(["model_id", "model_label", "feature", "human_name", "feature_family"], dropna=False)["importance"]
        .mean()
        .reset_index(name="mean_importance")
    )
    totals = feature_imp.groupby(["model_id", "model_label"])["mean_importance"].transform("sum")
    feature_imp["share_of_model_importance_pct"] = np.where(totals > 0, 100 * feature_imp["mean_importance"] / totals, np.nan)
    feature_imp = feature_imp.sort_values(["model_id", "mean_importance"], ascending=[True, False]).reset_index(drop=True)

    block = (
        feature_imp.groupby(["model_id", "model_label", "feature_family"], dropna=False)
        .agg(mean_importance=("mean_importance", "sum"), n_features_in_block=("feature", "nunique"))
        .reset_index()
    )
    totals_block = block.groupby(["model_id", "model_label"])["mean_importance"].transform("sum")
    block["share_of_model_importance_pct"] = np.where(totals_block > 0, 100 * block["mean_importance"] / totals_block, np.nan)
    block = block.sort_values(["model_id", "share_of_model_importance_pct"], ascending=[True, False]).reset_index(drop=True)
    return feature_imp, block



def choose_preferred_model(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    candidates = metrics[~metrics["model_type"].eq("baseline_rule")].copy()
    if candidates.empty:
        return pd.DataFrame()
    best_mae = float(candidates["mae"].min())
    margin = max(PREFERRED_COMPETITIVE_MARGIN_G, PREFERRED_COMPETITIVE_MARGIN_PCT_OF_BEST * best_mae)
    competitive = candidates[candidates["mae"] <= best_mae + margin].copy()
    # Prefer compact selected models over M6 when MAE is competitive; then fewer features.
    model_preference = {"M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5, "M6": 6}
    competitive["model_complexity_preference"] = competitive["model_id"].map(model_preference).fillna(99)
    selected = competitive.sort_values(["n_model_features", "mae", "model_complexity_preference"]).head(1).copy()
    selected["best_observed_mae"] = best_mae
    selected["competitive_margin_g"] = margin
    selected["selection_rule"] = "Choose the simplest model within a small MAE margin of the best RF model."
    return selected.drop(columns=["model_complexity_preference"])



def build_error_audit(predictions: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or metrics.empty:
        return pd.DataFrame()
    preds = predictions.copy()
    for col in ["observed_next_fresh_matter_g", "predicted_next_fresh_matter_g", "error_g", "absolute_error_g"]:
        preds[col] = pd.to_numeric(preds[col], errors="coerce")
    preds = preds[np.isfinite(preds["observed_next_fresh_matter_g"]) & np.isfinite(preds["predicted_next_fresh_matter_g"])].copy()
    if preds.empty:
        return pd.DataFrame()
    rows = []
    actual_flush_threshold = preds["observed_next_fresh_matter_g"].quantile(0.75)
    for (model_id, model_label), part in preds.groupby(["model_id", "model_label"], sort=False):
        mae = float(metrics.loc[metrics["model_id"].eq(model_id), "mae"].iloc[0]) if metrics["model_id"].eq(model_id).any() else float(part["absolute_error_g"].mean())
        large_under = part["error_g"] < -mae
        actual_flush = part["observed_next_fresh_matter_g"] >= actual_flush_threshold
        under_flush = large_under & actual_flush
        rows.append(
            {
                "model_id": model_id,
                "model_label": model_label,
                "mae_reference_g": mae,
                "actual_flush_threshold_g": float(actual_flush_threshold),
                "n_large_underpredictions": int(large_under.sum()),
                "n_actual_flush_rows": int(actual_flush.sum()),
                "n_underpredicted_actual_flush_rows": int(under_flush.sum()),
                "mean_abs_error_on_actual_flush_rows_g": float(part.loc[actual_flush, "absolute_error_g"].mean()) if actual_flush.any() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["n_underpredicted_actual_flush_rows", "mae_reference_g"]).reset_index(drop=True)



def build_operational_board(predictions: pd.DataFrame, preferred: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or preferred.empty:
        return pd.DataFrame()
    preferred_id = str(preferred["model_id"].iloc[0])
    part = predictions[predictions["model_id"].eq(preferred_id)].copy()
    if part.empty:
        return pd.DataFrame()
    for col in ["observed_next_fresh_matter_g", "predicted_next_fresh_matter_g", "absolute_error_g"]:
        part[col] = pd.to_numeric(part[col], errors="coerce")
    part = part[np.isfinite(part["observed_next_fresh_matter_g"]) & np.isfinite(part["predicted_next_fresh_matter_g"])].copy()
    if part.empty:
        return pd.DataFrame()
    model_mae = float(metrics.loc[metrics["model_id"].eq(preferred_id), "mae"].iloc[0])
    board = (
        part.groupby("target_date")
        .agg(
            forecast_total_g=("predicted_next_fresh_matter_g", "sum"),
            observed_total_g=("observed_next_fresh_matter_g", "sum"),
            n_treatments=("nitrogen_treatment", "nunique"),
            mean_abs_error_g=("absolute_error_g", "mean"),
        )
        .reset_index()
        .sort_values("target_date")
    )
    board["previous_observed_total_g"] = board["observed_total_g"].shift(1)
    board["forecast_change_vs_previous_observed_g"] = board["forecast_total_g"] - board["previous_observed_total_g"]
    board["forecast_change_vs_previous_observed_pct"] = 100 * board["forecast_change_vs_previous_observed_g"] / board["previous_observed_total_g"]
    # Aggregate error band. It is a practical communication band, not a formal prediction interval.
    board["forecast_lower_g"] = board["forecast_total_g"] - model_mae * np.sqrt(board["n_treatments"].clip(lower=1))
    board["forecast_upper_g"] = board["forecast_total_g"] + model_mae * np.sqrt(board["n_treatments"].clip(lower=1))
    board["outside_mae_band"] = (board["observed_total_g"] < board["forecast_lower_g"]) | (board["observed_total_g"] > board["forecast_upper_g"])
    board["underpredicted_peak"] = board["observed_total_g"] > board["forecast_upper_g"]

    def flag(change_pct: float) -> str:
        if pd.isna(change_pct):
            return "first window"
        if change_pct >= 20:
            return "Production flush risk"
        if change_pct <= -20:
            return "Supply drop risk"
        return "Stable window"

    board["operational_flag"] = board["forecast_change_vs_previous_observed_pct"].apply(flag)
    board["preferred_model_id"] = preferred_id
    board["preferred_model_label"] = str(preferred["model_label"].iloc[0])
    board["model_mae_reference_g"] = model_mae
    return board


# ---------------------------------------------------------------------------
# Writing/reporting
# ---------------------------------------------------------------------------


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)



def write_duckdb_tables(tables: dict[str, pd.DataFrame]) -> None:
    if duckdb is None:
        print("DuckDB not available; skipped DuckDB table writes.")
        return
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(DB_PATH) as con:
        for table_name, df in tables.items():
            con.register("tmp_df", df)
            con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM tmp_df")
            con.unregister("tmp_df")



def print_feature_sets(model_features: pd.DataFrame) -> None:
    print("\nFeature sets used:")
    for (model_id, model_label), part in model_features.groupby(["model_id", "model_label"], sort=False):
        features = [f for f in part["feature"].dropna().tolist() if str(f)]
        print(f"\n{model_id} — {model_label} ({len(features)} features)")
        grouped = part[part["feature"].astype(str).str.len() > 0].groupby("feature_family")["human_name"].apply(list)
        for family, names in grouped.items():
            print(f"  {family}:")
            for name in names:
                print(f"    - {name}")



def main() -> None:
    require_file(SELECTED_MODEL_FEATURES_PATH, "Run: python scripts/10_feature_audit_and_selection.py")
    forecast = load_forecast_features()
    treatment = load_treatment_fruit_features_optional()
    tagged = load_tagged_features_optional()
    df = join_treatment_fruit_features(forecast, treatment)
    df = join_tagged_features(df, tagged)
    df = add_compact_engineered_features(df)

    model_features = load_model_feature_sets()
    model_defs = model_definitions(model_features)

    predictions, raw_importances, history = run_rolling_models(df, model_defs)
    metrics = build_metrics(predictions)
    feature_importance, block_importance = summarize_importance(raw_importances)
    preferred = choose_preferred_model(metrics)
    error_audit = build_error_audit(predictions, metrics)
    operational_board = build_operational_board(predictions, preferred, metrics)

    outputs = {
        "strawberry_compact_model_predictions": predictions,
        "strawberry_compact_model_metrics": metrics,
        "strawberry_compact_model_training_history": history,
        "strawberry_compact_model_feature_importance": feature_importance,
        "strawberry_compact_model_block_importance": block_importance,
        "strawberry_compact_model_selection_summary": preferred,
        "strawberry_compact_model_error_audit": error_audit,
        "strawberry_compact_model_operational_board": operational_board,
        "strawberry_compact_model_feature_sets_used": model_features,
    }
    for name, table in outputs.items():
        write_csv(table, PROCESSED_DIR / f"{name}.csv")
    write_duckdb_tables(outputs)

    print("Compact sequential modelling complete.")
    print(f"Forecast rows available: {len(df)}")
    print(f"Prediction rows generated: {len(predictions)}")
    if not metrics.empty:
        print("\nModel comparison:")
        display_cols = ["model_id", "model_label", "n_predictions", "n_model_features", "feature_to_training_row_ratio", "mae", "rmse", "r2", "bias", "mae_improvement_pct_vs_persistence"]
        print(metrics[display_cols].to_string(index=False))
    if not preferred.empty:
        print("\nRecommended preferred model:")
        cols = ["model_id", "model_label", "n_model_features", "mae", "mae_improvement_pct_vs_persistence", "best_observed_mae", "competitive_margin_g", "selection_rule"]
        print(preferred[cols].to_string(index=False))
    if not error_audit.empty:
        print("\nFlush / underprediction audit:")
        cols = ["model_id", "model_label", "n_large_underpredictions", "n_actual_flush_rows", "n_underpredicted_actual_flush_rows", "mean_abs_error_on_actual_flush_rows_g"]
        print(error_audit[cols].to_string(index=False))
    print_feature_sets(model_features)

    print("\nOutputs written to:")
    for name in outputs:
        print(f"- {PROCESSED_DIR / f'{name}.csv'}")


if __name__ == "__main__":
    main()