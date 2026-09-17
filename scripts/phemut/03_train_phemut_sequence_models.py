#!/usr/bin/env python3
"""Train independent paper-informed PheMuT sequence baselines and models.

This script evaluates the sequence formulation:
- one plot = one sequence
- first N weeks are inputs
- each future week/horizon is predicted independently
- models are trained and validated within each season using K-fold splits over plots

It intentionally avoids the original PheMuT neural architecture/code. The goal is a
portfolio-grade audit: do compact tabular sequence baselines beat simple rules, and
are the resulting future curves genuinely anticipatory rather than lagged?

Inputs:
- data/processed/phemut/phemut_sequence_dataset.csv
- data/processed/phemut/phemut_sequence_feature_sets.csv

Outputs:
- phemut_sequence_model_predictions.csv
- phemut_sequence_model_metrics.csv
- phemut_sequence_aggregate_fold_board.csv
- phemut_sequence_aggregate_fold_metrics.csv
- phemut_sequence_aggregate_reconstructed_board.csv
- phemut_sequence_aggregate_reconstructed_metrics.csv
- phemut_sequence_timeliness_audit.csv
- phemut_sequence_model_selection_summary.csv
- phemut_sequence_feature_importance_detail.csv
- phemut_sequence_grouped_feature_importance.csv
- phemut_sequence_prediction_intervals.csv
- phemut_sequence_uncertainty_summary.csv
- phemut_sequence_paper_comparison.csv
- phemut_sequence_weather_driver_summary.csv
- phemut_sequence_stakeholder_summary.csv / .md

Important evaluation note:
The primary aggregate validation is calculated at held-out fold level. A full-field
out-of-fold reconstructed curve is also exported for illustration only, because
summing all out-of-fold predictions can create arithmetic artefacts for training-mean
baselines when folds are equal-sized.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

DEFAULT_PROCESSED_DIR = Path("data/processed/phemut")
DEFAULT_DB_PATH = Path("db/yield_forecasting.duckdb")
RANDOM_STATE = 42

# Published PheMuT paper metrics for the YOLO/LSTM yield forecaster.
# Used only for contextual comparison; protocols are not identical.
PAPER_REFERENCE_METRICS = pd.DataFrame([
    {
        "season": "2324",
        "paper_system": "PheMuT YOLO/LSTM",
        "paper_mae": 76.4,
        "paper_rmse": 92.6,
        "paper_r2": 0.630,
        "paper_correlation": 0.801,
        "paper_mape_pct": 30.2,
    },
    {
        "season": "2425",
        "paper_system": "PheMuT YOLO/LSTM",
        "paper_mae": 30.0,
        "paper_rmse": 36.2,
        "paper_r2": 0.249,
        "paper_correlation": 0.505,
        "paper_mape_pct": 174.3,
    },
])


@dataclass(frozen=True)
class MetricRow:
    season: str
    feature_set: str
    target_horizon_index: int | str
    target_date: str
    model_label: str
    validation_protocol: str
    n_predictions: int
    n_plots: int
    n_model_features: int
    mae: float
    rmse: float
    r2: float
    correlation: float
    bias_predicted_minus_observed_g: float
    mae_improvement_pct_vs_last_observed_baseline: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PheMuT sequence baselines and compact models.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory with phemut_sequence_dataset.csv and output destination.",
    )
    parser.add_argument(
        "--feature-set",
        default="paper_informed_nonreactive",
        help="Feature set name from phemut_sequence_feature_sets.csv.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Requested K-fold splits within each season/horizon. Reduced automatically when support is small.",
    )
    parser.add_argument(
        "--min-plots-per-horizon",
        type=int,
        default=12,
        help="Skip season/horizon evaluations with fewer plots than this.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fewer trees for quick iteration.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="DuckDB path. Use --no-db to skip database writes.",
    )
    parser.add_argument("--no-db", action="store_true", help="Do not write DuckDB tables.")
    return parser.parse_args()


def load_feature_set(processed_dir: Path, requested_name: str, sequence_df: pd.DataFrame) -> list[str]:
    path = processed_dir / "phemut_sequence_feature_sets.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature set file: {path}. Run 02_build_phemut_sequence_dataset.py first.")
    feature_sets = pd.read_csv(path)
    if requested_name not in set(feature_sets["feature_set"]):
        available = ", ".join(feature_sets["feature_set"].astype(str).tolist())
        raise ValueError(f"Feature set {requested_name!r} not found. Available: {available}")
    raw = feature_sets.loc[feature_sets["feature_set"] == requested_name, "features_json"].iloc[0]
    features = json.loads(raw)
    existing = [feature for feature in features if feature in sequence_df.columns]
    missing = [feature for feature in features if feature not in sequence_df.columns]
    if missing:
        print(f"Warning: dropping {len(missing)} missing features from requested feature set.")
    if not existing:
        raise ValueError("No model features remain after checking against sequence dataset columns.")
    forbidden = [f for f in existing if any(token in f.lower() for token in ["target", "next", "future", "error", "observed_for_baseline"])]
    if forbidden:
        raise ValueError(f"Feature leakage risk: forbidden target-like features in model set: {forbidden[:10]}")
    return existing


def build_models(fast: bool) -> dict[str, object]:
    n_trees = 150 if fast else 500
    return {
        "Ridge": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=5.0, random_state=RANDOM_STATE)),
            ]
        ),
        "ElasticNet": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.05, l1_ratio=0.2, max_iter=20_000, random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=n_trees,
                        max_depth=4,
                        min_samples_leaf=3,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Extra Trees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=n_trees,
                        max_depth=4,
                        min_samples_leaf=3,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=200 if not fast else 80,
                        learning_rate=0.04,
                        max_depth=2,
                        min_samples_leaf=3,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=250 if not fast else 100,
                        learning_rate=0.04,
                        max_leaf_nodes=8,
                        l2_regularization=0.5,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }




def base_feature_name(feature: str) -> tuple[str | None, str]:
    """Return input week index and semantic base name for flattened sequence feature."""
    text = str(feature)
    if text.startswith("w") and "_" in text:
        prefix, rest = text.split("_", 1)
        if prefix[1:].isdigit():
            return prefix, rest
    return None, text


def feature_group(feature: str) -> str:
    """Map individual flattened features to stakeholder-friendly groups."""
    _, base = base_feature_name(feature)
    low = base.lower()
    if any(token in low for token in ["flower", "green_fruit", "white_fruit", "pink_fruit", "unripe", "phenology"]):
        return "Phenology counts"
    if any(token in low for token in ["canopy", "area", "volume", "depth"]):
        return "Canopy structure"
    if any(token in low for token in ["weather", "gdd", "temp", "soil", "rh", "humidity", "dew", "rain", "solar", "radiation", "wind"]):
        return "Weather / GDD"
    if any(token in low for token in ["medallion", "cultivar", "plot_block", "plot_replicate"]):
        return "Cultivar / plot context"
    return "Other"


def weather_driver(feature: str) -> str:
    """Give a more specific label for weather/GDD features."""
    _, base = base_feature_name(feature)
    low = base.lower()
    if "gdd" in low:
        return "Growing degree days"
    if "solar" in low or "radiation" in low:
        return "Solar radiation"
    if "rain" in low:
        return "Rainfall"
    if "temp" in low:
        return "Temperature"
    if "rh" in low or "humidity" in low or "dew" in low:
        return "Humidity / dew point"
    if "wind" in low:
        return "Wind"
    return "Other weather"


def extract_feature_importance(fitted: object, features: list[str]) -> np.ndarray | None:
    """Extract comparable within-model importances from tree importances or linear coefficients."""
    model = fitted.named_steps.get("model") if hasattr(fitted, "named_steps") else fitted
    values = None
    method = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        method = "tree_feature_importance"
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float).reshape(-1))
        method = "absolute_standardized_coefficient"
    if values is None or len(values) != len(features):
        return None
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(values.sum())
    if total > 0:
        values = values / total
    return values


def importance_method(fitted: object) -> str | None:
    model = fitted.named_steps.get("model") if hasattr(fitted, "named_steps") else fitted
    if hasattr(model, "feature_importances_"):
        return "tree_feature_importance"
    if hasattr(model, "coef_"):
        return "absolute_standardized_coefficient"
    return None


def build_grouped_feature_importance(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    df = detail.copy()
    group_cols = ["season", "feature_set", "model_label", "target_horizon_index", "fold_id", "feature_group"]
    grouped = df.groupby(group_cols, dropna=False, as_index=False)["normalized_importance"].sum()
    summary_cols = ["season", "feature_set", "model_label", "feature_group"]
    summary = grouped.groupby(summary_cols, dropna=False, as_index=False).agg(
        mean_group_importance=("normalized_importance", "mean"),
        median_group_importance=("normalized_importance", "median"),
        n_model_fits=("normalized_importance", "size"),
    )
    totals = summary.groupby(["season", "feature_set", "model_label"], dropna=False)["mean_group_importance"].transform("sum")
    summary["importance_share_pct"] = np.where(totals > 0, summary["mean_group_importance"] / totals * 100, np.nan)
    return summary.sort_values(["season", "model_label", "importance_share_pct"], ascending=[True, True, False])


def build_weather_driver_summary(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    df = detail.copy()
    df = df[df["feature_group"].eq("Weather / GDD")]
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(["season", "feature_set", "model_label", "weather_driver"], dropna=False, as_index=False).agg(
        mean_importance=("normalized_importance", "mean"),
        total_importance=("normalized_importance", "sum"),
        n_feature_rows=("normalized_importance", "size"),
    )
    totals = grouped.groupby(["season", "feature_set", "model_label"], dropna=False)["total_importance"].transform("sum")
    grouped["weather_importance_share_pct"] = np.where(totals > 0, grouped["total_importance"] / totals * 100, np.nan)
    return grouped.sort_values(["season", "model_label", "weather_importance_share_pct"], ascending=[True, True, False])


def build_prediction_intervals(predictions: pd.DataFrame, lower_q: float = 0.10, upper_q: float = 0.90) -> pd.DataFrame:
    """Add simple residual-calibrated 80% intervals from other CV folds where possible."""
    if predictions.empty:
        return pd.DataFrame()
    df = predictions.copy()
    df["residual_observed_minus_predicted_g"] = df["target_yield_g"].astype(float) - df["prediction_g"].astype(float)
    rows = []
    grouping = ["season", "model_label", "target_horizon_index"]
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        mask = pd.Series(True, index=df.index)
        for col in grouping:
            mask &= df[col].astype(str).eq(str(row_dict[col]))
        if "fold_id" in df.columns:
            mask &= ~pd.to_numeric(df["fold_id"], errors="coerce").eq(float(row_dict.get("fold_id")))
        residuals = df.loc[mask, "residual_observed_minus_predicted_g"].dropna()
        # Fallback to same season/model excluding the fold if horizon-level support is too small.
        if residuals.size < 5:
            mask = df["season"].astype(str).eq(str(row_dict["season"])) & df["model_label"].astype(str).eq(str(row_dict["model_label"]))
            if "fold_id" in df.columns:
                mask &= ~pd.to_numeric(df["fold_id"], errors="coerce").eq(float(row_dict.get("fold_id")))
            residuals = df.loc[mask, "residual_observed_minus_predicted_g"].dropna()
        if residuals.empty:
            low_resid = high_resid = np.nan
        else:
            low_resid = float(residuals.quantile(lower_q))
            high_resid = float(residuals.quantile(upper_q))
        pred = float(row_dict["prediction_g"])
        lower = max(0.0, pred + low_resid) if pd.notna(low_resid) else np.nan
        upper = max(0.0, pred + high_resid) if pd.notna(high_resid) else np.nan
        target = float(row_dict["target_yield_g"])
        row_dict.update(
            {
                "interval_method": "other_fold_residual_p10_p90",
                "prediction_lower80_g": lower,
                "prediction_upper80_g": upper,
                "prediction_interval_width_g": upper - lower if pd.notna(lower) and pd.notna(upper) else np.nan,
                "interval_covered_target": bool(pd.notna(lower) and pd.notna(upper) and lower <= target <= upper),
                "calibration_residual_n": int(residuals.size),
            }
        )
        rows.append(row_dict)
    return pd.DataFrame(rows)


def build_uncertainty_summary(intervals: pd.DataFrame) -> pd.DataFrame:
    if intervals.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["season", "feature_set", "model_label"]
    for keys, group in intervals.groupby(group_cols, dropna=False, sort=True):
        season, feature_set, model_label = keys
        y_true = group["target_yield_g"].astype(float)
        y_pred = group["prediction_g"].astype(float)
        rows.append(
            {
                "season": season,
                "feature_set": feature_set,
                "model_label": model_label,
                "n_predictions": len(group),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "mean_interval_width_g": float(group["prediction_interval_width_g"].mean()),
                "median_interval_width_g": float(group["prediction_interval_width_g"].median()),
                "interval_coverage_pct": float(group["interval_covered_target"].mean() * 100),
                "mean_calibration_residual_n": float(group["calibration_residual_n"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["season", "mae"])


def build_paper_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    overall = metrics.loc[metrics["target_horizon_index"].astype(str).eq("all")].copy()
    if overall.empty:
        overall = metric_rows_for_overall(metrics)
    learned = overall[~overall["model_label"].astype(str).str.contains("baseline", case=False, na=False)].copy()
    if learned.empty:
        return pd.DataFrame()
    best = learned.sort_values(["season", "mae"]).groupby("season", as_index=False).head(1)
    comp = best.merge(PAPER_REFERENCE_METRICS, on="season", how="left")
    comp["our_pipeline"] = "Independent tabular sequence audit"
    comp = comp.rename(
        columns={
            "model_label": "our_best_model",
            "mae": "our_mae",
            "rmse": "our_rmse",
            "r2": "our_r2",
            "correlation": "our_correlation",
        }
    )
    for metric in ["mae", "rmse", "r2", "correlation"]:
        our_col = f"our_{metric}"
        paper_col = f"paper_{metric}"
        if our_col in comp.columns and paper_col in comp.columns:
            comp[f"delta_{metric}_our_minus_paper"] = comp[our_col] - comp[paper_col]
            comp[f"pct_delta_{metric}_our_vs_paper"] = np.where(comp[paper_col].abs() > 0, (comp[our_col] - comp[paper_col]) / comp[paper_col].abs() * 100, np.nan)
    comp["comparison_note"] = (
        "Context only: protocols differ. Paper uses original multimodal PheMuT/LSTM pipeline; this project uses processed tables, tabular models, "
        "within-season plot CV, and validation diagnostics."
    )
    preferred_cols = [
        "season", "paper_system", "our_pipeline", "our_best_model", "paper_mae", "our_mae", "pct_delta_mae_our_vs_paper",
        "paper_rmse", "our_rmse", "pct_delta_rmse_our_vs_paper", "paper_r2", "our_r2", "paper_correlation", "our_correlation", "comparison_note"
    ]
    return comp[[c for c in preferred_cols if c in comp.columns]].sort_values("season")


def build_stakeholder_summary(metrics: pd.DataFrame, aggregate_metrics: pd.DataFrame, uncertainty: pd.DataFrame, grouped_importance: pd.DataFrame, paper_comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if metrics.empty:
        return pd.DataFrame()
    overall = metrics.loc[metrics["target_horizon_index"].astype(str).eq("all")].copy()
    learned = overall[~overall["model_label"].astype(str).str.contains("baseline", case=False, na=False)]
    for season, group in learned.groupby("season", sort=True):
        best = group.sort_values("mae").iloc[0]
        agg = pd.DataFrame()
        if not aggregate_metrics.empty:
            agg = aggregate_metrics[(aggregate_metrics["season"].astype(str) == str(season)) & (~aggregate_metrics["model_label"].astype(str).str.contains("baseline", case=False, na=False))].sort_values("aggregate_mae_g")
        agg_row = agg.iloc[0] if not agg.empty else pd.Series(dtype=object)
        unc = pd.DataFrame()
        if not uncertainty.empty:
            unc = uncertainty[(uncertainty["season"].astype(str) == str(season)) & (uncertainty["model_label"].astype(str) == str(best["model_label"]))]
        unc_row = unc.iloc[0] if not unc.empty else pd.Series(dtype=object)
        imp = pd.DataFrame()
        if not grouped_importance.empty:
            imp = grouped_importance[(grouped_importance["season"].astype(str) == str(season)) & (grouped_importance["model_label"].astype(str) == str(best["model_label"]))].sort_values("importance_share_pct", ascending=False)
        top_drivers = "; ".join(
            f"{r.feature_group} ({r.importance_share_pct:.1f}%)" for r in imp.head(3).itertuples(index=False)
        ) if not imp.empty else "not available"
        paper = pd.DataFrame()
        if not paper_comparison.empty:
            paper = paper_comparison[paper_comparison["season"].astype(str) == str(season)]
        paper_note = "not compared"
        if not paper.empty:
            p = paper.iloc[0]
            paper_note = f"Paper MAE {p.get('paper_mae'):.1f}; independent audit MAE {p.get('our_mae'):.1f}. Not apples-to-apples."
        rows.append(
            {
                "season": season,
                "recommended_plot_model": best["model_label"],
                "plot_mae_g": best["mae"],
                "plot_improvement_pct_vs_last_observed": best["mae_improvement_pct_vs_last_observed_baseline"],
                "recommended_aggregate_model": agg_row.get("model_label", np.nan),
                "aggregate_mae_g": agg_row.get("aggregate_mae_g", np.nan),
                "aggregate_improvement_pct_vs_last_observed": agg_row.get("aggregate_mae_improvement_pct_vs_last_observed_baseline", np.nan),
                "uncertainty_coverage_pct": unc_row.get("interval_coverage_pct", np.nan),
                "mean_interval_width_g": unc_row.get("mean_interval_width_g", np.nan),
                "top_driver_groups": top_drivers,
                "paper_context": paper_note,
                "stakeholder_message": "Learned models beat reactive baselines within-season; use aggregate curves for volume planning, with uncertainty bands and data caveats.",
            }
        )
    return pd.DataFrame(rows)


def write_stakeholder_markdown(path: Path, stakeholder: pd.DataFrame) -> None:
    if stakeholder.empty:
        path.write_text("# Stakeholder summary\n\nNo stakeholder summary available.\n", encoding="utf-8")
        return
    lines = ["# Stakeholder one-page summary", "", "Purpose: communicate whether the processed strawberry data can support short-horizon yield planning.", ""]
    for row in stakeholder.itertuples(index=False):
        lines.extend([
            f"## Season {row.season}",
            f"- Recommended plot-level model: **{row.recommended_plot_model}** (MAE {row.plot_mae_g:.2f} g; improvement {row.plot_improvement_pct_vs_last_observed:.1f}% vs last-observed baseline).",
            f"- Recommended aggregate model: **{row.recommended_aggregate_model}** (held-out aggregate MAE {row.aggregate_mae_g:.2f} g; improvement {row.aggregate_improvement_pct_vs_last_observed:.1f}%).",
            f"- Main driver groups: {row.top_driver_groups}.",
            f"- Uncertainty: retrospective 80% interval coverage {row.uncertainty_coverage_pct:.1f}%; mean interval width {row.mean_interval_width_g:.2f} g.",
            f"- Paper context: {row.paper_context}",
            "- Caveat: within-season validation only; this is not evidence of commercial deployment readiness.",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")

def safe_rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def safe_corr(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    a = pd.Series(y_true, dtype="float64")
    b = pd.Series(y_pred, dtype="float64")
    mask = a.notna() & b.notna()
    if mask.sum() < 2 or a[mask].nunique() < 2 or b[mask].nunique() < 2:
        return np.nan
    return float(a[mask].corr(b[mask]))


def safe_r2(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y = pd.Series(y_true, dtype="float64")
    if y.notna().sum() < 2 or y.nunique(dropna=True) < 2:
        return np.nan
    return float(r2_score(y_true, y_pred))


def predict_train_mean(y_train: pd.Series, n: int) -> np.ndarray:
    return np.repeat(float(y_train.mean()), n)


def predict_cultivar_mean(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    global_mean = float(train["target_yield_g"].mean())
    if "cultivar" not in train.columns or train["cultivar"].nunique(dropna=True) < 2:
        return np.repeat(global_mean, len(test))
    means = train.groupby("cultivar")["target_yield_g"].mean().to_dict()
    return test["cultivar"].map(means).fillna(global_mean).to_numpy(dtype=float)


def evaluate_sequence_models(sequence_df: pd.DataFrame, features: list[str], feature_set: str, n_splits: int, min_plots: int, fast: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    models = build_models(fast=fast)

    df = sequence_df.copy()
    df["target_date"] = pd.to_datetime(df["target_date"])
    df = df.dropna(subset=["target_yield_g", "target_horizon_index", "season", "plot_id"])

    for (season, horizon), subset in df.groupby(["season", "target_horizon_index"], sort=True):
        subset = subset.sort_values("plot_id").reset_index(drop=True)
        n_plots = subset["plot_id"].nunique()
        if n_plots < min_plots:
            print(f"Skipping season={season}, horizon={horizon}: only {n_plots} plots.")
            continue
        k = min(n_splits, n_plots)
        if k < 2:
            continue
        kfold = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
        y_all = subset["target_yield_g"].astype(float)
        X_all = subset[features].apply(pd.to_numeric, errors="coerce")

        fold_baseline_predictions: dict[str, list[np.ndarray]] = {
            "Training horizon mean baseline": [],
            "Cultivar horizon mean baseline": [],
            "Last observed week-5 yield baseline": [],
            "Observed 5-week mean yield baseline": [],
            "Zero yield baseline": [],
        }
        fold_model_predictions: dict[str, list[np.ndarray]] = {model_label: [] for model_label in models}
        fold_indices: list[np.ndarray] = []

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(subset), start=1):
            train = subset.iloc[train_idx].copy()
            test = subset.iloc[test_idx].copy()
            X_train = X_all.iloc[train_idx]
            X_test = X_all.iloc[test_idx]
            y_train = y_all.iloc[train_idx]

            fold_indices.append(test_idx)
            fold_baseline_predictions["Training horizon mean baseline"].append(predict_train_mean(y_train, len(test_idx)))
            fold_baseline_predictions["Cultivar horizon mean baseline"].append(predict_cultivar_mean(train, test))
            fold_baseline_predictions["Last observed week-5 yield baseline"].append(test["baseline_last_observed_yield_g"].astype(float).to_numpy())
            fold_baseline_predictions["Observed 5-week mean yield baseline"].append(test["baseline_mean_observed_yield_g"].astype(float).to_numpy())
            fold_baseline_predictions["Zero yield baseline"].append(np.repeat(0.0, len(test_idx)))

            for model_label, estimator in models.items():
                fitted = clone(estimator)
                fitted.fit(X_train, y_train)
                method = importance_method(fitted)
                importances = extract_feature_importance(fitted, features)
                if importances is not None and method is not None:
                    for feature, score in zip(features, importances):
                        week_index, base_feature = base_feature_name(feature)
                        importance_rows.append(
                            {
                                "season": str(season),
                                "feature_set": feature_set,
                                "model_label": model_label,
                                "target_horizon_index": int(horizon),
                                "target_date": pd.Timestamp(subset["target_date"].iloc[0]).date().isoformat(),
                                "fold_id": int(fold_idx),
                                "importance_method": method,
                                "feature": feature,
                                "input_week": week_index,
                                "base_feature": base_feature,
                                "feature_group": feature_group(feature),
                                "weather_driver": weather_driver(feature),
                                "normalized_importance": float(score),
                            }
                        )
                fold_model_predictions[model_label].append(fitted.predict(X_test))

        target_date = pd.Timestamp(subset["target_date"].iloc[0]).date().isoformat()
        validation_protocol = f"within-season {k}-fold CV over plots; fixed input history; horizon-specific models"

        all_prediction_sources = {**fold_baseline_predictions, **fold_model_predictions}
        for model_label, fold_preds in all_prediction_sources.items():
            ordered_indices = np.concatenate(fold_indices)
            ordered_preds = np.concatenate(fold_preds)
            ordered_fold_ids = np.concatenate([
                np.repeat(fold_number, len(indices))
                for fold_number, indices in enumerate(fold_indices, start=1)
            ])
            pred_frame = subset.iloc[ordered_indices].copy()
            pred_frame["prediction_g"] = ordered_preds
            pred_frame["fold_id"] = ordered_fold_ids
            pred_frame["model_label"] = model_label
            pred_frame["feature_set"] = feature_set
            pred_frame["validation_protocol"] = validation_protocol
            pred_frame["n_model_features"] = 0 if "baseline" in model_label.lower() else len(features)

            for row in pred_frame.itertuples(index=False):
                predictions.append(
                    {
                        "season": row.season,
                        "plot_id": row.plot_id,
                        "cultivar": getattr(row, "cultivar", "unknown"),
                        "target_horizon_index": int(row.target_horizon_index),
                        "target_date": pd.Timestamp(row.target_date).date().isoformat(),
                        "input_start_date": pd.Timestamp(row.input_start_date).date().isoformat(),
                        "input_end_date": pd.Timestamp(row.input_end_date).date().isoformat(),
                        "forecast_horizon_days": int(row.forecast_horizon_days),
                        "target_yield_g": float(row.target_yield_g),
                        "prediction_g": float(row.prediction_g),
                        "baseline_last_observed_yield_g": float(row.baseline_last_observed_yield_g) if pd.notna(row.baseline_last_observed_yield_g) else np.nan,
                        "baseline_mean_observed_yield_g": float(row.baseline_mean_observed_yield_g) if pd.notna(row.baseline_mean_observed_yield_g) else np.nan,
                        "feature_set": row.feature_set,
                        "model_label": row.model_label,
                        "validation_protocol": row.validation_protocol,
                        "fold_id": int(row.fold_id),
                        "n_cv_folds": int(k),
                        "n_model_features": int(row.n_model_features),
                    }
                )

            y_true = pred_frame["target_yield_g"].astype(float)
            y_pred = pred_frame["prediction_g"].astype(float)
            last_baseline_pred = pred_frame["baseline_last_observed_yield_g"].astype(float)
            baseline_mae = float(mean_absolute_error(y_true, last_baseline_pred))
            mae = float(mean_absolute_error(y_true, y_pred))
            improvement = ((baseline_mae - mae) / baseline_mae * 100) if baseline_mae and baseline_mae > 0 else np.nan
            metric_rows.append(
                MetricRow(
                    season=str(season),
                    feature_set=feature_set,
                    target_horizon_index=int(horizon),
                    target_date=target_date,
                    model_label=model_label,
                    validation_protocol=validation_protocol,
                    n_predictions=len(pred_frame),
                    n_plots=int(pred_frame["plot_id"].nunique()),
                    n_model_features=0 if "baseline" in model_label.lower() else len(features),
                    mae=mae,
                    rmse=safe_rmse(y_true, y_pred),
                    r2=safe_r2(y_true, y_pred),
                    correlation=safe_corr(y_true, y_pred),
                    bias_predicted_minus_observed_g=float((y_pred - y_true).mean()),
                    mae_improvement_pct_vs_last_observed_baseline=float(improvement) if pd.notna(improvement) else np.nan,
                ).__dict__
            )

    return pd.DataFrame(predictions), pd.DataFrame(metric_rows), pd.DataFrame(importance_rows)


def build_overall_metrics(predictions: pd.DataFrame, horizon_metrics: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    for (season, feature_set, model_label), group in predictions.groupby(["season", "feature_set", "model_label"], sort=True):
        y_true = group["target_yield_g"].astype(float)
        y_pred = group["prediction_g"].astype(float)
        last_baseline = group["baseline_last_observed_yield_g"].astype(float)
        baseline_mae = float(mean_absolute_error(y_true, last_baseline))
        mae = float(mean_absolute_error(y_true, y_pred))
        rows.append(
            {
                "season": season,
                "feature_set": feature_set,
                "target_horizon_index": "all",
                "target_date": "all",
                "model_label": model_label,
                "validation_protocol": "within-season K-fold CV over plots, pooled across evaluated future horizons",
                "n_predictions": len(group),
                "n_plots": group["plot_id"].nunique(),
                "n_target_horizons": group["target_horizon_index"].nunique(),
                "n_model_features": int(group["n_model_features"].max()),
                "mae": mae,
                "rmse": safe_rmse(y_true, y_pred),
                "r2": safe_r2(y_true, y_pred),
                "correlation": safe_corr(y_true, y_pred),
                "bias_predicted_minus_observed_g": float((y_pred - y_true).mean()),
                "mae_improvement_pct_vs_last_observed_baseline": ((baseline_mae - mae) / baseline_mae * 100) if baseline_mae > 0 else np.nan,
            }
        )
    overall = pd.DataFrame(rows)
    return pd.concat([horizon_metrics, overall], ignore_index=True, sort=False)


def build_reconstructed_aggregate_board(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    board = (
        predictions.groupby(
            [
                "season",
                "feature_set",
                "model_label",
                "input_start_date",
                "input_end_date",
                "target_horizon_index",
                "target_date",
            ],
            as_index=False,
        )
        .agg(
            n_plots=("plot_id", "nunique"),
            observed_total_yield_g=("target_yield_g", "sum"),
            predicted_total_yield_g=("prediction_g", "sum"),
            last_observed_total_yield_baseline_g=("baseline_last_observed_yield_g", "sum"),
            mean_observed_history_total_yield_baseline_g=("baseline_mean_observed_yield_g", "sum"),
        )
        .sort_values(["season", "model_label", "target_horizon_index"])
    )
    board["aggregate_error_g"] = board["predicted_total_yield_g"] - board["observed_total_yield_g"]
    board["aggregate_abs_error_g"] = board["aggregate_error_g"].abs()
    board["validation_scope"] = "full_field_reconstructed_oof_curve_illustrative_not_for_selection"
    board["evaluation_warning"] = (
        "Illustrative only: summing all out-of-fold predictions can create arithmetic "
        "artefacts for fold-wise training-mean baselines. Use aggregate_fold metrics for selection."
    )
    return board


def build_aggregate_fold_board(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate predictions only within each held-out CV fold.

    This is the primary aggregate validation table. It avoids the full-field OOF
    reconstruction artefact where equal-sized folds and training-mean predictions
    can sum back to the observed whole-field total.
    """
    if predictions.empty:
        return pd.DataFrame()
    required = {"fold_id", "target_yield_g", "prediction_g", "plot_id"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Cannot build fold aggregate board; missing columns: {missing}")

    group_cols = [
        "season",
        "feature_set",
        "model_label",
        "fold_id",
        "input_start_date",
        "input_end_date",
        "target_horizon_index",
        "target_date",
    ]
    board = (
        predictions.groupby(group_cols, as_index=False)
        .agg(
            n_heldout_plots=("plot_id", "nunique"),
            observed_total_yield_g=("target_yield_g", "sum"),
            predicted_total_yield_g=("prediction_g", "sum"),
            last_observed_total_yield_baseline_g=("baseline_last_observed_yield_g", "sum"),
            mean_observed_history_total_yield_baseline_g=("baseline_mean_observed_yield_g", "sum"),
        )
        .sort_values(["season", "model_label", "fold_id", "target_horizon_index"])
    )
    board["aggregate_error_g"] = board["predicted_total_yield_g"] - board["observed_total_yield_g"]
    board["aggregate_abs_error_g"] = board["aggregate_error_g"].abs()
    board["validation_scope"] = "held_out_cv_fold_aggregate_primary"
    return board


def build_aggregate_metrics(board: pd.DataFrame) -> pd.DataFrame:
    if board.empty:
        return pd.DataFrame()
    rows = []
    for (season, feature_set, model_label), group in board.groupby(["season", "feature_set", "model_label"], sort=True):
        y_true = group["observed_total_yield_g"].astype(float)
        y_pred = group["predicted_total_yield_g"].astype(float)
        baseline = group["last_observed_total_yield_baseline_g"].astype(float)
        mae = float(mean_absolute_error(y_true, y_pred))
        baseline_mae = float(mean_absolute_error(y_true, baseline))
        rows.append(
            {
                "season": season,
                "feature_set": feature_set,
                "model_label": model_label,
                "validation_scope": group["validation_scope"].iloc[0] if "validation_scope" in group.columns else "aggregate",
                "n_target_horizons": group["target_horizon_index"].nunique(),
                "n_aggregate_rows": len(group),
                "n_folds": group["fold_id"].nunique() if "fold_id" in group.columns else np.nan,
                "mean_heldout_plots_per_row": float(group["n_heldout_plots"].mean()) if "n_heldout_plots" in group.columns else np.nan,
                "aggregate_mae_g": mae,
                "aggregate_rmse_g": safe_rmse(y_true, y_pred),
                "aggregate_r2": safe_r2(y_true, y_pred),
                "aggregate_correlation": safe_corr(y_true, y_pred),
                "aggregate_bias_predicted_minus_observed_g": float((y_pred - y_true).mean()),
                "aggregate_mae_improvement_pct_vs_last_observed_baseline": ((baseline_mae - mae) / baseline_mae * 100) if baseline_mae > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["season", "aggregate_mae_g", "model_label"])


def build_timeliness_audit(board: pd.DataFrame) -> pd.DataFrame:
    if board.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["season", "feature_set", "model_label"]
    if "fold_id" in board.columns:
        group_cols.append("fold_id")
    for group_key, group in board.groupby(group_cols, sort=True):
        if len(group_cols) == 4:
            season, feature_set, model_label, fold_id = group_key
        else:
            season, feature_set, model_label = group_key
            fold_id = np.nan
        g = group.sort_values("target_horizon_index").reset_index(drop=True)
        if len(g) < 2:
            continue
        y_true = g["observed_total_yield_g"].astype(float)
        y_pred = g["predicted_total_yield_g"].astype(float)
        same_corr = safe_corr(y_true, y_pred)
        same_mae = float(mean_absolute_error(y_true, y_pred))

        prev_obs = y_true.shift(1)
        prev_mask = prev_obs.notna()
        pred_vs_prev_corr = safe_corr(prev_obs[prev_mask], y_pred[prev_mask]) if prev_mask.sum() >= 2 else np.nan
        pred_vs_prev_mae = float(mean_absolute_error(prev_obs[prev_mask], y_pred[prev_mask])) if prev_mask.sum() else np.nan

        last_hist = g["last_observed_total_yield_baseline_g"].astype(float)
        pred_vs_last_hist_mae = float(mean_absolute_error(last_hist, y_pred))
        target_vs_last_hist_mae = float(mean_absolute_error(last_hist, y_true))

        observed_direction = y_true.diff().dropna().apply(np.sign)
        predicted_direction = y_pred.diff().dropna().apply(np.sign)
        direction_accuracy = float((observed_direction.values == predicted_direction.values).mean()) if len(observed_direction) else np.nan

        lag_reasons = []
        if pd.notna(pred_vs_prev_corr) and pd.notna(same_corr) and pred_vs_prev_corr > same_corr + 0.10:
            lag_reasons.append("prediction correlates more with previous future horizon than target horizon")
        if pd.notna(pred_vs_prev_mae) and pred_vs_prev_mae + 1e-9 < same_mae * 0.90:
            lag_reasons.append("prediction is substantially closer to previous future horizon than target")
        if direction_accuracy == 0:
            lag_reasons.append("future direction changes are not captured")
        if "last observed" in model_label.lower():
            lag_reasons.append("last-observed baseline is reactive by definition")

        rows.append(
            {
                "season": season,
                "feature_set": feature_set,
                "model_label": model_label,
                "fold_id": fold_id,
                "validation_scope": g["validation_scope"].iloc[0] if "validation_scope" in g.columns else "aggregate",
                "n_target_horizons": len(g),
                "mean_heldout_plots_per_row": float(g["n_heldout_plots"].mean()) if "n_heldout_plots" in g.columns else np.nan,
                "same_horizon_corr": same_corr,
                "prediction_vs_previous_horizon_corr": pred_vs_prev_corr,
                "same_horizon_mae_g": same_mae,
                "prediction_vs_previous_horizon_mae_g": pred_vs_prev_mae,
                "prediction_vs_last_history_mae_g": pred_vs_last_hist_mae,
                "target_vs_last_history_mae_g": target_vs_last_hist_mae,
                "direction_accuracy": direction_accuracy,
                "lag_warning": bool(lag_reasons),
                "lag_note": "; ".join(lag_reasons),
            }
        )
    return pd.DataFrame(rows).sort_values(["season", "lag_warning", "same_horizon_mae_g"], ascending=[True, False, True])


def build_selection_summary(metrics: pd.DataFrame, aggregate_metrics: pd.DataFrame, timeliness: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not metrics.empty:
        overall = metrics.loc[metrics["target_horizon_index"].astype(str).eq("all")].copy()
        for season, group in overall.groupby("season", sort=True):
            g = group.sort_values("mae")
            best = g.iloc[0]
            best_non_baseline = g.loc[~g["model_label"].str.contains("baseline", case=False, na=False)].head(1)
            candidate = best_non_baseline.iloc[0] if not best_non_baseline.empty else best
            rows.append(
                {
                    "season": season,
                    "selection_scope": "plot_level_all_horizons",
                    "best_by_mae_model": best["model_label"],
                    "best_by_mae": best["mae"],
                    "best_non_baseline_model": candidate["model_label"],
                    "best_non_baseline_mae": candidate["mae"],
                    "recommendation": "use baseline as operational benchmark" if "baseline" in str(best["model_label"]).lower() else "candidate learned model beats simple baseline",
                }
            )
    if not aggregate_metrics.empty:
        for season, group in aggregate_metrics.groupby("season", sort=True):
            g = group.sort_values("aggregate_mae_g")
            best = g.iloc[0]
            rows.append(
                {
                    "season": season,
                    "selection_scope": "aggregate_future_curve",
                    "best_by_mae_model": best["model_label"],
                    "best_by_mae": best["aggregate_mae_g"],
                    "best_non_baseline_model": g.loc[~g["model_label"].str.contains("baseline", case=False, na=False), "model_label"].head(1).iloc[0]
                    if (~g["model_label"].str.contains("baseline", case=False, na=False)).any()
                    else best["model_label"],
                    "best_non_baseline_mae": g.loc[~g["model_label"].str.contains("baseline", case=False, na=False), "aggregate_mae_g"].head(1).iloc[0]
                    if (~g["model_label"].str.contains("baseline", case=False, na=False)).any()
                    else best["aggregate_mae_g"],
                    "recommendation": "inspect future-curve timeliness before selecting",
                }
            )
    return pd.DataFrame(rows)


def write_markdown_report(path: Path, metrics: pd.DataFrame, aggregate_metrics: pd.DataFrame, timeliness: pd.DataFrame, selection: pd.DataFrame, paper_comparison: pd.DataFrame, grouped_importance: pd.DataFrame, uncertainty_summary: pd.DataFrame) -> None:
    sections = [
        "# PheMuT sequence model report",
        "",
        "Independent paper-informed sequence modelling: first observed weeks as input history, future weeks as explicit horizons, validation within season over plots.",
        "",
        "## Selection summary",
        selection.to_string(index=False) if not selection.empty else "No selection summary available.",
        "",
        "## Plot-level metrics: best overall rows",
        metrics.loc[metrics["target_horizon_index"].astype(str).eq("all")].sort_values(["season", "mae"]).head(30).to_string(index=False)
        if not metrics.empty
        else "No metrics available.",
        "",
        "## Aggregate future-curve metrics (held-out fold primary)",
        aggregate_metrics.to_string(index=False) if not aggregate_metrics.empty else "No aggregate metrics available.",
        "",
        "## Paper comparison context",
        paper_comparison.to_string(index=False) if not paper_comparison.empty else "No paper comparison available.",
        "",
        "## Grouped feature importance",
        grouped_importance.head(40).to_string(index=False) if not grouped_importance.empty else "No feature importance available.",
        "",
        "## Uncertainty summary",
        uncertainty_summary.to_string(index=False) if not uncertainty_summary.empty else "No uncertainty summary available.",
        "",
        "## Timeliness audit",
        timeliness.to_string(index=False) if not timeliness.empty else "No timeliness audit available.",
        "",
    ]
    path.write_text("\n".join(sections), encoding="utf-8")


def write_duckdb_tables(db_path: Path, tables: dict[str, pd.DataFrame]) -> None:
    if duckdb is None:
        print("DuckDB not installed; skipping DB writes.")
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        for table_name, df in tables.items():
            if df is None or len(df.columns) == 0:
                print(f"Skipping DuckDB table {table_name}: DataFrame has no columns.")
                continue
            con.register("_tmp_df", df)
            con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _tmp_df")
            con.unregister("_tmp_df")


def main() -> None:
    args = parse_args()
    processed_dir = args.processed_dir
    sequence_path = processed_dir / "phemut_sequence_dataset.csv"
    if not sequence_path.exists():
        raise FileNotFoundError(f"Missing input file: {sequence_path}. Run 02_build_phemut_sequence_dataset.py first.")

    sequence_df = pd.read_csv(sequence_path)
    features = load_feature_set(processed_dir, args.feature_set, sequence_df)
    predictions, horizon_metrics, feature_importance_detail = evaluate_sequence_models(
        sequence_df=sequence_df,
        features=features,
        feature_set=args.feature_set,
        n_splits=args.n_splits,
        min_plots=args.min_plots_per_horizon,
        fast=args.fast,
    )
    metrics = build_overall_metrics(predictions, horizon_metrics)
    grouped_feature_importance = build_grouped_feature_importance(feature_importance_detail)
    weather_driver_summary = build_weather_driver_summary(feature_importance_detail)
    prediction_intervals = build_prediction_intervals(predictions)
    uncertainty_summary = build_uncertainty_summary(prediction_intervals)
    paper_comparison = build_paper_comparison(metrics)
    aggregate_fold_board = build_aggregate_fold_board(predictions)
    aggregate_metrics = build_aggregate_metrics(aggregate_fold_board)
    aggregate_reconstructed_board = build_reconstructed_aggregate_board(predictions)
    aggregate_reconstructed_metrics = build_aggregate_metrics(aggregate_reconstructed_board)
    timeliness = build_timeliness_audit(aggregate_fold_board)
    selection = build_selection_summary(metrics, aggregate_metrics, timeliness)
    stakeholder_summary = build_stakeholder_summary(metrics, aggregate_metrics, uncertainty_summary, grouped_feature_importance, paper_comparison)

    processed_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(processed_dir / "phemut_sequence_model_predictions.csv", index=False)
    metrics.to_csv(processed_dir / "phemut_sequence_model_metrics.csv", index=False)
    feature_importance_detail.to_csv(processed_dir / "phemut_sequence_feature_importance_detail.csv", index=False)
    grouped_feature_importance.to_csv(processed_dir / "phemut_sequence_grouped_feature_importance.csv", index=False)
    weather_driver_summary.to_csv(processed_dir / "phemut_sequence_weather_driver_summary.csv", index=False)
    prediction_intervals.to_csv(processed_dir / "phemut_sequence_prediction_intervals.csv", index=False)
    uncertainty_summary.to_csv(processed_dir / "phemut_sequence_uncertainty_summary.csv", index=False)
    paper_comparison.to_csv(processed_dir / "phemut_sequence_paper_comparison.csv", index=False)
    stakeholder_summary.to_csv(processed_dir / "phemut_sequence_stakeholder_summary.csv", index=False)
    aggregate_fold_board.to_csv(processed_dir / "phemut_sequence_aggregate_fold_board.csv", index=False)
    aggregate_metrics.to_csv(processed_dir / "phemut_sequence_aggregate_fold_metrics.csv", index=False)
    aggregate_metrics.to_csv(processed_dir / "phemut_sequence_aggregate_metrics.csv", index=False)
    aggregate_reconstructed_board.to_csv(processed_dir / "phemut_sequence_aggregate_reconstructed_board.csv", index=False)
    aggregate_reconstructed_metrics.to_csv(processed_dir / "phemut_sequence_aggregate_reconstructed_metrics.csv", index=False)
    # Backward-compatible illustrative board for older Streamlit versions. Do not use for model selection.
    aggregate_reconstructed_board.to_csv(processed_dir / "phemut_sequence_aggregate_board.csv", index=False)
    timeliness.to_csv(processed_dir / "phemut_sequence_timeliness_audit.csv", index=False)
    selection.to_csv(processed_dir / "phemut_sequence_model_selection_summary.csv", index=False)
    write_markdown_report(processed_dir / "phemut_sequence_model_report.md", metrics, aggregate_metrics, timeliness, selection, paper_comparison, grouped_feature_importance, uncertainty_summary)
    write_stakeholder_markdown(processed_dir / "phemut_sequence_stakeholder_summary.md", stakeholder_summary)

    if not args.no_db:
        write_duckdb_tables(
            args.db_path,
            {
                "phemut_sequence_model_predictions": predictions,
                "phemut_sequence_model_metrics": metrics,
                "phemut_sequence_feature_importance_detail": feature_importance_detail,
                "phemut_sequence_grouped_feature_importance": grouped_feature_importance,
                "phemut_sequence_weather_driver_summary": weather_driver_summary,
                "phemut_sequence_prediction_intervals": prediction_intervals,
                "phemut_sequence_uncertainty_summary": uncertainty_summary,
                "phemut_sequence_paper_comparison": paper_comparison,
                "phemut_sequence_stakeholder_summary": stakeholder_summary,
                "phemut_sequence_aggregate_fold_board": aggregate_fold_board,
                "phemut_sequence_aggregate_fold_metrics": aggregate_metrics,
                "phemut_sequence_aggregate_metrics": aggregate_metrics,
                "phemut_sequence_aggregate_reconstructed_board": aggregate_reconstructed_board,
                "phemut_sequence_aggregate_reconstructed_metrics": aggregate_reconstructed_metrics,
                "phemut_sequence_aggregate_board": aggregate_reconstructed_board,
                "phemut_sequence_timeliness_audit": timeliness,
                "phemut_sequence_model_selection_summary": selection,
            },
        )

    print("PheMuT sequence model benchmark complete.")
    print(f"Feature set: {args.feature_set}")
    print(f"Model features: {len(features)}")
    print(f"Prediction rows generated: {len(predictions):,}")
    if not metrics.empty:
        print("\nBest overall plot-level model by season:")
        overall = metrics.loc[metrics["target_horizon_index"].astype(str).eq("all")].sort_values(["season", "mae"])
        print(overall.groupby("season", as_index=False).head(8)[[
            "season",
            "model_label",
            "n_predictions",
            "n_target_horizons",
            "n_model_features",
            "mae",
            "rmse",
            "r2",
            "correlation",
            "bias_predicted_minus_observed_g",
            "mae_improvement_pct_vs_last_observed_baseline",
        ]].to_string(index=False))
    if not aggregate_metrics.empty:
        print("\nAggregate future-curve metrics:")
        print(aggregate_metrics[[
            "season",
            "model_label",
            "validation_scope",
            "n_target_horizons",
            "n_aggregate_rows",
            "n_folds",
            "mean_heldout_plots_per_row",
            "aggregate_mae_g",
            "aggregate_rmse_g",
            "aggregate_r2",
            "aggregate_correlation",
            "aggregate_bias_predicted_minus_observed_g",
            "aggregate_mae_improvement_pct_vs_last_observed_baseline",
        ]].to_string(index=False))
    if not paper_comparison.empty:
        print("\nComparison with published PheMuT paper metrics:")
        print(paper_comparison.to_string(index=False))
    if not grouped_feature_importance.empty:
        print("\nGrouped feature importance preview:")
        print(grouped_feature_importance.groupby(["season", "model_label"], as_index=False).head(5).to_string(index=False))
    if not uncertainty_summary.empty:
        print("\nUncertainty summary preview:")
        print(uncertainty_summary.groupby("season", as_index=False).head(5).to_string(index=False))
    if not timeliness.empty:
        print("\nTimeliness audit warnings:")
        preview = timeliness.loc[timeliness["lag_warning"]].head(20)
        if preview.empty:
            print("No lag warnings triggered.")
        else:
            print(preview[[
                "season",
                "model_label",
                "fold_id",
                "validation_scope",
                "same_horizon_corr",
                "prediction_vs_previous_horizon_corr",
                "same_horizon_mae_g",
                "prediction_vs_previous_horizon_mae_g",
                "direction_accuracy",
                "lag_warning",
                "lag_note",
            ]].to_string(index=False))
    print("\nOutputs written to:")
    for filename in [
        "phemut_sequence_model_predictions.csv",
        "phemut_sequence_model_metrics.csv",
        "phemut_sequence_feature_importance_detail.csv",
        "phemut_sequence_grouped_feature_importance.csv",
        "phemut_sequence_weather_driver_summary.csv",
        "phemut_sequence_prediction_intervals.csv",
        "phemut_sequence_uncertainty_summary.csv",
        "phemut_sequence_paper_comparison.csv",
        "phemut_sequence_stakeholder_summary.csv",
        "phemut_sequence_stakeholder_summary.md",
        "phemut_sequence_aggregate_fold_board.csv",
        "phemut_sequence_aggregate_fold_metrics.csv",
        "phemut_sequence_aggregate_reconstructed_board.csv",
        "phemut_sequence_aggregate_reconstructed_metrics.csv",
        "phemut_sequence_aggregate_metrics.csv",
        "phemut_sequence_aggregate_board.csv",
        "phemut_sequence_timeliness_audit.csv",
        "phemut_sequence_model_selection_summary.csv",
        "phemut_sequence_model_report.md",
    ]:
        print(f"- {processed_dir / filename}")


if __name__ == "__main__":
    main()
