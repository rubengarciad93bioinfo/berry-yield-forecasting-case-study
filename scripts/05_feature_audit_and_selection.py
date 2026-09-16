#!/usr/bin/env python3
"""
Audit and select strawberry forecasting features before retraining compact models.

This script does not train models. It studies the candidate predictors and writes a
clean, defensible feature set for the next modelling pass.

Main additions vs the previous audit:
- Nitrogen is kept only as the categorical variable `nitrogen_treatment`.
- Mean-vs-median choices are handled explicitly where both representations exist.
- Fruit-size/treatment daily features from script 07 are joined before auditing, so
  sample-level mean/median candidates can be compared.
- Redundant rolling/lag variants are compressed into compact state/trend features.
- Model feature sets are written explicitly for script 11.

Run from repo root:
    python scripts/10_feature_audit_and_selection.py
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
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


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed" / "strawberry"
DB_PATH = ROOT / "db" / "yield_forecasting.duckdb"

FORECAST_FEATURES_PATH = PROCESSED_DIR / "strawberry_forecast_features.csv"
TAGGED_FEATURES_PATH = PROCESSED_DIR / "strawberry_tagged_fruit_daily_features.csv"
TREATMENT_FRUIT_FEATURES_PATH = PROCESSED_DIR / "strawberry_fruit_size_treatment_daily_features.csv"

TARGET_COL = "next_fresh_matter_g"
REDUNDANCY_RHO_THRESHOLD = 0.85
HIGH_MISSING_THRESHOLD = 0.40
VERY_HIGH_MISSING_THRESHOLD = 0.80
HIGH_SKEW_THRESHOLD = 1.0
MODERATE_SKEW_THRESHOLD = 0.75
HIGH_OUTLIER_THRESHOLD = 0.10
MODERATE_OUTLIER_THRESHOLD = 0.05


@dataclass(frozen=True)
class FeatureChoice:
    feature: str
    feature_family: str
    model_role: str
    included_in_default_compact: bool
    included_in_tagged_optional: bool
    rationale: str


# ---------------------------------------------------------------------------
# IO helpers
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
    """Treatment-specific fruit-size summaries from script 07.

    These are useful because they can include both mean and median sample summaries.
    In the current dataset they mostly come from 2022, so the audit can decide whether
    they are too sparse for default modelling.
    """
    if not TREATMENT_FRUIT_FEATURES_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(TREATMENT_FRUIT_FEATURES_PATH)
    if df.empty or not {"year", "date", "nitrogen_treatment"}.issubset(df.columns):
        return pd.DataFrame()
    df = df.copy()
    df["date"] = parse_dates(df["date"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    # Keep script-07 sample summaries. Avoid duplicate source metadata if present.
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

    # If forecast already contains a column with the same name, keep the original and add suffix to script-07 version.
    overlapping = [c for c in tmp.columns if c in df.columns and c not in {"year", "join_date", "nitrogen_treatment"}]
    rename_map = {c: f"script07_{c}" for c in overlapping}
    tmp = tmp.rename(columns=rename_map)

    merged = df.merge(
        tmp,
        on=["year", "join_date", "nitrogen_treatment"],
        how="left",
        validate="many_to_one",
    )
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
        is_size_signal = lower.startswith("tagged_diameter_mm_") or lower.startswith("tagged_length_mm_")
        is_problematic = "fresh_matter" in lower or "lifespan" in lower
        if is_size_signal and not is_problematic:
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


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


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

    # Compact fresh matter: current level + trend. Avoid keeping several redundant rolling means as defaults.
    current_fresh = numeric_col(out, "current_fresh_matter_g")
    previous_fresh = numeric_col(out, "previous_fresh_matter_g")
    change_fresh = numeric_col(out, "change_fresh_matter_g")
    if current_fresh is not None and previous_fresh is not None:
        out["fresh_matter_trend_g"] = current_fresh - previous_fresh
    elif change_fresh is not None:
        out["fresh_matter_trend_g"] = change_fresh

    # Keep recent means for audit/redundancy only; not selected by default.
    for src in ["rolling_mean_3_fresh_matter_g", "rolling_mean_2_fresh_matter_g"]:
        if src in out.columns:
            out["fresh_matter_recent_mean_g"] = pd.to_numeric(out[src], errors="coerce")
            break

    # Fruit number compact features.
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

    # Dry matter compact features.
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

    # Tagged-fruit compact aliases. Prefer median aliases when present; final mean/median choice is audited separately.
    diameter_current = first_existing(
        [
            "fruit_tagged_diameter_mm_median",
            "fruit_tagged_diameter_mm_mean",
            "fruit_tagged_diameter_median",
            "fruit_tagged_diameter_mean",
        ],
        out,
    )
    length_current = first_existing(
        [
            "fruit_tagged_length_mm_median",
            "fruit_tagged_length_mm_mean",
            "fruit_tagged_length_median",
            "fruit_tagged_length_mean",
        ],
        out,
    )
    if diameter_current:
        out["tagged_fruit_diameter_current_mm"] = pd.to_numeric(out[diameter_current], errors="coerce")
    if length_current:
        out["tagged_fruit_length_current_mm"] = pd.to_numeric(out[length_current], errors="coerce")

    diameter_trend = first_existing(
        [c for c in out.columns if c.startswith("fruit_tagged_diameter") and ("change" in c or "trend" in c or "diff" in c)],
        out,
    )
    length_trend = first_existing(
        [c for c in out.columns if c.startswith("fruit_tagged_length") and ("change" in c or "trend" in c or "diff" in c)],
        out,
    )
    if diameter_trend:
        out["tagged_fruit_diameter_trend_mm"] = pd.to_numeric(out[diameter_trend], errors="coerce")
    if length_trend:
        out["tagged_fruit_length_trend_mm"] = pd.to_numeric(out[length_trend], errors="coerce")

    tagged_n_obs = first_existing(
        [
            c
            for c in out.columns
            if c.startswith("fruit_tagged_")
            and ("n_non_missing" in c or "n_observed" in c or "n_observ" in c or "observations" in c)
            and ("diameter" in c or "length" in c)
        ],
        out,
    )
    if tagged_n_obs:
        out["tagged_fruit_n_observed"] = pd.to_numeric(out[tagged_n_obs], errors="coerce")

    return out


# ---------------------------------------------------------------------------
# Metadata and audit
# ---------------------------------------------------------------------------


def is_problematic_tagged_input(col: str) -> bool:
    lower = col.lower()
    return "tagged" in lower and ("fresh_matter" in lower or "lifespan" in lower)



def is_forbidden_feature(col: str) -> bool:
    lower = col.lower()
    forbidden_exact = {
        TARGET_COL,
        "current_observation_date",
        "target_date",
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
    if col in forbidden_exact:
        return True
    if col in {"forecast_horizon_days", "nitrogen_treatment", "treatment_n_level"}:
        return False
    if lower.startswith("next_") or lower.startswith("target_"):
        return True
    if "predicted" in lower or "observed" in lower:
        return True
    if any(metric in lower for metric in ["mae", "rmse", "r2", "bias", "error"]):
        return True
    if is_problematic_tagged_input(col):
        return True
    return False



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
    if lower == "year":
        return "year metadata"
    return "other"



def humanize_feature(feature: str) -> str:
    manual = {
        "nitrogen_treatment": "Nitrogen treatment category",
        "treatment_n_level": "Nitrogen dose numeric encoding",
        "forecast_horizon_days": "Forecast lead time",
        "origin_day_of_year": "Forecast day of year",
        "origin_month": "Forecast month",
        "origin_week_of_year": "Forecast week of year",
        "current_fresh_matter_g": "Current fresh matter",
        "previous_fresh_matter_g": "Previous fresh matter",
        "rolling_mean_2_fresh_matter_g": "2-observation fresh-matter average",
        "rolling_mean_3_fresh_matter_g": "3-observation fresh-matter average",
        "change_fresh_matter_g": "Fresh-matter change since previous observation",
        "fresh_matter_recent_mean_g": "Recent fresh-matter average",
        "fresh_matter_trend_g": "Fresh-matter trend since previous observation",
        "current_fruit_number": "Current fruit count",
        "previous_fruit_number": "Previous fruit count",
        "rolling_mean_2_fruit_number": "2-observation fruit-count average",
        "rolling_mean_3_fruit_number": "3-observation fruit-count average",
        "change_fruit_number": "Fruit-count change since previous observation",
        "fruit_number_recent_mean": "Recent fruit-count average",
        "fruit_number_trend": "Fruit-count trend since previous observation",
        "current_dry_matter_g": "Current dry matter",
        "previous_dry_matter_g": "Previous dry matter",
        "rolling_mean_2_dry_matter_g": "2-observation dry-matter average",
        "rolling_mean_3_dry_matter_g": "3-observation dry-matter average",
        "change_dry_matter_g": "Dry-matter change since previous observation",
        "dry_matter_recent_mean_g": "Recent dry-matter average",
        "dry_matter_trend_g": "Dry-matter trend since previous observation",
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
        "n_fruit_size_samples": "Number of fruit-size samples",
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
    label = re.sub(r"\bmm\b", "mm", label)
    label = re.sub(r"\bg\b", "g", label)
    return label[:1].upper() + label[1:]



def candidate_features_for_audit(df: pd.DataFrame) -> list[str]:
    out = []
    for col in df.columns:
        if col == TARGET_COL or is_forbidden_feature(col):
            continue
        if col in {"current_observation_date", "target_date", "date"}:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            out.append(col)
        else:
            if df[col].nunique(dropna=True) <= 30:
                out.append(col)
    if "treatment_n_level" in df.columns and "treatment_n_level" not in out:
        out.append("treatment_n_level")
    return sorted(dict.fromkeys(out))



def spearman_corr(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    pair = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(pair) < 4 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return np.nan, int(len(pair))
    return float(pair["x"].corr(pair["y"], method="spearman")), int(len(pair))



def categorical_eta_squared(x: pd.Series, y: pd.Series) -> tuple[float, int, str]:
    tmp = pd.DataFrame({"x": x.astype("string"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(tmp) < 4 or tmp["x"].nunique() < 2:
        return np.nan, int(len(tmp)), ""
    grand_mean = tmp["y"].mean()
    ss_between = tmp.groupby("x")["y"].agg(lambda s: len(s) * (s.mean() - grand_mean) ** 2).sum()
    ss_total = ((tmp["y"] - grand_mean) ** 2).sum()
    eta = float(ss_between / ss_total) if ss_total > 0 else np.nan
    means = tmp.groupby("x")["y"].mean().round(3).to_dict()
    return eta, int(len(tmp)), json.dumps(means, sort_keys=True)



def audit_feature(df: pd.DataFrame, feature: str) -> dict[str, object]:
    s = df[feature]
    y = df[TARGET_COL]
    family = infer_feature_family(feature)
    years = pd.to_numeric(df["year"], errors="coerce") if "year" in df.columns else pd.Series(index=df.index, dtype=float)

    base = {
        "feature": feature,
        "human_name": humanize_feature(feature),
        "feature_family": family,
        "dtype": str(s.dtype),
        "n_rows": int(len(df)),
        "missing_share": float(s.isna().mean()),
        "missing_share_2022": float(s[years == 2022].isna().mean()) if (years == 2022).any() else np.nan,
        "missing_share_2023": float(s[years == 2023].isna().mean()) if (years == 2023).any() else np.nan,
        "n_unique": int(s.nunique(dropna=True)),
    }

    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        num = pd.to_numeric(s, errors="coerce")
        q = num.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        q10, q25, q50, q75, q90 = [float(q.loc[p]) if not pd.isna(q.loc[p]) else np.nan for p in [0.10, 0.25, 0.50, 0.75, 0.90]]
        iqr = q75 - q25 if not (np.isnan(q75) or np.isnan(q25)) else np.nan
        if not np.isnan(iqr) and iqr > 0:
            low, high = q25 - 1.5 * iqr, q75 + 1.5 * iqr
            outlier_share = float(((num < low) | (num > high)).mean())
        else:
            outlier_share = 0.0
        rho, n_corr = spearman_corr(num, y)
        rho_2022, n_corr_2022 = spearman_corr(num[years == 2022], y[years == 2022]) if (years == 2022).any() else (np.nan, 0)
        rho_2023, n_corr_2023 = spearman_corr(num[years == 2023], y[years == 2023]) if (years == 2023).any() else (np.nan, 0)
        skew = float(num.skew()) if num.notna().sum() > 2 else np.nan
        mean = float(num.mean()) if num.notna().any() else np.nan
        median = float(num.median()) if num.notna().any() else np.nan
        if not np.isnan(mean) and abs(mean) > 1e-12 and not np.isnan(median):
            mean_median_relative_gap = float(abs(mean - median) / abs(mean))
        else:
            mean_median_relative_gap = np.nan
        base.update(
            {
                "feature_kind": "numeric",
                "mean": mean,
                "median": median,
                "std": float(num.std()) if num.notna().sum() > 1 else np.nan,
                "p10": q10,
                "p25": q25,
                "p50_median": q50,
                "p75": q75,
                "p90": q90,
                "iqr": float(iqr) if not np.isnan(iqr) else np.nan,
                "skewness": skew,
                "abs_skewness": abs(skew) if not np.isnan(skew) else np.nan,
                "outlier_share_iqr_rule": outlier_share,
                "mean_median_relative_gap": mean_median_relative_gap,
                "spearman_corr_with_target": rho,
                "abs_spearman_corr_with_target": abs(rho) if not np.isnan(rho) else np.nan,
                "n_corr_with_target": n_corr,
                "spearman_corr_with_target_2022": rho_2022,
                "spearman_corr_with_target_2023": rho_2023,
                "abs_year_to_year_corr_gap": abs(rho_2022 - rho_2023) if not (np.isnan(rho_2022) or np.isnan(rho_2023)) else np.nan,
                "n_corr_with_target_2022": n_corr_2022,
                "n_corr_with_target_2023": n_corr_2023,
                "target_association_metric": "spearman_rho",
                "category_target_means_json": "",
            }
        )
    else:
        eta, n_eta, means_json = categorical_eta_squared(s, y)
        base.update(
            {
                "feature_kind": "categorical",
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "p10": np.nan,
                "p25": np.nan,
                "p50_median": np.nan,
                "p75": np.nan,
                "p90": np.nan,
                "iqr": np.nan,
                "skewness": np.nan,
                "abs_skewness": np.nan,
                "outlier_share_iqr_rule": np.nan,
                "mean_median_relative_gap": np.nan,
                "spearman_corr_with_target": np.nan,
                "abs_spearman_corr_with_target": eta,
                "n_corr_with_target": n_eta,
                "spearman_corr_with_target_2022": np.nan,
                "spearman_corr_with_target_2023": np.nan,
                "abs_year_to_year_corr_gap": np.nan,
                "n_corr_with_target_2022": np.nan,
                "n_corr_with_target_2023": np.nan,
                "target_association_metric": "eta_squared_category_vs_target",
                "category_target_means_json": means_json,
            }
        )
    return base



def build_individual_audit(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    audit = pd.DataFrame([audit_feature(df, f) for f in features if f in df.columns])
    if audit.empty:
        return audit
    return audit.sort_values(
        ["feature_family", "abs_spearman_corr_with_target", "missing_share"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Mean / median representative audit
# ---------------------------------------------------------------------------


def canonical_mean_median_stem(feature: str) -> tuple[str, str] | None:
    """Return (stem, summary_type) for mean/median feature names."""
    if feature.endswith("_mean"):
        return feature[: -len("_mean")], "mean"
    if feature.endswith("_median"):
        return feature[: -len("_median")], "median"
    if feature.startswith("mean_"):
        return feature[len("mean_") :], "mean"
    if feature.startswith("median_"):
        return feature[len("median_") :], "median"
    return None



def build_mean_median_representative_choices(df: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame()
    by_feature = {row["feature"]: row for row in audit.to_dict("records")}
    groups: dict[str, dict[str, str]] = defaultdict(dict)
    for feature in audit.loc[audit["feature_kind"] == "numeric", "feature"]:
        parsed = canonical_mean_median_stem(str(feature))
        if parsed is not None:
            stem, summary_type = parsed
            groups[stem][summary_type] = str(feature)

    rows: list[dict[str, object]] = []
    for stem, members in sorted(groups.items()):
        mean_f = members.get("mean")
        median_f = members.get("median")
        if not mean_f or not median_f:
            continue
        mean_row = by_feature[mean_f]
        median_row = by_feature[median_f]
        mean_missing = float(mean_row.get("missing_share", 1.0) or 1.0)
        median_missing = float(median_row.get("missing_share", 1.0) or 1.0)
        mean_assoc = float(mean_row.get("abs_spearman_corr_with_target", 0.0) or 0.0)
        median_assoc = float(median_row.get("abs_spearman_corr_with_target", 0.0) or 0.0)
        mean_skew = abs(float(mean_row.get("skewness", 0.0) or 0.0))
        mean_outliers = float(mean_row.get("outlier_share_iqr_rule", 0.0) or 0.0)

        if mean_missing > median_missing + 0.05:
            chosen, reason = median_f, "Median has meaningfully lower missingness."
        elif median_missing > mean_missing + 0.05:
            chosen, reason = mean_f, "Mean has meaningfully lower missingness."
        elif mean_skew >= HIGH_SKEW_THRESHOLD or mean_outliers >= HIGH_OUTLIER_THRESHOLD:
            chosen, reason = median_f, "Mean-distribution is skewed or outlier-prone; median is safer."
        elif (mean_skew >= MODERATE_SKEW_THRESHOLD or mean_outliers >= MODERATE_OUTLIER_THRESHOLD) and median_assoc >= 0.90 * mean_assoc:
            chosen, reason = median_f, "Distribution is moderately skewed/outlier-prone and median keeps similar target association."
        elif median_assoc > mean_assoc + 0.03:
            chosen, reason = median_f, "Median has stronger target association without a missingness penalty."
        else:
            chosen, reason = mean_f, "Mean retained: no strong skew/outlier penalty and association is competitive."

        rows.append(
            {
                "stem": stem,
                "mean_feature": mean_f,
                "median_feature": median_f,
                "chosen_feature": chosen,
                "chosen_summary": "median" if chosen == median_f else "mean",
                "mean_missing_share": mean_missing,
                "median_missing_share": median_missing,
                "mean_abs_target_association": mean_assoc,
                "median_abs_target_association": median_assoc,
                "mean_abs_skewness": mean_skew,
                "mean_outlier_share": mean_outliers,
                "decision_reason": reason,
            }
        )
    return pd.DataFrame(rows)



def mean_median_choice_lookup(choices: pd.DataFrame) -> dict[str, str]:
    if choices.empty:
        return {}
    lookup = {}
    for _, row in choices.iterrows():
        lookup[str(row["mean_feature"])] = str(row["chosen_feature"])
        lookup[str(row["median_feature"])] = str(row["chosen_feature"])
    return lookup



def choose_best_representative(
    df: pd.DataFrame,
    candidates: list[str],
    audit: pd.DataFrame,
    mean_median_choices: pd.DataFrame,
    max_missing_for_default: float = 0.70,
) -> str | None:
    """Choose one candidate variable, respecting mean/median audit and basic quality."""
    present = [c for c in candidates if c in df.columns]
    if not present:
        return None
    by_feature = {row["feature"]: row for row in audit.to_dict("records")}
    mm_lookup = mean_median_choice_lookup(mean_median_choices)

    # If a present candidate belongs to a mean/median pair, replace it with the audited representative.
    present = [mm_lookup.get(c, c) for c in present]
    present = sorted(dict.fromkeys(present))

    def score(feature: str) -> tuple[float, float, float, int]:
        row = by_feature.get(feature, {})
        missing = float(row.get("missing_share", 1.0) or 1.0)
        assoc = float(row.get("abs_spearman_corr_with_target", 0.0) or 0.0)
        skew = abs(float(row.get("skewness", 0.0) or 0.0))
        outliers = float(row.get("outlier_share_iqr_rule", 0.0) or 0.0)
        robust_bonus = 1 if ("median" in feature or "_trend" in feature or "current_" in feature) else 0
        if missing > max_missing_for_default:
            return (-999.0, assoc, -missing, robust_bonus)
        return (assoc - 0.6 * missing - 0.10 * skew - 0.25 * outliers + 0.03 * robust_bonus, assoc, -missing, robust_bonus)

    ranked = sorted(present, key=score, reverse=True)
    return ranked[0] if score(ranked[0])[0] > -900 else None


# ---------------------------------------------------------------------------
# Redundancy analysis
# ---------------------------------------------------------------------------


class UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def clusters(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for item in self.parent:
            out[self.find(item)].append(item)
        return out



def build_redundancy_pairs(df: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame()
    numeric_features = audit.loc[audit["feature_kind"] == "numeric", "feature"].tolist()
    numeric_features = [f for f in numeric_features if f in df.columns and df[f].nunique(dropna=True) > 1]
    rows = []
    for i, a in enumerate(numeric_features):
        x = pd.to_numeric(df[a], errors="coerce")
        for b in numeric_features[i + 1 :]:
            y = pd.to_numeric(df[b], errors="coerce")
            pair = pd.DataFrame({"a": x, "b": y}).dropna()
            if len(pair) < 10 or pair["a"].nunique() < 2 or pair["b"].nunique() < 2:
                continue
            rho = float(pair["a"].corr(pair["b"], method="spearman"))
            if abs(rho) >= REDUNDANCY_RHO_THRESHOLD:
                rows.append(
                    {
                        "feature_a": a,
                        "feature_b": b,
                        "human_name_a": humanize_feature(a),
                        "human_name_b": humanize_feature(b),
                        "family_a": infer_feature_family(a),
                        "family_b": infer_feature_family(b),
                        "spearman_rho": rho,
                        "abs_spearman_rho": abs(rho),
                        "n_pairwise_non_missing": int(len(pair)),
                    }
                )
    return pd.DataFrame(rows).sort_values("abs_spearman_rho", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()



def representative_score(feature: str, audit_by_feature: dict[str, dict[str, object]]) -> tuple[float, float, float, int]:
    row = audit_by_feature.get(feature, {})
    missing = float(row.get("missing_share", 1.0) or 1.0)
    assoc = float(row.get("abs_spearman_corr_with_target", 0.0) or 0.0)
    outliers = float(row.get("outlier_share_iqr_rule", 0.0) or 0.0)
    skew = abs(float(row.get("skewness", 0.0) or 0.0))
    preferred_patterns = ["_trend", "current_", "forecast_horizon_days", "cumulative_gdd", "solar_radiation_7d", "tmean_7d", "median"]
    preference = sum(1 for p in preferred_patterns if p in feature)
    score = assoc - 0.65 * missing - 0.20 * outliers - 0.05 * skew + 0.025 * preference
    return (score, assoc, -missing, preference)



def build_redundancy_clusters(pairs: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    features = sorted(set(pairs["feature_a"]).union(pairs["feature_b"]))
    uf = UnionFind(features)
    for _, row in pairs.iterrows():
        uf.union(str(row["feature_a"]), str(row["feature_b"]))
    audit_by_feature = {row["feature"]: row for row in audit.to_dict("records")}
    rows = []
    for cluster_id, cluster_features in enumerate(uf.clusters().values(), start=1):
        if len(cluster_features) < 2:
            continue
        ranked = sorted(cluster_features, key=lambda f: representative_score(f, audit_by_feature), reverse=True)
        rows.append(
            {
                "cluster_id": cluster_id,
                "n_features": len(cluster_features),
                "representative_feature": ranked[0],
                "representative_human_name": humanize_feature(ranked[0]),
                "feature_family_summary": "; ".join(sorted(set(infer_feature_family(f) for f in cluster_features))),
                "features": "; ".join(ranked),
                "human_names": "; ".join(humanize_feature(f) for f in ranked),
                "selection_note": "High-correlation cluster. Keep one representative unless there is a strong agronomic reason to keep more.",
            }
        )
    return pd.DataFrame(rows).sort_values(["n_features", "cluster_id"], ascending=[False, True]).reset_index(drop=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------


def choose_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)



def add_choice(choices: list[FeatureChoice], feature: str | None, role: str, rationale: str, default: bool = True, optional: bool = False) -> None:
    if not feature:
        return
    choices.append(FeatureChoice(feature, infer_feature_family(feature), role, default, optional, rationale))



def choose_weather_features(df: pd.DataFrame) -> list[FeatureChoice]:
    choices: list[FeatureChoice] = []
    weather_defs = [
        (["tmean_7d", "tmean", "Tmean", "tmean_c", "mean_tmean_c"], "recent_temperature", "Recent mean temperature is kept as one compact short-term temperature signal."),
        (["gdd_base_10", "daily_gdd_base_10", "gdd"], "recent_thermal_time", "Daily/recent GDD captures short-term heat accumulation at the forecast origin."),
        (["cumulative_gdd_base_10", "cumulative_gdd", "cum_gdd_base_10"], "season_thermal_time", "Cumulative GDD captures crop-development stage during the season."),
        (["solar_radiation_7d", "radiation_7d", "rad_7d", "RAD_7d"], "recent_solar_radiation", "Recent solar radiation is a plausible berry growth/development signal."),
        (["cumulative_solar_radiation", "cum_solar_radiation", "cumulative_rad"], "season_solar_radiation", "Cumulative radiation captures seasonal energy exposure."),
        (["rhmean_7d", "RHmean_7d", "rhmean", "RHmean", "relative_humidity_7d"], "recent_humidity", "Humidity is kept as one recent summary rather than multiple highly related RH columns."),
    ]
    for candidates, role, rationale in weather_defs:
        add_choice(choices, choose_existing(df, candidates), role, rationale)
    return choices



def choose_fruit_size_features(df: pd.DataFrame, audit: pd.DataFrame, mm_choices: pd.DataFrame) -> list[FeatureChoice]:
    choices: list[FeatureChoice] = []
    groups = [
        (
            ["median_fruit_diameter_mm", "mean_fruit_diameter_mm", "sample_diameter_mm_median", "sample_diameter_mm_mean", "script07_sample_diameter_mm_median", "script07_sample_diameter_mm_mean"],
            "fruit_diameter",
            "One diameter summary chosen after mean-vs-median and missingness audit.",
        ),
        (
            ["median_fruit_length_mm", "mean_fruit_length_mm", "sample_length_mm_median", "sample_length_mm_mean", "script07_sample_length_mm_median", "script07_sample_length_mm_mean"],
            "fruit_length",
            "One length summary chosen after mean-vs-median and missingness audit.",
        ),
        (
            [
                "median_individual_fruit_fresh_weight_g",
                "mean_individual_fruit_fresh_weight_g",
                "sample_individual_fruit_fresh_weight_g_median",
                "sample_individual_fruit_fresh_weight_g_mean",
                "script07_sample_individual_fruit_fresh_weight_g_median",
                "script07_sample_individual_fruit_fresh_weight_g_mean",
            ],
            "individual_fruit_weight",
            "One individual-fruit-weight summary chosen after mean-vs-median and missingness audit.",
        ),
    ]
    for candidates, role, rationale in groups:
        chosen = choose_best_representative(df, candidates, audit, mm_choices, max_missing_for_default=0.70)
        add_choice(choices, chosen, role, rationale)
    return choices



def choose_tagged_features(df: pd.DataFrame, audit: pd.DataFrame, mm_choices: pd.DataFrame) -> list[FeatureChoice]:
    choices: list[FeatureChoice] = []
    # Compact aliases were created in add_compact_engineered_features. Keep only a small optional set.
    for feature, role in [
        ("tagged_fruit_diameter_current_mm", "tagged_diameter"),
        ("tagged_fruit_length_current_mm", "tagged_length"),
        ("tagged_fruit_diameter_trend_mm", "tagged_diameter_trend"),
        ("tagged_fruit_length_trend_mm", "tagged_length_trend"),
        ("tagged_fruit_n_observed", "tagged_n_observed"),
    ]:
        if feature in df.columns:
            choices.append(
                FeatureChoice(
                    feature,
                    "tagged fruit development",
                    role,
                    False,
                    True,
                    "Optional tagged-fruit feature. It is tested separately because it is date-level, not treatment-specific.",
                )
            )
    return choices



def build_selected_feature_choices(df: pd.DataFrame, audit: pd.DataFrame, mm_choices: pd.DataFrame) -> list[FeatureChoice]:
    choices: list[FeatureChoice] = []

    add_choice(
        choices,
        "nitrogen_treatment" if "nitrogen_treatment" in df.columns else None,
        "management_group",
        "Use nitrogen only as a categorical treatment group; treatment_n_level is dropped to avoid duplicate encoding.",
    )
    for feature, role, rationale in [
        ("forecast_horizon_days", "forecast_lead_time", "The lead time affects forecast difficulty."),
        ("origin_day_of_year", "season_timing", "Day-of-season gives a compact seasonal timing signal."),
    ]:
        if feature in df.columns:
            add_choice(choices, feature, role, rationale)

    choices.extend(choose_weather_features(df))

    for feature, role, rationale in [
        ("current_fruit_number", "current_fruit_load", "Current fruit count is a leading crop-load signal."),
        ("fruit_number_trend", "fruit_load_trend", "Fruit-count trend captures rising/falling crop load without multiple redundant rolling averages."),
        ("current_dry_matter_g", "current_dry_matter", "Dry matter is a crop-status signal distinct from fresh matter."),
        ("dry_matter_trend_g", "dry_matter_trend", "Dry-matter trend captures recent development without multiple redundant rolling averages."),
    ]:
        if feature in df.columns:
            add_choice(choices, feature, role, rationale)

    choices.extend(choose_fruit_size_features(df, audit, mm_choices))

    for feature, role, rationale in [
        ("current_fresh_matter_g", "current_production_state", "Current fresh matter anchors near-term production level."),
        ("fresh_matter_trend_g", "recent_production_trend", "Fresh-matter trend captures acceleration or decline without retaining several redundant lag/rolling columns."),
    ]:
        if feature in df.columns:
            add_choice(choices, feature, role, rationale)

    choices.extend(choose_tagged_features(df, audit, mm_choices))

    seen = set()
    unique = []
    for c in choices:
        if c.feature not in seen:
            unique.append(c)
            seen.add(c.feature)
    return unique



def redundancy_cluster_map(clusters: pd.DataFrame) -> dict[str, dict[str, object]]:
    mapping = {}
    if clusters.empty:
        return mapping
    for _, row in clusters.iterrows():
        for f in [x.strip() for x in str(row["features"]).split(";") if x.strip()]:
            mapping[f] = row.to_dict()
    return mapping



def build_recommendations(audit: pd.DataFrame, clusters: pd.DataFrame, choices: list[FeatureChoice], mm_choices: pd.DataFrame) -> pd.DataFrame:
    selected_default = {c.feature for c in choices if c.included_in_default_compact}
    selected_optional = {c.feature for c in choices if c.included_in_tagged_optional}
    choice_by_feature = {c.feature: c for c in choices}
    cluster_by_feature = redundancy_cluster_map(clusters)
    mm_lookup = mean_median_choice_lookup(mm_choices)

    rows = []
    for row in audit.to_dict("records"):
        feature = row["feature"]
        missing = float(row.get("missing_share", np.nan))
        skew = abs(float(row.get("skewness", 0.0) or 0.0)) if row.get("feature_kind") == "numeric" else np.nan
        outliers = float(row.get("outlier_share_iqr_rule", 0.0) or 0.0) if row.get("feature_kind") == "numeric" else np.nan
        in_cluster = feature in cluster_by_feature
        representative = cluster_by_feature.get(feature, {}).get("representative_feature", "")
        mm_chosen = mm_lookup.get(str(feature), "")

        if feature == "treatment_n_level":
            action = "drop_duplicate_encoding"
            reason = "Nitrogen is kept as categorical nitrogen_treatment, not duplicated as numeric treatment_n_level."
        elif feature in selected_default:
            action = "keep_default_compact"
            reason = choice_by_feature[feature].rationale
            if mm_chosen and mm_chosen == feature:
                reason += " This feature is the selected mean/median representative for its measurement."
        elif feature in selected_optional:
            action = "keep_optional_tagged_layer"
            reason = choice_by_feature[feature].rationale
        elif mm_chosen and mm_chosen != feature:
            action = "drop_mean_median_alternative"
            reason = f"Mean/median audit selected {mm_chosen} as the representative instead."
        elif feature.startswith("fruit_tagged_") or feature.startswith("tagged_fruit_"):
            action = "review_optional_tagged_layer"
            reason = "Tagged-fruit signal is audited as experimental context; only compact aliases are tested by default."
        elif feature in {
            "previous_fresh_matter_g",
            "rolling_mean_2_fresh_matter_g",
            "rolling_mean_3_fresh_matter_g",
            "change_fresh_matter_g",
            "fresh_matter_recent_mean_g",
        }:
            action = "drop_or_review_redundant_fresh_matter_history"
            reason = "Fresh-matter history is compressed to current_fresh_matter_g + fresh_matter_trend_g for interpretability."
        elif in_cluster:
            action = "drop_or_review_redundant"
            reason = f"Highly correlated with other predictors. Suggested cluster representative: {representative}."
        elif not np.isnan(missing) and missing >= VERY_HIGH_MISSING_THRESHOLD:
            action = "drop_high_missingness"
            reason = "Too many missing values for a small rolling dataset."
        elif not np.isnan(missing) and missing >= HIGH_MISSING_THRESHOLD:
            action = "review_missingness"
            reason = "Missingness is high; keep only with strong agronomic justification or robust imputation."
        elif row.get("feature_kind") == "numeric" and ((not np.isnan(skew) and skew > HIGH_SKEW_THRESHOLD) or (not np.isnan(outliers) and outliers > HIGH_OUTLIER_THRESHOLD)):
            action = "review_mean_median_or_outliers"
            reason = "Distribution is skewed/outlier-prone; median/percentile representation may be safer than a mean feature."
        else:
            action = "available_not_selected_initially"
            reason = "Available candidate but not selected in the compact first-pass feature set."

        rows.append(
            {
                **row,
                "selected_for_default_compact_model": feature in selected_default,
                "selected_for_optional_tagged_model": feature in selected_optional,
                "mean_median_selected_representative": mm_chosen,
                "recommended_action": action,
                "recommendation_reason": reason,
                "redundancy_cluster_representative": representative,
            }
        )
    rec = pd.DataFrame(rows)
    if rec.empty:
        return rec
    priority = {
        "keep_default_compact": 0,
        "keep_optional_tagged_layer": 1,
        "drop_duplicate_encoding": 2,
        "drop_mean_median_alternative": 3,
        "drop_or_review_redundant_fresh_matter_history": 4,
        "drop_or_review_redundant": 5,
        "review_mean_median_or_outliers": 6,
        "review_missingness": 7,
        "drop_high_missingness": 8,
        "review_optional_tagged_layer": 9,
        "available_not_selected_initially": 10,
    }
    rec["action_sort"] = rec["recommended_action"].map(priority).fillna(99)
    return rec.sort_values(["action_sort", "feature_family", "feature"]).drop(columns=["action_sort"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model feature sets for script 11
# ---------------------------------------------------------------------------


def build_model_feature_sets(df: pd.DataFrame, choices: list[FeatureChoice]) -> pd.DataFrame:
    def default_family(family: str) -> list[str]:
        return [c.feature for c in choices if c.feature_family == family and c.included_in_default_compact and c.feature in df.columns]

    calendar = default_family("calendar / lead time")
    nitrogen = default_family("nitrogen treatment")
    weather = default_family("weather / thermal time")
    fruit_load = default_family("fruit load")
    fruit_size = default_family("fruit size sample")
    dry_current = default_family("current dry matter state")
    dry_traj = default_family("dry matter trajectory")
    fresh_current = default_family("current fresh matter state")
    fresh_traj = default_family("fresh matter trajectory")
    optional_tagged = [c.feature for c in choices if c.included_in_tagged_optional and c.feature in df.columns]

    compact_engineered = {
        "fresh_matter_recent_mean_g",
        "fresh_matter_trend_g",
        "fruit_number_recent_mean",
        "fruit_number_trend",
        "dry_matter_recent_mean_g",
        "dry_matter_trend_g",
        "tagged_fruit_diameter_current_mm",
        "tagged_fruit_length_current_mm",
        "tagged_fruit_diameter_trend_mm",
        "tagged_fruit_length_trend_mm",
        "tagged_fruit_n_observed",
    }
    full_reference = []
    for col in candidate_features_for_audit(df):
        if col in {"treatment_n_level", "year"}:
            continue
        if col.startswith("fruit_tagged_") or col.startswith("tagged_fruit_"):
            continue
        if col in compact_engineered:
            continue
        full_reference.append(col)

    model_defs = [
        ("M0", "Persistence baseline", "baseline_rule", ["current_fresh_matter_g"] if "current_fresh_matter_g" in df.columns else [], "Rule: next fresh matter equals current fresh matter. Not trained as RF."),
        ("M0b", "Treatment mean baseline", "baseline_rule", ["nitrogen_treatment"] if "nitrogen_treatment" in df.columns else [], "Rule: use historical target mean for the treatment."),
        ("M1", "Calendar + nitrogen RF", "random_forest", sorted(dict.fromkeys(calendar + nitrogen)), "Tests timing and treatment category only."),
        ("M2", "+ weather / GDD RF", "random_forest", sorted(dict.fromkeys(calendar + nitrogen + weather)), "Adds weather, radiation and thermal-time context."),
        ("M3", "+ leading crop indicators RF", "random_forest", sorted(dict.fromkeys(calendar + nitrogen + weather + fruit_load + dry_current + dry_traj + fruit_size)), "Adds leading crop indicators while excluding direct fresh-matter history."),
        ("M4", "Compact crop/weather RF", "random_forest", sorted(dict.fromkeys(calendar + nitrogen + weather + fruit_load + dry_current + dry_traj + fruit_size + fresh_current + fresh_traj)), "Adds current fresh matter and a compact fresh-matter trend."),
        ("M5", "Compact crop/weather + tagged-fruit RF", "random_forest", sorted(dict.fromkeys(calendar + nitrogen + weather + fruit_load + dry_current + dry_traj + fruit_size + fresh_current + fresh_traj + optional_tagged)), "Optional test: add compact tagged-fruit diameter/length context."),
        ("M6", "Full raw RF reference", "random_forest_reference", sorted(dict.fromkeys(full_reference)), "Reference only: broader raw set without treatment_n_level or tagged layer; may contain redundant lags."),
    ]

    rows = []
    for model_id, model_label, model_type, features, note in model_defs:
        if features:
            for order, feature in enumerate(features, start=1):
                rows.append(
                    {
                        "model_id": model_id,
                        "model_label": model_label,
                        "model_type": model_type,
                        "feature_order": order,
                        "feature": feature,
                        "human_name": humanize_feature(feature),
                        "feature_family": infer_feature_family(feature),
                        "note": note,
                    }
                )
        else:
            rows.append({"model_id": model_id, "model_label": model_label, "model_type": model_type, "feature_order": 0, "feature": "", "human_name": "", "feature_family": "", "note": note})
    return pd.DataFrame(rows)



def build_selected_compact_feature_set(choices: list[FeatureChoice], audit: pd.DataFrame) -> pd.DataFrame:
    by_feature = {row["feature"]: row for row in audit.to_dict("records")}
    rows = []
    for order, c in enumerate(choices, start=1):
        row = by_feature.get(c.feature, {})
        rows.append(
            {
                "feature_order": order,
                "feature": c.feature,
                "human_name": humanize_feature(c.feature),
                "feature_family": c.feature_family,
                "model_role": c.model_role,
                "included_in_default_compact": c.included_in_default_compact,
                "included_in_tagged_optional": c.included_in_tagged_optional,
                "missing_share": row.get("missing_share", np.nan),
                "missing_share_2022": row.get("missing_share_2022", np.nan),
                "missing_share_2023": row.get("missing_share_2023", np.nan),
                "abs_target_association": row.get("abs_spearman_corr_with_target", np.nan),
                "target_association_metric": row.get("target_association_metric", ""),
                "skewness": row.get("skewness", np.nan),
                "outlier_share_iqr_rule": row.get("outlier_share_iqr_rule", np.nan),
                "rationale": c.rationale,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Write/report
# ---------------------------------------------------------------------------


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)



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



def print_model_feature_summary(model_features: pd.DataFrame) -> None:
    print("\nProposed sequential model feature sets:")
    for (model_id, model_label), part in model_features.groupby(["model_id", "model_label"], sort=False):
        features = [f for f in part["feature"].tolist() if isinstance(f, str) and f]
        print(f"\n{model_id} — {model_label} ({len(features)} features)")
        if features:
            grouped = part[part["feature"].astype(bool)].groupby("feature_family")["human_name"].apply(list)
            for family, names in grouped.items():
                print(f"  {family}:")
                for name in names:
                    print(f"    - {name}")
        else:
            print("  - no explicit feature list")



def main() -> None:
    forecast = load_forecast_features()
    treatment = load_treatment_fruit_features_optional()
    tagged = load_tagged_features_optional()

    df = join_treatment_fruit_features(forecast, treatment)
    df = join_tagged_features(df, tagged)
    df = add_compact_engineered_features(df)

    features = candidate_features_for_audit(df)
    audit = build_individual_audit(df, features)
    mm_choices = build_mean_median_representative_choices(df, audit)
    pairs = build_redundancy_pairs(df, audit)
    clusters = build_redundancy_clusters(pairs, audit)
    choices = build_selected_feature_choices(df, audit, mm_choices)
    recommendations = build_recommendations(audit, clusters, choices, mm_choices)
    compact_features = build_selected_compact_feature_set(choices, audit)
    model_features = build_model_feature_sets(df, choices)

    output_paths = {
        "strawberry_feature_audit_individual": PROCESSED_DIR / "strawberry_feature_audit_individual.csv",
        "strawberry_feature_mean_median_choices": PROCESSED_DIR / "strawberry_feature_mean_median_choices.csv",
        "strawberry_feature_redundancy_pairs": PROCESSED_DIR / "strawberry_feature_redundancy_pairs.csv",
        "strawberry_feature_redundancy_clusters": PROCESSED_DIR / "strawberry_feature_redundancy_clusters.csv",
        "strawberry_feature_selection_recommendation": PROCESSED_DIR / "strawberry_feature_selection_recommendation.csv",
        "strawberry_selected_compact_feature_set": PROCESSED_DIR / "strawberry_selected_compact_feature_set.csv",
        "strawberry_selected_model_feature_sets": PROCESSED_DIR / "strawberry_selected_model_feature_sets.csv",
    }
    tables = {
        "strawberry_feature_audit_individual": audit,
        "strawberry_feature_mean_median_choices": mm_choices,
        "strawberry_feature_redundancy_pairs": pairs,
        "strawberry_feature_redundancy_clusters": clusters,
        "strawberry_feature_selection_recommendation": recommendations,
        "strawberry_selected_compact_feature_set": compact_features,
        "strawberry_selected_model_feature_sets": model_features,
    }
    for name, path in output_paths.items():
        write_csv(tables[name], path)
    write_duckdb_tables(tables)

    print("Feature audit and selection complete.")
    print(f"Forecast rows audited: {len(df)}")
    print(f"Candidate features audited: {len(features)}")
    print(f"Treatment-specific fruit-size feature file found: {'yes' if not treatment.empty else 'no'}")
    print(f"Tagged feature file found: {'yes' if not tagged.empty else 'no'}")
    print(f"Mean/median candidate pairs evaluated: {len(mm_choices)}")
    print(f"High-correlation pairs |rho| >= {REDUNDANCY_RHO_THRESHOLD}: {len(pairs)}")
    print(f"Redundancy clusters: {len(clusters)}")

    if not mm_choices.empty:
        print("\nMean vs median choices:")
        print(
            mm_choices[["stem", "chosen_summary", "chosen_feature", "mean_abs_skewness", "mean_outlier_share", "decision_reason"]]
            .head(20)
            .to_string(index=False)
        )

    if not audit.empty:
        print("\nTop target-associated individual variables:")
        cols = ["human_name", "feature", "feature_family", "missing_share", "abs_spearman_corr_with_target", "target_association_metric"]
        print(
            audit.sort_values("abs_spearman_corr_with_target", ascending=False, na_position="last")
            .head(15)[cols]
            .to_string(index=False)
        )

    if not compact_features.empty:
        print("\nSelected default compact features:")
        print(
            compact_features.loc[compact_features["included_in_default_compact"], ["feature_family", "human_name", "feature", "missing_share", "rationale"]]
            .to_string(index=False)
        )
        optional = compact_features.loc[compact_features["included_in_tagged_optional"], ["feature_family", "human_name", "feature", "missing_share"]]
        if not optional.empty:
            print("\nOptional tagged-fruit features for testing only:")
            print(optional.to_string(index=False))

    print_model_feature_summary(model_features)

    print("\nOutputs written to:")
    for path in output_paths.values():
        print(f"- {path}")


if __name__ == "__main__":
    main()
