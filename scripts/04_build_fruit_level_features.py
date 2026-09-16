#!/usr/bin/env python3
"""
Build interpretable fruit-level features for the strawberry forecasting case study.

This script does NOT retrain any model. It prepares two fruit-level feature layers:

1) Treatment-specific size/weight samples
   - Files such as data_size_freshWeight_condition_2022_0N.csv
   - Joinable by year + date + nitrogen_treatment
   - In the current dataset these appear to be available for 2022 only.

2) Tagged-fruit longitudinal measurements
   - Files such as data_taggedFruit_diameter_2022.csv / 2023.csv
   - Joinable by year + date, but not by nitrogen treatment
   - Useful as global crop-development / phenology signals.

Outputs are written to data/processed/strawberry/ and mirrored into DuckDB when possible.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_MEASUREMENTS_ROOT = ROOT / "data" / "raw" / "strawberry_zenodo" / "measurements"
PROCESSED_DIR = ROOT / "data" / "processed" / "strawberry"
DB_PATH = ROOT / "db" / "yield_forecasting.duckdb"


TREATMENT_RE = re.compile(r"_(0N|50N|100N|150N)\.csv$", re.IGNORECASE)
YEAR_RE = re.compile(r"_(20\d{2})(?:_|\.)")


TAGGED_MEASUREMENT_MAP = {
    "diameter": "tagged_diameter_mm",
    "length": "tagged_length_mm",
    "freshmatter": "tagged_fresh_matter_g",
    "lifespan": "tagged_lifespan_code",
}


SIZE_COLUMN_MAP = {
    "Diameter": "sample_diameter_mm",
    "Length": "sample_length_mm",
    "Fresh weight": "sample_individual_fruit_fresh_weight_g",
    "Condition": "sample_condition_code",
}


def find_measurement_dir() -> Path:
    """Find the folder containing the measurement CSVs."""
    candidates = [
        RAW_MEASUREMENTS_ROOT / "measurements",
        RAW_MEASUREMENTS_ROOT,
        ROOT / "data" / "raw" / "strawberry_zenodo" / "measurements" / "measurements",
    ]
    for candidate in candidates:
        if candidate.exists() and list(candidate.glob("*.csv")):
            return candidate
    raise FileNotFoundError(
        "Could not find measurement CSV files. Expected them under "
        "data/raw/strawberry_zenodo/measurements/measurements/"
    )


def parse_year_from_name(path: Path) -> int | None:
    match = YEAR_RE.search(path.name)
    return int(match.group(1)) if match else None


def parse_treatment_from_name(path: Path) -> str | None:
    match = TREATMENT_RE.search(path.name)
    return match.group(1).upper() if match else None


def parse_tagged_measurement_type(path: Path) -> str:
    lower = path.name.lower()
    for raw_name, clean_name in TAGGED_MEASUREMENT_MAP.items():
        if raw_name in lower:
            return clean_name
    return "tagged_unknown"


def parse_dates(series: pd.Series) -> pd.Series:
    """Parse mixed date formats robustly across pandas versions."""
    clean = series.astype("string").str.strip()

    # Newer pandas versions no longer accept infer_datetime_format.
    # format="mixed" handles the combination of ISO dates and m/d/Y dates in this dataset.
    try:
        parsed = pd.to_datetime(clean, errors="coerce", format="mixed")
    except TypeError:
        parsed = pd.to_datetime(clean, errors="coerce")

    return parsed.dt.date


def safe_quantile(values: pd.Series, q: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.quantile(q))


def summarize_numeric_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: Iterable[str],
    prefix_lookup: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Create robust daily summaries for each numeric column separately."""
    summaries: list[pd.DataFrame] = []
    prefix_lookup = prefix_lookup or {}

    for value_col in value_cols:
        if value_col not in df.columns:
            continue
        numeric = df[group_cols + [value_col]].copy()
        numeric[value_col] = pd.to_numeric(numeric[value_col], errors="coerce")
        clean_prefix = prefix_lookup.get(value_col, value_col)

        grouped = (
            numeric.groupby(group_cols, dropna=False)[value_col]
            .agg(
                n_records="size",
                n_non_missing="count",
                mean="mean",
                median="median",
                std="std",
                min="min",
                max="max",
                p10=lambda s: safe_quantile(s, 0.10),
                p90=lambda s: safe_quantile(s, 0.90),
            )
            .reset_index()
        )

        rename_map = {
            col: f"{clean_prefix}_{col}"
            for col in grouped.columns
            if col not in group_cols
        }
        grouped = grouped.rename(columns=rename_map)
        summaries.append(grouped)

    if not summaries:
        return pd.DataFrame(columns=group_cols)

    out = summaries[0]
    for extra in summaries[1:]:
        out = out.merge(extra, on=group_cols, how="outer")
    return out


def build_treatment_size_features(measurement_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize and summarize treatment-specific fruit size/weight files."""
    files = sorted(measurement_dir.glob("data_size_freshWeight_condition_*.csv"))
    frames: list[pd.DataFrame] = []

    for path in files:
        year = parse_year_from_name(path)
        treatment = parse_treatment_from_name(path)
        if year is None or treatment is None:
            continue

        df = pd.read_csv(path)
        date_col = "Date" if "Date" in df.columns else "date" if "date" in df.columns else None
        if date_col is None:
            continue

        keep_cols = [date_col] + [col for col in SIZE_COLUMN_MAP if col in df.columns]
        part = df[keep_cols].copy()
        part = part.rename(columns={date_col: "date", **SIZE_COLUMN_MAP})
        part["date"] = parse_dates(part["date"])
        part["year"] = year
        part["nitrogen_treatment"] = treatment
        part["source_file"] = path.name
        frames.append(part)

    if not frames:
        empty = pd.DataFrame()
        return empty, empty

    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df.dropna(subset=["date"])

    numeric_cols = [col for col in SIZE_COLUMN_MAP.values() if col in long_df.columns]
    daily = summarize_numeric_by_group(
        long_df,
        group_cols=["year", "date", "nitrogen_treatment"],
        value_cols=numeric_cols,
    )

    # Convenience column: number of individual fruits measured in that date/treatment file.
    if "sample_diameter_mm_n_records" in daily.columns:
        daily["n_fruit_size_samples"] = daily["sample_diameter_mm_n_records"]
    elif "sample_individual_fruit_fresh_weight_g_n_records" in daily.columns:
        daily["n_fruit_size_samples"] = daily["sample_individual_fruit_fresh_weight_g_n_records"]

    return long_df, daily


def build_tagged_fruit_features(measurement_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize and summarize tagged-fruit longitudinal files."""
    files = sorted(measurement_dir.glob("data_taggedFruit_*.csv"))
    frames: list[pd.DataFrame] = []

    for path in files:
        year = parse_year_from_name(path)
        measurement_type = parse_tagged_measurement_type(path)
        if year is None:
            continue

        df = pd.read_csv(path)
        date_col = "Date" if "Date" in df.columns else "date" if "date" in df.columns else None
        if date_col is None:
            continue

        value_cols = [col for col in df.columns if col != date_col]
        part = df[[date_col] + value_cols].copy()
        part = part.rename(columns={date_col: "date"})
        part["date"] = parse_dates(part["date"])
        part = part.melt(
            id_vars=["date"],
            value_vars=value_cols,
            var_name="tagged_fruit_id",
            value_name="value",
        )
        part["value"] = pd.to_numeric(part["value"], errors="coerce")
        part["year"] = year
        part["measurement_type"] = measurement_type
        part["source_file"] = path.name
        frames.append(part)

    if not frames:
        empty = pd.DataFrame()
        return empty, empty

    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df.dropna(subset=["date"])

    summaries: list[pd.DataFrame] = []
    for measurement_type, part in long_df.groupby("measurement_type", dropna=False):
        daily = summarize_numeric_by_group(
            part,
            group_cols=["year", "date"],
            value_cols=["value"],
            prefix_lookup={"value": measurement_type},
        )
        summaries.append(daily)

    wide = summaries[0] if summaries else pd.DataFrame(columns=["year", "date"])
    for extra in summaries[1:]:
        wide = wide.merge(extra, on=["year", "date"], how="outer")

    wide = wide.sort_values(["year", "date"]).reset_index(drop=True)

    # Add simple temporal dynamics for non-destructive tagged-fruit signals.
    for col in [
        "tagged_diameter_mm_mean",
        "tagged_diameter_mm_median",
        "tagged_length_mm_mean",
        "tagged_length_mm_median",
        "tagged_fresh_matter_g_mean",
        "tagged_lifespan_code_mean",
    ]:
        if col in wide.columns:
            wide[f"{col}_change_since_previous_date"] = wide.groupby("year")[col].diff()
            wide[f"{col}_pct_change_since_previous_date"] = wide.groupby("year")[col].pct_change(fill_method=None) * 100

    return long_df, wide


def build_feature_availability_report(
    treatment_daily: pd.DataFrame,
    tagged_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Report how well fruit-level features overlap with the forecast feature table."""
    forecast_path = PROCESSED_DIR / "strawberry_forecast_features.csv"
    rows: list[dict[str, object]] = []

    if not forecast_path.exists():
        return pd.DataFrame(
            [
                {
                    "feature_layer": "forecast_features",
                    "join_key": "not_available",
                    "note": "strawberry_forecast_features.csv not found; run script 02 first to assess join coverage.",
                }
            ]
        )

    forecast = pd.read_csv(forecast_path)
    date_col = "current_observation_date" if "current_observation_date" in forecast.columns else "date"
    if date_col not in forecast.columns:
        return pd.DataFrame(
            [
                {
                    "feature_layer": "forecast_features",
                    "join_key": "not_detected",
                    "note": "Could not detect forecast origin date column.",
                }
            ]
        )

    forecast = forecast.copy()
    forecast["date"] = parse_dates(forecast[date_col])
    if "year" not in forecast.columns:
        forecast["year"] = pd.to_datetime(forecast["date"], errors="coerce").dt.year

    # Treatment-specific sample features.
    if not treatment_daily.empty:
        tmp = treatment_daily.copy()
        tmp["date"] = parse_dates(tmp["date"])
        merged = forecast.merge(
            tmp,
            on=["year", "date", "nitrogen_treatment"],
            how="left",
            suffixes=("", "_fruit_sample"),
        )
        feature_cols = [
            col
            for col in tmp.columns
            if col not in {"year", "date", "nitrogen_treatment"}
        ]
        for col in feature_cols:
            rows.append(
                {
                    "feature_layer": "treatment_specific_fruit_size_weight",
                    "join_key": "year + current_observation_date + nitrogen_treatment",
                    "feature": col,
                    "missing_share_in_forecast_rows": float(merged[col].isna().mean()),
                    "available_years": ", ".join(map(str, sorted(tmp["year"].dropna().unique()))),
                    "note": "Treatment-specific but currently appears limited to 2022 files.",
                }
            )

    # Date-level tagged fruit features.
    if not tagged_daily.empty:
        tmp = tagged_daily.copy()
        tmp["date"] = parse_dates(tmp["date"])
        merged = forecast.merge(
            tmp,
            on=["year", "date"],
            how="left",
            suffixes=("", "_tagged"),
        )
        feature_cols = [col for col in tmp.columns if col not in {"year", "date"}]
        for col in feature_cols:
            rows.append(
                {
                    "feature_layer": "date_level_tagged_fruit",
                    "join_key": "year + current_observation_date",
                    "feature": col,
                    "missing_share_in_forecast_rows": float(merged[col].isna().mean()),
                    "available_years": ", ".join(map(str, sorted(tmp["year"].dropna().unique()))),
                    "note": "Date-level crop-development signal; not treatment-specific.",
                }
            )

    return pd.DataFrame(rows)


def write_outputs(
    treatment_long: pd.DataFrame,
    treatment_daily: pd.DataFrame,
    tagged_long: pd.DataFrame,
    tagged_daily: pd.DataFrame,
    availability: pd.DataFrame,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "strawberry_fruit_size_treatment_long.csv": treatment_long,
        "strawberry_fruit_size_treatment_daily_features.csv": treatment_daily,
        "strawberry_tagged_fruit_long_values.csv": tagged_long,
        "strawberry_tagged_fruit_daily_features.csv": tagged_daily,
        "strawberry_fruit_level_feature_availability_report.csv": availability,
    }

    for filename, df in outputs.items():
        path = PROCESSED_DIR / filename
        df.to_csv(path, index=False)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    try:
        table_map = {
            "strawberry_fruit_size_treatment_long": treatment_long,
            "strawberry_fruit_size_treatment_daily_features": treatment_daily,
            "strawberry_tagged_fruit_long_values": tagged_long,
            "strawberry_tagged_fruit_daily_features": tagged_daily,
            "strawberry_fruit_level_feature_availability_report": availability,
        }
        for table_name, df in table_map.items():
            con.register("tmp_df", df)
            con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM tmp_df")
            con.unregister("tmp_df")
    finally:
        con.close()


def main() -> None:
    measurement_dir = find_measurement_dir()
    treatment_long, treatment_daily = build_treatment_size_features(measurement_dir)
    tagged_long, tagged_daily = build_tagged_fruit_features(measurement_dir)
    availability = build_feature_availability_report(treatment_daily, tagged_daily)
    write_outputs(treatment_long, treatment_daily, tagged_long, tagged_daily, availability)

    print("Fruit-level feature build complete.")
    print(f"Measurement directory: {measurement_dir}")
    print("\nTreatment-specific fruit size/weight layer:")
    print(f"  long rows: {len(treatment_long)}")
    print(f"  daily feature rows: {len(treatment_daily)}")
    if not treatment_daily.empty:
        print(
            "  years:", ", ".join(map(str, sorted(treatment_daily["year"].dropna().unique())))
        )
        print(
            "  treatments:", ", ".join(sorted(treatment_daily["nitrogen_treatment"].dropna().unique()))
        )

    print("\nTagged-fruit date-level layer:")
    print(f"  long rows: {len(tagged_long)}")
    print(f"  daily feature rows: {len(tagged_daily)}")
    if not tagged_daily.empty:
        print("  years:", ", ".join(map(str, sorted(tagged_daily["year"].dropna().unique()))))

    print("\nAvailability report preview:")
    if availability.empty:
        print("  No availability rows generated.")
    else:
        cols = [
            c
            for c in [
                "feature_layer",
                "join_key",
                "feature",
                "missing_share_in_forecast_rows",
                "available_years",
                "note",
            ]
            if c in availability.columns
        ]
        print(availability[cols].head(20).to_string(index=False))

    print("\nOutputs written to:")
    for filename in [
        "strawberry_fruit_size_treatment_long.csv",
        "strawberry_fruit_size_treatment_daily_features.csv",
        "strawberry_tagged_fruit_long_values.csv",
        "strawberry_tagged_fruit_daily_features.csv",
        "strawberry_fruit_level_feature_availability_report.csv",
    ]:
        print(f"- {PROCESSED_DIR / filename}")


if __name__ == "__main__":
    main()
