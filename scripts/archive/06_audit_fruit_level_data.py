#!/usr/bin/env python3
"""
Audit strawberry fruit-level / fruit-development measurement files.

Run from the repository root:
    python scripts/06_audit_fruit_level_data.py

This script does not change the forecasting model. It only inventories the raw
fruit-level files, reshapes what can be reshaped safely, and writes diagnostic
outputs so we can decide whether these data are safe/useful to include as
predictors.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RAW_DIR = Path("data/raw/strawberry_zenodo")
PROCESSED_DIR = Path("data/processed/strawberry")
MEASUREMENTS_DIR = RAW_DIR / "measurements"
MEASUREMENTS_ZIP = RAW_DIR / "measurements.zip"

TREATMENTS = ("0N", "50N", "100N", "150N")


def ensure_measurements_extracted() -> None:
    """Extract measurements.zip if the measurements folder is missing."""
    if MEASUREMENTS_DIR.exists() and any(MEASUREMENTS_DIR.rglob("*.csv")):
        return
    if not MEASUREMENTS_ZIP.exists():
        raise FileNotFoundError(
            f"Could not find {MEASUREMENTS_ZIP}. Run script 00 first or place measurements.zip there."
        )
    MEASUREMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(MEASUREMENTS_ZIP, "r") as zf:
        zf.extractall(MEASUREMENTS_DIR)


def read_csv(path: Path) -> pd.DataFrame:
    """Read CSV and normalize obvious date columns."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if col.lower() == "date":
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def infer_year_from_name(path: Path) -> int | None:
    match = re.search(r"20\d{2}", path.name)
    return int(match.group(0)) if match else None


def infer_treatment_from_name(path: Path) -> str | None:
    for treatment in TREATMENTS:
        if re.search(rf"(^|_){re.escape(treatment)}($|_|\.)", path.name):
            return treatment
    return None


def infer_measurement_type(path: Path) -> str:
    name = path.stem
    name = re.sub(r"data_", "", name)
    name = re.sub(r"_20\d{2}", "", name)
    for treatment in TREATMENTS:
        name = re.sub(rf"_{re.escape(treatment)}$", "", name)
    return name


def is_fruit_level_candidate(path: Path) -> bool:
    name = path.name.lower()
    return any(
        token in name
        for token in [
            "taggedfruit",
            "size_freshweight_condition",
            "fruit_diameter",
            "fruit_length",
            "fruit_lifespan",
        ]
    )


def numeric_summary(df: pd.DataFrame) -> dict[str, float | int]:
    numeric = df.select_dtypes(include=[np.number])
    values = numeric.to_numpy().ravel() if not numeric.empty else np.array([])
    values = values[~pd.isna(values)]
    return {
        "numeric_columns": int(numeric.shape[1]),
        "numeric_values": int(values.size),
        "numeric_min": float(np.min(values)) if values.size else np.nan,
        "numeric_median": float(np.median(values)) if values.size else np.nan,
        "numeric_max": float(np.max(values)) if values.size else np.nan,
    }


def melt_wide_tagged_fruit(path: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Convert tagged fruit wide tables to long format when Date exists."""
    date_cols = [c for c in df.columns if c.lower() == "date"]
    if not date_cols:
        return pd.DataFrame()

    date_col = date_cols[0]
    value_cols = [c for c in df.columns if c != date_col]
    if not value_cols:
        return pd.DataFrame()

    long = df.melt(
        id_vars=[date_col],
        value_vars=value_cols,
        var_name="fruit_id_raw",
        value_name="value",
    )
    long = long.rename(columns={date_col: "date"})
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    long["year"] = infer_year_from_name(path)
    long["measurement_type"] = infer_measurement_type(path)
    long["source_file"] = path.name
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    # Best-effort treatment parsing from fruit ID if labels contain 0N/50N/100N/150N.
    pattern = "(" + "|".join(re.escape(t) for t in TREATMENTS) + ")"
    long["nitrogen_treatment"] = long["fruit_id_raw"].astype(str).str.extract(pattern, expand=False)
    return long.dropna(subset=["date"])


def normalize_size_weight_condition(path: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Normalize per-fruit size/fresh-weight/condition tables as far as possible."""
    out = df.copy()
    out["source_file"] = path.name
    out["year"] = infer_year_from_name(path)
    out["nitrogen_treatment"] = infer_treatment_from_name(path)
    out["measurement_type"] = infer_measurement_type(path)

    # Normalize Date if present.
    for col in out.columns:
        if col.lower() == "date":
            out = out.rename(columns={col: "date"})
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            break

    return out


def infer_joinability(path: Path, df: pd.DataFrame, long_df: pd.DataFrame | None = None) -> dict[str, object]:
    has_date = any(c.lower() == "date" for c in df.columns) or (
        long_df is not None and "date" in long_df.columns and long_df["date"].notna().any()
    )
    treatment_from_filename = infer_treatment_from_name(path)
    treatment_in_columns = any(c in df.columns for c in ["nitrogen_treatment", "N treatment", "N_treatment"])
    treatment_in_long = (
        long_df is not None
        and "nitrogen_treatment" in long_df.columns
        and long_df["nitrogen_treatment"].notna().any()
    )

    if treatment_from_filename:
        treatment_source = "filename"
    elif treatment_in_columns:
        treatment_source = "column"
    elif treatment_in_long:
        treatment_source = "fruit_id"
    else:
        treatment_source = "not_detected"

    can_join_by_date_treatment = bool(has_date and treatment_source != "not_detected")
    return {
        "source_file": path.name,
        "has_date": bool(has_date),
        "treatment_source": treatment_source,
        "can_join_to_forecast_by_date_treatment": can_join_by_date_treatment,
    }


def summarize_long_values(long: pd.DataFrame) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame()

    group_cols = ["source_file", "measurement_type", "year"]
    if "nitrogen_treatment" in long.columns and long["nitrogen_treatment"].notna().any():
        group_cols.append("nitrogen_treatment")
    if "date" in long.columns:
        group_cols.append("date")

    return (
        long.groupby(group_cols, dropna=False)
        .agg(
            n_records=("value", "size"),
            n_non_missing=("value", lambda s: int(s.notna().sum())),
            mean_value=("value", "mean"),
            median_value=("value", "median"),
            min_value=("value", "min"),
            max_value=("value", "max"),
        )
        .reset_index()
    )


def main() -> None:
    ensure_measurements_extracted()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(MEASUREMENTS_DIR.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {MEASUREMENTS_DIR}")

    inventory_rows: list[dict[str, object]] = []
    joinability_rows: list[dict[str, object]] = []
    long_frames: list[pd.DataFrame] = []
    normalized_size_frames: list[pd.DataFrame] = []
    preview_rows: list[pd.DataFrame] = []

    for path in csv_files:
        df = read_csv(path)
        candidate = is_fruit_level_candidate(path)
        summary = numeric_summary(df)

        inventory_rows.append(
            {
                "source_file": path.name,
                "relative_path": str(path.relative_to(Path.cwd())) if path.is_relative_to(Path.cwd()) else str(path),
                "is_fruit_level_candidate": candidate,
                "year": infer_year_from_name(path),
                "nitrogen_treatment_from_filename": infer_treatment_from_name(path),
                "measurement_type": infer_measurement_type(path),
                "n_rows": int(df.shape[0]),
                "n_columns": int(df.shape[1]),
                "columns": ", ".join(df.columns),
                "missing_values_total": int(df.isna().sum().sum()),
                **summary,
            }
        )

        if not candidate:
            continue

        long_df = pd.DataFrame()
        if "taggedfruit" in path.name.lower():
            long_df = melt_wide_tagged_fruit(path, df)
            if not long_df.empty:
                long_frames.append(long_df)
        elif "size_freshweight_condition" in path.name.lower():
            normalized = normalize_size_weight_condition(path, df)
            normalized_size_frames.append(normalized)

            # If there is a clear numeric measurement table, also create a long-ish copy
            # for broad summaries. Keep all original columns in the normalized output.
            numeric_cols = [c for c in normalized.select_dtypes(include=[np.number]).columns if c != "year"]
            date_col = "date" if "date" in normalized.columns else None
            id_vars = [c for c in [date_col, "year", "nitrogen_treatment", "measurement_type", "source_file"] if c]
            if numeric_cols and id_vars:
                long_df = normalized.melt(
                    id_vars=id_vars,
                    value_vars=numeric_cols,
                    var_name="measurement_column",
                    value_name="value",
                )
                long_df["fruit_id_raw"] = np.nan
                long_frames.append(long_df)

        joinability_rows.append(infer_joinability(path, df, long_df))

        sample = df.head(5).copy()
        sample.insert(0, "source_file", path.name)
        preview_rows.append(sample)

    inventory = pd.DataFrame(inventory_rows)
    joinability = pd.DataFrame(joinability_rows)

    long_all = pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame()
    size_all = pd.concat(normalized_size_frames, ignore_index=True) if normalized_size_frames else pd.DataFrame()
    preview = pd.concat(preview_rows, ignore_index=True, sort=False) if preview_rows else pd.DataFrame()
    daily_summary = summarize_long_values(long_all)

    outputs = {
        "fruit_level_file_inventory": inventory,
        "fruit_level_joinability_report": joinability,
        "fruit_level_long_values": long_all,
        "fruit_level_daily_summary": daily_summary,
        "fruit_level_size_weight_condition_normalized": size_all,
        "fruit_level_raw_preview": preview,
    }

    for name, table in outputs.items():
        path = PROCESSED_DIR / f"strawberry_{name}.csv"
        table.to_csv(path, index=False)

    print("Fruit-level audit complete.")
    print(f"CSV files scanned: {len(csv_files)}")
    print(f"Fruit-level candidate files: {int(inventory['is_fruit_level_candidate'].sum())}")
    print("\nJoinability summary:")
    if not joinability.empty:
        print(joinability[["source_file", "has_date", "treatment_source", "can_join_to_forecast_by_date_treatment"]].to_string(index=False))
    else:
        print("No fruit-level candidate files found.")

    print("\nOutputs written to:")
    for name in outputs:
        print(f"- {PROCESSED_DIR / ('strawberry_' + name + '.csv')}")


if __name__ == "__main__":
    main()

