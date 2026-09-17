#!/usr/bin/env python3
"""
Profile the public PheMuT strawberry forecasting dataset before modelling.

This script intentionally does not use the original PheMuT training code. It only
inspects the published CSV files and converts the wide observation sheets into a
long audit table so we can decide whether the data are suitable for an independent
forecasting pipeline.

Expected input layout, for example:
    data/2324_GNV_processed/2324_weather.csv
    data/2324_GNV_processed/240108/consolidated_summary_with_yield.csv
    data/2324_GNV_processed/counting_yield/240108.csv
    data/2425_GNV_processed/2425_weather.csv
    ...

Typical usage from the cloned PheMuT repo:
    python 00_profile_phemut_dataset.py --data-dir data

Typical usage from this case-study repo after copying/cloning the PheMuT data:
    python scripts/phemut/00_profile_phemut_dataset.py \
        --data-dir ../dataset_scouting/phemut_check/data \
        --out-dir data/processed/phemut
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path("data/raw/phemut")
DEFAULT_OUT_DIR = Path("data/processed/phemut")


@dataclass(frozen=True)
class FileContext:
    path: Path
    relative_path: str
    season: str | None
    observation_date: pd.Timestamp | None
    file_kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile PheMuT CSV files and produce data-suitability audit outputs."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the PheMuT data folder or its CSV contents.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory where profile CSV/Markdown outputs will be written.",
    )
    parser.add_argument(
        "--long-output",
        choices=["preview", "full", "none"],
        default="full",
        help="Whether to write the full long measurement table, a preview, or no long table.",
    )
    return parser.parse_args()


def normalize_column_name(name: object) -> str:
    return str(name).strip()


def normalize_plot_id(name: object) -> str:
    return str(name).strip().upper()


def normalize_row_label(label: object) -> str:
    raw = str(label).strip()
    low = raw.lower().strip()
    low = low.replace("-", "_").replace(" ", "_")
    low = re.sub(r"_+", "_", low)

    aliases = {
        "fl": "flower_count",
        "flower": "flower_count",
        "flowers": "flower_count",
        "strawberry_flower": "flower_count",
        "g": "green_fruit_count",
        "green": "green_fruit_count",
        "strawberry_green": "green_fruit_count",
        "green_fruit": "green_fruit_count",
        "w": "white_fruit_count",
        "white": "white_fruit_count",
        "strawberry_white": "white_fruit_count",
        "white_fruit": "white_fruit_count",
        "r": "red_or_ripe_fruit_count",
        "red": "red_or_ripe_fruit_count",
        "ripe": "red_or_ripe_fruit_count",
        "strawberry_red": "red_or_ripe_fruit_count",
        "strawberry_ripe": "red_or_ripe_fruit_count",
    }
    return aliases.get(low, low)


def is_likely_yield_label(label: object) -> bool:
    normalized = normalize_row_label(label)
    tokens = (
        "yield",
        "harvest",
        "weight",
        "mass",
        "production",
        "marketable",
        "kg",
        "lb",
        "lbs",
        "gram",
        "grams",
        "g_per",
        "fruit_weight",
        "fresh_weight",
    )
    return any(token in normalized for token in tokens)


def parse_yymmdd(text: str) -> pd.Timestamp | None:
    match = re.search(r"(?<!\d)(\d{6})(?!\d)", text)
    if not match:
        return None
    value = match.group(1)
    try:
        return pd.to_datetime("20" + value, format="%Y%m%d")
    except ValueError:
        return None


def infer_season(path: Path) -> str | None:
    for part in path.parts:
        match = re.search(r"(\d{4})_GNV_processed", part, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    match = re.search(r"(\d{4})_GNV_processed", str(path), flags=re.IGNORECASE)
    return match.group(1) if match else None


def infer_file_kind(path: Path) -> str:
    name = path.name.lower()
    parent = path.parent.name.lower()
    if "weather" in name:
        return "weather"
    if "consolidated_summary_with_yield" in name:
        return "consolidated_summary_with_yield"
    if parent == "counting_yield":
        return "counting_yield"
    return "other_csv"


def infer_file_context(path: Path, data_dir: Path) -> FileContext:
    relative = str(path.relative_to(data_dir)) if path.is_relative_to(data_dir) else str(path)
    return FileContext(
        path=path,
        relative_path=relative,
        season=infer_season(path),
        observation_date=parse_yymmdd(str(path)),
        file_kind=infer_file_kind(path),
    )


def list_csv_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. Provide --data-dir pointing at the PheMuT data folder."
        )
    csvs = sorted(data_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")
    return csvs


def safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def non_empty_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        name = normalize_column_name(col)
        if not name or name.lower().startswith("unnamed"):
            # Keep first unnamed column as row-label candidate elsewhere, but
            # ignore duplicate trailing unnamed columns as data columns.
            continue
        cols.append(name)
    return cols


def find_row_label_column(df: pd.DataFrame) -> str:
    # Consolidated files usually use "Unnamed: 0"; counting_yield files use
    # "Counting". Fall back to the first column because the wide tables use rows
    # as variable labels and plot ids as columns.
    preferred = ["Counting", "counting", "Unnamed: 0", "variable", "Variable", "label", "Label"]
    for candidate in preferred:
        if candidate in df.columns:
            return candidate
    return str(df.columns[0])


def find_plot_columns(df: pd.DataFrame, row_label_col: str) -> list[str]:
    plot_columns: list[str] = []
    for col in df.columns:
        name = normalize_column_name(col)
        if name == row_label_col:
            continue
        if name.lower().startswith("unnamed"):
            # Some 2425 files contain a duplicate trailing label column named
            # Unnamed: 41. It should not be treated as a plot.
            continue
        if re.fullmatch(r"[A-Za-z]+\d+", name):
            plot_columns.append(name)
    return plot_columns


def parse_plot_id(plot_id: str) -> dict[str, object]:
    clean = normalize_plot_id(plot_id)
    match = re.fullmatch(r"([A-Z]+)(\d+)", clean)
    if not match:
        return {
            "plot_id": clean,
            "plot_prefix": None,
            "plot_numeric_code": None,
            "plot_block_code": None,
            "plot_replicate_code": None,
        }
    prefix, digits = match.groups()
    return {
        "plot_id": clean,
        "plot_prefix": prefix,
        "plot_numeric_code": digits,
        "plot_block_code": digits[0] if len(digits) >= 1 else None,
        "plot_replicate_code": digits[1:] if len(digits) >= 2 else None,
    }


def profile_manifest(csvs: Iterable[Path], data_dir: Path) -> pd.DataFrame:
    rows = []
    for path in csvs:
        context = infer_file_context(path, data_dir)
        try:
            df = safe_read_csv(path)
            status = "READ_OK"
            error = ""
        except Exception as exc:  # pragma: no cover - defensive report path
            df = pd.DataFrame()
            status = "READ_ERROR"
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "relative_path": context.relative_path,
                "file_kind": context.file_kind,
                "season": context.season,
                "observation_date": context.observation_date.date().isoformat()
                if context.observation_date is not None
                else None,
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "columns_json": json.dumps([str(col) for col in df.columns], ensure_ascii=False),
                "status": status,
                "error": error,
            }
        )
    return pd.DataFrame(rows)


def profile_weather_file(path: Path, data_dir: Path) -> dict[str, object]:
    context = infer_file_context(path, data_dir)
    df = safe_read_csv(path)
    row: dict[str, object] = {
        "relative_path": context.relative_path,
        "season": context.season,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "has_datetime_column": "Date Time" in df.columns,
    }
    if "Date Time" in df.columns:
        dt = pd.to_datetime(df["Date Time"], errors="coerce")
        diffs = dt.sort_values().diff().dropna().dt.total_seconds() / 60
        row.update(
            {
                "min_datetime": dt.min(),
                "max_datetime": dt.max(),
                "n_unique_timestamps": int(dt.nunique(dropna=True)),
                "median_interval_minutes": float(diffs.median()) if not diffs.empty else np.nan,
                "missing_datetime_share": float(dt.isna().mean()),
            }
        )
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    row["numeric_columns_json"] = json.dumps(numeric_cols, ensure_ascii=False)
    row["overall_missing_share"] = float(df.isna().mean().mean()) if len(df.columns) else np.nan
    for col in [
        "Soil Temp (C)",
        "Temp @ 60cm (C)",
        "Temp @ 2m (C)",
        "Relative Humidity (%)",
        "Rainfall Amount (in)",
        "Solar Radiation (w/m2)",
    ]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            key = re.sub(r"[^a-z0-9]+", "_", col.lower()).strip("_")
            row[f"{key}_mean"] = float(values.mean())
            row[f"{key}_missing_share"] = float(values.isna().mean())
    return row


def build_measurement_long(path: Path, data_dir: Path) -> pd.DataFrame:
    context = infer_file_context(path, data_dir)
    df = safe_read_csv(path)
    row_label_col = find_row_label_column(df)
    plot_columns = find_plot_columns(df, row_label_col)
    if not plot_columns:
        return pd.DataFrame()

    long_df = df[[row_label_col] + plot_columns].melt(
        id_vars=[row_label_col],
        value_vars=plot_columns,
        var_name="plot_id_raw",
        value_name="value",
    )
    long_df = long_df.rename(columns={row_label_col: "row_label"})
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df["row_label"] = long_df["row_label"].astype(str).str.strip()
    long_df["normalized_label"] = long_df["row_label"].map(normalize_row_label)
    long_df["is_candidate_yield"] = long_df["row_label"].map(is_likely_yield_label)
    long_df["relative_path"] = context.relative_path
    long_df["file_kind"] = context.file_kind
    long_df["season"] = context.season
    long_df["observation_date"] = context.observation_date
    long_df["plot_id"] = long_df["plot_id_raw"].map(normalize_plot_id)

    plot_meta = pd.DataFrame([parse_plot_id(plot) for plot in long_df["plot_id"].unique()])
    long_df = long_df.merge(plot_meta, on="plot_id", how="left")

    cols = [
        "season",
        "observation_date",
        "file_kind",
        "plot_id",
        "plot_prefix",
        "plot_numeric_code",
        "plot_block_code",
        "plot_replicate_code",
        "row_label",
        "normalized_label",
        "is_candidate_yield",
        "value",
        "relative_path",
    ]
    return long_df[cols]


def profile_measurement_file(path: Path, data_dir: Path) -> dict[str, object]:
    context = infer_file_context(path, data_dir)
    df = safe_read_csv(path)
    row_label_col = find_row_label_column(df)
    plot_columns = find_plot_columns(df, row_label_col)
    labels = []
    candidate_yield_labels = []
    if row_label_col in df.columns:
        labels = sorted(df[row_label_col].dropna().astype(str).str.strip().unique().tolist())
        candidate_yield_labels = [label for label in labels if is_likely_yield_label(label)]
    values = df[plot_columns].apply(pd.to_numeric, errors="coerce") if plot_columns else pd.DataFrame()
    return {
        "relative_path": context.relative_path,
        "file_kind": context.file_kind,
        "season": context.season,
        "observation_date": context.observation_date,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "row_label_column": row_label_col,
        "n_plot_columns": len(plot_columns),
        "plot_columns_json": json.dumps([normalize_plot_id(col) for col in plot_columns], ensure_ascii=False),
        "row_labels_json": json.dumps(labels, ensure_ascii=False),
        "normalized_labels_json": json.dumps(sorted({normalize_row_label(label) for label in labels}), ensure_ascii=False),
        "candidate_yield_labels_json": json.dumps(candidate_yield_labels, ensure_ascii=False),
        "n_candidate_yield_labels": len(candidate_yield_labels),
        "overall_missing_share_in_plot_values": float(values.isna().mean().mean()) if not values.empty else np.nan,
        "min_value": float(values.min().min()) if not values.empty else np.nan,
        "max_value": float(values.max().max()) if not values.empty else np.nan,
    }


def build_row_label_frequency(measurement_long: pd.DataFrame) -> pd.DataFrame:
    if measurement_long.empty:
        return pd.DataFrame()
    out = (
        measurement_long.groupby(["file_kind", "row_label", "normalized_label"], dropna=False)
        .agg(
            n_values=("value", "size"),
            n_non_missing_values=("value", "count"),
            n_observation_dates=("observation_date", "nunique"),
            n_seasons=("season", "nunique"),
            n_plots=("plot_id", "nunique"),
            mean_value=("value", "mean"),
            median_value=("value", "median"),
            min_value=("value", "min"),
            max_value=("value", "max"),
            missing_share=("value", lambda s: float(s.isna().mean())),
            is_candidate_yield=("is_candidate_yield", "max"),
        )
        .reset_index()
        .sort_values(["is_candidate_yield", "file_kind", "normalized_label"], ascending=[False, True, True])
    )
    numeric_cols = out.select_dtypes(include="number").columns
    out[numeric_cols] = out[numeric_cols].round(4)
    return out


def build_plot_profile(measurement_long: pd.DataFrame) -> pd.DataFrame:
    if measurement_long.empty:
        return pd.DataFrame()
    out = (
        measurement_long.groupby(
            ["plot_id", "plot_prefix", "plot_numeric_code", "plot_block_code", "plot_replicate_code"],
            dropna=False,
        )
        .agg(
            n_records=("value", "size"),
            n_non_missing_values=("value", "count"),
            n_file_kinds=("file_kind", "nunique"),
            n_observation_dates=("observation_date", "nunique"),
            n_seasons=("season", "nunique"),
            n_labels=("normalized_label", "nunique"),
        )
        .reset_index()
        .sort_values(["plot_prefix", "plot_numeric_code", "plot_id"])
    )
    return out


def build_candidate_yield_summary(measurement_long: pd.DataFrame) -> pd.DataFrame:
    if measurement_long.empty:
        return pd.DataFrame()
    candidates = measurement_long.loc[measurement_long["is_candidate_yield"]].copy()
    if candidates.empty:
        return pd.DataFrame(
            [
                {
                    "finding": "No obvious yield/harvest/weight row labels found by heuristic.",
                    "status": "REVIEW_REQUIRED",
                }
            ]
        )
    out = (
        candidates.groupby(["file_kind", "row_label", "normalized_label"], dropna=False)
        .agg(
            n_values=("value", "size"),
            n_non_missing_values=("value", "count"),
            n_observation_dates=("observation_date", "nunique"),
            n_seasons=("season", "nunique"),
            n_plots=("plot_id", "nunique"),
            mean_value=("value", "mean"),
            median_value=("value", "median"),
            min_value=("value", "min"),
            max_value=("value", "max"),
            missing_share=("value", lambda s: float(s.isna().mean())),
        )
        .reset_index()
        .sort_values(["n_observation_dates", "n_non_missing_values"], ascending=False)
    )
    numeric_cols = out.select_dtypes(include="number").columns
    out[numeric_cols] = out[numeric_cols].round(4)
    return out


def build_summary_findings(
    manifest: pd.DataFrame,
    weather_profile: pd.DataFrame,
    measurement_profile: pd.DataFrame,
    measurement_long: pd.DataFrame,
    candidate_yield_summary: pd.DataFrame,
) -> pd.DataFrame:
    n_csv_files = len(manifest)
    n_seasons = int(manifest["season"].nunique(dropna=True)) if "season" in manifest else 0
    n_weather_files = int((manifest["file_kind"] == "weather").sum()) if "file_kind" in manifest else 0
    n_measurement_files = int(manifest["file_kind"].isin(["consolidated_summary_with_yield", "counting_yield"]).sum())
    n_observation_dates = int(measurement_long["observation_date"].nunique(dropna=True)) if not measurement_long.empty else 0
    n_plots = int(measurement_long["plot_id"].nunique(dropna=True)) if not measurement_long.empty else 0
    labels = sorted(measurement_long["normalized_label"].dropna().unique().tolist()) if not measurement_long.empty else []
    n_candidate_yield_rows = 0
    has_candidate_yield = False
    if not candidate_yield_summary.empty and "status" not in candidate_yield_summary.columns:
        n_candidate_yield_rows = len(candidate_yield_summary)
        has_candidate_yield = True

    median_weather_interval = np.nan
    if not weather_profile.empty and "median_interval_minutes" in weather_profile.columns:
        median_weather_interval = float(pd.to_numeric(weather_profile["median_interval_minutes"], errors="coerce").median())

    possible_forecast = n_seasons >= 2 and n_observation_dates >= 8 and n_plots >= 10 and has_candidate_yield
    if possible_forecast:
        suitability = "PROMISING_BUT_REVIEW_TARGET"
        recommendation = (
            "Data look suitable for an independent forecasting audit, provided the candidate yield rows "
            "are confirmed as the operational target and feature availability is checked temporally."
        )
    elif n_seasons >= 2 and n_observation_dates >= 8 and n_plots >= 10:
        suitability = "PROMISING_STRUCTURE_TARGET_UNCLEAR"
        recommendation = (
            "The temporal/plot/weather structure is promising, but no obvious yield row was detected. "
            "Inspect row labels and documentation before modelling."
        )
    else:
        suitability = "LIMITED_FOR_FORECASTING"
        recommendation = (
            "The dataset may be useful for descriptive analysis, but the current profile does not yet show "
            "enough seasons, plots, observation dates, and target rows for robust forecasting."
        )

    rows = [
        {"metric": "csv_files", "value": n_csv_files, "status": "INFO", "note": "All CSVs found under the selected data directory."},
        {"metric": "seasons", "value": n_seasons, "status": "PASS" if n_seasons >= 2 else "WARN", "note": "At least two seasons are needed for season holdout validation."},
        {"metric": "weather_files", "value": n_weather_files, "status": "PASS" if n_weather_files >= 2 else "WARN", "note": "Weather files provide exogenous predictors."},
        {"metric": "measurement_files", "value": n_measurement_files, "status": "PASS" if n_measurement_files else "FAIL", "note": "Counting/consolidated files provide phenology/yield rows."},
        {"metric": "observation_dates", "value": n_observation_dates, "status": "PASS" if n_observation_dates >= 8 else "WARN", "note": "Repeated dates are needed for next-window forecasting."},
        {"metric": "plot_units", "value": n_plots, "status": "PASS" if n_plots >= 10 else "WARN", "note": "Plot/unit replication supports plot-level modelling."},
        {"metric": "normalized_row_labels", "value": len(labels), "status": "INFO", "note": ", ".join(labels[:20])},
        {"metric": "candidate_yield_rows", "value": n_candidate_yield_rows, "status": "PASS" if has_candidate_yield else "REVIEW_REQUIRED", "note": "Heuristic detection based on yield/harvest/weight/mass labels."},
        {"metric": "median_weather_interval_minutes", "value": median_weather_interval, "status": "PASS" if pd.notna(median_weather_interval) and median_weather_interval <= 60 else "WARN", "note": "High-frequency weather can be aggregated into daily/rolling features."},
        {"metric": "overall_suitability", "value": suitability, "status": suitability, "note": recommendation},
    ]
    return pd.DataFrame(rows)



def dataframe_to_markdown_no_extra_dependency(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """Render a small DataFrame as markdown without requiring pandas[tabulate]."""
    if df.empty:
        return ""
    out = df.copy()
    if max_rows is not None:
        out = out.head(max_rows)
    out = out.fillna("")
    columns = [str(c) for c in out.columns]
    rows = [[str(value) for value in row] for row in out.to_numpy()]
    widths = []
    for idx, column in enumerate(columns):
        width = len(column)
        for row in rows:
            width = max(width, len(row[idx]))
        widths.append(width)

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"

    header = fmt_row(columns)
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [fmt_row(row) for row in rows]
    return "\n".join([header, separator, *body])


def write_markdown_report(
    out_path: Path,
    summary: pd.DataFrame,
    row_frequency: pd.DataFrame,
    candidate_yield_summary: pd.DataFrame,
) -> None:
    def value_for(metric: str) -> object:
        matches = summary.loc[summary["metric"] == metric, "value"]
        return matches.iloc[0] if not matches.empty else "n/a"

    def note_for(metric: str) -> object:
        matches = summary.loc[summary["metric"] == metric, "note"]
        return matches.iloc[0] if not matches.empty else ""

    top_labels = dataframe_to_markdown_no_extra_dependency(row_frequency, max_rows=20) if not row_frequency.empty else "No row labels found."
    candidate_text = dataframe_to_markdown_no_extra_dependency(candidate_yield_summary) if not candidate_yield_summary.empty else "No candidate yield summary."

    report = f"""# PheMuT dataset profile

## Executive read

- CSV files: **{value_for('csv_files')}**
- Seasons: **{value_for('seasons')}**
- Observation dates: **{value_for('observation_dates')}**
- Plot/unit columns: **{value_for('plot_units')}**
- Candidate yield rows detected: **{value_for('candidate_yield_rows')}**
- Overall suitability: **{value_for('overall_suitability')}**

{note_for('overall_suitability')}

## Row-label frequency preview

{top_labels}

## Candidate yield rows

{candidate_text}

## Recommended next checks

1. Confirm which row label is the actual yield target and its unit.
2. Confirm whether plot prefixes such as B/M represent variety, treatment, block, or another experimental factor.
3. Build a tidy table at `season × observation_date × plot_id`.
4. Aggregate high-frequency weather into daily, cumulative, and rolling features available before each forecast date.
5. Run a leakage/timeliness audit before trusting any high R² or visually good curve.
"""
    out_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = list_csv_files(data_dir)

    manifest = profile_manifest(csvs, data_dir)
    manifest.to_csv(out_dir / "phemut_file_manifest.csv", index=False)

    weather_rows = []
    measurement_rows = []
    long_parts = []

    for path in csvs:
        context = infer_file_context(path, data_dir)
        if context.file_kind == "weather":
            weather_rows.append(profile_weather_file(path, data_dir))
        elif context.file_kind in {"consolidated_summary_with_yield", "counting_yield"}:
            measurement_rows.append(profile_measurement_file(path, data_dir))
            long = build_measurement_long(path, data_dir)
            if not long.empty:
                long_parts.append(long)

    weather_profile = pd.DataFrame(weather_rows)
    measurement_profile = pd.DataFrame(measurement_rows)
    measurement_long = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame()

    if not weather_profile.empty:
        weather_profile.to_csv(out_dir / "phemut_weather_profile.csv", index=False)
    if not measurement_profile.empty:
        measurement_profile.to_csv(out_dir / "phemut_measurement_file_profile.csv", index=False)

    row_frequency = build_row_label_frequency(measurement_long)
    plot_profile = build_plot_profile(measurement_long)
    candidate_yield_summary = build_candidate_yield_summary(measurement_long)

    if not row_frequency.empty:
        row_frequency.to_csv(out_dir / "phemut_row_label_frequency.csv", index=False)
    if not plot_profile.empty:
        plot_profile.to_csv(out_dir / "phemut_plot_id_profile.csv", index=False)
    if not candidate_yield_summary.empty:
        candidate_yield_summary.to_csv(out_dir / "phemut_candidate_yield_rows.csv", index=False)

    summary = build_summary_findings(
        manifest=manifest,
        weather_profile=weather_profile,
        measurement_profile=measurement_profile,
        measurement_long=measurement_long,
        candidate_yield_summary=candidate_yield_summary,
    )
    summary.to_csv(out_dir / "phemut_profile_summary.csv", index=False)

    if args.long_output == "full" and not measurement_long.empty:
        measurement_long.to_csv(out_dir / "phemut_measurement_long.csv", index=False)
    elif args.long_output == "preview" and not measurement_long.empty:
        measurement_long.head(5000).to_csv(out_dir / "phemut_measurement_long_preview.csv", index=False)

    write_markdown_report(
        out_dir / "phemut_profile_report.md",
        summary=summary,
        row_frequency=row_frequency,
        candidate_yield_summary=candidate_yield_summary,
    )

    print("PheMuT dataset profile complete.")
    print(f"Data directory: {data_dir}")
    print(f"CSV files found: {len(csvs)}")

    print("\nSummary:")
    display_cols = ["metric", "value", "status", "note"]
    print(summary[display_cols].to_string(index=False))

    if not row_frequency.empty:
        print("\nTop row labels:")
        print(
            row_frequency[[
                "file_kind",
                "row_label",
                "normalized_label",
                "n_observation_dates",
                "n_seasons",
                "n_plots",
                "missing_share",
                "is_candidate_yield",
            ]]
            .head(25)
            .to_string(index=False)
        )

    print("\nOutputs written to:")
    for path in sorted(out_dir.glob("phemut_*")):
        print(f"- {path}")


if __name__ == "__main__":
    main()
