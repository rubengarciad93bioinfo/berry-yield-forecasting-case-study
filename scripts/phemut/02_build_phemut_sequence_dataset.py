#!/usr/bin/env python3
"""Build paper-informed PheMuT sequence forecast dataset.

This script intentionally does not reuse the PheMuT paper code. It only uses the
paper's experimental framing to build an independent, auditable dataset:

- one plot = one sequence
- first N observed weeks are the input history (default: 5)
- future observed weeks are explicit forecast horizons
- model features exclude current/previous yield by default to reduce lagging
- the target remains harvested Yield (g), as provided by the public CSVs

Input:
- data/processed/phemut/phemut_tidy_observations.csv

Outputs:
- data/processed/phemut/phemut_sequence_dataset.csv
- data/processed/phemut/phemut_sequence_input_panel.csv
- data/processed/phemut/phemut_sequence_quality_report.csv
- data/processed/phemut/phemut_sequence_feature_sets.csv
- data/processed/phemut/phemut_sequence_target_summary.csv
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

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None


DEFAULT_PROCESSED_DIR = Path("data/processed/phemut")
DEFAULT_DB_PATH = Path("db/yield_forecasting.duckdb")

# Paper-informed core visual features: FL, G, W, P, canopy area, canopy volume.
# Red/ripe is deliberately excluded from the default model feature set because it
# is very close to immediate/current harvest and can make forecasts reactive.
CORE_PHENOLOGY = [
    "flower_count",
    "green_fruit_count",
    "white_fruit_count",
    "pink_fruit_count",
]
OPTIONAL_REACTIVE_PHENOLOGY = ["red_or_ripe_fruit_count"]
CORE_CANOPY = ["canopy_area", "canopy_volume"]
OPTIONAL_CANOPY = ["canopy_depth"]

# We use interpretable physical weather summaries rather than the paper's learned
# weather embeddings, because this is an independent portfolio pipeline.
PREFERRED_WEATHER = [
    # Compact weekly weather context. We deliberately avoid dumping every raw
    # weather column into the model because each season has only 40 plot
    # sequences.
    "tmean_2m_c_7d_mean",
    "soil_temp_mean_c_7d_mean",
    "rh_mean_pct_7d_mean",
    "rainfall_in_7d_sum",
    "solar_radiation_mean_wm2_7d_mean",
    "gdd_base_10_c_7d_sum",
    "cumulative_gdd_base_10_c",
    "cumulative_rainfall_in",
    "cumulative_solar_radiation_sum_wm2_records",
]

# Coefficients are a small paper-informed agronomic feature, not copied code.
PHENOLOGY_STAGE_WEIGHTS = {
    "flower_count": 0.2,
    "green_fruit_count": 0.4,
    "white_fruit_count": 0.8,
    "pink_fruit_count": 0.9,
}


@dataclass(frozen=True)
class QualityCheck:
    check: str
    status: str
    value: object
    threshold: object
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-informed PheMuT sequence dataset.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory containing phemut_tidy_observations.csv and where outputs are written.",
    )
    parser.add_argument(
        "--input-weeks",
        type=int,
        default=5,
        help="Number of initial observation weeks used as input history.",
    )
    parser.add_argument(
        "--include-red-feature-set",
        action="store_true",
        help="Also define an exploratory feature set that includes red/ripe counts. Default models should use the non-reactive feature set.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="DuckDB path. Use --no-db to skip database writes.",
    )
    parser.add_argument("--no-db", action="store_true", help="Do not write DuckDB tables.")
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def clean_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_semantic_features(observations: pd.DataFrame) -> pd.DataFrame:
    """Add cultivar, plot structure, and paper-informed phenology summaries."""
    df = observations.copy()
    df["observation_date"] = pd.to_datetime(df["observation_date"])

    if "plot_prefix" not in df.columns:
        df["plot_prefix"] = df["plot_id"].astype(str).str.extract(r"^([A-Za-z]+)", expand=False).str.upper()
    else:
        df["plot_prefix"] = df["plot_prefix"].astype(str).str.upper().replace({"NAN": np.nan})

    if "plot_block" not in df.columns:
        df["plot_block"] = pd.to_numeric(df["plot_id"].astype(str).str.extract(r"(\d)", expand=False), errors="coerce")
    if "plot_replicate" not in df.columns:
        df["plot_replicate"] = pd.to_numeric(df["plot_id"].astype(str).str.extract(r"\d(\d)$", expand=False), errors="coerce")

    # Paper layout: B rows correspond to Florida Brilliance, M rows to Florida Medallion.
    df["cultivar"] = np.select(
        [df["plot_prefix"].eq("B"), df["plot_prefix"].eq("M")],
        ["Florida Brilliance", "Florida Medallion"],
        default="unknown",
    )
    df["is_medallion"] = (df["cultivar"] == "Florida Medallion").astype(int)

    numeric_candidates = set(CORE_PHENOLOGY + OPTIONAL_REACTIVE_PHENOLOGY + CORE_CANOPY + OPTIONAL_CANOPY + ["yield_g"]) | {
        col for col in df.columns if _looks_like_weather_feature(col)
    }
    df = clean_numeric(df, numeric_candidates)

    for col in CORE_PHENOLOGY:
        if col not in df.columns:
            df[col] = np.nan

    df["total_unripe_visible_count"] = df[CORE_PHENOLOGY].sum(axis=1, skipna=True)
    df["phenology_weighted_index"] = 0.0
    any_weighted = False
    for col, weight in PHENOLOGY_STAGE_WEIGHTS.items():
        if col in df.columns:
            df["phenology_weighted_index"] += df[col].fillna(0) * weight
            any_weighted = True
    if not any_weighted:
        df["phenology_weighted_index"] = np.nan

    if "canopy_area" in df.columns:
        df["phenology_weighted_index_per_area"] = df["phenology_weighted_index"] / df["canopy_area"].replace(0, np.nan)
        df["total_unripe_visible_count_per_area"] = df["total_unripe_visible_count"] / df["canopy_area"].replace(0, np.nan)
    else:
        df["phenology_weighted_index_per_area"] = np.nan
        df["total_unripe_visible_count_per_area"] = np.nan

    return df


def _looks_like_weather_feature(col: str) -> bool:
    tokens = [
        "tmean",
        "tmin",
        "tmax",
        "soil_temp",
        "rh_",
        "rainfall",
        "solar",
        "gdd",
        "dew_point",
        "wind_speed",
        "cumulative_",
    ]
    return any(token in col for token in tokens)


def choose_weather_features(df: pd.DataFrame) -> list[str]:
    selected = [col for col in PREFERRED_WEATHER if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    # Fallback: if column names differ, include a small interpretable subset of weather-looking numeric columns.
    if len(selected) < 4:
        extras = [
            col
            for col in df.columns
            if _looks_like_weather_feature(col)
            and pd.api.types.is_numeric_dtype(df[col])
            and col not in selected
            and not col.startswith("target")
            and "next" not in col
        ]
        selected.extend(extras[:12])
    return selected


def build_feature_catalog(observations: pd.DataFrame, include_red: bool) -> tuple[list[str], dict[str, list[str]]]:
    core_visual = [col for col in CORE_PHENOLOGY + CORE_CANOPY if col in observations.columns]
    derived = [
        col
        for col in [
            "total_unripe_visible_count",
            "phenology_weighted_index",
            "phenology_weighted_index_per_area",
            "total_unripe_visible_count_per_area",
        ]
        if col in observations.columns
    ]
    weather = choose_weather_features(observations)
    base = core_visual + derived + weather

    feature_sets: dict[str, list[str]] = {
        "paper_informed_nonreactive": base,
        "paper_informed_no_weather": core_visual + derived,
        "weather_only_context": weather,
    }
    if include_red and "red_or_ripe_fruit_count" in observations.columns:
        feature_sets["exploratory_includes_red_ripe"] = ["red_or_ripe_fruit_count"] + base
    return base, feature_sets


def build_sequence_dataset(observations: pd.DataFrame, input_weeks: int, include_red: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    require_columns(observations, ["season", "observation_date", "plot_id", "yield_g"], "phemut_tidy_observations")
    df = add_semantic_features(observations)
    _, feature_sets = build_feature_catalog(df, include_red=include_red)
    default_features = feature_sets["paper_informed_nonreactive"]

    sequence_rows: list[dict[str, object]] = []
    input_panel_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []

    for season, season_df in df.groupby("season", sort=True):
        season_df = season_df.sort_values(["observation_date", "plot_id"])
        dates = sorted(pd.to_datetime(season_df["observation_date"].dropna().unique()))
        if len(dates) <= input_weeks:
            continue
        input_dates = dates[:input_weeks]
        target_dates = dates[input_weeks:]
        input_last_date = input_dates[-1]

        for plot_id, plot_df in season_df.groupby("plot_id", sort=True):
            plot_df = plot_df.sort_values("observation_date").set_index("observation_date", drop=False)
            metadata_source = plot_df.iloc[0]
            base_row: dict[str, object] = {
                "season": season,
                "plot_id": plot_id,
                "plot_prefix": metadata_source.get("plot_prefix", np.nan),
                "cultivar": metadata_source.get("cultivar", "unknown"),
                "is_medallion": int(metadata_source.get("is_medallion", 0)) if pd.notna(metadata_source.get("is_medallion", np.nan)) else np.nan,
                "plot_block": metadata_source.get("plot_block", np.nan),
                "plot_replicate": metadata_source.get("plot_replicate", np.nan),
                "input_weeks": input_weeks,
                "input_start_date": input_dates[0],
                "input_end_date": input_last_date,
                "n_target_weeks_available": len(target_dates),
            }

            observed_yields = []
            n_complete_input_weeks = 0
            for week_idx, input_date in enumerate(input_dates, start=1):
                prefix = f"w{week_idx}"
                if input_date in plot_df.index:
                    obs = plot_df.loc[input_date]
                    if isinstance(obs, pd.DataFrame):
                        obs = obs.iloc[0]
                    n_complete_input_weeks += 1
                    yield_value = obs.get("yield_g", np.nan)
                    observed_yields.append(yield_value)
                    base_row[f"{prefix}_date"] = input_date
                    base_row[f"{prefix}_yield_observed_for_baseline_g"] = yield_value
                    for feature in default_features:
                        base_row[f"{prefix}_{feature}"] = obs.get(feature, np.nan)
                    input_panel_rows.append(
                        {
                            "season": season,
                            "plot_id": plot_id,
                            "input_week_index": week_idx,
                            "observation_date": input_date,
                            "yield_observed_for_baseline_g": yield_value,
                            **{feature: obs.get(feature, np.nan) for feature in default_features},
                        }
                    )
                else:
                    base_row[f"{prefix}_date"] = input_date
                    base_row[f"{prefix}_yield_observed_for_baseline_g"] = np.nan
                    for feature in default_features:
                        base_row[f"{prefix}_{feature}"] = np.nan

            observed_yields_series = pd.to_numeric(pd.Series(observed_yields), errors="coerce")
            base_row["n_complete_input_weeks"] = n_complete_input_weeks
            base_row["baseline_last_observed_yield_g"] = observed_yields_series.dropna().iloc[-1] if observed_yields_series.notna().any() else np.nan
            base_row["baseline_mean_observed_yield_g"] = observed_yields_series.mean()
            base_row["baseline_zero_yield_g"] = 0.0

            for horizon_idx, target_date in enumerate(target_dates, start=1):
                if target_date not in plot_df.index:
                    continue
                target_obs = plot_df.loc[target_date]
                if isinstance(target_obs, pd.DataFrame):
                    target_obs = target_obs.iloc[0]
                target_yield = target_obs.get("yield_g", np.nan)
                if pd.isna(target_yield):
                    continue
                row = base_row.copy()
                row.update(
                    {
                        "target_horizon_index": horizon_idx,
                        "target_date": target_date,
                        "forecast_horizon_days": int((pd.Timestamp(target_date) - pd.Timestamp(input_last_date)).days),
                        "target_yield_g": float(target_yield),
                        "target_delta_vs_last_observed_g": float(target_yield - row["baseline_last_observed_yield_g"])
                        if pd.notna(row["baseline_last_observed_yield_g"])
                        else np.nan,
                    }
                )
                sequence_rows.append(row)
                target_rows.append(
                    {
                        "season": season,
                        "plot_id": plot_id,
                        "target_horizon_index": horizon_idx,
                        "target_date": target_date,
                        "target_yield_g": float(target_yield),
                    }
                )

    sequence_df = pd.DataFrame(sequence_rows)
    input_panel = pd.DataFrame(input_panel_rows)
    target_panel = pd.DataFrame(target_rows)

    # Build final feature sets using the flattened default columns that are actually present.
    flattened_feature_sets: dict[str, list[str]] = {}
    for feature_set_name, raw_features in feature_sets.items():
        flattened = []
        for week_idx in range(1, input_weeks + 1):
            flattened.extend([f"w{week_idx}_{feature}" for feature in raw_features if f"w{week_idx}_{feature}" in sequence_df.columns])
        # Spatial/cultivar context is simple and not plot-ID memorisation.
        for meta_feature in ["is_medallion", "plot_block", "plot_replicate"]:
            if meta_feature in sequence_df.columns:
                flattened.append(meta_feature)
        flattened_feature_sets[feature_set_name] = flattened

    return sequence_df, input_panel, target_panel, flattened_feature_sets


def build_quality_report(sequence_df: pd.DataFrame, input_panel: pd.DataFrame, feature_sets: dict[str, list[str]], input_weeks: int) -> pd.DataFrame:
    checks: list[QualityCheck] = []

    def status(condition: bool) -> str:
        return "PASS" if condition else "WARN"

    checks.append(QualityCheck("sequence_target_rows", status(len(sequence_df) > 0), len(sequence_df), "> 0", "Rows are plot x future horizon."))
    checks.append(QualityCheck("seasons", status(sequence_df.get("season", pd.Series(dtype=str)).nunique() >= 2), sequence_df.get("season", pd.Series(dtype=str)).nunique(), ">= 2", "Both seasons should be represented, but models will be fit within season."))
    checks.append(QualityCheck("input_weeks", "PASS" if input_weeks == 5 else "WARN", input_weeks, "5", "Paper-informed default uses the first five observed weeks."))
    checks.append(QualityCheck("plots", status(sequence_df.get("plot_id", pd.Series(dtype=str)).nunique() >= 20), sequence_df.get("plot_id", pd.Series(dtype=str)).nunique(), ">= 20", "Plot-level sequences provide validation units."))
    checks.append(QualityCheck("target_horizons", status(sequence_df.get("target_horizon_index", pd.Series(dtype=float)).nunique() >= 3), sequence_df.get("target_horizon_index", pd.Series(dtype=float)).nunique(), ">= 3", "Future horizon outputs should exist after the input history."))

    if "n_complete_input_weeks" in sequence_df.columns:
        incomplete_share = (sequence_df["n_complete_input_weeks"] < input_weeks).mean() if len(sequence_df) else np.nan
        checks.append(QualityCheck("incomplete_input_sequence_share", "PASS" if pd.notna(incomplete_share) and incomplete_share == 0 else "WARN", round(float(incomplete_share), 4) if pd.notna(incomplete_share) else np.nan, "0", "Each sequence should contain all input weeks."))

    target_missing = sequence_df["target_yield_g"].isna().mean() if "target_yield_g" in sequence_df.columns and len(sequence_df) else np.nan
    checks.append(QualityCheck("target_missing_share", "PASS" if pd.notna(target_missing) and target_missing == 0 else "WARN", round(float(target_missing), 4) if pd.notna(target_missing) else np.nan, "0", "Target yield availability."))

    default_features = feature_sets.get("paper_informed_nonreactive", [])
    forbidden_patterns = re.compile(r"(^|_)yield|target|next|future|error|observed_for_baseline", flags=re.IGNORECASE)
    forbidden = [feature for feature in default_features if forbidden_patterns.search(feature)]
    checks.append(QualityCheck("yield_like_features_in_default_model", "PASS" if not forbidden else "FAIL", len(forbidden), "0", "Default model features should not contain current/previous yield or target-like columns."))
    checks.append(QualityCheck("default_model_feature_count", status(5 <= len(default_features) <= 120), len(default_features), "5-120", "Feature count should stay compact enough for 40 plot sequences per season."))

    if len(sequence_df) and default_features:
        existing_default = [col for col in default_features if col in sequence_df.columns]
        missing_share = sequence_df[existing_default].isna().mean().mean() if existing_default else np.nan
        checks.append(QualityCheck("default_feature_mean_missing_share", "PASS" if pd.notna(missing_share) and missing_share < 0.25 else "WARN", round(float(missing_share), 4) if pd.notna(missing_share) else np.nan, "< 0.25", "Mean missingness across selected model features."))

    return pd.DataFrame([c.__dict__ for c in checks])


def build_feature_sets_table(feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_set": name,
                "n_features": len(features),
                "features_json": json.dumps(features),
                "features_pipe": " | ".join(features),
            }
            for name, features in feature_sets.items()
        ]
    )


def build_target_summary(sequence_df: pd.DataFrame) -> pd.DataFrame:
    if sequence_df.empty:
        return pd.DataFrame()
    summary = (
        sequence_df.groupby(["season", "target_horizon_index", "target_date"], as_index=False)
        .agg(
            n_plots=("plot_id", "nunique"),
            total_target_yield_g=("target_yield_g", "sum"),
            mean_plot_target_yield_g=("target_yield_g", "mean"),
            median_plot_target_yield_g=("target_yield_g", "median"),
            zero_yield_plot_share=("target_yield_g", lambda s: float((s == 0).mean())),
            mean_delta_vs_last_observed_g=("target_delta_vs_last_observed_g", "mean"),
        )
        .sort_values(["season", "target_horizon_index"])
    )
    return summary


def write_markdown_report(path: Path, quality: pd.DataFrame, target_summary: pd.DataFrame, feature_sets: pd.DataFrame) -> None:
    lines = [
        "# PheMuT sequence dataset report",
        "",
        "This report describes the independent paper-informed sequence dataset.",
        "The default model feature set excludes yield history and red/ripe counts to reduce reactive one-window lag behaviour.",
        "",
        "## Quality checks",
        "",
        quality.to_string(index=False) if not quality.empty else "No quality checks available.",
        "",
        "## Target summary",
        "",
        target_summary.to_string(index=False) if not target_summary.empty else "No target summary available.",
        "",
        "## Feature sets",
        "",
        feature_sets[["feature_set", "n_features"]].to_string(index=False) if not feature_sets.empty else "No feature sets available.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


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
    tidy_path = processed_dir / "phemut_tidy_observations.csv"
    if not tidy_path.exists():
        raise FileNotFoundError(f"Missing input file: {tidy_path}. Run 01_build_phemut_tidy_dataset.py first.")

    observations = pd.read_csv(tidy_path)
    sequence_df, input_panel, target_panel, feature_sets = build_sequence_dataset(
        observations,
        input_weeks=args.input_weeks,
        include_red=args.include_red_feature_set,
    )
    quality = build_quality_report(sequence_df, input_panel, feature_sets, input_weeks=args.input_weeks)
    feature_sets_table = build_feature_sets_table(feature_sets)
    target_summary = build_target_summary(sequence_df)

    processed_dir.mkdir(parents=True, exist_ok=True)
    sequence_df.to_csv(processed_dir / "phemut_sequence_dataset.csv", index=False)
    input_panel.to_csv(processed_dir / "phemut_sequence_input_panel.csv", index=False)
    target_panel.to_csv(processed_dir / "phemut_sequence_target_panel.csv", index=False)
    quality.to_csv(processed_dir / "phemut_sequence_quality_report.csv", index=False)
    feature_sets_table.to_csv(processed_dir / "phemut_sequence_feature_sets.csv", index=False)
    target_summary.to_csv(processed_dir / "phemut_sequence_target_summary.csv", index=False)
    write_markdown_report(processed_dir / "phemut_sequence_report.md", quality, target_summary, feature_sets_table)

    if not args.no_db:
        write_duckdb_tables(
            args.db_path,
            {
                "phemut_sequence_dataset": sequence_df,
                "phemut_sequence_input_panel": input_panel,
                "phemut_sequence_target_panel": target_panel,
                "phemut_sequence_quality_report": quality,
                "phemut_sequence_feature_sets": feature_sets_table,
                "phemut_sequence_target_summary": target_summary,
            },
        )

    print("PheMuT sequence dataset build complete.")
    print(f"Rows: {len(sequence_df):,}")
    if not sequence_df.empty:
        print(f"Seasons: {sequence_df['season'].nunique()}")
        print(f"Plot units: {sequence_df['plot_id'].nunique()}")
        print(f"Target horizons: {sorted(sequence_df['target_horizon_index'].dropna().unique().astype(int).tolist())}")
        print(f"Default feature count: {len(feature_sets.get('paper_informed_nonreactive', []))}")
        print("\nTarget summary:")
        print(target_summary.to_string(index=False))
    print("\nQuality report:")
    print(quality.to_string(index=False))
    print("\nOutputs written to:")
    for filename in [
        "phemut_sequence_dataset.csv",
        "phemut_sequence_input_panel.csv",
        "phemut_sequence_target_panel.csv",
        "phemut_sequence_quality_report.csv",
        "phemut_sequence_feature_sets.csv",
        "phemut_sequence_target_summary.csv",
        "phemut_sequence_report.md",
    ]:
        print(f"- {processed_dir / filename}")


if __name__ == "__main__":
    main()
