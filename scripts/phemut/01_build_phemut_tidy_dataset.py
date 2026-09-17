#!/usr/bin/env python3
"""Build a tidy PheMuT strawberry observation dataset.

This script intentionally does not use the modelling code from the PheMuT
repository. It only reads the published CSV data layout and converts the wide
per-date/per-plot files into tidy analysis tables suitable for an independent
forecasting audit.

Expected input layout, for example:

data/
├── 2324_GNV_processed/
│   ├── 2324_weather.csv
│   ├── 240108/consolidated_summary_with_yield.csv
│   └── counting_yield/240108.csv
└── 2425_GNV_processed/
    ├── 2425_weather.csv
    ├── 250107/consolidated_summary_with_yield.csv
    └── counting_yield/250107.csv

Main outputs:
- phemut_tidy_observations.csv
- phemut_daily_weather.csv
- phemut_data_quality_report.csv
- phemut_tidy_summary.csv
- phemut_counting_crosscheck.csv

Optionally writes the same tables into the project DuckDB database.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import duckdb
except ImportError:  # pragma: no cover - optional convenience output
    duckdb = None


DEFAULT_DATA_DIR = Path("data/raw/phemut/data")
DEFAULT_OUT_DIR = Path("data/processed/phemut")
DEFAULT_DB_PATH = Path("db/yield_forecasting.duckdb")

PLOT_ID_RE = re.compile(r"^([A-Za-z]+)(\d)(\d)$")
DATE_DIR_RE = re.compile(r"^(\d{6})$")
SEASON_RE = re.compile(r"^(\d{4})_.*processed$", re.IGNORECASE)

LABEL_ALIASES = {
    "fl": "flower_count",
    "flower": "flower_count",
    "flowers": "flower_count",
    "strawberry_flower": "flower_count",
    "flower_count": "flower_count",
    "g": "green_fruit_count",
    "green": "green_fruit_count",
    "strawberry_green": "green_fruit_count",
    "green_fruit": "green_fruit_count",
    "green_fruit_count": "green_fruit_count",
    "w": "white_fruit_count",
    "white": "white_fruit_count",
    "strawberry_white": "white_fruit_count",
    "white_fruit": "white_fruit_count",
    "white_fruit_count": "white_fruit_count",
    "r": "red_or_ripe_fruit_count",
    "red": "red_or_ripe_fruit_count",
    "ripe": "red_or_ripe_fruit_count",
    "strawberry_red": "red_or_ripe_fruit_count",
    "red_fruit_count": "red_or_ripe_fruit_count",
    "ripe_fruit_count": "red_or_ripe_fruit_count",
    "red_or_ripe_fruit_count": "red_or_ripe_fruit_count",
    "p": "pink_fruit_count",
    "pink": "pink_fruit_count",
    "strawberry_pink": "pink_fruit_count",
    "pink_fruit_count": "pink_fruit_count",
    "area": "canopy_area",
    "depth": "canopy_depth",
    "volume": "canopy_volume",
    "yield": "yield_g",
    "yield_g": "yield_g",
    "yield_(g)": "yield_g",
    "yield (g)": "yield_g",
}

MEASUREMENT_COLUMNS = [
    "flower_count",
    "green_fruit_count",
    "white_fruit_count",
    "pink_fruit_count",
    "red_or_ripe_fruit_count",
    "canopy_area",
    "canopy_depth",
    "canopy_volume",
    "yield_g",
]

PHENOLOGY_COUNT_COLUMNS = [
    "flower_count",
    "green_fruit_count",
    "white_fruit_count",
    "pink_fruit_count",
    "red_or_ripe_fruit_count",
]

WEATHER_ROLLING_WINDOWS = (3, 7, 14)


@dataclass(frozen=True)
class QualityCheck:
    check: str
    status: str
    value: object
    threshold: object
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tidy PheMuT observation tables.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the PheMuT data folder.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory where processed PheMuT outputs will be written.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="DuckDB path. Use --no-db to skip database writes.",
    )
    parser.add_argument("--no-db", action="store_true", help="Do not write DuckDB tables.")
    return parser.parse_args()


def normalize_label(label: object) -> str | None:
    if label is None or pd.isna(label):
        return None
    raw = str(label).strip()
    if not raw or raw.lower() in {"nan", "none", "unnamed: 41", "1"}:
        return None
    key = raw.lower().strip()
    key = key.replace("%", "pct")
    key = key.replace("/", "_")
    key = key.replace("-", "_")
    key = key.replace(" ", "_")
    key = re.sub(r"_+", "_", key)
    key = key.strip("_")
    return LABEL_ALIASES.get(key, key)


def season_from_path(path: Path) -> str:
    for part in path.parts:
        match = SEASON_RE.match(part)
        if match:
            return match.group(1)
    # Fallback: first 4-digit token in the path.
    for part in path.parts:
        token = re.search(r"(\d{4})", part)
        if token:
            return token.group(1)
    return "unknown"


def date_from_path(path: Path) -> pd.Timestamp:
    for part in reversed(path.parts):
        match = DATE_DIR_RE.match(part)
        if match:
            raw = match.group(1)
            year = 2000 + int(raw[:2])
            month = int(raw[2:4])
            day = int(raw[4:6])
            return pd.Timestamp(year=year, month=month, day=day)
    raise ValueError(f"Could not infer YYMMDD observation date from path: {path}")


def canonical_plot_id(column: object) -> str | None:
    if column is None or pd.isna(column):
        return None
    raw = str(column).strip()
    if not raw or raw.lower().startswith("unnamed"):
        return None
    plot_id = raw.upper()
    return plot_id if PLOT_ID_RE.match(plot_id) else None


def add_plot_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed = out["plot_id"].astype(str).str.extract(PLOT_ID_RE)
    out["plot_prefix"] = parsed[0]
    out["plot_block"] = pd.to_numeric(parsed[1], errors="coerce").astype("Int64")
    out["plot_replicate"] = pd.to_numeric(parsed[2], errors="coerce").astype("Int64")
    return out


def discover_measurement_files(data_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(data_dir.rglob("*.csv")):
        name = path.name.lower()
        if name.endswith("weather.csv"):
            file_kind = "weather"
        elif name == "consolidated_summary_with_yield.csv":
            file_kind = "consolidated_summary_with_yield"
        elif "counting_yield" in {p.lower() for p in path.parts}:
            file_kind = "counting_yield"
        else:
            file_kind = "other_csv"

        observation_date = None
        if file_kind in {"consolidated_summary_with_yield", "counting_yield"}:
            try:
                observation_date = date_from_path(path)
            except ValueError:
                observation_date = pd.NaT

        rows.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(data_dir)),
                "file_kind": file_kind,
                "season": season_from_path(path),
                "observation_date": observation_date,
                "file_size_bytes": path.stat().st_size,
            }
        )
    return pd.DataFrame(rows)


def wide_measurement_to_long(path: Path, file_kind: str, season: str, observation_date: pd.Timestamp) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if raw.empty:
        return pd.DataFrame()

    label_column = raw.columns[0]
    plot_columns = [col for col in raw.columns[1:] if canonical_plot_id(col) is not None]
    if not plot_columns:
        return pd.DataFrame()

    data = raw[[label_column] + plot_columns].copy()
    data = data.rename(columns={label_column: "raw_row_label"})
    data["measurement"] = data["raw_row_label"].map(normalize_label)
    data = data.dropna(subset=["measurement"])
    data = data.loc[data["measurement"].isin(MEASUREMENT_COLUMNS)]
    if data.empty:
        return pd.DataFrame()

    long = data.melt(
        id_vars=["raw_row_label", "measurement"],
        value_vars=plot_columns,
        var_name="plot_id",
        value_name="value",
    )
    long["plot_id"] = long["plot_id"].map(canonical_plot_id)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long["season"] = str(season)
    long["observation_date"] = pd.to_datetime(observation_date)
    long["file_kind"] = file_kind
    long["source_file"] = str(path)
    long = long.dropna(subset=["plot_id"])
    long = add_plot_metadata(long)
    return long[
        [
            "season",
            "observation_date",
            "plot_id",
            "plot_prefix",
            "plot_block",
            "plot_replicate",
            "file_kind",
            "raw_row_label",
            "measurement",
            "value",
            "source_file",
        ]
    ]


def build_measurement_long(manifest: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for row in manifest.itertuples(index=False):
        if row.file_kind not in {"consolidated_summary_with_yield", "counting_yield"}:
            continue
        frames.append(
            wide_measurement_to_long(
                Path(row.path),
                file_kind=row.file_kind,
                season=str(row.season),
                observation_date=pd.to_datetime(row.observation_date),
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def pivot_source(long: pd.DataFrame, file_kind: str) -> pd.DataFrame:
    source = long.loc[long["file_kind"] == file_kind].copy()
    if source.empty:
        return pd.DataFrame()
    source = source.drop_duplicates(
        subset=["season", "observation_date", "plot_id", "measurement"],
        keep="last",
    )
    wide = source.pivot_table(
        index=["season", "observation_date", "plot_id", "plot_prefix", "plot_block", "plot_replicate"],
        columns="measurement",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    for col in MEASUREMENT_COLUMNS:
        if col not in wide.columns:
            wide[col] = np.nan
    return wide


def build_counting_crosscheck(consolidated: pd.DataFrame, counting: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "measurement",
        "n_compared",
        "mean_abs_difference",
        "max_abs_difference",
        "share_exact_match",
    ]
    if consolidated.empty or counting.empty:
        return pd.DataFrame(columns=columns)

    key = ["season", "observation_date", "plot_id"]
    common = [col for col in MEASUREMENT_COLUMNS if col in consolidated.columns and col in counting.columns]
    rows = []
    merged = consolidated[key + common].merge(
        counting[key + common],
        on=key,
        how="inner",
        suffixes=("_consolidated", "_counting"),
    )
    for col in common:
        left = f"{col}_consolidated"
        right = f"{col}_counting"
        if left not in merged.columns or right not in merged.columns:
            continue
        diff = merged[left] - merged[right]
        both = merged[left].notna() & merged[right].notna()
        rows.append(
            {
                "measurement": col,
                "n_compared": int(both.sum()),
                "mean_abs_difference": float(diff[both].abs().mean()) if both.any() else np.nan,
                "max_abs_difference": float(diff[both].abs().max()) if both.any() else np.nan,
                "share_exact_match": float((diff[both].abs() < 1e-9).mean()) if both.any() else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def load_weather_files(manifest: pd.DataFrame) -> pd.DataFrame:
    frames = []
    weather_files = manifest.loc[manifest["file_kind"] == "weather"]
    for row in weather_files.itertuples(index=False):
        path = Path(row.path)
        raw = pd.read_csv(path)
        if raw.empty or "Date Time" not in raw.columns:
            continue
        raw = raw.copy()
        raw["season"] = str(row.season)
        raw["date_time"] = pd.to_datetime(raw["Date Time"], errors="coerce")
        raw = raw.dropna(subset=["date_time"])
        raw["date"] = raw["date_time"].dt.floor("D")
        frames.append(raw)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    existing = set(columns)
    for candidate in candidates:
        if candidate in existing:
            return candidate
    return None


def aggregate_weather_daily(weather_raw: pd.DataFrame) -> pd.DataFrame:
    if weather_raw.empty:
        return pd.DataFrame()

    df = weather_raw.copy()
    temp_2m = first_existing(df.columns, ["Temp @ 2m (C)", "Temp @ 60cm (C)", "Temp @ 10m (C)"])
    temp_60cm = first_existing(df.columns, ["Temp @ 60cm (C)"])
    soil_temp = first_existing(df.columns, ["Soil Temp (C)"])
    rh = first_existing(df.columns, ["Relative Humidity (%)"])
    rainfall = first_existing(df.columns, ["Rainfall Amount (in)"])
    wind = first_existing(df.columns, ["Wind Speed (mph)"])
    solar = first_existing(df.columns, ["Solar Radiation (w/m2)"])
    dew = first_existing(df.columns, ["Dew Point Temp (C)"])

    for col in [temp_2m, temp_60cm, soil_temp, rh, rainfall, wind, solar, dew]:
        if col is not None:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    agg_spec: dict[str, tuple[str, str]] = {
        "n_weather_records": ("date_time", "size"),
    }
    if temp_2m:
        agg_spec.update({"tmean_2m_c": (temp_2m, "mean"), "tmin_2m_c": (temp_2m, "min"), "tmax_2m_c": (temp_2m, "max")})
    if temp_60cm:
        agg_spec["tmean_60cm_c"] = (temp_60cm, "mean")
    if soil_temp:
        agg_spec["soil_temp_mean_c"] = (soil_temp, "mean")
    if rh:
        agg_spec["rh_mean_pct"] = (rh, "mean")
    if rainfall:
        agg_spec["rainfall_in"] = (rainfall, "sum")
    if wind:
        agg_spec["wind_speed_mean_mph"] = (wind, "mean")
    if solar:
        # The source is 15-min records. Mean is robust; sum is a proxy for daily radiation load.
        agg_spec["solar_radiation_mean_wm2"] = (solar, "mean")
        agg_spec["solar_radiation_sum_wm2_records"] = (solar, "sum")
    if dew:
        agg_spec["dew_point_mean_c"] = (dew, "mean")

    daily = df.groupby(["season", "date"], as_index=False).agg(**agg_spec).sort_values(["season", "date"])
    if "tmean_2m_c" in daily.columns:
        daily["gdd_base_10_c"] = (daily["tmean_2m_c"] - 10).clip(lower=0)
    else:
        daily["gdd_base_10_c"] = np.nan

    for col in ["rainfall_in", "solar_radiation_sum_wm2_records"]:
        if col not in daily.columns:
            daily[col] = np.nan

    daily["weather_date"] = pd.to_datetime(daily["date"])
    return daily


def add_weather_rolling_features(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return daily.copy()

    out = daily.copy().sort_values(["season", "weather_date"])
    rolling_specs = {
        "tmean_2m_c": "mean",
        "soil_temp_mean_c": "mean",
        "rh_mean_pct": "mean",
        "rainfall_in": "sum",
        "solar_radiation_mean_wm2": "mean",
        "solar_radiation_sum_wm2_records": "sum",
        "gdd_base_10_c": "sum",
    }

    for season, idx in out.groupby("season").groups.items():
        ordered = out.loc[idx].sort_values("weather_date")
        for col, op in rolling_specs.items():
            if col not in ordered.columns:
                continue
            for window in WEATHER_ROLLING_WINDOWS:
                target = f"{col}_{window}d_{op}"
                values = ordered[col].rolling(window=window, min_periods=1)
                out.loc[ordered.index, target] = values.sum().values if op == "sum" else values.mean().values

        # Cumulative features within each season.
        for col in ["gdd_base_10_c", "rainfall_in", "solar_radiation_sum_wm2_records"]:
            if col in ordered.columns:
                out.loc[ordered.index, f"cumulative_{col}"] = ordered[col].fillna(0).cumsum().values

    return out


def attach_weather_features(observations: pd.DataFrame, daily_weather: pd.DataFrame) -> pd.DataFrame:
    if observations.empty or daily_weather.empty:
        return observations.copy()

    weather_feature_cols = [
        col
        for col in daily_weather.columns
        if col not in {"date", "weather_date"}
    ]
    merged = observations.merge(
        daily_weather[weather_feature_cols + ["weather_date"]],
        left_on=["season", "observation_date"],
        right_on=["season", "weather_date"],
        how="left",
    )
    merged = merged.drop(columns=["weather_date"], errors="ignore")
    return merged


def build_quality_report(observations: pd.DataFrame, daily_weather: pd.DataFrame, crosscheck: pd.DataFrame) -> pd.DataFrame:
    checks: list[QualityCheck] = []

    def status(condition: bool) -> str:
        return "PASS" if condition else "WARN"

    checks.append(QualityCheck("observation_rows", status(len(observations) > 0), len(observations), "> 0", "Tidy rows produced."))
    checks.append(QualityCheck("seasons", status(observations["season"].nunique() >= 2), observations["season"].nunique(), ">= 2", "Needed for season-holdout validation."))
    checks.append(QualityCheck("observation_dates", status(observations["observation_date"].nunique() >= 10), observations["observation_date"].nunique(), ">= 10", "Repeated dates are needed for next-window forecasting."))
    checks.append(QualityCheck("plot_units", status(observations["plot_id"].nunique() >= 20), observations["plot_id"].nunique(), ">= 20", "Plot/unit replication supports plot-level modelling."))

    dupes = observations.duplicated(["season", "observation_date", "plot_id"]).sum()
    checks.append(QualityCheck("duplicate_season_date_plot", "PASS" if dupes == 0 else "FAIL", int(dupes), "0", "One row per season/date/plot is expected."))

    if "yield_g" in observations.columns:
        missing_yield = observations["yield_g"].isna().mean()
        negative_yield = (observations["yield_g"] < 0).sum()
        checks.append(QualityCheck("missing_yield_share", "PASS" if missing_yield < 0.05 else "WARN", round(float(missing_yield), 4), "< 0.05", "Yield must be present for supervised forecasting."))
        checks.append(QualityCheck("negative_yield_rows", "PASS" if negative_yield == 0 else "FAIL", int(negative_yield), "0", "Yield should not be negative."))
    else:
        checks.append(QualityCheck("yield_column_present", "FAIL", False, True, "Could not identify yield_g."))

    for col in PHENOLOGY_COUNT_COLUMNS:
        if col in observations.columns:
            missing = observations[col].isna().mean()
            neg = (observations[col] < 0).sum()
            checks.append(QualityCheck(f"missing_{col}_share", "PASS" if missing < 0.10 else "WARN", round(float(missing), 4), "< 0.10", "Phenology count availability."))
            checks.append(QualityCheck(f"negative_{col}_rows", "PASS" if neg == 0 else "FAIL", int(neg), "0", "Phenology counts should not be negative."))

    weather_rows = len(daily_weather)
    checks.append(QualityCheck("daily_weather_rows", status(weather_rows > 0), weather_rows, "> 0", "Daily weather features produced."))
    if "n_weather_records" in observations.columns:
        missing_weather = observations["n_weather_records"].isna().mean()
        checks.append(QualityCheck("missing_weather_join_share", "PASS" if missing_weather < 0.05 else "WARN", round(float(missing_weather), 4), "< 0.05", "Observation dates should join to weather features."))

    if not crosscheck.empty and "measurement" in crosscheck.columns:
        yield_check = crosscheck.loc[crosscheck["measurement"] == "yield_g"]
        if not yield_check.empty:
            mad = float(yield_check["mean_abs_difference"].iloc[0])
            checks.append(QualityCheck("counting_vs_consolidated_yield_mad", "PASS" if mad < 1e-6 else "WARN", round(mad, 6), "0", "Yield row consistency across published file types."))

    return pd.DataFrame([check.__dict__ for check in checks])


def build_summary(observations: pd.DataFrame) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame()
    summary = pd.DataFrame(
        [
            {"metric": "rows", "value": len(observations)},
            {"metric": "seasons", "value": observations["season"].nunique()},
            {"metric": "observation_dates", "value": observations["observation_date"].nunique()},
            {"metric": "plot_units", "value": observations["plot_id"].nunique()},
            {"metric": "min_observation_date", "value": observations["observation_date"].min().date().isoformat()},
            {"metric": "max_observation_date", "value": observations["observation_date"].max().date().isoformat()},
            {"metric": "yield_missing_share", "value": round(float(observations["yield_g"].isna().mean()), 4) if "yield_g" in observations.columns else np.nan},
            {"metric": "mean_yield_g", "value": round(float(observations["yield_g"].mean()), 3) if "yield_g" in observations.columns else np.nan},
            {"metric": "median_yield_g", "value": round(float(observations["yield_g"].median()), 3) if "yield_g" in observations.columns else np.nan},
        ]
    )
    return summary


def write_duckdb_tables(db_path: Path, tables: dict[str, pd.DataFrame]) -> None:
    if duckdb is None:
        print("DuckDB is not installed; skipping DB writes.")
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        for table_name, df in tables.items():
            # DuckDB cannot register a DataFrame with zero columns. This can happen
            # for optional audit tables on partial/dirty source folders. CSV output is
            # still written, but the database table is skipped loudly.
            if df is None or len(df.columns) == 0:
                print(f"Skipping DuckDB table {table_name}: DataFrame has no columns.")
                continue
            con.register("_tmp_df", df)
            con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _tmp_df")
            con.unregister("_tmp_df")
    finally:
        con.close()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"PheMuT data directory not found: {data_dir}")

    manifest = discover_measurement_files(data_dir)
    measurement_long = build_measurement_long(manifest)
    consolidated = pivot_source(measurement_long, "consolidated_summary_with_yield")
    counting = pivot_source(measurement_long, "counting_yield")
    crosscheck = build_counting_crosscheck(consolidated, counting)

    if consolidated.empty:
        raise RuntimeError("No consolidated_summary_with_yield rows could be converted into tidy observations.")

    observations = consolidated.copy().sort_values(["season", "observation_date", "plot_id"])
    weather_raw = load_weather_files(manifest)
    daily_weather = aggregate_weather_daily(weather_raw)
    daily_weather = add_weather_rolling_features(daily_weather)
    observations_with_weather = attach_weather_features(observations, daily_weather)

    quality = build_quality_report(observations_with_weather, daily_weather, crosscheck)
    summary = build_summary(observations_with_weather)

    # CSV outputs.
    manifest.to_csv(out_dir / "phemut_file_manifest.csv", index=False)
    measurement_long.to_csv(out_dir / "phemut_measurement_long.csv", index=False)
    observations_with_weather.to_csv(out_dir / "phemut_tidy_observations.csv", index=False)
    daily_weather.to_csv(out_dir / "phemut_daily_weather.csv", index=False)
    counting.to_csv(out_dir / "phemut_counting_tidy_observations.csv", index=False)
    crosscheck.to_csv(out_dir / "phemut_counting_crosscheck.csv", index=False)
    quality.to_csv(out_dir / "phemut_data_quality_report.csv", index=False)
    summary.to_csv(out_dir / "phemut_tidy_summary.csv", index=False)

    if not args.no_db:
        write_duckdb_tables(
            args.db_path,
            {
                "phemut_file_manifest": manifest,
                "phemut_measurement_long": measurement_long,
                "phemut_tidy_observations": observations_with_weather,
                "phemut_daily_weather": daily_weather,
                "phemut_counting_tidy_observations": counting,
                "phemut_counting_crosscheck": crosscheck,
                "phemut_data_quality_report": quality,
                "phemut_tidy_summary": summary,
            },
        )

    print("PheMuT tidy dataset build complete.")
    print(f"Rows: {len(observations_with_weather):,}")
    print(f"Seasons: {observations_with_weather['season'].nunique()}")
    print(f"Observation dates: {observations_with_weather['observation_date'].nunique()}")
    print(f"Plot units: {observations_with_weather['plot_id'].nunique()}")
    if "yield_g" in observations_with_weather.columns:
        print(f"Yield missing share: {observations_with_weather['yield_g'].isna().mean():.4f}")
    print("\nQuality report:")
    print(quality.to_string(index=False))
    print("\nOutputs written to:")
    for path in [
        "phemut_tidy_observations.csv",
        "phemut_daily_weather.csv",
        "phemut_data_quality_report.csv",
        "phemut_tidy_summary.csv",
        "phemut_counting_crosscheck.csv",
    ]:
        print(f"- {out_dir / path}")


if __name__ == "__main__":
    main()
