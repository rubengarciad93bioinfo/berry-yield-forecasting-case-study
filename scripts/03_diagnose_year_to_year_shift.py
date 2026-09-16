#!/usr/bin/env python3
"""
Diagnose year-to-year strawberry target/weather shift before modelling.

This script is intentionally model-independent. It reads only the feature table
from script 02 and the weather table from script 01/02, then writes diagnostic
outputs explaining why blind year-to-year forecasting is difficult and why a
rolling in-season validation setup is more defensible.

It does NOT read old model prediction/metric tables, so it is safe in a clean
pipeline run after deleting db/yield_forecasting.duckdb and processed outputs.

Run from the repository root:
    python scripts/03_diagnose_year_to_year_shift.py
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


DB_PATH = Path("db/yield_forecasting.duckdb")
OUTDIR = Path("data/processed/strawberry")
OUTDIR.mkdir(parents=True, exist_ok=True)

FORECAST_FEATURES_PATH = OUTDIR / "strawberry_forecast_features.csv"
WEATHER_PATH = OUTDIR / "strawberry_weather_daily.csv"

TARGET = "next_fresh_matter_g"
REQUIRED_FORECAST_COLUMNS = {
    "year",
    "nitrogen_treatment",
    "current_observation_date",
    "target_date",
    "forecast_horizon_days",
    TARGET,
    "current_fruit_number",
    "current_fresh_matter_g",
    "cumulative_gdd_base_10",
    "cumulative_solar_radiation",
    "tmean_7d",
    "solar_radiation_7d",
}
REQUIRED_WEATHER_COLUMNS = {
    "year",
    "date",
    "tmean_c",
    "tmax_c",
    "tmin_c",
    "rhmean_pct",
    "solar_radiation",
    "cumulative_gdd_base_10",
    "cumulative_solar_radiation",
}


# ---------------------------------------------------------------------------
# Loading / validation
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
    require_file(
        FORECAST_FEATURES_PATH,
        "Run: python scripts/02_build_forecast_features.py",
    )
    df = pd.read_csv(FORECAST_FEATURES_PATH)
    missing = REQUIRED_FORECAST_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"{FORECAST_FEATURES_PATH} is missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df["current_observation_date"] = parse_dates(df["current_observation_date"])
    df["target_date"] = parse_dates(df["target_date"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    numeric_cols = [
        TARGET,
        "forecast_horizon_days",
        "current_fruit_number",
        "current_fresh_matter_g",
        "cumulative_gdd_base_10",
        "cumulative_solar_radiation",
        "tmean_7d",
        "solar_radiation_7d",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["year", "nitrogen_treatment", TARGET]).copy()



def load_weather_daily() -> pd.DataFrame:
    require_file(
        WEATHER_PATH,
        "Run: python scripts/01_build_strawberry_dataset.py",
    )
    df = pd.read_csv(WEATHER_PATH)
    missing = REQUIRED_WEATHER_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{WEATHER_PATH} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = parse_dates(df["date"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for col in REQUIRED_WEATHER_COLUMNS.difference({"year", "date"}):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["year", "date"]).copy()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def safe_pct_change(new: float, old: float) -> float:
    if pd.isna(new) or pd.isna(old) or old == 0:
        return np.nan
    return float((new - old) / abs(old) * 100.0)



def build_forecast_dataset_summary(forecast_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": "forecast_rows",
                "value": int(len(forecast_df)),
                "detail": "Rows available for next-observation forecasting.",
            },
            {
                "metric": "years",
                "value": ", ".join(map(str, sorted(forecast_df["year"].dropna().unique()))),
                "detail": "Campaign years present in the forecast feature table.",
            },
            {
                "metric": "nitrogen_treatments",
                "value": ", ".join(sorted(forecast_df["nitrogen_treatment"].dropna().unique())),
                "detail": "Nitrogen treatment groups present in the forecast feature table.",
            },
            {
                "metric": "mean_forecast_horizon_days",
                "value": round(float(forecast_df["forecast_horizon_days"].mean()), 3),
                "detail": "Average lead time between forecast origin and target observation.",
            },
            {
                "metric": "max_forecast_horizon_days",
                "value": round(float(forecast_df["forecast_horizon_days"].max()), 3),
                "detail": "Longest lead time between forecast origin and target observation.",
            },
        ]
    )



def build_target_shift_summary(forecast_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        forecast_df.groupby(["year", "nitrogen_treatment"], as_index=False)
        .agg(
            n_rows=(TARGET, "size"),
            n_target_non_missing=(TARGET, "count"),
            mean_next_fresh_matter_g=(TARGET, "mean"),
            median_next_fresh_matter_g=(TARGET, "median"),
            std_next_fresh_matter_g=(TARGET, "std"),
            p10_next_fresh_matter_g=(TARGET, lambda s: s.quantile(0.10)),
            p90_next_fresh_matter_g=(TARGET, lambda s: s.quantile(0.90)),
            mean_current_fresh_matter_g=("current_fresh_matter_g", "mean"),
            median_current_fresh_matter_g=("current_fresh_matter_g", "median"),
            mean_current_fruit_number=("current_fruit_number", "mean"),
            median_current_fruit_number=("current_fruit_number", "median"),
            mean_cumulative_gdd=("cumulative_gdd_base_10", "mean"),
            mean_cumulative_solar_radiation=("cumulative_solar_radiation", "mean"),
            mean_tmean_7d=("tmean_7d", "mean"),
            mean_solar_radiation_7d=("solar_radiation_7d", "mean"),
        )
        .sort_values(["year", "nitrogen_treatment"])
        .reset_index(drop=True)
    )

    wide_rows: list[dict[str, object]] = []
    for treatment in sorted(summary["nitrogen_treatment"].dropna().unique()):
        subset = summary.loc[summary["nitrogen_treatment"] == treatment]
        if not {2022, 2023}.issubset(set(subset["year"].astype(int))):
            continue

        row_2022 = subset.loc[subset["year"].astype(int) == 2022].iloc[0]
        row_2023 = subset.loc[subset["year"].astype(int) == 2023].iloc[0]

        mean_target_2022 = float(row_2022["mean_next_fresh_matter_g"])
        mean_target_2023 = float(row_2023["mean_next_fresh_matter_g"])
        median_target_2022 = float(row_2022["median_next_fresh_matter_g"])
        median_target_2023 = float(row_2023["median_next_fresh_matter_g"])

        wide_rows.append(
            {
                "nitrogen_treatment": treatment,
                "n_rows_2022": int(row_2022["n_rows"]),
                "n_rows_2023": int(row_2023["n_rows"]),
                "mean_target_2022": mean_target_2022,
                "mean_target_2023": mean_target_2023,
                "target_difference_2023_minus_2022": mean_target_2023 - mean_target_2022,
                "target_pct_change_2023_vs_2022": safe_pct_change(mean_target_2023, mean_target_2022),
                "median_target_2022": median_target_2022,
                "median_target_2023": median_target_2023,
                "median_target_difference_2023_minus_2022": median_target_2023 - median_target_2022,
                "median_target_pct_change_2023_vs_2022": safe_pct_change(median_target_2023, median_target_2022),
                "fruit_number_difference_2023_minus_2022": float(row_2023["mean_current_fruit_number"] - row_2022["mean_current_fruit_number"]),
                "current_fresh_matter_difference_2023_minus_2022": float(row_2023["mean_current_fresh_matter_g"] - row_2022["mean_current_fresh_matter_g"]),
                "gdd_difference_2023_minus_2022": float(row_2023["mean_cumulative_gdd"] - row_2022["mean_cumulative_gdd"]),
                "solar_difference_2023_minus_2022": float(row_2023["mean_cumulative_solar_radiation"] - row_2022["mean_cumulative_solar_radiation"]),
                "tmean_7d_difference_2023_minus_2022": float(row_2023["mean_tmean_7d"] - row_2022["mean_tmean_7d"]),
                "solar_7d_difference_2023_minus_2022": float(row_2023["mean_solar_radiation_7d"] - row_2022["mean_solar_radiation_7d"]),
            }
        )

    shift = pd.DataFrame(wide_rows).sort_values("nitrogen_treatment").reset_index(drop=True)
    return summary, shift



def build_weather_shift_summary(weather_df: pd.DataFrame) -> pd.DataFrame:
    return (
        weather_df.groupby("year", as_index=False)
        .agg(
            n_weather_days=("date", "size"),
            min_weather_date=("date", "min"),
            max_weather_date=("date", "max"),
            mean_tmean_c=("tmean_c", "mean"),
            mean_tmax_c=("tmax_c", "mean"),
            mean_tmin_c=("tmin_c", "mean"),
            mean_rhmean_pct=("rhmean_pct", "mean"),
            mean_solar_radiation=("solar_radiation", "mean"),
            final_cumulative_gdd=("cumulative_gdd_base_10", "max"),
            final_cumulative_solar_radiation=("cumulative_solar_radiation", "max"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )



def build_customer_findings(
    forecast_summary: pd.DataFrame,
    shift_by_treatment: pd.DataFrame,
    weather_shift: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    n_rows = forecast_summary.loc[forecast_summary["metric"] == "forecast_rows", "value"].iloc[0]
    mean_horizon = forecast_summary.loc[
        forecast_summary["metric"] == "mean_forecast_horizon_days", "value"
    ].iloc[0]

    rows.append(
        {
            "finding": "Forecasting setup",
            "result": f"The forecast table contains {n_rows} next-observation records with an average lead time of {float(mean_horizon):.1f} days.",
            "interpretation": "This supports a short-horizon in-season production-window forecast, not a full commercial yield forecast.",
        }
    )

    if not shift_by_treatment.empty:
        mean_abs_target_shift = float(
            shift_by_treatment["target_pct_change_2023_vs_2022"].abs().mean()
        )
        largest = shift_by_treatment.assign(
            abs_shift=lambda d: d["target_pct_change_2023_vs_2022"].abs()
        ).sort_values("abs_shift", ascending=False).iloc[0]

        rows.append(
            {
                "finding": "Season shift",
                "result": f"Average absolute treatment-level target shift between 2022 and 2023 was {mean_abs_target_shift:.1f}%.",
                "interpretation": "Large year-to-year differences mean that a model trained on one previous season should be treated cautiously.",
            }
        )
        rows.append(
            {
                "finding": "Largest treatment shift",
                "result": (
                    f"{largest['nitrogen_treatment']} changed by "
                    f"{float(largest['target_pct_change_2023_vs_2022']):.1f}% in mean next-window fresh matter."
                ),
                "interpretation": "Treatment responses were not stable across years, so treatment alone is not enough to explain production-window changes.",
            }
        )

    if {2022, 2023}.issubset(set(weather_shift["year"].astype(int))):
        w2022 = weather_shift.loc[weather_shift["year"].astype(int) == 2022].iloc[0]
        w2023 = weather_shift.loc[weather_shift["year"].astype(int) == 2023].iloc[0]

        rows.append(
            {
                "finding": "Thermal time difference",
                "result": (
                    f"Final cumulative GDD changed from {float(w2022['final_cumulative_gdd']):.1f} "
                    f"in 2022 to {float(w2023['final_cumulative_gdd']):.1f} in 2023."
                ),
                "interpretation": "Crop-development timing differs between campaigns, so forecasts need in-season updating rather than a blind year-to-year transfer.",
            }
        )
        rows.append(
            {
                "finding": "Radiation difference",
                "result": (
                    f"Final cumulative solar radiation changed from {float(w2022['final_cumulative_solar_radiation']):.1f} "
                    f"in 2022 to {float(w2023['final_cumulative_solar_radiation']):.1f} in 2023."
                ),
                "interpretation": "Differences in seasonal energy exposure are a plausible contributor to year-to-year production differences.",
            }
        )

    rows.append(
        {
            "finding": "Modelling implication",
            "result": "This diagnostic step is independent of model outputs and is safe to run before model training.",
            "interpretation": "Model performance claims should come from the later rolling-validation script, while this script explains the data-shift context.",
        }
    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_outputs(outputs: dict[str, pd.DataFrame]) -> None:
    for table_name, table_df in outputs.items():
        csv_path = OUTDIR / f"{table_name}.csv"
        table_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    try:
        # Remove stale table from the old version of this script if it exists.
        con.execute("DROP TABLE IF EXISTS strawberry_forecast_metrics_by_treatment")

        for table_name, table_df in outputs.items():
            con.execute(f"DROP TABLE IF EXISTS {table_name}")
            con.register("tmp_df", table_df)
            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM tmp_df")
            con.unregister("tmp_df")
    finally:
        con.close()

    stale_metrics_path = OUTDIR / "strawberry_forecast_metrics_by_treatment.csv"
    if stale_metrics_path.exists():
        stale_metrics_path.unlink()
        print(f"Removed stale model-dependent output: {stale_metrics_path}")



def main() -> None:
    forecast_df = load_forecast_features()
    weather_df = load_weather_daily()

    forecast_summary = build_forecast_dataset_summary(forecast_df)
    target_summary, shift_by_treatment = build_target_shift_summary(forecast_df)
    weather_shift = build_weather_shift_summary(weather_df)
    customer_findings = build_customer_findings(
        forecast_summary=forecast_summary,
        shift_by_treatment=shift_by_treatment,
        weather_shift=weather_shift,
    )

    outputs = {
        "strawberry_forecast_dataset_summary": forecast_summary,
        "strawberry_target_summary_by_year_treatment": target_summary,
        "strawberry_year_to_year_shift_by_treatment": shift_by_treatment,
        "strawberry_weather_shift_summary": weather_shift,
        "strawberry_year_to_year_diagnostic_findings": customer_findings,
        # Compatibility name used by the current dashboard. Content is now diagnostic-only.
        "strawberry_forecast_customer_findings": customer_findings,
    }

    write_outputs(outputs)

    print("\nForecast dataset summary:")
    print(forecast_summary.to_string(index=False))

    print("\nYear-to-year target shift by treatment:")
    print(shift_by_treatment.to_string(index=False))

    print("\nWeather shift summary:")
    print(weather_shift.to_string(index=False))

    print("\nCustomer-facing diagnostic findings:")
    print(customer_findings.to_string(index=False))


if __name__ == "__main__":
    main()
