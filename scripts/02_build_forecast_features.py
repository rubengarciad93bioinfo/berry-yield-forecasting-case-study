from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


DB_PATH = Path("db/yield_forecasting.duckdb")

OUTDIR = Path("data/processed/strawberry")
OUTDIR.mkdir(parents=True, exist_ok=True)


TARGET_CURRENT = "fresh_matter_g"
TARGET_NEXT = "next_fresh_matter_g"


def parse_treatment_n(treatment: str) -> float:
    if pd.isna(treatment):
        return np.nan

    return float(str(treatment).replace("N", ""))


def load_yield_timeseries() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT * FROM strawberry_yield_timeseries").fetchdf()
    con.close()

    df["date"] = pd.to_datetime(df["date"])
    return df


def add_within_treatment_forecast_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["year", "nitrogen_treatment", "date"]).copy()

    group_cols = ["year", "nitrogen_treatment"]

    df["forecast_origin_date"] = df["date"]
    df["target_date"] = df.groupby(group_cols)["date"].shift(-1)

    df[TARGET_NEXT] = df.groupby(group_cols)[TARGET_CURRENT].shift(-1)
    df["next_fruit_number"] = df.groupby(group_cols)["fruit_number"].shift(-1)
    df["next_dry_matter_g"] = df.groupby(group_cols)["dry_matter_g"].shift(-1)

    df["forecast_horizon_days"] = (
        df["target_date"] - df["forecast_origin_date"]
    ).dt.days

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["year", "nitrogen_treatment", "date"]).copy()

    group_cols = ["year", "nitrogen_treatment"]

    lag_columns = [
        "fruit_number",
        "fresh_matter_g",
        "dry_matter_g",
        "mean_fruit_diameter_mm",
        "mean_individual_fruit_fresh_weight_g",
        "mean_tagged_fruit_diameter_mm",
        "mean_tagged_fruit_length_mm",
    ]

    for column in lag_columns:
        if column not in df.columns:
            continue

        df[f"previous_{column}"] = df.groupby(group_cols)[column].shift(1)
        df[f"change_{column}"] = df[column] - df[f"previous_{column}"]

    df["previous_observation_date"] = df.groupby(group_cols)["date"].shift(1)
    df["days_since_previous_observation"] = (
        df["date"] - df["previous_observation_date"]
    ).dt.days

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["year", "nitrogen_treatment", "date"]).copy()

    group_cols = ["year", "nitrogen_treatment"]

    rolling_columns = [
        "fruit_number",
        "fresh_matter_g",
        "dry_matter_g",
    ]

    frames = []

    for _, group in df.groupby(group_cols, dropna=False):
        group = group.copy()

        for column in rolling_columns:
            if column not in group.columns:
                continue

            group[f"rolling_mean_2_{column}"] = (
                group[column]
                .shift(1)
                .rolling(window=2, min_periods=1)
                .mean()
            )

            group[f"rolling_mean_3_{column}"] = (
                group[column]
                .shift(1)
                .rolling(window=3, min_periods=1)
                .mean()
            )

        frames.append(group)

    return pd.concat(frames, ignore_index=True)


def build_forecast_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["treatment_n_level"] = df["nitrogen_treatment"].apply(parse_treatment_n)

    df = add_within_treatment_forecast_targets(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # Keep rows where the next target exists.
    forecast_df = df.loc[df[TARGET_NEXT].notna()].copy()

    forecast_df = forecast_df.rename(
        columns={
            "date": "current_observation_date",
            "fresh_matter_g": "current_fresh_matter_g",
            "fruit_number": "current_fruit_number",
            "dry_matter_g": "current_dry_matter_g",
        }
    )

    # Explicit feature groups for later modelling/dashboard explanation.
    forecast_df["forecast_task"] = "next_observation_fresh_matter"

    useful_columns = [
        "forecast_task",
        "year",
        "nitrogen_treatment",
        "treatment_n_level",
        "current_observation_date",
        "target_date",
        "forecast_horizon_days",
        TARGET_NEXT,
        "next_fruit_number",
        "next_dry_matter_g",
        "current_fruit_number",
        "current_fresh_matter_g",
        "current_dry_matter_g",
        "previous_fruit_number",
        "previous_fresh_matter_g",
        "previous_dry_matter_g",
        "change_fruit_number",
        "change_fresh_matter_g",
        "change_dry_matter_g",
        "rolling_mean_2_fruit_number",
        "rolling_mean_3_fruit_number",
        "rolling_mean_2_fresh_matter_g",
        "rolling_mean_3_fresh_matter_g",
        "rolling_mean_2_dry_matter_g",
        "rolling_mean_3_dry_matter_g",
        "days_since_previous_observation",
        "day_of_year",
        "days_since_weather_start",
        "tmax_c",
        "tmin_c",
        "tmean_c",
        "rhmean_pct",
        "solar_radiation",
        "gdd_base_10",
        "cumulative_gdd_base_10",
        "cumulative_solar_radiation",
        "tmean_7d",
        "solar_radiation_7d",
        "rhmean_7d",
        "n_fruit_size_samples",
        "mean_fruit_diameter_mm",
        "mean_fruit_length_mm",
        "mean_individual_fruit_fresh_weight_g",
        "mean_tagged_fruit_diameter_mm",
        "n_tagged_fruit_diameter_mm_observations",
        "mean_tagged_fruit_length_mm",
        "n_tagged_fruit_length_mm_observations",
        "mean_tagged_fruit_fresh_matter_g",
        "n_tagged_fruit_fresh_matter_g_observations",
        "mean_tagged_fruit_lifespan",
        "n_tagged_fruit_lifespan_observations",
    ]

    available_columns = [col for col in useful_columns if col in forecast_df.columns]

    return forecast_df[available_columns].sort_values(
        ["year", "nitrogen_treatment", "current_observation_date"]
    )


def build_feature_quality_report(forecast_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add_check(check, status, value, detail):
        rows.append(
            {
                "check": check,
                "status": status,
                "value": value,
                "detail": detail,
            }
        )

    add_check(
        "forecast_rows",
        "PASS" if len(forecast_df) > 0 else "FAIL",
        len(forecast_df),
        "Rows available for next-observation forecasting.",
    )

    add_check(
        "unique_year_treatment_origin",
        "PASS"
        if forecast_df.duplicated(
            subset=["year", "nitrogen_treatment", "current_observation_date"]
        ).sum()
        == 0
        else "FAIL",
        int(
            forecast_df.duplicated(
                subset=["year", "nitrogen_treatment", "current_observation_date"]
            ).sum()
        ),
        "Each year/treatment/current date should appear once.",
    )

    add_check(
        "target_missing_share",
        "PASS" if forecast_df[TARGET_NEXT].isna().mean() == 0 else "FAIL",
        float(forecast_df[TARGET_NEXT].isna().mean()),
        "Share of missing next fresh matter targets after filtering.",
    )

    add_check(
        "forecast_horizon_days_min",
        "PASS" if forecast_df["forecast_horizon_days"].min() > 0 else "FAIL",
        int(forecast_df["forecast_horizon_days"].min()),
        "Minimum number of days between current observation and target date.",
    )

    add_check(
        "forecast_horizon_days_max",
        "PASS",
        int(forecast_df["forecast_horizon_days"].max()),
        "Maximum number of days between current observation and target date.",
    )

    core_feature_columns = [
        "current_fruit_number",
        "current_fresh_matter_g",
        "current_dry_matter_g",
        "cumulative_gdd_base_10",
        "cumulative_solar_radiation",
        "tmean_7d",
        "solar_radiation_7d",
    ]

    for column in core_feature_columns:
        if column not in forecast_df.columns:
            continue

        missing_share = forecast_df[column].isna().mean()

        add_check(
            f"missing_share_{column}",
            "PASS" if missing_share < 0.2 else "WARN",
            round(float(missing_share), 3),
            f"Share of missing values in {column}.",
        )

    return pd.DataFrame(rows)


def main():
    yield_df = load_yield_timeseries()

    forecast_df = build_forecast_dataset(yield_df)
    quality_df = build_feature_quality_report(forecast_df)

    forecast_path = OUTDIR / "strawberry_forecast_features.csv"
    quality_path = OUTDIR / "strawberry_forecast_feature_quality.csv"

    forecast_df.to_csv(forecast_path, index=False)
    quality_df.to_csv(quality_path, index=False)

    con = duckdb.connect(str(DB_PATH))

    tables = {
        "strawberry_forecast_features": forecast_df,
        "strawberry_forecast_feature_quality": quality_df,
    }

    for table_name, table_df in tables.items():
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.register("tmp_df", table_df)
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")

    con.close()

    print("Saved:")
    print(forecast_path)
    print(quality_path)

    print("\nForecast feature dataset:")
    print(forecast_df.head())
    print("\nShape:", forecast_df.shape)

    print("\nForecast horizon summary:")
    print(forecast_df["forecast_horizon_days"].describe())

    print("\nRows by year:")
    print(
        forecast_df.groupby("year", as_index=False)
        .agg(
            n_rows=("forecast_task", "size"),
            n_dates=("current_observation_date", "nunique"),
            n_treatments=("nitrogen_treatment", "nunique"),
            min_origin=("current_observation_date", "min"),
            max_origin=("current_observation_date", "max"),
            min_target=("target_date", "min"),
            max_target=("target_date", "max"),
        )
    )

    print("\nQuality report:")
    print(quality_df)


if __name__ == "__main__":
    main()
