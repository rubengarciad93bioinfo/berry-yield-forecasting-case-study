from pathlib import Path
from zipfile import ZipFile

import duckdb
import numpy as np
import pandas as pd


RAW_DIR = Path("data/raw/strawberry_zenodo")
MEASUREMENTS_ZIP = RAW_DIR / "measurements.zip"

OUTDIR = Path("data/processed/strawberry")
OUTDIR.mkdir(parents=True, exist_ok=True)

DB_PATH = Path("db/yield_forecasting.duckdb")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

TREATMENTS = ["0N", "50N", "100N", "150N"]
GDD_BASE_TEMP = 10.0


def parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def ensure_measurements_extracted():
    csv_files = list(RAW_DIR.rglob("*.csv"))

    if csv_files:
        return

    if not MEASUREMENTS_ZIP.exists():
        raise FileNotFoundError(
            f"Could not find CSV files or {MEASUREMENTS_ZIP}. "
            "Run scripts/00_download_strawberry_zenodo.py first."
        )

    extract_dir = RAW_DIR / "measurements"
    extract_dir.mkdir(exist_ok=True)

    with ZipFile(MEASUREMENTS_ZIP, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


def find_file(filename: str) -> Path:
    matches = list(RAW_DIR.rglob(filename))

    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {RAW_DIR}")

    return matches[0]


def read_treatment_timeseries(filename: str, year: int, variable_name: str) -> pd.DataFrame:
    path = find_file(filename)
    df = pd.read_csv(path)

    df["date"] = parse_date(df["Date"])

    long_df = df.melt(
        id_vars=["date"],
        value_vars=[col for col in TREATMENTS if col in df.columns],
        var_name="nitrogen_treatment",
        value_name=variable_name,
    )

    long_df["year"] = year
    long_df["source_file"] = filename

    return long_df[
        [
            "year",
            "date",
            "nitrogen_treatment",
            variable_name,
            "source_file",
        ]
    ]


def build_core_measurements() -> pd.DataFrame:
    datasets = []

    variables = {
        "fruit_number": "data_fruitNumber_{year}.csv",
        "fresh_matter_g": "data_freshMatter_{year}.csv",
        "dry_matter_g": "data_dryMatter_{year}.csv",
    }

    for year in [2022, 2023]:
        year_frames = []

        for variable_name, pattern in variables.items():
            filename = pattern.format(year=year)
            frame = read_treatment_timeseries(
                filename=filename,
                year=year,
                variable_name=variable_name,
            )
            year_frames.append(frame.drop(columns=["source_file"]))

        merged = year_frames[0]

        for frame in year_frames[1:]:
            merged = merged.merge(
                frame,
                on=["year", "date", "nitrogen_treatment"],
                how="outer",
            )

        datasets.append(merged)

    return pd.concat(datasets, ignore_index=True)


def build_weather_daily() -> pd.DataFrame:
    frames = []

    for year in [2022, 2023]:
        path = find_file(f"weather_daily_{year}.csv")
        df = pd.read_csv(path)

        df = df.rename(
            columns={
                "Date": "date",
                "Tmax": "tmax_c",
                "Tmin": "tmin_c",
                "Tmean": "tmean_c",
                "RHmax": "rhmax_pct",
                "RHmin": "rhmin_pct",
                "RHmean": "rhmean_pct",
                "RAD": "solar_radiation",
            }
        )

        df["date"] = parse_date(df["date"])
        df["year"] = year
        df = df.sort_values("date")

        df["gdd_base_10"] = (df["tmean_c"] - GDD_BASE_TEMP).clip(lower=0)
        df["cumulative_gdd_base_10"] = df["gdd_base_10"].cumsum()
        df["cumulative_solar_radiation"] = df["solar_radiation"].cumsum()
        df["day_of_year"] = df["date"].dt.dayofyear
        df["days_since_weather_start"] = (df["date"] - df["date"].min()).dt.days

        df["tmean_7d"] = df["tmean_c"].rolling(window=7, min_periods=1).mean()
        df["solar_radiation_7d"] = (
            df["solar_radiation"].rolling(window=7, min_periods=1).mean()
        )
        df["rhmean_7d"] = df["rhmean_pct"].rolling(window=7, min_periods=1).mean()

        frames.append(df)

    weather = pd.concat(frames, ignore_index=True)

    return weather[
        [
            "year",
            "date",
            "day_of_year",
            "days_since_weather_start",
            "tmax_c",
            "tmin_c",
            "tmean_c",
            "rhmax_pct",
            "rhmin_pct",
            "rhmean_pct",
            "solar_radiation",
            "gdd_base_10",
            "cumulative_gdd_base_10",
            "cumulative_solar_radiation",
            "tmean_7d",
            "solar_radiation_7d",
            "rhmean_7d",
        ]
    ]


def build_fruit_size_samples() -> pd.DataFrame:
    frames = []

    for path in sorted(RAW_DIR.rglob("data_size_freshWeight_condition_*.csv")):
        parts = path.stem.split("_")
        year = int(parts[-2])
        treatment = parts[-1]

        df = pd.read_csv(path)
        df["date"] = parse_date(df["Date"])
        df["year"] = year
        df["nitrogen_treatment"] = treatment

        df = df.rename(
            columns={
                "Diameter": "fruit_diameter_mm",
                "Length": "fruit_length_mm",
                "Fresh weight": "individual_fruit_fresh_weight_g",
                "Condition": "fruit_condition",
            }
        )

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    fruit_size = pd.concat(frames, ignore_index=True)

    summary = (
        fruit_size.groupby(["year", "date", "nitrogen_treatment"], as_index=False)
        .agg(
            n_fruit_size_samples=("individual_fruit_fresh_weight_g", "size"),
            mean_fruit_diameter_mm=("fruit_diameter_mm", "mean"),
            mean_fruit_length_mm=("fruit_length_mm", "mean"),
            mean_individual_fruit_fresh_weight_g=(
                "individual_fruit_fresh_weight_g",
                "mean",
            ),
            n_condition_observed=("fruit_condition", lambda x: x.notna().sum()),
        )
    )

    return summary


def read_tagged_fruit_file(path: Path, variable_name: str, year: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = parse_date(df["Date"])

    value_cols = [col for col in df.columns if col != "Date" and col != "date"]

    long_df = df.melt(
        id_vars=["date"],
        value_vars=value_cols,
        var_name="tagged_fruit_id",
        value_name=variable_name,
    )

    long_df["year"] = year

    return long_df


def build_tagged_fruit_daily_summary() -> pd.DataFrame:
    variable_lookup = {
        "diameter": "tagged_fruit_diameter_mm",
        "length": "tagged_fruit_length_mm",
        "freshMatter": "tagged_fruit_fresh_matter_g",
        "lifespan": "tagged_fruit_lifespan",
    }

    summary_frames = []

    for key, variable_name in variable_lookup.items():
        frames = []

        for path in sorted(RAW_DIR.rglob(f"data_taggedFruit_{key}_*.csv")):
            year = int(path.stem.split("_")[-1])
            frames.append(read_tagged_fruit_file(path, variable_name, year))

        if not frames:
            continue

        long_df = pd.concat(frames, ignore_index=True)

        summary = (
            long_df.groupby(["year", "date"], as_index=False)
            .agg(
                **{
                    f"mean_{variable_name}": (variable_name, "mean"),
                    f"n_{variable_name}_observations": (
                        variable_name,
                        lambda x: x.notna().sum(),
                    ),
                }
            )
        )

        summary_frames.append(summary)

    if not summary_frames:
        return pd.DataFrame()

    merged = summary_frames[0]

    for frame in summary_frames[1:]:
        merged = merged.merge(frame, on=["year", "date"], how="outer")

    return merged


def build_plant_biomass_samples() -> pd.DataFrame:
    frames = []

    for year in [2022, 2023]:
        path = find_file(f"data_plantBiomass_{year}.csv")
        df = pd.read_csv(path)

        id_col = "Unnamed: 0"
        df = df.rename(columns={id_col: "sample_id"})

        long_df = df.melt(
            id_vars=["sample_id"],
            value_vars=[col for col in TREATMENTS if col in df.columns],
            var_name="nitrogen_treatment",
            value_name="plant_biomass_g",
        )

        long_df["year"] = year
        frames.append(long_df)

    return pd.concat(frames, ignore_index=True)[
        [
            "year",
            "sample_id",
            "nitrogen_treatment",
            "plant_biomass_g",
        ]
    ]


def build_quality_report(yield_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    checks = []

    def add_check(check_name, status, value, detail):
        checks.append(
            {
                "check": check_name,
                "status": status,
                "value": value,
                "detail": detail,
            }
        )

    add_check(
        "yield_timeseries_rows",
        "PASS" if len(yield_df) > 0 else "FAIL",
        len(yield_df),
        "Rows in the joined treatment-date yield table.",
    )

    duplicated_rows = yield_df.duplicated(
        subset=["year", "date", "nitrogen_treatment"]
    ).sum()

    add_check(
        "duplicate_year_date_treatment",
        "PASS" if duplicated_rows == 0 else "FAIL",
        int(duplicated_rows),
        "Each year/date/nitrogen treatment should appear once.",
    )

    missing_weather = yield_df["tmean_c"].isna().sum()

    add_check(
        "missing_weather_after_join",
        "PASS" if missing_weather == 0 else "WARN",
        int(missing_weather),
        "Rows without matching daily weather.",
    )

    for column in ["fruit_number", "fresh_matter_g", "dry_matter_g"]:
        if column in yield_df.columns:
            negative_values = (yield_df[column] < 0).sum()

            add_check(
                f"negative_{column}",
                "PASS" if negative_values == 0 else "FAIL",
                int(negative_values),
                f"Negative values found in {column}.",
            )

            missing_share = yield_df[column].isna().mean()

            add_check(
                f"missing_share_{column}",
                "PASS" if missing_share < 0.3 else "WARN",
                round(float(missing_share), 3),
                f"Share of missing values in {column}.",
            )

    impossible_temps = (
        (weather_df["tmax_c"] < weather_df["tmin_c"])
        | (weather_df["tmax_c"] > 60)
        | (weather_df["tmin_c"] < -20)
    ).sum()

    add_check(
        "weather_temperature_sanity",
        "PASS" if impossible_temps == 0 else "FAIL",
        int(impossible_temps),
        "Checks Tmax >= Tmin and plausible temperature bounds.",
    )

    return pd.DataFrame(checks)


def main():
    ensure_measurements_extracted()

    core = build_core_measurements()
    weather = build_weather_daily()
    fruit_size = build_fruit_size_samples()
    tagged_fruit = build_tagged_fruit_daily_summary()
    plant_biomass = build_plant_biomass_samples()

    yield_ts = core.merge(
        weather,
        on=["year", "date"],
        how="left",
    )

    if not fruit_size.empty:
        yield_ts = yield_ts.merge(
            fruit_size,
            on=["year", "date", "nitrogen_treatment"],
            how="left",
        )

    if not tagged_fruit.empty:
        yield_ts = yield_ts.merge(
            tagged_fruit,
            on=["year", "date"],
            how="left",
        )

    yield_ts = yield_ts.sort_values(["year", "date", "nitrogen_treatment"])

    quality_report = build_quality_report(yield_ts, weather)

    yield_path = OUTDIR / "strawberry_yield_timeseries.csv"
    weather_path = OUTDIR / "strawberry_weather_daily.csv"
    biomass_path = OUTDIR / "strawberry_plant_biomass_samples.csv"
    quality_path = OUTDIR / "strawberry_data_quality_report.csv"

    yield_ts.to_csv(yield_path, index=False)
    weather.to_csv(weather_path, index=False)
    plant_biomass.to_csv(biomass_path, index=False)
    quality_report.to_csv(quality_path, index=False)

    con = duckdb.connect(str(DB_PATH))

    tables = {
        "strawberry_yield_timeseries": yield_ts,
        "strawberry_weather_daily": weather,
        "strawberry_plant_biomass_samples": plant_biomass,
        "strawberry_data_quality_report": quality_report,
    }

    for table_name, table_df in tables.items():
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.register("tmp_df", table_df)
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")

    con.close()

    print("Saved:")
    print(yield_path)
    print(weather_path)
    print(biomass_path)
    print(quality_path)
    print(DB_PATH)

    print("\nJoined strawberry yield time series:")
    print(yield_ts.head())
    print("\nShape:", yield_ts.shape)

    print("\nQuality report:")
    print(quality_report)


if __name__ == "__main__":
    main()
