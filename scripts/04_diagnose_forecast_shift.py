from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


DB_PATH = Path("db/yield_forecasting.duckdb")
OUTDIR = Path("data/processed/strawberry")
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET = "next_fresh_matter_g"


def load_table(table_name: str) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH))
    df = con.execute(f"SELECT * FROM {table_name}").fetchdf()
    con.close()
    return df


def safe_r2(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan

    denominator = np.sum((y_true - np.mean(y_true)) ** 2)

    if denominator == 0:
        return np.nan

    numerator = np.sum((y_true - y_pred) ** 2)
    return 1 - numerator / denominator


def build_target_shift_summary(forecast_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        forecast_df.groupby(["year", "nitrogen_treatment"], as_index=False)
        .agg(
            n_rows=(TARGET, "size"),
            mean_next_fresh_matter_g=(TARGET, "mean"),
            median_next_fresh_matter_g=(TARGET, "median"),
            std_next_fresh_matter_g=(TARGET, "std"),
            mean_current_fresh_matter_g=("current_fresh_matter_g", "mean"),
            mean_current_fruit_number=("current_fruit_number", "mean"),
            mean_cumulative_gdd=("cumulative_gdd_base_10", "mean"),
            mean_cumulative_solar_radiation=("cumulative_solar_radiation", "mean"),
            mean_tmean_7d=("tmean_7d", "mean"),
            mean_solar_radiation_7d=("solar_radiation_7d", "mean"),
        )
    )

    wide_rows = []

    for treatment in sorted(summary["nitrogen_treatment"].dropna().unique()):
        subset = summary.loc[summary["nitrogen_treatment"] == treatment].copy()

        if set(subset["year"]) >= {2022, 2023}:
            row_2022 = subset.loc[subset["year"] == 2022].iloc[0]
            row_2023 = subset.loc[subset["year"] == 2023].iloc[0]

            wide_rows.append(
                {
                    "nitrogen_treatment": treatment,
                    "mean_target_2022": row_2022["mean_next_fresh_matter_g"],
                    "mean_target_2023": row_2023["mean_next_fresh_matter_g"],
                    "target_difference_2023_minus_2022": row_2023[
                        "mean_next_fresh_matter_g"
                    ]
                    - row_2022["mean_next_fresh_matter_g"],
                    "target_pct_change_2023_vs_2022": (
                        row_2023["mean_next_fresh_matter_g"]
                        - row_2022["mean_next_fresh_matter_g"]
                    )
                    / row_2022["mean_next_fresh_matter_g"]
                    * 100,
                    "fruit_number_difference_2023_minus_2022": row_2023[
                        "mean_current_fruit_number"
                    ]
                    - row_2022["mean_current_fruit_number"],
                    "gdd_difference_2023_minus_2022": row_2023[
                        "mean_cumulative_gdd"
                    ]
                    - row_2022["mean_cumulative_gdd"],
                    "solar_difference_2023_minus_2022": row_2023[
                        "mean_cumulative_solar_radiation"
                    ]
                    - row_2022["mean_cumulative_solar_radiation"],
                }
            )

    shift = pd.DataFrame(wide_rows)

    return summary, shift


def build_weather_shift_summary(weather_df: pd.DataFrame) -> pd.DataFrame:
    return (
        weather_df.groupby("year", as_index=False)
        .agg(
            n_weather_days=("date", "size"),
            mean_tmean_c=("tmean_c", "mean"),
            mean_tmax_c=("tmax_c", "mean"),
            mean_tmin_c=("tmin_c", "mean"),
            mean_rhmean_pct=("rhmean_pct", "mean"),
            mean_solar_radiation=("solar_radiation", "mean"),
            final_cumulative_gdd=("cumulative_gdd_base_10", "max"),
            final_cumulative_solar_radiation=("cumulative_solar_radiation", "max"),
        )
    )


def build_model_metrics_by_treatment(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = ["model", "model_label", "nitrogen_treatment"]

    for keys, group in predictions_df.groupby(group_cols):
        model, model_label, treatment = keys

        y_true = group["observed"].astype(float).values
        y_pred = group["predicted"].astype(float).values

        rows.append(
            {
                "model": model,
                "model_label": model_label,
                "nitrogen_treatment": treatment,
                "n_rows": len(group),
                "mae": np.mean(np.abs(y_true - y_pred)),
                "rmse": np.sqrt(np.mean((y_true - y_pred) ** 2)),
                "bias": np.mean(y_pred - y_true),
                "r2": safe_r2(y_true, y_pred),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["model", "nitrogen_treatment"]
    )


def build_customer_findings(metrics_df, shift_by_treatment, weather_shift):
    best_model = metrics_df.sort_values("mae").iloc[0]
    persistence = metrics_df.loc[
        metrics_df["model"] == "baseline_persistence"
    ].iloc[0]

    best_mae_gain = persistence["mae"] - best_model["mae"]
    best_mae_gain_pct = best_mae_gain / persistence["mae"] * 100

    mean_abs_target_shift = shift_by_treatment[
        "target_pct_change_2023_vs_2022"
    ].abs().mean()

    gdd_2022 = weather_shift.loc[
        weather_shift["year"] == 2022, "final_cumulative_gdd"
    ].iloc[0]
    gdd_2023 = weather_shift.loc[
        weather_shift["year"] == 2023, "final_cumulative_gdd"
    ].iloc[0]

    rows = [
        {
            "finding": "Strict year-to-year validation",
            "result": (
                f"The best model was '{best_model['model_label']}' with "
                f"MAE={best_model['mae']:.2f} and R²={best_model['r2']:.3f}."
            ),
            "interpretation": (
                "The model does not generalize strongly from 2022 to 2023. "
                "This suggests that one previous season is not enough for a robust standalone forecast."
            ),
        },
        {
            "finding": "Baseline comparison",
            "result": (
                f"Best MAE improvement vs persistence baseline: "
                f"{best_mae_gain:.2f} g ({best_mae_gain_pct:.1f}%)."
            ),
            "interpretation": (
                "Complex models should not be trusted unless they beat a simple operational baseline."
            ),
        },
        {
            "finding": "Season shift",
            "result": (
                f"Average absolute treatment-level target shift between 2022 and 2023: "
                f"{mean_abs_target_shift:.1f}%."
            ),
            "interpretation": (
                "Large season-to-season differences make blind year-to-year forecasting difficult."
            ),
        },
        {
            "finding": "Thermal time difference",
            "result": (
                f"Final cumulative GDD changed from {gdd_2022:.1f} in 2022 "
                f"to {gdd_2023:.1f} in 2023."
            ),
            "interpretation": (
                "Weather and crop-development timing differ between campaigns, so forecasts need in-season updating."
            ),
        },
    ]

    return pd.DataFrame(rows)


def main():
    forecast_df = load_table("strawberry_forecast_features")
    weather_df = load_table("strawberry_weather_daily")
    predictions_df = load_table("strawberry_forecast_predictions")
    metrics_df = load_table("strawberry_forecast_model_metrics")

    forecast_df["current_observation_date"] = pd.to_datetime(
        forecast_df["current_observation_date"]
    )
    forecast_df["target_date"] = pd.to_datetime(forecast_df["target_date"])
    weather_df["date"] = pd.to_datetime(weather_df["date"])

    target_summary, shift_by_treatment = build_target_shift_summary(forecast_df)
    weather_shift = build_weather_shift_summary(weather_df)
    metrics_by_treatment = build_model_metrics_by_treatment(predictions_df)
    customer_findings = build_customer_findings(
        metrics_df=metrics_df,
        shift_by_treatment=shift_by_treatment,
        weather_shift=weather_shift,
    )

    outputs = {
        "strawberry_target_summary_by_year_treatment": target_summary,
        "strawberry_year_to_year_shift_by_treatment": shift_by_treatment,
        "strawberry_weather_shift_summary": weather_shift,
        "strawberry_forecast_metrics_by_treatment": metrics_by_treatment,
        "strawberry_forecast_customer_findings": customer_findings,
    }

    con = duckdb.connect(str(DB_PATH))

    for table_name, table_df in outputs.items():
        csv_path = OUTDIR / f"{table_name}.csv"
        table_df.to_csv(csv_path, index=False)

        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.register("tmp_df", table_df)
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")

        print(f"Saved: {csv_path}")

    con.close()

    print("\nYear-to-year target shift by treatment:")
    print(shift_by_treatment)

    print("\nWeather shift summary:")
    print(weather_shift)

    print("\nCustomer-facing findings:")
    print(customer_findings)


if __name__ == "__main__":
    main()
