from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st


DB_PATH = Path("db/yield_forecasting.duckdb")


st.set_page_config(
    page_title="Strawberry Yield Forecasting Case Study",
    page_icon="🍓",
    layout="wide",
)


def file_mtime(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_mtime


@st.cache_data
def load_table(table_name: str, db_mtime: float) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH))
    df = con.execute(f"SELECT * FROM {table_name}").fetchdf()
    con.close()
    return df


def convert_dates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column])
    return df


def get_row(df: pd.DataFrame, column: str, value: str):
    subset = df.loc[df[column] == value]
    if subset.empty:
        return None
    return subset.iloc[0]


def fmt_number(value, digits=2):
    if pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def forecast_confidence_label(best_rolling_row) -> str:
    r2 = best_rolling_row["r2"]
    improvement = best_rolling_row["mae_improvement_pct_vs_persistence"]

    if r2 > 0.4 and improvement > 15:
        return "Medium"
    if improvement > 0:
        return "Low-to-moderate"
    return "Low"


def status_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("status", as_index=False)
        .agg(n_checks=("check", "size"))
        .sort_values("status")
    )


if not DB_PATH.exists():
    st.error(
        "Database not found. Run scripts 01–05 first to build the processed tables."
    )
    st.stop()


db_mtime = file_mtime(DB_PATH)

yield_ts = load_table("strawberry_yield_timeseries", db_mtime)
forecast_features = load_table("strawberry_forecast_features", db_mtime)
raw_quality = load_table("strawberry_data_quality_report", db_mtime)
forecast_quality = load_table("strawberry_forecast_feature_quality", db_mtime)

strict_metrics = load_table("strawberry_forecast_model_metrics", db_mtime)
strict_predictions = load_table("strawberry_forecast_predictions", db_mtime)

rolling_metrics = load_table("strawberry_rolling_forecast_metrics", db_mtime)
rolling_predictions = load_table("strawberry_rolling_forecast_predictions", db_mtime)
rolling_training_history = load_table(
    "strawberry_rolling_forecast_training_history",
    db_mtime,
)
rolling_effects = load_table("strawberry_rolling_forecast_feature_effects", db_mtime)

customer_findings = load_table("strawberry_forecast_customer_findings", db_mtime)
year_shift = load_table("strawberry_year_to_year_shift_by_treatment", db_mtime)
weather_shift = load_table("strawberry_weather_shift_summary", db_mtime)

yield_ts = convert_dates(yield_ts, ["date"])
forecast_features = convert_dates(
    forecast_features,
    ["current_observation_date", "target_date"],
)
strict_predictions = convert_dates(
    strict_predictions,
    ["current_observation_date", "target_date"],
)
rolling_predictions = convert_dates(
    rolling_predictions,
    ["current_observation_date", "target_date"],
)
rolling_training_history = convert_dates(
    rolling_training_history,
    ["forecast_origin_date"],
)


st.title("🍓 Strawberry Yield Forecasting Case Study")

st.markdown(
    """
    A compact forecasting case study built around a practical agronomy question:

    **Can next-observation strawberry fresh matter be forecast from weather, GDD,
    fruit counts, crop development signals, and treatment information?**

    The workflow separates two scenarios:

    **A. Strict year-to-year audit:** train on 2022 and test on 2023.  
    **B. Rolling in-season forecast:** update the model during 2023 as new observations become available.
    """
)


# ---------------------------------------------------------------------
# Executive summary
# ---------------------------------------------------------------------

st.header("1. Executive summary")

strict_best = strict_metrics.sort_values("mae").iloc[0]
strict_persistence = get_row(strict_metrics, "model", "baseline_persistence")

rolling_best = rolling_metrics.sort_values("mae").iloc[0]
rolling_persistence = get_row(rolling_metrics, "model", "rolling_persistence")

confidence = forecast_confidence_label(rolling_best)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Forecast rows",
    f"{len(forecast_features):,}",
)

col2.metric(
    "Forecast horizon",
    f"{forecast_features['forecast_horizon_days'].mean():.1f} days",
)

col3.metric(
    "Best rolling MAE",
    f"{rolling_best['mae']:.2f} g",
)

col4.metric(
    "Rolling uplift vs persistence",
    f"{rolling_best['mae_improvement_pct_vs_persistence']:.1f}%",
)

st.info(
    f"""
    **Main result:** strict year-to-year forecasting did not beat the simple persistence
    baseline. However, the rolling in-season random forest improved MAE by
    **{rolling_best['mae_improvement_pct_vs_persistence']:.1f}%** versus persistence.

    **Forecast confidence:** **{confidence}**.  
    This is a useful prototype workflow, but not yet a production-grade forecasting system.
    """
)


# ---------------------------------------------------------------------
# Data readiness
# ---------------------------------------------------------------------

st.header("2. Data readiness audit")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Raw joined dataset checks")
    st.dataframe(raw_quality, use_container_width=True, hide_index=True)

with col2:
    st.subheader("Forecast feature checks")
    st.dataframe(forecast_quality, use_container_width=True, hide_index=True)

fail_count = int(
    (raw_quality["status"].eq("FAIL").sum())
    + (forecast_quality["status"].eq("FAIL").sum())
)

warn_count = int(
    (raw_quality["status"].eq("WARN").sum())
    + (forecast_quality["status"].eq("WARN").sum())
)

if fail_count == 0 and warn_count == 0:
    st.success(
        "No failing or warning data-quality checks were found in the current processed dataset."
    )
elif fail_count == 0:
    st.warning(
        f"{warn_count} warning checks were found. These should be reviewed before using the forecast operationally."
    )
else:
    st.error(
        f"{fail_count} failing checks and {warn_count} warning checks were found."
    )


# ---------------------------------------------------------------------
# Forecasting setup
# ---------------------------------------------------------------------

st.header("3. Forecasting setup")

setup_col1, setup_col2 = st.columns(2)

with setup_col1:
    st.markdown(
        """
        **Target**

        `next_fresh_matter_g`

        This is the fresh matter measured at the next observation date for the same
        nitrogen treatment.
        """
    )

    setup_table = pd.DataFrame(
        [
            {
                "Input group": "Treatment / management",
                "Examples": "0N, 50N, 100N, 150N",
            },
            {
                "Input group": "Weather",
                "Examples": "temperature, relative humidity, solar radiation",
            },
            {
                "Input group": "Thermal time",
                "Examples": "daily GDD, cumulative GDD base 10",
            },
            {
                "Input group": "Crop status",
                "Examples": "current fruit number, current fresh matter, dry matter",
            },
            {
                "Input group": "Recent development",
                "Examples": "previous values, changes, rolling means",
            },
        ]
    )

    st.dataframe(setup_table, use_container_width=True, hide_index=True)

with setup_col2:
    horizon_summary = (
        forecast_features.groupby("year", as_index=False)
        .agg(
            n_rows=("forecast_task", "size"),
            n_origin_dates=("current_observation_date", "nunique"),
            min_horizon=("forecast_horizon_days", "min"),
            max_horizon=("forecast_horizon_days", "max"),
            mean_horizon=("forecast_horizon_days", "mean"),
        )
    )

    st.markdown("**Forecast horizon by year**")
    st.dataframe(horizon_summary, use_container_width=True, hide_index=True)

    fig = px.histogram(
        forecast_features,
        x="forecast_horizon_days",
        nbins=12,
        title="Forecast horizon distribution",
        labels={"forecast_horizon_days": "Forecast horizon in days"},
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# Year-to-year audit
# ---------------------------------------------------------------------

st.header("4. A — Strict year-to-year forecast audit")

st.markdown(
    """
    First, the workflow tests a strict temporal holdout:

    **Train on 2022 → Test on 2023**

    This is intentionally conservative. It checks whether one historical season is enough
    to generalize to the next one.
    """
)

fig = px.bar(
    strict_metrics.sort_values("mae"),
    x="model_label",
    y="mae",
    hover_data=["rmse", "r2", "bias", "mae_pct_of_mean_observed"],
    title="Strict year-to-year validation: MAE by model",
    labels={
        "model_label": "Model",
        "mae": "MAE (g)",
    },
)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    strict_metrics[
        [
            "model_label",
            "mae",
            "rmse",
            "r2",
            "bias",
            "mae_improvement_pct_vs_persistence",
            "note",
        ]
    ].sort_values("mae"),
    use_container_width=True,
    hide_index=True,
)

if strict_best["model"] == "baseline_persistence":
    st.warning(
        """
        The best strict year-to-year model is the persistence baseline. More complex
        models did not improve generalization from 2022 to 2023. This is a useful
        warning: one previous season is not enough for a robust standalone forecast.
        """
    )
else:
    st.info(
        f"""
        The best strict year-to-year model was **{strict_best['model_label']}**,
        with MAE={strict_best['mae']:.2f} g.
        """
    )

with st.expander("Why did year-to-year forecasting struggle?"):
    st.subheader("Treatment-level shift between 2022 and 2023")

    fig = px.bar(
        year_shift,
        x="nitrogen_treatment",
        y="target_pct_change_2023_vs_2022",
        title="Target shift by nitrogen treatment: 2023 vs 2022",
        labels={
            "nitrogen_treatment": "Nitrogen treatment",
            "target_pct_change_2023_vs_2022": "Target change (%)",
        },
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(year_shift, use_container_width=True, hide_index=True)

    st.subheader("Weather shift summary")
    st.dataframe(weather_shift, use_container_width=True, hide_index=True)

    st.subheader("Customer-facing findings")
    st.dataframe(customer_findings, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------
# Rolling forecast
# ---------------------------------------------------------------------

st.header("5. B — Rolling in-season forecast")

st.markdown(
    """
    The second scenario is closer to operational yield forecasting.

    For each 2023 forecast date, the model is trained with all previous information:

    - all 2022 observations;
    - plus 2023 observations already known before the forecast date.

    This simulates an in-season workflow where forecasts are updated as new crop
    observations arrive.
    """
)

fig = px.bar(
    rolling_metrics.sort_values("mae"),
    x="model_label",
    y="mae",
    hover_data=[
        "rmse",
        "r2",
        "bias",
        "mae_improvement_pct_vs_persistence",
    ],
    title="Rolling in-season validation: MAE by model",
    labels={
        "model_label": "Model",
        "mae": "MAE (g)",
    },
)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(
    rolling_metrics[
        [
            "model_label",
            "n_predictions",
            "mae",
            "rmse",
            "r2",
            "bias",
            "mae_improvement_pct_vs_persistence",
        ]
    ].sort_values("mae"),
    use_container_width=True,
    hide_index=True,
)

st.success(
    f"""
    The best rolling model was **{rolling_best['model_label']}**.

    It achieved MAE={rolling_best['mae']:.2f} g and improved MAE by
    **{rolling_best['mae_improvement_pct_vs_persistence']:.1f}%**
    versus the rolling persistence baseline.
    """
)

selected_rolling_model = st.selectbox(
    "Select rolling model to inspect",
    rolling_metrics.sort_values("mae")["model"].tolist(),
    format_func=lambda model: rolling_metrics.set_index("model")
    .loc[model, "model_label"],
)

selected_preds = rolling_predictions.loc[
    rolling_predictions["model"] == selected_rolling_model
].copy()

st.subheader("Observed vs predicted fresh matter")

fig = px.scatter(
    selected_preds,
    x="observed",
    y="predicted",
    hover_data=[
        "nitrogen_treatment",
        "current_observation_date",
        "target_date",
        "forecast_horizon_days",
        "current_fruit_number",
        "current_fresh_matter_g",
    ],
    title="Rolling forecast: predicted vs observed",
    labels={
        "observed": "Observed next fresh matter (g)",
        "predicted": "Predicted next fresh matter (g)",
    },
)

min_value = min(selected_preds["observed"].min(), selected_preds["predicted"].min())
max_value = max(selected_preds["observed"].max(), selected_preds["predicted"].max())

fig.add_shape(
    type="line",
    x0=min_value,
    y0=min_value,
    x1=max_value,
    y1=max_value,
    line={"dash": "dash"},
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("How the rolling training set grows")

fig = px.line(
    rolling_training_history,
    x="forecast_origin_date",
    y=[
        "n_previous_year_training_rows",
        "n_current_year_training_rows",
    ],
    markers=True,
    title="Training data available at each forecast update",
    labels={
        "forecast_origin_date": "Forecast origin date",
        "value": "Training rows",
        "variable": "Training source",
    },
)
st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# Customer one-pager
# ---------------------------------------------------------------------

st.header("6. Customer-facing forecast summary")

st.markdown(
    f"""
    ### Forecast conclusion

    The strict year-to-year model audit shows that **one historical season is not enough**
    to produce a reliable standalone forecast for the next season.

    The rolling in-season approach performs better: once current-season observations are
    incorporated, the best model improves MAE by **{rolling_best['mae_improvement_pct_vs_persistence']:.1f}%**
    versus a simple persistence baseline.

    ### Recommended operational interpretation

    This prototype supports **in-season forecast updating**, not fully automated
    season-ahead forecasting.

    ### Highest-value data improvements

    1. Add more seasons and locations.
    2. Track consistent fruit counts before each harvest window.
    3. Record cultivar/block/plot metadata.
    4. Capture management events such as irrigation, fertigation and pruning.
    5. Validate forecasts against commercial harvest totals.
    """
)

st.warning(
    """
    This dashboard is a forecasting-readiness prototype. It demonstrates data profiling,
    feature engineering, baseline comparison, temporal validation and stakeholder-facing
    interpretation. It should not be treated as a production yield forecast without more
    seasons, locations and operational metadata.
    """
)


# ---------------------------------------------------------------------
# Technical appendix
# ---------------------------------------------------------------------

st.header("7. Technical appendix")

with st.expander("Rolling model feature effects"):
    if rolling_effects.empty:
        st.info("No feature effects available for the selected rolling models.")
    else:
        selected_effects = rolling_effects.loc[
            rolling_effects["model"] == selected_rolling_model
        ].copy()

        if selected_effects.empty:
            st.info("No feature effects available for this model.")
        else:
            effect_summary = (
                selected_effects.groupby(
                    ["model", "model_label", "feature", "effect_type"],
                    as_index=False,
                )
                .agg(
                    mean_effect=("effect_value", "mean"),
                    mean_abs_effect=("abs_effect_value", "mean"),
                )
                .sort_values("mean_abs_effect", ascending=False)
                .head(20)
            )

            fig = px.bar(
                effect_summary,
                x="mean_abs_effect",
                y="feature",
                orientation="h",
                title="Top rolling model feature effects",
                labels={
                    "mean_abs_effect": "Mean absolute effect",
                    "feature": "Feature",
                },
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(effect_summary, use_container_width=True, hide_index=True)

with st.expander("Rolling predictions"):
    st.dataframe(rolling_predictions, use_container_width=True, hide_index=True)

with st.expander("Strict year-to-year predictions"):
    st.dataframe(strict_predictions, use_container_width=True, hide_index=True)

with st.expander("Forecast feature dataset"):
    st.dataframe(forecast_features, use_container_width=True, hide_index=True)

with st.expander("Joined strawberry time series"):
    st.dataframe(yield_ts, use_container_width=True, hide_index=True)
