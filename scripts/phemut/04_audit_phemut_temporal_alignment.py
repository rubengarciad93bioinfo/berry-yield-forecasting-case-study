#!/usr/bin/env python3
"""Audit PheMuT temporal alignment and lag behaviour.

This script is deliberately diagnostic: it does not train models. It checks
whether the tidy observations and forecast-feature tables are aligned as
intended before interpreting model results.

Main questions:
1. Does each forecast row use yield/features from the forecast origin date?
2. Does next_yield_g equal the yield observed at the target date for the same plot?
3. Do aggregate current/target values equal the sum of plot-level observations?
4. Are selected model features free of obvious future-target columns?
5. Do model curves look lagged because they track the previous observed window?

Outputs:
- phemut_temporal_alignment_quality.csv
- phemut_temporal_alignment_plot_sample.csv
- phemut_temporal_alignment_aggregate_sample.csv
- phemut_temporal_alignment_feature_leakage_audit.csv
- phemut_temporal_alignment_lag_audit.csv
- phemut_temporal_alignment_report.md
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

DEFAULT_PROCESSED_DIR = Path("data/processed/phemut")

PHENOLOGY_COLUMNS = [
    "flower_count",
    "green_fruit_count",
    "white_fruit_count",
    "pink_fruit_count",
    "red_or_ripe_fruit_count",
]
CANOPY_COLUMNS = ["canopy_area", "canopy_depth", "canopy_volume"]
FORBIDDEN_FEATURE_PATTERNS = [
    r"^next_",
    r"^target_",
    r"_next_",
    r"future",
    r"observed_yield",
    r"predicted_yield",
    r"error_g$",
    r"absolute_error",
]


@dataclass(frozen=True)
class Check:
    check: str
    status: str
    value: object
    threshold: object
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit PheMuT temporal alignment and model timeliness.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--sample-plots", type=int, default=5, help="Number of plots to include in the human-readable sample.")
    return parser.parse_args()


def read_csv_if_exists(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def corr_safe(a: pd.Series, b: pd.Series) -> float:
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < 3:
        return np.nan
    if aligned.iloc[:, 0].std() == 0 or aligned.iloc[:, 1].std() == 0:
        return np.nan
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def mae_safe(a: pd.Series, b: pd.Series) -> float:
    aligned = pd.concat([a, b], axis=1).dropna()
    if aligned.empty:
        return np.nan
    return float(mean_absolute_error(aligned.iloc[:, 0], aligned.iloc[:, 1]))


def status_pass_fail(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def status_pass_warn(condition: bool) -> str:
    return "PASS" if condition else "WARN"


def build_plot_alignment(plot_features: pd.DataFrame, observations: pd.DataFrame) -> tuple[pd.DataFrame, list[Check]]:
    checks: list[Check] = []
    if plot_features.empty or observations.empty:
        checks.append(Check("plot_alignment_available", "FAIL", 0, "> 0", "Missing plot features or tidy observations."))
        return pd.DataFrame(), checks

    obs = observations.copy()
    obs["observation_date"] = pd.to_datetime(obs["observation_date"])
    pf = plot_features.copy()
    pf["observation_date"] = pd.to_datetime(pf["observation_date"])
    pf["target_date"] = pd.to_datetime(pf["target_date"])

    current_lookup = obs[["season", "observation_date", "plot_id", "yield_g"]].rename(
        columns={"yield_g": "current_yield_lookup_g"}
    )
    target_lookup = obs[["season", "observation_date", "plot_id", "yield_g"]].rename(
        columns={"observation_date": "target_date", "yield_g": "target_yield_lookup_g"}
    )

    aligned = pf.merge(current_lookup, on=["season", "observation_date", "plot_id"], how="left")
    aligned = aligned.merge(target_lookup, on=["season", "target_date", "plot_id"], how="left")
    aligned["current_yield_diff_g"] = aligned["yield_g"] - aligned["current_yield_lookup_g"]
    aligned["target_yield_diff_g"] = aligned["next_yield_g"] - aligned["target_yield_lookup_g"]
    aligned["target_after_origin"] = aligned["target_date"] > aligned["observation_date"]

    current_max_abs = float(aligned["current_yield_diff_g"].abs().max()) if len(aligned) else np.nan
    target_max_abs = float(aligned["target_yield_diff_g"].abs().max()) if len(aligned) else np.nan
    missing_current = float(aligned["current_yield_lookup_g"].isna().mean()) if len(aligned) else np.nan
    missing_target = float(aligned["target_yield_lookup_g"].isna().mean()) if len(aligned) else np.nan
    non_future = int((~aligned["target_after_origin"]).sum())

    checks.extend(
        [
            Check("plot_current_yield_lookup_missing_share", status_pass_fail(missing_current == 0), round(missing_current, 6), "0", "Every plot forecast origin should map to a current tidy observation."),
            Check("plot_target_yield_lookup_missing_share", status_pass_fail(missing_target == 0), round(missing_target, 6), "0", "Every plot target date should map to a future tidy observation."),
            Check("plot_current_yield_max_abs_diff_g", status_pass_fail(pd.notna(current_max_abs) and current_max_abs < 1e-9), round(current_max_abs, 9) if pd.notna(current_max_abs) else np.nan, "0", "Feature current yield must equal tidy yield at forecast origin."),
            Check("plot_target_yield_max_abs_diff_g", status_pass_fail(pd.notna(target_max_abs) and target_max_abs < 1e-9), round(target_max_abs, 9) if pd.notna(target_max_abs) else np.nan, "0", "Target next_yield_g must equal tidy yield at target date."),
            Check("plot_non_future_target_rows", status_pass_fail(non_future == 0), non_future, "0", "All target dates must be after forecast origin dates."),
            Check("plot_forecast_horizon_min_days", status_pass_warn(aligned["forecast_horizon_days"].min() > 0), int(aligned["forecast_horizon_days"].min()), "> 0", "Forecast horizon should be strictly positive."),
            Check("plot_forecast_horizon_max_days", status_pass_warn(aligned["forecast_horizon_days"].max() <= 21), int(aligned["forecast_horizon_days"].max()), "<= 21", "Long gaps should be reviewed."),
        ]
    )

    sample_cols = [
        "season",
        "plot_id",
        "plot_prefix",
        "plot_block",
        "plot_replicate",
        "observation_date",
        "target_date",
        "forecast_horizon_days",
        "yield_g",
        "next_yield_g",
        "delta_next_yield_g",
        "current_yield_lookup_g",
        "target_yield_lookup_g",
        "current_yield_diff_g",
        "target_yield_diff_g",
    ]
    for col in PHENOLOGY_COLUMNS + CANOPY_COLUMNS:
        if col in aligned.columns:
            sample_cols.append(col)
    return aligned[[col for col in sample_cols if col in aligned.columns]].copy(), checks


def build_aggregate_alignment(aggregate_features: pd.DataFrame, observations: pd.DataFrame) -> tuple[pd.DataFrame, list[Check]]:
    checks: list[Check] = []
    if aggregate_features.empty or observations.empty:
        checks.append(Check("aggregate_alignment_available", "FAIL", 0, "> 0", "Missing aggregate features or tidy observations."))
        return pd.DataFrame(), checks

    obs = observations.copy()
    obs["observation_date"] = pd.to_datetime(obs["observation_date"])
    observed_agg = (
        obs.groupby(["season", "observation_date"], as_index=False)
        .agg(
            lookup_aggregate_yield_g=("yield_g", "sum"),
            lookup_n_plots=("plot_id", "nunique"),
            lookup_mean_plot_yield_g=("yield_g", "mean"),
        )
    )

    af = aggregate_features.copy()
    af["observation_date"] = pd.to_datetime(af["observation_date"])
    af["target_date"] = pd.to_datetime(af["target_date"])

    current_lookup = observed_agg.rename(columns={"lookup_aggregate_yield_g": "current_lookup_aggregate_yield_g", "lookup_n_plots": "current_lookup_n_plots", "lookup_mean_plot_yield_g": "current_lookup_mean_plot_yield_g"})
    target_lookup = observed_agg.rename(columns={"observation_date": "target_date", "lookup_aggregate_yield_g": "target_lookup_aggregate_yield_g", "lookup_n_plots": "target_lookup_n_plots", "lookup_mean_plot_yield_g": "target_lookup_mean_plot_yield_g"})

    aligned = af.merge(current_lookup, on=["season", "observation_date"], how="left")
    aligned = aligned.merge(target_lookup, on=["season", "target_date"], how="left")
    aligned["current_aggregate_diff_g"] = aligned["aggregate_yield_g"] - aligned["current_lookup_aggregate_yield_g"]
    aligned["target_aggregate_diff_g"] = aligned["aggregate_next_yield_g"] - aligned["target_lookup_aggregate_yield_g"]
    aligned["target_after_origin"] = aligned["target_date"] > aligned["observation_date"]

    current_max_abs = float(aligned["current_aggregate_diff_g"].abs().max()) if len(aligned) else np.nan
    target_max_abs = float(aligned["target_aggregate_diff_g"].abs().max()) if len(aligned) else np.nan
    missing_current = float(aligned["current_lookup_aggregate_yield_g"].isna().mean()) if len(aligned) else np.nan
    missing_target = float(aligned["target_lookup_aggregate_yield_g"].isna().mean()) if len(aligned) else np.nan
    non_future = int((~aligned["target_after_origin"]).sum())
    n_plot_mismatch = int((aligned.get("n_plots", aligned["current_lookup_n_plots"]) != aligned["current_lookup_n_plots"]).sum()) if "n_plots" in aligned.columns else 0

    checks.extend(
        [
            Check("aggregate_current_lookup_missing_share", status_pass_fail(missing_current == 0), round(missing_current, 6), "0", "Every aggregate forecast origin should map to current aggregate observations."),
            Check("aggregate_target_lookup_missing_share", status_pass_fail(missing_target == 0), round(missing_target, 6), "0", "Every aggregate target date should map to future aggregate observations."),
            Check("aggregate_current_yield_max_abs_diff_g", status_pass_fail(pd.notna(current_max_abs) and current_max_abs < 1e-9), round(current_max_abs, 9) if pd.notna(current_max_abs) else np.nan, "0", "Aggregate current yield must equal sum of plot yields at forecast origin."),
            Check("aggregate_target_yield_max_abs_diff_g", status_pass_fail(pd.notna(target_max_abs) and target_max_abs < 1e-9), round(target_max_abs, 9) if pd.notna(target_max_abs) else np.nan, "0", "Aggregate target yield must equal sum of plot yields at target date."),
            Check("aggregate_non_future_target_rows", status_pass_fail(non_future == 0), non_future, "0", "All aggregate target dates must be after forecast origin dates."),
            Check("aggregate_n_plot_mismatch_rows", status_pass_warn(n_plot_mismatch == 0), n_plot_mismatch, "0", "Aggregate feature n_plots should equal tidy plot count at forecast origin."),
        ]
    )

    keep = [
        "season",
        "observation_date",
        "target_date",
        "forecast_horizon_days",
        "n_plots",
        "current_lookup_n_plots",
        "target_lookup_n_plots",
        "aggregate_yield_g",
        "aggregate_next_yield_g",
        "aggregate_delta_next_yield_g",
        "current_lookup_aggregate_yield_g",
        "target_lookup_aggregate_yield_g",
        "current_aggregate_diff_g",
        "target_aggregate_diff_g",
        "aggregate_flower_count",
        "aggregate_green_fruit_count",
        "aggregate_white_fruit_count",
        "aggregate_pink_fruit_count",
        "aggregate_red_or_ripe_fruit_count",
    ]
    return aligned[[col for col in keep if col in aligned.columns]].copy(), checks


def build_feature_leakage_audit(processed_dir: Path, plot_features: pd.DataFrame, aggregate_features: pd.DataFrame) -> tuple[pd.DataFrame, list[Check]]:
    checks: list[Check] = []
    feature_sets_path = processed_dir / "phemut_feature_sets.csv"
    if feature_sets_path.exists():
        feature_sets = pd.read_csv(feature_sets_path)
        if "feature" not in feature_sets.columns:
            feature_sets = pd.DataFrame()
    else:
        # Fallback: audit all feature columns by table. This is less precise than
        # the model feature set, but still catches obvious future-target columns.
        plot_exclude = {"next_yield_g", "delta_next_yield_g", "target_date", "forecast_task"}
        aggregate_exclude = {"aggregate_next_yield_g", "aggregate_delta_next_yield_g", "target_date", "forecast_task"}
        rows = []
        for level, df, exclude in [("plot", plot_features, plot_exclude), ("aggregate", aggregate_features, aggregate_exclude)]:
            for col in df.columns:
                if col not in exclude:
                    rows.append({"task_id": "unknown", "task_label": "unknown", "level": level, "feature": col})
        feature_sets = pd.DataFrame(rows)

    if feature_sets.empty:
        checks.append(Check("feature_set_available", "WARN", 0, "> 0", "No phemut_feature_sets.csv available yet; run script 03 after script 02."))
        return pd.DataFrame(columns=["task_id", "task_label", "level", "feature", "future_target_feature_warning"]), checks

    pattern = re.compile("|".join(FORBIDDEN_FEATURE_PATTERNS), flags=re.IGNORECASE)
    audit = feature_sets.copy()
    audit["future_target_feature_warning"] = audit["feature"].astype(str).str.contains(pattern)
    n_warn = int(audit["future_target_feature_warning"].sum())
    checks.append(Check("future_target_features_selected", status_pass_fail(n_warn == 0), n_warn, "0", "Selected features should not include target/next/future/error columns."))
    return audit, checks


def build_lag_audit(aggregate_board: pd.DataFrame) -> tuple[pd.DataFrame, list[Check]]:
    checks: list[Check] = []
    columns = [
        "task_id",
        "task_label",
        "level",
        "target_mode",
        "model_id",
        "model_label",
        "n_target_windows",
        "same_window_corr",
        "forecast_vs_current_window_corr",
        "forecast_vs_previous_observed_target_corr",
        "same_window_mae_g",
        "forecast_vs_current_window_mae_g",
        "forecast_vs_previous_observed_target_mae_g",
        "direction_accuracy_vs_current",
        "lag_warning",
        "lag_note",
    ]
    if aggregate_board.empty:
        checks.append(Check("model_lag_audit_available", "WARN", 0, "> 0", "No phemut_aggregate_board.csv available yet; run script 03 to audit model timeliness."))
        return pd.DataFrame(columns=columns), checks

    board = aggregate_board.copy()
    for col in ["forecast_origin_date", "target_date"]:
        if col in board.columns:
            board[col] = pd.to_datetime(board[col])

    rows = []
    group_cols = ["task_id", "task_label", "level", "target_mode", "model_id", "model_label"]
    for key, group in board.sort_values("target_date").groupby(group_cols, dropna=False):
        g = group.sort_values("target_date").reset_index(drop=True)
        forecast = g["forecasted_next_window_yield_g"]
        observed = g["observed_next_window_yield_g"]
        current = g["current_window_yield_g"]
        previous_target_observed = observed.shift(1)

        same_corr = corr_safe(forecast, observed)
        current_corr = corr_safe(forecast, current)
        prev_target_corr = corr_safe(forecast, previous_target_observed)
        same_mae = mae_safe(forecast, observed)
        current_mae = mae_safe(forecast, current)
        prev_target_mae = mae_safe(forecast, previous_target_observed)

        direction = g[["forecast_change_vs_current_pct", "observed_change_vs_current_pct"]].replace([np.inf, -np.inf], np.nan).dropna()
        if direction.empty:
            direction_accuracy = np.nan
        else:
            direction_accuracy = float((np.sign(direction["forecast_change_vs_current_pct"]) == np.sign(direction["observed_change_vs_current_pct"])).mean())

        lag_warning = False
        notes = []
        if pd.notna(current_corr) and pd.notna(same_corr) and current_corr > same_corr + 0.10:
            lag_warning = True
            notes.append("forecast aligns more with current window than target")
        if pd.notna(current_mae) and pd.notna(same_mae) and current_mae < same_mae * 0.90:
            lag_warning = True
            notes.append("forecast is closer to current window than target")
        if pd.notna(prev_target_corr) and pd.notna(same_corr) and prev_target_corr > same_corr + 0.10:
            lag_warning = True
            notes.append("forecast aligns more with previous target than target")
        if "persistence" in str(key[group_cols.index("model_id")]).lower():
            notes.append("persistence lag is expected by definition")

        row = dict(zip(group_cols, key))
        row.update(
            {
                "n_target_windows": len(g),
                "same_window_corr": same_corr,
                "forecast_vs_current_window_corr": current_corr,
                "forecast_vs_previous_observed_target_corr": prev_target_corr,
                "same_window_mae_g": same_mae,
                "forecast_vs_current_window_mae_g": current_mae,
                "forecast_vs_previous_observed_target_mae_g": prev_target_mae,
                "direction_accuracy_vs_current": direction_accuracy,
                "lag_warning": bool(lag_warning),
                "lag_note": "; ".join(notes) if notes else "no strong lag signal by this heuristic",
            }
        )
        rows.append(row)

    audit = pd.DataFrame(rows, columns=columns).sort_values(["lag_warning", "same_window_mae_g"], ascending=[False, True])
    n_lag = int(audit["lag_warning"].sum()) if not audit.empty else 0
    non_persistence = audit.loc[~audit["model_id"].astype(str).str.contains("persistence", case=False, na=False)] if not audit.empty else audit
    n_non_persistence_lag = int(non_persistence["lag_warning"].sum()) if not non_persistence.empty else 0
    checks.append(Check("lag_warning_models", status_pass_warn(n_lag == 0), n_lag, "0", "Number of model/task combinations with one-window lag warning."))
    checks.append(Check("non_persistence_lag_warning_models", status_pass_warn(n_non_persistence_lag == 0), n_non_persistence_lag, "0", "Lag warnings on learned/non-persistence models are more concerning."))
    return audit, checks


def make_human_plot_sample(plot_alignment: pd.DataFrame, sample_plots: int) -> pd.DataFrame:
    if plot_alignment.empty:
        return plot_alignment
    latest = sorted(plot_alignment["season"].astype(str).dropna().unique())[-1]
    latest_df = plot_alignment.loc[plot_alignment["season"].astype(str) == latest].copy()
    plots = sorted(latest_df["plot_id"].dropna().unique().tolist())[:sample_plots]
    sample = latest_df.loc[latest_df["plot_id"].isin(plots)].sort_values(["plot_id", "observation_date"])
    keep = [
        "season",
        "plot_id",
        "observation_date",
        "target_date",
        "forecast_horizon_days",
        "yield_g",
        "next_yield_g",
        "delta_next_yield_g",
        "flower_count",
        "green_fruit_count",
        "white_fruit_count",
        "pink_fruit_count",
        "red_or_ripe_fruit_count",
        "current_yield_diff_g",
        "target_yield_diff_g",
    ]
    return sample[[col for col in keep if col in sample.columns]].head(80)


def make_human_aggregate_sample(aggregate_alignment: pd.DataFrame) -> pd.DataFrame:
    if aggregate_alignment.empty:
        return aggregate_alignment
    latest = sorted(aggregate_alignment["season"].astype(str).dropna().unique())[-1]
    sample = aggregate_alignment.loc[aggregate_alignment["season"].astype(str) == latest].copy().sort_values("observation_date")
    keep = [
        "season",
        "observation_date",
        "target_date",
        "forecast_horizon_days",
        "n_plots",
        "aggregate_yield_g",
        "aggregate_next_yield_g",
        "aggregate_delta_next_yield_g",
        "current_aggregate_diff_g",
        "target_aggregate_diff_g",
        "aggregate_flower_count",
        "aggregate_green_fruit_count",
        "aggregate_white_fruit_count",
        "aggregate_pink_fruit_count",
        "aggregate_red_or_ripe_fruit_count",
    ]
    return sample[[col for col in keep if col in sample.columns]]


def write_report(path: Path, quality: pd.DataFrame, plot_sample: pd.DataFrame, aggregate_sample: pd.DataFrame, lag_audit: pd.DataFrame) -> None:
    def csv_preview(df: pd.DataFrame, n: int = 10) -> str:
        if df.empty:
            return "No rows."
        return df.head(n).to_csv(index=False).strip()

    text = f"""# PheMuT temporal alignment audit

## Quality checks

```csv
{csv_preview(quality, 50)}
```

## Plot-level alignment sample

```csv
{csv_preview(plot_sample, 20)}
```

## Aggregate alignment sample

```csv
{csv_preview(aggregate_sample, 20)}
```

## Lag/timeliness audit sample

```csv
{csv_preview(lag_audit, 20)}
```
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    processed_dir = args.processed_dir

    observations = read_csv_if_exists(processed_dir / "phemut_tidy_observations.csv", parse_dates=["observation_date"])
    plot_features = read_csv_if_exists(processed_dir / "phemut_forecast_features_plot_level.csv", parse_dates=["observation_date", "target_date"])
    aggregate_features = read_csv_if_exists(processed_dir / "phemut_forecast_features_aggregate.csv", parse_dates=["observation_date", "target_date"])
    aggregate_board = read_csv_if_exists(processed_dir / "phemut_aggregate_board.csv", parse_dates=["forecast_origin_date", "target_date"])

    all_checks: list[Check] = []
    plot_alignment, checks = build_plot_alignment(plot_features, observations)
    all_checks.extend(checks)
    aggregate_alignment, checks = build_aggregate_alignment(aggregate_features, observations)
    all_checks.extend(checks)
    feature_audit, checks = build_feature_leakage_audit(processed_dir, plot_features, aggregate_features)
    all_checks.extend(checks)
    lag_audit, checks = build_lag_audit(aggregate_board)
    all_checks.extend(checks)

    plot_sample = make_human_plot_sample(plot_alignment, args.sample_plots)
    aggregate_sample = make_human_aggregate_sample(aggregate_alignment)
    quality = pd.DataFrame([check.__dict__ for check in all_checks])

    processed_dir.mkdir(parents=True, exist_ok=True)
    quality.to_csv(processed_dir / "phemut_temporal_alignment_quality.csv", index=False)
    plot_sample.to_csv(processed_dir / "phemut_temporal_alignment_plot_sample.csv", index=False)
    aggregate_sample.to_csv(processed_dir / "phemut_temporal_alignment_aggregate_sample.csv", index=False)
    feature_audit.to_csv(processed_dir / "phemut_temporal_alignment_feature_leakage_audit.csv", index=False)
    lag_audit.to_csv(processed_dir / "phemut_temporal_alignment_lag_audit.csv", index=False)
    write_report(processed_dir / "phemut_temporal_alignment_report.md", quality, plot_sample, aggregate_sample, lag_audit)

    print("PheMuT temporal alignment audit complete.")
    print("\nQuality checks:")
    print(quality.to_string(index=False))
    print("\nPlot-level human alignment sample:")
    sample_cols = ["season", "plot_id", "observation_date", "target_date", "yield_g", "next_yield_g", "delta_next_yield_g", "target_yield_diff_g"]
    print(plot_sample[[col for col in sample_cols if col in plot_sample.columns]].head(20).to_string(index=False))
    print("\nAggregate human alignment sample:")
    agg_cols = ["season", "observation_date", "target_date", "aggregate_yield_g", "aggregate_next_yield_g", "aggregate_delta_next_yield_g", "target_aggregate_diff_g"]
    print(aggregate_sample[[col for col in agg_cols if col in aggregate_sample.columns]].head(20).to_string(index=False))

    if not lag_audit.empty:
        print("\nLag/timeliness warnings preview:")
        lag_cols = ["task_label", "model_label", "same_window_corr", "forecast_vs_current_window_corr", "same_window_mae_g", "forecast_vs_current_window_mae_g", "direction_accuracy_vs_current", "lag_warning", "lag_note"]
        print(lag_audit[[col for col in lag_cols if col in lag_audit.columns]].head(12).to_string(index=False))

    print("\nOutputs written to:")
    for filename in [
        "phemut_temporal_alignment_quality.csv",
        "phemut_temporal_alignment_plot_sample.csv",
        "phemut_temporal_alignment_aggregate_sample.csv",
        "phemut_temporal_alignment_feature_leakage_audit.csv",
        "phemut_temporal_alignment_lag_audit.csv",
        "phemut_temporal_alignment_report.md",
    ]:
        print(f"- {processed_dir / filename}")


if __name__ == "__main__":
    main()

