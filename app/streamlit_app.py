"""Presentation-focused Streamlit dashboard for the independent PheMuT-inspired case study.

Reads processed outputs produced by the independent pipeline under data/processed/phemut.
No original PheMuT research code is required by this app.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from textwrap import dedent
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DEFAULT_PROCESSED_DIR = Path("data/processed/phemut")
PRIMARY_AGGREGATE_SCOPE = "held_out_cv_fold_aggregate_primary"
ILLUSTRATIVE_SCOPE = "full_field_reconstructed_oof_curve_illustrative_not_for_selection"
BASELINE_PATTERN = "baseline|Zero yield|Training horizon mean|Observed 5-week mean|Last observed"


st.set_page_config(
    page_title="Strawberry Yield Forecasting Audit",
    page_icon="🍓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------

st.markdown(
    dedent(
        """
        <style>
        :root {
            --bg: #070b10;
            --panel: #101722;
            --panel-2: #111827;
            --line: rgba(148, 163, 184, 0.24);
            --line-strong: rgba(148, 163, 184, 0.36);
            --text: #f8fafc;
            --muted: #b8c2cc;
            --soft: #93a4b7;
            --accent: #7cc7ff;
            --green: #55d27a;
            --yellow: #f7c948;
            --red: #ff5c77;
            --pink: #ff6f91;
            --purple: #b794f4;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 0%, rgba(255, 92, 119, 0.13), transparent 26rem),
                radial-gradient(circle at 100% 8%, rgba(124, 199, 255, 0.10), transparent 24rem),
                radial-gradient(circle at 60% 100%, rgba(85, 210, 122, 0.07), transparent 24rem),
                var(--bg);
            color: var(--text);
        }

        .block-container {
            max-width: min(1480px, calc(100vw - 1rem));
            padding: 3.60rem 1.05rem 2.0rem 1.05rem !important;
        }
        [data-testid="stHeader"] {
            background: rgba(7, 11, 16, 0.96);
            backdrop-filter: blur(8px);
        }
        div[data-testid="stTabs"] {
            margin-top: .20rem;
        }

        h1, h2, h3, h4, h5, h6, p, li, label, div, span {
            color: var(--text);
        }
        h1, h2, h3 { letter-spacing: -0.035em; }
        h1 { font-size: 2.25rem !important; line-height: 1.08 !important; margin-bottom: .45rem !important; }
        h2 { margin-top: .25rem !important; font-size: 2.55rem !important; line-height: 1.10 !important; margin-bottom: .55rem !important; }
        h3 { font-size: 1.18rem !important; }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.99), rgba(7, 11, 16, 0.98));
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] * { color: var(--text) !important; }
        .side-brand {
            padding: 1.00rem 1.00rem 1.05rem 1.00rem;
            margin: .40rem 0 .95rem 0;
            border: 1px solid rgba(124,199,255,.30);
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(255,92,119,.24), rgba(17,24,39,.96) 55%, rgba(85,210,122,.16));
            box-shadow: 0 18px 45px rgba(0,0,0,.32);
        }
        .side-title {
            font-size: 1.50rem;
            font-weight: 950;
            line-height: 1.03;
            letter-spacing: -0.055em;
            color: #fff;
            margin-top: .50rem;
        }
        .side-kicker {
            color: var(--green);
            font-size: .62rem;
            font-weight: 900;
            letter-spacing: .13em;
            text-transform: uppercase;
        }
        .side-subtitle {
            color: var(--muted);
            font-size: .78rem;
            line-height: 1.36;
            margin-top: .55rem;
        }
        .side-profile {
            padding: .92rem .95rem;
            margin: .80rem 0 .95rem 0;
            border: 1px solid rgba(148,163,184,.24);
            border-radius: 16px;
            background: rgba(17,24,39,.58);
        }
        .side-profile-name {
            color: #fff;
            font-size: 1.02rem;
            font-weight: 850;
            margin-bottom: .18rem;
        }
        .side-profile-role {
            color: var(--muted);
            font-size: .76rem;
            line-height: 1.35;
            margin-bottom: .55rem;
        }
        .side-link {
            display: block;
            color: #dff4ff !important;
            text-decoration: none;
            font-size: .78rem;
            font-weight: 750;
            margin-top: .28rem;
        }
        .side-stack {
            color: var(--soft);
            font-size: .72rem;
            line-height: 1.35;
            margin-top: .55rem;
        }

        .hero-card {
            padding: 1.18rem 1.38rem;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(255, 92, 119, 0.20), rgba(17, 24, 39, 0.96) 48%, rgba(85, 210, 122, 0.10));
            border: 1px solid var(--line-strong);
            box-shadow: 0 22px 60px rgba(0,0,0,.30);
            margin-bottom: 1.00rem;
        }
        .eyebrow {
            color: var(--green);
            font-size: .78rem;
            font-weight: 850;
            letter-spacing: .14em;
            text-transform: uppercase;
            margin-bottom: .46rem;
        }
        .hero-title {
            color: #fff;
            font-size: 2.05rem;
            font-weight: 900;
            line-height: 1.05;
            margin-bottom: .62rem;
        }
        .hero-subtitle {
            color: var(--muted);
            font-size: .96rem;
            line-height: 1.48;
            max-width: 1280px;
        }

        .metric-card, .note-card, .good-card, .warn-card, .risk-card, .mini-card {
            background: rgba(17, 24, 39, .92);
            border: 1px solid var(--line);
            border-radius: 17px;
            padding: .95rem 1.00rem;
            box-shadow: 0 14px 34px rgba(0,0,0,.20);
        }
        .metric-card {
            min-height: 106px;
            height: 106px;
            margin-bottom: 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-sizing: border-box;
        }
        .card-grid {
            display: grid;
            gap: .78rem;
            align-items: stretch;
            margin: .75rem 0 1.05rem 0;
        }
        .card-grid.cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
        .card-grid.cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
        .card-grid.cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .card-grid .mini-card { height: 100%; margin-bottom: 0; box-sizing: border-box; }
        @media (max-width: 1100px) {
            .card-grid.cols-4 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .card-grid.cols-3 { grid-template-columns: repeat(1, minmax(0, 1fr)); }
        }
        .metric-label {
            color: var(--soft);
            font-size: .62rem;
            font-weight: 850;
            letter-spacing: .10em;
            text-transform: uppercase;
            margin-bottom: .44rem;
        }
        .metric-value {
            color: #fff;
            font-size: 1.34rem;
            font-weight: 900;
            line-height: 1.02;
            margin-bottom: .35rem;
            word-break: normal;
        }
        .metric-note, .card-body, .mini-body {
            color: var(--muted);
            font-size: .78rem;
            line-height: 1.32;
        }
        .card-title, .mini-title {
            color: #fff;
            font-weight: 850;
            margin-bottom: .35rem;
        }
        .mini-card { min-height: 102px; margin-bottom: .55rem; }
        .good-card, .warn-card, .note-card, .risk-card {
            margin: .70rem 0 .95rem 0;
            line-height: 1.52;
        }
        .good-card { background: rgba(85, 210, 122, .12); border-left: 6px solid var(--green); }
        .warn-card { background: rgba(247, 201, 72, .13); border-left: 6px solid var(--yellow); }
        .note-card { background: rgba(124, 199, 255, .12); border-left: 6px solid var(--accent); }
        .risk-card { background: rgba(255, 92, 119, .13); border-left: 6px solid var(--red); }

        .pill {
            display: inline-block;
            border-radius: 999px;
            padding: .18rem .58rem;
            border: 1px solid rgba(124,199,255,.35);
            color: #dff4ff;
            background: rgba(124,199,255,.12);
            font-size: .82rem;
            font-weight: 750;
            margin-right: .35rem;
            margin-bottom: .35rem;
        }
        .pill-green {
            border-color: rgba(85,210,122,.35);
            color: #defce8;
            background: rgba(85,210,122,.13);
        }
        .pill-yellow {
            border-color: rgba(247,201,72,.35);
            color: #fff4bf;
            background: rgba(247,201,72,.13);
        }
        .section-kicker {
            color: var(--green);
            font-size: .86rem;
            font-weight: 900;
            letter-spacing: .12em;
            text-transform: uppercase;
            margin: .35rem 0 .15rem 0;
        }
        .chart-caption {
            color: #fff;
            font-weight: 850;
            margin: 1.05rem 0 .55rem 0;
            line-height: 1.25;
        }
        .chart-subtitle {
            color: var(--muted);
            font-size: .86rem;
            margin: -.35rem 0 .65rem 0;
            line-height: 1.42;
        }
        .section-intro {
            color: var(--muted);
            font-size: .95rem;
            line-height: 1.42;
            max-width: 980px;
            margin-bottom: .65rem;
        }
        .small-help {
            color: var(--muted);
            font-size: .84rem;
            line-height: 1.4;
            margin-top: .25rem;
        }
        .muted { color: var(--muted); }
        .thin-rule { border-top: 1px solid var(--line); margin: 1.05rem 0 1.20rem 0; }

        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(148,163,184,.18);
            border-radius: 13px;
            overflow: hidden;
            background: rgba(17,24,39,.55);
        }
        div[data-testid="stExpander"] {
            background: rgba(17, 24, 39, .48);
            border: 1px solid rgba(148,163,184,.22);
            border-radius: 13px;
        }


        div[data-testid="stPlotlyChart"] {
            background: rgba(7, 11, 16, .12);
            border-radius: 16px;
        }
        hr { border: none; border-top: 1px solid var(--line); margin: 1.15rem 0; }

        @media (max-width: 900px) {
            .block-container { padding-top: 3.75rem !important; padding-left: .70rem; padding-right: .70rem; }
            .hero-title { font-size: 1.78rem; }
            .metric-value { font-size: 1.35rem; }
            .stTabs [data-baseweb="tab"] { padding: .55rem .62rem; }
        }
        </style>
        """
    ),
    unsafe_allow_html=True,
)


PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#f8fafc", size=12),
    margin=dict(l=18, r=18, t=30, b=28),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, font=dict(size=10)),
)


# -----------------------------------------------------------------------------
# Small UI helpers
# -----------------------------------------------------------------------------


def html_block(html: str) -> None:
    st.markdown(dedent(html).strip(), unsafe_allow_html=True)


def metric_card(label: str, value: str, note: str = "") -> None:
    html_block(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(str(label))}</div>
            <div class="metric-value">{escape(str(value))}</div>
            <div class="metric-note">{escape(str(note))}</div>
        </div>
        """
    )


def metric_card_markup(label: str, value: str, note: str = "") -> str:
    # Keep HTML fully left-aligned/compact. Indented HTML inside Markdown can
    # render as a code block in Streamlit instead of real HTML.
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{escape(str(label))}</div>'
        f'<div class="metric-value">{escape(str(value))}</div>'
        f'<div class="metric-note">{escape(str(note))}</div>'
        '</div>'
    )


def mini_card_markup(title: str, body: str) -> str:
    return (
        '<div class="mini-card">'
        f'<div class="mini-title">{escape(str(title))}</div>'
        f'<div class="mini-body">{body}</div>'
        '</div>'
    )


def card_grid(markup_items: list[str], columns: int = 4) -> None:
    columns = max(1, min(columns, 4))
    body = "".join(markup_items)
    html_block(f'<div class="card-grid cols-{columns}">{body}</div>')


def message_card(title: str, body: str, tone: str = "note") -> None:
    css = {"good": "good-card", "warn": "warn-card", "risk": "risk-card", "note": "note-card"}.get(tone, "note-card")
    html_block(
        f"""
        <div class="{css}">
            <div class="card-title">{escape(str(title))}</div>
            <div class="card-body">{body}</div>
        </div>
        """
    )


def mini_card(title: str, body: str) -> None:
    html_block(
        f"""
        <div class="mini-card">
            <div class="mini-title">{escape(str(title))}</div>
            <div class="mini-body">{body}</div>
        </div>
        """
    )


def pills(values: Iterable[object], tone: str = "blue") -> None:
    css = "pill-green" if tone == "green" else "pill-yellow" if tone == "yellow" else ""
    body = "".join(f'<span class="pill {css}">{escape(str(v))}</span>' for v in values if pd.notna(v))
    html_block(body or '<span class="muted">No values available.</span>')


def chart_caption(text: str, subtitle: str | None = None) -> None:
    html_block(f'<div class="chart-caption">{escape(str(text))}</div>')
    if subtitle:
        html_block(f'<div class="chart-subtitle">{escape(str(subtitle))}</div>')


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------


def fmt_float(value: object, digits: int = 2) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):,.{digits}f}"
    except Exception:
        return "—"


def fmt_int(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{int(round(float(value))):,}"
    except Exception:
        return "—"


def fmt_pct(value: object, digits: int = 1) -> str:
    formatted = fmt_float(value, digits)
    return "—" if formatted == "—" else f"{formatted}%"


def safe_unique(values: Iterable[object]) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        if value is None or pd.isna(value):
            continue
        cleaned.append(str(value))
    return sorted(set(cleaned))

def season_label(value: object) -> str:
    text = str(value)
    mapping = {
        "2324": "2023–24",
        "2425": "2024–25",
        "2023-24": "2023–24",
        "2024-25": "2024–25",
    }
    return mapping.get(text, text)


def season_control(label: str, seasons: list[str], key: str, include_all: bool = False) -> str | None:
    if not seasons:
        return "All" if include_all else None
    options = (["All"] if include_all else []) + seasons
    display = {option: ("All" if option == "All" else season_label(option)) for option in options}
    selected = st.segmented_control(
        label,
        options,
        default=options[0],
        key=key,
        format_func=lambda option: display.get(option, str(option)),
    )
    return selected


def display_season_value(value: object) -> str:
    return season_label(value) if pd.notna(value) else "—"


def is_baseline_label(label: object) -> bool:
    return bool(pd.Series([str(label)]).str.contains(BASELINE_PATTERN, case=False, regex=True, na=False).iloc[0])


@st.cache_data(show_spinner=False)
def load_csv(processed_dir: str, filename: str) -> pd.DataFrame:
    path = Path(processed_dir) / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - UI defensive path
        st.warning(f"Could not read {filename}: {exc}")
        return pd.DataFrame()

    for col in ["season", "plot_id", "cultivar", "model_label", "feature_set", "validation_scope"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "season" in df.columns:
        df["season"] = df["season"].map(season_label)
    return df


def load_artifacts(processed_dir: Path) -> dict[str, pd.DataFrame]:
    processed_dir_str = str(processed_dir)
    return {
        "tidy_summary": load_csv(processed_dir_str, "phemut_tidy_summary.csv"),
        "tidy_quality": load_csv(processed_dir_str, "phemut_data_quality_report.csv"),
        "sequence_quality": load_csv(processed_dir_str, "phemut_sequence_quality_report.csv"),
        "target_summary": load_csv(processed_dir_str, "phemut_sequence_target_summary.csv"),
        "feature_sets": load_csv(processed_dir_str, "phemut_sequence_feature_sets.csv"),
        "sequence_dataset": load_csv(processed_dir_str, "phemut_sequence_dataset.csv"),
        "model_metrics": load_csv(processed_dir_str, "phemut_sequence_model_metrics.csv"),
        "model_predictions": load_csv(processed_dir_str, "phemut_sequence_model_predictions.csv"),
        "aggregate_metrics": load_csv(processed_dir_str, "phemut_sequence_aggregate_metrics.csv"),
        "aggregate_fold_metrics": load_csv(processed_dir_str, "phemut_sequence_aggregate_fold_metrics.csv"),
        "aggregate_fold_board": load_csv(processed_dir_str, "phemut_sequence_aggregate_fold_board.csv"),
        "aggregate_reconstructed_board": load_csv(processed_dir_str, "phemut_sequence_aggregate_reconstructed_board.csv"),
        "aggregate_reconstructed_metrics": load_csv(processed_dir_str, "phemut_sequence_aggregate_reconstructed_metrics.csv"),
        "timeliness": load_csv(processed_dir_str, "phemut_sequence_timeliness_audit.csv"),
        "selection": load_csv(processed_dir_str, "phemut_sequence_model_selection_summary.csv"),
        "grouped_importance": load_csv(processed_dir_str, "phemut_sequence_grouped_feature_importance.csv"),
        "weather_driver_summary": load_csv(processed_dir_str, "phemut_sequence_weather_driver_summary.csv"),
        "prediction_intervals": load_csv(processed_dir_str, "phemut_sequence_prediction_intervals.csv"),
        "uncertainty_summary": load_csv(processed_dir_str, "phemut_sequence_uncertainty_summary.csv"),
        "stakeholder_summary": load_csv(processed_dir_str, "phemut_sequence_stakeholder_summary.csv"),
    }


def available_seasons(artifacts: dict[str, pd.DataFrame]) -> list[str]:
    seasons: list[str] = []
    for key in ["model_metrics", "aggregate_metrics", "target_summary", "sequence_dataset"]:
        df = artifacts.get(key, pd.DataFrame())
        if not df.empty and "season" in df.columns:
            seasons.extend(safe_unique(df["season"]))
    return sorted(set(seasons))


def metric_rows_for_overall(model_metrics: pd.DataFrame) -> pd.DataFrame:
    if model_metrics.empty:
        return model_metrics
    df = model_metrics.copy()
    if "n_target_horizons" in df.columns:
        overall = df[pd.to_numeric(df["n_target_horizons"], errors="coerce").notna()].copy()
        if not overall.empty:
            return overall
    if "target_horizon_index" in df.columns:
        numeric_cols = [
            "mae",
            "rmse",
            "r2",
            "correlation",
            "bias_predicted_minus_observed_g",
            "mae_improvement_pct_vs_last_observed_baseline",
        ]
        group_cols = [col for col in ["season", "feature_set", "model_label", "n_model_features"] if col in df.columns]
        available_numeric = [col for col in numeric_cols if col in df.columns]
        if group_cols and available_numeric:
            return df.groupby(group_cols, dropna=False)[available_numeric].mean().reset_index()
    return df


def primary_aggregate_metrics(aggregate_metrics: pd.DataFrame) -> pd.DataFrame:
    if aggregate_metrics.empty:
        return aggregate_metrics
    df = aggregate_metrics.copy()
    if "validation_scope" in df.columns:
        primary = df[df["validation_scope"].astype(str).eq(PRIMARY_AGGREGATE_SCOPE)].copy()
        if not primary.empty:
            return primary
    return df


def best_rows_by_season(metrics: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    if metrics.empty or metric_col not in metrics.columns or "season" not in metrics.columns:
        return pd.DataFrame()
    df = primary_aggregate_metrics(metrics) if metric_col == "aggregate_mae_g" else metric_rows_for_overall(metrics)
    if df.empty:
        return pd.DataFrame()
    rows = []
    for _, sub in df.dropna(subset=[metric_col]).groupby("season", dropna=False):
        rows.append(sub.sort_values(metric_col, ascending=True).iloc[0])
    return pd.DataFrame(rows)


def selected_best_model(metrics: pd.DataFrame, season: str | None, metric_col: str = "mae") -> str | None:
    if metrics.empty or "model_label" not in metrics.columns:
        return None
    df = primary_aggregate_metrics(metrics) if metric_col == "aggregate_mae_g" else metric_rows_for_overall(metrics)
    if season is not None and "season" in df.columns:
        df = df[df["season"].astype(str).eq(str(season))]
    if metric_col not in df.columns or df.empty:
        return None
    learned = df[~df["model_label"].apply(is_baseline_label)].copy()
    if not learned.empty:
        return str(learned.sort_values(metric_col).iloc[0]["model_label"])
    return str(df.sort_values(metric_col).iloc[0]["model_label"])


def filtered_season(df: pd.DataFrame, season: str | None) -> pd.DataFrame:
    if df.empty or season is None or "season" not in df.columns:
        return df.copy()
    return df[df["season"].astype(str).eq(str(season))].copy()


def compact_columns(df: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
    cols = [c for c in preferred if c in df.columns]
    if not cols:
        return df
    return df[cols]


def clean_feature_group_label(value: object) -> str:
    text = str(value)
    mapping = {
        "Cultivar / plot context": "Cultivar/context",
        "Other crop/context": "Other context",
    }
    return mapping.get(text, text)


def clean_driver_summary_text(value: object) -> str:
    text = str(value)
    if not text or text == "nan":
        return "Phenology counts and canopy structure"
    parts = [part.strip() for part in text.split(";") if part.strip()]
    cleaned: list[str] = []
    for part in parts:
        lowered = part.lower()
        if "cultivar" in lowered or "plot context" in lowered or "context" in lowered:
            continue
        if "weather" in lowered and ("0.0%" in lowered or "0%" in lowered):
            continue
        cleaned.append(part.replace("Cultivar / plot context", "Cultivar/context"))
    return "; ".join(cleaned) if cleaned else "Phenology counts and canopy structure"


def has_positive_weather_signal(weather_driver_summary: pd.DataFrame, season: str | None, model_label: str | None) -> bool:
    required = {"season", "model_label", "weather_importance_share_pct"}
    if weather_driver_summary.empty or not required.issubset(weather_driver_summary.columns):
        return False
    df = filtered_season(weather_driver_summary, season)
    if model_label:
        df = df[df["model_label"].astype(str).eq(str(model_label))]
    values = pd.to_numeric(df.get("weather_importance_share_pct"), errors="coerce")
    return bool(values.fillna(0).gt(0.05).any())


# -----------------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------------


def apply_layout(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(title=dict(text=""))
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    if height is not None:
        fig.update_layout(height=height)
    return fig


def plot_target_summary(target_summary: pd.DataFrame) -> go.Figure | None:
    required = {"season", "target_horizon_index", "total_target_yield_g"}
    if target_summary.empty or not required.issubset(target_summary.columns):
        return None
    df = target_summary.copy()
    df["season"] = df["season"].astype(str)
    df["season_display"] = df["season"].map(season_label)
    df["horizon"] = "H" + pd.to_numeric(df["target_horizon_index"], errors="coerce").astype("Int64").astype(str)
    fig = px.bar(
        df,
        x="horizon",
        y="total_target_yield_g",
        color="season_display",
        barmode="group",
        labels={"horizon": "Future horizon", "total_target_yield_g": "Total target yield (g)", "season_display": "Season"},
        text_auto=".2s",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    return apply_layout(fig, height=315)


def plot_model_skill(metrics: pd.DataFrame, season: str | None) -> go.Figure | None:
    if metrics.empty or "model_label" not in metrics.columns or "mae" not in metrics.columns:
        return None
    df = filtered_season(metric_rows_for_overall(metrics), season)
    if df.empty:
        return None
    df = df.sort_values("mae", ascending=False).tail(12).copy()
    df["model_type"] = np.where(df["model_label"].apply(is_baseline_label), "Baseline", "Learned model")
    fig = px.bar(
        df,
        x="mae",
        y="model_label",
        color="model_type",
        orientation="h",
        hover_data=[c for c in ["rmse", "r2", "correlation", "mae_improvement_pct_vs_last_observed_baseline"] if c in df.columns],
        labels={"mae": "MAE (g per plot-horizon)", "model_label": "Model", "model_type": "Type"},
        text="mae",
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
    fig.update_yaxes(categoryorder="total ascending")
    return apply_layout(fig, height=300)


def plot_model_improvement(metrics: pd.DataFrame) -> go.Figure | None:
    if metrics.empty or "mae_improvement_pct_vs_last_observed_baseline" not in metrics.columns:
        return None
    df = metric_rows_for_overall(metrics).copy()
    df = df[~df["model_label"].apply(is_baseline_label)]
    if df.empty:
        return None
    df["season_display"] = df["season"].map(season_label) if "season" in df.columns else "Season"
    df = df.sort_values(["season", "mae_improvement_pct_vs_last_observed_baseline"], ascending=[True, False])
    fig = px.bar(
        df,
        x="model_label",
        y="mae_improvement_pct_vs_last_observed_baseline",
        color="season_display",
        barmode="group",
        labels={"model_label": "Model", "mae_improvement_pct_vs_last_observed_baseline": "MAE improvement vs last-observed baseline (%)", "season_display": "Season"},
        text="mae_improvement_pct_vs_last_observed_baseline",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_tickangle=-18)
    return apply_layout(fig, height=315)


def plot_aggregate_metric(aggregate_metrics: pd.DataFrame, season: str | None) -> go.Figure | None:
    if aggregate_metrics.empty or "model_label" not in aggregate_metrics.columns or "aggregate_mae_g" not in aggregate_metrics.columns:
        return None
    df = filtered_season(primary_aggregate_metrics(aggregate_metrics), season)
    if df.empty:
        return None
    df = df.sort_values("aggregate_mae_g", ascending=False).tail(12).copy()
    df["model_type"] = np.where(df["model_label"].apply(is_baseline_label), "Baseline", "Learned model")
    fig = px.bar(
        df,
        x="aggregate_mae_g",
        y="model_label",
        color="model_type",
        orientation="h",
        hover_data=[c for c in ["aggregate_rmse_g", "aggregate_r2", "aggregate_correlation", "aggregate_mae_improvement_pct_vs_last_observed_baseline"] if c in df.columns],
        labels={"aggregate_mae_g": "Held-out aggregate MAE (g)", "model_label": "Model", "model_type": "Type"},
        text="aggregate_mae_g",
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
    fig.update_yaxes(categoryorder="total ascending")
    return apply_layout(fig, height=300)


def plot_predictions_scatter(predictions: pd.DataFrame, season: str, model_label: str) -> go.Figure | None:
    required = {"season", "model_label", "target_yield_g", "prediction_g"}
    if predictions.empty or not required.issubset(predictions.columns):
        return None
    df = predictions[(predictions["season"].astype(str) == str(season)) & (predictions["model_label"].astype(str) == str(model_label))].copy()
    if df.empty:
        return None
    color_col = "target_horizon_index" if "target_horizon_index" in df.columns else None
    fig = px.scatter(
        df,
        x="target_yield_g",
        y="prediction_g",
        color=color_col,
        hover_data=[c for c in ["plot_id", "cultivar", "target_date", "fold_id"] if c in df.columns],
        labels={"target_yield_g": "Observed yield (g)", "prediction_g": "Predicted yield (g)", "target_horizon_index": "Horizon"},
        opacity=0.78,
    )
    max_value = np.nanmax([df["target_yield_g"].max(), df["prediction_g"].max()])
    min_value = np.nanmin([df["target_yield_g"].min(), df["prediction_g"].min()])
    if np.isfinite(max_value) and np.isfinite(min_value):
        fig.add_trace(
            go.Scatter(
                x=[min_value, max_value],
                y=[min_value, max_value],
                mode="lines",
                name="1:1 reference",
                line=dict(dash="dot", width=1),
            )
        )
    return apply_layout(fig, height=300)


def plot_aggregate_curve(board: pd.DataFrame, season: str, model_label: str, fold_id: int | None) -> go.Figure | None:
    required = {"season", "model_label", "target_horizon_index", "observed_total_yield_g", "predicted_total_yield_g"}
    if board.empty or not required.issubset(board.columns):
        return None
    df = board[(board["season"].astype(str) == str(season)) & (board["model_label"].astype(str) == str(model_label))].copy()
    if fold_id is not None and "fold_id" in df.columns:
        df = df[pd.to_numeric(df["fold_id"], errors="coerce").eq(float(fold_id))]
    if df.empty:
        return None
    df = df.sort_values("target_horizon_index")
    x_label = "target_date" if "target_date" in df.columns else "target_horizon_index"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_label], y=df["observed_total_yield_g"], mode="lines+markers", name="Observed held-out total", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=df[x_label], y=df["predicted_total_yield_g"], mode="lines+markers", name="Predicted held-out total", line=dict(width=3)))
    if "last_observed_total_yield_baseline_g" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[x_label],
                y=df["last_observed_total_yield_baseline_g"],
                mode="lines+markers",
                name="Last-observed baseline",
                line=dict(dash="dash", width=2),
            )
        )
    fig.update_layout(
        xaxis_title="Target date" if x_label == "target_date" else "Future horizon",
        yaxis_title="Yield (g)",
    )
    return apply_layout(fig, height=340)


def plot_fold_errors(board: pd.DataFrame, season: str, model_label: str) -> go.Figure | None:
    required = {"season", "model_label", "fold_id", "target_horizon_index", "aggregate_abs_error_g"}
    if board.empty or not required.issubset(board.columns):
        return None
    df = board[(board["season"].astype(str) == str(season)) & (board["model_label"].astype(str) == str(model_label))].copy()
    if df.empty:
        return None
    df["horizon"] = "H" + pd.to_numeric(df["target_horizon_index"], errors="coerce").astype("Int64").astype(str)
    fig = px.box(
        df,
        x="horizon",
        y="aggregate_abs_error_g",
        points="all",
        labels={"horizon": "Future horizon", "aggregate_abs_error_g": "Absolute aggregate error (g)"},
    )
    return apply_layout(fig, height=300)


def plot_lag_scatter(timeliness: pd.DataFrame, season: str | None) -> go.Figure | None:
    required = {"same_horizon_mae_g", "prediction_vs_previous_horizon_mae_g", "model_label"}
    if timeliness.empty or not required.issubset(timeliness.columns):
        return None
    df = filtered_season(timeliness, season)
    if df.empty:
        return None
    df["model_type"] = np.where(df["model_label"].apply(is_baseline_label), "Baseline", "Learned model")
    fig = px.scatter(
        df,
        x="same_horizon_mae_g",
        y="prediction_vs_previous_horizon_mae_g",
        color="model_type",
        symbol="lag_warning" if "lag_warning" in df.columns else None,
        hover_data=[c for c in ["season", "model_label", "fold_id", "lag_note"] if c in df.columns],
        labels={
            "same_horizon_mae_g": "MAE vs target horizon",
            "prediction_vs_previous_horizon_mae_g": "MAE vs previous horizon",
            "model_type": "Type",
        },

    )
    max_value = np.nanmax([df["same_horizon_mae_g"].max(), df["prediction_vs_previous_horizon_mae_g"].max()])
    if np.isfinite(max_value):
        fig.add_trace(go.Scatter(x=[0, max_value], y=[0, max_value], mode="lines", name="Equal error", line=dict(dash="dot", width=1)))
    return apply_layout(fig, height=340)


def plot_lag_warning_summary(timeliness: pd.DataFrame, season: str | None) -> go.Figure | None:
    if timeliness.empty or "model_label" not in timeliness.columns or "lag_warning" not in timeliness.columns:
        return None
    df = filtered_season(timeliness, season)
    if df.empty:
        return None
    df = df.copy()
    df["model_type"] = np.where(df["model_label"].apply(is_baseline_label), "Reactive baseline", "Learned model")
    df["lag_warning"] = df["lag_warning"].fillna(False).astype(bool)
    summary = (
        df.groupby(["season", "model_type", "lag_warning"], dropna=False)
        .size()
        .reset_index(name="n_rows")
    )
    summary["status"] = np.where(summary["lag_warning"], "Lag warning", "No warning")
    facet = "season" if season in (None, "All") and summary["season"].nunique() > 1 else None
    fig = px.bar(
        summary,
        x="model_type",
        y="n_rows",
        color="status",
        barmode="group",
        facet_col=facet,
        labels={"model_type": "Model group", "n_rows": "Audited model-fold rows", "status": "Status"},
        text="n_rows",
    )
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside", cliponaxis=False)
    fig.for_each_annotation(lambda ann: ann.update(text=ann.text.replace("season=", "Season ")))
    return apply_layout(fig, height=300)





def plot_grouped_importance(grouped_importance: pd.DataFrame, season: str | None, model_label: str | None) -> go.Figure | None:
    required = {"season", "model_label", "feature_group", "importance_share_pct"}
    if grouped_importance.empty or not required.issubset(grouped_importance.columns):
        return None
    df = filtered_season(grouped_importance, season)
    if model_label:
        df = df[df["model_label"].astype(str).eq(str(model_label))]
    if df.empty:
        return None
    df["importance_share_pct"] = pd.to_numeric(df["importance_share_pct"], errors="coerce").fillna(0)
    weather_zero = df["feature_group"].astype(str).str.contains("Weather", case=False, na=False) & df["importance_share_pct"].le(0.05)
    df = df[~weather_zero].copy()
    if df.empty:
        return None
    df = df.sort_values("importance_share_pct", ascending=True)
    df["feature_group_display"] = df["feature_group"].apply(clean_feature_group_label)
    fig = px.bar(
        df,
        x="importance_share_pct",
        y="feature_group_display",
        orientation="h",
        text="importance_share_pct",
        labels={"importance_share_pct": "Mean importance share (%)", "feature_group_display": "Feature group"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    return apply_layout(fig, height=310)


def plot_weather_driver_summary(weather_driver_summary: pd.DataFrame, season: str | None, model_label: str | None) -> go.Figure | None:
    required = {"season", "model_label", "weather_driver", "weather_importance_share_pct"}
    if weather_driver_summary.empty or not required.issubset(weather_driver_summary.columns):
        return None
    df = filtered_season(weather_driver_summary, season)
    if model_label:
        df = df[df["model_label"].astype(str).eq(str(model_label))]
    if df.empty:
        return None
    df["weather_importance_share_pct"] = pd.to_numeric(df["weather_importance_share_pct"], errors="coerce").fillna(0)
    if not df["weather_importance_share_pct"].gt(0.05).any():
        return None
    df = df.sort_values("weather_importance_share_pct", ascending=True)
    fig = px.bar(
        df,
        x="weather_importance_share_pct",
        y="weather_driver",
        orientation="h",
        text="weather_importance_share_pct",
        labels={"weather_importance_share_pct": "Share of weather importance (%)", "weather_driver": "Weather driver"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    return apply_layout(fig, height=300)


def plot_uncertainty_summary(uncertainty_summary: pd.DataFrame, season: str | None) -> go.Figure | None:
    required = {"season", "model_label", "interval_coverage_pct", "mean_interval_width_g"}
    if uncertainty_summary.empty or not required.issubset(uncertainty_summary.columns):
        return None
    df = filtered_season(uncertainty_summary, season)
    df = df[~df["model_label"].apply(is_baseline_label)].copy()
    if df.empty:
        return None
    df = df.sort_values("mae" if "mae" in df.columns else "mean_interval_width_g").head(8)
    fig = px.scatter(
        df,
        x="mean_interval_width_g",
        y="interval_coverage_pct",
        size="n_predictions" if "n_predictions" in df.columns else None,
        hover_name="model_label",
        labels={"mean_interval_width_g": "Mean 80% interval width (g)", "interval_coverage_pct": "Retrospective interval coverage (%)"},
    )
    fig.add_hline(y=80, line_dash="dot", annotation_text="Nominal 80%")
    return apply_layout(fig, height=300)

# -----------------------------------------------------------------------------
# Page pieces
# NOTE: title/branding lives in the sidebar; no top hero/banner is rendered.

# -----------------------------------------------------------------------------


def render_header(artifacts: dict[str, pd.DataFrame], processed_dir: Path) -> None:
    missing_core = [
        filename
        for key, filename in [
            ("sequence_quality", "phemut_sequence_quality_report.csv"),
            ("model_metrics", "phemut_sequence_model_metrics.csv"),
            ("aggregate_metrics", "phemut_sequence_aggregate_metrics.csv"),
        ]
        if artifacts[key].empty
    ]
    with st.expander("Pipeline status", expanded=bool(missing_core)):
        st.caption(f"Processed outputs: `{processed_dir}`")
        if missing_core:
            st.warning("Missing required pipeline outputs: " + ", ".join(missing_core))
        else:
            st.success("Core sequence/model outputs found.")


def render_metric_strip(artifacts: dict[str, pd.DataFrame]) -> None:
    best_plot = best_rows_by_season(artifacts["model_metrics"], "mae")
    best_agg = best_rows_by_season(artifacts["aggregate_metrics"], "aggregate_mae_g")

    cards: list[str] = []
    if not best_plot.empty:
        row = best_plot.sort_values("mae").iloc[0]
        cards.append(metric_card_markup("Best plot-level MAE", fmt_float(row.get("mae")), f"{row.get('model_label', '—')} · {display_season_value(row.get('season', '—'))}"))
        cards.append(metric_card_markup("Plot-level improvement", fmt_pct(row.get("mae_improvement_pct_vs_last_observed_baseline")), "vs last-observed yield baseline"))
    else:
        cards.append(metric_card_markup("Best plot-level MAE", "—", "Run sequence model benchmark"))
        cards.append(metric_card_markup("Plot-level improvement", "—", "Run sequence model benchmark"))

    if not best_agg.empty:
        row = best_agg.sort_values("aggregate_mae_g").iloc[0]
        cards.append(metric_card_markup("Best held-out aggregate MAE", fmt_float(row.get("aggregate_mae_g")), f"{row.get('model_label', '—')} · {display_season_value(row.get('season', '—'))}"))
        cards.append(metric_card_markup("Aggregate improvement", fmt_pct(row.get("aggregate_mae_improvement_pct_vs_last_observed_baseline")), "held-out fold aggregate"))
    else:
        cards.append(metric_card_markup("Best held-out aggregate MAE", "—", "Run corrected aggregate evaluation"))
        cards.append(metric_card_markup("Aggregate improvement", "—", "Run corrected aggregate evaluation"))

    card_grid(cards, columns=4)


def render_overview(artifacts: dict[str, pd.DataFrame]) -> None:
    html_block('<div class="section-kicker">Overview</div>')
    st.header("Forecasting validation at a glance")
    st.markdown(
        '<div class="section-intro">Each plot is treated as a short weekly sequence: first five observed weeks predict future harvest weeks. Skill is measured within each season against simple baselines.</div>',
        unsafe_allow_html=True,
    )
    render_metric_strip(artifacts)

    card_grid([
        mini_card_markup("Forecast unit", "<b>Plot sequence</b><br/>One small field unit tracked across weeks."),
        mini_card_markup("Primary validation", "<b>Held-out plots</b><br/>Predictions are made for plots not used to fit that fold."),
        mini_card_markup("Operational view", "<b>Volume planning</b><br/>Held-out plot predictions are summed to test aggregate planning value."),
    ], columns=3)

    left, right = st.columns([1.0, 1.0])
    with left:
        chart_caption("Learned-model improvement over baseline", "Positive values mean lower MAE than the last-observed yield baseline.")
        improvement_fig = plot_model_improvement(artifacts["model_metrics"])
        if improvement_fig is not None:
            st.plotly_chart(improvement_fig, width="stretch", key="overview_improvement", config={"displayModeBar": False})
        else:
            st.info("Model metrics are not available yet.")
    with right:
        chart_caption("Future harvest targets", "Yield scale differs strongly by season; evaluation is therefore within-season.")
        fig = plot_target_summary(artifacts["target_summary"])
        if fig is not None:
            st.plotly_chart(fig, width="stretch", key="overview_targets", config={"displayModeBar": False})
        else:
            st.info("Target summary not available yet.")

def render_data_audit(artifacts: dict[str, pd.DataFrame]) -> None:
    html_block('<div class="section-kicker">Data audit</div>')
    st.header("Are the processed data ready for forecasting?")
    st.markdown('<div class="section-intro">This page checks target availability, plot/date consistency, feature completeness, and whether the model inputs avoid obvious yield/target leakage.</div>', unsafe_allow_html=True)

    tidy = artifacts["tidy_quality"]
    seq = artifacts["sequence_quality"]
    target = artifacts["target_summary"]

    def count_status(df: pd.DataFrame, status: str) -> int:
        if df.empty or "status" not in df.columns:
            return 0
        return int(df["status"].astype(str).str.upper().eq(status).sum())

    n_features = "—"
    if not artifacts["feature_sets"].empty and "features_json" in artifacts["feature_sets"].columns:
        first = artifacts["feature_sets"].iloc[0]["features_json"]
        try:
            n_features = fmt_int(len(json.loads(first)))
        except Exception:
            n_features = "—"
    card_grid([
        metric_card_markup("Tidy PASS checks", fmt_int(count_status(tidy, "PASS")), "observation-level"),
        metric_card_markup("Sequence PASS checks", fmt_int(count_status(seq, "PASS")), "sequence-level"),
        metric_card_markup("Target rows", fmt_int(len(target)), "season × horizon"),
        metric_card_markup("Default features", n_features, "non-reactive set"),
    ], columns=4)

    left, right = st.columns([1.0, .95])
    with left:
        chart_caption("Target yield by future horizon")
        fig = plot_target_summary(target)
        if fig is not None:
            st.plotly_chart(fig, width="stretch", key="data_audit_targets", config={"displayModeBar": False})
    with right:
        chart_caption("Target summary table", "Each row is one season × future harvest horizon.")
        if target.empty:
            st.info("Target summary artifact not available.")
        else:
            st.dataframe(
                compact_columns(
                    target,
                    ["season", "target_horizon_index", "target_date", "n_plots", "total_target_yield_g", "mean_plot_target_yield_g", "zero_yield_plot_share"],
                ),
                use_container_width=True,
                hide_index=True,
                height=245,
            )

    with st.expander("Quality reports", expanded=False):
        tabs = st.tabs(["Tidy observations", "Sequence dataset", "Features"])
        with tabs[0]:
            if not tidy.empty:
                st.dataframe(tidy, use_container_width=True, hide_index=True, height=360)
            else:
                st.info("Tidy quality report not available.")
        with tabs[1]:
            if not seq.empty:
                st.dataframe(seq, use_container_width=True, hide_index=True, height=360)
            else:
                st.info("Sequence quality report not available.")
        with tabs[2]:
            fs = artifacts["feature_sets"].copy()
            if fs.empty:
                st.info("Feature set artifact not available.")
            else:
                if "features_json" in fs.columns:
                    fs["n_features"] = fs["features_json"].apply(lambda v: len(json.loads(v)) if isinstance(v, str) and v.startswith("[") else np.nan)
                st.dataframe(fs, use_container_width=True, hide_index=True, height=360)

def render_model_validation(artifacts: dict[str, pd.DataFrame], seasons: list[str]) -> None:
    html_block('<div class="section-kicker">Model validation</div>')
    st.header("Do learned models beat reactive baselines?")
    st.markdown('<div class="section-intro">The main metric is plot-level MAE across future horizons. Lower MAE is better; positive improvement means the model beats the last-observed yield baseline.</div>', unsafe_allow_html=True)
    metrics = artifacts["model_metrics"]
    predictions = artifacts["model_predictions"]
    if metrics.empty:
        st.info("Run `scripts/phemut/03_train_phemut_sequence_models.py --fast` first.")
        return

    season = season_control("Season", seasons, key="model_season") if seasons else None
    overall = filtered_season(metric_rows_for_overall(metrics), season)
    best = overall.sort_values("mae").iloc[0] if not overall.empty and "mae" in overall.columns else None

    if best is not None:
        card_grid([
            metric_card_markup("Selected best model", str(best.get("model_label", "—")), f"{display_season_value(best.get('season', season))}"),
            metric_card_markup("MAE", fmt_float(best.get("mae")), "g per plot-horizon"),
            metric_card_markup("R²", fmt_float(best.get("r2"), 3), "held-out plots"),
            metric_card_markup("Improvement", fmt_pct(best.get("mae_improvement_pct_vs_last_observed_baseline")), "vs baseline"),
        ], columns=4)

    left, right = st.columns([1.05, .95])
    with left:
        chart_caption(f"Plot-level MAE ranking — {season_label(season)}")
        fig = plot_model_skill(metrics, season)
        if fig is not None:
            st.plotly_chart(fig, width="stretch", key="model_skill_rank", config={"displayModeBar": False})
    with right:
        models = safe_unique(overall.get("model_label", []))
        default_model = selected_best_model(metrics, season, "mae")
        model = st.selectbox("Prediction scatter model", models, index=models.index(default_model) if default_model in models else 0, key="scatter_model") if models else None
        if season is not None and model:
            chart_caption(f"Observed vs predicted — {model}", "Dotted line = perfect agreement.")
            scatter = plot_predictions_scatter(predictions, season, model)
            if scatter is not None:
                st.plotly_chart(scatter, width="stretch", key="model_scatter", config={"displayModeBar": False})
            else:
                st.info("Prediction rows not available for this model/season.")

    with st.expander("Metric tables", expanded=False):
        st.dataframe(
            compact_columns(
                overall.sort_values("mae"),
                ["season", "model_label", "n_predictions", "n_target_horizons", "mae", "rmse", "r2", "correlation", "bias_predicted_minus_observed_g", "mae_improvement_pct_vs_last_observed_baseline"],
            ),
            use_container_width=True,
            hide_index=True,
            height=360,
        )
        horizon_rows = filtered_season(metrics, season)
        if "target_horizon_index" in horizon_rows.columns:
            horizon_rows = horizon_rows[pd.to_numeric(horizon_rows["target_horizon_index"], errors="coerce").notna()]
        if not horizon_rows.empty:
            st.caption("Horizon-level detail")
            sort_cols = [c for c in ["target_horizon_index", "mae"] if c in horizon_rows.columns]
            st.dataframe(horizon_rows.sort_values(sort_cols), use_container_width=True, hide_index=True, height=360)

def render_aggregate_forecast(artifacts: dict[str, pd.DataFrame], seasons: list[str]) -> None:
    html_block('<div class="section-kicker">Aggregate forecast</div>')
    st.header("Can predictions support harvest-volume planning?")
    st.markdown('<div class="section-intro">Aggregate evaluation sums predictions only for plots held out from training in the current fold. A fold is one validation group of plots, usually about 8 plots out of 40.</div>', unsafe_allow_html=True)

    metrics = artifacts["aggregate_metrics"]
    fold_board = artifacts["aggregate_fold_board"]
    if metrics.empty:
        st.info("Aggregate metrics are not available yet.")
        return

    season = season_control("Season", seasons, key="agg_season") if seasons else None
    primary = filtered_season(primary_aggregate_metrics(metrics), season)
    best = primary.sort_values("aggregate_mae_g").iloc[0] if not primary.empty and "aggregate_mae_g" in primary.columns else None

    if best is not None:
        card_grid([
            metric_card_markup("Best aggregate model", str(best.get("model_label", "—")), f"{display_season_value(best.get('season', season))}"),
            metric_card_markup("Aggregate MAE", fmt_float(best.get("aggregate_mae_g")), "g per fold curve"),
            metric_card_markup("Aggregate R²", fmt_float(best.get("aggregate_r2"), 3), "held-out folds"),
            metric_card_markup("Improvement", fmt_pct(best.get("aggregate_mae_improvement_pct_vs_last_observed_baseline")), "vs baseline"),
        ], columns=4)

    season_board = filtered_season(fold_board, season)
    models = safe_unique(season_board.get("model_label", []))
    default_model = selected_best_model(metrics, season, "aggregate_mae_g")
    model = st.selectbox("Aggregate model", models, index=models.index(default_model) if default_model in models else 0, key="agg_model") if models else None
    folds = sorted(pd.to_numeric(season_board.get("fold_id", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique().tolist())
    fold = st.selectbox("Validation fold", folds, index=0, key="agg_fold", help="One held-out group of plots. The model is fitted without these plots, then predictions are summed for this group.") if folds else None

    left, right = st.columns([1.0, 1.0])
    with left:
        chart_caption(f"Held-out aggregate MAE ranking — {season_label(season)}")
        fig = plot_aggregate_metric(metrics, season)
        if fig is not None:
            st.plotly_chart(fig, width="stretch", key="aggregate_skill_rank", config={"displayModeBar": False})
    with right:
        if season is not None and model:
            chart_caption(f"Held-out aggregate curve — {model}, fold {fold}", "Observed total vs predicted total for plots not used in training.")
            curve = plot_aggregate_curve(fold_board, season, model, fold)
            if curve is not None:
                st.plotly_chart(curve, width="stretch", key="aggregate_curve", config={"displayModeBar": False})

    if season is not None and model:
        box = plot_fold_errors(fold_board, season, model)
        if box is not None:
            with st.expander("Absolute error by horizon", expanded=False):
                st.plotly_chart(box, width="stretch", key="aggregate_error_box", config={"displayModeBar": False})

    with st.expander("Primary aggregate metrics table", expanded=False):
        st.dataframe(
            compact_columns(
                primary.sort_values("aggregate_mae_g"),
                ["season", "model_label", "validation_scope", "n_aggregate_rows", "n_folds", "mean_heldout_plots_per_row", "aggregate_mae_g", "aggregate_rmse_g", "aggregate_r2", "aggregate_correlation", "aggregate_mae_improvement_pct_vs_last_observed_baseline"],
            ),
            use_container_width=True,
            hide_index=True,
            height=380,
        )

    with st.expander("Illustrative reconstructed full-field curve", expanded=False):
        message_card(
            "Use only for communication",
            "This reconstructs a full-field out-of-fold curve for visualization. It is not used for model selection because recombining all folds can create arithmetic artifacts for some baselines.",
            "warn",
        )
        recon = filtered_season(artifacts["aggregate_reconstructed_board"], season)
        if recon.empty:
            st.info("Reconstructed board not available.")
        else:
            st.dataframe(recon.head(300), use_container_width=True, hide_index=True, height=360)

def render_drivers(artifacts: dict[str, pd.DataFrame], seasons: list[str]) -> None:
    html_block('<div class="section-kicker">Forecast drivers</div>')
    st.header("Which signals drive the forecast?")
    st.markdown(
        '<div class="section-intro">This page shows model reliance, not causal proof. Feature importance is grouped into crop-relevant buckets instead of showing 98 individual flattened features.</div>',
        unsafe_allow_html=True,
    )
    grouped = artifacts.get("grouped_importance", pd.DataFrame())
    weather = artifacts.get("weather_driver_summary", pd.DataFrame())
    if grouped.empty:
        st.info("Run the updated sequence model script to generate grouped feature importance outputs.")
        return

    season = season_control("Season", seasons, key="driver_season") if seasons else None
    season_grouped = filtered_season(grouped, season)
    models = safe_unique(season_grouped.get("model_label", []))
    default_model = selected_best_model(artifacts.get("model_metrics", pd.DataFrame()), season, "mae")
    model = st.selectbox("Model", models, index=models.index(default_model) if default_model in models else 0, key="driver_model") if models else None

    message_card(
        "How to read this page",
        "These results describe what the fitted model relied on in this validation setup. They should not be read as causal agronomic effects. Weather/GDD features were included as agronomic context, but their separate contribution is hard to isolate here because all plots share the same weekly weather within a season.",
        "note",
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        chart_caption("Grouped model reliance")
        fig = plot_grouped_importance(grouped, season, model)
        if fig is not None:
            st.plotly_chart(fig, width="stretch", key="grouped_importance", config={"displayModeBar": False})
    with c2:
        chart_caption("Weather/GDD diagnostic")
        wfig = plot_weather_driver_summary(weather, season, model)
        if wfig is not None:
            st.plotly_chart(wfig, width="stretch", key="weather_driver_summary", config={"displayModeBar": False})
        else:
            message_card(
                "No robust separate weather/GDD signal in this model",
                "Weather is agronomically relevant and was included in the feature set. In this small within-season tabular validation, observed crop state carried most of the predictive signal and weather did not appear as a separate model-reliance driver.",
                "warn",
            )

    table = season_grouped[season_grouped["model_label"].astype(str).eq(str(model))].copy() if model else season_grouped.copy()
    if not table.empty and "feature_group" in table.columns:
        if "importance_share_pct" in table.columns:
            table["importance_share_pct"] = pd.to_numeric(table["importance_share_pct"], errors="coerce").fillna(0)
            weather_zero = table["feature_group"].astype(str).str.contains("Weather", case=False, na=False) & table["importance_share_pct"].le(0.05)
            table = table[~weather_zero].copy()
        table["feature_group"] = table["feature_group"].apply(clean_feature_group_label)
    with st.expander("Model reliance table", expanded=False):
        st.dataframe(
            compact_columns(
                table.sort_values("importance_share_pct", ascending=False) if "importance_share_pct" in table.columns else table,
                ["season", "model_label", "feature_group", "importance_share_pct", "mean_group_importance", "n_model_fits"],
            ),
            use_container_width=True,
            hide_index=True,
            height=260,
        )

def render_uncertainty(artifacts: dict[str, pd.DataFrame], seasons: list[str]) -> None:
    html_block('<div class="section-kicker">Forecast uncertainty</div>')
    st.header("How uncertain are the predictions?")
    st.markdown('<div class="section-intro">Intervals are simple residual-calibrated 80% bands from other CV folds. They are useful for communication, not a production-grade probabilistic forecast.</div>', unsafe_allow_html=True)
    summary = artifacts.get("uncertainty_summary", pd.DataFrame())
    intervals = artifacts.get("prediction_intervals", pd.DataFrame())
    if summary.empty:
        st.info("Run the updated sequence model script to generate uncertainty outputs.")
        return
    season = season_control("Season", seasons, key="uncertainty_season") if seasons else None
    df = filtered_season(summary, season)
    learned = df[~df["model_label"].apply(is_baseline_label)] if "model_label" in df.columns else df
    best = learned.sort_values("mae").iloc[0] if not learned.empty and "mae" in learned.columns else None
    if best is not None:
        card_grid([
            metric_card_markup("Selected model", str(best.get("model_label", "—")), season_label(best.get("season", season))),
            metric_card_markup("MAE", fmt_float(best.get("mae")), "g per plot-horizon"),
            metric_card_markup("80% coverage", fmt_pct(best.get("interval_coverage_pct")), "retrospective CV"),
            metric_card_markup("Mean band width", fmt_float(best.get("mean_interval_width_g")), "g"),
        ], columns=4)
    fig = plot_uncertainty_summary(summary, season)
    if fig is not None:
        chart_caption("Coverage vs interval width", "A useful forecast should balance narrower bands with acceptable coverage.")
        st.plotly_chart(fig, width="stretch", key="uncertainty_summary", config={"displayModeBar": False})
    st.dataframe(
        compact_columns(
            df.sort_values("mae") if "mae" in df.columns else df,
            ["season", "model_label", "n_predictions", "mae", "mean_interval_width_g", "median_interval_width_g", "interval_coverage_pct", "mean_calibration_residual_n"],
        ),
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("Prediction-level intervals", expanded=False):
        if intervals.empty:
            st.info("Prediction interval rows not available.")
        else:
            view = filtered_season(intervals, season)
            st.dataframe(
                compact_columns(
                    view.head(500),
                    ["season", "plot_id", "model_label", "target_horizon_index", "target_date", "target_yield_g", "prediction_g", "prediction_lower80_g", "prediction_upper80_g", "interval_covered_target"],
                ),
                use_container_width=True,
                hide_index=True,
            )


def render_stakeholder_summary(artifacts: dict[str, pd.DataFrame]) -> None:
    html_block('<div class="section-kicker">Stakeholder summary</div>')
    st.header("One-page decision summary")
    st.markdown('<div class="section-intro">This page translates the technical validation into a short operational readout: model choice, expected error, aggregate planning value, drivers, uncertainty, and caveats.</div>', unsafe_allow_html=True)
    summary = artifacts.get("stakeholder_summary", pd.DataFrame())
    if summary.empty:
        st.info("Run the updated sequence model script to generate the stakeholder summary.")
        return
    display_summary = summary.drop(columns=["paper_context"], errors="ignore").copy()
    if "top_driver_groups" in display_summary.columns:
        display_summary["top_driver_groups"] = display_summary["top_driver_groups"].apply(clean_driver_summary_text)

    for row in display_summary.itertuples(index=False):
        message_card(
            f"Season {season_label(row.season)}",
            f"<b>Plot model:</b> {escape(str(row.recommended_plot_model))} · MAE {float(row.plot_mae_g):.2f} g · improvement {float(row.plot_improvement_pct_vs_last_observed):.1f}%<br/>"
            f"<b>Aggregate model:</b> {escape(str(row.recommended_aggregate_model))} · MAE {float(row.aggregate_mae_g):.2f} g · improvement {float(row.aggregate_improvement_pct_vs_last_observed):.1f}%<br/>"
            f"<b>Main model signals:</b> {escape(str(row.top_driver_groups))}<br/>"
            f"<b>Uncertainty:</b> 80% interval coverage {float(row.uncertainty_coverage_pct):.1f}%; mean band width {float(row.mean_interval_width_g):.2f} g.",
            "note",
        )
    with st.expander("Detailed summary table", expanded=False):
        st.dataframe(display_summary, use_container_width=True, hide_index=True, height=240)

def render_lag_audit(artifacts: dict[str, pd.DataFrame], seasons: list[str]) -> None:
    html_block('<div class="section-kicker">Lag audit</div>')
    st.header("Does the forecast anticipate rather than just follow?")
    st.markdown('<div class="section-intro">This audit compares each aggregate prediction with the correct target horizon and with the previous horizon. A lag warning means a curve may be following the previous harvest window rather than anticipating the target.</div>', unsafe_allow_html=True)
    timeliness = artifacts["timeliness"]
    if timeliness.empty:
        st.info("Timeliness audit not available yet.")
        return

    season = season_control("Season", seasons, key="lag_season", include_all=True) if seasons else "All"
    df = timeliness.copy() if season == "All" else filtered_season(timeliness, season)
    if df.empty:
        st.info("No timeliness rows for this selection.")
        return

    warning_bool = df["lag_warning"].fillna(False).astype(bool) if "lag_warning" in df.columns else pd.Series(False, index=df.index)
    baseline_bool = df["model_label"].apply(is_baseline_label) if "model_label" in df.columns else pd.Series(False, index=df.index)
    learned_bool = ~baseline_bool

    card_grid([
        metric_card_markup("Audited rows", fmt_int(len(df)), "model × fold combinations"),
        metric_card_markup("Total warnings", fmt_int(warning_bool.sum()), "lag_warning = True"),
        metric_card_markup("Baseline warnings", fmt_int((warning_bool & baseline_bool).sum()), "expected for reactive baselines"),
        metric_card_markup("Learned warnings", fmt_int((warning_bool & learned_bool).sum()), "inspect these first"),
    ], columns=4)

    summary_fig = plot_lag_warning_summary(df, None)
    if summary_fig is not None:
        chart_caption("Lag warnings by model group", "The goal is to keep warnings concentrated in reactive baselines rather than learned models.")
        st.plotly_chart(summary_fig, width="stretch", key="lag_warning_summary", config={"displayModeBar": False})

    with st.expander("Detailed timeliness scatter", expanded=False):
        fig = plot_lag_scatter(df, None)
        if fig is not None:
            chart_caption("Target alignment vs previous-horizon alignment", "Points below the dotted line are closer to the previous horizon than to the target horizon, which can indicate reactive behaviour.")
            st.plotly_chart(fig, width="stretch", key="lag_scatter", config={"displayModeBar": False})

    left, right = st.columns([1, 1])
    with left:
        chart_caption("Learned-model warnings")
        learned_warn = df[warning_bool & learned_bool]
        if learned_warn.empty:
            message_card("No learned-model lag warnings in this view", "Warnings are concentrated in reactive baselines, which is expected.", "good")
        else:
            st.dataframe(
                compact_columns(
                    learned_warn.sort_values([c for c in ["season", "same_horizon_mae_g"] if c in learned_warn.columns]),
                    ["season", "model_label", "fold_id", "same_horizon_corr", "prediction_vs_previous_horizon_corr", "same_horizon_mae_g", "prediction_vs_previous_horizon_mae_g", "direction_accuracy", "lag_note"],
                ),
                use_container_width=True,
                hide_index=True,
            )
    with right:
        chart_caption("Baseline warnings")
        base_warn = df[warning_bool & baseline_bool]
        if base_warn.empty:
            st.info("No baseline warnings in this view.")
        else:
            st.dataframe(
                compact_columns(
                    base_warn.sort_values([c for c in ["season", "same_horizon_mae_g"] if c in base_warn.columns]),
                    ["season", "model_label", "fold_id", "same_horizon_mae_g", "prediction_vs_previous_horizon_mae_g", "direction_accuracy", "lag_note"],
                ),
                use_container_width=True,
                hide_index=True,
            )


def render_lag_audit_compact(artifacts: dict[str, pd.DataFrame], seasons: list[str]) -> None:
    timeliness = artifacts.get("timeliness", pd.DataFrame())
    if timeliness.empty:
        st.info("Timeliness audit not available yet.")
        return

    season = season_control("Season", seasons, key="lag_limitations_season", include_all=True) if seasons else "All"
    df = timeliness.copy() if season == "All" else filtered_season(timeliness, season)
    if df.empty:
        st.info("No timeliness rows for this selection.")
        return

    warning_bool = df["lag_warning"].fillna(False).astype(bool) if "lag_warning" in df.columns else pd.Series(False, index=df.index)
    baseline_bool = df["model_label"].apply(is_baseline_label) if "model_label" in df.columns else pd.Series(False, index=df.index)
    learned_bool = ~baseline_bool

    card_grid([
        metric_card_markup("Audited rows", fmt_int(len(df)), "model × fold combinations"),
        metric_card_markup("Total warnings", fmt_int(warning_bool.sum()), "lag_warning = True"),
        metric_card_markup("Baseline warnings", fmt_int((warning_bool & baseline_bool).sum()), "expected for reactive baselines"),
        metric_card_markup("Learned warnings", fmt_int((warning_bool & learned_bool).sum()), "checked separately"),
    ], columns=4)

    learned_warn = df[warning_bool & learned_bool]
    if learned_warn.empty:
        message_card("No learned-model lag warnings in this view", "Warnings are concentrated in reactive baselines, which is expected.", "good")
    else:
        message_card("Some learned-model lag warnings", "A small number of learned model/fold combinations behave closer to the previous harvest window than the target. This is why lag checks are kept as a caveat rather than hidden.", "warn")
        st.dataframe(
            compact_columns(
                learned_warn.sort_values([c for c in ["season", "same_horizon_mae_g"] if c in learned_warn.columns]),
                ["season", "model_label", "fold_id", "same_horizon_corr", "prediction_vs_previous_horizon_corr", "same_horizon_mae_g", "prediction_vs_previous_horizon_mae_g", "direction_accuracy", "lag_note"],
            ),
            use_container_width=True,
            hide_index=True,
            height=260,
        )


def render_limitations(artifacts: dict[str, pd.DataFrame], seasons: list[str]) -> None:
    html_block('<div class="section-kicker">Limitations</div>')
    st.header("Honest interpretation")

    c1, c2, c3 = st.columns(3)
    with c1:
        mini_card("What this demonstrates", "A reproducible audit pipeline for yield forecasting data: profiling, temporal alignment, sequence reformulation, baseline comparison, aggregate validation, and dashboard communication.")
    with c2:
        mini_card("What it does not claim", "It does not use raw images directly, does not claim production deployment readiness, and does not present cross-farm generalisation as proven.")
    with c3:
        mini_card("Why this is useful", "The project emphasises validation discipline: model skill, lag checks, and aggregate planning value are separated rather than mixed into one headline number.")

    message_card(
        "Portfolio framing",
        "<b>Recommended wording:</b><br/>I built an independent forecasting validation pipeline focused on data suitability, temporal alignment, baseline comparison, sequence modelling, uncertainty, and stakeholder-ready diagnostics.",
        "note",
    )
    message_card(
        "Key caveat",
        "The two seasons should be interpreted carefully because yield scale and management conditions differ. The dashboard therefore emphasizes within-season validation and does not present cross-season transfer as proven. Driver results are model evidence, not causal agronomic proof.",
        "warn",
    )

    with st.expander("Technical lag/timeliness audit", expanded=False):
        render_lag_audit_compact(artifacts, seasons)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="side-brand">
                <div class="side-kicker">Independent forecasting validation</div>
                <div class="side-title">🍓 Strawberry yield forecasting</div>
                <div class="side-subtitle">Sequence-based validation with processed plot-level berry data.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="side-brand">
                <div class="card-title">Rubén García Domínguez</div>
                <div class="side-subtitle">
                    Portfolio case study for agricultural forecasting and data-quality workflows.<br/><br/>
                    <a href="https://github.com/rubengarciad93bioinfo" target="_blank">GitHub profile ↗</a><br/>
                    <a href="https://github.com/rubengarciad93bioinfo/berry-yield-forecasting-case-study" target="_blank">Project repository ↗</a><br/><br/>
                    Python · pandas · scikit-learn · Streamlit · Plotly
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        processed_dir_input = str(DEFAULT_PROCESSED_DIR)

    processed_dir = Path(processed_dir_input)
    artifacts = load_artifacts(processed_dir)
    seasons = available_seasons(artifacts)

    missing_core = [
        filename
        for key, filename in [
            ("sequence_quality", "phemut_sequence_quality_report.csv"),
            ("model_metrics", "phemut_sequence_model_metrics.csv"),
            ("aggregate_metrics", "phemut_sequence_aggregate_metrics.csv"),
        ]
        if artifacts[key].empty
    ]
    

    tab_labels = [
        "Overview",
        "Data audit",
        "Model validation",
        "Aggregate forecast",
        "Drivers",
        "Uncertainty",
        "Stakeholder summary",
        "Limitations",
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        render_overview(artifacts)
    with tabs[1]:
        render_data_audit(artifacts)
    with tabs[2]:
        render_model_validation(artifacts, seasons)
    with tabs[3]:
        render_aggregate_forecast(artifacts, seasons)
    with tabs[4]:
        render_drivers(artifacts, seasons)
    with tabs[5]:
        render_uncertainty(artifacts, seasons)
    with tabs[6]:
        render_stakeholder_summary(artifacts)
    with tabs[7]:
        render_limitations(artifacts, seasons)


if __name__ == "__main__":
    main()
