# Strawberry Yield Forecasting Validation Dashboard

Portfolio case study by **Rubén García Domínguez**.

Live app: https://berry-yield-forecasting-case-study.streamlit.app/  
GitHub profile: https://github.com/rubengarciad93bioinfo

## Overview

This project is an independent yield-forecasting validation pipeline for strawberry production data.

The goal is not to reproduce the original PheMuT research model. Instead, the project demonstrates a practical workflow for:

- profiling agricultural forecasting data;
- checking target availability and temporal alignment;
- reformulating weekly observations into sequence-based forecast examples;
- comparing learned models against simple baselines;
- validating both plot-level predictions and aggregate harvest-volume planning;
- communicating uncertainty, model limitations, and stakeholder-facing diagnostics.

## Why this project is relevant

Agricultural forecasting is not only about fitting a model. A useful workflow also needs to answer:

- Is the dataset suitable for forecasting?
- Are features available before the target date?
- Does the model beat simple operational baselines?
- Does the forecast help with aggregate volume planning?
- Are predictions anticipatory, or just lagging behind recent yield?
- Can the result be explained clearly to non-technical stakeholders?

This dashboard is structured around those questions.

## Data source and attribution

This case study uses processed public data from the PheMuT strawberry yield forecasting project:

- PheMuT repository: https://github.com/Sycamorers/PheMuT
- Paper: *PheMuT: A phenology-informed multi-modal time-series model for strawberry yield forecasting*

No original PheMuT research code is used in this dashboard. The pipeline and Streamlit app were built independently for portfolio and validation purposes.

## Modelling approach

Each plot is treated as a short weekly sequence. The first five observed weeks are used to predict future harvest weeks.

The workflow evaluates:

- plot-level model skill;
- held-out plot validation;
- aggregate forecast value by summing held-out plot predictions;
- simple baselines such as last-observed yield and horizon means;
- uncertainty intervals calibrated from cross-validation residuals.

The dashboard emphasizes within-season validation because the two seasons differ strongly in yield scale and management conditions.

## Dashboard sections

### Overview

High-level validation results and headline metrics.

### Data audit

Checks target availability, plot/date consistency, feature completeness, and obvious leakage risks.

### Model validation

Compares learned models with simple reactive baselines at plot level.

### Aggregate forecast

Tests whether held-out plot predictions can support harvest-volume planning.

### Drivers

Summarizes model reliance by feature groups such as phenology counts and canopy structure. These results are model diagnostics, not causal agronomic claims.

Weather/GDD features are included because they are agronomically relevant, but their independent contribution is difficult to isolate in this small within-season tabular setup because all plots share the same weekly weather.

### Uncertainty

Shows simple residual-calibrated 80% prediction intervals for communication.

### Stakeholder summary

Condenses model choice, forecast error, aggregate planning value, main signals, uncertainty, and caveats into a short operational readout.

### Limitations

States what the project demonstrates and what it does not claim.

## Project structure

```text
app/
  streamlit_app.py

scripts/phemut/
  00_profile_phemut_dataset.py
  01_build_phemut_tidy_dataset.py
  02_build_phemut_sequence_dataset.py
  03_train_phemut_sequence_models.py
  04_audit_phemut_temporal_alignment.py

data/processed/phemut/
  Processed dashboard-ready outputs
```

## Run locally

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app/streamlit_app.py
```

## Reproduce the processed outputs

First, clone or download the public PheMuT repository outside this project:

```bash
git clone https://github.com/Sycamorers/PheMuT.git ../PheMuT
```

Then point the scripts to the local PheMuT data folder:

```bash
export PHEMUT_DATA_DIR="../PheMuT/data"
```

If you cloned or downloaded the PheMuT repository somewhere else, replace `../PheMuT/data` with your local path to the PheMuT `data` folder.

Then run the pipeline:

```bash
python scripts/phemut/00_profile_phemut_dataset.py \
  --data-dir "$PHEMUT_DATA_DIR" \
  --out-dir data/processed/phemut \
  --long-output full

python scripts/phemut/01_build_phemut_tidy_dataset.py \
  --data-dir "$PHEMUT_DATA_DIR" \
  --out-dir data/processed/phemut

python scripts/phemut/02_build_phemut_sequence_dataset.py \
  --processed-dir data/processed/phemut

python scripts/phemut/03_train_phemut_sequence_models.py \
  --processed-dir data/processed/phemut \
  --fast

python scripts/phemut/04_audit_phemut_temporal_alignment.py \
  --processed-dir data/processed/phemut
```

## Main caveats

This is a portfolio validation project, not a production forecasting system.

It does not:

- use raw imagery directly;
- claim cross-farm generalisation;
- claim causal agronomic effects from feature importance;
- claim deployment readiness.

Its main purpose is to demonstrate forecasting validation discipline, data-quality thinking, baseline comparison, aggregate evaluation, uncertainty communication, and stakeholder-ready presentation.

## Tech stack

Python, pandas, NumPy, scikit-learn, Streamlit, Plotly.
