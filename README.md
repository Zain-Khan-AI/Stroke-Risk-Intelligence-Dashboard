# Stroke Risk Intelligence Dashboard

A healthcare analytics dashboard built with **Streamlit** for exploring stroke indicators, patient demographics, and risk patterns.

## Features

- **Interactive filters** — cohort (gender, residence, work type, smoking status) and clinical (hypertension, heart disease) filters with range sliders for age, glucose, and BMI
- **Executive KPI strip** — live-updating metrics: patient count, stroke rate, risk score, elevated risk share, average age
- **4-tab analytics workspace:**
  - **Overview** — age distribution, stroke outcome mix, clinical indicators snapshot
  - **Risk Patterns** — age/outcome breakdown, glucose box plots, smoking status analysis, correlation matrix, BMI & glucose band distributions
  - **Stratification** — composite risk score bands, observed stroke rate by risk band, age-group risk heatmap
  - **Data** — searchable, exportable patient-level table with CSV download
- **Dark mode** — full dark theme with matched chart backgrounds
- **Custom CSS** — gradient backgrounds, card-based UI, responsive layout

## Tech Stack

Python · Streamlit · Pandas · Plotly · Matplotlib · Seaborn

## Dataset

Uses the [Healthcare Stroke Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (~5,100 records) with features: age, hypertension, heart disease, marital status, work type, residence, glucose level, BMI, smoking status, and stroke outcome.

## Run Locally

```bash
pip install streamlit pandas plotly matplotlib seaborn
python -m streamlit run app.py
```

Or double-click `run_dashboard.bat` (Windows).
