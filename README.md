# Global Weather Forecasting — ML Analysis (Study Reference)

**Study project:** I am working through this codebase to learn end-to-end machine learning workflows. The original code was written by someone else; I keep it here as a learning reference.

## What the project covers
- Exploratory analysis of weather data across 15 world cities (temperature trends, correlations, anomalies, spatial and climate patterns)
- Regression models compared: Linear Regression, Ridge, Random Forest, Gradient Boosting, and a Voting ensemble — evaluated with MAE, MSE, and R-squared
- Permutation feature importance, time-series anomaly detection, forecast visualizations

## What I am learning from it
- EDA with pandas/matplotlib/seaborn
- Training and comparing sklearn regressors
- Model evaluation (MAE, MSE, R2) and feature importance
- Structuring an analysis project (analysis script + report builder + figures)

## Files
- `analysis.py` — data prep, EDA, modeling, evaluation, figures
- `build_report.py` — report generation
- `figures/` — all charts
- `Weather_Forecasting_Report.pdf` — full write-up

```bash
pip install -r requirements.txt
python analysis.py
```
