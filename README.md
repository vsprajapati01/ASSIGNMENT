# Global Weather Forecasting — ML Analysis

**Data science project:** end-to-end analysis and temperature forecasting using the Global Weather Repository dataset (Kaggle), built with Python (pandas, scikit-learn, matplotlib, seaborn).

## What it does
- Exploratory analysis of weather data across 15 world cities (temperature trends, correlations, anomalies, spatial and climate patterns)
- Trains and compares regression models — Linear Regression, Ridge, Random Forest, Gradient Boosting, and a Voting ensemble — evaluated with MAE, MSE, and R-squared
- Permutation feature importance to explain what drives temperature predictions
- Time-series anomaly detection and forecast visualizations

## Results
- 8 publication-style figures in `figures/` (EDA, correlations, anomalies, model comparison, feature importance, spatial, climate, forecast)
- Full written report: `Weather_Forecasting_Report.pdf`

## Reproduce
```bash
pip install -r requirements.txt
python analysis.py      # runs EDA + modeling, saves figures/
python build_report.py  # builds the PDF report
```

## Files
- `analysis.py` — data prep, EDA, modeling, evaluation, figures
- `build_report.py` — report generation
- `figures/` — all charts
- `Weather_Forecasting_Report.pdf` — full write-up

*Self-directed data science project, 2026.*
