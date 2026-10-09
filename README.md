# Global Weather Forecasting — ML Analysis

End-to-end machine learning project: temperature analysis and forecasting across 15 world cities using the Global Weather Repository dataset (Kaggle). Built with Python — pandas, scikit-learn, matplotlib, seaborn.

## Pipeline
1. **Data** — global weather observations for 15 cities (New York, London, Tokyo, Mumbai, Sydney, Cairo, Sao Paulo, Toronto, Berlin, Beijing, Nairobi, Dubai, Moscow, Buenos Aires, Mexico City), with city metadata (lat/lon, continent, climate type) and climatological baselines.
2. **EDA** — temperature distributions, correlation heatmaps, time-series plots with anomaly detection, spatial and climate-pattern charts (`figures/fig1`–`fig3`, `fig6`–`fig7`).
3. **Modeling** — regression models trained and compared: Linear Regression, Ridge, Random Forest, Gradient Boosting, and a Voting ensemble (`figures/fig4`).
4. **Evaluation** — MAE, MSE, and R-squared; permutation feature importance to explain drivers (`figures/fig5`).
5. **Forecasting** — forward temperature forecasts with visualizations (`figures/fig8`).
6. **Report** — `build_report.py` compiles everything into `Weather_Forecasting_Report.pdf`.

## Reproduce
```bash
pip install -r requirements.txt
python analysis.py      # EDA + modeling + figures
python build_report.py  # PDF report
```

## Files
- `analysis.py` — data prep, EDA, modeling, evaluation, figures
- `build_report.py` — report generation
- `figures/` — all charts
- `Weather_Forecasting_Report.pdf` — full write-up
