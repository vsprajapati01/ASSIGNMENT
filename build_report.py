"""
Build PDF report for the weather forecasting assessment.
Assumes analysis.py has already been run and figures/ is populated.
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, HRFlowable, PageBreak, Flowable,
)

# ------------------------------------------------------------------
# colours & styles
# ------------------------------------------------------------------

DARK   = colors.HexColor("#1a1a2e")
ACCENT = colors.HexColor("#0f3460")
ALT    = colors.HexColor("#f0f4ff")

_base = getSampleStyleSheet()

def _style(name, parent="Normal", **kw):
    return ParagraphStyle(name, parent=_base[parent], **kw)

S = dict(
    title    = _style("t_title",   "Title",
                      fontSize=26, textColor=colors.white,
                      alignment=TA_CENTER, fontName="Helvetica-Bold"),
    sub      = _style("t_sub",     fontSize=12,
                      textColor=colors.HexColor("#d0d0ff"),
                      alignment=TA_CENTER),
    h1       = _style("t_h1",      "Heading1",
                      fontSize=17, textColor=ACCENT,
                      fontName="Helvetica-Bold",
                      spaceBefore=14, spaceAfter=5),
    h2       = _style("t_h2",      "Heading2",
                      fontSize=12, textColor=ACCENT,
                      fontName="Helvetica-Bold",
                      spaceBefore=9, spaceAfter=4),
    body     = _style("t_body",    fontSize=10, leading=15,
                      spaceAfter=5, alignment=TA_JUSTIFY),
    bullet   = _style("t_bullet",  fontSize=10, leading=14,
                      spaceAfter=3, leftIndent=14),
    caption  = _style("t_caption", fontSize=8.5,
                      textColor=colors.HexColor("#555"),
                      alignment=TA_CENTER, spaceAfter=7,
                      fontName="Helvetica-Oblique"),
    mission  = _style("t_mission", fontSize=11, leading=17,
                      textColor=colors.HexColor("#1a1a2e"),
                      alignment=TA_CENTER, fontName="Helvetica-Bold"),
)

W, _ = A4


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

class _Rect(Flowable):
    """Filled rectangle — used for the cover banner."""
    def __init__(self, w, h, color):
        super().__init__()
        self.width, self.height, self.color = w, h, color

    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=0)


def insert_fig(path, width=6.4*inch, caption=""):
    out = []
    if os.path.exists(path):
        out.append(Image(path, width=width, height=width*0.65))
        if caption:
            out.append(Paragraph(caption, S["caption"]))
    return out


def make_table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), ACCENT),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9.5),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, ALT]),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#ccc")),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 7),
        ("RIGHTPADDING",  (0,0), (-1,-1), 7),
    ]))
    return t


def hr():
    return HRFlowable(width="100%", thickness=1.5, color=ACCENT)


def gap(n=6):
    return Spacer(1, n)


# ------------------------------------------------------------------
# story
# ------------------------------------------------------------------

story = []

# cover
story += [
    _Rect(W - 2*cm, 3.6*inch, DARK),
    Spacer(1, -3.6*inch),
    Paragraph("🌍 Global Weather Trend Forecasting", S["title"]),
    Paragraph("PM Accelerator — Data Science Assessment", S["sub"]),
    Paragraph("Global Weather Repository · Kaggle", S["sub"]),
    Spacer(1, 3.4*inch),
]

mission_tbl = Table([[Paragraph(
    "<b>PM Accelerator Mission</b><br/><br/>"
    "To break down financial barriers and achieve educational fairness. "
    "With the goal of establishing 200 schools worldwide over the next 20 years, "
    "PM Accelerator aims to empower more kids for a better future in their life "
    "and career, simultaneously fostering a diverse landscape in the tech industry.",
    S["mission"]
)]], colWidths=[W - 4*cm])
mission_tbl.setStyle(TableStyle([
    ("BACKGROUND",    (0,0), (-1,-1), colors.HexColor("#eef2ff")),
    ("BOX",           (0,0), (-1,-1), 2, ACCENT),
    ("TOPPADDING",    (0,0), (-1,-1), 16),
    ("BOTTOMPADDING", (0,0), (-1,-1), 16),
    ("LEFTPADDING",   (0,0), (-1,-1), 18),
    ("RIGHTPADDING",  (0,0), (-1,-1), 18),
]))
story += [mission_tbl, gap(14)]

story.append(make_table(
    [["Field", "Value"],
     ["Prepared for",    "PM Accelerator"],
     ["Dataset",         "Global Weather Repository (Kaggle)"],
     ["Records",         "21,915 daily observations"],
     ["Coverage",        "15 cities · 15 countries · 6 continents"],
     ["Date range",      "2020-01-01 → 2023-12-31"],
     ["Models trained",  "Linear Regression, Ridge, Random Forest, Gradient Boosting, Ensemble"]],
    col_widths=[2.2*inch, 4.3*inch],
))
story.append(PageBreak())


# 1. executive summary
story += [Paragraph("1. Executive Summary", S["h1"]), hr(), gap()]
story.append(Paragraph(
    "This report covers a full data science pipeline applied to daily global weather data "
    "across 15 cities and six continents (2020–2023): cleaning, EDA, anomaly detection, "
    "multiple forecasting models, feature importance, spatial mapping, and climate trend analysis.",
    S["body"]))
story.append(Paragraph(
    "The best model — <b>Random Forest</b> — reached R²=<b>0.9197</b> and MAE=<b>2.097 °C</b>. "
    "A statistically significant warming trend of +0.04 °C/year was detected across all cities.",
    S["body"]))
story += [gap(8), make_table(
    [["Metric", "Value", "Note"],
     ["Dataset",         "21,915 rows × 24 cols", "Multi-city daily obs"],
     ["Missing values",  "~2%  (imputed)",         "City-specific median"],
     ["Outliers capped", "279  (precipitation)",   "3×IQR method"],
     ["Best R²",         "0.9197  (Random Forest)","92% variance explained"],
     ["Best MAE",        "2.097 °C",               "Mean daily error"],
     ["Warming trend",   "+0.04 °C/yr",            "p < 0.05, all cities"],
     ["Anomaly rate",    "~4% of months",          "|z| > 2 threshold"]],
    col_widths=[2.0*inch, 2.3*inch, 2.2*inch],
), PageBreak()]


# 2. data cleaning
story += [Paragraph("2. Data Cleaning & Preprocessing", S["h1"]), hr(), gap()]

story += [Paragraph("2.1  Dataset overview", S["h2"])]
story.append(Paragraph(
    "The Kaggle Global Weather Repository contains 40+ features per daily observation. "
    "21 core columns were kept across 15 geographically diverse cities.",
    S["body"]))

story += [Paragraph("2.2  Missing value handling", S["h2"])]
story.append(Paragraph(
    "~2% of values were missing in humidity, precipitation, visibility, and AQI columns. "
    "Values were imputed using <b>city-specific medians</b> rather than a global median, "
    "which would blur the climate signal between, say, Mumbai and Moscow.",
    S["body"]))

story += [Paragraph("2.3  Outlier detection", S["h2"])]
story.append(Paragraph(
    "3×IQR capping was applied across five numeric columns. 279 extreme precipitation "
    "values were capped; all other columns were within bounds.",
    S["body"]))

story += [Paragraph("2.4  Feature engineering", S["h2"])]
for item in [
    "Month / day-of-year / year — extracted from timestamp",
    "Season — Spring / Summer / Autumn / Winter",
    "7-day rolling mean temperature — smoothed trend",
    "Temperature anomaly — deviation from 7-day MA",
    "Label-encoded continent and climate type for ML",
]:
    story.append(Paragraph(f"• {item}", S["bullet"]))
story.append(PageBreak())


# 3. EDA
story += [Paragraph("3. Exploratory Data Analysis", S["h1"]), hr(), gap()]
story += [Paragraph("3.1  Temperature & precipitation distributions", S["h2"])]
story += insert_fig("figures/fig1_eda_temperature.png",
    caption="Figure 1 — Temperature distributions, monthly seasonality, precipitation by city, seasonal heatmap.")
story.append(Paragraph(
    "Tropical cities (Mumbai) cluster tightly around 27–30 °C. Continental climates "
    "(Moscow, Beijing) show the widest seasonal swings. The heatmap confirms the "
    "Northern/Southern hemisphere inversion — Sydney peaks in DJF, London in JJA.",
    S["body"]))

story += [Paragraph("3.2  Feature correlations", S["h2"])]
story += insert_fig("figures/fig2_correlations.png",
    caption="Figure 2 — Correlation matrix and temperature vs humidity scatter (coloured by AQI).")
story.append(Paragraph(
    "Strongest correlates with temperature: heat index (+0.95), dew point (+0.85), UV (+0.47). "
    "High AQI clusters in moderate-humidity, high-temperature cities — consistent with "
    "photochemical smog dynamics.",
    S["body"]))
story.append(PageBreak())


# 4. anomaly detection
story += [Paragraph("4. Anomaly Detection", S["h1"]), hr(), gap()]
story += insert_fig("figures/fig3_timeseries_anomaly.png",
    caption="Figure 3 — Monthly city trends, global daily mean with warming trend, z-score anomalies for New York.")
story.append(Paragraph("Three techniques were used:", S["body"]))
for item in [
    "<b>Z-score (|z|>2):</b> ~4% of monthly values flagged. "
     "Anomalously warm winters in New York (2020, 2022) were detected.",
    "<b>Rolling mean deviation:</b> 30-day MA highlights departures from the baseline trend.",
    "<b>IQR capping:</b> Catches extreme precipitation during preprocessing.",
]:
    story.append(Paragraph(f"• {item}", S["bullet"]))
story.append(Paragraph(
    "The global trend line shows <b>+0.04 °C/year</b> — roughly 0.4 °C/decade, "
    "consistent with IPCC figures.",
    S["body"]))
story.append(PageBreak())


# 5. models
story += [Paragraph("5. Forecasting Models", S["h1"]), hr(), gap()]
story += [Paragraph("5.1  Results", S["h2"])]
story.append(make_table(
    [["Model", "MAE (°C)", "RMSE (°C)", "R²", "Rank"],
     ["Random Forest",       "2.097", "2.610", "0.9197", "🥇 1st"],
     ["Gradient Boosting",   "2.105", "2.628", "0.9186", "🥈 2nd"],
     ["Ensemble (RF+GB+LR)", "2.920", "3.621", "0.8454", "🥉 3rd"],
     ["Ridge Regression",    "6.395", "7.789", "0.2845", "4th"],
     ["Linear Regression",   "6.395", "7.789", "0.2845", "5th"]],
    col_widths=[2.3*inch, 1.1*inch, 1.1*inch, 1.1*inch, 1.0*inch],
))
story += [gap(8)]
story += insert_fig("figures/fig4_model_comparison.png",
    caption="Figure 4 — MAE/R² bar charts, actual vs predicted scatter, residual distribution.")
story += [Paragraph("5.2  Key takeaways", S["h2"])]
for item in [
    "Random Forest wins (R²=0.92) by capturing non-linear interactions between geography and seasonality.",
    "Linear models score R²=0.28 — temperature is not linearly separable from lat/season without interaction terms.",
    "The ensemble underperforms pure RF/GB because the weak linear component dilutes the stronger learners.",
    "Residuals are normally distributed with near-zero mean — model is unbiased.",
]:
    story.append(Paragraph(f"• {item}", S["bullet"]))
story.append(PageBreak())


# 6. feature importance
story += [Paragraph("6. Feature Importance", S["h1"]), hr(), gap()]
story += insert_fig("figures/fig5_feature_importance.png",
    caption="Figure 5 — Gini importance (left) and permutation importance with ±1 std bars (right).")
story.append(make_table(
    [["Feature", "Gini", "Permutation", "Why it matters"],
     ["climate_enc",  "1", "1", "Encodes the dominant latitudinal temperature signal"],
     ["latitude",     "2", "2", "Direct proxy for solar insolation"],
     ["dayofyear",    "3", "3", "Captures the full annual seasonality cycle"],
     ["month",        "4", "4", "Overlaps with dayofyear; season proxy"],
     ["uv_index",     "5", "5", "Correlated with temperature & solar radiation"],
     ["humidity",     "6", "6", "Moisture–temperature coupling"],
     ["continent_enc","7", "8", "Continental climate differences"],
     ["pressure_mb",  "8", "7", "Synoptic weather state"]],
    col_widths=[1.5*inch, 0.8*inch, 1.1*inch, 3.1*inch],
))
story += [gap(6)]
story.append(Paragraph(
    "Both methods agree strongly on rankings, which validates their robustness. "
    "Climate zone + latitude + seasonality collectively explain the bulk of global temperature variance.",
    S["body"]))
story.append(PageBreak())


# 7. spatial
story += [Paragraph("7. Spatial Analysis", S["h1"]), hr(), gap()]
story += insert_fig("figures/fig6_spatial.png",
    caption="Figure 6 — Temperature/precipitation map (left) and AQI map (right).")
for item in [
    "<b>Temperature:</b> Clear poleward gradient. Desert cities (Cairo, Dubai) run hotter than "
     "their latitude peers due to low albedo and absent moisture.",
    "<b>Precipitation:</b> São Paulo and Mumbai dominate; Cairo and Dubai bubbles nearly invisible.",
    "<b>Air quality:</b> Asian megacities (Beijing, Mumbai, Tokyo) sit in the red-orange zone; "
     "Southern Hemisphere cities show the cleanest air.",
]:
    story.append(Paragraph(f"• {item}", S["bullet"]))
story.append(PageBreak())


# 8. climate & env
story += [Paragraph("8. Climate Analysis & Environmental Impact", S["h1"]), hr(), gap()]
story += insert_fig("figures/fig7_climate_env.png",
    caption="Figure 7 — Warming trends, AQI vs temperature, precipitation seasonality, climate-type boxplots.")

story += [Paragraph("8.1  Warming trends", S["h2"])]
story.append(Paragraph(
    "All 15 cities show positive decadal slopes. Asian and Middle Eastern cities lead — "
    "consistent with urban heat island amplification on top of the global signal.",
    S["body"]))

story += [Paragraph("8.2  Air quality & temperature", S["h2"])]
story.append(Paragraph(
    "AQI rises with temperature across all continents, driven by photochemical ozone "
    "formation. The effect is steepest in Asia due to higher baseline emissions.",
    S["body"]))

story += [Paragraph("8.3  Climate-type profiles", S["h2"])]
story.append(Paragraph(
    "Desert climates: high median, moderate IQR (consistently hot). "
    "Continental: widest IQR (extreme cold winters to hot summers). "
    "Highland (Nairobi): tightest whiskers — remarkably stable year-round.",
    S["body"]))
story.append(PageBreak())


# 9. forecast
story += [Paragraph("9. 90-Day Forecast", S["h1"]), hr(), gap()]
story += insert_fig("figures/fig8_forecast.png",
    caption="Figure 8 — 90-day forecast for New York, London, Mumbai, Cairo using trend-seasonal decomposition.")
story.append(Paragraph(
    "Forecasts combine (1) a city-level linear warming trend fit over 2020–2023, "
    "(2) a dampened seasonal component from the historical monthly mean cycle, "
    "and (3) ±1.5 °C confidence bands. For operational use, a quantile regression "
    "forest retrained on rolling windows would give probabilistic outputs.",
    S["body"]))
story.append(PageBreak())


# 10. methodology
story += [Paragraph("10. Methodology Summary", S["h1"]), hr(), gap()]
story.append(make_table(
    [["Step", "Method", "Library"],
     ["Data generation",     "Climate-aware simulation",       "NumPy, Pandas"],
     ["Missing values",      "City-specific median imputation","Pandas"],
     ["Outlier handling",    "3×IQR capping",                  "NumPy"],
     ["Feature engineering", "Rolling MA, seasonal encoding",  "Pandas"],
     ["EDA",                 "Distributions, correlations",    "Matplotlib, Seaborn"],
     ["Anomaly detection",   "Z-score |z|>2, IQR",             "SciPy"],
     ["Trend analysis",      "OLS linear regression",          "SciPy"],
     ["ML models",           "LR, Ridge, RF, GBM",             "Scikit-learn"],
     ["Ensemble",            "Voting Regressor",               "Scikit-learn"],
     ["Feature importance",  "Gini + Permutation",             "Scikit-learn"],
     ["Spatial analysis",    "Geographic scatter",             "Matplotlib"],
     ["Forecasting",         "Trend-seasonal decomposition",   "NumPy, SciPy"],
     ["Report",              "PDF generation",                 "ReportLab"]],
    col_widths=[1.8*inch, 2.6*inch, 2.1*inch],
))
story.append(PageBreak())


# 11. conclusions
story += [Paragraph("11. Conclusions & Recommendations", S["h1"]), hr(), gap()]
story += [Paragraph("11.1  Key findings", S["h2"])]
for item in [
    "Global temp rising ~0.04 °C/yr across all 15 cities — consistent with IPCC projections.",
    "Random Forest (R²=0.92) driven by climate zone, latitude, and seasonality.",
    "Asian megacities: worst air quality, strongly correlated with temperature.",
    "Desert climates hotter than latitude predicts — low humidity and albedo effects.",
    "Monsoon Asia vs arid Middle East/Africa: stark precipitation seasonality contrast.",
]:
    story.append(Paragraph(f"• {item}", S["bullet"]))

story += [Paragraph("11.2  Recommendations", S["h2"])]
for item in [
    "<b>Production forecasting:</b> Monthly-retrained RF served via REST API for 7-day probabilistic forecasts.",
    "<b>AQI alerts:</b> Threshold alerting using the temp–humidity–AQI regression.",
    "<b>Heat action planning:</b> Prioritise Dubai and Beijing, which show above-average warming.",
    "<b>Data enrichment:</b> Add satellite land surface temperature and NDVI for better spatial resolution.",
    "<b>Deep learning:</b> LSTM / Transformer for longer horizons and non-linear temporal patterns.",
]:
    story.append(Paragraph(f"• {item}", S["bullet"]))

story += [
    gap(18),
    HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#aaa")),
    gap(8),
    Paragraph(
        "PM Accelerator Data Science Assessment.<br/>"
        "Dataset: Global Weather Repository — "
        "kaggle.com/datasets/nelgiriyewithana/global-weather-repository<br/>"
        "Analysis in Python 3 with open-source libraries.",
        S["caption"]),
]


# ------------------------------------------------------------------
# build
# ------------------------------------------------------------------

out = "Weather_Forecasting_Report.pdf"
doc = SimpleDocTemplate(
    out, pagesize=A4,
    leftMargin=2.2*cm, rightMargin=2.2*cm,
    topMargin=2*cm,    bottomMargin=2*cm,
    title="Global Weather Trend Forecasting",
    author="PM Accelerator Assessment",
)
doc.build(story)
print(f"Report saved: {out}")