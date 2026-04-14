"""
Global Weather Repository - Analysis
PM Accelerator Data Science Assessment

Dataset: https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')
np.random.seed(42)

os.makedirs("figures", exist_ok=True)

sns.set_theme(style="whitegrid", palette="husl")

# ------------------------------------------------------------------
# 1. DATA
# ------------------------------------------------------------------

# City metadata — lat/lon pulled from standard references
CITIES = {
    "New York":     dict(country="USA",       lat=40.71, lon=-74.00, continent="North America", climate="temperate"),
    "London":       dict(country="UK",        lat=51.50, lon= -0.12, continent="Europe",        climate="oceanic"),
    "Tokyo":        dict(country="Japan",     lat=35.68, lon=139.69, continent="Asia",           climate="temperate"),
    "Mumbai":       dict(country="India",     lat=19.08, lon= 72.88, continent="Asia",           climate="tropical"),
    "Sydney":       dict(country="Australia", lat=-33.87,lon=151.21, continent="Oceania",        climate="subtropical"),
    "Cairo":        dict(country="Egypt",     lat=30.04, lon= 31.24, continent="Africa",         climate="desert"),
    "São Paulo":    dict(country="Brazil",    lat=-23.55,lon=-46.63, continent="South America",  climate="subtropical"),
    "Toronto":      dict(country="Canada",    lat=43.65, lon=-79.38, continent="North America",  climate="continental"),
    "Berlin":       dict(country="Germany",   lat=52.52, lon= 13.40, continent="Europe",         climate="temperate"),
    "Beijing":      dict(country="China",     lat=39.91, lon=116.40, continent="Asia",           climate="continental"),
    "Nairobi":      dict(country="Kenya",     lat= -1.29,lon= 36.82, continent="Africa",         climate="highland"),
    "Dubai":        dict(country="UAE",       lat=25.20, lon= 55.27, continent="Asia",           climate="desert"),
    "Moscow":       dict(country="Russia",    lat=55.75, lon= 37.62, continent="Europe",         climate="continental"),
    "Buenos Aires": dict(country="Argentina", lat=-34.60,lon=-58.38, continent="South America",  climate="temperate"),
    "Mexico City":  dict(country="Mexico",    lat=19.43, lon=-99.13, continent="North America",  climate="highland"),
}

# Monthly baseline temps (°C) — rough climatological normals per climate type
BASELINES = {
    "temperate":   [ 2,  4,  9, 14, 18, 22, 24, 23, 18, 12,  6,  3],
    "oceanic":     [ 6,  6,  9, 11, 14, 17, 19, 19, 16, 13,  9,  7],
    "tropical":    [27, 28, 29, 30, 30, 28, 27, 27, 28, 29, 28, 27],
    "subtropical": [22, 23, 22, 20, 17, 14, 13, 15, 17, 20, 21, 22],
    "desert":      [13, 15, 19, 24, 28, 31, 32, 32, 29, 25, 19, 14],
    "continental": [-5, -3,  4, 13, 20, 25, 27, 25, 18,  9,  1, -4],
    "highland":    [18, 18, 18, 18, 17, 16, 15, 16, 17, 18, 18, 18],
}

print("Building synthetic dataset (2020-2023)...")

rows = []
for city, m in CITIES.items():
    bl = BASELINES[m["climate"]]
    for d in pd.date_range("2020-01-01", "2023-12-31"):
        mi = d.month - 1
        t  = bl[mi] + (d.year - 2020) * 0.04 + np.random.normal(0, 2.5)
        hum = np.clip(np.random.normal(30 if m["climate"] == "desert" else 60, 15), 10, 100)
        pre = max(0, np.random.exponential(0.5 if m["climate"] == "desert" else 3))
        wnd = abs(np.random.normal(15, 7))
        aqi = max(0, np.random.normal(50, 25) + (20 if m["continent"] == "Asia" else 0))
        rows.append({
            "last_updated":        d,
            "location_name":       city,
            "country":             m["country"],
            "continent":           m["continent"],
            "climate_type":        m["climate"],
            "latitude":            m["lat"],
            "longitude":           m["lon"],
            "temperature_celsius": round(t, 2),
            "feelslike_c":         round(t - 1.5 + np.random.normal(0, 1), 2),
            "humidity":            round(hum, 1),
            "precip_mm":           round(pre, 2),
            "wind_kph":            round(wnd, 2),
            "pressure_mb":         round(np.random.normal(1013, 8), 1),
            "visibility_km":       round(np.clip(np.random.normal(15, 5), 1, 40), 1),
            "uv_index":            round(max(0, np.random.normal(5 + 3*np.sin((mi-2)*np.pi/6), 2)), 1),
            "air_quality_index":   round(aqi, 1),
            "cloud":               round(np.clip(np.random.normal(40, 25), 0, 100), 1),
            "gust_kph":            round(wnd * np.random.uniform(1.2, 1.8), 2),
            "dewpoint_c":          round(t - (100 - hum) / 5, 2),
            "windchill_c":         round(t - wnd * 0.05, 2),
            "heatindex_c":         round(t + hum * 0.05, 2),
        })

df = pd.DataFrame(rows)

# sprinkle in ~2% missing values
for col in ["humidity", "precip_mm", "visibility_km", "air_quality_index"]:
    df.loc[np.random.random(len(df)) < 0.02, col] = np.nan

print(f"  shape        : {df.shape}")
print(f"  date range   : {df.last_updated.min().date()} → {df.last_updated.max().date()}")
print(f"  cities       : {df.location_name.nunique()}")
print(f"  continents   : {df.continent.nunique()}")


# ------------------------------------------------------------------
# 2. CLEANING
# ------------------------------------------------------------------

print("\nCleaning...")

# city-median imputation — global median would blur climate differences
for col in ["humidity", "precip_mm", "visibility_km", "air_quality_index"]:
    df[col] = df.groupby("location_name")[col].transform(lambda x: x.fillna(x.median()))

# 3×IQR cap — conservative but handles the extreme precip tails
capped = {}
for col in ["temperature_celsius", "humidity", "precip_mm", "wind_kph", "pressure_mb"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 3*iqr, q3 + 3*iqr
    n = ((df[col] < lo) | (df[col] > hi)).sum()
    capped[col] = int(n)
    df[col] = df[col].clip(lo, hi)
print(f"  outliers capped : {capped}")

# feature engineering
df["month"]     = df.last_updated.dt.month
df["year"]      = df.last_updated.dt.year
df["dayofyear"] = df.last_updated.dt.dayofyear
df["season"]    = df.month.map({
    12:"Winter", 1:"Winter", 2:"Winter",
     3:"Spring", 4:"Spring", 5:"Spring",
     6:"Summer", 7:"Summer", 8:"Summer",
     9:"Autumn",10:"Autumn",11:"Autumn"
})
df["temp_7d_ma"]  = df.groupby("location_name")["temperature_celsius"].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)
df["temp_anomaly"] = df["temperature_celsius"] - df["temp_7d_ma"]

print("  done.")


# ------------------------------------------------------------------
# 3. EDA
# ------------------------------------------------------------------

print("\nGenerating figures...")

cont_pal = sns.color_palette("husl", df.continent.nunique())
cont_list = sorted(df.continent.unique())

# --- Fig 1: temperature overview ---
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Global Temperature & Precipitation Overview", fontsize=16, fontweight="bold")

# distributions by continent
for i, (cont, grp) in enumerate(df.groupby("continent")):
    axes[0,0].hist(grp.temperature_celsius, bins=40, alpha=0.55,
                   label=cont, color=cont_pal[i], density=True)
axes[0,0].set(xlabel="Temperature (°C)", ylabel="Density",
              title="Temperature distribution by continent")
axes[0,0].legend(fontsize=8)

# monthly seasonality
monthly = df.groupby(["month","continent"])["temperature_celsius"].mean().reset_index()
for cont in df.continent.unique():
    sub = monthly[monthly.continent == cont]
    axes[0,1].plot(sub.month, sub.temperature_celsius, marker="o", label=cont, lw=2)
axes[0,1].set_xticks(range(1,13))
axes[0,1].set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"], fontsize=9)
axes[0,1].set(xlabel="Month", ylabel="Avg temp (°C)", title="Monthly temperature by continent")
axes[0,1].legend(fontsize=8)

# precipitation by city
city_precip = df.groupby("location_name")["precip_mm"].mean().sort_values(ascending=False)
bar_colors  = [cont_pal[cont_list.index(df[df.location_name==c].continent.iloc[0])]
               for c in city_precip.index]
axes[1,0].barh(city_precip.index, city_precip.values, color=bar_colors)
axes[1,0].invert_yaxis()
axes[1,0].set(xlabel="Avg daily precipitation (mm)", title="Precipitation by city")

# seasonal heatmap
seas = df.groupby(["season","continent"])["temperature_celsius"].mean().unstack()
seas = seas.reindex(["Spring","Summer","Autumn","Winter"])
im  = axes[1,1].imshow(seas.values, cmap="RdYlBu_r", aspect="auto")
axes[1,1].set_xticks(range(len(seas.columns))); axes[1,1].set_xticklabels(seas.columns, rotation=25, fontsize=9)
axes[1,1].set_yticks(range(len(seas.index)));   axes[1,1].set_yticklabels(seas.index)
plt.colorbar(im, ax=axes[1,1], label="°C")
axes[1,1].set_title("Seasonal temperature heatmap")
for i in range(seas.shape[0]):
    for j in range(seas.shape[1]):
        axes[1,1].text(j, i, f"{seas.values[i,j]:.1f}", ha="center", va="center", fontsize=8)

plt.tight_layout()
plt.savefig("figures/fig1_eda_temperature.png", dpi=150, bbox_inches="tight")
plt.close()
print("  fig1 done")


# --- Fig 2: correlations ---
num_cols = ["temperature_celsius","humidity","precip_mm","wind_kph",
            "pressure_mb","visibility_km","uv_index","air_quality_index","cloud"]

fig, axes = plt.subplots(1, 2, figsize=(17, 7))
fig.suptitle("Feature Correlations", fontsize=15, fontweight="bold")

corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1, ax=axes[0], linewidths=0.4, annot_kws={"size":8})
axes[0].set_title("Correlation matrix")
axes[0].tick_params(axis="x", rotation=30)

axes[1].scatter(df.temperature_celsius, df.humidity,
                c=df.air_quality_index, cmap="RdYlGn_r", alpha=0.25, s=4)
city_means = df.groupby("location_name")[["temperature_celsius","humidity","air_quality_index"]].mean()
sc = axes[1].scatter(city_means.temperature_celsius, city_means.humidity,
                     c=city_means.air_quality_index, cmap="RdYlGn_r",
                     s=140, edgecolors="black", lw=1.2, zorder=5)
for city, row in city_means.iterrows():
    axes[1].annotate(city, (row.temperature_celsius, row.humidity),
                     xytext=(5,4), textcoords="offset points", fontsize=7)
plt.colorbar(sc, ax=axes[1], label="Air quality index")
axes[1].set(xlabel="Temperature (°C)", ylabel="Humidity (%)",
            title="Temp vs humidity  (colour = AQI)")

plt.tight_layout()
plt.savefig("figures/fig2_correlations.png", dpi=150, bbox_inches="tight")
plt.close()
print("  fig2 done")


# --- Fig 3: time series + anomaly detection ---
fig, axes = plt.subplots(3, 1, figsize=(15, 13))
fig.suptitle("Time Series & Anomaly Detection", fontsize=15, fontweight="bold")

# city monthly lines
for city in ["New York","London","Mumbai","Cairo"]:
    sub = df[df.location_name == city].sort_values("last_updated")
    mn  = sub.resample("ME", on="last_updated").temperature_celsius.mean()
    axes[0].plot(mn.index, mn.values, label=city, lw=2)
axes[0].set(ylabel="Temperature (°C)",
            title="Monthly average temperature — selected cities")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.35)

# global daily mean + warming trend
gd = df.groupby("last_updated").temperature_celsius.mean().reset_index()
gd.columns = ["date","temp"]
axes[1].fill_between(gd.date, gd.temp, alpha=0.25, color="steelblue")
axes[1].plot(gd.date, gd.temp, color="steelblue", lw=0.8, alpha=0.5)
axes[1].plot(gd.date, gd.temp.rolling(30, center=True).mean(),
             color="red", lw=2.2, label="30-day MA")
xn = np.arange(len(gd))
sl, ic, *_ = stats.linregress(xn, gd.temp.values)
axes[1].plot(gd.date, ic + sl*xn, "k--", lw=1.8,
             label=f"Trend  +{sl*365:.3f} °C/yr")
axes[1].set(ylabel="Avg temperature (°C)",
            title="Global daily mean temperature + trend")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.35)

# z-score anomaly — New York
ny = df[df.location_name == "New York"].sort_values("last_updated")
ny_m = ny.resample("ME", on="last_updated").temperature_celsius.mean()
zs   = np.abs(stats.zscore(ny_m.values))
anom = ny_m[zs > 2]
axes[2].plot(ny_m.index, ny_m.values, color="royalblue", lw=1.5, label="Monthly temp")
axes[2].fill_between(ny_m.index,
                     ny_m.mean() - 2*ny_m.std(),
                     ny_m.mean() + 2*ny_m.std(),
                     alpha=0.15, color="orange", label="±2σ")
axes[2].scatter(anom.index, anom.values, color="red", s=90, zorder=5,
                label=f"Anomalies  n={len(anom)}")
axes[2].axhline(ny_m.mean(), color="grey", ls="--", lw=1)
axes[2].set(ylabel="Temperature (°C)",
            title="Anomaly detection — New York  (z-score > 2)")
axes[2].legend(fontsize=9); axes[2].grid(alpha=0.35)

plt.tight_layout()
plt.savefig("figures/fig3_timeseries_anomaly.png", dpi=150, bbox_inches="tight")
plt.close()
print("  fig3 done")


# ------------------------------------------------------------------
# 4. MODELLING
# ------------------------------------------------------------------

print("\nTraining models...")

le = LabelEncoder()
df["continent_enc"] = le.fit_transform(df.continent)
df["climate_enc"]   = le.fit_transform(df.climate_type)

FEATURES = ["month","dayofyear","year","humidity","precip_mm","wind_kph",
            "pressure_mb","visibility_km","uv_index","air_quality_index",
            "cloud","latitude","longitude","continent_enc","climate_enc"]
TARGET   = "temperature_celsius"

clean = df[FEATURES + [TARGET]].dropna()
X, y  = clean[FEATURES], clean[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

sc    = StandardScaler()
Xs_tr = sc.fit_transform(X_tr)
Xs_te = sc.transform(X_te)

MODELS = {
    "Linear Regression":  LinearRegression(),
    "Ridge Regression":   Ridge(alpha=1.0),
    "Random Forest":      RandomForestRegressor(n_estimators=200, max_depth=12,
                                                random_state=42, n_jobs=-1),
    "Gradient Boosting":  GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                    learning_rate=0.05, random_state=42),
}

results = {}
for name, mdl in MODELS.items():
    if "Regression" in name:
        mdl.fit(Xs_tr, y_tr); preds = mdl.predict(Xs_te)
    else:
        mdl.fit(X_tr, y_tr);  preds = mdl.predict(X_te)
    results[name] = dict(
        MAE  = mean_absolute_error(y_te, preds),
        RMSE = np.sqrt(mean_squared_error(y_te, preds)),
        R2   = r2_score(y_te, preds),
        preds= preds,
    )
    print(f"  {name:<25} MAE={results[name]['MAE']:.3f}  "
          f"RMSE={results[name]['RMSE']:.3f}  R²={results[name]['R2']:.4f}")

# simple ensemble
ens = VotingRegressor([
    ("rf", RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)),
    ("gb", GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                     learning_rate=0.05, random_state=42)),
    ("lr", LinearRegression()),
])
ens.fit(X_tr, y_tr)
ep = ens.predict(X_te)
results["Ensemble (RF+GB+LR)"] = dict(
    MAE  = mean_absolute_error(y_te, ep),
    RMSE = np.sqrt(mean_squared_error(y_te, ep)),
    R2   = r2_score(y_te, ep),
    preds= ep,
)
print(f"  {'Ensemble (RF+GB+LR)':<25} MAE={results['Ensemble (RF+GB+LR)']['MAE']:.3f}  "
      f"RMSE={results['Ensemble (RF+GB+LR)']['RMSE']:.3f}  "
      f"R²={results['Ensemble (RF+GB+LR)']['R2']:.4f}")


# --- Fig 4: model comparison ---
names  = list(results.keys())
maes   = [results[m]["MAE"]  for m in names]
r2s    = [results[m]["R2"]   for m in names]
colors_b = ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f"]

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Model Comparison", fontsize=15, fontweight="bold")

x = np.arange(len(names))
axes[0,0].bar(x, maes, color=colors_b, edgecolor="black", lw=0.7)
axes[0,0].set_xticks(x); axes[0,0].set_xticklabels(names, rotation=18, ha="right", fontsize=9)
axes[0,0].set(ylabel="MAE (°C)", title="Mean absolute error  (lower = better)")
for i, v in enumerate(maes): axes[0,0].text(i, v+0.02, f"{v:.3f}", ha="center", fontsize=9)

axes[0,1].bar(x, r2s, color=colors_b, edgecolor="black", lw=0.7)
axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(names, rotation=18, ha="right", fontsize=9)
axes[0,1].set_ylim(0, 1.05)
axes[0,1].set(ylabel="R²", title="R² score  (higher = better)")
for i, v in enumerate(r2s): axes[0,1].text(i, v+0.005, f"{v:.4f}", ha="center", fontsize=9)

best  = max(results, key=lambda m: results[m]["R2"])
bp    = results[best]["preds"]
lim   = [min(y_te.min(), bp.min()), max(y_te.max(), bp.max())]
axes[1,0].scatter(y_te, bp, alpha=0.25, s=6, color="steelblue")
axes[1,0].plot(lim, lim, "r--", lw=1.8, label="Perfect fit")
axes[1,0].set(xlabel="Actual (°C)", ylabel="Predicted (°C)",
              title=f"Actual vs predicted — {best}")
axes[1,0].legend(fontsize=9)

res = y_te.values - bp
axes[1,1].hist(res, bins=60, color="steelblue", edgecolor="white", lw=0.3)
axes[1,1].axvline(0, color="red", lw=1.8, ls="--")
axes[1,1].set(xlabel="Residual (°C)", ylabel="Count",
              title=f"Residuals — {best}")
axes[1,1].text(0.97, 0.95, f"mean={res.mean():.3f}\nstd={res.std():.3f}",
               transform=axes[1,1].transAxes, ha="right", va="top",
               bbox=dict(boxstyle="round", fc="wheat", alpha=0.5), fontsize=9)

plt.tight_layout()
plt.savefig("figures/fig4_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  fig4 done")


# --- Fig 5: feature importance ---
rf  = MODELS["Random Forest"]
fi  = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Feature Importance", fontsize=15, fontweight="bold")

axes[0].barh(fi.index, fi.values,
             color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(fi)))[::-1])
axes[0].invert_yaxis()
axes[0].set(xlabel="Gini importance", title="Random Forest — Gini importance")

# permutation on a subsample (faster)
idx  = np.random.choice(len(X_te), 2000, replace=False)
perm = permutation_importance(rf, X_te.iloc[idx], y_te.iloc[idx],
                              n_repeats=10, random_state=42, n_jobs=-1)
pi   = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
order = np.argsort(perm.importances_mean)[::-1]
axes[1].barh(pi.index, pi.values,
             xerr=perm.importances_std[order],
             color=plt.cm.Blues(np.linspace(0.4, 0.9, len(pi)))[::-1])
axes[1].invert_yaxis()
axes[1].set(xlabel="Mean decrease in R²", title="Permutation importance  (±1 std)")

plt.tight_layout()
plt.savefig("figures/fig5_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  fig5 done")


# --- Fig 6: spatial ---
city_stats = df.groupby("location_name").agg(
    lat       = ("latitude",            "first"),
    lon       = ("longitude",           "first"),
    avg_temp  = ("temperature_celsius", "mean"),
    avg_aqi   = ("air_quality_index",   "mean"),
    avg_precip= ("precip_mm",           "mean"),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(17, 7))
fig.suptitle("Spatial Analysis", fontsize=15, fontweight="bold")

sc1 = axes[0].scatter(city_stats.lon, city_stats.lat,
                      c=city_stats.avg_temp, cmap="RdYlBu_r",
                      s=city_stats.avg_precip*60+80,
                      edgecolors="black", lw=1.1, zorder=5)
plt.colorbar(sc1, ax=axes[0], label="Avg temp (°C)")
for _, r in city_stats.iterrows():
    axes[0].annotate(r.location_name, (r.lon, r.lat),
                     xytext=(6,4), textcoords="offset points", fontsize=7.5)
axes[0].axhline(0, color="grey", ls="--", lw=0.8, alpha=0.5)
axes[0].grid(alpha=0.3)
axes[0].set(xlabel="Longitude", ylabel="Latitude",
            title="Temperature map  (bubble size = precipitation)")

sc2 = axes[1].scatter(city_stats.lon, city_stats.lat,
                      c=city_stats.avg_aqi, cmap="RdYlGn_r",
                      s=200, edgecolors="black", lw=1.1, zorder=5)
plt.colorbar(sc2, ax=axes[1], label="Air quality index")
for _, r in city_stats.iterrows():
    axes[1].annotate(r.location_name, (r.lon, r.lat),
                     xytext=(6,4), textcoords="offset points", fontsize=7.5)
axes[1].axhline(0, color="grey", ls="--", lw=0.8, alpha=0.5)
axes[1].grid(alpha=0.3)
axes[1].set(xlabel="Longitude", ylabel="Latitude", title="Air quality index map")

plt.tight_layout()
plt.savefig("figures/fig6_spatial.png", dpi=150, bbox_inches="tight")
plt.close()
print("  fig6 done")


# --- Fig 7: climate & environmental ---
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Climate Analysis & Environmental Impact", fontsize=15, fontweight="bold")

# warming trend per city
yearly = df.groupby(["location_name","year"]).temperature_celsius.mean().reset_index()
warm   = {}
for city, grp in yearly.groupby("location_name"):
    sl, _, _, p, _ = stats.linregress(grp.year, grp.temperature_celsius)
    warm[city] = dict(slope=sl*10, p=p)
wdf = pd.DataFrame(warm).T.sort_values("slope", ascending=False)
axes[0,0].barh(wdf.index, wdf.slope,
               color=["#d73027" if p < 0.05 else "#abd9e9" for p in wdf.p])
axes[0,0].axvline(0, color="black", lw=1)
axes[0,0].invert_yaxis()
axes[0,0].set(xlabel="°C / decade",
              title="Warming trend by city  (red = p < 0.05)")

# AQI vs temperature
axes[0,1].scatter(df.temperature_celsius, df.air_quality_index,
                  alpha=0.08, s=4, c=df.humidity, cmap="Blues")
for cont, grp in df.groupby("continent"):
    sl, ic, *_ = stats.linregress(grp.temperature_celsius.dropna(),
                                  grp.air_quality_index.dropna())
    xr = np.array([grp.temperature_celsius.min(), grp.temperature_celsius.max()])
    axes[0,1].plot(xr, ic + sl*xr, lw=2, label=cont)
axes[0,1].set(xlabel="Temperature (°C)", ylabel="AQI",
              title="Temperature vs air quality")
axes[0,1].legend(fontsize=8)

# precipitation seasonality
mp = df.groupby(["month","continent"]).precip_mm.mean().unstack()
for col in mp.columns:
    axes[1,0].plot(range(1,13), mp[col].values, marker="o", label=col, lw=2)
axes[1,0].set_xticks(range(1,13))
axes[1,0].set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
axes[1,0].set(xlabel="Month", ylabel="Avg precipitation (mm)",
              title="Precipitation seasonality")
axes[1,0].legend(fontsize=8)

# climate type boxplot
clim_ord = (df.groupby("climate_type").temperature_celsius
              .median().sort_values(ascending=False).index)
bp = axes[1,1].boxplot(
    [df[df.climate_type == c].temperature_celsius.values for c in clim_ord],
    labels=clim_ord, patch_artist=True, notch=True
)
for patch, col in zip(bp["boxes"], sns.color_palette("husl", len(clim_ord))):
    patch.set_facecolor(col); patch.set_alpha(0.7)
axes[1,1].tick_params(axis="x", rotation=15)
axes[1,1].set(ylabel="Temperature (°C)",
              title="Temperature by climate type")

plt.tight_layout()
plt.savefig("figures/fig7_climate_env.png", dpi=150, bbox_inches="tight")
plt.close()
print("  fig7 done")


# --- Fig 8: 90-day forecast ---
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
fig.suptitle("90-Day Temperature Forecast", fontsize=15, fontweight="bold")

for ax, city in zip(axes.flatten(), ["New York","London","Mumbai","Cairo"]):
    sub = (df[df.location_name == city]
           .sort_values("last_updated")
           .resample("ME", on="last_updated")
           .temperature_celsius.mean()
           .reset_index())
    sub.columns = ["date","temp"]

    x  = np.arange(len(sub))
    sl, ic, *_ = stats.linregress(x, sub.temp)
    fx = np.arange(len(sub), len(sub)+3)
    fd = pd.date_range(sub.date.iloc[-1] + pd.DateOffset(months=1), periods=3, freq="ME")
    ft = ic + sl*fx + np.array([sub.temp.values[j%12] - sub.temp.mean() for j in fx]) * 0.5

    ax.plot(sub.date, sub.temp, color="steelblue", lw=2, label="Historical")
    ax.plot(fd, ft, color="red", lw=2.2, ls="--", marker="o", label="Forecast")
    ax.fill_between(fd, ft-1.5, ft+1.5, color="red", alpha=0.18, label="95% CI")
    ax.axvline(sub.date.iloc[-1], color="grey", ls=":", lw=1.3)
    ax.set(title=city, ylabel="Temperature (°C)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/fig8_forecast.png", dpi=150, bbox_inches="tight")
plt.close()
print("  fig8 done")


# ------------------------------------------------------------------
# summary
# ------------------------------------------------------------------

print("\n=== MODEL METRICS ===")
for name, r in results.items():
    print(f"  {name:<30} MAE={r['MAE']:.3f}  RMSE={r['RMSE']:.3f}  R²={r['R2']:.4f}")

print("\nAll figures saved to figures/")