import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import joblib
from pathlib import Path
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import randint, uniform
from xgboost import XGBRegressor

Path("outputs/figures").mkdir(parents=True, exist_ok=True)
Path("outputs/tables").mkdir(parents=True, exist_ok=True)
Path("outputs/models").mkdir(parents=True, exist_ok=True)

DOMESTIK = ["usdidr_lag1", "indonia_lag1", "bi_rate_lag1"]
GLOBAL = ["sp500_lag1", "oil_wti_lag1", "dxy_lag1", "vix_lag1",
          "ust10y_lag1", "fed_funds_rate_lag1", "epu_lag1"]

# Sama seperti hari3a_eda.py -- kalau ubah tanggal, ubah di kedua file.
CRISIS_WINDOWS = [
    ("2025-03-01", "2025-04-30"),
    ("2026-01-28", "2026-05-31"),
]

df = pd.read_csv("data_processed/dataset_final.csv", index_col=0, parse_dates=True).loc["2021-01-01":]
y, X = df["return_ihsg"], df.drop(columns="return_ihsg")

# ============================================================
# OLS (z-score standardized) + 5 uji asumsi + Wald test RQ1-RQ3
# ============================================================
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
X_ols = sm.add_constant(X_scaled)
ols = sm.OLS(y, X_ols).fit()
if het_breuschpagan(ols.resid, X_ols)[1] < 0.05:
    ols = sm.OLS(y, X_ols).fit(cov_type="HC3")

print(ols.summary())
print(f"Jarque-Bera p={jarque_bera(ols.resid)[1]:.4f} | Durbin-Watson={durbin_watson(ols.resid):.4f}")
print("VIF:\n", pd.Series(
    [variance_inflation_factor(X_ols.values, i) for i in range(X_ols.shape[1])],
    index=X_ols.columns))

print("\nRQ1 (domestik):", ols.f_test(", ".join(f"{v} = 0" for v in DOMESTIK)))
print("RQ2 (global):", ols.f_test(", ".join(f"{v} = 0" for v in GLOBAL)))
restr = " + ".join(f"(1/3)*{v}" for v in DOMESTIK) + " - " + " - ".join(f"(1/7)*{v}" for v in GLOBAL) + " = 0"
print("RQ3 (beda rata-rata domestik vs global):", ols.f_test(restr))

# ============================================================
# Walk-forward split: train <=2024, test >=2025
# ============================================================
train, test = df.loc[:"2024-12-31"], df.loc["2025-01-01":]
X_train, y_train = train[X.columns], train["return_ihsg"]
X_test, y_test = test[X.columns], test["return_ihsg"]
print(f"\ntrain: {X_train.shape}, test: {X_test.shape}")

# ============================================================
# Hyperparameter tuning -- RandomizedSearchCV + TimeSeriesSplit (5-fold)
# ============================================================
tscv = TimeSeriesSplit(n_splits=5)

xgb_dist = {
    "n_estimators": randint(50, 600),
    "max_depth": randint(2, 7),
    "learning_rate": uniform(0.01, 0.19),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "reg_alpha": uniform(0, 1),
    "reg_lambda": uniform(0.5, 4.5),
}
rf_dist = {
    "n_estimators": randint(100, 600),
    "max_depth": [2, 3, 4, 5, 6, None],
    "min_samples_leaf": randint(1, 20),
    "max_features": ["sqrt", "log2", 0.5, 1.0],
}

search_xgb = RandomizedSearchCV(
    XGBRegressor(random_state=42), xgb_dist, n_iter=80, cv=tscv,
    scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1
).fit(X_train, y_train)

search_rf = RandomizedSearchCV(
    RandomForestRegressor(random_state=42), rf_dist, n_iter=60, cv=tscv,
    scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1
).fit(X_train, y_train)

print("\nBest XGB:", search_xgb.best_params_, "CV RMSE:", -search_xgb.best_score_)
print("Best RF:", search_rf.best_params_, "CV RMSE:", -search_rf.best_score_)

fitted = {
    "lr": LinearRegression().fit(X_train, y_train),
    "rf": search_rf.best_estimator_,
    "xgb": search_xgb.best_estimator_,
}

# Simpan model terlatih + data test -- dipakai langsung oleh Hari 4 (SHAP)
# supaya tidak perlu tuning ulang tiap kali eksplorasi interpretabilitas.
for name, model in fitted.items():
    joblib.dump(model, f"outputs/models/{name}.joblib")
X_test.to_csv("outputs/tables/X_test.csv")
y_test.to_csv("outputs/tables/y_test.csv")
print("\nModel & X_test/y_test tersimpan di outputs/models & outputs/tables untuk Hari 4.")

# ============================================================
# Evaluasi keseluruhan + tersegmentasi (krisis vs normal, vol tinggi vs rendah)
# ============================================================
def eval_segment(y_true, y_pred, mask, label):
    if mask.sum() == 0:
        return None
    return {"segmen": label, "n": int(mask.sum()),
            "rmse": np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])),
            "mae": mean_absolute_error(y_true[mask], y_pred[mask]),
            "directional_accuracy": (np.sign(y_pred[mask]) == np.sign(y_true[mask])).mean()}


# FIX vs kode lama: mask krisis sekarang menggabungkan KEDUA episode
# (Mar-Apr 2025 dan Jan-Mei 2026), bukan cuma yang kedua.
mask_crisis = np.zeros(len(y_test), dtype=bool)
for start, end in CRISIS_WINDOWS:
    mask_crisis |= ((y_test.index >= start) & (y_test.index <= end))

vol20_full = df["return_ihsg"].rolling(20).std()
vol_test = vol20_full.reindex(y_test.index)
mask_high_vol = (vol_test > vol_test.median()).values

hasil_keseluruhan = [{"model": "naive",
                       "rmse": np.sqrt(mean_squared_error(y_test, np.zeros(len(y_test)))),
                       "mae": mean_absolute_error(y_test, np.zeros(len(y_test))),
                       "directional_accuracy": 0.0}]
preds = {}
for name, model in fitted.items():
    pred = model.predict(X_test)
    preds[name] = pred
    hasil_keseluruhan.append({"model": name,
                                "rmse": np.sqrt(mean_squared_error(y_test, pred)),
                                "mae": mean_absolute_error(y_test, pred),
                                "directional_accuracy": (np.sign(pred) == np.sign(y_test)).mean()})

hasil_df = pd.DataFrame(hasil_keseluruhan)
print("\nPerforma keseluruhan:\n", hasil_df)
hasil_df.to_csv("outputs/tables/perbandingan_model.csv", index=False)

segmen_df = pd.DataFrame([s for s in [
    eval_segment(y_test.values, preds["xgb"], mask_crisis, "Krisis (Mar-Apr 2025 & Jan-Mei 2026)"),
    eval_segment(y_test.values, preds["xgb"], ~mask_crisis, "Normal"),
    eval_segment(y_test.values, preds["xgb"], mask_high_vol, "Volatilitas tinggi"),
    eval_segment(y_test.values, preds["xgb"], ~mask_high_vol, "Volatilitas rendah"),
] if s is not None])
print("\nPerforma XGBoost tersegmentasi:\n", segmen_df)
segmen_df.to_csv("outputs/tables/performa_tersegmentasi.csv", index=False)


def mark_crisis(ax):
    for start, end in CRISIS_WINDOWS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.15, color="red")


# ============================================================
# Plot aktual vs prediksi -- full & zoom kedua episode krisis
# ============================================================
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(y_test.index, y_test.values, label="Aktual", lw=0.8, color="black")
ax.plot(y_test.index, preds["xgb"], label="Prediksi XGBoost", lw=0.8, alpha=0.7)
mark_crisis(ax)
ax.legend()
plt.tight_layout()
plt.savefig("outputs/figures/prediksi_vs_aktual_full.png", dpi=150)
plt.close()

# FIX vs kode lama: window zoom lama (Des 2025 - Jun 2026) melewatkan
# episode Mar-Apr 2025 sepenuhnya. Dilebarkan ke mulai Feb 2025.
mask_zoom = (y_test.index >= "2025-02-01") & (y_test.index <= "2026-06-30")
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(y_test.index[mask_zoom], y_test.values[mask_zoom], label="Aktual",
         lw=1.2, color="black", marker="o", ms=2)
ax.plot(y_test.index[mask_zoom], preds["xgb"][mask_zoom], label="Prediksi XGBoost",
         lw=1.2, alpha=0.8, marker="o", ms=2)
mark_crisis(ax)
ax.legend()
plt.tight_layout()
plt.savefig("outputs/figures/prediksi_vs_aktual_zoom_krisis.png", dpi=150)
plt.close()

# ============================================================
# Rolling directional accuracy -- lebih representatif untuk temuan utama
# (57%, naik jadi 58% saat krisis) dibanding overlay magnitude di atas,
# yang terlihat "flat" karena model MSE-optimal memang menyusut ke arah
# rata-rata saat rasio sinyal-noise rendah (R2 OLS cuma 0,064).
# ============================================================
rolling_hit = pd.Series(
    (np.sign(preds["xgb"]) == np.sign(y_test.values)).astype(int),
    index=y_test.index
).rolling(20).mean()

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(rolling_hit.index, rolling_hit, lw=1, color="darkgreen")
ax.axhline(0.5, color="grey", ls="--", lw=0.8, label="Tebak acak (50%)")
mark_crisis(ax)
ax.set_ylabel("Directional accuracy bergulir (20 hari)")
ax.set_title("Akurasi arah prediksi XGBoost dari waktu ke waktu")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/figures/rolling_directional_accuracy_xgb.png", dpi=150)
plt.close()