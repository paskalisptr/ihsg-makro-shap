import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import binomtest
from sklearn.metrics import mean_squared_error, mean_absolute_error

Path("outputs/tables").mkdir(parents=True, exist_ok=True)

MACRO_COLS = ["usdidr_lag1", "sp500_lag1", "oil_wti_lag1", "dxy_lag1", "vix_lag1",
              "ust10y_lag1", "fed_funds_rate_lag1", "epu_lag1", "indonia_lag1", "bi_rate_lag1"]

df = pd.read_csv("data_processed/dataset_final.csv", index_col=0, parse_dates=True).loc["2021-01-01":]
y = df["return_ihsg"]
X = df[MACRO_COLS]

train_mask = df.index <= "2024-12-31"
test_mask = df.index >= "2025-01-01"
y_train, y_test = y[train_mask], y[test_mask]
X_train, X_test = X[train_mask], X[test_mask]
print(f"train: {len(y_train)}, test: {len(y_test)}")

# ============================================================
# Pemilihan orde (p,d,q) via AIC pada data latih. d=0 karena
# return_ihsg sudah terbukti stasioner (uji ADF, Hari 3a) --
# differencing tambahan justru akan merusak interpretasi.
# Grid kecil (p,q <= 3) supaya tidak overfit ke data latih yang
# cuma 969 observasi -- ARIMA orde tinggi di data sekecil ini
# berisiko tidak stabil.
# ============================================================
print("\n=== Pemilihan orde ARIMAX via AIC ===")
best_aic, best_order = np.inf, None
for p in range(4):
    for q in range(4):
        if p == 0 and q == 0:
            continue
        try:
            model = SARIMAX(y_train, exog=X_train, order=(p, 0, q),
                             enforce_stationarity=True, enforce_invertibility=True)
            fit = model.fit(disp=False)
            print(f"ARIMAX({p},0,{q})  AIC={fit.aic:.2f}")
            if fit.aic < best_aic:
                best_aic, best_order = fit.aic, (p, 0, q)
        except Exception as e:
            print(f"ARIMAX({p},0,{q})  GAGAL: {e}")

print(f"\nOrde terbaik: ARIMAX{best_order}, AIC={best_aic:.2f}")

# ============================================================
# Fit ulang model terbaik di seluruh data latih, forecast satu-langkah
# berulang di data uji (exog test diberikan penuh -- ini konsisten
# dengan setup ML kita: prediksi t+1 pakai fitur makro t, bukan
# multi-step forecast tanpa exog).
# ============================================================
final_model = SARIMAX(y_train, exog=X_train, order=best_order,
                       enforce_stationarity=True, enforce_invertibility=True)
final_fit = final_model.fit(disp=False)
print("\n", final_fit.summary())

pred = final_fit.get_forecast(steps=len(y_test), exog=X_test).predicted_mean.values

rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
dir_acc = (np.sign(pred) == np.sign(y_test.values)).mean()
k, n = int((np.sign(pred) == np.sign(y_test.values)).sum()), len(y_test)
p_binom = binomtest(k, n, 0.5, alternative="greater").pvalue

print(f"\nARIMAX{best_order}: RMSE={rmse:.6f}  MAE={mae:.6f}  "
      f"akurasi_arah={dir_acc:.4f} (k={k}/{n})  p_binom={p_binom:.4f}")

pd.DataFrame([{
    "model": f"ARIMAX{best_order}", "aic": best_aic, "rmse": rmse, "mae": mae,
    "directional_accuracy": dir_acc, "k": k, "n": n, "p_binomial": p_binom,
}]).to_csv("outputs/tables/arimax_hasil.csv", index=False)

print("\nPembanding (dari hasil sebelumnya):")
print("  XGBoost regresi (hari3b)       : akurasi=0.5706, k=198/347")
print("  Logistic Regression klasifikasi: akurasi=0.5994, k=208/347")
