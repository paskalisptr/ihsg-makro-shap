import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import randint, uniform
from xgboost import XGBClassifier

Path("outputs/tables").mkdir(parents=True, exist_ok=True)

MACRO_COLS = ["usdidr_lag1", "sp500_lag1", "oil_wti_lag1", "dxy_lag1", "vix_lag1",
              "ust10y_lag1", "fed_funds_rate_lag1", "epu_lag1", "indonia_lag1", "bi_rate_lag1"]
N_LAG = 5  # jumlah lag return_ihsg sendiri yang dicoba sebagai fitur autoregresif

# ============================================================
# Muat data, tambah fitur autoregresif (lag return_ihsg sendiri).
# PENTING soal kebocoran data: return_ihsg_lag1 = return_ihsg.shift(1),
# yaitu return HARI SEBELUMNYA -- bukan return hari yang sama yang mau
# diprediksi. Ini prinsip sama seperti 10 fitur makro yang sudah di-lag,
# jadi tidak ada informasi masa depan yang bocor ke fitur.
# ============================================================
df = pd.read_csv("data_processed/dataset_final.csv", index_col=0, parse_dates=True).loc["2021-01-01":]

for lag in range(1, N_LAG + 1):
    df[f"return_ihsg_ar_lag{lag}"] = df["return_ihsg"].shift(lag)

AR_COLS = [f"return_ihsg_ar_lag{lag}" for lag in range(1, N_LAG + 1)]
df = df.dropna()  # buang baris awal yang belum punya lag5 lengkap

# Target biner: arah pergerakan (1 = naik, 0 = turun/tetap)
y_dir = (df["return_ihsg"] > 0).astype(int)

train_mask = df.index <= "2024-12-31"
test_mask = df.index >= "2025-01-01"
y_train, y_test = y_dir[train_mask], y_dir[test_mask]
print(f"train: {train_mask.sum()}, test: {test_mask.sum()}")
print(f"Baseline (selalu tebak kelas mayoritas) di test: {max(y_test.mean(), 1 - y_test.mean()):.4f}")

feature_sets = {
    "makro_saja": MACRO_COLS,
    "makro_plus_ar": MACRO_COLS + AR_COLS,
}

tscv = TimeSeriesSplit(n_splits=5)
xgb_dist = {
    "n_estimators": randint(50, 600), "max_depth": randint(2, 7),
    "learning_rate": uniform(0.01, 0.19), "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4), "reg_alpha": uniform(0, 1),
    "reg_lambda": uniform(0.5, 4.5),
}
rf_dist = {
    "n_estimators": randint(100, 600), "max_depth": [2, 3, 4, 5, 6, None],
    "min_samples_leaf": randint(1, 20), "max_features": ["sqrt", "log2", 0.5, 1.0],
}

hasil = []
for fs_name, cols in feature_sets.items():
    X_train, X_test = df.loc[train_mask, cols], df.loc[test_mask, cols]

    search_xgb = RandomizedSearchCV(
        XGBClassifier(random_state=42, eval_metric="logloss"), xgb_dist, n_iter=60, cv=tscv,
        scoring="accuracy", random_state=42, n_jobs=-1
    ).fit(X_train, y_train)
    search_rf = RandomizedSearchCV(
        RandomForestClassifier(random_state=42), rf_dist, n_iter=50, cv=tscv,
        scoring="accuracy", random_state=42, n_jobs=-1
    ).fit(X_train, y_train)
    logreg = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    models = {"logreg": logreg, "rf": search_rf.best_estimator_, "xgb": search_xgb.best_estimator_}
    for model_name, model in models.items():
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, proba)
        k, n = int((pred == y_test.values).sum()), len(y_test)
        p_binom = binomtest(k, n, 0.5, alternative="greater").pvalue
        hasil.append({
            "fitur": fs_name, "model": model_name, "akurasi": acc, "auc": auc,
            "k": k, "n": n, "p_binomial": p_binom,
            "signifikan": "ya" if p_binom < 0.05 else "tidak",
        })
        print(f"{fs_name:15s} {model_name:8s} akurasi={acc:.4f}  AUC={auc:.4f}  "
              f"k={k}/{n}  p_binom={p_binom:.4f}")
    if fs_name == "makro_plus_ar":
        print("Best XGB (makro+AR):", search_xgb.best_params_)
        print("Best RF  (makro+AR):", search_rf.best_params_)

hasil_df = pd.DataFrame(hasil)
hasil_df.to_csv("outputs/tables/klasifikasi_vs_regresi_akurasi.csv", index=False)
print("\n=== Ringkasan lengkap ===")
print(hasil_df.to_string(index=False))

# ============================================================
# Perbandingan apple-to-apple dengan baseline lama (regresi lalu ambil
# tanda). Baseline lama: XGBoost regresi, macro-saja, akurasi arah 57,1%
# (k=198/347) -- dari hari3b_modeling.py, TIDAK dihitung ulang di sini,
# cuma dikutip sebagai pembanding tetap.
# ============================================================
baseline_lama = {"pendekatan": "Regresi -> ambil tanda (XGBoost, makro-saja, hari3b)",
                  "akurasi": 198 / 347, "k": 198, "n": 347}
print(f"\nPembanding: {baseline_lama['pendekatan']} = {baseline_lama['akurasi']:.4f} "
      f"(k={baseline_lama['k']}/{baseline_lama['n']})")