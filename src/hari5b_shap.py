import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import randint, uniform, binomtest, mannwhitneyu
from xgboost import XGBClassifier

Path("outputs/figures").mkdir(parents=True, exist_ok=True)
Path("outputs/tables").mkdir(parents=True, exist_ok=True)
Path("outputs/models_klasifikasi").mkdir(parents=True, exist_ok=True)

DOMESTIK = ["usdidr_lag1", "indonia_lag1", "bi_rate_lag1"]
GLOBAL = ["sp500_lag1", "oil_wti_lag1", "dxy_lag1", "vix_lag1",
          "ust10y_lag1", "fed_funds_rate_lag1", "epu_lag1"]
MACRO_COLS = DOMESTIK + GLOBAL

# ============================================================
# Muat data, target biner (1=naik, 0=turun/tetap). Fitur autoregresif
# TIDAK dipakai di sini -- terbukti menurunkan signifikansi RF/XGBoost
# di eksperimen Hari 5a, jadi model utama tetap makro-saja.
# ============================================================
df = pd.read_csv("data_processed/dataset_final.csv", index_col=0, parse_dates=True).loc["2021-01-01":]
X = df[MACRO_COLS]
y_dir = (df["return_ihsg"] > 0).astype(int)

train_mask = df.index <= "2024-12-31"
test_mask = df.index >= "2025-01-01"
X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y_dir[train_mask], y_dir[test_mask]
print(f"train: {len(X_train)}, test: {len(X_test)}")

# ============================================================
# Tuning + fit ketiga model klasifikasi (mirror struktur Hari 3,
# scoring="accuracy" karena itu metrik yang benar-benar dilaporkan)
# ============================================================
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

search_xgb = RandomizedSearchCV(
    XGBClassifier(random_state=42, eval_metric="logloss"), xgb_dist, n_iter=60, cv=tscv,
    scoring="accuracy", random_state=42, n_jobs=-1
).fit(X_train, y_train)
search_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42), rf_dist, n_iter=50, cv=tscv,
    scoring="accuracy", random_state=42, n_jobs=-1
).fit(X_train, y_train)
logreg = LogisticRegression(max_iter=1000).fit(X_train, y_train)

fitted = {"logreg": logreg, "rf": search_rf.best_estimator_, "xgb": search_xgb.best_estimator_}
for name, model in fitted.items():
    joblib.dump(model, f"outputs/models_klasifikasi/{name}.joblib")
X_test.to_csv("outputs/tables/X_test_klasifikasi.csv")
y_test.to_csv("outputs/tables/y_test_klasifikasi.csv")

print("\nBest XGB:", search_xgb.best_params_)
print("Best RF:", search_rf.best_params_)

# ============================================================
# Evaluasi -- konfirmasi ulang angka Hari 5a (harus identik, seed sama)
# ============================================================
hasil = []
for name, model in fitted.items():
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    k, n = int((pred == y_test.values).sum()), len(y_test)
    p_binom = binomtest(k, n, 0.5, alternative="greater").pvalue
    hasil.append({"model": name, "akurasi": acc, "auc": auc, "k": k, "n": n, "p_binomial": p_binom})
    print(f"{name:8s} akurasi={acc:.4f}  AUC={auc:.4f}  k={k}/{n}  p_binom={p_binom:.4f}")

pd.DataFrame(hasil).to_csv("outputs/tables/performa_klasifikasi.csv", index=False)

# ============================================================
# SHAP untuk ketiga model klasifikasi
# ============================================================
masker_lr = shap.maskers.Independent(X_train, max_samples=len(X_train))
explainers = {
    "logreg": shap.LinearExplainer(fitted["logreg"], masker_lr),
    "rf": shap.TreeExplainer(fitted["rf"]),
    "xgb": shap.TreeExplainer(fitted["xgb"]),
}

shap_values = {}
for name, expl in explainers.items():
    sv = expl.shap_values(X_test)
    # TreeExplainer untuk classifier kadang mengembalikan list [kelas0, kelas1]
    # tergantung versi shap -- ambil kelas 1 (naik) secara eksplisit.
    if isinstance(sv, list):
        sv = sv[1]
    if sv.ndim == 3:  # sebagian versi shap: (n_samples, n_features, n_classes)
        sv = sv[:, :, 1]
    shap_values[name] = pd.DataFrame(sv, columns=X_test.columns, index=X_test.index)
    mean_abs = shap_values[name].abs().mean().sort_values(ascending=False)
    print(f"\nSHAP {name} -- 5 fitur teratas:\n{mean_abs.head()}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(mean_abs.index[::-1], mean_abs.values[::-1])
    ax.set_xlabel("Mean |SHAP value| (log-odds naik)")
    ax.set_title(f"Kepentingan fitur -- {name.upper()} (klasifikasi)")
    plt.tight_layout()
    plt.savefig(f"outputs/figures/shap_klasifikasi_importance_{name}.png", dpi=150)
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    shap.summary_plot(shap_values[name].values, X_test, show=False, plot_size=None)
    plt.tight_layout()
    plt.savefig(f"outputs/figures/shap_klasifikasi_summary_{name}.png", dpi=150)
    plt.close()

# ============================================================
# Rolling dominansi domestik-global + Mann-Whitney U (model utama: logreg)
# ============================================================
sv_main = shap_values["logreg"]
domestik_abs = sv_main[DOMESTIK].abs().sum(axis=1)
global_abs = sv_main[GLOBAL].abs().sum(axis=1)

roll_domestik = domestik_abs.rolling(20).mean()
roll_global = global_abs.rolling(20).mean()
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(roll_domestik.index, roll_domestik, label="Domestik (rolling 20 hari)", lw=1)
ax.plot(roll_global.index, roll_global, label="Global (rolling 20 hari)", lw=1)
ax.set_ylabel("Rata-rata |SHAP| bergulir (log-odds)")
ax.set_title("Dominansi SHAP domestik vs global -- model klasifikasi (Logistic Regression)")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/figures/shap_klasifikasi_rolling_domglobal.png", dpi=150)
plt.close()

vol20 = df["return_ihsg"].rolling(20).std()
vol_test = vol20.reindex(y_test.index)
mask_high_vol = (vol_test > vol_test.median()).values
diff_shap = (global_abs - domestik_abs).values
u_stat, p_value = mannwhitneyu(diff_shap[mask_high_vol], diff_shap[~mask_high_vol], alternative="two-sided")
n1, n2 = int(mask_high_vol.sum()), int((~mask_high_vol).sum())
r_rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
print(f"\nMann-Whitney U (klasifikasi, logreg): U={u_stat:.1f}, p={p_value:.4f}, r={r_rank_biserial:.4f}")

pd.DataFrame([{
    "model": "logreg", "U": u_stat, "p_value": p_value, "effect_size_r": r_rank_biserial,
    "n_tinggi": n1, "n_rendah": n2,
}]).to_csv("outputs/tables/mann_whitney_klasifikasi.csv", index=False)

print("\nSelesai. Bandingkan file outputs/figures/shap_klasifikasi_*.png dan "
      "outputs/tables/performa_klasifikasi.csv dengan hasil Hari 4 (model regresi).")