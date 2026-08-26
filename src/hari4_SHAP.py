import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error

Path("outputs/figures").mkdir(parents=True, exist_ok=True)
Path("outputs/tables").mkdir(parents=True, exist_ok=True)

DOMESTIK = ["usdidr_lag1", "indonia_lag1", "bi_rate_lag1"]
GLOBAL = ["sp500_lag1", "oil_wti_lag1", "dxy_lag1", "vix_lag1",
          "ust10y_lag1", "fed_funds_rate_lag1", "epu_lag1"]

# Anotasi visual di grafik SAJA -- bukan dasar uji statistik.
# Uji statistik (Mann-Whitney U di bawah) pakai split volatilitas dari EDA,
# sesuai kesepakatan: klaim kuantitatif = data-driven, tanggal = ilustratif.
CRISIS_WINDOWS = [
    ("2025-03-01", "2025-04-30"),
    ("2026-01-28", "2026-05-31"),
]

# ============================================================
# Muat data & model dari Hari 3 (split train/test direkonstruksi ulang --
# deterministik berbasis tanggal, jadi identik dengan hari3b_modeling.py)
# ============================================================
df = pd.read_csv("data_processed/dataset_final.csv", index_col=0, parse_dates=True).loc["2021-01-01":]
y, X = df["return_ihsg"], df.drop(columns="return_ihsg")
train = df.loc[:"2024-12-31"]
X_train, y_train = train[X.columns], train["return_ihsg"]

X_test = pd.read_csv("outputs/tables/X_test.csv", index_col=0, parse_dates=True)
y_test = pd.read_csv("outputs/tables/y_test.csv", index_col=0, parse_dates=True).iloc[:, 0]

fitted = {name: joblib.load(f"outputs/models/{name}.joblib") for name in ["lr", "rf", "xgb"]}

# ============================================================
# SHAP per model
# ============================================================
masker_lr = shap.maskers.Independent(X_train, max_samples=len(X_train))
explainers = {
    "lr": shap.LinearExplainer(fitted["lr"], masker_lr),
    "rf": shap.TreeExplainer(fitted["rf"]),
    "xgb": shap.TreeExplainer(fitted["xgb"]),
}
print("max_samples masker:", masker_lr.max_samples)

shap_values = {}
for name, expl in explainers.items():
    sv = expl.shap_values(X_test)
    shap_values[name] = pd.DataFrame(sv, columns=X_test.columns, index=X_test.index)
    mean_abs = shap_values[name].abs().mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(mean_abs.index[::-1], mean_abs.values[::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Kepentingan fitur -- {name.upper()}")
    plt.tight_layout()
    plt.savefig(f"outputs/figures/shap_importance_{name}.png", dpi=150)
    plt.close()

    print(f"\nSHAP {name} -- 5 fitur teratas:\n{mean_abs.head()}")

    # Summary/beeswarm plot -- menunjukkan ARAH pengaruh (bukan cuma besaran).
    # Wajib untuk paper interpretabilitas: penguji akan tanya "S&P500 naik,
    # IHSG naik atau turun?" -- bar chart mean|SHAP| di atas tidak menjawab itu.
    fig, ax = plt.subplots(figsize=(7, 5))
    shap.summary_plot(shap_values[name].values, X_test, show=False, plot_size=None)
    plt.tight_layout()
    plt.savefig(f"outputs/figures/shap_summary_{name}.png", dpi=150)
    plt.close()

# ============================================================
# Rolling-window SHAP: dominansi global vs domestik dari waktu ke waktu
# Model utama: XGBoost (RMSE test terbaik, directional accuracy setara RF)
# ============================================================
sv_xgb = shap_values["xgb"]
domestik_abs = sv_xgb[DOMESTIK].abs().sum(axis=1)
global_abs = sv_xgb[GLOBAL].abs().sum(axis=1)

WINDOW = 20
roll_domestik = domestik_abs.rolling(WINDOW).mean()
roll_global = global_abs.rolling(WINDOW).mean()

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(roll_domestik.index, roll_domestik, label="Domestik (rolling 20 hari)", lw=1)
ax.plot(roll_global.index, roll_global, label="Global (rolling 20 hari)", lw=1)
for start, end in CRISIS_WINDOWS:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.12, color="red")
ax.set_ylabel("Rata-rata |SHAP| bergulir")
ax.set_title("Dominansi SHAP domestik vs global (merah = periode ilustratif, bukan dasar uji)")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/figures/shap_rolling_domestik_vs_global.png", dpi=150)
plt.close()

# ============================================================
# Mann-Whitney U -- uji utama RQ4, berbasis split volatilitas (objektif, EDA-driven)
# H0: distribusi (|SHAP_global| - |SHAP_domestik|) sama antara periode
#     volatilitas tinggi vs rendah pada data uji. Independent samples,
#     non-parametrik (SHAP tidak dijamin normal).
# ============================================================
vol20 = df["return_ihsg"].rolling(20).std()
vol_test = vol20.reindex(y_test.index)
mask_high_vol = (vol_test > vol_test.median()).values

diff_shap = (global_abs - domestik_abs).values
u_stat, p_value = mannwhitneyu(
    diff_shap[mask_high_vol], diff_shap[~mask_high_vol], alternative="two-sided"
)
print(f"\nMann-Whitney U (dominansi global-domestik, vol tinggi vs rendah): "
      f"U={u_stat:.1f}, p={p_value:.4f}")

n1, n2 = int(mask_high_vol.sum()), int((~mask_high_vol).sum())
# Rank-biserial correlation -- effect size untuk Mann-Whitney U, supaya
# p=0.70 dilaporkan bukan cuma "tidak signifikan" tapi juga "seberapa kecil".
r_rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

mwu_result = pd.DataFrame([{
    "pembanding": "vol_tinggi_vs_rendah",
    "U": u_stat,
    "p_value": p_value,
    "effect_size_r": r_rank_biserial,
    "median_diff_vol_tinggi": float(np.median(diff_shap[mask_high_vol])),
    "median_diff_vol_rendah": float(np.median(diff_shap[~mask_high_vol])),
    "n_tinggi": n1,
    "n_rendah": n2,
}])
mwu_result.to_csv("outputs/tables/mann_whitney_shap.csv", index=False)
print(mwu_result.T)

# ============================================================
# Ablation study: domestik-saja vs global-saja vs penuh
# Hyperparameter DISAMAKAN dengan model penuh hasil tuning Hari 3
# (clone dari model tersimpan) -- supaya isolasi efek fitur, bukan efek tuning.
# ============================================================
feature_sets = {"domestik_saja": DOMESTIK, "global_saja": GLOBAL, "penuh": list(X.columns)}
ablation_rows = []
for fs_name, cols in feature_sets.items():
    for model_name in ["rf", "xgb"]:
        model = clone(fitted[model_name])
        model.fit(X_train[cols], y_train)
        pred = model.predict(X_test[cols])
        ablation_rows.append({
            "fitur": fs_name,
            "model": model_name,
            "rmse": np.sqrt(mean_squared_error(y_test, pred)),
            "mae": mean_absolute_error(y_test, pred),
            "directional_accuracy": (np.sign(pred) == np.sign(y_test)).mean(),
        })

ablation_df = pd.DataFrame(ablation_rows)
print("\nHasil ablation:\n", ablation_df)
ablation_df.to_csv("outputs/tables/ablation_study.csv", index=False)