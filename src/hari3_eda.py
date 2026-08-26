import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("outputs/figures").mkdir(parents=True, exist_ok=True)
Path("outputs/tables").mkdir(parents=True, exist_ok=True)

# Dua episode volatilitas 2025-2026 yang relevan untuk Pembahasan.
# Tanggal Mar-Apr 2025 diverifikasi via web search: trading halt IHSG
# terjadi 18 Maret 2025 dan 8 & 11 April 2025 (tarif resiprokal Trump,
# diumumkan 2 April 2025). Persempit rentang ini kalau perlu presisi lebih.
CRISIS_WINDOWS = [
    ("2025-03-01", "2025-04-30"),  # tarif Trump "Liberation Day" + Danantara
    ("2026-01-28", "2026-05-31"),  # MSCI reclass / BI rate darurat
]

df = pd.read_csv("data_processed/dataset_final.csv", index_col=0, parse_dates=True).loc["2021-01-01":]
print("Dataset final:", df.shape, df.index.min(), "s/d", df.index.max())

y = df["return_ihsg"]
X = df.drop(columns="return_ihsg")

# ============================================================
# Statistik deskriptif
# ============================================================
print("\n=== Statistik deskriptif ===")
print(df.describe().T)

# ============================================================
# Uji Augmented Dickey-Fuller (stasioneritas)
# H0: ada unit root (non-stasioner). p<0.05 -> tolak H0 -> stasioner.
# WAJIB sebelum OLS: regresi antar variabel non-stasioner berisiko
# spurious regression (Granger & Newbold, 1974 -- verifikasi sitasi
# persis sebelum dikutip di naskah).
# ============================================================
from statsmodels.tsa.stattools import adfuller

print("\n=== Uji Augmented Dickey-Fuller (stasioneritas) ===")
adf_rows = []
for col in df.columns:
    stat, pval, *_ = adfuller(df[col].dropna())
    status = "Stasioner" if pval < 0.05 else "TIDAK STASIONER"
    adf_rows.append({"variabel": col, "adf_stat": stat, "p_value": pval, "status": status})
    print(f"{col:22s} ADF={stat:8.3f}  p={pval:.4f}  -> {status}")

adf_df = pd.DataFrame(adf_rows)
adf_df.to_csv("outputs/tables/uji_adf.csv", index=False)
if (adf_df["status"] == "TIDAK STASIONER").any():
    print("\nPERINGATAN: ada variabel tidak stasioner -- OLS/RQ1-3 berisiko "
          "spurious regression, perlu differencing tambahan sebelum lanjut.")

print("\n=== Uji ADF (stasioneritas) ===")
# Wajib untuk regresi time-series: data non-stasioner berisiko spurious
# regression (hubungan signifikan yang cuma kebetulan, bukan riil).
# Ekspektasi: 9 fitur sudah dalam bentuk diff harian (mean~0) -> harusnya
# stasioner otomatis. vix_lag1 dipertahankan dalam level (bukan diff) --
# ini SATU-SATUNYA fitur yang perlu dibuktikan stasioner secara empiris,
# bukan diasumsikan dari teori (VIX secara teori mean-reverting).
for col in df.columns:
    stat, pval, *_ = adfuller(df[col].dropna())
    status = "Stasioner" if pval < 0.05 else "TIDAK STASIONER (unit root)"
    print(f"{col:25s} ADF stat={stat:8.3f}  p={pval:.4f}  -> {status}")

vol20 = y.rolling(20).std()


def mark_crisis(ax):
    for start, end in CRISIS_WINDOWS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.2, color="red")


# ============================================================
# Return & volatilitas bergulir, dengan kedua episode krisis ditandai
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axes[0].plot(df.index, y, lw=0.6)
mark_crisis(axes[0])
axes[0].set_ylabel("Return harian")
axes[1].plot(vol20.index, vol20, lw=0.9, color="darkorange")
mark_crisis(axes[1])
axes[1].set_ylabel("Volatilitas bergulir 20 hari")
plt.tight_layout()
plt.savefig("outputs/figures/eda_return_dan_volatilitas.png", dpi=150)
plt.close()

# ============================================================
# Histogram distribusi return
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(y, bins=60)
ax.set_xlabel("Return harian IHSG")
plt.tight_layout()
plt.savefig("outputs/figures/eda_return_histogram.png", dpi=150)
plt.close()

# ============================================================
# Heatmap korelasi antar fitur (cek multikolinearitas kasar sebelum OLS)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
corr = X.corr()
im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns)
plt.colorbar(im)
plt.tight_layout()
plt.savefig("outputs/figures/eda_correlation_heatmap.png", dpi=150)
plt.close()

# ============================================================
# Deteksi outlier ekstrem (>3xIQR)
# ============================================================
q1, q3 = y.quantile([0.25, 0.75])
iqr = q3 - q1
outliers = df[(y < q1 - 3 * iqr) | (y > q3 + 3 * iqr)]
print(f"\nOutlier ekstrem (>3xIQR): {len(outliers)} hari, "
      f"{len(outliers.loc[:'2024-12-31'])} di train, {len(outliers.loc['2025-01-01':])} di test")
print(outliers[["return_ihsg"]].sort_values("return_ihsg"))