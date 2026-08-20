# ============================================================
# HARI 1 -- AKUISISI DATA
# Analisis Faktor Makroekonomi Domestik-Global terhadap Return IHSG
# ============================================================
# CATATAN PENTING SEBELUM DIJALANKAN:
# 1. Skrip ini HARUS dijalankan di Google Colab atau lingkungan dengan
#    akses internet penuh -- tidak bisa dites di sandbox pembuatan draf ini
#    karena domain finance data (Yahoo Finance, FRED) tidak diizinkan di sana.
# 2. Daftarkan API key FRED gratis dulu di https://fred.stlouisfed.org/docs/api/api_key.html
#    sebelum menjalankan bagian FRED.
# 3. BI Rate dan INDONIA TIDAK bisa diambil otomatis -- unduh manual dari
#    situs resmi Bank Indonesia (bi.go.id), lalu ikuti format loading di
#    bagian akhir skrip ini.
# ============================================================

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred

pd.set_option("display.width", 120)

OUTPUT_DIR = "data_raw"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_DATE = "2021-01-01"
END_DATE = "2026-08-31"

# ------------------------------------------------------------
# 1. DATA DARI YAHOO FINANCE (6 variabel, semuanya genuinely harian)
# ------------------------------------------------------------
YF_TICKERS = {
    "ihsg":   "^JKSE",      # Target: harga penutupan IHSG
    "usdidr": "USDIDR=X",   # Domestik: kurs
    "sp500":  "^GSPC",      # Global: indeks S&P 500
    "oil_wti": "CL=F",      # Global: harga minyak WTI
    "dxy":    "DX-Y.NYB",   # Global: Dollar Index
    "vix":    "^VIX",       # Global: CBOE Volatility Index
}

def fetch_yahoo_series(ticker: str, name: str) -> pd.Series:
    """Ambil data Close harian dari Yahoo Finance, kembalikan sebagai Series bernama `name`."""
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"Data kosong untuk ticker {ticker}. Cek koneksi atau validitas ticker.")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.name = name
    return close

yahoo_data = {}
for name, ticker in YF_TICKERS.items():
    print(f"Mengambil {name} ({ticker})...")
    try:
        yahoo_data[name] = fetch_yahoo_series(ticker, name)
        yahoo_data[name].to_csv(f"{OUTPUT_DIR}/{name}.csv")
        print(f"  -> {len(yahoo_data[name])} baris tersimpan.")
    except Exception as e:
        print(f"  -> GAGAL: {e}. Cek ticker '{ticker}' masih aktif di Yahoo Finance.")

# ------------------------------------------------------------
# 2. DATA DARI FRED (2 variabel)
# ------------------------------------------------------------
# GANTI dengan API key Anda sendiri, jangan commit key ini ke repo publik
FRED_API_KEY = "ISI_API_KEY_ANDA_DI_SINI"
fred = Fred(api_key=FRED_API_KEY)

FRED_SERIES = {
    "fed_funds_rate": "FEDFUNDS",  # Domestik? bukan -- ini GLOBAL: suku bunga acuan The Fed
    "ust10y":         "DGS10",     # Global: yield obligasi pemerintah AS 10 tahun
}

fred_data = {}
for name, series_id in FRED_SERIES.items():
    print(f"Mengambil {name} ({series_id}) dari FRED...")
    try:
        s = fred.get_series(series_id, observation_start=START_DATE, observation_end=END_DATE)
        s.name = name
        fred_data[name] = s
        s.to_csv(f"{OUTPUT_DIR}/{name}.csv")
        print(f"  -> {len(s)} baris tersimpan.")
    except Exception as e:
        print(f"  -> GAGAL: {e}. Cek API key FRED sudah benar.")

# Catatan: FEDFUNDS di FRED biasanya bulanan (rata-rata efektif bulanan),
# bukan harian. Perlakukan sebagai step function seperti BI Rate --
# forward-fill ke hari bursa, JANGAN interpolasi linear.

# ------------------------------------------------------------
# 3. US ECONOMIC POLICY UNCERTAINTY INDEX (bulanan -- perlu lag publikasi)
# ------------------------------------------------------------
# EPU index TIDAK tersedia lewat API otomatis yang stabil. Unduh manual:
#   https://www.policyuncertainty.com/us_monthly.html
#   -> unduh file CSV/XLS "US_Policy_Uncertainty_Data.xlsx", simpan sebagai
#      data_raw/epu_raw.csv dengan kolom minimal: Year, Month, News_Based_Policy_Uncert_Index
#
# PENTING: EPU index bulan M baru dipublikasikan sekitar awal bulan M+1.
# Kode di bawah menerapkan lag publikasi 1 bulan secara eksplisit --
# JANGAN gunakan nilai bulan M langsung di tanggal-tanggal bulan M
# (itu look-ahead bias).

def load_epu_with_publication_lag(path: str = f"{OUTPUT_DIR}/epu_raw.csv") -> pd.Series:
    """
    Baca EPU mentah (Year, Month, index value), lalu geser tanggal berlaku
    ke awal bulan berikutnya untuk mensimulasikan lag publikasi riil.
    """
    raw = pd.read_csv(path)
    raw["period_end"] = pd.to_datetime(
        raw["Year"].astype(str) + "-" + raw["Month"].astype(str) + "-01"
    ) + pd.offsets.MonthEnd(0)
    # Nilai baru valid dipakai mulai tanggal 5 bulan berikutnya (asumsi konservatif,
    # sesuaikan jika Anda menemukan tanggal rilis resmi yang lebih presisi)
    raw["valid_from"] = raw["period_end"] + pd.Timedelta(days=5)
    epu_col = [c for c in raw.columns if "Policy_Uncert" in c or "epu" in c.lower()]
    if not epu_col:
        raise ValueError("Kolom EPU tidak ditemukan, cek nama kolom di file mentah.")
    raw = raw.rename(columns={epu_col[0]: "epu"})
    return raw.set_index("valid_from")["epu"].sort_index()

try:
    epu_series = load_epu_with_publication_lag()
    print(f"EPU dimuat: {len(epu_series)} baris (sebelum forward-fill ke harian).")
except FileNotFoundError:
    print("epu_raw.csv belum diunduh -- lengkapi manual dari policyuncertainty.com sebelum Hari 2.")

# ------------------------------------------------------------
# 4. BI RATE DAN INDONIA (manual, dari situs resmi Bank Indonesia)
# ------------------------------------------------------------
# Unduh manual dari:
#   BI Rate  : https://www.bi.go.id/id/statistik/indikator/bi-rate.aspx
#   INDONIA  : https://www.bi.go.id/id/statistik/informasi-kurs/indonia/Default.aspx
# Simpan masing-masing sebagai data_raw/bi_rate_raw.csv dan data_raw/indonia_raw.csv
# dengan minimal dua kolom: tanggal, nilai.
#
# Template loading (sesuaikan nama kolom persis setelah Anda unduh filenya):
#
# bi_rate = pd.read_csv(f"{OUTPUT_DIR}/bi_rate_raw.csv", parse_dates=["tanggal"])
# bi_rate = bi_rate.set_index("tanggal")["nilai"].rename("bi_rate").sort_index()
#
# indonia = pd.read_csv(f"{OUTPUT_DIR}/indonia_raw.csv", parse_dates=["tanggal"])
# indonia = indonia.set_index("tanggal")["nilai"].rename("indonia").sort_index()

print("\n=== RINGKASAN ===")
print("Otomatis (Yahoo Finance):", list(yahoo_data.keys()))
print("Otomatis (FRED):", list(fred_data.keys()))
print("Perlu unduh manual: epu_raw.csv, bi_rate_raw.csv, indonia_raw.csv")
print("Semua file mentah tersimpan di folder:", OUTPUT_DIR)
print("\nLanjutkan ke Hari 2 (penggabungan data) setelah 3 file manual di atas lengkap.")
