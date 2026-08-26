import os
import sys
from pathlib import Path
import pandas as pd
import yfinance as yf
from fredapi import Fred

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FRED_API_KEY

START, END = "2020-01-01", "2026-06-30"
OUT = "data_raw"
os.makedirs(OUT, exist_ok=True)

# --- Yahoo Finance: 6 ticker sekali download ---
tickers = {"^JKSE": "ihsg", "USDIDR=X": "usdidr", "^GSPC": "sp500",
           "CL=F": "oil_wti", "DX-Y.NYB": "dxy", "^VIX": "vix"}
yf_data = yf.download(list(tickers), start=START, end=END, auto_adjust=True)["Close"]
yf_data = yf_data.rename(columns=tickers)
for col in yf_data.columns:
    yf_data[col].dropna().to_csv(f"{OUT}/{col}.csv")

# --- FRED: 2 series harian (DFF, bukan FEDFUNDS yang bulanan) ---
fred = Fred(api_key=FRED_API_KEY)
for series_id, name in {"DFF": "fed_funds_rate", "DGS10": "ust10y"}.items():
    s = fred.get_series(series_id, observation_start=START, observation_end=END).rename(name)
    s = s.ffill()  # isi gap hari libur finansial AS yang bukan weekend
    s.to_csv(f"{OUT}/{name}.csv")

# --- Tanggal campuran (Excel kadang parse otomatis, kadang enggak) ---
BULAN_ID = {"januari":1,"jan":1,"februari":2,"feb":2,"maret":3,"mar":3,"april":4,"apr":4,
            "mei":5,"juni":6,"jun":6,"juli":7,"jul":7,"agustus":8,"agu":8,"agt":8,
            "september":9,"sep":9,"oktober":10,"okt":10,"november":11,"nov":11,"desember":12,"des":12}

def parse_tanggal(series):
    hasil = pd.to_datetime(series, format="mixed", dayfirst=True, errors="coerce")
    for i in hasil[hasil.isna()].index:
        d, b, y = str(series[i]).split()
        hasil[i] = pd.Timestamp(int(y), BULAN_ID[b.lower()], int(d))
    return hasil

def clean_cols(df):
    df.columns = df.columns.str.replace("<br>", " ", regex=False).str.strip()
    return df

assert parse_tanggal(pd.Series(["23-Jun-26", "29 Mei 2026"]))[1] == pd.Timestamp(2026, 5, 29)

# --- EPU harian ---
epu_raw = clean_cols(pd.read_csv(f"{OUT}/epu_daily_raw.csv"))
epu = pd.Series(epu_raw["daily_policy_index"].values,
                 index=pd.to_datetime(epu_raw[["year", "month", "day"]]),
                 name="epu").sort_index().loc[START:END]
epu.to_csv(f"{OUT}/epu_daily.csv")

# --- BI Rate ---
bi = clean_cols(pd.read_csv(f"{OUT}/bi_rate_raw.csv", sep=";"))
bi["Tanggal"] = parse_tanggal(bi["Tanggal"])
bi_rate = bi.dropna(subset=["Tanggal"]).assign(
    bi_rate=lambda d: d["BI-7Day-RR"].astype(str).str.replace("%", "").str.strip().astype(float)
).set_index("Tanggal")["bi_rate"].sort_index().loc[START:END]
bi_rate.to_csv(f"{OUT}/bi_rate.csv")

# --- INDONIA ---
ind = clean_cols(pd.read_csv(f"{OUT}/indonia_raw.csv", sep=";"))
ind["Tanggal Publikasi"] = parse_tanggal(ind["Tanggal Publikasi"])
indonia = ind.dropna(subset=["Tanggal Publikasi"]).assign(
    indonia=lambda d: d["IndONIA (%)"].astype(str).str.replace(",", ".").astype(float)
).set_index("Tanggal Publikasi")["indonia"].sort_index().loc[START:END]
indonia.to_csv(f"{OUT}/indonia.csv")

print({"ihsg": len(yf_data["ihsg"].dropna()), "epu": len(epu),
       "bi_rate": len(bi_rate), "indonia": len(indonia)})