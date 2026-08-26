import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

OUT = "data_raw"

ihsg = pd.read_csv(f"{OUT}/ihsg.csv", index_col=0, parse_dates=True)["ihsg"]
df = pd.DataFrame({"ihsg": ihsg})

# variabel yang sudah harian (kalender/bursa) -- reindex ke hari bursa IHSG, ffill dulu baru potong
daily_vars = ["usdidr", "sp500", "oil_wti", "dxy", "vix", "ust10y", "fed_funds_rate", "epu_daily", "indonia"]
for name in daily_vars:
    s = pd.read_csv(f"{OUT}/{name}.csv", index_col=0, parse_dates=True).iloc[:, 0]
    col = name.replace("_daily", "")
    df[col] = s.reindex(df.index.union(s.index)).ffill().reindex(df.index)

# bi_rate cuma berubah saat RDG -- ffill ke kalender penuh dulu, baru potong ke hari bursa
bi = pd.read_csv(f"{OUT}/bi_rate.csv", index_col=0, parse_dates=True)["bi_rate"]
bi_full = bi.reindex(pd.date_range(bi.index.min(), df.index.max())).ffill()
df["bi_rate"] = bi_full.reindex(df.index)

df = df.dropna()  # buang baris awal sebelum bi_rate/variabel lain punya nilai
assert df.isna().sum().sum() == 0

# return log IHSG + lag 1 hari semua fitur makro
df["return_ihsg"] = np.log(df["ihsg"] / df["ihsg"].shift(1))
fitur = [c for c in df.columns if c not in ("ihsg", "return_ihsg")]
df_model = df[fitur].shift(1).join(df["return_ihsg"]).dropna()
df_model.columns = [c if c == "return_ihsg" else f"{c}_lag1" for c in df_model.columns]

df_model.to_csv("data_processed/dataset_final.csv")
print(df_model.shape, df_model.index.min(), "s/d", df_model.index.max())

for c in df_model.columns:
    p = adfuller(df_model[c].dropna())[1]
    print(f"{c:20s} ADF p={p:.4f}", "stasioner" if p < 0.05 else "TIDAK stasioner -> perlu diff")

non_stationary = ["usdidr", "sp500", "oil_wti", "dxy", "ust10y",
                   "fed_funds_rate", "epu", "indonia", "bi_rate"]
for col in non_stationary:
    df_model[f"{col}_lag1"] = df_model[f"{col}_lag1"].diff()
df_model = df_model.dropna()

# validasi ulang -- pastikan diff beneran bikin stasioner, bukan asal transform
for c in df_model.columns:
    p = adfuller(df_model[c].dropna())[1]
    tag = "stasioner" if p < 0.05 else "MASIH TIDAK stasioner -- perlu dicek manual"
    print(f"{c:20s} ADF p={p:.4f} {tag}")

df_model.to_csv("data_processed/dataset_final.csv")
print(df_model.shape)