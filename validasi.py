import pandas as pd
import glob

for f in sorted(glob.glob("data_raw/*.csv")):
    if f.endswith("_raw.csv"):
        continue
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    col = df.columns[0]
    print(f"{f:28s} n={len(df):5d}  {df.index.min().date()} s/d {df.index.max().date()}  "
          f"NaN={df[col].isna().sum():3d}  dup={df.index.duplicated().sum():3d}  "
          f"min={df[col].min():10.4f}  max={df[col].max():10.4f}")