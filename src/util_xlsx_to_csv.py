# src/util_xlsx_to_csv.py
import pandas as pd
from pathlib import Path

src = Path("results/segmentation.xlsx")
dst = Path("results/segmentation.csv")

df = pd.read_excel(src, dtype={"item_id": str, "session": str, "pass": str, "flag": str})

# restore original row order
df = df.sort_values(["session", "pass", "position"], kind="stable").reset_index(drop=True)

# clamp float precision — Excel may have stored 101.2 as 101.19999999
for c in ["start_sec", "end_sec"]:
    df[c] = df[c].astype(float).round(2)
df["match_ratio"] = df["match_ratio"].astype(float).round(3)
for c in ["position", "n_whisper_words_matched"]:
    df[c] = df[c].astype(int)

df.to_csv(dst, index=False, lineterminator="\n")
print(f"wrote {dst} — {len(df)} rows, {df['flag'].value_counts().to_dict()}")