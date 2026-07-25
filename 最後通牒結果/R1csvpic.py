import pandas as pd

# 元データ読み込み
df = pd.read_csv("2025最後通牒結果.csv")

# 必要な5列だけ抽出
df_r1 = df.iloc[:, :5]

# ファイル出力
df_r1.to_csv("R1_round.csv", index=False, encoding="utf-8-sig")
