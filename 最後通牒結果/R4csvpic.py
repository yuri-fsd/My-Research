import pandas as pd

# 元データを読み込み
df = pd.read_csv("2025最後通牒結果.csv")

# 1〜4列目（0〜3）と8列目（7）を抽出
df_r4 = df.iloc[:, [0, 1, 2, 3, 7]]

# 新しいCSVとして出力
df_r4.to_csv("R4_round.csv", index=False, encoding="utf-8-sig")