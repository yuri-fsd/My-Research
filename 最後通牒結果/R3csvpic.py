import pandas as pd

# 元データを読み込み
df = pd.read_csv("2025最後通牒結果.csv")

# 1〜4列目（0〜3）と6列目（5）を抽出
df_r2 = df.iloc[:, [0, 1, 2, 3, 6]]

# 新しいCSVとして出力
df_r2.to_csv("R3_round.csv", index=False, encoding="utf-8-sig")