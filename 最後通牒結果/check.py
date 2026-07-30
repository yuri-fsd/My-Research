import pandas as pd

# ファイルを読み込み（ファイル名を自分の環境に合わせて）
df = pd.read_csv("最後通牒結果/2025最後通牒結果.csv")

# 各被験者×実験者性別×ボイスチェンジャーの出現回数をカウント
group_counts = df.groupby(["氏名", "被験者の性別", "実験者の性別", "ボイスチェンジャー"]).size().reset_index(name="count")

# 重複している条件を抽出
duplicates = group_counts[group_counts["count"] > 1]

print("▼重複している条件：")
print(duplicates)
