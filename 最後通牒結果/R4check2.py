import pandas as pd

df = pd.read_csv("最後通牒結果/2025最後通牒結果.csv")

# 男性実験者だけ
df_male = df[df["実験者の性別"] == "男性"]

# VCなし
no_vc = df_male[df_male["ボイスチェンジャー"] == "なし"]
print("VCなし R4")
print(no_vc["受諾拒否_R4"].value_counts())

# VCあり
vc = df_male[df_male["ボイスチェンジャー"] == "あり"]
print("VCあり R4")
print(vc["受諾拒否_R4"].value_counts())