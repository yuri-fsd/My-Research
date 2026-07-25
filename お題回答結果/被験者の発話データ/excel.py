import pandas as pd

# 入力
df = pd.read_csv("情報単位スコア_箱ひげ用.csv")

# 条件列
condition_cols = [
    "男被_男性実_なし",
    "男被_男性実_あり",
    "女被_女性実_なし",
    "女被_女性実_あり",
    "男被_女性実_なし",
    "男被_女性実_あり",
    "女被_男性実_なし",
    "女被_男性実_あり"
]

# 条件ごとに「空白を除いて10件ずつ」集める
result = {}

for col in condition_cols:
    values = df[col].dropna().tolist()
    assert len(values) == 10, f"{col} の件数が {len(values)} 件です（10件のはず）"
    result[col] = values

# 横持ち箱ひげ用DataFrame
boxplot_df = pd.DataFrame(result)

# 保存
boxplot_df.to_csv(
    "情報単位スコア_箱ひげ最終.csv",
    index=False,
    encoding="utf-8-sig"
)

print("完成：情報単位スコア_箱ひげ最終.csv（全条件10件ずつ）")
