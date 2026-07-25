import pandas as pd

df = pd.read_csv("距離と回数_差分まとめ.csv")

# 行動タイプ分類
def classify(dist, count):
    if dist > 0 and count > 0:
        return "Approach"
    elif dist < 0 and count > 0:
        return "Hesitation"
    elif dist < 0 and count < 0:
        return "Cautious"
    else:
        return "Smooth"

df["BehaviorType"] = df.apply(lambda x: classify(
    x["距離差分(あり-なし)"], x["回数差分(あり-なし)"]
), axis=1)

# -------------------------
# 条件ごとの抽出
# -------------------------

results = {}

# ① VC有無（全体）
results["VC有無（全体）"] = df["BehaviorType"].value_counts()

# ② 男性被験者
results["男性被験者"] = df[df["被験者性別"]=="男性"]["BehaviorType"].value_counts()

# ③ 女性被験者
results["女性被験者"] = df[df["被験者性別"]=="女性"]["BehaviorType"].value_counts()

# ④ 男性被 × 男性実
cond = (df["被験者性別"]=="男性") & (df["実験者性別"]=="男性")
results["男性被 × 男性実"] = df[cond]["BehaviorType"].value_counts()

# ⑤ 女性被 × 女性実
cond = (df["被験者性別"]=="女性") & (df["実験者性別"]=="女性")
results["女性被 × 女性実"] = df[cond]["BehaviorType"].value_counts()

# ⑥ 男性被 × 女性実
cond = (df["被験者性別"]=="男性") & (df["実験者性別"]=="女性")
results["男性被 × 女性実"] = df[cond]["BehaviorType"].value_counts()

# ⑦ 女性被 × 男性実
cond = (df["被験者性別"]=="女性") & (df["実験者性別"]=="男性")
results["女性被 × 男性実"] = df[cond]["BehaviorType"].value_counts()

# -------------------------
# 1つの巨大な表にまとめる
# -------------------------

table = pd.DataFrame(results).fillna(0).astype(int)

print("\n==============================")
print(" 行動タイプ × 条件（ゆりあ専用）")
print("==============================")
print(table)
