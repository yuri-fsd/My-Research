# 同性相手異性相手全体
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# データ読み込み
# =========================
df = pd.read_csv("R1_round.csv")

# =========================
# 前処理
# =========================
df["relation"] = np.where(
    df["被験者の性別"] == df["実験者の性別"],
    "Same", "Different"
)

# =========================
# VCごとに平均
# =========================
mean_df = df.groupby(
    ["氏名","relation","ボイスチェンジャー"]
)["第１ラウンド"].mean().reset_index()

# 横持ち
pivot = mean_df.pivot(
    index=["氏名","relation"],
    columns="ボイスチェンジャー",
    values="第１ラウンド"
).dropna()

# 差分（VCあり − なし）
pivot["diff"] = pivot["あり"] - pivot["なし"]

# =========================
# 分割
# =========================
same = pivot.loc[pivot.index.get_level_values("relation")=="Same","diff"]
diff = pivot.loc[pivot.index.get_level_values("relation")=="Different","diff"]

# =========================
# 効果量
# =========================
def dz(x):
    return x.mean() / x.std(ddof=1)

print("=== Same Gender ===")
print("Mean:", same.mean())
print("Cohen's d:", dz(same))

print("\n=== Different Gender ===")
print("Mean:", diff.mean())
print("Cohen's d:", dz(diff))

# =========================
# 可視化
# =========================
means = [same.mean(), diff.mean()]
ses = [
    same.std(ddof=1)/np.sqrt(len(same)),
    diff.std(ddof=1)/np.sqrt(len(diff))
]

plt.figure(figsize=(6,5))

# jitter（見やすくする）
jitter = 0.05
plt.scatter(np.zeros(len(same)) + np.random.uniform(-jitter, jitter, len(same)), same,
            color="blue", alpha=0.6, label="Same Gender")
plt.scatter(np.ones(len(diff)) + np.random.uniform(-jitter, jitter, len(diff)), diff,
            color="red", alpha=0.6, label="Different Gender")

# 平均＋エラーバー
plt.errorbar([0,1], means, yerr=ses, fmt="o-", color="black", capsize=5)

# 0ライン
plt.axhline(0, linestyle="--", color="blue")

# 軸
plt.xticks([0,1], ["Same", "Different"])
plt.ylabel("Difference in Offer (VC On − Off)")
plt.xlabel("Interaction Type")

# タイトル
plt.title("Effect of Voice Changer on Ultimatum Offer")

# 凡例
plt.legend()

# 範囲
plt.ylim(-60, 60)

# 見切れ防止
plt.tight_layout()

plt.show()

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # =========================
# # データ読み込み
# # =========================
# df = pd.read_csv("R1_round.csv")

# # =========================
# # 前処理
# # =========================
# df["relation"] = np.where(
#     df["被験者の性別"] == df["実験者の性別"],
#     "Same", "Different"
# )

# # =========================
# # 関数化（再利用）
# # =========================
# def analyze_and_plot(data, title):

#     mean_df = data.groupby(
#         ["氏名","relation","ボイスチェンジャー"]
#     )["第１ラウンド"].mean().reset_index()

#     pivot = mean_df.pivot(
#         index=["氏名","relation"],
#         columns="ボイスチェンジャー",
#         values="第１ラウンド"
#     ).dropna()

#     pivot["diff"] = pivot["あり"] - pivot["なし"]

#     same = pivot.loc[pivot.index.get_level_values("relation")=="Same","diff"]
#     diff = pivot.loc[pivot.index.get_level_values("relation")=="Different","diff"]

#     def dz(x):
#         return x.mean() / x.std(ddof=1)

#     print(f"\n=== {title} ===")
#     print("Same Mean:", same.mean(), " d:", dz(same))
#     print("Different Mean:", diff.mean(), " d:", dz(diff))

#     # =========================
#     # グラフ
#     # =========================
#     means = [same.mean(), diff.mean()]
#     ses = [
#         same.std(ddof=1)/np.sqrt(len(same)),
#         diff.std(ddof=1)/np.sqrt(len(diff))
#     ]

#     plt.figure(figsize=(6,5))

#     jitter = 0.05
#     plt.scatter(np.zeros(len(same)) + np.random.uniform(-jitter, jitter, len(same)),
#                 same, color="blue", alpha=0.6, label="Same Gender")

#     plt.scatter(np.ones(len(diff)) + np.random.uniform(-jitter, jitter, len(diff)),
#                 diff, color="red", alpha=0.6, label="Different Gender")

#     plt.errorbar([0,1], means, yerr=ses, fmt="o-", color="black", capsize=5)

#     # 0ライン（灰色に変更おすすめ）
#     plt.axhline(0, linestyle="--", color="gray")

#     plt.xticks([0,1], ["Same", "Different"])
#     plt.ylabel("Difference in Offer (VC On − Off)")
#     plt.xlabel("Interaction Type")

#     plt.title(title)

#     plt.legend()
#     plt.ylim(-60, 60)
#     plt.tight_layout()

#     plt.show()

# # =========================
# # 男性被験者
# # =========================
# male_df = df[df["被験者の性別"] == "男性"]
# analyze_and_plot(male_df, "Male Participants")

# # =========================
# # 女性被験者
# # =========================
# female_df = df[df["被験者の性別"] == "女性"]
# analyze_and_plot(female_df, "Female Participants")