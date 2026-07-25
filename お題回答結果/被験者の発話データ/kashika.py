# 同性異性相手全体の分析
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# データ読み込み
# =========================
df = pd.read_csv("被験者の発話データ/情報単位スコア一覧.csv")

# =========================
# 前処理
# =========================
df["participant_gender"] = df["participant_gender"].replace({"男":"男性","女":"女性"})

# 同性 / 異性
df["relation"] = np.where(
    df["participant_gender"] == df["experimenter_gender"],
    "Same", "Different"
)

# =========================
# VCごとに平均（トピック無視）
# =========================
mean_df = df.groupby(
    ["name","relation","voice_changer"]
)["info_unit_score"].mean().reset_index()

# 横持ち
pivot = mean_df.pivot(
    index=["name","relation"],
    columns="voice_changer",
    values="info_unit_score"
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

# 個人データ
plt.scatter(np.zeros(len(same)), same, color="blue", alpha=0.6, label="Same Gender")
plt.scatter(np.ones(len(diff)), diff, color="red", alpha=0.6, label="Different Gender")

# 平均＋エラーバー
plt.errorbar([0,1], means, yerr=ses, fmt="o-", color="black", capsize=5)

# 0ライン
plt.axhline(0, linestyle="--", color="blue")

# 軸
plt.xticks([0,1], ["Same", "Different"])
plt.ylabel("Difference in Self-Disclosure (VC On − Off)")
plt.xlabel("Interaction Type")

# タイトル
plt.title("Effect of Voice Changer on Self-Disclosure")

# 凡例
plt.legend()

# 範囲（スライドに合わせる）
plt.ylim(-15,15)

# 見切れ防止🔥
plt.tight_layout()

plt.show()

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # =========================
# # データ読み込み
# # =========================
# df = pd.read_csv("被験者の発話データ/情報単位スコア一覧.csv")

# # 性別統一
# df["participant_gender"] = df["participant_gender"].replace({"男":"男性","女":"女性"})

# # 同性 / 異性
# df["relation"] = np.where(
#     df["participant_gender"] == df["experimenter_gender"],
#     "Same", "Different"
# )

# # =========================
# # 関数化（ここ重要🔥）
# # =========================
# def analyze_and_plot(data, title):

#     mean_df = data.groupby(
#         ["name","relation","voice_changer"]
#     )["info_unit_score"].mean().reset_index()

#     pivot = mean_df.pivot(
#         index=["name","relation"],
#         columns="voice_changer",
#         values="info_unit_score"
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

#     plt.axhline(0, linestyle="--", color="blue")

#     plt.xticks([0,1], ["Same", "Different"])
#     plt.ylabel("Difference in Self-Disclosure (VC On − Off)")
#     plt.xlabel("Interaction Type")

#     plt.title(title)

#     plt.legend()
#     plt.ylim(-15, 15)
#     plt.tight_layout()

#     plt.show()

# # =========================
# # 男性被験者
# # =========================
# male_df = df[df["participant_gender"] == "男性"]
# analyze_and_plot(male_df, "Male Participants")

# # =========================
# # 女性被験者
# # =========================
# female_df = df[df["participant_gender"] == "女性"]
# analyze_and_plot(female_df, "Female Participants")