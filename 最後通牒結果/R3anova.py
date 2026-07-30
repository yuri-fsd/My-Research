import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# R3データの読み込み
df = pd.read_csv("最後通牒結果/R3_round.csv")

# 列名の統一z
df.columns = ["participant", "participant_gender", "experimenter_gender", "voice", "offer"]

# 三要因分散分析モデル
model = ols('offer ~ C(voice) * C(participant_gender) * C(experimenter_gender)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)


# t検定（ボイスチェンジャー有無の2群間比較）
# 平均値の可視化
df.columns = ["participant", "participant_gender", "experimenter_gender", "voice", "offer"]
df["voice"] = df["voice"].replace({"あり": "voicechanger", "なし": "not_voicechanger"})

# グループ分け
group_with = df[df["voice"] == "voicechanger"]["offer"]
group_without = df[df["voice"] == "not_voicechanger"]["offer"]

# t検定（Welchのt検定）
t, p = ttest_ind(group_with, group_without, equal_var=False)

# 平均と標準偏差をまとめて出力
summary = df.groupby("voice")["offer"].agg(["mean", "std", "count"])
print(summary)
print(f"\nt = {t:.3f}, p = {p:.3f}")
print(f"Mean (with) = {group_with.mean():.2f}, Mean (without) = {group_without.mean():.2f}")

# 可視化（箱ひげ図）
df.columns = ["participant", "participant_gender", "experimenter_gender", "voice", "offer"]
df["voice"] = df["voice"].replace({"あり": "voicechanger", "なし": "not_voicechanger"})

# 箱ひげ図（中央値＋平均ともに実線）
plt.figure(figsize=(7,6))
plt.boxplot(
    [df[df["voice"] == "not_voicechanger"]["offer"],
     df[df["voice"] == "voicechanger"]["offer"]],
    labels=["not_voicechanger", "voicechanger"],
    patch_artist=True,
    boxprops=dict(facecolor="#D6EAF8", color="#2874A6", linewidth=1.5),
    medianprops=dict(color="#1A5276", linewidth=2, linestyle="-"),  # 中央値：実線
    meanline=True,     # 平均線を描画
    showmeans=True,
    meanprops=dict(color="green", linewidth=2, linestyle="-"),  # 平均：実線
    whiskerprops=dict(color="#2874A6"),
    capprops=dict(color="#2874A6")
)

plt.ylabel("Offer Amount")
plt.xlabel("Voice Changer Condition")
plt.title("Boxplot of Offer Amounts with Mean and Median (Round 3)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
