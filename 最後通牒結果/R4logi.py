import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===== データ読み込み =====
df = pd.read_csv("最後通牒結果/R4_round.csv")

# ===== 前処理 =====
df.columns = ["participant", "participant_gender", "experimenter_gender", "voice", "accept"]
df["voice"] = df["voice"].replace({"あり": "voicechanger", "なし": "not_voicechanger"})
df["participant_gender"] = df["participant_gender"].replace({"男性": "male", "女性": "female"})
df["experimenter_gender"] = df["experimenter_gender"].replace({"男性": "male", "女性": "female"})
df["accept"] = df["accept"].replace({"受諾": 1, "拒否": 0})  # 受諾=1／拒否=0

# ===== ロジスティック回帰分析 =====
model = smf.logit(
    formula="accept ~ C(voice) * C(participant_gender) * C(experimenter_gender)",
    data=df
).fit()

print("\n=== Logistic Regression Summary ===")
print(model.summary())

# オッズ比（解釈しやすい形で出力）
odds_ratios = pd.DataFrame({
    "Variable": model.params.index,
    "Odds Ratio": model.params.apply(lambda x: round(np.exp(x), 3)),
    "p-value": model.pvalues.round(3)
})
print("\n=== Odds Ratios ===")
print(odds_ratios)

# 擬似決定係数（Pseudo R²）
llf = model.llf  # 対数尤度
llnull = model.llnull
pseudo_r2 = 1 - (llf / llnull)
print(f"\nPseudo R² = {pseudo_r2:.3f}")

# ===== ヒートマップ（平均受諾率） =====
pivot_table = df.pivot_table(
    values="accept",
    index=["participant_gender", "experimenter_gender"],
    columns="voice",
    aggfunc="mean"
)

print("\n=== 平均受諾率（Participant × Experimenter × Voice） ===")
print(pivot_table)

# ヒートマップ描画
plt.figure(figsize=(7,5))
sns.heatmap(
    pivot_table,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=0,
    vmax=1,
    linewidths=0.5,
    cbar_kws={'label': 'Acceptance Rate'}
)
plt.title("Acceptance Rate by Voice, Participant Gender, and Experimenter Gender (R4)")
plt.ylabel("Participant × Experimenter")
plt.xlabel("Voice Changer Condition")
plt.tight_layout()
plt.show()
