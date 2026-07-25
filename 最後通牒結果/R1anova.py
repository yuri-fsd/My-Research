import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# ===== データ読み込み =====
df = pd.read_csv("R1_round.csv")

# 列名の統一
df.columns = ["participant", "participant_gender", "experimenter_gender", "voice", "offer"]
df["voice"] = df["voice"].replace({"あり": "voicechanger", "なし": "not_voicechanger"})

# ===== 性別ごとの分析 =====
for gender in ["男性", "女性"]:
    print(f"\n==============================")
    print(f"被験者性別: {gender}")
    print("==============================")

    # 指定性別のみ抽出
    df_gender = df[df["participant_gender"] == gender]

    # --- ANOVA ---
    model = ols('offer ~ C(voice) * C(experimenter_gender)', data=df_gender).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print("\n【ANOVA結果】")
    print(anova_table)

    # --- t検定 ---
    group_with = df_gender[df_gender["voice"] == "voicechanger"]["offer"]
    group_without = df_gender[df_gender["voice"] == "not_voicechanger"]["offer"]
    t, p = ttest_ind(group_with, group_without, equal_var=False)

    print("\n【t検定結果】")
    print(f"t = {t:.3f}, p = {p:.3f}")
    print(f"Mean (with) = {group_with.mean():.2f}, Mean (without) = {group_without.mean():.2f}")

    # --- 平均と標準偏差 ---
    summary = df_gender.groupby("voice")["offer"].agg(["mean", "std", "count"])
    print("\n【平均・標準偏差】")
    print(summary)

    # --- 可視化（箱ひげ図） ---
    plt.figure(figsize=(7,6))
    plt.boxplot(
        [df_gender[df_gender["voice"] == "not_voicechanger"]["offer"],
         df_gender[df_gender["voice"] == "voicechanger"]["offer"]],
        labels=["not_voicechanger", "voicechanger"],
        patch_artist=True,
        showmeans=True,       # 平均を表示
        meanline=True,        # 平均を線で
        showfliers=True,      # 外れ値（丸）を表示
        flierprops=dict(marker='o', color='red', markersize=6),  # 外れ値：赤丸
        boxprops=dict(facecolor="#D6EAF8" if gender == "男性" else "#F9EBEA",
                      color="#2874A6" if gender == "男性" else "#922B21",
                      linewidth=1.5),
        medianprops=dict(color="#1A5276" if gender == "男性" else "#641E16",
                         linewidth=2, linestyle="-"),
        meanprops=dict(color="green", linewidth=2, linestyle="-"),
        whiskerprops=dict(color="#2874A6" if gender == "男性" else "#922B21"),
        capprops=dict(color="#2874A6" if gender == "男性" else "#922B21")
    )

    plt.ylabel("Offer Amount")
    plt.xlabel("Voice Changer Condition")
    plt.title(f"Boxplot of Offer Amounts by Voice (Round 1, {gender})")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
