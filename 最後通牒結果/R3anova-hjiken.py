import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# ===== データ読み込み =====
df = pd.read_csv("最後通牒結果/R3_round.csv")

# ===== 前処理 =====
df.columns = ["participant", "participant_gender", "experimenter_gender", "voice", "offer"]
df["voice"] = df["voice"].replace({"あり": "voicechanger", "なし": "not_voicechanger"})

# ===== 同性／異性の組み合わせを追加 =====
df["pair_type"] = df.apply(
    lambda x: "same_gender" if x["participant_gender"] == x["experimenter_gender"] else "different_gender",
    axis=1
)

# ===== 同性・異性 × 被験者性別ごとに分析 =====
for pair_type in ["same_gender", "different_gender"]:
    for gender in ["男性", "女性"]:
        print(f"\n==============================")
        print(f"被験者×実験者の組み合わせ: {pair_type}（被験者: {gender}）")
        print("==============================")

        # 条件抽出
        df_pair = df[(df["pair_type"] == pair_type) & (df["participant_gender"] == gender)]

        # データが不足していたらスキップ
        if len(df_pair) < 5:
            print("データ数が少ないためスキップ")
            continue

        # --- ANOVA ---
        model = ols('offer ~ C(voice)', data=df_pair).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print("\n【ANOVA結果】")
        print(anova_table)

        # --- t検定（ボイスチェンジャーあり／なし） ---
        group_with = df_pair[df_pair["voice"] == "voicechanger"]["offer"]
        group_without = df_pair[df_pair["voice"] == "not_voicechanger"]["offer"]
        t, p = ttest_ind(group_with, group_without, equal_var=False)

        print("\n【t検定結果】")
        print(f"t = {t:.3f}, p = {p:.3f}")
        print(f"Mean (with) = {group_with.mean():.2f}, Mean (without) = {group_without.mean():.2f}")

        # --- 平均と標準偏差 ---
        summary = df_pair.groupby("voice")["offer"].agg(["mean", "std", "count"])
        print("\n【平均・標準偏差】")
        print(summary)

        # --- 可視化（箱ひげ図） ---
        plt.figure(figsize=(7,6))
        plt.boxplot(
            [df_pair[df_pair["voice"] == "not_voicechanger"]["offer"],
             df_pair[df_pair["voice"] == "voicechanger"]["offer"]],
            labels=["not_voicechanger", "voicechanger"],
            patch_artist=True,
            showmeans=True,       # 平均を表示
            meanline=True,        # 平均を線で
            showfliers=True,      # 外れ値（丸）を表示
            flierprops=dict(marker='o', color='red', markersize=6),
            boxprops=dict(facecolor="#E8F8F5" if pair_type == "same_gender" else "#FDEDEC",
                          color="#1ABC9C" if pair_type == "same_gender" else "#C0392B",
                          linewidth=1.5),
            medianprops=dict(color="#117A65" if pair_type == "same_gender" else "#922B21",
                             linewidth=2, linestyle="-"),
            meanprops=dict(color="green", linewidth=2, linestyle="-"),
            whiskerprops=dict(color="#1ABC9C" if pair_type == "same_gender" else "#C0392B"),
            capprops=dict(color="#1ABC9C" if pair_type == "same_gender" else "#C0392B")
        )

        plt.ylabel("Offer Amount")
        plt.xlabel("Voice Changer Condition")
        plt.title(f"Round 3 Offer: {pair_type} × 被験者 {gender}")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
