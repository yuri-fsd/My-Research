import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind
import numpy as np

# ===== データ読み込み =====
df = pd.read_csv("情報単位スコア一覧.csv")

# 列名整理
df = df.rename(columns={
    "name": "participant",
    "participant_gender": "p_gender",
    "experimenter_gender": "e_gender",
    "voice_changer": "voice",
    "info_unit_score": "score"
})

df["voice"] = df["voice"].replace({"あり": "vc", "なし": "no_vc"})

# ===== 条件定義 =====
conditions = {
    "全体": df,
    "男性被験者": df[df["p_gender"] == "男"],
    "女性被験者": df[df["p_gender"] == "女"],
    "男性×男性": df[(df["p_gender"] == "男") & (df["e_gender"] == "男性")],
    "女性×女性": df[(df["p_gender"] == "女") & (df["e_gender"] == "女性")],
    "男性×女性": df[(df["p_gender"] == "男") & (df["e_gender"] == "女性")],
    "女性×男性": df[(df["p_gender"] == "女") & (df["e_gender"] == "男性")],
}

# ===== 分析 =====
for label, sub in conditions.items():
    print("\n==============================")
    print(f"条件: {label}")
    print("==============================")

    if sub["voice"].nunique() < 2:
        print("VC条件が不足しているためスキップ")
        continue

    vc = sub[sub["voice"] == "vc"]["score"]
    no_vc = sub[sub["voice"] == "no_vc"]["score"]

    if len(vc) < 2 or len(no_vc) < 2:
        print("データ数不足のためスキップ")
        continue

    # --- ANOVA ---
    model = ols("score ~ C(voice)", data=sub).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print("\n【ANOVA】")
    print(anova_table)

    # --- t検定（Welch） ---
    t, p = ttest_ind(vc, no_vc, equal_var=False)

    print("\n【t検定】")
    print(f"t = {t:.3f}, p = {p:.3f}")

    # --- Cohen's d ---
    pooled_std = np.sqrt((vc.var(ddof=1) + no_vc.var(ddof=1)) / 2)
    d = (vc.mean() - no_vc.mean()) / pooled_std if pooled_std != 0 else np.nan

    print("\n【Cohen's d】")
    print(f"d = {d:.3f}")

    # --- 平均・SD ---
    print("\n【平均・標準偏差】")
    summary = sub.groupby("voice")["score"].agg(["mean", "std", "count"])
    print(summary)
