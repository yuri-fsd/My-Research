
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind

# ===== データ読み込み =====
df = pd.read_csv("./情報単位スコア一覧.csv")

df = df.rename(columns={
    "name": "participant",
    "participant_gender": "p_gender",
    "experimenter_gender": "e_gender",
    "voice_changer": "voice",
    "info_unit_score": "score"
})

# ===== 表記統一 =====
df["p_gender"] = df["p_gender"].replace({
    "男性": "男", "女性": "女"
})

df["e_gender"] = df["e_gender"].replace({
    "男": "男性", "女": "女性"
})

df["voice"] = df["voice"].replace({
    "あり": "vc", "なし": "no_vc"
})

df["score"] = pd.to_numeric(df["score"], errors="coerce")

# ===== 地声条件のみ抽出 =====
df = df[df["voice"] == "no_vc"].copy()

# ===== 比較条件 =====
conditions = {
    "男性相手": df[df["e_gender"] == "男性"],
    "女性相手": df[df["e_gender"] == "女性"]
}

# ===== 分析 =====
for label, sub in conditions.items():

    print("\n========================================")
    print(f"条件：{label}（VCなし）")
    print("========================================")

    male = sub[sub["p_gender"] == "男"]["score"].dropna()
    female = sub[sub["p_gender"] == "女"]["score"].dropna()

    print("\n【平均・標準偏差】")
    print(f"男性 N={len(male)}, 平均={male.mean():.3f}, SD={male.std():.3f}")
    print(f"女性 N={len(female)}, 平均={female.mean():.3f}, SD={female.std():.3f}")

    if len(male) < 2 or len(female) < 2:
        print("データ不足のためスキップ")
        continue

    # --- ANOVA ---
    model = ols("score ~ C(p_gender)", data=sub).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print("\n【ANOVA】")
    print(anova_table)

    # --- Welchのt検定 ---
    t, p = ttest_ind(male, female, equal_var=False)

    print("\n【t検定】")
    print(f"t = {t:.3f}, p = {p:.4f}")

    # --- Cohen's d（独立2群） ---
    n1, n2 = len(male), len(female)

    pooled_sd = np.sqrt(
        ((n1 - 1) * male.var(ddof=1)
         + (n2 - 1) * female.var(ddof=1))
        / (n1 + n2 - 2)
    )

    d = (male.mean() - female.mean()) / pooled_sd if pooled_sd > 0 else np.nan

    print("\n【Cohen's d】")
    print(f"d = {d:.3f}")

    if d > 0:
        print("男性被験者の方が自己開示量が多い")
    elif d < 0:
        print("女性被験者の方が自己開示量が多い")

# ===== Excel箱ひげ図用CSV =====

box_data = {}

for e_gender in ["男性", "女性"]:
    for p_gender in ["男", "女"]:

        values = (
            df[
                (df["e_gender"] == e_gender) &
                (df["p_gender"] == p_gender)
            ]["score"]
            .dropna()
            .reset_index(drop=True)
        )

        label = (
            f"{e_gender}相手_"
            f"{'男性' if p_gender == '男' else '女性'}被験者"
        )

        box_data[label] = values

box_df = pd.DataFrame(box_data)

box_df.to_csv(
    "自己開示_地声_箱ひげ図用.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n【箱ひげ図用データ】")
print(box_df)
print("\nCSV出力完了")
