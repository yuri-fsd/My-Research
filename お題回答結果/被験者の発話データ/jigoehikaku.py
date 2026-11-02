
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind

# ========================================
# 1. データ読み込み
# ========================================

df = pd.read_csv("情報単位スコア一覧.csv")

df = df.rename(columns={
    "name": "participant",
    "participant_gender": "p_gender",
    "experimenter_gender": "e_gender",
    "voice_changer": "voice",
    "info_unit_score": "score"
})

# 表記の統一
df["p_gender"] = df["p_gender"].replace({
    "男性": "男", "女性": "女"
})

df["e_gender"] = df["e_gender"].replace({
    "男": "男性", "女": "女性"
})

df["voice"] = df["voice"].replace({
    "あり": "vc",
    "なし": "no_vc"
})

df["score"] = pd.to_numeric(df["score"], errors="coerce")

# ========================================
# 2. 地声条件のみ抽出
# ========================================

df = df[
    (df["voice"] == "no_vc") &
    (df["p_gender"].isin(["男", "女"])) &
    (df["e_gender"].isin(["男性", "女性"]))
].copy()

# ========================================
# 3. 比較条件
# ========================================

conditions = {
    "全体": df,
    "男性相手": df[df["e_gender"] == "男性"],
    "女性相手": df[df["e_gender"] == "女性"]
}

results = []

# ========================================
# 4. 統計分析
# ========================================

for label, sub in conditions.items():

    print("\n========================================")
    print(f"条件：{label}（VCなし）")
    print("========================================")

    male = sub[sub["p_gender"] == "男"]["score"].dropna()
    female = sub[sub["p_gender"] == "女"]["score"].dropna()

    if len(male) < 2 or len(female) < 2:
        print("データ数不足のためスキップ")
        continue

    # --- 平均・標準偏差 ---
    print("\n【平均・標準偏差】")
    print(f"男性 N={len(male)}, 平均={male.mean():.3f}, SD={male.std(ddof=1):.3f}")
    print(f"女性 N={len(female)}, 平均={female.mean():.3f}, SD={female.std(ddof=1):.3f}")

    # --- ANOVA ---
    model = ols("score ~ C(p_gender)", data=sub.dropna(subset=["score"])).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print("\n【ANOVA】")
    print(anova_table)

    # --- Welchのt検定 ---
    t, p = ttest_ind(male, female, equal_var=False)

    print("\n【Welchのt検定】")
    print(f"t = {t:.3f}, p = {p:.4f}")

    # --- Cohen's d（独立2群） ---
    n1, n2 = len(male), len(female)
    s1, s2 = male.std(ddof=1), female.std(ddof=1)

    pooled_sd = np.sqrt(
        ((n1 - 1) * s1**2 + (n2 - 1) * s2**2)
        / (n1 + n2 - 2)
    )

    d = (
        (male.mean() - female.mean()) / pooled_sd
        if pooled_sd > 0 else np.nan
    )

    # --- Hedges' g（小標本補正） ---
    dfree = n1 + n2 - 2
    correction = 1 - 3 / (4 * dfree - 1)
    g = d * correction

    print("\n【効果量】")
    print(f"Cohen's d = {d:.3f}")
    print(f"Hedges' g = {g:.3f}")

    if d > 0:
        print("方向：男性の方が自己開示量が多い")
    elif d < 0:
        print("方向：女性の方が自己開示量が多い")
    else:
        print("方向：男女の平均値が同じ")

    results.append({
        "比較条件": label,
        "男性N": n1,
        "女性N": n2,
        "男性平均": male.mean(),
        "女性平均": female.mean(),
        "男性SD": s1,
        "女性SD": s2,
        "平均差_男性-女性": male.mean() - female.mean(),
        "t値": t,
        "p値": p,
        "Cohens_d": d,
        "Hedges_g": g
    })

# ========================================
# 5. 分析結果をCSV保存
# ========================================

result_df = pd.DataFrame(results)

result_df.to_csv(
    "自己開示_地声_男女比較_統計結果.csv",
    index=False,
    encoding="utf-8-sig"
)

# ========================================
# 6. 箱ひげ図用CSV作成
# ========================================

# Excelで列ごとに箱ひげ図を作れる形式
box_data = {}

for e_gender in ["男性", "女性"]:
    for p_gender in ["男", "女"]:

        values = df[
            (df["e_gender"] == e_gender) &
            (df["p_gender"] == p_gender)
        ]["score"].dropna().reset_index(drop=True)

        label = (
            f"{'男性' if p_gender == '男' else '女性'}被験者"
            f"×{e_gender}相手"
        )

        box_data[label] = values

box_df = pd.DataFrame(box_data)

box_df.to_csv(
    "自己開示_地声_箱ひげ図用.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n========================================")
print("CSV出力完了")
print("========================================")
