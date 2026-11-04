
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind

# =====================================
# 1. データ読み込み
# =====================================
df = pd.read_csv("./距離測定/2025距離測定結果.csv")

# タイムスタンプ列を削除
df = df.drop(columns=[df.columns[0]])

# 性別表記の統一
df["被験者の性別"] = df["被験者の性別"].replace({
    "男": "男性",
    "女": "女性"
})

# =====================================
# 2. 地声条件の最終停止距離を抽出
# =====================================
rows = []

for _, r in df.iterrows():

    name = r["氏名"]
    p_gender = r["被験者の性別"]

    for exp_gender in ["男性", "女性"]:

        # 地声条件の距離測定列を番号順に取得
        cols = [
            f"なしー{exp_gender}{i}"
            for i in range(1, 10)
            if f"なしー{exp_gender}{i}" in df.columns
        ]

        if not cols:
            raise ValueError(
                f"測定列が見つかりません：なしー{exp_gender}"
            )

        # 数値に変換し、欠損値を除外
        values = pd.to_numeric(
            r[cols],
            errors="coerce"
        ).dropna()

        # 最後に記録された距離を採用
        final_distance = (
            values.iloc[-1]
            if len(values) > 0
            else np.nan
        )

        rows.append({
            "氏名": name,
            "被験者性別": p_gender,
            "実験者性別": exp_gender,
            "距離": final_distance
        })

df_dist = pd.DataFrame(rows)

# 欠損値を除外
df_dist = df_dist.dropna(subset=["距離"]).copy()

print("\n========================================")
print("地声条件：最終停止距離")
print("========================================")
print(df_dist.to_string(index=False))

# =====================================
# 3. 男女比較
# =====================================
conditions = {
    "男性相手": df_dist[
        df_dist["実験者性別"] == "男性"
    ],
    "女性相手": df_dist[
        df_dist["実験者性別"] == "女性"
    ]
}

for label, sub in conditions.items():

    print("\n========================================")
    print(f"条件：{label}（VCなし）")
    print("========================================")

    male = sub[
        sub["被験者性別"] == "男性"
    ]["距離"].dropna()

    female = sub[
        sub["被験者性別"] == "女性"
    ]["距離"].dropna()

    # --- 平均・標準偏差 ---
    print("\n【平均・標準偏差】")
    print(
        f"男性 N={len(male)}, "
        f"平均={male.mean():.3f}, "
        f"SD={male.std():.3f}"
    )
    print(
        f"女性 N={len(female)}, "
        f"平均={female.mean():.3f}, "
        f"SD={female.std():.3f}"
    )

    if len(male) < 2 or len(female) < 2:
        print("データ不足のためスキップ")
        continue

    # --- ANOVA ---
    model = ols(
        "距離 ~ C(被験者性別)",
        data=sub
    ).fit()

    anova_table = sm.stats.anova_lm(
        model,
        typ=2
    )

    print("\n【ANOVA】")
    print(anova_table)

    # --- Welchのt検定 ---
    t, p = ttest_ind(
        male,
        female,
        equal_var=False
    )

    print("\n【t検定】")
    print(f"t = {t:.3f}, p = {p:.4f}")

    # --- Cohen's d（独立2群） ---
    n1 = len(male)
    n2 = len(female)

    pooled_sd = np.sqrt(
        (
            (n1 - 1) * male.var(ddof=1)
            + (n2 - 1) * female.var(ddof=1)
        )
        / (n1 + n2 - 2)
    )

    d = (
        (male.mean() - female.mean()) / pooled_sd
        if pooled_sd > 0
        else np.nan
    )

    print("\n【Cohen's d】")
    print(f"d = {d:.3f}")

    if d > 0:
        print("男性被験者の方が対話距離が遠い")
    elif d < 0:
        print("女性被験者の方が対話距離が遠い")
    else:
        print("男女の平均対話距離は同じ")

# =====================================
# 4. Excel箱ひげ図用CSV
# =====================================
box_data = {}

for exp_gender in ["男性", "女性"]:
    for p_gender in ["男性", "女性"]:

        values = (
            df_dist[
                (df_dist["実験者性別"] == exp_gender) &
                (df_dist["被験者性別"] == p_gender)
            ]["距離"]
            .dropna()
            .reset_index(drop=True)
        )

        label = (
            f"{exp_gender}相手_"
            f"{p_gender}被験者"
        )

        box_data[label] = values

box_df = pd.DataFrame(box_data)

box_df.to_csv(
    "./距離測定/対話距離_地声_箱ひげ図用.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n========================================")
print("Excel箱ひげ図用データ")
print("========================================")
print(box_df.to_string(index=False))

print("\nCSV出力完了：./距離測定/対話距離_地声_箱ひげ図用.csv")
