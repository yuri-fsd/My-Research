# ============================================================
# 0. import
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import ttest_rel


# ============================================================
# フォント
# ============================================================

plt.rcParams["font.family"] = "Hiragino Sans"


# ============================================================
# ① データ読み込み
# ============================================================

df_ult = pd.read_csv("最後通牒結果/R1_round.csv")
df_body = pd.read_csv("最後通牒結果/2025身体化感覚アンケート.csv")


# ============================================================
# ② 列名整理
# ============================================================

df_ult.columns = df_ult.columns.str.strip()
df_body.columns = df_body.columns.str.strip()


# ============================================================
# ③ 名前クリーニング
# ============================================================

def clean_name(df):

    df["氏名"] = (
        df["氏名"]
        .astype(str)
        .str.strip()
        .str.replace("　", "", regex=False)
        .str.replace(" ", "", regex=False)
    )

    return df


df_ult = clean_name(df_ult)
df_body = clean_name(df_body)


# ============================================================
# ④ 性別表記を統一
# ============================================================

df_ult["被験者の性別"] = (
    df_ult["被験者の性別"]
    .replace({
        "男": "男性",
        "女": "女性"
    })
)

df_ult["実験者の性別"] = (
    df_ult["実験者の性別"]
    .replace({
        "男": "男性",
        "女": "女性"
    })
)


# ============================================================
# ⑤ 同性 / 異性
# ============================================================

df_ult["relation"] = np.where(
    df_ult["被験者の性別"] == df_ult["実験者の性別"],
    "Same",
    "Different"
)

df_ult["relation_num"] = np.where(
    df_ult["relation"] == "Same",
    0,
    1
)


# ============================================================
# ⑥ 身体化感覚スコア
# ============================================================

numeric_cols = df_body.columns[-12:]

# 所有感
df_body["ownership"] = (
    df_body[numeric_cols[:4]]
    .mean(axis=1)
)

# 主体感
df_body["agency"] = (
    df_body[numeric_cols[4:8]]
    .mean(axis=1)
)

# 自己表象
df_body["self_rep"] = (
    df_body[numeric_cols[8:12]]
    .mean(axis=1)
)


# ============================================================
# ⑦ 被験者ごとの身体化感覚平均
# ============================================================

df_body_mean = (
    df_body
    .groupby("氏名")[
        ["ownership", "agency", "self_rep"]
    ]
    .mean()
    .reset_index()
)


# ============================================================
# ⑧ VCあり / なしを横持ちに変換
# ============================================================

df_ult_wide = df_ult.pivot_table(
    index=[
        "氏名",
        "relation",
        "relation_num",
        "被験者の性別"
    ],
    columns="ボイスチェンジャー",
    values="第１ラウンド",
    aggfunc="mean"
).reset_index()


df_ult_wide = df_ult_wide.rename(columns={
    "なし": "absent",
    "あり": "present"
})


# ============================================================
# ⑨ VCあり − VCなし
# ============================================================

df_ult_wide["diff"] = (
    df_ult_wide["present"]
    - df_ult_wide["absent"]
)


# ============================================================
# ⑩ 身体化感覚データと結合
# ============================================================

df_plot = pd.merge(
    df_ult_wide,
    df_body_mean,
    on="氏名"
)


# ============================================================
# ⑪ 男性被験者だけ抽出
# ★ 主体感の高低では分けない
# ============================================================

male_all = df_plot[
    df_plot["被験者の性別"] == "男性"
].copy()


male_participants = (
    male_all[
        ["氏名", "被験者の性別", "agency"]
    ]
    .drop_duplicates(subset="氏名")
    .sort_values("agency")
)


print("\n========================================")
print("★ 分析対象：男性被験者 全員")
print("========================================")

print(
    male_participants.to_string(index=False)
)

print(
    "\n男性被験者数 =",
    male_all["氏名"].nunique()
)


# ============================================================
# ⑫ 男性 → 男性 / 男性 → 女性 に分割
# ============================================================

# 男性被験者 → 男性実験者
male_male = male_all[
    male_all["relation"] == "Same"
].copy()

# 男性被験者 → 女性実験者
male_female = male_all[
    male_all["relation"] == "Different"
].copy()


print("\n========================================")
print("男性被験者の条件別人数")
print("========================================")

print(
    "男性 → 男性 N =",
    len(male_male)
)

print(
    "男性 → 女性 N =",
    len(male_female)
)


# ============================================================
# ⑬ 各条件のデータを表示
# ============================================================

print("\n========================================")
print("男性 → 男性")
print("========================================")

print(
    male_male[
        ["氏名", "agency", "absent", "present", "diff"]
    ].sort_values("agency").to_string(index=False)
)


print("\n========================================")
print("男性 → 女性")
print("========================================")

print(
    male_female[
        ["氏名", "agency", "absent", "present", "diff"]
    ].sort_values("agency").to_string(index=False)
)


# ============================================================
# ⑭ 単純傾き OLS
# ============================================================

def regression_stats(df, label):

    print("\n")
    print("=" * 70)
    print(label)
    print("=" * 70)

    if len(df) < 3:

        print("Not enough data")

        return None

    X = sm.add_constant(
        df["agency"]
    )

    y = df["diff"]

    model = sm.OLS(
        y,
        X
    ).fit()

    print(
        f"N = {len(df)}"
    )

    print(
        f"Mean diff = "
        f"{df['diff'].mean():.3f}"
    )

    print(
        f"β = "
        f"{model.params['agency']:.4f}"
    )

    print(
        f"p = "
        f"{model.pvalues['agency']:.4f}"
    )

    print(
        f"R² = "
        f"{model.rsquared:.4f}"
    )

    print("\n--- OLS Summary ---")

    print(
        model.summary()
    )

    return model


# ============================================================
# ⑮ 男性被験者 → 男性実験者 OLS
# ============================================================

male_male_model = regression_stats(
    male_male,
    "Male Participants - Male Partner"
)


# ============================================================
# ⑯ 男性被験者 → 女性実験者 OLS
# ============================================================

male_female_model = regression_stats(
    male_female,
    "Male Participants - Female Partner"
)


# ============================================================
# ⑰ 男性被験者全員で相手性別との交互作用 OLS
# ============================================================

print("\n")
print("=" * 70)
print("Male Participants Interaction")
print("diff ~ agency * partner gender")
print("=" * 70)


interaction_model = smf.ols(
    "diff ~ agency * relation_num",
    data=male_all
).fit()


print(
    interaction_model.summary()
)


# ============================================================
# ⑱ 交互作用の値
# ============================================================

interaction_term = "agency:relation_num"


print("\n========================================")
print("男性被験者全員：Agency × 相手性別の交互作用")
print("========================================")


print(
    f"β = "
    f"{interaction_model.params[interaction_term]:.4f}"
)

print(
    f"p = "
    f"{interaction_model.pvalues[interaction_term]:.4f}"
)

print(
    f"R² = "
    f"{interaction_model.rsquared:.4f}"
)


# ============================================================
# ⑲ 男性被験者全員：OLS回帰プロット
# ============================================================

plt.figure(
    figsize=(8, 6)
)


# 男性被験者 → 男性実験者
sns.regplot(
    data=male_male,
    x="agency",
    y="diff",
    color="blue",
    label="Male → Male",
    scatter_kws={
        "s": 80
    },
    line_kws={
        "linewidth": 2
    },
    ci=95
)


# 男性被験者 → 女性実験者
sns.regplot(
    data=male_female,
    x="agency",
    y="diff",
    color="red",
    label="Male → Female",
    scatter_kws={
        "s": 80
    },
    line_kws={
        "linewidth": 2
    },
    ci=95
)


# VCあり − VCなし = 0
plt.axhline(
    y=0,
    linestyle="--",
    color="gray",
    linewidth=1.5
)


plt.xlabel(
    "Sense of Agency",
    fontsize=12
)

plt.ylabel(
    "Offer Difference (VC On − Off)",
    fontsize=12
)

plt.title(
    "Male Participants",
    fontsize=14
)


# 縦軸
plt.ylim(-100, 100)

plt.yticks(
    range(-100, 101, 25)
)


plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


# 画像保存
plt.savefig(
    "男性被験者全員_OLS回帰.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# ⑳ 対応のある t 検定
# ============================================================

def paired_t_test(df, label):

    data = df[
        ["present", "absent"]
    ].dropna()

    present = data["present"]
    absent = data["absent"]

    t_stat, p_value = ttest_rel(
        present,
        absent
    )

    difference = (
        present
        - absent
    )

    # Cohen's dz
    if difference.std(ddof=1) != 0:

        cohen_dz = (
            difference.mean()
            / difference.std(ddof=1)
        )

    else:

        cohen_dz = np.nan


    print("\n")
    print("=" * 70)
    print(label)
    print("=" * 70)

    print(
        f"N = {len(data)}"
    )

    print(
        f"VCなし 平均 = "
        f"{absent.mean():.3f}"
    )

    print(
        f"VCあり 平均 = "
        f"{present.mean():.3f}"
    )

    print(
        f"平均差（VCあり − VCなし） = "
        f"{difference.mean():.3f}"
    )

    print(
        f"t({len(data)-1}) = "
        f"{t_stat:.3f}"
    )

    print(
        f"p = "
        f"{p_value:.4f}"
    )

    print(
        f"Cohen's dz = "
        f"{cohen_dz:.3f}"
    )

    if p_value < 0.05:

        print(
            "→ 有意差あり"
        )

    else:

        print(
            "→ 有意差なし"
        )


# ============================================================
# ㉑ 男性被験者 → 男性相手
# ============================================================

paired_t_test(
    male_male,
    "男性被験者 → 男性相手"
)


# ============================================================
# ㉒ 男性被験者 → 女性相手
# ============================================================

paired_t_test(
    male_female,
    "男性被験者 → 女性相手"
)


# ============================================================
# ㉓ 最終確認
# ============================================================

print("\n")
print("=" * 70)
print("★ 最終確認")
print("=" * 70)

print(
    f"男性被験者数 = "
    f"{male_all['氏名'].nunique()}"
)

print(
    f"男性 → 男性 N = "
    f"{len(male_male)}"
)

print(
    f"男性 → 女性 N = "
    f"{len(male_female)}"
)

print("\n※ 主体感による高群・低群への分類は行っていません")