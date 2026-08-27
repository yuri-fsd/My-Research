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

# 男性被験者だけに絞った後は
# Same      = 男性 → 男性
# Different = 男性 → 女性

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


print("\n========================================")
print("全データ")
print("========================================")

print(
    "Total N =",
    len(df_plot)
)


# ============================================================
# ⑪ 男女20名全体から主体感中央値を計算
#
# ★ ここでは男性だけにしない
# ★ 男性10名 + 女性10名で中央値を算出
# ============================================================

participant_agency = (
    df_plot[
        ["氏名", "被験者の性別", "agency"]
    ]
    .drop_duplicates(subset="氏名")
)


agency_median = participant_agency["agency"].median()


print("\n========================================")
print("全被験者の主体感中央値")
print("========================================")

print(
    "被験者数 =",
    len(participant_agency)
)

print(
    f"Agency Median = {agency_median:.3f}"
)


# ============================================================
# ⑫ ★全20名から主体感が中央値より高い被験者を抽出
#
# ★ < ではなく > になっていることが重要
# ============================================================

high_agency_names = participant_agency.loc[
    participant_agency["agency"] > agency_median,
    "氏名"
]


high_agency_all = df_plot[
    df_plot["氏名"].isin(high_agency_names)
].copy()


print("\n========================================")
print("全被験者：高主体感群")
print("========================================")

print(
    "高主体感被験者数 =",
    high_agency_all["氏名"].nunique()
)


# ============================================================
# ⑬ 高主体感群の全被験者を表示
# ============================================================

high_participants_all = (
    high_agency_all[
        ["氏名", "被験者の性別", "agency"]
    ]
    .drop_duplicates(subset="氏名")
    .sort_values("agency")
)


print("\n========================================")
print("全被験者：高主体感群")
print("========================================")

print(
    high_participants_all.to_string(index=False)
)


# ============================================================
# ⑭ ★高主体感群から男性被験者だけ抽出
# ============================================================

male_high_agency = high_agency_all[
    high_agency_all["被験者の性別"] == "男性"
].copy()


male_high_participants = (
    male_high_agency[
        ["氏名", "被験者の性別", "agency"]
    ]
    .drop_duplicates(subset="氏名")
    .sort_values("agency")
)


print("\n========================================")
print("★ 最終分析対象：高主体感 × 男性被験者")
print("========================================")

print(
    male_high_participants.to_string(index=False)
)

print(
    "\n男性被験者数 =",
    male_high_agency["氏名"].nunique()
)


# ============================================================
# ⑮ 男性 → 男性 / 男性 → 女性 に分割
# ============================================================

# 男性被験者 → 男性実験者
male_male = male_high_agency[
    male_high_agency["relation"] == "Same"
].copy()


# 男性被験者 → 女性実験者
male_female = male_high_agency[
    male_high_agency["relation"] == "Different"
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
# ⑯ 単純傾き OLS
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
# ⑰ 高主体感男性 → 男性実験者 OLS
# ============================================================

male_male_model = regression_stats(
    male_male,
    "High Agency Male Participants - Male Partner"
)


# ============================================================
# ⑱ 高主体感男性 → 女性実験者 OLS
# ============================================================

male_female_model = regression_stats(
    male_female,
    "High Agency Male Participants - Female Partner"
)


# ============================================================
# ⑲ 高主体感の男性被験者のみで交互作用 OLS
#
# diff ~ agency × 相手性別
#
# relation_num
# 0 = 男性相手
# 1 = 女性相手
# ============================================================

print("\n")
print("=" * 70)
print("High Agency Male Participants Interaction")
print("diff ~ agency * partner gender")
print("=" * 70)


interaction_model = smf.ols(
    "diff ~ agency * relation_num",
    data=male_high_agency
).fit()


print(
    interaction_model.summary()
)


# ============================================================
# ⑳ 交互作用の値
# ============================================================

interaction_term = "agency:relation_num"


print("\n========================================")
print("高主体感・男性被験者のみ：交互作用")
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
# ㉑ 高主体感・男性被験者のみ：OLS回帰プロット
# ============================================================

plt.figure(
    figsize=(8, 6)
)


# ============================================================
# 男性被験者 → 男性実験者
# ============================================================

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


# ============================================================
# 男性被験者 → 女性実験者
# ============================================================

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
    "High Agency - Male Participants",
    fontsize=14
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


# 画像保存
plt.savefig(
    "高主体感群_男性被験者_OLS回帰.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# ㉒ 対応のある t 検定
# ============================================================

def paired_t_test(df, label):

    data = df[
        ["present", "absent"]
    ].dropna()


    present = data["present"]
    absent = data["absent"]


    # 対応のある t 検定
    t_stat, p_value = ttest_rel(
        present,
        absent
    )


    # VCあり − VCなし
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
# ㉓ 高主体感男性 → 男性相手
# ============================================================

paired_t_test(
    male_male,
    "高主体感：男性被験者 → 男性相手"
)


# ============================================================
# ㉔ 高主体感男性 → 女性相手
# ============================================================

paired_t_test(
    male_female,
    "高主体感：男性被験者 → 女性相手"
)


# ============================================================
# ㉕ 最終確認
# ============================================================

print("\n")
print("=" * 70)
print("★ 最終確認")
print("=" * 70)


print(
    f"全被験者数 = "
    f"{participant_agency['氏名'].nunique()}"
)

print(
    f"全20名から算出した主体感中央値 = "
    f"{agency_median:.3f}"
)

print(
    f"中央値より高い全被験者数 = "
    f"{high_agency_all['氏名'].nunique()}"
)

print(
    f"分析対象の男性被験者数 = "
    f"{male_high_agency['氏名'].nunique()}"
)

print(
    f"男性 → 男性 N = "
    f"{len(male_male)}"
)

print(
    f"男性 → 女性 N = "
    f"{len(male_female)}"
)