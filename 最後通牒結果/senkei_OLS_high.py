import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

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

# interaction用
# Same = 0
# Different = 1
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
# ⑧ VCあり / なし を横持ちに変換
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


# 列名変更
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
print("Total N =", len(df_plot))


# ============================================================
# ⑪ 男女すべての被験者から主体感中央値を計算
# ============================================================

# 同じ被験者がSame/Differentで複数行あるため、
# 被験者1人につき1つの主体感だけ取り出して中央値を計算する
participant_agency = (
    df_plot[
        ["氏名", "被験者の性別", "agency"]
    ]
    .drop_duplicates(subset="氏名")
)


agency_median = participant_agency["agency"].median()


print("\n========================================")
print("主体感の中央値")
print("========================================")

print(f"被験者数 = {len(participant_agency)}")
print(f"Agency Median = {agency_median:.3f}")


# ============================================================
# ⑫ 主体感が中央値より高い被験者だけ抽出
# ============================================================
high_agency_names = participant_agency.loc[
    participant_agency["agency"] > agency_median,
    "氏名"
]


high_agency = df_plot[
    df_plot["氏名"].isin(high_agency_names)
].copy()


print("\n========================================")
print("高主体感群")
print("========================================")

print(
    "高主体感被験者数 =",
    high_agency["氏名"].nunique()
)

print(
    "高主体感群データ数 =",
    len(high_agency)
)


# ============================================================
# ⑬ 高主体感群に誰が入っているか確認
# ============================================================
high_participants = (
    high_agency[
        ["氏名", "被験者の性別", "agency"]
    ]
    .drop_duplicates(subset="氏名")
    .sort_values("agency")
)


print("\n========================================")
print("高主体感群の被験者")
print("========================================")

print(high_participants.to_string(index=False))


# ============================================================
# ⑭ Same / Different に分割
# ============================================================
same_df = high_agency[
    high_agency["relation"] == "Same"
].copy()


different_df = high_agency[
    high_agency["relation"] == "Different"
].copy()


print("\nSame Gender N =", len(same_df))
print("Different Gender N =", len(different_df))


# ============================================================
# ⑮ 単純傾き OLS
# ============================================================
def regression_stats(df, label):

    print("\n")
    print("=" * 70)
    print(label)
    print("=" * 70)

    if len(df) < 3:
        print("Not enough data")
        return

    X = sm.add_constant(df["agency"])
    y = df["diff"]

    model = sm.OLS(
        y,
        X
    ).fit()

    print(f"N = {len(df)}")
    print(f"Mean diff = {df['diff'].mean():.3f}")

    print(
        f"β = {model.params['agency']:.4f}"
    )

    print(
        f"p = {model.pvalues['agency']:.4f}"
    )

    print(
        f"R² = {model.rsquared:.4f}"
    )

    print("\n--- OLS Summary ---")
    print(model.summary())


# ============================================================
# ⑯ Same Gender のOLS
# ============================================================
regression_stats(
    same_df,
    "High Agency Group - Same Gender"
)


# ============================================================
# ⑰ Different Gender のOLS
# ============================================================
regression_stats(
    different_df,
    "High Agency Group - Different Gender"
)


# ============================================================
# ⑱ 高主体感群で interaction OLS
#
# diff ~ agency × relation
#
# Same = 0
# Different = 1
# ============================================================
print("\n")
print("=" * 70)
print("High Agency Group Interaction")
print("diff ~ agency * relation")
print("=" * 70)


interaction_model = smf.ols(
    "diff ~ agency * relation_num",
    data=high_agency
).fit()


print(interaction_model.summary())


# ============================================================
# ⑲ interactionの重要な値だけ表示
# ============================================================
interaction_term = "agency:relation_num"


print("\n========================================")
print("交互作用")
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
# ⑳ 高主体感群：OLS回帰プロット
# ============================================================
plt.figure(
    figsize=(8, 6)
)


# -------------------------
# Same Gender
# -------------------------
sns.regplot(
    data=same_df,
    x="agency",
    y="diff",
    color="blue",
    label="Same Gender",
    scatter_kws={
        "s": 80
    },
    line_kws={
        "linewidth": 2
    },
    ci=95
)


# -------------------------
# Different Gender
# -------------------------
sns.regplot(
    data=different_df,
    x="agency",
    y="diff",
    color="red",
    label="Different Gender",
    scatter_kws={
        "s": 80
    },
    line_kws={
        "linewidth": 2
    },
    ci=95
)


# VCあり−なし = 0
plt.axhline(
    y=0,
    linestyle="--",
    color="gray",
    linewidth=1.5
)


# 軸
plt.xlabel(
    "Sense of Agency",
    fontsize=12
)

plt.ylabel(
    "Offer Difference (VC On − Off)",
    fontsize=12
)


# タイトル
plt.title(
    "High Agency Group",
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
    "高主体感群_OLS回帰.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()

# ============================================================
# ㉑ 高主体感群：Excel箱ひげ図用データ作成
# ============================================================

# 男性被験者 × 男性実験者
male_male = high_agency[
    (high_agency["被験者の性別"] == "男性") &
    (high_agency["relation"] == "Same")
]

# 男性被験者 × 女性実験者
male_female = high_agency[
    (high_agency["被験者の性別"] == "男性") &
    (high_agency["relation"] == "Different")
]

# 女性被験者 × 女性実験者
female_female = high_agency[
    (high_agency["被験者の性別"] == "女性") &
    (high_agency["relation"] == "Same")
]

# 女性被験者 × 男性実験者
female_male = high_agency[
    (high_agency["被験者の性別"] == "女性") &
    (high_agency["relation"] == "Different")
]


# ============================================================
# ㉒ 各条件のVCあり・なしを列にする
# ============================================================

boxplot_excel = pd.DataFrame({

    # 男性被験者 → 男性相手
    "男性→男性 VCあり":
        male_male["present"].reset_index(drop=True),

    "男性→男性 VCなし":
        male_male["absent"].reset_index(drop=True),

    # 男性被験者 → 女性相手
    "男性→女性 VCあり":
        male_female["present"].reset_index(drop=True),

    "男性→女性 VCなし":
        male_female["absent"].reset_index(drop=True),

    # 女性被験者 → 女性相手
    "女性→女性 VCあり":
        female_female["present"].reset_index(drop=True),

    "女性→女性 VCなし":
        female_female["absent"].reset_index(drop=True),

    # 女性被験者 → 男性相手
    "女性→男性 VCあり":
        female_male["present"].reset_index(drop=True),

    "女性→男性 VCなし":
        female_male["absent"].reset_index(drop=True)
})


# ============================================================
# ㉓ 確認
# ============================================================

print("\n========================================")
print("高主体感群：箱ひげ図用データ")
print("========================================")

print(boxplot_excel)


print("\n各条件の人数")

print(
    "男性→男性:",
    len(male_male)
)

print(
    "男性→女性:",
    len(male_female)
)

print(
    "女性→女性:",
    len(female_female)
)

print(
    "女性→男性:",
    len(female_male)
)


# ============================================================
# ㉔ CSV保存
# ============================================================

boxplot_excel.to_csv(
    "高主体感群_箱ひげ図用.csv",
    index=False,
    encoding="utf-8-sig"
)


print("\n========================================")
print("保存完了")
print("========================================")

print("高主体感群_箱ひげ図用.csv")