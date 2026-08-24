# 有意差が出た線形回帰の所有感の結果を見て，所有感が高い被験者だけを抽出して，その被験者の発話データを分析する．
# なお，被験者の性別は分けないものとする

# 各被験者の4条件の所有感を平均 → その16人の値の中央値を出す → 中央値より高い被験者を抽出 → その被験者名を表示 → その人たちだけで、同じ同性・異性の全体線形回帰
# ============================================================
# 高所有感群のみ
# 所有感 × 自己開示差の線形回帰
#
# 高群判定：
#   各被験者の所有感（VCあり・なし）の平均
#   → 全被験者の中央値
#   → 中央値より高い被験者を抽出
#
# 線形回帰：
#   横軸 = 身体化感覚アンケートの所有感
#          （VCあり・なしの平均）
#
#   縦軸 = 自己開示スコア
#          （VCあり − VCなし）
#
#   赤 = 同性相手
#   青 = 異性相手
# ============================================================


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm


# =========================
# ① データ読み込み
# =========================

df_dis = pd.read_csv(
    "お題回答結果/被験者の発話データ/情報単位スコア一覧.csv"
)

df_body = pd.read_csv(
    "お題回答結果/被験者の発話データ/2025身体化感覚アンケート.csv"
)


# =========================
# ② 列名整理
# =========================

df_dis = df_dis.rename(columns={"name": "氏名"})

df_dis.columns = df_dis.columns.str.strip()
df_body.columns = df_body.columns.str.strip()


# =========================
# ③ 名前クリーニング
# =========================

def clean_name(df):

    df["氏名"] = (
        df["氏名"]
        .astype(str)
        .str.strip()
        .str.replace("　", "", regex=False)
        .str.replace(" ", "", regex=False)
    )

    return df


df_dis = clean_name(df_dis)
df_body = clean_name(df_body)


# =========================
# ④ 被験者性別
# =========================

female_names = [
    "﨑谷瑠愛",
    "小島夏美",
    "谷川舞桜",
    "久保田翔帆",
    "小西真結",
    "北野安樹子",
    "渡邉怜",
    "葭田桃花",
    "利光唯",
    "細見涼乃"
]


df_dis["participant_gender"] = np.where(
    df_dis["氏名"].isin(female_names),
    "女性",
    "男性"
)


# =========================
# ⑤ 実験者性別を統一
# =========================

df_dis["experimenter_gender"] = (
    df_dis["experimenter_gender"]
    .replace({
        "男": "男性",
        "女": "女性"
    })
)


# =========================
# ⑥ 同性 / 異性を判定
# =========================

df_dis["relation"] = np.where(
    df_dis["participant_gender"]
    == df_dis["experimenter_gender"],
    "Same",
    "Different"
)


# ============================================================
# ⑦ 所有感を計算
# ============================================================

numeric_cols = df_body.columns[-12:]

# 最初の4項目を所有感として使用
df_body["ownership"] = (
    df_body[numeric_cols[:4]]
    .mean(axis=1)
)


# ============================================================
# ⑧ 各被験者の所有感
#    VCあり・なしを平均
# ============================================================

ownership_mean = (
    df_body
    .groupby("氏名")["ownership"]
    .mean()
    .reset_index()
)


print("\n============================")
print("各被験者の所有感（VCあり・なし平均）")
print("============================")

print(
    ownership_mean
    .sort_values(
        "ownership",
        ascending=False
    )
    .to_string(index=False)
)


# ============================================================
# ⑨ 所有感の中央値
# ============================================================

ownership_median = (
    ownership_mean["ownership"]
    .median()
)


print("\n============================")
print("所有感の中央値")
print("============================")

print(
    f"中央値 = {ownership_median:.3f}"
)


# ============================================================
# ⑩ 中央値より高い被験者
# ============================================================

high_group = ownership_mean[
    ownership_mean["ownership"]
    > ownership_median
].copy()


high_names = high_group["氏名"].tolist()


print("\n============================")
print("所有感が中央値より高い被験者")
print("============================")


for _, row in high_group.sort_values(
    "ownership",
    ascending=False
).iterrows():

    print(
        f"{row['氏名']}："
        f"{row['ownership']:.3f}"
    )


print("\n============================")
print("抽出結果")
print("============================")

print(
    f"全被験者数："
    f"{len(ownership_mean)}"
)

print(
    f"高所有感群："
    f"{len(high_names)}人"
)


print("\n被験者名")

for name in high_names:
    print(name)


# ============================================================
# ⑪ 高所有感群の名前だけ使って
#    自己開示データを抽出
# ============================================================

df_dis_high = df_dis[
    df_dis["氏名"].isin(high_names)
].copy()


# ============================================================
# ⑫ 自己開示を
#    同性 / 異性 × VCあり / なし
#    に分ける
# ============================================================

df_dis_wide = df_dis_high.pivot_table(

    index=[
        "氏名",
        "relation"
    ],

    columns="voice_changer",

    values="info_unit_score",

    aggfunc="mean"

).reset_index()


# VC表記統一
df_dis_wide = df_dis_wide.rename(columns={

    0: "absent",
    1: "present",

    "なし": "absent",
    "あり": "present",

    False: "absent",
    True: "present"
})


# ============================================================
# ⑬ 自己開示差
#
# VCあり − VCなし
# ============================================================

df_dis_wide["diff"] = (
    df_dis_wide["present"]
    - df_dis_wide["absent"]
)


# ============================================================
# ⑭ 横軸用の所有感を作る
#
# ★ 2025身体化感覚アンケート.csv から取得
# ★ VCあり・なしの平均
# ============================================================

ownership_x = (
    df_body[
        df_body["氏名"].isin(high_names)
    ]
    .groupby("氏名")["ownership"]
    .mean()
    .reset_index()
)


# 分かりやすく列名変更
ownership_x = ownership_x.rename(
    columns={
        "ownership": "ownership_x"
    }
)


print("\n============================")
print("線形回帰のX軸に使用する所有感")
print("（身体化感覚アンケートのVCあり・なし平均）")
print("============================")

print(
    ownership_x
    .sort_values(
        "ownership_x",
        ascending=False
    )
    .to_string(index=False)
)


# ============================================================
# ⑮ 自己開示差と所有感を結合
# ============================================================

df_plot = pd.merge(

    df_dis_wide,

    ownership_x,

    on="氏名",

    how="inner"
)


# ============================================================
# ⑯ 最終確認
# ============================================================

print("\n============================")
print("線形回帰に使用するデータ")
print("============================")

print(
    df_plot[
        [
            "氏名",
            "relation",
            "ownership_x",
            "absent",
            "present",
            "diff"
        ]
    ]
    .sort_values(
        [
            "relation",
            "ownership_x"
        ]
    )
    .to_string(index=False)
)


# ============================================================
# ⑰ 同性 / 異性
# ============================================================

same_df = df_plot[
    df_plot["relation"] == "Same"
].copy()


different_df = df_plot[
    df_plot["relation"] == "Different"
].copy()


print("\n============================")
print("同性相手データ")
print("============================")

print(
    same_df[
        [
            "氏名",
            "ownership_x",
            "diff"
        ]
    ].to_string(index=False)
)


print("\n============================")
print("異性相手データ")
print("============================")

print(
    different_df[
        [
            "氏名",
            "ownership_x",
            "diff"
        ]
    ].to_string(index=False)
)


# ============================================================
# ⑱ 線形回帰
# ============================================================

def regression_stats(df, label):

    X = df["ownership_x"]
    y = df["diff"]

    X = sm.add_constant(X)

    model = sm.OLS(
        y,
        X
    ).fit()


    print(f"\n--- {label} ---")

    print(
        f"N        : "
        f"{len(df)}"
    )

    print(
        f"β (slope): "
        f"{model.params['ownership_x']:.3f}"
    )

    print(
        f"p-value  : "
        f"{model.pvalues['ownership_x']:.3f}"
    )

    print(
        f"R²       : "
        f"{model.rsquared:.3f}"
    )


print("\n============================")
print("高所有感群のみの線形回帰")
print("============================")


regression_stats(
    same_df,
    "同性相手"
)


regression_stats(
    different_df,
    "異性相手"
)


# ============================================================
# ⑲ グラフ
#
# 赤 = 同性相手
# 青 = 異性相手
# ============================================================

plt.figure(
    figsize=(8, 6)
)


# 同性相手 = 赤
sns.regplot(
    data=same_df,
    x="ownership_x",
    y="diff",
    color="red",
    label="Same Gender",
    scatter_kws={
        "s": 80
    }
)


# 異性相手 = 青
sns.regplot(
    data=different_df,
    x="ownership_x",
    y="diff",
    color="blue",
    label="Different Gender",
    scatter_kws={
        "s": 80
    }
)


# VCあり − なし = 0
plt.axhline(
    0,
    linestyle="--",
    color="gray"
)


plt.xlabel(
    "Ownership Score (Mean of VC On and Off)"
)

plt.ylabel(
    "Self-Disclosure Difference (VC On − Off)"
)

plt.title(
    "Ownership vs Self-Disclosure\n"
    "(High Ownership Group)"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()