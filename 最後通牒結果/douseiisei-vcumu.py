# ============================================================
# 所有感 × 相手性別による自己開示スコア差
#
# 全被験者を使用（男女合算）
# 所有感の中央値による分類なし
#
# 【赤プロット：地声】
#   同性相手 - 異性相手
#
#   男性被験者：
#       男性相手 - 女性相手
#
#   女性被験者：
#       女性相手 - 男性相手
#
#
# 【青プロット：VC】
#   仮想的同性相手 - 仮想的異性相手
#
#   VCによって被験者の声の性別が反転するため、
#
#   男性被験者（VC後は女性声）：
#       女性相手 - 男性相手
#
#   女性被験者（VC後は男性声）：
#       男性相手 - 女性相手
#
#
# 横軸 = 所有感
# 縦軸 = 自己開示スコア差
# ============================================================


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm


# ============================================================
# ① データ読み込み
# ============================================================

df_dis = pd.read_csv(
    "お題回答結果/被験者の発話データ/情報単位スコア一覧.csv"
)

df_body = pd.read_csv(
    "お題回答結果/被験者の発話データ/2025身体化感覚アンケート.csv"
)


# ============================================================
# ② 列名整理
# ============================================================

df_dis = df_dis.rename(
    columns={
        "name": "氏名"
    }
)

df_dis.columns = df_dis.columns.str.strip()
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


df_dis = clean_name(df_dis)
df_body = clean_name(df_body)


# ============================================================
# ④ 被験者の性別
# ============================================================

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


# ============================================================
# ⑤ 実験者性別を統一
# ============================================================

df_dis["experimenter_gender"] = (
    df_dis["experimenter_gender"]
    .replace({
        "男": "男性",
        "女": "女性"
    })
)


# ============================================================
# ⑥ VC表記を統一
# ============================================================

df_dis["VC"] = (
    df_dis["voice_changer"]
    .replace({
        0: "VCなし",
        1: "VCあり",
        "なし": "VCなし",
        "あり": "VCあり",
        False: "VCなし",
        True: "VCあり"
    })
)


# ============================================================
# ⑦ 所有感を計算
#
# 身体化感覚アンケートの最後12項目のうち
# 最初の4項目を所有感として使用
# ============================================================

numeric_cols = df_body.columns[-12:]

df_body["ownership"] = (
    df_body[numeric_cols[:4]]
    .mean(axis=1)
)


# ============================================================
# ⑧ 各被験者の所有感
# ============================================================

ownership_all = (
    df_body
    .groupby("氏名")["ownership"]
    .mean()
    .reset_index()
)


print("\n========================================")
print("全被験者の所有感")
print("========================================")

print(
    ownership_all
    .sort_values("ownership")
    .to_string(index=False)
)

print(
    f"\n全被験者数 = {len(ownership_all)}"
)


# ============================================================
# ⑨ 地声（VCなし）の自己開示差を計算
#
# 同性相手 - 異性相手
#
# 男性：
#   男性相手 - 女性相手
#
# 女性：
#   女性相手 - 男性相手
# ============================================================

natural_df = df_dis[
    df_dis["VC"] == "VCなし"
].copy()


def calc_natural_difference(group):

    participant_gender = group["participant_gender"].iloc[0]

    # ----------------------------
    # 男性被験者
    # 男性相手 - 女性相手
    # ----------------------------

    if participant_gender == "男性":

        same_score = group.loc[
            group["experimenter_gender"] == "男性",
            "info_unit_score"
        ].mean()

        different_score = group.loc[
            group["experimenter_gender"] == "女性",
            "info_unit_score"
        ].mean()

    # ----------------------------
    # 女性被験者
    # 女性相手 - 男性相手
    # ----------------------------

    else:

        same_score = group.loc[
            group["experimenter_gender"] == "女性",
            "info_unit_score"
        ].mean()

        different_score = group.loc[
            group["experimenter_gender"] == "男性",
            "info_unit_score"
        ].mean()


    return pd.Series({

        "participant_gender":
            participant_gender,

        "same_score":
            same_score,

        "different_score":
            different_score,

        # 同性 - 異性
        "difference":
            same_score - different_score
    })


natural_diff = (
    natural_df
    .groupby("氏名")
    .apply(calc_natural_difference)
    .reset_index()
)


# ============================================================
# ⑩ VCありの自己開示差を計算
#
# 仮想的同性相手 - 仮想的異性相手
#
# ★ VCで声の性別が反転することに注意
#
# 男性被験者
#   → VC後は女性声
#   → 女性相手が「仮想的同性」
#   → 男性相手が「仮想的異性」
#
#   女性相手 - 男性相手
#
#
# 女性被験者
#   → VC後は男性声
#   → 男性相手が「仮想的同性」
#   → 女性相手が「仮想的異性」
#
#   男性相手 - 女性相手
# ============================================================

vc_df = df_dis[
    df_dis["VC"] == "VCあり"
].copy()


def calc_vc_difference(group):

    participant_gender = group["participant_gender"].iloc[0]

    # ----------------------------
    # 男性被験者
    #
    # VC後は女性声
    #
    # 仮想的同性 = 女性相手
    # 仮想的異性 = 男性相手
    # ----------------------------

    if participant_gender == "男性":

        virtual_same_score = group.loc[
            group["experimenter_gender"] == "女性",
            "info_unit_score"
        ].mean()

        virtual_different_score = group.loc[
            group["experimenter_gender"] == "男性",
            "info_unit_score"
        ].mean()

    # ----------------------------
    # 女性被験者
    #
    # VC後は男性声
    #
    # 仮想的同性 = 男性相手
    # 仮想的異性 = 女性相手
    # ----------------------------

    else:

        virtual_same_score = group.loc[
            group["experimenter_gender"] == "男性",
            "info_unit_score"
        ].mean()

        virtual_different_score = group.loc[
            group["experimenter_gender"] == "女性",
            "info_unit_score"
        ].mean()


    return pd.Series({

        "participant_gender":
            participant_gender,

        "virtual_same_score":
            virtual_same_score,

        "virtual_different_score":
            virtual_different_score,

        # 仮想的同性 - 仮想的異性
        "difference":
            virtual_same_score - virtual_different_score
    })


vc_diff = (
    vc_df
    .groupby("氏名")
    .apply(calc_vc_difference)
    .reset_index()
)


# ============================================================
# ⑪ Ownershipを結合
# ============================================================

red_plot = pd.merge(
    natural_diff,
    ownership_all,
    on="氏名",
    how="inner"
)


blue_plot = pd.merge(
    vc_diff,
    ownership_all,
    on="氏名",
    how="inner"
)


# ============================================================
# ⑫ 実際に計算された差を確認
# ============================================================

print("\n========================================")
print("🔴 地声：同性相手 - 異性相手")
print("========================================")

print(
    red_plot[
        [
            "氏名",
            "participant_gender",
            "ownership",
            "same_score",
            "different_score",
            "difference"
        ]
    ]
    .sort_values("ownership")
    .to_string(index=False)
)

print(
    f"\n赤 N = {len(red_plot)}"
)


print("\n========================================")
print("🔵 VC：仮想的同性相手 - 仮想的異性相手")
print("========================================")

print(
    blue_plot[
        [
            "氏名",
            "participant_gender",
            "ownership",
            "virtual_same_score",
            "virtual_different_score",
            "difference"
        ]
    ]
    .sort_values("ownership")
    .to_string(index=False)
)

print(
    f"\n青 N = {len(blue_plot)}"
)


# ============================================================
# ⑬ 線形回帰
#
# 横軸：Ownership
# 縦軸：自己開示スコア差
# ============================================================

def regression_stats(df, label):

    data = (
        df[
            [
                "ownership",
                "difference"
            ]
        ]
        .dropna()
    )

    X = data["ownership"]

    y = data["difference"]

    X = sm.add_constant(X)

    model = sm.OLS(
        y,
        X
    ).fit()


    print(
        f"\n--- {label} ---"
    )

    print(
        f"N         : {len(data)}"
    )

    print(
        f"切片      : "
        f"{model.params['const']:.3f}"
    )

    print(
        f"β (slope) : "
        f"{model.params['ownership']:.3f}"
    )

    print(
        f"p-value   : "
        f"{model.pvalues['ownership']:.3f}"
    )

    print(
        f"R²        : "
        f"{model.rsquared:.3f}"
    )

    return model


# ============================================================
# ⑭ 回帰結果
# ============================================================

print("\n========================================")
print("所有感 × 相手性別による自己開示スコア差")
print("全被験者・男女合算")
print("========================================")


model_red = regression_stats(
    red_plot,
    "🔴 地声：同性相手 - 異性相手"
)


model_blue = regression_stats(
    blue_plot,
    "🔵 VC：仮想的同性相手 - 仮想的異性相手"
)


# ============================================================
# ⑮ グラフ
# ============================================================

plt.figure(
    figsize=(8, 6)
)


# ------------------------------------------------------------
# 🔴 地声
# 同性相手 - 異性相手
# ------------------------------------------------------------

sns.regplot(
    data=red_plot,

    x="ownership",

    y="difference",

    color="red",

    label="Natural Voice: Same - Different Gender",

    scatter_kws={
        "s": 80
    },

    line_kws={
        "linewidth": 2
    }
)


# ------------------------------------------------------------
# 🔵 VC
# 仮想的同性相手 - 仮想的異性相手
# ------------------------------------------------------------

sns.regplot(
    data=blue_plot,

    x="ownership",

    y="difference",

    color="blue",

    label="VC: Virtual Same - Virtual Different Gender",

    scatter_kws={
        "s": 80
    },

    line_kws={
        "linewidth": 2
    }
)


# ============================================================
# ⑯ グラフ設定
# ============================================================

plt.axhline(
    y=0,
    color="gray",
    linestyle="--",
    linewidth=1
)

plt.xlabel(
    "Ownership Score"
)

plt.ylabel(
    "Self-Disclosure Difference Score"
)

plt.title(
    "Ownership vs. Self-Disclosure Difference"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()