# ============================================================
# import
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm


# ============================================================
# ① データ読み込み
# ============================================================

df_ult = pd.read_csv(
    "最後通牒結果/R1_round.csv"
)

df_body = pd.read_csv(
    "最後通牒結果/2025身体化感覚アンケート.csv"
)


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
# ⑤ VC表記を統一
# ============================================================

df_ult["VC"] = (
    df_ult["ボイスチェンジャー"]
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
# ⑥ 男性被験者のみ抽出
# ============================================================

df_male = df_ult[
    df_ult["被験者の性別"] == "男性"
].copy()


print("\n========================================")
print("男性被験者")
print("========================================")

print(
    df_male["氏名"]
    .drop_duplicates()
    .sort_values()
    .to_string(index=False)
)

print(
    f"\n男性被験者数 = {df_male['氏名'].nunique()}"
)


# ============================================================
# ⑦ Ownershipを計算
#
# 身体化感覚アンケートの最後12項目のうち
# 最初の4項目をOwnershipとして使用
#
# ※中央値による高群・低群分けはしない
# ============================================================

numeric_cols = df_body.columns[-12:]

df_body["ownership"] = (
    df_body[numeric_cols[:4]]
    .mean(axis=1)
)


# 各被験者のOwnershipを1つの値にする

ownership_all = (
    df_body
    .groupby("氏名")["ownership"]
    .mean()
    .reset_index()
)


# 男性被験者のOwnershipだけ確認

male_names = df_male["氏名"].unique()

ownership_male = ownership_all[
    ownership_all["氏名"].isin(male_names)
].copy()


print("\n========================================")
print("男性被験者のOwnership")
print("========================================")

print(
    ownership_male
    .sort_values("ownership")
    .to_string(index=False)
)

print(
    f"\nOwnership N = {len(ownership_male)}"
)


# ============================================================
# ⑧ 赤プロット用データ
#
# VCなし
#
# （男性被験者 → 男性相手）
#             -
# （男性被験者 → 女性相手）
#
# Y赤 = 男性相手 - 女性相手
# ============================================================

red_source = df_male[
    df_male["VC"] == "VCなし"
].copy()


# 男性相手・女性相手を横持ちにする

red_wide = red_source.pivot_table(
    index="氏名",
    columns="実験者の性別",
    values="第１ラウンド",
    aggfunc="mean"
).reset_index()


# ★ 赤の差分
#
# 男性相手 - 女性相手

red_wide["red_diff"] = (
    red_wide["男性"]
    -
    red_wide["女性"]
)


# Ownershipを結合

red_plot = pd.merge(
    red_wide,
    ownership_male,
    on="氏名",
    how="inner"
)


# ============================================================
# ⑨ 青プロット用データ
#
# VCあり
#
# （男性被験者 → 女性相手）
#             -
# （男性被験者 → 男性相手）
#
# Y青 = 女性相手 - 男性相手
# ============================================================

blue_source = df_male[
    df_male["VC"] == "VCあり"
].copy()


# 男性相手・女性相手を横持ちにする

blue_wide = blue_source.pivot_table(
    index="氏名",
    columns="実験者の性別",
    values="第１ラウンド",
    aggfunc="mean"
).reset_index()


# ★ 青の差分
#
# 女性相手 - 男性相手

blue_wide["blue_diff"] = (
    blue_wide["女性"]
    -
    blue_wide["男性"]
)


# Ownershipを結合

blue_plot = pd.merge(
    blue_wide,
    ownership_male,
    on="氏名",
    how="inner"
)


# ============================================================
# ⑩ 赤の計算結果を確認
# ============================================================

print("\n========================================")
print("赤：VCなし")
print("男性相手 - 女性相手")
print("========================================")

print(
    red_plot[
        [
            "氏名",
            "男性",
            "女性",
            "red_diff",
            "ownership"
        ]
    ]
    .sort_values("ownership")
    .to_string(index=False)
)

print(
    f"\n赤 N = {len(red_plot)}"
)


# ============================================================
# ⑪ 青の計算結果を確認
# ============================================================

print("\n========================================")
print("青：VCあり")
print("女性相手 - 男性相手")
print("========================================")

print(
    blue_plot[
        [
            "氏名",
            "女性",
            "男性",
            "blue_diff",
            "ownership"
        ]
    ]
    .sort_values("ownership")
    .to_string(index=False)
)

print(
    f"\n青 N = {len(blue_plot)}"
)


# ============================================================
# ⑫ 赤の線形回帰
#
# X = Ownership
#
# Y =
# VCなし
# （男性→男性相手）-（男性→女性相手）
# ============================================================

red_reg = (
    red_plot[
        [
            "ownership",
            "red_diff"
        ]
    ]
    .dropna()
)


X_red = sm.add_constant(
    red_reg["ownership"]
)

y_red = red_reg["red_diff"]


model_red = sm.OLS(
    y_red,
    X_red
).fit()


print("\n========================================")
print("赤：線形回帰")
print("VCなし：男性相手 - 女性相手")
print("========================================")

print(
    f"N = {len(red_reg)}"
)

print(
    f"β = {model_red.params['ownership']:.3f}"
)

print(
    f"p = {model_red.pvalues['ownership']:.3f}"
)

print(
    f"R² = {model_red.rsquared:.3f}"
)


# ============================================================
# ⑬ 青の線形回帰
#
# X = Ownership
#
# Y =
# VCあり
# （男性→女性相手）-（男性→男性相手）
# ============================================================

blue_reg = (
    blue_plot[
        [
            "ownership",
            "blue_diff"
        ]
    ]
    .dropna()
)


X_blue = sm.add_constant(
    blue_reg["ownership"]
)

y_blue = blue_reg["blue_diff"]


model_blue = sm.OLS(
    y_blue,
    X_blue
).fit()


print("\n========================================")
print("青：線形回帰")
print("VCあり：女性相手 - 男性相手")
print("========================================")

print(
    f"N = {len(blue_reg)}"
)

print(
    f"β = {model_blue.params['ownership']:.3f}"
)

print(
    f"p = {model_blue.pvalues['ownership']:.3f}"
)

print(
    f"R² = {model_blue.rsquared:.3f}"
)


# ============================================================
# ⑭ 結果をまとめて表示
# ============================================================

print("\n========================================")
print("回帰結果まとめ")
print("========================================")

print(
    "赤："
    f"β = {model_red.params['ownership']:.3f}, "
    f"p = {model_red.pvalues['ownership']:.3f}, "
    f"R² = {model_red.rsquared:.3f}"
)

print(
    "青："
    f"β = {model_blue.params['ownership']:.3f}, "
    f"p = {model_blue.pvalues['ownership']:.3f}, "
    f"R² = {model_blue.rsquared:.3f}"
)


# ============================================================
# ⑮ グラフ
#
# ★ グラフは1枚
# ★ 赤と青は別々に回帰
# ============================================================

plt.figure(
    figsize=(8, 6)
)


# ============================================================
# 赤
#
# X = Ownership
#
# Y =
# VCなし
# 男性相手 - 女性相手
# ============================================================

sns.regplot(
    data=red_plot,

    x="ownership",

    y="red_diff",

    color="red",

    label="VC Off: Male - Female",

    scatter_kws={
        "s": 80
    },

    line_kws={
        "linewidth": 2
    }
)


# ============================================================
# 青
#
# X = Ownership
#
# Y =
# VCあり
# 女性相手 - 男性相手
# ============================================================

sns.regplot(
    data=blue_plot,

    x="ownership",

    y="blue_diff",

    color="blue",

    label="VC On: Female - Male",

    scatter_kws={
        "s": 80
    },

    line_kws={
        "linewidth": 2
    }
)


# ============================================================
# 0ライン
# ============================================================

plt.axhline(
    y=0,
    linestyle="--",
    color="gray"
)


# ============================================================
# グラフ設定
# ============================================================

plt.xlabel(
    "Ownership Score"
)

plt.ylabel(
    "Offer Difference"
)

plt.title(
    "Ownership vs. Ultimatum Offer Difference"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()