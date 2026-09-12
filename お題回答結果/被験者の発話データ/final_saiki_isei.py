# ============================================================
# 仮想的に異性の場合
# 所有感 × 自己開示スコアの線形回帰
#
# 所有感の中央値は取らない
# 高所有感群 / 低所有感群には分けない
# 全被験者をそのまま使用
# 被験者の性別は分けずに合算
#
# 【赤プロット】
#   VCなし
#   男性被験者 → 女性相手
#   女性被験者 → 男性相手
#
# 【青プロット】
#   VCあり
#   男性被験者 → 男性相手
#   女性被験者 → 女性相手
#
# 横軸 = 所有感
# 縦軸 = 自己開示スコア
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
        .str.replace(
            "　",
            "",
            regex=False
        )
        .str.replace(
            " ",
            "",
            regex=False
        )
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
#
# 中央値は取らない
# 高群 / 低群には分けない
# 全被験者を使用
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
# ⑨ 赤プロットを抽出
#
# VCなし
#
# 男性被験者 → 女性相手
# 女性被験者 → 男性相手
#
# 男女は分けずに合算
# ============================================================

red_df = df_dis[

    (df_dis["VC"] == "VCなし")

    &

    (
        # 男性 → 女性
        (
            (df_dis["participant_gender"] == "男性")
            &
            (df_dis["experimenter_gender"] == "女性")
        )

        |

        # 女性 → 男性
        (
            (df_dis["participant_gender"] == "女性")
            &
            (df_dis["experimenter_gender"] == "男性")
        )
    )

].copy()


# ============================================================
# ⑩ 青プロットを抽出
#
# VCあり
#
# 男性被験者 → 男性相手
# 女性被験者 → 女性相手
#
# 男女は分けずに合算
# ============================================================

blue_df = df_dis[

    (df_dis["VC"] == "VCあり")

    &

    (
        # 男性 → 男性
        (
            (df_dis["participant_gender"] == "男性")
            &
            (df_dis["experimenter_gender"] == "男性")
        )

        |

        # 女性 → 女性
        (
            (df_dis["participant_gender"] == "女性")
            &
            (df_dis["experimenter_gender"] == "女性")
        )
    )

].copy()


# ============================================================
# ⑪ 所有感を結合
# ============================================================

red_plot = pd.merge(
    red_df,
    ownership_all,
    on="氏名",
    how="inner"
)


blue_plot = pd.merge(
    blue_df,
    ownership_all,
    on="氏名",
    how="inner"
)


# ============================================================
# ⑫ 線形回帰に使用するデータを確認
# ============================================================

print("\n========================================")
print("赤：VCなし 男性→女性 ＋ 女性→男性")
print("========================================")

print(
    red_plot[
        [
            "氏名",
            "participant_gender",
            "experimenter_gender",
            "ownership",
            "info_unit_score"
        ]
    ]
    .sort_values("ownership")
    .to_string(index=False)
)

print(
    f"\n赤 N = {len(red_plot)}"
)


print("\n========================================")
print("青：VCあり 男性→男性 ＋ 女性→女性")
print("========================================")

print(
    blue_plot[
        [
            "氏名",
            "participant_gender",
            "experimenter_gender",
            "ownership",
            "info_unit_score"
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
# ============================================================

def regression_stats(df, label):

    data = (
        df[
            [
                "ownership",
                "info_unit_score"
            ]
        ]
        .dropna()
    )

    # 横軸：所有感
    X = data["ownership"]

    # 縦軸：自己開示スコア
    y = data["info_unit_score"]

    # 切片を追加
    X = sm.add_constant(X)

    # OLS線形回帰
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
print("所有感 × 自己開示スコア")
print("全被験者・中央値による分類なし")
print("========================================")


model_red = regression_stats(
    red_plot,
    "赤：VCなし 男性→女性 ＋ 女性→男性"
)


model_blue = regression_stats(
    blue_plot,
    "青：VCあり 男性→男性 ＋ 女性→女性"
)


# ============================================================
# ⑮ グラフ
# ============================================================

plt.figure(
    figsize=(8, 6)
)


# ------------------------------------------------------------
# 赤
#
# VCなし
# 男性 → 女性
# 女性 → 男性
# ------------------------------------------------------------

sns.regplot(
    data=red_plot,

    x="ownership",

    y="info_unit_score",

    color="red",

    label="Natural Voice: Different Gender",

    scatter_kws={
        "s": 80
    },

    line_kws={
        "linewidth": 2
    }
)


# ------------------------------------------------------------
# 青
#
# VCあり
# 男性 → 男性
# 女性 → 女性
# ------------------------------------------------------------

sns.regplot(
    data=blue_plot,

    x="ownership",

    y="info_unit_score",

    color="blue",

    label="VC: Virtual Different Gender",

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

plt.xlabel(
    "Ownership Score"
)

plt.ylabel(
    "Self-Disclosure Score"
)

plt.title(
    "Ownership vs. Self-Disclosure"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()