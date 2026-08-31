# ============================================================
# 0. import
# ============================================================

import pandas as pd
import numpy as np
from tabulate import tabulate


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
# ⑤ 同性 / 異性を判定
# ============================================================

df_ult["relation"] = np.where(
    df_ult["被験者の性別"] == df_ult["実験者の性別"],
    "同性",
    "異性"
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
# ⑦ 被験者ごとの主体感平均
# ============================================================

participant_agency = (
    df_body
    .groupby("氏名")["agency"]
    .mean()
    .reset_index()
)


# ============================================================
# ⑧ 被験者性別を追加
# ============================================================

participant_gender = (
    df_ult[
        [
            "氏名",
            "被験者の性別"
        ]
    ]
    .drop_duplicates(subset="氏名")
)


participant_agency = pd.merge(
    participant_agency,
    participant_gender,
    on="氏名",
    how="left"
)


# ============================================================
# ⑨ 全20名の主体感中央値
# ============================================================

agency_median = participant_agency[
    "agency"
].median()


print("\n")
print("=" * 90)
print("主体感中央値")
print("=" * 90)

print(
    f"被験者数 = {len(participant_agency)}"
)

print(
    f"Agency Median = {agency_median:.3f}"
)


# ============================================================
# ⑩ 高主体感群 / 低主体感群
# ============================================================

high_names = participant_agency.loc[
    participant_agency["agency"] > agency_median,
    "氏名"
].tolist()


low_names = participant_agency.loc[
    participant_agency["agency"] < agency_median,
    "氏名"
].tolist()


print("\n")
print("=" * 90)
print("群分け")
print("=" * 90)

print(
    f"高主体感群 = {len(high_names)} 名"
)

print(
    f"低主体感群 = {len(low_names)} 名"
)


# ============================================================
# ⑪ 比較表を作る関数
#
# 比較条件
#
#   地声 × 同性相手
#          VS
#   VCあり × 異性相手
#
# 使用データ
#   最後通牒ゲーム「第1ラウンド」の提示額
# ============================================================

def make_comparison_table(
    target_names,
    group_name
):

    # --------------------------------------------------------
    # 対象群だけ抽出
    # --------------------------------------------------------

    df_group = df_ult[
        df_ult["氏名"].isin(target_names)
    ].copy()


    # --------------------------------------------------------
    # 地声 × 同性相手
    # --------------------------------------------------------

    natural_same = df_group[
        (df_group["ボイスチェンジャー"] == "なし")
        &
        (df_group["relation"] == "同性")
    ][
        [
            "氏名",
            "被験者の性別",
            "第１ラウンド"
        ]
    ].copy()


    natural_same = natural_same.rename(
        columns={
            "第１ラウンド":
            "地声×同性相手"
        }
    )


    # --------------------------------------------------------
    # VCあり × 異性相手
    # --------------------------------------------------------

    vc_different = df_group[
        (df_group["ボイスチェンジャー"] == "あり")
        &
        (df_group["relation"] == "異性")
    ][
        [
            "氏名",
            "第１ラウンド"
        ]
    ].copy()


    vc_different = vc_different.rename(
        columns={
            "第１ラウンド":
            "VC×異性相手"
        }
    )


    # --------------------------------------------------------
    # 2条件を結合
    # --------------------------------------------------------

    comparison = pd.merge(
        natural_same,
        vc_different,
        on="氏名",
        how="inner"
    )


    # --------------------------------------------------------
    # 差
    #
    # VCあり×異性 − 地声×同性
    # --------------------------------------------------------

    comparison["差"] = (
        comparison["VC×異性相手"]
        -
        comparison["地声×同性相手"]
    )


    # --------------------------------------------------------
    # 絶対差
    # --------------------------------------------------------

    comparison["絶対差"] = (
        comparison["差"].abs()
    )


    # --------------------------------------------------------
    # 主体感を追加
    # --------------------------------------------------------

    comparison = pd.merge(
        comparison,
        participant_agency[
            [
                "氏名",
                "agency"
            ]
        ],
        on="氏名",
        how="left"
    )


    # --------------------------------------------------------
    # 主体感の低い順に並べる
    # --------------------------------------------------------

    comparison = (
        comparison
        .sort_values("agency")
        .reset_index(drop=True)
    )


    # ========================================================
    # 表示用データ
    # ========================================================

    display_table = comparison[
        [
            "氏名",
            "被験者の性別",
            "agency",
            "地声×同性相手",
            "VC×異性相手",
            "差",
            "絶対差"
        ]
    ].copy()


    # ========================================================
    # 平均値
    # ========================================================

    natural_mean = comparison[
        "地声×同性相手"
    ].mean()


    vc_mean = comparison[
        "VC×異性相手"
    ].mean()


    diff_mean = comparison[
        "差"
    ].mean()


    abs_diff_mean = comparison[
        "絶対差"
    ].mean()


    agency_mean = comparison[
        "agency"
    ].mean()


    # ========================================================
    # 平均行
    # ========================================================

    mean_row = pd.DataFrame({
        "氏名": [
            "平均"
        ],

        "被験者の性別": [
            "-"
        ],

        "agency": [
            agency_mean
        ],

        "地声×同性相手": [
            natural_mean
        ],

        "VC×異性相手": [
            vc_mean
        ],

        "差": [
            diff_mean
        ],

        "絶対差": [
            abs_diff_mean
        ]
    })


    # ========================================================
    # 表の一番下に平均を追加
    # ========================================================

    display_table = pd.concat(
        [
            display_table,
            mean_row
        ],
        ignore_index=True
    )


    # ========================================================
    # ターミナルに表を表示
    # ========================================================

    print("\n\n")
    print("=" * 110)
    print(
        f"★ {group_name}："
        "地声×同性相手 vs VCあり×異性相手"
    )
    print("=" * 110)


    print(
        tabulate(
            display_table,
            headers="keys",
            tablefmt="grid",
            showindex=False,
            floatfmt=".1f"
        )
    )


    # ========================================================
    # 男女別の平均
    # ========================================================

    gender_summary = []


    for gender in [
        "男性",
        "女性"
    ]:

        gender_df = comparison[
            comparison["被験者の性別"]
            == gender
        ]


        if len(gender_df) == 0:
            continue


        gender_summary.append({

            "被験者の性別":
                gender,

            "N":
                len(gender_df),

            "地声×同性相手":
                gender_df[
                    "地声×同性相手"
                ].mean(),

            "VC×異性相手":
                gender_df[
                    "VC×異性相手"
                ].mean(),

            "平均差":
                gender_df[
                    "差"
                ].mean(),

            "平均絶対差":
                gender_df[
                    "絶対差"
                ].mean()
        })


    gender_summary = pd.DataFrame(
        gender_summary
    )


    print("\n")
    print(
        f"【{group_name}：男女別平均】"
    )


    print(
        tabulate(
            gender_summary,
            headers="keys",
            tablefmt="grid",
            showindex=False,
            floatfmt=".1f"
        )
    )


    return comparison


# ============================================================
# ⑫ 高主体感群
# ============================================================

high_comparison = make_comparison_table(
    high_names,
    "高主体感群"
)


# ============================================================
# ⑬ 低主体感群
# ============================================================

low_comparison = make_comparison_table(
    low_names,
    "低主体感群"
)


# ============================================================
# ⑭ 高主体感群 vs 低主体感群
#
# 最後に平均値だけまとめた比較表を出す
# ============================================================

summary_table = pd.DataFrame({

    "群": [
        "高主体感群",
        "低主体感群"
    ],

    "N": [
        len(high_comparison),
        len(low_comparison)
    ],

    "地声×同性相手": [
        high_comparison[
            "地声×同性相手"
        ].mean(),

        low_comparison[
            "地声×同性相手"
        ].mean()
    ],

    "VC×異性相手": [
        high_comparison[
            "VC×異性相手"
        ].mean(),

        low_comparison[
            "VC×異性相手"
        ].mean()
    ],

    "平均差": [
        high_comparison[
            "差"
        ].mean(),

        low_comparison[
            "差"
        ].mean()
    ],

    "平均絶対差": [
        high_comparison[
            "絶対差"
        ].mean(),

        low_comparison[
            "絶対差"
        ].mean()
    ]
})


print("\n\n")
print("=" * 110)
print("★ 高主体感群 vs 低主体感群")
print("=" * 110)


print(
    tabulate(
        summary_table,
        headers="keys",
        tablefmt="grid",
        showindex=False,
        floatfmt=".1f"
    )
)


# ============================================================
# ⑮ 平均絶対差の差
# ============================================================

high_abs_mean = high_comparison[
    "絶対差"
].mean()


low_abs_mean = low_comparison[
    "絶対差"
].mean()


difference_between_groups = (
    high_abs_mean
    -
    low_abs_mean
)


print("\n")
print("=" * 90)
print("平均絶対差")
print("=" * 90)


print(
    f"高主体感群 = "
    f"{high_abs_mean:.2f}"
)


print(
    f"低主体感群 = "
    f"{low_abs_mean:.2f}"
)


print(
    f"高 − 低 = "
    f"{difference_between_groups:+.2f}"
)