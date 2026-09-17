# ============================================================
# 仮想的な同性条件の比較
# Paired TOST ＋ グラフ作成
#
# 【比較する2条件】
#
# 条件A：地声 × 同性相手
#   男性被験者 → 男性相手
#   女性被験者 → 女性相手
#
# 条件B：VC × 異性相手
#   男性被験者 → 女性相手
#   女性被験者 → 男性相手
#
# ★ 全被験者20名を使用
# ★ 主体感・所有感による分類は行わない
# ★ 同じ被験者の2条件なので対応あり
#
# SESOI：
# Yee & Bailenson (2007) の自己開示で報告された
# Cohen's d = 0.38 を参考に
# 標準化効果量 ±0.38 を等価性境界とする
# ============================================================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# ① データ読み込み
# ============================================================

df_dis = pd.read_csv(
    "お題回答結果/被験者の発話データ/情報単位スコア一覧.csv"
)


# ============================================================
# ② 列名整理
# ============================================================

df_dis = df_dis.rename(
    columns={"name": "氏名"}
)

df_dis.columns = df_dis.columns.str.strip()


# ============================================================
# ③ 名前クリーニング
# ============================================================

df_dis["氏名"] = (
    df_dis["氏名"]
    .astype(str)
    .str.strip()
    .str.replace("　", "", regex=False)
    .str.replace(" ", "", regex=False)
)


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
# ⑤ 実験者の性別を統一
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
# ⑦ 条件A
# 地声 × 同性相手
#
# 男性 → 男性
# 女性 → 女性
# ============================================================

natural_same = df_dis[

    (df_dis["VC"] == "VCなし")

    &

    (
        (
            (df_dis["participant_gender"] == "男性")
            &
            (df_dis["experimenter_gender"] == "男性")
        )

        |

        (
            (df_dis["participant_gender"] == "女性")
            &
            (df_dis["experimenter_gender"] == "女性")
        )
    )

][
    [
        "氏名",
        "participant_gender",
        "info_unit_score"
    ]
].copy()


natural_same = natural_same.rename(
    columns={
        "info_unit_score": "地声_同性"
    }
)


# ============================================================
# ⑧ 条件B
# VC × 異性相手
#
# 男性 → 女性
# 女性 → 男性
#
# VC使用時には自分の声の性別が変わるため、
# 「仮想的な同性条件」として扱う
# ============================================================

vc_opposite = df_dis[

    (df_dis["VC"] == "VCあり")

    &

    (
        (
            (df_dis["participant_gender"] == "男性")
            &
            (df_dis["experimenter_gender"] == "女性")
        )

        |

        (
            (df_dis["participant_gender"] == "女性")
            &
            (df_dis["experimenter_gender"] == "男性")
        )
    )

][
    [
        "氏名",
        "info_unit_score"
    ]
].copy()


vc_opposite = vc_opposite.rename(
    columns={
        "info_unit_score": "VC_異性"
    }
)


# ============================================================
# ⑨ 同じ被験者同士を結合
# ============================================================

paired = pd.merge(
    natural_same,
    vc_opposite,
    on="氏名",
    how="inner"
)


paired = paired.dropna(
    subset=[
        "地声_同性",
        "VC_異性"
    ]
)


# ============================================================
# ⑩ 個人ごとの差
#
# 差 = VC×異性 − 地声×同性
#
# 0に近いほど2条件が近い
# ============================================================

paired["差"] = (
    paired["VC_異性"]
    -
    paired["地声_同性"]
)

paired["絶対差"] = (
    paired["差"].abs()
)


# ============================================================
# ⑪ 使用データを表示
# ============================================================

print("\n========================================")
print("仮想的な同性条件")
print("地声×同性 vs VC×異性")
print("========================================")

print(
    paired[
        [
            "氏名",
            "participant_gender",
            "地声_同性",
            "VC_異性",
            "差",
            "絶対差"
        ]
    ].to_string(index=False)
)


print("\n========================================")
print("被験者数")
print("========================================")

print(
    f"N = {len(paired)}"
)


# ============================================================
# ⑫ 記述統計
# ============================================================

mean_natural = paired["地声_同性"].mean()

mean_vc = paired["VC_異性"].mean()

mean_diff = paired["差"].mean()

mean_abs_diff = paired["絶対差"].mean()


print("\n========================================")
print("記述統計")
print("========================================")

print(
    f"地声×同性 平均 = {mean_natural:.3f}"
)

print(
    f"VC×異性 平均   = {mean_vc:.3f}"
)

print(
    f"平均差          = {mean_diff:.3f}"
)

print(
    f"平均絶対差      = {mean_abs_diff:.3f}"
)


# ============================================================
# ⑬ 差得点の統計量
# ============================================================

diff = paired["差"].to_numpy()

n = len(diff)

sd_diff = np.std(
    diff,
    ddof=1
)

se_diff = (
    sd_diff
    /
    np.sqrt(n)
)

# ============================================================
# ⑭ SESOI
#
# Yee & Bailenson (2007) の自己開示で報告された
# Cohen's d = 0.38 を参考に、
#
# 標準化効果量の尺度で
#
# SESOI = [-0.38, +0.38]
#
# と設定する。
#
# ★ 0.38 × SD のような生得点への変換は行わない
# ============================================================

SESOI_d = 0.38

lower_bound = -SESOI_d
upper_bound = +SESOI_d


print("\n========================================")
print("SESOI")
print("========================================")

print(
    f"標準化効果量上の等価性境界 = "
    f"[{lower_bound:.2f}, {upper_bound:.2f}]"
)


# ============================================================
# ⑮ Cohen's dz
#
# 同じ被験者の2条件を比較しているため、
# 対応ありデータの標準化平均差 dz を使用
#
# dz = 平均差 / 差得点のSD
# ============================================================

cohens_dz = (
    mean_diff
    /
    sd_diff
)


print("\n========================================")
print("標準化平均差")
print("========================================")

print(
    f"Cohen's dz = {cohens_dz:.3f}"
)


# ============================================================
# ⑯ 対応ありt統計量
#
# paired t の t 値と dz には
#
# t = dz × √N
#
# の関係がある
# ============================================================

t_observed = (
    mean_diff
    /
    se_diff
)


# ============================================================
# ⑰ TOST
#
# SESOI = ±0.38 を
# dz の尺度でそのまま使用
#
# dz = t / √N
#
# なので、SESOI ±0.38 に対応する
# t分布上の非心度は
#
# λ = ±0.38 × √N
#
# ============================================================

ncp_lower = (
    lower_bound
    *
    np.sqrt(n)
)

ncp_upper = (
    upper_bound
    *
    np.sqrt(n)
)


# ------------------------------------------------------------
# 下側検定
#
# H0：dz <= -0.38
# H1：dz >  -0.38
# ------------------------------------------------------------

p_lower = stats.nct.sf(
    t_observed,
    df=n - 1,
    nc=ncp_lower
)


# ------------------------------------------------------------
# 上側検定
#
# H0：dz >= +0.38
# H1：dz <  +0.38
# ------------------------------------------------------------

p_upper = stats.nct.cdf(
    t_observed,
    df=n - 1,
    nc=ncp_upper
)


# ============================================================
# ⑱ dz の90%信頼区間
#
# 非心t分布を用いて、
# 非心度 λ の90%CIを求める。
#
# dz = λ / √N
#
# なので最後に√Nで割って
# dzのCIへ変換する。
# ============================================================

from scipy.optimize import brentq


alpha = 0.10


def find_ncp_for_cdf(target_probability):

    def function(ncp):

        return (
            stats.nct.cdf(
                t_observed,
                df=n - 1,
                nc=ncp
            )
            -
            target_probability
        )

    return brentq(
        function,
        -100,
        100
    )


# 非心度の下限・上限
ncp_ci_lower = find_ncp_for_cdf(
    1 - alpha / 2
)

ncp_ci_upper = find_ncp_for_cdf(
    alpha / 2
)


# dz尺度へ変換
dz_ci_lower = (
    ncp_ci_lower
    /
    np.sqrt(n)
)

dz_ci_upper = (
    ncp_ci_upper
    /
    np.sqrt(n)
)


# ============================================================
# ⑲ 結果表示
# ============================================================

print("\n========================================")
print("TOST 結果")
print("========================================")

print(
    f"N = {n}"
)

print(
    f"平均差（情報単位） = {mean_diff:.3f}"
)

print(
    f"差得点のSD = {sd_diff:.3f}"
)

print(
    f"Cohen's dz = {cohens_dz:.3f}"
)

print(
    f"dz の90% CI = "
    f"[{dz_ci_lower:.3f}, {dz_ci_upper:.3f}]"
)

print(
    f"SESOI = "
    f"[{lower_bound:.3f}, {upper_bound:.3f}]"
)

print()

print(
    f"下側検定 "
    f"(H0: dz <= {lower_bound:.2f}) "
    f"p = {p_lower:.4f}"
)

print(
    f"上側検定 "
    f"(H0: dz >= {upper_bound:.2f}) "
    f"p = {p_upper:.4f}"
)


# ============================================================
# ⑳ 最終判定
# ============================================================

equivalent = (
    (p_lower < 0.05)
    and
    (p_upper < 0.05)
)


print("\n========================================")
print("最終判定")
print("========================================")


if equivalent:

    print(
        "両方の片側検定が p < .05"
    )

    print(
        "→ SESOI ±0.38 の範囲内で"
        "統計的等価性が確認された"
    )

else:

    print(
        "少なくとも一方の片側検定が p >= .05"
    )

    print(
        "→ SESOI ±0.38 の範囲内での"
        "統計的等価性は確認されなかった"
    )


# ============================================================
# ㉑ グラフ1
#
# 各被験者の2条件を線で結ぶ
# ============================================================

plt.figure(
    figsize=(8, 6)
)


for _, row in paired.iterrows():

    plt.plot(
        [0, 1],
        [
            row["地声_同性"],
            row["VC_異性"]
        ],
        marker="o",
        alpha=0.45,
        linewidth=1
    )


# 平均値
plt.plot(
    [0, 1],
    [
        mean_natural,
        mean_vc
    ],
    marker="o",
    linewidth=4,
    markersize=10,
    label="Mean"
)


plt.xticks(
    [0, 1],
    [
        "Natural Voice\nSame-Gender Partner",
        "VC\nOpposite-Gender Partner"
    ]
)


plt.ylabel(
    "Self-Disclosure Score"
)


plt.title(
    "Virtual Same-Gender Condition"
)


plt.grid(
    axis="y",
    alpha=0.3
)


plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# ㉒ グラフ2
#
# 標準化効果量 dz によるTOSTの可視化
#
# 横軸：
# Cohen's dz
#
# 灰色：
# SESOI = [-0.38, +0.38]
#
# 点：
# 今回観測された dz
#
# 横線：
# dz の90%信頼区間
#
# 90%CI全体がSESOI内に入れば
# 統計的等価性が示される
# ============================================================

plt.figure(
    figsize=(9, 4)
)


# SESOI
plt.axvspan(
    lower_bound,
    upper_bound,
    alpha=0.20,
    label="SESOI (-0.38 to +0.38)"
)


# 効果量0
plt.axvline(
    0,
    linestyle="--",
    linewidth=1.5
)


# SESOI境界
plt.axvline(
    lower_bound,
    linestyle=":",
    linewidth=2
)

plt.axvline(
    upper_bound,
    linestyle=":",
    linewidth=2
)


# dz + 90% CI
plt.errorbar(
    cohens_dz,
    0,
    xerr=[
        [
            cohens_dz - dz_ci_lower
        ],
        [
            dz_ci_upper - cohens_dz
        ]
    ],
    fmt="o",
    markersize=10,
    capsize=8,
    linewidth=2
)


# SESOIの値
plt.text(
    lower_bound,
    0.25,
    "-0.38",
    ha="center"
)

plt.text(
    upper_bound,
    0.25,
    "+0.38",
    ha="center"
)


# dz
plt.text(
    cohens_dz,
    -0.25,
    f"dz = {cohens_dz:.3f}",
    ha="center"
)


plt.yticks([])


plt.xlabel(
    "Standardized Mean Difference (Cohen's dz)\n"
    "(VC Opposite-Gender − Natural Voice Same-Gender)"
)


plt.title(
    "TOST Equivalence Test (SESOI = ±0.38)"
)


plt.xlim(
    min(
        lower_bound,
        dz_ci_lower
    ) - 0.15,
    max(
        upper_bound,
        dz_ci_upper
    ) + 0.15
)


plt.legend()

plt.tight_layout()

plt.show()