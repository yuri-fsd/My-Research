import pandas as pd
from scipy.stats import pearsonr, spearmanr

# ==================================================
# 読み込み
# ==================================================

body = pd.read_csv(
    "身体化感覚/2025身体化感覚アンケート（回答） - フォームの回答 1.csv"
)

post = pd.read_csv(
    "被験者の発話データ/2025実験後アンケート（回答） - フォームの回答 1.csv"
)

# ==================================================
# 名前整形
# ==================================================

for df_ in [body, post]:

    df_["氏名"] = (
        df_["氏名"]
        .astype(str)
        .str.strip()
        .str.replace("　", "", regex=False)
        .str.replace(" ", "", regex=False)
    )

# 渡辺→渡邉 統一
body["氏名"] = body["氏名"].replace({
    "渡辺怜": "渡邉怜"
})

# ==================================================
# 主体感（Agency）
# Q9,Q10,Q11,Q12(逆転)
# ==================================================

body["agency"] = (
    body.iloc[:, 11] +
    body.iloc[:, 12] +
    body.iloc[:, 13] +
    (8 - body.iloc[:, 14])
) / 4

# ==================================================
# 被験者ごと平均
# ==================================================

agency_df = (
    body.groupby("氏名")["agency"]
    .mean()
    .reset_index()
)

# ==================================================
# マージ
# ==================================================

df = pd.merge(
    agency_df,
    post,
    on="氏名"
)

print("人数 =", len(df))

# ==================================================
# 文字列 → 数値
# ==================================================

adapt_map = {
    "適応できなかった": 1,
    "少し時間がかかった": 2,
    "すぐに対応できた": 3
}

expression_map = {
    "あまり影響を受けなかった": 1,
    "多少影響を受けた": 2,
    "強く影響を受けた": 3
}

reuse_map = {
    "あまり使用したくない": 1,
    "使用したい": 2,
    "ぜひまた使用してみたい": 3
}

df["異なる声のコミュニケーションに対応できましたか？"] = (
    df["異なる声のコミュニケーションに対応できましたか？"]
    .map(adapt_map)
)

df["ボイスチェンジャーが自己表現に影響を与えたと感じましたか？"] = (
    df["ボイスチェンジャーが自己表現に影響を与えたと感じましたか？"]
    .map(expression_map)
)

df["またボイスチェンジャーを使って話したいと思いますか？"] = (
    df["またボイスチェンジャーを使って話したいと思いますか？"]
    .map(reuse_map)
)

# ==================================================
# 性別
# ==================================================

female_names = [
    "小島夏美",
    "久保田翔帆",
    "小西真結",
    "北野安樹子",
    "渡邉怜",
    "細見涼乃",
    "葭田桃花",
    "谷川舞桜",
    "利光唯",
    "﨑谷瑠愛"
]

df["gender"] = df["氏名"].apply(
    lambda x: "女性" if x in female_names else "男性"
)

# ==================================================
# 分析項目
# ==================================================

target_cols = [

    "異なる声のコミュニケーションに対応できましたか？",

    "ボイスチェンジャーが自己表現に影響を与えたと感じましたか？",

    "またボイスチェンジャーを使って話したいと思いますか？",

    "ボイスチェンジャーで変換した声は地声よりも気に入った",

    "ボイスチェンジャーを使用した方が自己開示しやすかった  ",

    "ボイスチェンジャーを使用した方が交渉で強気だった  ",

    "実験を進めていくにつれて，ボイスチェンジャーで変換した声が自分の声のように感じた",

    "ボイスチェンジャーを使用した場合では，普段の自分と違うキャラになったと感じた"

]

# ==================================================
# 相関分析
# ==================================================

def run_corr(data, title):

    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)

    for col in target_cols:

        tmp = data[["agency", col]].copy()

        tmp["agency"] = pd.to_numeric(
            tmp["agency"],
            errors="coerce"
        )

        tmp[col] = pd.to_numeric(
            tmp[col],
            errors="coerce"
        )

        tmp = tmp.dropna()

        if len(tmp) < 3:
            continue

        try:

            r1, p1 = pearsonr(
                tmp["agency"],
                tmp[col]
            )

            r2, p2 = spearmanr(
                tmp["agency"],
                tmp[col]
            )

            print("\n")
            print("【", col, "】")
            print(f"Pearson  r = {r1:.3f}   p = {p1:.3f}")
            print(f"Spearman r = {r2:.3f}   p = {p2:.3f}")

        except Exception as e:

            print("\n")
            print(col)
            print("ERROR:", e)

# ==================================================
# 全体
# ==================================================

run_corr(
    df,
    "全体"
)

# ==================================================
# 男性
# ==================================================

run_corr(
    df[df["gender"] == "男性"],
    "男性"
)

# ==================================================
# 女性
# ==================================================

run_corr(
    df[df["gender"] == "女性"],
    "女性"
)

# ==================================================
# 主体感ランキング
# ==================================================

print("\n")
print("=" * 80)
print("主体感ランキング")
print("=" * 80)

rank_df = df.sort_values(
    "agency",
    ascending=False
)

cols = [
    "氏名",
    "gender",
    "agency",
    "ボイスチェンジャーを使用した方が自己開示しやすかった  ",
    "実験を進めていくにつれて，ボイスチェンジャーで変換した声が自分の声のように感じた",
    "ボイスチェンジャーを使用した場合では，普段の自分と違うキャラになったと感じた"
]

print(rank_df[cols])

# ==================================================
# Spearman相関ランキング
# ==================================================

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

plt.rcParams["font.family"] = "Hiragino Sans"

results = []

for col in target_cols:

    tmp = df[["agency", col]].copy()

    tmp["agency"] = pd.to_numeric(
        tmp["agency"],
        errors="coerce"
    )

    tmp[col] = pd.to_numeric(
        tmp[col],
        errors="coerce"
    )

    tmp = tmp.dropna()

    r, p = spearmanr(
        tmp["agency"],
        tmp[col]
    )

    results.append([
        col,
        r,
        p,
        abs(r)
    ])

corr_df = pd.DataFrame(
    results,
    columns=[
        "項目",
        "Spearman_r",
        "p",
        "abs_r"
    ]
)

corr_df = corr_df.sort_values(
    "abs_r",
    ascending=False
)

print("\n===== 主体感との相関ランキング =====")
print(corr_df)

# CSV保存
corr_df.to_csv(
    "agency_correlation_ranking.csv",
    index=False,
    encoding="utf-8-sig"
)

# グラフ
plt.figure(figsize=(10, 6))

sns.barplot(
    data=corr_df,
    y="項目",
    x="Spearman_r"
)

plt.axvline(
    0,
    color="black",
    linestyle="--"
)

plt.xlabel("Spearman correlation")
plt.ylabel("")
plt.title("Agency と実験後アンケートの相関")

plt.tight_layout()

plt.savefig(
    "agency_correlation_ranking.png",
    dpi=300
)

plt.show()

# ==================================================
# 相関グラフ作成関数
# ==================================================

def make_corr_graph(data, title, save_name):

    results = []

    for col in target_cols:

        tmp = data[["agency", col]].copy()

        tmp["agency"] = pd.to_numeric(
            tmp["agency"],
            errors="coerce"
        )

        tmp[col] = pd.to_numeric(
            tmp[col],
            errors="coerce"
        )

        tmp = tmp.dropna()

        if len(tmp) < 3:
            continue

        r, p = spearmanr(
            tmp["agency"],
            tmp[col]
        )

        results.append([
            col,
            r,
            p,
            abs(r)
        ])

    corr_df = pd.DataFrame(
        results,
        columns=[
            "項目",
            "Spearman_r",
            "p",
            "abs_r"
        ]
    )

    corr_df = corr_df.sort_values(
        "abs_r",
        ascending=False
    )

    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(corr_df)

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=corr_df,
        y="項目",
        x="Spearman_r"
    )

    plt.axvline(
        0,
        color="black",
        linestyle="--"
    )

    plt.xlabel("Spearman correlation")
    plt.ylabel("")
    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        save_name,
        dpi=300
    )

    plt.show()


# ==================================================
# 男性
# ==================================================

make_corr_graph(
    df[df["gender"] == "男性"],
    "Agency と実験後アンケートの相関（男性）",
    "Agency_Correlation_Male.png"
)

# ==================================================
# 女性
# ==================================================

make_corr_graph(
    df[df["gender"] == "女性"],
    "Agency と実験後アンケートの相関（女性）",
    "Agency_Correlation_Female.png"
)