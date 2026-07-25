import pandas as pd
from scipy.stats import chi2_contingency

# =========================
# 1. データ読み込み
# =========================
df = pd.read_csv("2025距離測定結果.csv")

# =========================
# 2. 条件列の整理
# =========================
conditions = {
    "なし_男性": [c for c in df.columns if c.startswith("なしー男性")],
    "なし_女性": [c for c in df.columns if c.startswith("なしー女性")],
    "あり_男性": [c for c in df.columns if c.startswith("ありー男性")],
    "あり_女性": [c for c in df.columns if c.startswith("ありー女性")]
}

# =========================
# 3. 被験者 × VC で最終距離・回数を算出
# =========================
records = []

for _, row in df.iterrows():
    for key, cols in conditions.items():
        values = row[cols].dropna().astype(float)
        if len(values) == 0:
            continue

        records.append({
            "氏名": row["氏名"],
            "被験者性別": row["被験者の性別"],
            "実験者性別": "男性" if "男性" in key else "女性",
            "VC": "あり" if "あり" in key else "なし",
            "最終距離": values.max(),
            "回数": len(values)
        })

long_df = pd.DataFrame(records)

# =========================
# 4. 被験者ごとに平均（基準）を作る
# =========================
mean_df = (
    long_df
    .groupby("氏名")[["最終距離", "回数"]]
    .mean()
    .rename(columns={
        "最終距離": "距離_平均",
        "回数": "回数_平均"
    })
)

long_df = long_df.merge(mean_df, on="氏名")

# =========================
# 5. 平均との差で①〜④に分類
# =========================
def classify(row):
    if row["最終距離"] > row["距離_平均"]:
        dist = "近い"
    elif row["最終距離"] < row["距離_平均"]:
        dist = "遠い"
    else:
        return None

    if row["回数"] > row["回数_平均"]:
        cnt = "多い"
    elif row["回数"] < row["回数_平均"]:
        cnt = "少ない"
    else:
        return None

    if dist == "近い" and cnt == "多い":
        return "① 近い×多い"
    elif dist == "近い" and cnt == "少ない":
        return "② 近い×少ない"
    elif dist == "遠い" and cnt == "多い":
        return "③ 遠い×多い"
    else:
        return "④ 遠い×少ない"

long_df["パターン"] = long_df.apply(classify, axis=1)
long_df = long_df.dropna(subset=["パターン"])

# =========================
# 6. VC × パターンの分割表＋カイ二乗
# =========================
def chi_square_test(df_sub, title):
    table = pd.crosstab(df_sub["VC"], df_sub["パターン"])

    print("\n==============================")
    print(title)
    print("==============================")
    print(table)

    if table.shape[0] >= 2 and table.shape[1] >= 2:
        chi2, p, dof, _ = chi2_contingency(table)
        print(f"\nχ² = {chi2:.3f}, dof = {dof}, p = {p:.4f}")
    else:
        print("\n※ 検定不可（カテゴリ不足）")

# =========================
# 7. 指定された6条件
# =========================

chi_square_test(
    long_df[long_df["被験者性別"] == "男性"],
    "男性被験者：VCあり vs なし"
)

chi_square_test(
    long_df[long_df["被験者性別"] == "女性"],
    "女性被験者：VCあり vs なし"
)

chi_square_test(
    long_df[
        (long_df["被験者性別"] == "女性") &
        (long_df["実験者性別"] == "女性")
    ],
    "女性被験者 × 女性実験者"
)

chi_square_test(
    long_df[
        (long_df["被験者性別"] == "女性") &
        (long_df["実験者性別"] == "男性")
    ],
    "女性被験者 × 男性実験者"
)

chi_square_test(
    long_df[
        (long_df["被験者性別"] == "男性") &
        (long_df["実験者性別"] == "女性")
    ],
    "男性被験者 × 女性実験者"
)

chi_square_test(
    long_df[
        (long_df["被験者性別"] == "男性") &
        (long_df["実験者性別"] == "男性")
    ],
    "男性被験者 × 男性実験者"
)
