# ==========================================
# ③-1 各因子スコアを 性別×VC有無 ごとで比較（Two-way ANOVA）
# ==========================================
import pandas as pd
import numpy as np
import pingouin as pg

# ---------- ① 氏名→性別 マッピング ----------
name_to_gender = {
    # 男性
    "杉江鎌": "男性",
    "清水拓海": "男性",
    "伊藤駿之介": "男性",
    "西岡謙介": "男性",
    "北村佳資": "男性",
    "日野快人": "男性",
    "本多快": "男性",
    "飛澤佑季": "男性",
    "池田滉成": "男性",
    "横山秀侑": "男性",  # ← 旧「しゅう」

    # 女性
    "﨑谷瑠愛": "女性",
    "小島夏美": "女性",
    "谷川舞桜": "女性",
    "小西真結": "女性",
    "利光唯": "女性",
    "久保田翔帆": "女性",  # ← 修正済み
    "北野安樹子": "女性",
    "渡辺怜": "女性",
    "細見涼乃": "女性",
    "葭田桃花": "女性",
}

# ---------- ② データ読み込み ----------
path = "2025身体化感覚アンケート（回答） - フォームの回答 1.csv"
df = pd.read_csv(path)
df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip()

# ---------- ③ 列名のゆるやかマッチ ----------
def find_col(keyword):
    hits = [c for c in df.columns if keyword in c]
    if not hits:
        raise ValueError(f"列が見つからない: {keyword}")
    return hits[0]

cols = {
    "Q1": find_col("自分自身の手足のように感じた"),
    "Q2": find_col("ようではないと感じた"),
    "Q3": find_col("自分の身体」として受け入れていた"),
    "Q4": find_col("自分の身体」だと感じる"),
    "Q5": find_col("同じ場所にいるように感じた"),
    "Q6": find_col("別の場所にあるように感じた"),
    "Q7": find_col("一体化しているように感じた"),
    "Q8": find_col("目から見ている"),
    "Q9": find_col("意図と動きの一致"),
    "Q10": find_col("操作できた"),
    "Q11": find_col("リアルタイム同期感"),
    "Q12": find_col("コントロール喪失感"),
}

# ---------- ④ スコア計算 ----------
items = df[list(cols.values())].apply(pd.to_numeric, errors="coerce")
for q in ["Q2", "Q6", "Q12"]:
    items[cols[q]] = 8 - items[cols[q]]

df_fac = pd.DataFrame({
    "身体所有感": items[[cols["Q1"], cols["Q3"], cols["Q4"], cols["Q8"]]].mean(axis=1),
    "自己位置感": items[[cols["Q5"], cols["Q6"], cols["Q7"]]].mean(axis=1),
    "行動主体感": items[[cols["Q9"], cols["Q10"], cols["Q11"], cols["Q12"]]].mean(axis=1),
})

# ---------- ⑤ メタ情報 ----------
meta = pd.DataFrame({
    "氏名": df["氏名"],
    "性別": df["氏名"].map(name_to_gender),
    "VC": df[find_col("ボイスチェンジャー")].astype(str).str.strip().replace({
        "はい": "Yes", "いいえ": "No", "有": "Yes", "無": "No"
    })
})

# ---------- ⑥ データ統合 ----------
data = pd.concat([meta, df_fac], axis=1).dropna(subset=["性別", "VC"])

# ---------- ⑦ 各群の平均値 ----------
summary = (
    data.groupby(["性別", "VC"])[["身体所有感", "自己位置感", "行動主体感"]]
        .mean().round(2).reset_index()
)
print("\n=== 各群の平均値（性別×VC） ===")
print(summary.to_string(index=False))

# ---------- ⑧ 二要因分散分析（ANOVA） ----------
for fac in ["身体所有感", "自己位置感", "行動主体感"]:
    print(f"\n=== {fac} の二要因分散分析（性別×VC） ===")
    print(pg.anova(dv=fac, between=["性別", "VC"], data=data, detailed=True))
