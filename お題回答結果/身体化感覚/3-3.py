# ==========================================
# ③-3 条件 × 因子 の交互作用（三要因分散分析）
# ==========================================
import pandas as pd
import pingouin as pg

# === データ読み込み ===
df = pd.read_csv("2025身体化感覚アンケート（回答） - フォームの回答 1.csv")
df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip()

# === 列名確認 ===
def find_col(keyword):
    hits = [c for c in df.columns if keyword in c]
    return hits[0] if hits else None

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

# === 数値化＆逆転項目 ===
items = df[list(cols.values())].apply(pd.to_numeric, errors="coerce")
for q in ["Q2", "Q6", "Q12"]:
    items[cols[q]] = 8 - items[cols[q]]

# === 因子スコア ===
df["Body_Ownership"] = items[[cols["Q1"], cols["Q3"], cols["Q4"], cols["Q8"]]].mean(axis=1)
df["Self_Location"]  = items[[cols["Q5"], cols["Q6"], cols["Q7"]]].mean(axis=1)
df["Agency"]         = items[[cols["Q9"], cols["Q10"], cols["Q11"], cols["Q12"]]].mean(axis=1)

# === 条件情報 ===
df["VC"] = df["直前の実験でボイスチェンジャーを使用しましたか"].map({"はい": "Yes", "いいえ": "No"})

# 性別リスト（前回の20名リストから）
gender_map = {
    "杉江鎌": "男性", "﨑谷瑠愛": "女性", "清水拓海": "男性", "伊藤駿之介": "男性",
    "小島夏美": "女性", "西岡謙介": "男性", "北村佳資": "男性", "日野快人": "男性",
    "本多快": "男性", "飛澤佑季": "男性", "谷川舞桜": "女性",
    "久保田翔帆": "女性", "小西真結": "女性", "利光唯": "女性", "横山秀侑": "男性",
    "池田滉成": "男性", "北野安樹子": "女性", "渡辺怜": "女性", "細見涼乃": "女性", "葭田桃花": "女性"
}
df["性別"] = df["氏名"].map(gender_map)

# === ロング形式に変換（因子を1列にまとめる）===
df_long = df.melt(id_vars=["氏名", "性別", "VC"],
                  value_vars=["Body_Ownership", "Self_Location", "Agency"],
                  var_name="Factor", value_name="Score")

# === 三要因分散分析 ===
anova = pg.anova(dv="Score", between=["性別", "VC", "Factor"], data=df_long, detailed=True)
print("\n=== 三要因分散分析（性別 × VC × 因子） ===")
print(anova.round(4))
