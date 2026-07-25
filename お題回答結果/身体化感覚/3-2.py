# ==========================================
# ③-2 因子間相関分析（CSVから計算・英語軸版）
# ==========================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === データ読み込み ===
df = pd.read_csv("2025身体化感覚アンケート（回答） - フォームの回答 1.csv")
df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip()

# === 質問文のマッピング ===
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

# === 数値変換 & 逆転処理 ===
items = df[list(cols.values())].apply(pd.to_numeric, errors="coerce")
for q in ["Q2", "Q6", "Q12"]:
    items[cols[q]] = 8 - items[cols[q]]

# === 因子スコア作成（英語表記） ===
df_fac = pd.DataFrame({
    "Body Ownership": items[[cols["Q1"], cols["Q3"], cols["Q4"], cols["Q8"]]].mean(axis=1),
    "Self Location": items[[cols["Q5"], cols["Q6"], cols["Q7"]]].mean(axis=1),
    "Agency": items[[cols["Q9"], cols["Q10"], cols["Q11"], cols["Q12"]]].mean(axis=1)
})

# === 相関係数の計算 ===
corr = df_fac.corr(method="pearson").round(2)
print("\n=== Correlation among Embodiment Factors (Pearson r) ===")
print(corr)

# === ヒートマップ ===
sns.set(style="white", font="Arial", font_scale=1.3)
plt.figure(figsize=(5,4))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, square=True,
            cbar_kws={"shrink": .8}, fmt=".2f", linewidths=0.5)
plt.title("Correlation among Embodiment Factors (Pearson r)", fontsize=14)
plt.tight_layout()
plt.savefig("Embodiment_Correlation_Heatmap.png", dpi=400, bbox_inches="tight")
plt.show()
