# ==========================================
# 性別 × ボイスチェンジャー × 因子の交互作用グラフ
# （男性＝青系、女性＝赤系で区別）
# ==========================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === データ読み込み ===
df = pd.read_csv("2025身体化感覚アンケート（回答） - フォームの回答 1.csv")
df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip()

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

# === 数値変換＆逆転項目 ===
items = df[list(cols.values())].apply(pd.to_numeric, errors="coerce")
for q in ["Q2", "Q6", "Q12"]:
    items[cols[q]] = 8 - items[cols[q]]

# === 因子スコア ===
df["Ownership"] = items[[cols["Q1"], cols["Q3"], cols["Q4"], cols["Q8"]]].mean(axis=1)
df["Location"]  = items[[cols["Q5"], cols["Q6"], cols["Q7"]]].mean(axis=1)
df["Agency"]    = items[[cols["Q9"], cols["Q10"], cols["Q11"], cols["Q12"]]].mean(axis=1)

# === 条件付与 ===
df["VC"] = df["直前の実験でボイスチェンジャーを使用しましたか"].map({"はい": "Yes", "いいえ": "No"})
gender_map = {
    "杉江鎌": "Male", "﨑谷瑠愛": "Female", "清水拓海": "Male", "伊藤駿之介": "Male",
    "小島夏美": "Female", "西岡謙介": "Male", "北村佳資": "Male", "日野快人": "Male",
    "本多快": "Male", "飛澤佑季": "Male", "谷川舞桜": "Female",
    "久保田翔帆": "Female", "小西真結": "Female", "利光唯": "Female", "横山秀侑": "Male",
    "池田滉成": "Male", "北野安樹子": "Female", "渡辺怜": "Female", "細見涼乃": "Female", "葭田桃花": "Female"
}
df["Gender"] = df["氏名"].map(gender_map)

# === ロング形式 ===
df_long = df.melt(id_vars=["氏名", "Gender", "VC"],
                  value_vars=["Ownership", "Location", "Agency"],
                  var_name="Factor", value_name="Score")

# === グラフ設定 ===
sns.set(style="whitegrid", font="Arial", font_scale=1.3)
plt.figure(figsize=(8, 5))

# パレット設定（青系：男性, 赤系：女性）
palette = {
    ("Male", "No"): "#1f77b4",      # 青
    ("Male", "Yes"): "#66b2ff",     # 水色
    ("Female", "No"): "#d62728",    # 赤
    ("Female", "Yes"): "#ff9999"    # ピンク
}

# 描画
for (gender, vc), group_data in df_long.groupby(["Gender", "VC"]):
    plt.plot(
        ["Ownership", "Location", "Agency"],
        group_data.groupby("Factor")["Score"].mean(),
        marker="o", linewidth=2.5, markersize=8,
        label=f"{gender} / VC-{vc}",
        color=palette[(gender, vc)]
    )

plt.title("Interaction: Gender × VC × Embodiment Factors", fontsize=15, pad=15)
plt.xlabel("Embodiment Factor", fontsize=13)
plt.ylabel("Mean Score (1–7)", fontsize=13)
plt.ylim(4.5, 6.6)
plt.legend(title="", loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.tight_layout()
plt.savefig("Gender_VC_Factor_Interaction.png", dpi=500, bbox_inches="tight")
plt.show()
