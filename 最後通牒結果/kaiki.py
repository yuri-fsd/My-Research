import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# ① データ読み込み
# =========================
df_ult = pd.read_csv("R1_round.csv")
df_body = pd.read_csv("2025身体化感覚アンケート.csv")

# =========================
# ② 列名整理
# =========================
df_body.columns = df_body.columns.str.strip()

# =========================
# ③ 名前クリーニング
# =========================
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

# =========================
# ④ 身体化スコア作成
# =========================
body_cols = [
    "自身のアバターの手足は, 自分自身の手足のように感じた",
    "自身のアバターを「自分の身体」として受け入れていた",
    "自分の身体はアバターと同じ場所にいるように感じた",
    "自分の意識がアバターと一体化しているように感じた",
    "自分はアバターの目から見ているように感じた"
]

df_body["body_score"] = df_body[body_cols].mean(axis=1)

# 被験者ごと平均
df_body_mean = df_body.groupby("氏名")["body_score"].mean().reset_index()

# =========================
# ⑤ 最後通牒（あり・なし分ける）
# =========================
print("voice changer:", df_ult["ボイスチェンジャー"].unique())

df_ult_wide = df_ult.pivot_table(
    index=["氏名", "実験者の性別"],
    columns="ボイスチェンジャー",
    values="第１ラウンド",
    aggfunc="mean"
).reset_index()

# 列名統一
df_ult_wide = df_ult_wide.rename(columns={
    "なし": "absent",
    "あり": "present"
})

# 差分（あり − なし）
df_ult_wide["diff"] = df_ult_wide["present"] - df_ult_wide["absent"]

# =========================
# ⑥ 結合
# =========================
df_plot = pd.merge(
    df_ult_wide,
    df_body_mean,
    on="氏名",
    how="inner"
)

print(df_plot.head())
print("N =", len(df_plot))

# =========================
# ⑦ グラフ
# =========================
plt.figure(figsize=(8,6))

# 男性実験者（青）
sns.regplot(
    data=df_plot[df_plot["実験者の性別"]=="男性"],
    x="body_score", y="diff",
    color="blue",
    label="Male Experimenter",
    scatter_kws={"s":80}
)

# 女性実験者（赤）
sns.regplot(
    data=df_plot[df_plot["実験者の性別"]=="女性"],
    x="body_score", y="diff",
    color="red",
    label="Female Experimenter",
    scatter_kws={"s":80}
)

# 横線
plt.axhline(0, linestyle="--", color="gray")

# 英語ラベル
plt.xlabel("Agency Score")
plt.ylabel("Offer Difference (Present - Absent)")
plt.title("Effect of Agency on Ultimatum Offers by Experimenter Gender")

plt.legend()
plt.grid(True)

plt.show()