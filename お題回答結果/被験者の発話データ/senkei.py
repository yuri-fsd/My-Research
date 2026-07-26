import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# ① データ読み込み
# =========================
df_dis = pd.read_csv("お題回答結果/被験者の発話データ/情報単位スコア一覧.csv")
df_body = pd.read_csv("お題回答結果/被験者の発話データ/2025身体化感覚アンケート.csv")

# =========================
# ② 列名整理
# =========================
df_dis = df_dis.rename(columns={"name": "氏名"})
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

df_dis = clean_name(df_dis)
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
# ⑤ 自己開示（あり・なし分ける）
# =========================
print("voice_changer:", df_dis["voice_changer"].unique())

df_dis_wide = df_dis.pivot_table(
    index=["氏名", "experimenter_gender"],
    columns="voice_changer",
    values="info_unit_score",
    aggfunc="mean"
).reset_index()

# 値に応じて対応（どれでも動くように）
df_dis_wide = df_dis_wide.rename(columns={
    0: "absent",
    1: "present",
    "なし": "absent",
    "あり": "present",
    False: "absent",
    True: "present"
})

# 差分（あり − なし）
df_dis_wide["diff"] = df_dis_wide["present"] - df_dis_wide["absent"]

# =========================
# ⑥ 結合
# =========================
df_plot = pd.merge(
    df_dis_wide,
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
    data=df_plot[df_plot["experimenter_gender"]=="男性"],
    x="body_score", y="diff",
    color="blue",
    label="Male Experimenter",
    scatter_kws={"s":80}
)

# 女性実験者（赤）
sns.regplot(
    data=df_plot[df_plot["experimenter_gender"]=="女性"],
    x="body_score", y="diff",
    color="red",
    label="Female Experimenter",
    scatter_kws={"s":80}
)

# 横線（差分0）
plt.axhline(0, linestyle="--", color="gray")

# 英語ラベル
plt.xlabel("Agency Score")
plt.ylabel("Self-Disclosure Difference (Present - Absent)")
plt.title("Effect of Agency on Self-Disclosure Difference by Experimenter Gender")

plt.legend()
plt.grid(True)

plt.show()


