# # 同性相手異性相手全体
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import statsmodels.api as sm

# # =========================
# # ① データ読み込み
# # =========================
# df_dis = pd.read_csv("被験者の発話データ/情報単位スコア一覧.csv")
# df_body = pd.read_csv("被験者の発話データ/2025身体化感覚アンケート.csv")

# # =========================
# # ② 列名整理
# # =========================
# df_dis = df_dis.rename(columns={"name": "氏名"})
# df_body.columns = df_body.columns.str.strip()

# # =========================
# # ③ 名前クリーニング
# # =========================
# def clean_name(df):
#     df["氏名"] = (
#         df["氏名"]
#         .astype(str)
#         .str.strip()
#         .str.replace("　", "", regex=False)
#         .str.replace(" ", "", regex=False)
#     )
#     return df

# df_dis = clean_name(df_dis)
# df_body = clean_name(df_body)

# # =========================
# # ④ 性別統一
# # =========================
# df_dis["participant_gender"] = df_dis["participant_gender"].replace({"男":"男性","女":"女性"})

# # =========================
# # ⑤ 同性 / 異性
# # =========================
# df_dis["relation"] = np.where(
#     df_dis["participant_gender"] == df_dis["experimenter_gender"],
#     "Same", "Different"
# )

# # =========================
# # ⑥ 身体化スコア（3分割🔥）
# # =========================
# numeric_cols = df_body.columns[-12:]

# df_body["ownership"] = df_body[numeric_cols[:4]].mean(axis=1)
# df_body["agency"] = df_body[numeric_cols[4:8]].mean(axis=1)
# df_body["self_rep"] = df_body[numeric_cols[8:12]].mean(axis=1)

# df_body_mean = df_body.groupby("氏名")[["ownership","agency","self_rep"]].mean().reset_index()

# # =========================
# # ⑦ 自己開示 diff（被験者内）
# # =========================
# df_dis_wide = df_dis.pivot_table(
#     index=["氏名","relation"],
#     columns="voice_changer",
#     values="info_unit_score",
#     aggfunc="mean"
# ).reset_index()

# df_dis_wide = df_dis_wide.rename(columns={
#     "なし": "absent",
#     "あり": "present"
# })

# df_dis_wide["diff"] = df_dis_wide["present"] - df_dis_wide["absent"]

# # =========================
# # ⑧ 結合
# # =========================
# df_plot = pd.merge(df_dis_wide, df_body_mean, on="氏名")

# print("N =", len(df_plot))

# # =========================
# # ⑨ 回帰関数🔥
# # =========================
# def regression_stats(df, x_col, label):

#     X = df[x_col]
#     y = df["diff"]

#     X = sm.add_constant(X)
#     model = sm.OLS(y, X).fit()

#     print(f"\n--- {label} ---")
#     print(f"β (slope): {model.params[x_col]:.3f}")
#     print(f"p-value : {model.pvalues[x_col]:.3f}")
#     print(f"R²      : {model.rsquared:.3f}")

# # =========================
# # ⑩ グラフ関数🔥
# # =========================
# def plot_reg(x_col, title):

#     plt.figure(figsize=(8,6))

#     same_df = df_plot[df_plot["relation"]=="Same"]
#     diff_df = df_plot[df_plot["relation"]=="Different"]

#     # 回帰
#     sns.regplot(data=same_df, x=x_col, y="diff",
#                 color="blue", label="Same Gender", scatter_kws={"s":80})

#     sns.regplot(data=diff_df, x=x_col, y="diff",
#                 color="red", label="Different Gender", scatter_kws={"s":80})

#     # ===== 数値出力 =====
#     regression_stats(same_df, x_col, f"{title} (Same)")
#     regression_stats(diff_df, x_col, f"{title} (Different)")

#     # ===== 見た目 =====
#     plt.axhline(0, linestyle="--", color="gray")

#     plt.xlabel(x_col.capitalize())
#     plt.ylabel("Self-Disclosure Difference (VC On − Off)")
#     plt.title(title)

#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()

#     plt.show()

# # =========================
# # ⑪ 実行🔥
# # =========================
# plot_reg("ownership", "Ownership vs Self-Disclosure")
# plot_reg("agency", "Agency vs Self-Disclosure")
# plot_reg("self_rep", "Self-Representation vs Self-Disclosure")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# =========================
# フォント（文字化け防止）
# =========================
plt.rcParams["font.family"] = "Hiragino Sans"

# =========================
# ① データ読み込み
# =========================
df_dis = pd.read_csv("お題回答結果/被験者の発話データ/情報単位スコア一覧.csv")
df_body = pd.read_csv("お題回答結果/被験者の発話データ/2025身体化感覚アンケート.csv")

# =========================
# ② 前処理
# =========================
df_dis = df_dis.rename(columns={"name": "氏名"})
df_body.columns = df_body.columns.str.strip()

def clean_name(df):
    df["氏名"] = (
        df["氏名"]
        .astype(str)
        .str.strip()
        .str.replace("　","")
        .str.replace(" ","")
    )
    return df

df_dis = clean_name(df_dis)
df_body = clean_name(df_body)

# 性別統一
df_dis["participant_gender"] = df_dis["participant_gender"].replace({"男":"男性","女":"女性"})

# 同性 / 異性
df_dis["relation"] = np.where(
    df_dis["participant_gender"] == df_dis["experimenter_gender"],
    "Same", "Different"
)

# =========================
# ③ 身体化スコア（3分割）
# =========================
numeric_cols = df_body.columns[-12:]

df_body["ownership"] = df_body[numeric_cols[:4]].mean(axis=1)
df_body["agency"] = df_body[numeric_cols[4:8]].mean(axis=1)
df_body["self_rep"] = df_body[numeric_cols[8:12]].mean(axis=1)

df_body_mean = df_body.groupby("氏名")[["ownership","agency","self_rep"]].mean().reset_index()

# =========================
# ④ VC差分（被験者内）
# =========================
df_dis_wide = df_dis.pivot_table(
    index=["氏名","relation","participant_gender"],
    columns="voice_changer",
    values="info_unit_score",
    aggfunc="mean"
).reset_index()

df_dis_wide = df_dis_wide.rename(columns={
    "なし": "absent",
    "あり": "present"
})

df_dis_wide["diff"] = df_dis_wide["present"] - df_dis_wide["absent"]

# =========================
# ⑤ 結合
# =========================
df_plot = pd.merge(df_dis_wide, df_body_mean, on="氏名")

print("Total N =", len(df_plot))

# =========================
# ⑥ 英語ラベル
# =========================
label_map = {
    "ownership": "Sense of Ownership",
    "agency": "Sense of Agency",
    "self_rep": "Self-Representation"
}

# =========================
# ⑦ 回帰
# =========================
def regression_stats(df, x_col, label):

    print(f"\n--- {label} ---")
    print("N =", len(df))

    if len(df) < 2:
        print("Not enough data")
        return

    X = sm.add_constant(df[x_col])
    y = df["diff"]

    model = sm.OLS(y, X).fit()

    print(f"Mean diff = {df['diff'].mean():.3f}")
    print(f"β = {model.params[x_col]:.3f}")
    print(f"p = {model.pvalues[x_col]:.3f}")
    print(f"R² = {model.rsquared:.3f}")

# =========================
# ⑧ 分析関数
# =========================
def analyze_gender(participant):

    sub = df_plot[df_plot["participant_gender"] == participant]

    p_label = "Male" if participant == "男性" else "Female"

    print(f"\n======================")
    print(f"{p_label} Participants")
    print(f"======================")

    for var in ["ownership","agency","self_rep"]:

        same_df = sub[sub["relation"]=="Same"]
        diff_df = sub[sub["relation"]=="Different"]

        # ===== 数値 =====
        regression_stats(same_df, var, f"{p_label} - Same - {var}")
        regression_stats(diff_df, var, f"{p_label} - Different - {var}")

        # ===== グラフ =====
        plt.figure(figsize=(6,5))

        sns.regplot(
            data=same_df, x=var, y="diff",
            color="blue",
            label="Same-Gender Interaction",
            scatter_kws={"s":80}
        )

        sns.regplot(
            data=diff_df, x=var, y="diff",
            color="red",
            label="Different-Gender Interaction",
            scatter_kws={"s":80}
        )

        # 0ライン
        plt.axhline(0, linestyle="--", color="gray")

        # ラベル
        plt.xlabel(label_map[var])
        plt.ylabel("Change in Self-Disclosure (VC On − Off)")

        # タイトル
        plt.title(f"{p_label}: {label_map[var]}")

        # 見た目
        plt.legend()
        plt.grid(True)

        # ⭐ 見切れ防止（最重要🔥）
        plt.ylim(-11, 11)
        plt.yticks(range(-10, 11, 2))

        plt.tight_layout()
        plt.show()

# =========================
# ⑨ 実行
# =========================
analyze_gender("男性")
analyze_gender("女性")
