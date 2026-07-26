import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# =========================
# ① データ読み込み
# =========================
df_ult = pd.read_csv("最後通牒結果/R1_round.csv")
df_body = pd.read_csv("最後通牒結果/2025身体化感覚アンケート.csv")

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
# ④ 同性 / 異性
# =========================
df_ult["relation"] = np.where(
    df_ult["被験者の性別"] == df_ult["実験者の性別"],
    "Same",
    "Different"
)

# interaction用
df_ult["relation_num"] = np.where(
    df_ult["relation"] == "Same",
    0,
    1
)

# =========================
# ⑤ 身体化スコア
# =========================
numeric_cols = df_body.columns[-12:]

df_body["ownership"] = df_body[numeric_cols[:4]].mean(axis=1)
df_body["agency"] = df_body[numeric_cols[4:8]].mean(axis=1)
df_body["self_rep"] = df_body[numeric_cols[8:12]].mean(axis=1)

df_body_mean = df_body.groupby("氏名")[
    ["ownership","agency","self_rep"]
].mean().reset_index()

# =========================
# ⑥ Ultimatum差分
# =========================
df_ult_wide = df_ult.pivot_table(
    index=["氏名","relation","relation_num"],
    columns="ボイスチェンジャー",
    values="第１ラウンド",
    aggfunc="mean"
).reset_index()

df_ult_wide = df_ult_wide.rename(columns={
    "なし": "absent",
    "あり": "present"
})

df_ult_wide["diff"] = (
    df_ult_wide["present"]
    - df_ult_wide["absent"]
)

# =========================
# ⑦ 結合
# =========================
df_plot = pd.merge(
    df_ult_wide,
    df_body_mean,
    on="氏名"
)

print("Total N =", len(df_plot))

# =========================
# ⑧ 単純傾き
# =========================
def regression_stats(df, x_col, label):

    X = sm.add_constant(df[x_col])
    y = df["diff"]

    model = sm.OLS(y, X).fit()

    print("\n")
    print("=" * 70)
    print(label)
    print("=" * 70)

    print(f"N = {len(df)}")
    print(f"Mean diff = {df['diff'].mean():.3f}")
    print(f"β = {model.params[x_col]:.3f}")
    print(f"p = {model.pvalues[x_col]:.3f}")
    print(f"R² = {model.rsquared:.3f}")

# =========================
# ⑨ interaction
# =========================
def interaction_analysis(x_col):

    print("\n")
    print("=" * 70)
    print(f"Regression: diff ~ {x_col} * relation")
    print("=" * 70)

    formula = f"diff ~ {x_col} * relation_num"

    model = smf.ols(
        formula=formula,
        data=df_plot
    ).fit()

    print(model.summary())

# =========================
# ⑩ グラフ
# =========================
def plot_reg(x_col, title):

    plt.figure(figsize=(8,6))

    same_df = df_plot[
        df_plot["relation"]=="Same"
    ]

    diff_df = df_plot[
        df_plot["relation"]=="Different"
    ]

    # ===== 回帰線 =====
    sns.regplot(
        data=same_df,
        x=x_col,
        y="diff",
        color="blue",
        label="Same Gender",
        scatter_kws={"s":80}
    )

    sns.regplot(
        data=diff_df,
        x=x_col,
        y="diff",
        color="red",
        label="Different Gender",
        scatter_kws={"s":80}
    )

    # ===== 単純傾き =====
    regression_stats(
        same_df,
        x_col,
        f"{title} (Same)"
    )

    regression_stats(
        diff_df,
        x_col,
        f"{title} (Different)"
    )

    # ===== interaction =====
    interaction_analysis(x_col)

    # ===== 見た目 =====
    plt.axhline(
        0,
        linestyle="--",
        color="gray"
    )

    plt.xlabel(x_col)
    plt.ylabel(
        "Offer Difference (VC On − Off)"
    )

    plt.title(title)

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.show()

# =========================
# ⑪ 実行
# =========================
plot_reg(
    "ownership",
    "Ownership vs Ultimatum Offer"
)

plot_reg(
    "agency",
    "Agency vs Ultimatum Offer"
)

plot_reg(
    "self_rep",
    "Self-Representation vs Ultimatum Offer"
)

# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
# import statsmodels.formula.api as smf

# # =========================
# # フォント
# # =========================
# plt.rcParams["font.family"] = "Hiragino Sans"

# # =========================
# # ① データ読み込み
# # =========================
# df_dis = pd.read_csv("R1_round.csv")
# df_body = pd.read_csv("2025身体化感覚アンケート.csv")

# # =========================
# # ② 前処理
# # =========================
# df_body.columns = df_body.columns.str.strip()

# def clean_name(df):

#     df["氏名"] = (
#         df["氏名"]
#         .astype(str)
#         .str.strip()
#         .str.replace("　", "")
#         .str.replace(" ", "")
#     )

#     return df

# df_dis = clean_name(df_dis)
# df_body = clean_name(df_body)

# # =========================
# # ③ 性別統一
# # =========================
# df_dis["被験者の性別"] = (
#     df_dis["被験者の性別"]
#     .replace({
#         "男": "男性",
#         "女": "女性"
#     })
# )

# # =========================
# # ④ 同性 / 異性
# # =========================
# df_dis["relation"] = np.where(
#     df_dis["被験者の性別"]
#     == df_dis["実験者の性別"],
#     "Same",
#     "Different"
# )

# # interaction用
# df_dis["relation_num"] = np.where(
#     df_dis["relation"] == "Same",
#     0,
#     1
# )

# # =========================
# # ⑤ 身体化スコア
# # =========================
# numeric_cols = df_body.columns[-12:]

# # 所有感
# df_body["ownership"] = (
#     df_body[numeric_cols[:4]]
#     .mean(axis=1)
# )

# # 主体感
# df_body["agency"] = (
#     df_body[numeric_cols[4:8]]
#     .mean(axis=1)
# )

# # 自己表象
# df_body["self_rep"] = (
#     df_body[numeric_cols[8:12]]
#     .mean(axis=1)
# )

# # 被験者平均
# df_body_mean = (
#     df_body.groupby("氏名")[
#         ["ownership", "agency", "self_rep"]
#     ]
#     .mean()
#     .reset_index()
# )

# # =========================
# # ⑥ VC差分
# # =========================
# df_dis_wide = df_dis.pivot_table(
#     index=[
#         "氏名",
#         "relation",
#         "relation_num",
#         "被験者の性別"
#     ],
#     columns="ボイスチェンジャー",
#     values="第１ラウンド",
#     aggfunc="mean"
# ).reset_index()

# # 列名変更
# df_dis_wide = df_dis_wide.rename(columns={
#     "なし": "absent",
#     "あり": "present"
# })

# # 差分
# df_dis_wide["diff"] = (
#     df_dis_wide["present"]
#     - df_dis_wide["absent"]
# )

# # =========================
# # ⑦ 結合
# # =========================
# df_plot = pd.merge(
#     df_dis_wide,
#     df_body_mean,
#     on="氏名"
# )

# print("Total N =", len(df_plot))

# # =========================
# # ⑧ 英語ラベル
# # =========================
# label_map = {
#     "ownership": "Sense of Ownership",
#     "agency": "Sense of Agency",
#     "self_rep": "Self-Representation"
# }

# # =========================
# # ⑨ 単純傾き
# # =========================
# def regression_stats(df, x_col, label):

#     print("\n")
#     print("=" * 70)
#     print(label)
#     print("=" * 70)

#     if len(df) < 2:

#         print("Not enough data")
#         return

#     X = sm.add_constant(df[x_col])
#     y = df["diff"]

#     model = sm.OLS(y, X).fit()

#     print(f"N = {len(df)}")
#     print(f"Mean diff = {df['diff'].mean():.3f}")
#     print(f"β = {model.params[x_col]:.3f}")
#     print(f"p = {model.pvalues[x_col]:.3f}")
#     print(f"R² = {model.rsquared:.3f}")

# # =========================
# # ⑩ interaction回帰
# # =========================
# def interaction_analysis(df, x_col, label):

#     print("\n")
#     print("=" * 70)

#     print(
#         f"Regression: diff ~ {x_col} * relation"
#     )

#     print(label)

#     print("=" * 70)

#     model = smf.ols(
#         f"diff ~ {x_col} * relation_num",
#         data=df
#     ).fit()

#     print(model.summary())

# # =========================
# # ⑪ メイン
# # =========================
# for var in [
#     "ownership",
#     "agency",
#     "self_rep"
# ]:

#     print("\n======================")
#     print(label_map[var])
#     print("======================")

#     for participant in [
#         "男性",
#         "女性"
#     ]:

#         sub = df_plot[
#             df_plot["被験者の性別"]
#             == participant
#         ]

#         p_label = (
#             "Male"
#             if participant == "男性"
#             else "Female"
#         )

#         # =====================
#         # Same / Different
#         # =====================
#         same_df = sub[
#             sub["relation"] == "Same"
#         ]

#         diff_df = sub[
#             sub["relation"] == "Different"
#         ]

#         # =====================
#         # 単純傾き
#         # =====================
#         regression_stats(
#             same_df,
#             var,
#             f"{p_label} - Same"
#         )

#         regression_stats(
#             diff_df,
#             var,
#             f"{p_label} - Different"
#         )

#         # =====================
#         # interaction
#         # =====================
#         interaction_analysis(
#             sub,
#             var,
#             f"{p_label} Participants"
#         )

#         # =====================
#         # グラフ
#         # =====================
#         plt.figure(figsize=(6,5))

#         sns.regplot(
#             data=same_df,
#             x=var,
#             y="diff",
#             color="blue",
#             label="Same-Gender Interaction",
#             scatter_kws={"s":80}
#         )

#         sns.regplot(
#             data=diff_df,
#             x=var,
#             y="diff",
#             color="red",
#             label="Different-Gender Interaction",
#             scatter_kws={"s":80}
#         )

#         # 横線
#         plt.axhline(
#             0,
#             linestyle="--",
#             color="gray"
#         )

#         # ラベル
#         plt.xlabel(
#             label_map[var]
#         )

#         plt.ylabel(
#             "Offer Difference (VC On − Off)"
#         )

#         # タイトル
#         plt.title(
#             f"{label_map[var]} ({p_label})"
#         )

#         # 見た目
#         plt.legend()
#         plt.grid(True)

#         # 見切れ防止
#         plt.ylim(-50, 55)

#         plt.yticks(
#             range(-50, 56, 10)
#         )

#         plt.margins(y=0.1)

#         plt.tight_layout()

#         plt.show()