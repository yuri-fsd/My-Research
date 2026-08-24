# ============================================================
# Ownership × Different Character Score
# Spearman Rank Correlation Analysis
#
# 1. Calculate ownership score for each participant
# 2. Calculate median ownership score
# 3. Extract participants above the median
# 4. Analyze:
#    "ボイスチェンジャーを使用した場合では，
#     普段の自分と違うキャラになったと感じた"
# 5. Calculate Spearman rank correlations
# 6. Plot results using English labels only
# ============================================================


# ============================================================
# Import
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ============================================================
# ① Load data
# ============================================================

# Embodiment questionnaire
df_body = pd.read_csv(
    "お題回答結果/被験者の発話データ/2025身体化感覚アンケート.csv"
)

# Post-experiment questionnaire
df_post = pd.read_csv(
    "お題回答結果/被験者の発話データ/2025実験後アンケート.csv"
)


# ============================================================
# ② Clean column names
# ============================================================

df_body.columns = df_body.columns.str.strip()
df_post.columns = df_post.columns.str.strip()


# ============================================================
# ③ Clean participant names
# ============================================================

def clean_name(df):

    df["氏名"] = (
        df["氏名"]
        .astype(str)
        .str.strip()
        .str.replace("　", "", regex=False)
        .str.replace(" ", "", regex=False)
    )

    return df


df_body = clean_name(df_body)
df_post = clean_name(df_post)


# ============================================================
# ④ Ownership questionnaire items
# ============================================================

ownership_cols = [

    "自身のアバターの手足は, 自分自身の手足のように感じた",

    "自身のアバターを「自分の身体」として受け入れていた",

    "自分の身体はアバターと同じ場所にいるように感じた",

    "自分の意識がアバターと一体化しているように感じた",

    "自分はアバターの目から見ているように感じた"

]


# ============================================================
# ⑤ Calculate ownership score for each questionnaire response
# ============================================================

df_body["ownership"] = (
    df_body[ownership_cols]
    .mean(axis=1)
)


# ============================================================
# ⑥ Calculate mean ownership score for each participant
#
# Each participant has VC On / VC Off responses.
# These are averaged to obtain one ownership score per person.
# ============================================================

ownership_person_mean = (
    df_body
    .groupby("氏名")["ownership"]
    .mean()
    .reset_index()
)


# ============================================================
# ⑦ Display ownership scores
# ============================================================

print("\n=============================================")
print("各被験者の所有感平均")
print("=============================================")

print(
    ownership_person_mean
    .sort_values(
        "ownership",
        ascending=False
    )
    .to_string(index=False)
)


# ============================================================
# ⑧ Calculate median ownership score
# ============================================================

ownership_median = (
    ownership_person_mean["ownership"]
    .median()
)


print("\n=============================================")
print("所有感の中央値")
print("=============================================")

print(
    f"{ownership_median:.3f}"
)


# ============================================================
# ⑨ Extract High Ownership Group
#
# Strictly ABOVE the median
# ============================================================

high_ownership = (
    ownership_person_mean[
        ownership_person_mean["ownership"]
        > ownership_median
    ]
    .copy()
)


high_names = (
    high_ownership["氏名"]
    .tolist()
)


print("\n=============================================")
print("高所有感群")
print("=============================================")

print(
    high_ownership
    .sort_values(
        "ownership",
        ascending=False
    )
    .to_string(index=False)
)


print(
    f"\n高所有感群の人数："
    f"{len(high_ownership)}"
)


# ============================================================
# ⑩ Post-experiment questionnaire item
# ============================================================

question_col = (
    "ボイスチェンジャーを使用した場合では，"
    "普段の自分と違うキャラになったと感じた"
)


# ============================================================
# ⑪ Check questionnaire column
# ============================================================

if question_col not in df_post.columns:

    print("\n⚠️ 指定したアンケート項目が見つかりません。")
    print("\nCSV内の列名：")

    for i, col in enumerate(df_post.columns):
        print(i, repr(col))

    raise KeyError(
        f"Column not found: {question_col}"
    )


# ============================================================
# ⑫ Extract questionnaire score
# ============================================================

df_character = (
    df_post[
        [
            "氏名",
            question_col
        ]
    ]
    .copy()
)


df_character = df_character.rename(
    columns={
        question_col: "different_character"
    }
)


# Convert to numeric
df_character["different_character"] = pd.to_numeric(
    df_character["different_character"],
    errors="coerce"
)


# Remove missing values
df_character = df_character.dropna(
    subset=["different_character"]
)


# ============================================================
# ⑬ If multiple responses exist per participant,
#    calculate participant mean
# ============================================================

df_character_mean = (
    df_character
    .groupby("氏名")["different_character"]
    .mean()
    .reset_index()
)


# ============================================================
# ⑭ Merge ownership and questionnaire data
# ============================================================

df_all = pd.merge(
    ownership_person_mean,
    df_character_mean,
    on="氏名",
    how="inner"
)


# ============================================================
# ⑮ Display all participants
# ============================================================

print("\n=============================================")
print("全被験者：所有感 × 普段と違うキャラになった程度")
print("=============================================")

print(
    df_all
    .sort_values(
        "ownership",
        ascending=False
    )
    .to_string(index=False)
)


# ============================================================
# ⑯ Extract High Ownership Group
# ============================================================

df_high = (
    df_all[
        df_all["氏名"].isin(high_names)
    ]
    .copy()
)


print("\n=============================================")
print("高所有感群：アンケート結果")
print("=============================================")

print(
    df_high
    .sort_values(
        "ownership",
        ascending=False
    )
    .to_string(index=False)
)


# ============================================================
# ⑰ Basic statistics for High Ownership Group
# ============================================================

print("\n=============================================")
print("高所有感群：アンケート基本統計")
print("=============================================")


print(
    f"人数："
    f"{len(df_high)}"
)

print(
    f"平均："
    f"{df_high['different_character'].mean():.3f}"
)

print(
    f"中央値："
    f"{df_high['different_character'].median():.3f}"
)

print(
    f"標準偏差："
    f"{df_high['different_character'].std():.3f}"
)

print(
    f"最小値："
    f"{df_high['different_character'].min():.0f}"
)

print(
    f"最大値："
    f"{df_high['different_character'].max():.0f}"
)


# ============================================================
# ⑱ Distribution of questionnaire responses
# ============================================================

print("\n=============================================")
print("高所有感群：回答の分布")
print("=============================================")


for score in range(1, 8):

    count = (
        df_high["different_character"]
        .eq(score)
        .sum()
    )

    print(
        f"{score}：{count}人"
    )


# ============================================================
# ⑲ Spearman correlation
# High Ownership Group
# ============================================================

rho_high, p_high = spearmanr(
    df_high["ownership"],
    df_high["different_character"]
)


print("\n=============================================")
print("高所有感群：Spearman順位相関")
print("=============================================")

print(
    f"ρ = {rho_high:.3f}"
)

print(
    f"p = {p_high:.3f}"
)


# ============================================================
# ⑳ Spearman correlation
# All Participants
# ============================================================

rho_all, p_all = spearmanr(
    df_all["ownership"],
    df_all["different_character"]
)


print("\n=============================================")
print("全被験者：Spearman順位相関")
print("=============================================")

print(
    f"ρ = {rho_all:.3f}"
)

print(
    f"p = {p_all:.3f}"
)


# ============================================================
# ㉑ Plot function
#
# IMPORTANT:
# All text displayed in the figure is English.
# Participant names are NOT displayed.
# ============================================================

def plot_spearman(
    df,
    title
):

    # ----------------------------------------
    # Spearman correlation
    # ----------------------------------------

    rho, p = spearmanr(
        df["ownership"],
        df["different_character"]
    )


    # ----------------------------------------
    # Figure
    # ----------------------------------------

    plt.figure(
        figsize=(8, 6)
    )


    # ----------------------------------------
    # Scatter plot
    # ----------------------------------------

    plt.scatter(
        df["ownership"],
        df["different_character"],
        s=100
    )


    # ----------------------------------------
    # Axis labels
    # ----------------------------------------

    plt.xlabel(
        "Ownership Score",
        fontsize=12
    )

    plt.ylabel(
        "Different Character Score",
        fontsize=12
    )


    # ----------------------------------------
    # Title
    # ----------------------------------------

    plt.title(
        title,
        fontsize=14
    )


    # ----------------------------------------
    # Likert scale
    # ----------------------------------------

    plt.yticks(
        range(1, 8)
    )

    plt.ylim(
        0.5,
        7.5
    )


    # ----------------------------------------
    # Grid
    # ----------------------------------------

    plt.grid(
        True,
        alpha=0.3
    )


    # ----------------------------------------
    # Spearman result
    # ----------------------------------------

    plt.text(
        0.05,
        0.95,

        f"Spearman rho = {rho:.3f}\n"
        f"p = {p:.3f}",

        transform=plt.gca().transAxes,

        verticalalignment="top",

        fontsize=12,

        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8
        )
    )


    plt.tight_layout()

    plt.show()


# ============================================================
# ㉒ High Ownership Group Plot
# ============================================================

plot_spearman(
    df_high,

    "Ownership vs. Different Character Score\n"
    "(High Ownership Group)"
)


# ============================================================
# ㉓ All Participants Plot
# ============================================================

plot_spearman(
    df_all,

    "Ownership vs. Different Character Score\n"
    "(All Participants)"
)