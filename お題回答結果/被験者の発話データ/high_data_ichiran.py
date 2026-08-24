import pandas as pd

# ============================================================
# ① データ読み込み
# ============================================================

# 自己開示データ
df_dis = pd.read_csv(
    "お題回答結果/被験者の発話データ/情報単位スコア一覧.csv"
)

# 身体化感覚アンケート
df_body = pd.read_csv(
    "お題回答結果/被験者の発話データ/2025身体化感覚アンケート.csv"
)


# ============================================================
# ② 列名整理
# ============================================================

df_dis = df_dis.rename(columns={"name": "氏名"})

df_dis.columns = df_dis.columns.str.strip()
df_body.columns = df_body.columns.str.strip()


# ============================================================
# ③ 名前クリーニング
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


df_dis = clean_name(df_dis)
df_body = clean_name(df_body)


# ============================================================
# ④ 所有感スコアを計算
# ============================================================

# これまでの分析で使用している所有感4項目
ownership_cols = [
    "自身のアバターの手足は, 自分自身の手足のように感じた",
    "自身のアバターを「自分の身体」として受け入れていた",
    "自分の身体はアバターと同じ場所にいるように感じた",
    "自分の意識がアバターと一体化しているように感じた"
]

# 各回答（VCあり・なし）の所有感
df_body["ownership"] = (
    df_body[ownership_cols]
    .mean(axis=1)
)


# ============================================================
# ⑤ 各被験者の所有感平均
#    VCあり・なしをまとめた平均
# ============================================================

ownership_person_mean = (
    df_body
    .groupby("氏名")["ownership"]
    .mean()
    .reset_index()
)


print("\n============================")
print("各被験者の所有感平均")
print("============================")

print(
    ownership_person_mean
    .sort_values("ownership", ascending=False)
    .to_string(index=False)
)


# ============================================================
# ⑥ 所有感の中央値
# ============================================================

ownership_median = ownership_person_mean["ownership"].median()


print("\n============================")
print("所有感の中央値")
print("============================")

print(f"中央値 = {ownership_median:.3f}")


# ============================================================
# ⑦ 中央値より高い被験者を抽出
# ============================================================

high_ownership = ownership_person_mean[
    ownership_person_mean["ownership"] > ownership_median
].copy()

high_names = high_ownership["氏名"].tolist()


print("\n============================")
print("所有感が中央値より高い被験者")
print("============================")

for _, row in high_ownership.sort_values(
    "ownership",
    ascending=False
).iterrows():

    print(
        f"{row['氏名']}："
        f"{row['ownership']:.3f}"
    )


print("\n============================")
print("抽出結果")
print("============================")

print(f"全被験者数：{len(ownership_person_mean)}")
print(f"中央値より高い被験者数：{len(high_names)}")

print("\n被験者名")

for name in high_names:
    print(name)


# ============================================================
# ⑧ 高所有感群だけ自己開示データから抽出
# ============================================================

df_compare = df_dis[
    df_dis["氏名"].isin(high_names)
].copy()


# ============================================================
# ⑨ 同性相手・異性相手を判定
# ============================================================

# participant_gender と experimenter_gender を統一
df_compare["participant_gender"] = (
    df_compare["participant_gender"]
    .replace({
        "男": "男性",
        "女": "女性"
    })
)

df_compare["experimenter_gender"] = (
    df_compare["experimenter_gender"]
    .replace({
        "男": "男性",
        "女": "女性"
    })
)


df_compare["relation"] = df_compare.apply(
    lambda row:
        "Same"
        if row["participant_gender"]
        == row["experimenter_gender"]
        else "Different",
    axis=1
)


# ============================================================
# ⑩ VCなし × 同性相手
# ============================================================

same_no_vc = (
    df_compare[
        (df_compare["relation"] == "Same")
        &
        (df_compare["voice_changer"] == "なし")
    ]
    .groupby("氏名")["info_unit_score"]
    .mean()
    .reset_index()
    .rename(
        columns={
            "info_unit_score": "VCなし×同性"
        }
    )
)


# ============================================================
# ⑪ VCあり × 異性相手
# ============================================================

different_vc = (
    df_compare[
        (df_compare["relation"] == "Different")
        &
        (df_compare["voice_changer"] == "あり")
    ]
    .groupby("氏名")["info_unit_score"]
    .mean()
    .reset_index()
    .rename(
        columns={
            "info_unit_score": "VCあり×異性"
        }
    )
)


# ============================================================
# ⑫ 2条件を結合
# ============================================================

comparison = pd.merge(
    same_no_vc,
    different_vc,
    on="氏名",
    how="outer"
)


# 差
comparison["差（VCあり異性 − VCなし同性）"] = (
    comparison["VCあり×異性"]
    - comparison["VCなし×同性"]
)


# 絶対差
# 「どれくらい近いか」を確認しやすくするため
comparison["絶対差"] = (
    comparison["差（VCあり異性 − VCなし同性）"]
    .abs()
)


# ============================================================
# ⑬ 被験者ごとの結果
# ============================================================

print("\n=============================================")
print("高所有感群：自己開示量の比較")
print("=============================================")

print(
    comparison
    .sort_values("氏名")
    .to_string(index=False)
)


# ============================================================
# ⑭ 高所有感群全体の平均
# ============================================================

print("\n=============================================")
print("高所有感群の平均")
print("=============================================")

print(
    f"VCなし × 同性相手："
    f"{comparison['VCなし×同性'].mean():.3f}"
)

print(
    f"VCあり × 異性相手："
    f"{comparison['VCあり×異性'].mean():.3f}"
)

print(
    f"平均差（VCあり異性 − VCなし同性）："
    f"{comparison['差（VCあり異性 − VCなし同性）'].mean():.3f}"
)

print(
    f"平均絶対差："
    f"{comparison['絶対差'].mean():.3f}"
)