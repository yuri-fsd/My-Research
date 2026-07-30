import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = "Hiragino Sans"

# ==================================================
# 発話データ
# ==================================================

speech_df = pd.read_csv("最後通牒結果/R1_round.csv")

speech_df = speech_df.rename(columns={
    "氏名": "name",
    "被験者の性別": "participant_gender",
    "実験者の性別": "experimenter_gender",
    "ボイスチェンジャー": "voice_changer",
    "第１ラウンド": "info_unit_score"
})

# ==================================================
# 身体化感覚アンケート
# ==================================================

body_df = pd.read_csv(
    "2025身体化感覚アンケート.csv"
)

# ==================================================
# 発話データ前処理
# ==================================================

speech_df["participant_gender"] = (
    speech_df["participant_gender"]
    .astype(str)
    .str.strip()
)

speech_df["experimenter_gender"] = (
    speech_df["experimenter_gender"]
    .astype(str)
    .str.strip()
)

speech_df["voice_changer"] = (
    speech_df["voice_changer"]
    .astype(str)
    .str.strip()
)

speech_df["name"] = (
    speech_df["name"]
    .astype(str)
    .str.strip()
)

speech_df["relation"] = np.where(
    speech_df["participant_gender"]
    ==
    speech_df["experimenter_gender"],
    "Same",
    "Different"
)

speech_df["info_unit_score"] = pd.to_numeric(
    speech_df["info_unit_score"],
    errors="coerce"
)

speech_df = speech_df.dropna(
    subset=["info_unit_score"]
)

# ==================================================
# VC表記統一
# ==================================================

speech_df["voice_changer"] = (
    speech_df["voice_changer"]
    .replace({
        "有り": "あり",
        "無し": "なし"
    })
)

# ==================================================
# 身体化感覚アンケート
# ==================================================

body_df = pd.read_csv(
    "2025身体化感覚アンケート.csv"
)

# ==================================================
# 発話データ前処理
# ==================================================

speech_df["participant_gender"] = (
    speech_df["participant_gender"]
    .replace({
        "男":"男性",
        "女":"女性"
    })
)

speech_df["relation"] = np.where(
    speech_df["participant_gender"]
    ==
    speech_df["experimenter_gender"],
    "Same",
    "Different"
)

speech_df["voice_changer"] = (
    speech_df["voice_changer"]
    .astype(str)
)

speech_df["name"] = (
    speech_df["name"]
    .astype(str)
    .str.strip()
)

# ==================================================
# アンケート前処理
# ==================================================

body_df["氏名"] = (
    body_df["氏名"]
    .astype(str)
    .str.strip()
)

body_df["voice_changer"] = (
    body_df[
        "直前の実験でボイスチェンジャーを使用しましたか"
    ]
    .replace({
        "はい":"あり",
        "いいえ":"なし"
    })
)

# ==================================================
# 逆転項目
# ==================================================

body_df["Q2_r"] = 8 - body_df.iloc[:,4]
body_df["Q6_r"] = 8 - body_df.iloc[:,8]
body_df["Q12_r"] = 8 - body_df.iloc[:,14]

# ==================================================
# Ownership
# ==================================================

body_df["ownership"] = (
    body_df.iloc[:,3]
    + body_df["Q2_r"]
    + body_df.iloc[:,5]
    + body_df.iloc[:,6]
) / 4

# ==================================================
# Self Location
# ==================================================

body_df["self_location"] = (
    body_df.iloc[:,7]
    + body_df["Q6_r"]
    + body_df.iloc[:,9]
    + body_df.iloc[:,10]
) / 4

# ==================================================
# Agency
# ==================================================

body_df["agency"] = (
    body_df.iloc[:,11]
    + body_df.iloc[:,12]
    + body_df.iloc[:,13]
    + body_df["Q12_r"]
) / 4

# ==================================================
# マージ
# ==================================================

df = pd.merge(
    speech_df,
    body_df,
    left_on=["name","voice_changer"],
    right_on=["氏名","voice_changer"],
    how="inner"
)

print("被験者数")
print(df["name"].nunique())

# ==================================================
# z変換
# ==================================================

for scale in [
    "ownership",
    "agency",
    "self_location"
]:
    df[f"{scale}_z"] = StandardScaler().fit_transform(
        df[[scale]]
    )

# ==================================================
# VC効果量作成
# ==================================================

def create_effect_df(data):

    tmp = (
        data.pivot_table(
            index=["name","relation"],
            columns="voice_changer",
            values="info_unit_score"
        )
        .reset_index()
    )

    tmp["vc_effect"] = (
        tmp["あり"]
        -
        tmp["なし"]
    )

    return tmp

# ==================================================
# LMM
# ==================================================

def run_lmm(data, scale):

    print("\n")
    print("="*60)
    print(scale.upper())
    print("="*60)

    formula = (
        f"info_unit_score "
        f"~ voice_changer "
        f"* relation "
        f"* {scale}_z"
    )

    model = smf.mixedlm(
        formula=formula,
        data=data,
        groups=data["name"],
        re_formula="~voice_changer"
    )

    try:

        result = model.fit(
            method="lbfgs",
            maxiter=1000
        )

    except:

        print(
            "Random slope失敗 → Random interceptのみ"
        )

        model = smf.mixedlm(
            formula=formula,
            data=data,
            groups=data["name"]
        )

        result = model.fit()

    print(result.summary())

    return result

# ==================================================
# LMM Prediction Plot
# ==================================================

def plot_lmm_prediction(
    data,
    scale,
    result,
    title
):

    x = np.linspace(
        data[scale].min(),
        data[scale].max(),
        100
    )

    pred_df = pd.DataFrame()

    for vc in ["あり", "なし"]:

        for rel in ["Same", "Different"]:

            tmp = pd.DataFrame({

                scale: x,

                f"{scale}_z":
                    (
                        x
                        -
                        data[scale].mean()
                    )
                    /
                    data[scale].std(ddof=0),

                "voice_changer": vc,

                "relation": rel

            })

            tmp["pred"] = result.predict(tmp)

            tmp["VC"] = vc
            tmp["Relation"] = rel

            pred_df = pd.concat(
                [pred_df, tmp],
                ignore_index=True
            )

    plt.figure(figsize=(8,6))

    # 4本の予測直線のみ
    sns.lineplot(
        data=pred_df,
        x=scale,
        y="pred",
        hue="VC",
        style="Relation",
        linewidth=4
    )

    plt.title(
        title,
        fontsize=14
    )

    plt.xlabel(
        scale,
        fontsize=12
    )

    plt.ylabel(
        "提示額",
        fontsize=12
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()
    
    # ==================================================
# 性別ごと
# ==================================================

female_df = (
    df[
        df["participant_gender"] == "女性"
    ]
    .copy()
)

male_df = (
    df[
        df["participant_gender"] == "男性"
    ]
    .copy()
)

# ==================================================
# 全体
# ==================================================

print("\n\n")
print("########################")
print("OVERALL")
print("########################")

for scale in [
    "ownership",
    "agency",
    "self_location"
]:

    result = run_lmm(
        df,
        scale
    )

    plot_lmm_prediction(
        df,
        scale,
        result,
        f"Overall - {scale}"
    )

# ==================================================
# 女性
# ==================================================

print("\n\n")
print("########################")
print("FEMALE")
print("########################")

for scale in [
    "ownership",
    "agency",
    "self_location"
]:

    result = run_lmm(
        female_df,
        scale
    )

    plot_lmm_prediction(
        female_df,
        scale,
        result,
        f"Female - {scale}"
    )

# ==================================================
# 男性
# ==================================================

print("\n\n")
print("########################")
print("MALE")
print("########################")

for scale in [
    "ownership",
    "agency",
    "self_location"
]:

    result = run_lmm(
        male_df,
        scale
    )

    plot_lmm_prediction(
        male_df,
        scale,
        result,
        f"Male - {scale}"
    )