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

speech_df = pd.read_csv(
    "被験者の発話データ/情報単位スコア一覧.csv"
)

# ==================================================
# 身体化感覚アンケート
# ==================================================

body_df = pd.read_csv(
    "被験者の発話データ/2025身体化感覚アンケート.csv"
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

print(f"\n総被験者数: {df['name'].nunique()} 名")

# ==================================================
# 日本語ラベル
# ==================================================

scale_labels = {
    "ownership":"所有感",
    "agency":"主体感",
    "self_location":"自己表象"
}

# ==================================================
# LMM
# ==================================================

def run_lmm(data, scale, group_name):

    print("\n")
    print("#"*60)
    print(f"【{group_name}】 尺度: {scale_labels[scale]}")
    print("#"*60)

    data_copy = data.copy()

    data_copy[f"{scale}_z"] = (
        StandardScaler()
        .fit_transform(data_copy[[scale]])
    )

    formula = (
        f"info_unit_score "
        f"~ voice_changer "
        f"* relation "
        f"* {scale}_z"
    )

    model = smf.mixedlm(
        formula=formula,
        data=data_copy,
        groups=data_copy["name"],
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
            data=data_copy,
            groups=data_copy["name"]
        )

        result = model.fit()

    print(result.summary())

    return result, data_copy

# ==================================================
# グラフ
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

    for vc in ["あり","なし"]:

        for rel in ["Same","Different"]:

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
                [pred_df,tmp],
                ignore_index=True
            )

    plt.figure(figsize=(8,6))

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
        scale_labels[scale],
        fontsize=12
    )

    plt.ylabel(
        "自己開示スコア",
        fontsize=12
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()

# ==================================================
# 性別ごとデータ
# ==================================================

female_df = (
    df[
        df["participant_gender"]=="女性"
    ]
    .copy()
)

male_df = (
    df[
        df["participant_gender"]=="男性"
    ]
    .copy()
)

# ==================================================
# グループ
# ==================================================

groups = [

    {
        "name":"全体",
        "data":df
    },

    {
        "name":"女性被験者",
        "data":female_df
    },

    {
        "name":"男性被験者",
        "data":male_df
    }

]

# ==================================================
# 分析する尺度
# ==================================================

scales = [
    "ownership",
    "agency",
    "self_location"
]

# ==================================================
# 実行
# ==================================================

for group in groups:

    for scale in scales:

        result, data_with_z = run_lmm(
            group["data"],
            scale,
            group["name"]
        )

        plot_lmm_prediction(
            data_with_z,
            scale,
            result,
            f"{group['name']} × {scale_labels[scale]}"
        )