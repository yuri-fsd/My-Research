import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = "Hiragino Sans"

# ==================================================
# 1. 最後通牒ゲーム（提示額）データの読み込み
# ==================================================
# ※実際のファイル名（"R1_round.csv"等）に合わせて適宜変更してください
speech_df = pd.read_csv("最後通牒結果/R1_round.csv")

speech_df = speech_df.rename(columns={
    "氏名": "name",
    "被験者の性別": "participant_gender",
    "実験者の性別": "experimenter_gender",
    "ボイスチェンジャー": "voice_changer",
    "第１ラウンド": "info_unit_score"  # 共通数式適用のために変数名はそのまま維持
})

# ==================================================
# 2. 身体化感覚アンケートデータの読み込み
# ==================================================
body_df = pd.read_csv("2025身体化感覚アンケート.csv")

# ==================================================
# 3. ゲームデータの前処理
# ==================================================
speech_df["participant_gender"] = speech_df["participant_gender"].astype(str).str.strip()
speech_df["experimenter_gender"] = speech_df["experimenter_gender"].astype(str).str.strip()
speech_df["voice_changer"] = speech_df["voice_changer"].astype(str).str.strip()
speech_df["name"] = speech_df["name"].astype(str).str.strip()

# ペア関係（同性・異性）の判定
speech_df["relation"] = np.where(
    speech_df["participant_gender"] == speech_df["experimenter_gender"],
    "Same",
    "Different"
)

# 提示額を数値型に変換して欠損値を除去
speech_df["info_unit_score"] = pd.to_numeric(speech_df["info_unit_score"], errors="coerce")
speech_df = speech_df.dropna(subset=["info_unit_score"])

# 表記の統一
speech_df["voice_changer"] = speech_df["voice_changer"].replace({"有り": "あり", "無し": "なし"})
speech_df["participant_gender"] = speech_df["participant_gender"].replace({"男": "男性", "女": "女性"})

# ==================================================
# 4. アンケートデータの前処理（指標の計算）
# ==================================================
body_df["氏名"] = body_df["氏名"].astype(str).str.strip()

# 逆転項目の処理
body_df["Q2_r"] = 8 - body_df.iloc[:, 4]
body_df["Q6_r"] = 8 - body_df.iloc[:, 8]
body_df["Q12_r"] = 8 - body_df.iloc[:, 14]

# 身体化感覚 3大因子の平均値算出
body_df["ownership"] = (body_df.iloc[:, 3] + body_df["Q2_r"] + body_df.iloc[:, 5] + body_df.iloc[:, 6]) / 4
body_df["self_location"] = (body_df.iloc[:, 7] + body_df["Q6_r"] + body_df.iloc[:, 9] + body_df.iloc[:, 10]) / 4
body_df["agency"] = (body_df.iloc[:, 11] + body_df.iloc[:, 12] + body_df.iloc[:, 13] + body_df["Q12_r"]) / 4

# 被験者ごとに一意の身体化スコアテーブルを作成（重複マージを防止）
body_summary = body_df[["氏名", "ownership", "self_location", "agency"]].drop_duplicates(subset=["氏名"])

# ==================================================
# 5. データセットの統合（氏名のみで安全にマージ）
# ==================================================
df = pd.merge(
    speech_df,
    body_summary,
    left_on="name",
    right_on="氏名",
    how="inner"
)

print(f"\n総被験者数: {df['name'].nunique()} 名")

# ==================================================
# 6. 日本語表記ラベル辞書（レイアウト用）
# ==================================================
scale_labels = {
    "ownership": "所有感",
    "agency": "主体感",
    "self_location": "自己表象"
}

# ==================================================
# 7. LMM（線形混合モデル）実行関数
# ==================================================
def run_lmm(data, scale, group_name):
    print("\n")
    print("#"*60)
    print(f"【{group_name}】 尺度: {scale_labels[scale]}")
    print("#"*60)

    data_copy = data.copy()
    
    # 選択されたグループ（全体・女性・男性）の内部でその都度スタンドアロンなz変換を実行
    data_copy[f"{scale}_z"] = StandardScaler().fit_transform(data_copy[[scale]])

    formula = f"info_unit_score ~ voice_changer * relation * {scale}_z"

    # Random slopeモデル
    model = smf.mixedlm(
        formula=formula,
        data=data_copy,
        groups=data_copy["name"],
        re_formula="~voice_changer"
    )

    try:
        result = model.fit(method="lbfgs", maxiter=1000)
    except:
        print("Random slope失敗 → Random interceptのみに切り替え")
        model = smf.mixedlm(
            formula=formula,
            data=data_copy,
            groups=data_copy["name"]
        )
        result = model.fit()

    print(result.summary())
    return result, data_copy

# ==================================================
# 8. 予測プロット関数（1枚目と完全に同じレイアウト）
# ==================================================
def plot_lmm_prediction(data_z, scale, result, title):
    # 軸の範囲を生データの最小〜最大値で設定
    x = np.linspace(data_z[scale].min(), data_z[scale].max(), 100)
    pred_df = pd.DataFrame()

    for vc in ["あり", "なし"]:
        for rel in ["Same", "Different"]:
            tmp = pd.DataFrame({
                scale: x,
                f"{scale}_z": (x - data_z[scale].mean()) / data_z[scale].std(ddof=0),
                "voice_changer": vc,
                "relation": rel
            })

            tmp["pred"] = result.predict(tmp)
            tmp["VC"] = vc
            tmp["Relation"] = rel

            pred_df = pd.concat([pred_df, tmp], ignore_index=True)

    plt.figure(figsize=(8, 6))

    # 4本の予測直線のみを太線でプロット（散布図は非表示にしてスッキリさせる）
    sns.lineplot(
        data=pred_df,
        x=scale,
        y="pred",
        hue="VC",
        style="Relation",
        linewidth=4
    )

    plt.title(title, fontsize=14)
    plt.xlabel(scale_labels[scale], fontsize=12)
    plt.ylabel("提示額", fontsize=12)  # 最後通牒ゲームの指標に修正
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==================================================
# 9. グループ分割とループ実行
# ==================================================
female_df = df[df["participant_gender"] == "女性"].copy()
male_df = df[df["participant_gender"] == "男性"].copy()

groups = [
    {"name": "全体", "data": df},
    {"name": "女性被験者", "data": female_df},
    {"name": "男性被験者", "data": male_df}
]

scales = ["ownership", "agency", "self_location"]

for group in groups:
    for scale in scales:
        # LMM実行
        result, data_with_z = run_lmm(group["data"], scale, group["name"])

        # 1枚目と全く同じ日本語タイトルルールでグラフを出力
        plot_lmm_prediction(
            data_with_z,
            scale,
            result,
            f"{group['name']} × {scale_labels[scale]}"
        )