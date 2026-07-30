# 条件ごとの受諾率ヒートマップ
# 横軸：ボイスチェンジャーの有無
# 縦軸：被験者の性別（＋オプションで実験者性別も考慮）
# 色：各条件での平均受諾率（0〜1）

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# データ読み込み
df = pd.read_csv("最後通牒結果/R4_round.csv")
df.columns = ["participant", "participant_gender", "experimenter_gender", "voice", "accept"]

# カテゴリ変換
df["voice"] = df["voice"].replace({"あり": "voicechanger", "なし": "not_voicechanger"})
df["participant_gender"] = df["participant_gender"].replace({"男性": "male", "女性": "female"})
df["experimenter_gender"] = df["experimenter_gender"].replace({"男性": "male", "女性": "female"})
df["accept"] = df["accept"].replace({"受諾": 1, "拒否": 0})

# 💡 被験者性別 × ボイスチェンジャーで平均受諾率を算出
pivot_table = df.pivot_table(
    values="accept",
    index="participant_gender",
    columns="voice",
    aggfunc="mean"
)

print(pivot_table)  # 平均受諾率の表を確認

# ヒートマップ描画
plt.figure(figsize=(6,4))
sns.heatmap(
    pivot_table,
    annot=True, fmt=".2f", cmap="coolwarm", vmin=0, vmax=1, cbar_kws={'label': 'Acceptance Rate'}
)
plt.title("Acceptance Rate by Voice Condition and Participant Gender (R4)")
plt.ylabel("Participant Gender")
plt.xlabel("Voice Changer Condition")
plt.tight_layout()
plt.show()

# 女性：ボイスチェンジャー使用で受諾率が下がる（0.60→0.55）
# 男性：ボイスチェンジャー使用で受諾率が上がる（0.60→0.65）

# 性別によってボイスチェンジャーの影響の“方向”が逆。

# これ、実は ロジスティック回帰の交互作用項が表していたものと一致
# （p=0.51 で有意ではなかったけど、傾向としては同じ方向）