import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===== データ読み込み =====
df = pd.read_csv("最後通牒結果/R4_round.csv")
df.columns = ["participant", "participant_gender", "experimenter_gender", "voice", "accept"]

# ===== カテゴリ変換 =====
df["voice"] = df["voice"].replace({"あり": "voicechanger", "なし": "not_voicechanger"})
df["participant_gender"] = df["participant_gender"].replace({"男性": "male", "女性": "female"})
df["experimenter_gender"] = df["experimenter_gender"].replace({"男性": "male", "女性": "female"})
df["accept"] = df["accept"].replace({"受諾": 1, "拒否": 0})

# ===== 被験者性別 × 実験者性別 × ボイスチェンジャーの組み合わせで平均受諾率を算出 =====
pivot_table = df.pivot_table(
    values="accept",
    index=["participant_gender", "experimenter_gender"],
    columns="voice",
    aggfunc="mean"
)

print("=== 平均受諾率（Participant × Experimenter × Voice） ===")
print(pivot_table)
print("\n")

# ===== ヒートマップ描画 =====
plt.figure(figsize=(8,5))
sns.heatmap(
    pivot_table,
    annot=True, fmt=".2f",
    cmap="coolwarm", vmin=0, vmax=1,
    linewidths=0.5, linecolor="gray",
    cbar_kws={'label': 'Acceptance Rate'}
)
plt.title("Acceptance Rate by Participant × Experimenter × Voice Condition (R4)", fontsize=12)
plt.ylabel("Participant × Experimenter Gender")
plt.xlabel("Voice Changer Condition")
plt.tight_layout()
plt.show()

# ===== 結果の傾向コメント（自動出力） =====
print("▼ Interpretation")
for (p_gen, e_gen), row in pivot_table.iterrows():
    diff = row["voicechanger"] - row["not_voicechanger"]
    direction = "上昇" if diff > 0 else "低下" if diff < 0 else "変化なし"
    print(f"{p_gen} × {e_gen}: 受諾率が{direction} ({row['not_voicechanger']:.2f} → {row['voicechanger']:.2f})")
