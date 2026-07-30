import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===== データ読み込み =====
df = pd.read_csv("最後通牒結果/R4_round.csv")

# ===== 前処理 =====
df.columns = ["participant", "participant_gender", "experimenter_gender", "voice", "accept"]
df["voice"] = df["voice"].replace({"あり": "voicechanger", "なし": "not_voicechanger"})
df["participant_gender"] = df["participant_gender"].replace({"男性": "male", "女性": "female"})
df["experimenter_gender"] = df["experimenter_gender"].replace({"男性": "male", "女性": "female"})
df["accept"] = df["accept"].replace({"受諾": 1, "拒否": 0})

# ===== 関数定義：Cohen's d =====
def cohens_d(x1, x2):
    n1, n2 = len(x1), len(x2)
    s1, s2 = x1.std(ddof=1), x2.std(ddof=1)
    s_pooled = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
    return (x1.mean() - x2.mean()) / s_pooled if s_pooled > 0 else np.nan

# ===== ① 全体（性別を考慮しない） =====
with_voice = df[df["voice"] == "voicechanger"]["accept"]
without_voice = df[df["voice"] == "not_voicechanger"]["accept"]
d_total = cohens_d(with_voice, without_voice)
print(f"\n=== 全体 ===")
print(f"Cohen's d (全体, ボイスあり vs なし) = {d_total:.2f}")

# ===== ② 被験者の性別ごと =====
print("\n=== 被験者性別ごとの Cohen's d ===")
for gender in ["male", "female"]:
    subset = df[df["participant_gender"] == gender]
    with_v = subset[subset["voice"] == "voicechanger"]["accept"]
    without_v = subset[subset["voice"] == "not_voicechanger"]["accept"]
    d = cohens_d(with_v, without_v)
    print(f"{gender}: d = {d:.2f}")

# ===== ③ 被験者×実験者の組み合わせ =====
print("\n=== 被験者×実験者の組み合わせごとの Cohen's d ===")
results = []
for p_gender in ["male", "female"]:
    for e_gender in ["male", "female"]:
        subset = df[
            (df["participant_gender"] == p_gender) &
            (df["experimenter_gender"] == e_gender)
        ]
        if len(subset) < 2:
            continue
        with_v = subset[subset["voice"] == "voicechanger"]["accept"]
        without_v = subset[subset["voice"] == "not_voicechanger"]["accept"]
        d = cohens_d(with_v, without_v)
        results.append({
            "Participant": p_gender,
            "Experimenter": e_gender,
            "Cohen_d": round(d, 2)
        })

# 結果を表形式で表示
results_df = pd.DataFrame(results)
print(results_df)

# ===== ヒートマップ =====
pivot_d = results_df.pivot(index="Participant", columns="Experimenter", values="Cohen_d")
plt.figure(figsize=(6,5))
sns.heatmap(
    pivot_d, annot=True, fmt=".2f",
    cmap="coolwarm", center=0, vmin=-0.5, vmax=0.5,
    cbar_kws={'label': "Cohen's d"}
)
plt.title("Cohen's d for Acceptance Rate by Participant × Experimenter (R4)")
plt.ylabel("Participant Gender")
plt.xlabel("Experimenter Gender")
plt.tight_layout()
plt.show()
