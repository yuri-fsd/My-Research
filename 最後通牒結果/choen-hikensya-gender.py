import pandas as pd
import numpy as np


# ✅ Cohen's d 
def cohens_d(x1, x2):
    n1, n2 = len(x1), len(x2)
    s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
    return (np.mean(x1) - np.mean(x2)) / s_pooled

# ✅ 関数：ラウンド別に性別ごと Cohen’s d を算出
def analyze_round(file, label, value_col):
    df = pd.read_csv(file)
    df.columns = ["participant", "participant_gender", "experimenter_gender", "voice", value_col]
    
    # カテゴリ統一
    df["voice"] = df["voice"].replace({"あり": "voicechanger", "なし": "not_voicechanger"})
    df["participant_gender"] = df["participant_gender"].replace({"男性": "male", "女性": "female"})
    df["experimenter_gender"] = df["experimenter_gender"].replace({"男性": "male", "女性": "female"})

    print(f"\n==============================")
    print(f"{label} : Cohen's d by Gender")
    print("==============================")

    results = []
    
    for gender in ["male", "female"]:
        subset = df[df["participant_gender"] == gender]
        group_with = subset[subset["voice"] == "voicechanger"][value_col]
        group_without = subset[subset["voice"] == "not_voicechanger"][value_col]
        d = cohens_d(group_with, group_without)
        mean_with = group_with.mean()
        mean_without = group_without.mean()
        results.append({
            "Gender": gender,
            "d": round(d, 3),
            "Mean_with": round(mean_with, 2),
            "Mean_without": round(mean_without, 2),
            "n": len(subset)
        })
    
    # 結果をDataFrameで表示
    result_df = pd.DataFrame(results)
    print(result_df)
    return result_df


# 🟦 R1：初回提案
r1_df = analyze_round("R1_round.csv", "R1: First Offer Amount", "offer")

# 🟩 R3：2回目提案
r3_df = analyze_round("R3_round.csv", "R3: Second Offer Amount", "offer")

# 🟥 R4：受諾率（1=受諾, 0=拒否に変換）
df4 = pd.read_csv("R4_round.csv")
df4.columns = ["participant", "participant_gender", "experimenter_gender", "voice", "accept"]
df4["accept"] = df4["accept"].replace({"受諾": 1, "拒否": 0})
df4.to_csv("R4_round_numeric.csv", index=False)

r4_df = analyze_round("R4_round_numeric.csv", "R4: Acceptance Rate", "accept")

# ====================================
# ✅ まとめ表示
# ====================================
print("\n==============================")
print("Summary of All Rounds (Cohen's d by Gender)")
print("==============================")
summary = pd.concat([
    r1_df.assign(Round="R1"),
    r3_df.assign(Round="R3"),
    r4_df.assign(Round="R4")
])[["Round", "Gender", "d", "Mean_with", "Mean_without", "n"]]

print(summary)
