import pandas as pd
import numpy as np

# 同性・異性の実験者との組み合わせで行動が変わるか
import pandas as pd
import numpy as np

# ====================================
# ✅ Cohen's d 関数
# ====================================
def cohens_d(x1, x2):
    n1, n2 = len(x1), len(x2)
    s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / (n1 + n2 - 2))
    return (np.mean(x1) - np.mean(x2)) / s_pooled

# ====================================
# ✅ 分析関数（4条件×Rごと）
# ====================================
def analyze_d_by_pair(file, label, value_col):
    df = pd.read_csv(file)
    df.columns = ["participant", "participant_gender", "experimenter_gender", "voice", value_col]
    df["voice"] = df["voice"].replace({"あり": "voicechanger", "なし": "not_voicechanger"})
    df["participant_gender"] = df["participant_gender"].replace({"男性": "male", "女性": "female"})
    df["experimenter_gender"] = df["experimenter_gender"].replace({"男性": "male", "女性": "female"})
    
    print(f"\n==============================")
    print(f"{label} : Cohen's d by Participant × Experimenter Gender")
    print("==============================")

    results = []
    # 4条件（male-male, male-female, female-male, female-female）
    for p_gender in ["male", "female"]:
        for e_gender in ["male", "female"]:
            subset = df[
                (df["participant_gender"] == p_gender) & 
                (df["experimenter_gender"] == e_gender)
            ]
            if subset.empty:
                continue
            group_with = subset[subset["voice"] == "voicechanger"][value_col]
            group_without = subset[subset["voice"] == "not_voicechanger"][value_col]
            d = cohens_d(group_with, group_without)
            mean_with = group_with.mean()
            mean_without = group_without.mean()
            results.append({
                "Participant_gender": p_gender,
                "Experimenter_gender": e_gender,
                "d": round(d, 3),
                "Mean_with": round(mean_with, 2),
                "Mean_without": round(mean_without, 2),
                "n": len(subset)
            })
    result_df = pd.DataFrame(results)
    print(result_df)
    return result_df


# ====================================
# 🟦 R1
# ====================================
r1_df = analyze_d_by_pair("R1_round.csv", "R1: First Offer", "offer")

# ====================================
# 🟩 R3
# ====================================
r3_df = analyze_d_by_pair("R3_round.csv", "R3: Second Offer", "offer")

# ====================================
# 🟥 R4
# ====================================
df4 = pd.read_csv("R4_round.csv")
df4.columns = ["participant", "participant_gender", "experimenter_gender", "voice", "accept"]
df4["accept"] = df4["accept"].replace({"受諾": 1, "拒否": 0})
df4.to_csv("R4_round_numeric.csv", index=False)
r4_df = analyze_d_by_pair("R4_round_numeric.csv", "R4: Acceptance Rate", "accept")

# ====================================
# ✅ 全体まとめ
# ====================================
print("\n==============================")
print("Summary of All Rounds (Cohen's d by Participant × Experimenter Gender)")
print("==============================")

summary = pd.concat([
    r1_df.assign(Round="R1"),
    r3_df.assign(Round="R3"),
    r4_df.assign(Round="R4")
])[["Round", "Participant_gender", "Experimenter_gender", "d", "Mean_with", "Mean_without", "n"]]

print(summary)
