import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ================================
# 1. データ（行動タイプ × 条件）
# ================================
data = {
    "Attraction": [7,2,5,1,1,1,4],   # 旧Approach
    "Avoidance":  [8,5,3,3,1,2,2],   # 旧Cautious
    "Hesitation": [7,2,5,2,5,0,0],   # 旧Hesitation
    "Smooth":     [18,11,7,4,3,7,4]  # 旧Smooth
}

# ================================
# 2. 条件名（英語）
# ================================
conditions = [
    "VC Overall",
    "Male Participant",
    "Female Participant",
    "Male → Male",
    "Female → Female",
    "Male → Female",
    "Female → Male"
]

df = pd.DataFrame(data, index=conditions)

# ================================
# 3. ヒートマップ描画
# ================================
plt.figure(figsize=(10,6))
sns.heatmap(df, annot=True, fmt="d", cmap="Blues")

plt.title("Psychological Behavior Types × Conditions", fontsize=15)
plt.xlabel("Behavior Type", fontsize=12)
plt.ylabel("Condition", fontsize=12)

plt.tight_layout()
plt.show()
