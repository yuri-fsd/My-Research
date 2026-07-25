import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== データ読み込み =====
df = pd.read_csv("情報単位スコア一覧.csv")

df = df.rename(columns={
    "participant_gender": "p_gender",
    "experimenter_gender": "e_gender",
    "voice_changer": "voice",
    "info_unit_score": "score"
})

df["voice"] = df["voice"].replace({"あり": "VCあり", "なし": "VCなし"})

# ===== 条件定義 =====
conditions = [
    ("全体", df),
    ("男性被験者", df[df["p_gender"] == "男"]),
    ("女性被験者", df[df["p_gender"] == "女"]),
    ("男性×男性", df[(df["p_gender"] == "男") & (df["e_gender"] == "男性")]),
    ("女性×女性", df[(df["p_gender"] == "女") & (df["e_gender"] == "女性")]),
    ("男性×女性", df[(df["p_gender"] == "男") & (df["e_gender"] == "女性")]),
    ("女性×男性", df[(df["p_gender"] == "女") & (df["e_gender"] == "男性")]),
]

# ===== 平均とSDを集計 =====
means_vc = []
means_no = []
stds_vc = []
stds_no = []
labels = []

for label, sub in conditions:
    labels.append(label)
    vc = sub[sub["voice"] == "VCあり"]["score"]
    no = sub[sub["voice"] == "VCなし"]["score"]

    means_vc.append(vc.mean())
    means_no.append(no.mean())
    stds_vc.append(vc.std())
    stds_no.append(no.std())

# ===== グラフ =====
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, means_no, width, yerr=stds_no, label="VCなし", capsize=4)
plt.bar(x + width/2, means_vc, width, yerr=stds_vc, label="VCあり", capsize=4)

plt.xticks(x, labels, rotation=30)
plt.ylabel("情報単位数（平均）")
plt.title("VC有無による情報単位数の比較（条件別）")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
