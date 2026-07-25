import pandas as pd
from scipy.stats import ttest_1samp
import numpy as np

# =====================================
# 1. データ読み込み（ワイド形式）
# =====================================
df = pd.read_csv("2025距離測定結果.csv")

# タイムスタンプ削除
df = df.drop(columns=[df.columns[0]])

# 数値化
for c in df.columns[2:]:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# =====================================
# 2. 接近回数をカウントする関数
# =====================================
def count_approach(row, cond, exp):
    cols = [f"{cond}ー{exp}{i}" for i in range(1, 10)]
    vals = [row[c] for c in cols if c in row and not pd.isna(row[c])]
    return len(vals)

# =====================================
# 3. 近づき回数のデータフレーム作成
# =====================================
rows = []

for _, r in df.iterrows():
    name = r["氏名"]
    p_gender = r["被験者の性別"]

    for exp_gender in ["男性", "女性"]:
        # なし条件
        cnt_none = len([c for c in df.columns if c.startswith(f"なしー{exp_gender}") and not pd.isna(r[c])])
        # あり条件
        cnt_vc = len([c for c in df.columns if c.startswith(f"ありー{exp_gender}") and not pd.isna(r[c])])

        rows.append({
            "氏名": name,
            "被験者性別": p_gender,
            "実験者性別": exp_gender,
            "なし回数": cnt_none,
            "あり回数": cnt_vc,
            "差分": cnt_vc - cnt_none  # 差分（あり - なし）
        })

df_cnt = pd.DataFrame(rows)

# =====================================
# 4. t検定 & Cohen's d 関数
# =====================================
def analyze(label, sub):
    diffs = sub["差分"]
    if len(diffs) < 2:
        print(f"\n=== {label} ===\nサンプル不足")
        return
    mean_diff = diffs.mean()
    t, p = ttest_1samp(diffs, 0)
    d = mean_diff / diffs.std(ddof=1)

    print(f"\n=== {label} ===")
    print(f"平均差分 = {mean_diff:.3f}")
    print(f"t = {t:.3f}, p = {p:.4f}")
    print(f"Cohen’s d = {d:.3f}")

# =====================================
# 5. 7条件で分析
# =====================================

# VC有無（全体）
analyze("VC有無（全体）", df_cnt)

# 男性被験者
analyze("男性被験者", df_cnt[df_cnt["被験者性別"]=="男性"])

# 女性被験者
analyze("女性被験者", df_cnt[df_cnt["被験者性別"]=="女性"])

# 男性被 × 男性実
analyze("男性被 × 男性実",
        df_cnt[(df_cnt["被験者性別"]=="男性")&(df_cnt["実験者性別"]=="男性")])

# 女性被 × 女性実
analyze("女性被 × 女性実",
        df_cnt[(df_cnt["被験者性別"]=="女性")&(df_cnt["実験者性別"]=="女性")])

# 男性被 × 女性実
analyze("男性被 × 女性実",
        df_cnt[(df_cnt["被験者性別"]=="男性")&(df_cnt["実験者性別"]=="女性")])

# 女性被 × 男性実
analyze("女性被 × 男性実",
        df_cnt[(df_cnt["被験者性別"]=="女性")&(df_cnt["実験者性別"]=="男性")])
