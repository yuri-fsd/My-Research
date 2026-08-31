import pandas as pd
from scipy.stats import ttest_1samp
import numpy as np

# ================================
# 1. ロング形式データ読み込み
# ================================/
df = pd.read_csv("距離測定2025距離測定結果_long.csv")
df["距離"] = pd.to_numeric(df["距離"], errors="coerce")

# ================================
# 2. 最終試行のみ抽出
# ================================
df_last = df.sort_values("試行").dropna(subset=["試行"])
df_last = df_last.loc[df_last.groupby(
    ["氏名", "条件", "実験者性別"]
)["試行"].idxmax()]

# ================================
# 3. あり/なしのwideデータ作成
# ================================
wide = df_last.pivot_table(
    index=["氏名", "被験者性別", "実験者性別"],
    columns="条件",
    values="距離"
).reset_index()

# 両方そろっているデータのみ
wide = wide.dropna(subset=["あり", "なし"])

# 差分
wide["差分"] = wide["あり"] - wide["なし"]

# ================================
# 4. 分析関数（差分 → t検定 → Cohen d）
# ================================
def analyze(label, df_sub):
    diffs = df_sub["差分"]
    if len(diffs) < 2:
        print(f"\n=== {label} ===")
        print("サンプル不足")
        return

    mean_diff = diffs.mean()
    t_stat, p_val = ttest_1samp(diffs, 0)
    d = mean_diff / diffs.std(ddof=1)

    print(f"\n=== {label} ===")
    print(f"平均差分 = {mean_diff:.3f}")
    print(f"t = {t_stat:.3f}, p = {p_val:.4f}")
    print(f"Cohen’s d = {d:.3f}")

# ================================
# 5. 7条件の分析を一括実行
# ================================

# 1. VC有無（全体）
analyze("VC有無（全体）", wide)

# 2. 男性被験者
analyze("男性被験者", wide[wide["被験者性別"] == "男性"])

# 3. 女性被験者
analyze("女性被験者", wide[wide["被験者性別"] == "女性"])

# 4. 男性被 × 男性実
analyze("男性被 × 男性実", 
        wide[(wide["被験者性別"] == "男性") & (wide["実験者性別"] == "男性")])

# 5. 女性被 × 女性実
analyze("女性被 × 女性実",
        wide[(wide["被験者性別"] == "女性") & (wide["実験者性別"] == "女性")])

# 6. 男性被 × 女性実
analyze("男性被 × 女性実",
        wide[(wide["被験者性別"] == "男性") & (wide["実験者性別"] == "女性")])

# 7. 女性被 × 男性実
analyze("女性被 × 男性実",
        wide[(wide["被験者性別"] == "女性") & (wide["実験者性別"] == "男性")])
