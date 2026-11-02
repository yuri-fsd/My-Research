from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ttest_ind
from statsmodels.formula.api import ols

# ===== データ読み込み =====
base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "情報単位スコア一覧.csv"

df = pd.read_csv(csv_path)

# 列名整理
# 既存の CSV の列名に合わせて、分析用に簡潔な名前へ変換する
# 例: participant_gender -> p_gender, experimenter_gender -> e_gender
# 例: voice_changer -> voice, info_unit_score -> score

df = df.rename(columns={
    "name": "participant",
    "participant_gender": "p_gender",
    "experimenter_gender": "e_gender",
    "voice_changer": "voice",
    "info_unit_score": "score",
})

# 文字列の前後空白を除去し、性別の表記ゆれを吸収する
# 例: 「男性」/「男」, 「女性」/「女」などを統一
for col in ["p_gender", "e_gender"]:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].replace({"男性": "男", "女性": "女", "male": "男", "female": "女"})

# voice 条件の変換
# あり -> vc, なし -> no_vc
# 文字列として扱うことで、比較時にエラーを減らす

df["voice"] = df["voice"].astype(str).str.strip().replace({"あり": "vc", "なし": "no_vc"})

df["score"] = pd.to_numeric(df["score"], errors="coerce")

df = df.dropna(subset=["score", "p_gender", "e_gender", "voice"]).copy()

# ===== 条件定義 =====
# 変数が「男/女」のどちらかである前提で、条件ごとの差分を見やすくする
conditions = {
    "全体": df,
    "男性被験者": df[df["p_gender"] == "男"],
    "女性被験者": df[df["p_gender"] == "女"],
    "男性×男性": df[(df["p_gender"] == "男") & (df["e_gender"] == "男")],
    "女性×女性": df[(df["p_gender"] == "女") & (df["e_gender"] == "女")],
    "男性×女性": df[(df["p_gender"] == "男") & (df["e_gender"] == "女")],
    "女性×男性": df[(df["p_gender"] == "女") & (df["e_gender"] == "男")],
}

# ===== 分析 =====
for label, sub in conditions.items():
    print("\n==============================")
    print(f"条件: {label}")
    print("==============================")

    if sub.empty:
        print("対象データがありません。スキップ")
        continue

    if sub["voice"].nunique() < 2:
        print("VC条件が不足しているためスキップ")
        continue

    vc = sub[sub["voice"] == "vc"]["score"]
    no_vc = sub[sub["voice"] == "no_vc"]["score"]

    if len(vc) < 2 or len(no_vc) < 2:
        print("データ数不足のためスキップ")
        continue

    # --- ANOVA ---
    model = ols("score ~ C(voice)", data=sub).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print("\n【ANOVA】")
    print(anova_table)

    # --- t検定（Welch） ---
    t, p = ttest_ind(vc, no_vc, equal_var=False)

    print("\n【t検定】")
    print(f"t = {t:.3f}, p = {p:.3f}")

    # --- Cohen's d ---
    if len(vc) > 1 and len(no_vc) > 1:
        pooled_std = np.sqrt((vc.var(ddof=1) + no_vc.var(ddof=1)) / 2)
        d = (vc.mean() - no_vc.mean()) / pooled_std if pooled_std != 0 else np.nan
    else:
        d = np.nan

    print("\n【Cohen's d】")
    print(f"d = {d:.3f}")

    # --- 平均・SD ---
    print("\n【平均・標準偏差】")
    summary = sub.groupby("voice")["score"].agg(["mean", "std", "count"])
    print(summary)
