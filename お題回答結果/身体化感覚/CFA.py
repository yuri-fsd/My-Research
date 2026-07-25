# ==========================================
#  12項目 身体化感覚アンケート 確証的因子分析 (CFA)
#  手法：Frontiers 2020 と同等 (Gonzalez-Franco & Peck)
# ==========================================

import pandas as pd
from semopy import Model, semplot

# ---------- ① データ読み込み ----------
path = "2025身体化感覚アンケート（回答） - フォームの回答 1.csv"
df = pd.read_csv(path)
df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip()

# ---------- ② 各質問項目 ----------
cols = [
 '自身のアバターの手足は, 自分自身の手足のように感じた',                  
 '自身のアバターの手足は, 自分自身の手足のようではないと感じた',  # 逆転
 '自身のアバターを「自分の身体」として受け入れていた',                    
 '自身のアバターを「自分の身体」だと感じることができた',                  
 '自分の身体はアバターと同じ場所にいるように感じた',                    
 '自分の身体はアバターとは別の場所にあるように感じた',  # 逆転
 '自分の意識がアバターと一体化しているように感じた',                    
 '自分はアバターの目から見ているように感じた',                          
 '自分が「こう動こう」と思ったときに，アバターもそのとおりに動いた →意図と動きの一致',  
 'アバターの動きを自分の思いどおりに操作できた →全体的な操作感',       
 'アバターは自分の動きに連動した →リアルタイム同期感',                 
 'アバターは自分とは関係なく勝手に動いていた →コントロール喪失感'      # 逆転
]

# ---------- ③ データ整形 ----------
items = df[cols].apply(pd.to_numeric, errors="coerce")

# 逆転項目(Q2, Q6, Q12)を反転 (1〜7 → 8−x)
for c in [cols[1], cols[5], cols[11]]:
    items[c] = 8 - items[c]

# 欠損除去
X = items.dropna(axis=0).reset_index(drop=True)

# ---------- ④ モデル定義（3因子モデル） ----------
desc = """
Body_Ownership =~ Q1 + Q3 + Q4 + Q8
Self_Location  =~ Q5 + Q6 + Q7
Agency         =~ Q9 + Q10 + Q11 + Q12
"""

# 列名をQ1〜Q12に置き換え
X.columns = [f"Q{i+1}" for i in range(12)]

# ---------- ⑤ CFA実行 ----------
model = Model(desc)
res = model.fit(X)

# ---------- ⑥ 結果表示 ----------
print("\n=== モデル適合度指標 ===")
from semopy import calc_stats

stats_df = calc_stats(model)  # ← これだけでOK（引数は model だけ）
# 欲しい指標だけ表示
for k in ["CFI", "TLI", "RMSEA", "AIC", "BIC", "chi2", "chi2 p-value", "DoF"]:
    if k in stats_df.columns:
        print(f"{k}: {stats_df.loc['Value', k]:.3f}")

print("\n=== 各パスの因子負荷量 ===")
estimates = model.inspect()
print(estimates[["lval", "op", "rval", "Estimate"]])

# ---------- ⑦ パス図を出力（任意） ----------
semplot(model, "CFA_embodiment_model.png")
