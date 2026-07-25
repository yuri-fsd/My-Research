# ==========================================
#  12項目 身体化感覚アンケート 信頼性分析
#  手法：Cronbach’s α（Frontiers 2020 と同等）
# ==========================================

import pandas as pd
import numpy as np

# ---------- ① CSV読み込み ----------
path = "2025身体化感覚アンケート（回答） - フォームの回答 1.csv"
df = pd.read_csv(path)
df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip()

# ---------- ② 各質問項目 ----------
cols = [
 '自身のアバターの手足は, 自分自身の手足のように感じた',                  # Q1
 '自身のアバターの手足は, 自分自身の手足のようではないと感じた',            # Q2 (逆転)
 '自身のアバターを「自分の身体」として受け入れていた',                        # Q3
 '自身のアバターを「自分の身体」だと感じることができた',                      # Q4
 '自分の身体はアバターと同じ場所にいるように感じた',                          # Q5
 '自分の身体はアバターとは別の場所にあるように感じた',                        # Q6 (逆転)
 '自分の意識がアバターと一体化しているように感じた',                          # Q7
 '自分はアバターの目から見ているように感じた',                                # Q8
 '自分が「こう動こう」と思ったときに，アバターもそのとおりに動いた →意図と動きの一致',  # Q9
 'アバターの動きを自分の思いどおりに操作できた →全体的な操作感',             # Q10
 'アバターは自分の動きに連動した →リアルタイム同期感',                       # Q11
 'アバターは自分とは関係なく勝手に動いていた →コントロール喪失感'            # Q12 (逆転)
]

# ---------- ③ 数値化と逆転項目処理 ----------
items = df[cols].apply(pd.to_numeric, errors="coerce")
for c in [cols[1], cols[5], cols[11]]:  # Q2, Q6, Q12を反転
    items[c] = 8 - items[c]
X = items.dropna(axis=0).reset_index(drop=True)

# ---------- ④ Cronbach's α 関数 ----------
def cronbach_alpha(df):
    df = df.dropna(axis=0)
    N = df.shape[1]
    variances = df.var(axis=0, ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    return (N / (N - 1)) * (1 - variances.sum() / total_var)

# ---------- ⑤ 各因子ごとに計算 ----------
body_ownership = X[[cols[0], cols[2], cols[3], cols[7]]]   # 身体所有感
self_location  = X[[cols[4], cols[5], cols[6]]]             # 自己位置感
agency         = X[[cols[8], cols[9], cols[10], cols[11]]]  # 行動主体感

alpha_body = cronbach_alpha(body_ownership)
alpha_loc  = cronbach_alpha(self_location)
alpha_ag   = cronbach_alpha(agency)
alpha_total = cronbach_alpha(X)

# ---------- ⑥ 結果表示 ----------
result = pd.DataFrame({
    "因子": ["身体所有感", "自己位置感", "行動主体感", "全体"],
    "Cronbach’s α": [alpha_body, alpha_loc, alpha_ag, alpha_total]
}).round(3)

print("=== Cronbach’s α（信頼性分析結果）===")
print(result)
