import pandas as pd
import re

# ==============================
# 1. データ読み込み & タイムスタンプ削除
# ==============================
df = pd.read_csv("2025距離測定結果.csv")

# 先頭列（タイムスタンプ）を削除
df = df.drop(columns=[df.columns[0]])

# ==============================
# 2. ロング形式へ変換
# ==============================
long_rows = []

for _, row in df.iterrows():
    name = row["氏名"]
    gender = row["被験者の性別"]
    
    # 氏名・性別以外のすべての列を処理
    for col in df.columns[2:]:
        value = row[col]

        # 値が空ならスキップ
        if pd.isna(value):
            continue
        
        # 列名を解析（例： "なしー男性3"）
        m = re.match(r"(なし|あり)ー(男性|女性)(\d*)", col)
        if m:
            condition = m.group(1)        # なし / あり
            exp_gender = m.group(2)       # 男性 / 女性
            trial_str = m.group(3)        # 試行番号（1〜5）
            trial = int(trial_str) if trial_str else None

            # 距離を整数文字列に変換（366.0 → "366"）
            try:
                value_str = str(int(float(value)))
            except:
                value_str = str(value)

            long_rows.append({
                "氏名": name,
                "被験者性別": gender,
                "条件": condition,
                "実験者性別": exp_gender,
                "試行": trial,
                "距離": value_str
            })

# ==============================
# 3. データフレーム化 & 保存
# ==============================
df_long = pd.DataFrame(long_rows)

# 保存（ファイル名はそのまま指定）
df_long.to_csv("2025距離測定結果_long.csv", index=False)

df_long.head()
