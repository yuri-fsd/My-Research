import pandas as pd
import json
import os
import re

# ======== 設定 ========
BASE_DIR = "/Users/amazo/Documents/修論/結果分析/被験者の発話データ"
TXT_FILES = {
    "映画": os.path.join(BASE_DIR, "映画.txt"),
    "趣味": os.path.join(BASE_DIR, "趣味.txt"),
    "行ってよかった場所": os.path.join(BASE_DIR, "行ってよかった場所.txt"),
    "好きな食べ物": os.path.join(BASE_DIR, "好きな食べ物.txt"),
}
OUT_TOPIC_PATH = os.path.join(BASE_DIR, "self_disclosure_by_topic.csv")
OUT_TOTAL_PATH = os.path.join(BASE_DIR, "self_disclosure_by_person.csv")

# ======== 自己開示判定キーワード ========
KEYWORDS = [
    "私", "僕", "自分", "俺", "思う", "感じる", "～たい", "たい", "好き", "苦手", 
    "行った", "見た", "経験", "印象", "感動", "欲しい", "うれしい", "楽しい", "悲しい", "怖い", "面白い"
]

# ======== 被験者・条件情報（20人×4条件） ========
info = []
base_data = [
    ("杉江鎌", "男", "なし→あり"),
    ("﨑谷瑠愛", "女", "なし→あり"),
    ("清水拓海", "男", "なし→あり"),
    ("伊藤駿之介", "男", "なし→あり"),
    ("小島夏美", "女", "あり→なし"),
    ("西岡謙介", "男", "あり→なし"),
    ("北村佳資", "男", "あり→なし"),
    ("日野快人", "男", "あり→なし"),
    ("本多快", "男", "なし→あり"),
    ("飛澤佑季", "男", "あり→なし"),
    ("谷川舞桜", "女", "あり→なし"),
    ("久保田翔帆", "女", "なし→あり"),
    ("小西真結", "女", "なし→あり"),
    ("利光唯", "女", "なし→あり"),
    ("横山秀侑", "男", "なし→あり"),
    ("池田滉成", "男", "あり→なし"),
    ("北野安樹子", "女", "なし→あり"),
    ("渡辺怜", "女", "あり→なし"),
    ("細見涼乃", "女", "あり→なし"),
    ("葭田桃花", "女", "あり→なし"),
]

# 各人に4条件展開（あり女性, あり男性, なし女性, なし男性）
topics = {
    ("あり", "女性"): "趣味",
    ("あり", "男性"): "映画",
    ("なし", "女性"): "行ってよかった場所",
    ("なし", "男性"): "好きな食べ物"
}

for name, gender, order in base_data:
    for vc_condition, exp_gender in [("あり", "女性"), ("あり", "男性"), ("なし", "女性"), ("なし", "男性")]:
        topic = topics[(vc_condition, exp_gender)]
        info.append([name, gender, order, vc_condition, exp_gender, topic])

participant_df = pd.DataFrame(info, columns=["name", "gender", "vc_order", "vc_condition", "experimenter_gender", "topic"])

# ======== 自己開示単位のカウント関数（精密版） ========
def count_disclosure_units(text):
    sentences = re.split(r"[。．！？!?\n]", text)
    count = 0
    for s in sentences:
        for k in KEYWORDS:
            count += s.count(k)
    return count

# ======== txtファイルの読み込み ========
def load_text_data(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        data = json.loads(text)
        return [tuple(item.split(" ", 1)) for item in data]
    except Exception as e:
        print(f"⚠️ JSON読み込みエラー: {path} -> {e}")
        return []

# ======== 各トピックの集計 ========
records = []
missing = []

for topic_name, path in TXT_FILES.items():
    items = load_text_data(path)
    if not items:
        print(f"⚠️ {topic_name} にデータが読み込めませんでした")
        continue
    for name, content in items:
        c90 = count_disclosure_units(content[:len(content)//2])
        c60 = count_disclosure_units(content[len(content)//2:])
        records.append([name, topic_name, c90, c60, c90 + c60])

topic_df = pd.DataFrame(records, columns=["name", "topic", "count_90s", "count_60s", "count_total"])

# ======== 条件情報と結合 ========
merged_df = pd.merge(topic_df, participant_df, on=["name", "topic"], how="left")

# 欠損確認
missing_names = merged_df[merged_df["gender"].isna()]["name"].unique()
if len(missing_names) > 0:
    print("⚠️ 条件情報が見つからなかった被験者:", list(missing_names))
else:
    print("✅ 全被験者20名分の条件情報を統合")

# ======== 1. トピックごとのスコア ========
merged_df.to_csv(OUT_TOPIC_PATH, index=False, encoding="utf-8-sig")

# ======== 2. 被験者ごとの合計スコア ========
total_df = (
    merged_df.groupby(["name", "gender", "vc_order"])
    [["count_90s", "count_60s", "count_total"]]
    .sum()
    .reset_index()
)
total_df.rename(columns={"count_90s": "total_90s", "count_60s": "total_60s", "count_total": "total_all"}, inplace=True)
total_df.to_csv(OUT_TOTAL_PATH, index=False, encoding="utf-8-sig")

# ======== お題ごとの平均 ========
topic_means = (
    merged_df.groupby("topic")[["count_90s", "count_60s", "count_total"]]
    .mean().round(2)
)
topic_means.to_csv(os.path.join(BASE_DIR, "self_disclosure_topic_means.csv"))

print("✅ 4トピック×条件統合分析 完了！")
print("📁 トピック別CSV:", OUT_TOPIC_PATH)
print("📁 被験者別CSV:", OUT_TOTAL_PATH)
print(topic_means)
