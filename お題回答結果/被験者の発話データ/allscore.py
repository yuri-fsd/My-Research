import csv
import os
import re

BASE_DIR = "/Users/amazo/Documents/修論/実験結果/お題回答結果/被験者の発話データ"
OUT_PATH = os.path.join(BASE_DIR, "情報単位スコア一覧.csv")

TOPIC_FILES = {
    "映画": "映画.csv",
    "趣味": "趣味.csv",
    "好きな食べ物": "好きな食べ物.csv",
    "行ってよかった場所": "行ってよかった場所.csv",
}

participant_info = {
    "杉江鎌": ("男",),
    "﨑谷瑠愛": ("女",),
    "清水拓海": ("男",),
    "伊藤駿之介": ("男",),
    "小島夏美": ("女",),
    "西岡謙介": ("男",),
    "北村佳資": ("男",),
    "日野快人": ("男",),
    "本多快": ("男",),
    "飛澤佑季": ("男",),
    "谷川舞桜": ("女",),
    "久保田翔帆": ("女",),
    "小西真結": ("女",),
    "利光唯": ("女",),
    "横山秀侑": ("男",),
    "池田滉成": ("男",),
    "北野安樹子": ("女",),
    "渡辺怜": ("女",),
    "細見涼乃": ("女",),
    "葭田桃花": ("女",),
}

vc_order_info = {
    "杉江鎌": "なし→あり",
    "﨑谷瑠愛": "なし→あり",
    "清水拓海": "なし→あり",
    "伊藤駿之介": "なし→あり",
    "本多快": "なし→あり",
    "久保田翔帆": "なし→あり",
    "小西真結": "なし→あり",
    "利光唯": "なし→あり",
    "横山秀侑": "なし→あり",
    "北野安樹子": "なし→あり",

    "小島夏美": "あり→なし",
    "西岡謙介": "あり→なし",
    "北村佳資": "あり→なし",
    "日野快人": "あり→なし",
    "飛澤佑季": "あり→なし",
    "谷川舞桜": "あり→なし",
    "池田滉成": "あり→なし",
    "渡辺怜": "あり→なし",
    "細見涼乃": "あり→なし",
    "葭田桃花": "あり→なし",
}

# ★ お題 → 条件対応
topic_condition = {
    "趣味": ("あり", "女性"),
    "映画": ("あり", "男性"),
    "行ってよかった場所": ("なし", "女性"),
    "好きな食べ物": ("なし", "男性"),
}

rows = []

for topic, filename in TOPIC_FILES.items():
    path = os.path.join(BASE_DIR, filename)

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = r["name"]
            text = r["result"]

            m = re.search(r"情報単位数[:：]\s*(\d+)", text)
            score = int(m.group(1)) if m else None

            participant_gender = participant_info.get(name, ("不明",))[0]
            voice_changer, experimenter_gender = topic_condition[topic]
            vc_order = vc_order_info.get(name, "不明")

            rows.append({
                "name": name,
                "topic": topic,
                "info_unit_score": score,
                "participant_gender": participant_gender,
                "experimenter_gender": experimenter_gender,
                "voice_changer": voice_changer,
                "voice_changer_order": vc_order,
            })

with open(OUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "name",
            "topic",
            "info_unit_score",
            "participant_gender",
            "experimenter_gender",
            "voice_changer",
            "voice_changer_order",
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ 条件付きCSVを出力しました → {OUT_PATH}")
