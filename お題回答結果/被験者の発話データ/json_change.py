import os
import json
import re

# ======== 設定 ========
BASE_DIR = "/Users/amazo/Documents/修論/実験結果/お題回答結果/被験者の発話データ"
INPUT_FILE = os.path.join(BASE_DIR, "映画_文字起こし.txt")  # 元のファイル
OUTPUT_FILE = os.path.join(BASE_DIR, "映画.txt")  # JSON保存先

# ======== 処理 ========
def convert_transcript_to_json(path_in, path_out):
    with open(path_in, "r", encoding="utf-8") as f:
        text = f.read()

    # 改行やタブを統一
    text = re.sub(r"\r", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)  # 連続改行を1つに
    text = text.strip()

    # 名前の候補リスト（ファイル順に処理）
    NAMES = [
        "杉江鎌","﨑谷瑠愛","清水拓海","伊藤駿之介","小島夏美","西岡謙介",
        "北村佳資","日野快人","本多快","飛澤佑季","谷川舞桜","久保田翔帆",
        "小西真結","利光唯","横山秀侑","池田滉成","北野安樹子","渡辺怜",
        "細見涼乃","葭田桃花"
    ]

    # 名前の位置を検索
    positions = []
    for name in NAMES:
        match = re.search(name, text)
        if match:
            positions.append((match.start(), name))

    positions.sort()

    # 各話者ごとにテキストを切り出し
    data = []
    for i, (pos, name) in enumerate(positions):
        start = pos + len(name)
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        content = text[start:end].strip()
        data.append([name, content])

    # JSON保存
    with open(path_out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ {os.path.basename(path_out)} をJSON形式に変換しました（{len(data)}件）")
    print("人数:", len(data))
    print("抜け:", [n for n in NAMES if n not in [x[0] for x in data]])

# ======== 実行 ========
convert_transcript_to_json(INPUT_FILE, OUTPUT_FILE)

