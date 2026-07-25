from pathlib import Path
import re

# ===== 設定 =====
BASE_DIR = Path("/Users/amazo/Documents/修論/結果分析/被験者の発話データ")
FILES = [
    "映画.txt",
    "趣味.txt",
    "行ってよかった場所.txt",
    "好きな食べ物.txt"
]

SPEAKERS = [
    "杉江鎌","﨑谷瑠愛","清水拓海","伊藤駿之介","小島夏美","西岡謙介",
    "北村佳資","日野快人","本多快","飛澤佑季","谷川舞桜","久保田翔帆",
    "小西真結","利光唯","利光唯","横山秀侑","池田滉成",
    "北野安樹子","渡辺怜","細見涼乃","葭田桃花"
]

# ===== 整形関数 =====
def clean_text_lines(lines):
    cleaned = []
    current_speaker = None
    current_block = []

    def normalize_name(name):
        return re.sub(r"[ 　：:\-・]", "", name)

    normalized_speakers = [normalize_name(n) for n in SPEAKERS]

    for line in lines:
        line = line.strip()
        if not line:
            continue
        norm_line = normalize_name(line)

        if any(norm_line == s for s in normalized_speakers):
            if current_speaker and current_block:
                cleaned.append(f"{current_speaker} {' '.join(current_block)}")
                current_block = []
            current_speaker = line
        else:
            current_block.append(line)

    if current_speaker and current_block:
        cleaned.append(f"{current_speaker} {' '.join(current_block)}")

    return cleaned


# ===== メイン処理 =====
def run_all():
    print("==== 4ファイル整形開始 ====\n")

    for filename in FILES:
        path = BASE_DIR / filename
        if not path.exists():
            print(f"❌ ファイルが見つかりません: {path}")
            continue

        text = path.read_text(encoding="utf-8")

        # すでに整形済み（JSON風）ならスキップ
        if text.strip().startswith("[") and text.strip().endswith("]"):
            print(f"⏩ {filename} はすでに整形済みのためスキップ")
            continue

        lines = text.splitlines()
        cleaned = clean_text_lines(lines)

        # JSON風フォーマット
        formatted = "[\n" + ",\n".join(f'  "{c}"' for c in cleaned) + "\n]\n"

        # 話者数チェック（20人いれば「整列済み」と判定）
        detected_speakers = sum(1 for s in SPEAKERS if any(s in c for c in cleaned))
        if detected_speakers == 20:
            path.write_text(formatted, encoding="utf-8")
            print(f"✅ 整形＆上書き完了 → {filename}（{len(cleaned)}件）")
        else:
            print(f"⚠ {filename} は話者が {detected_speakers} 人しか検出されませんでした（上書きしません）")

    print("\n==== 完了 ====")


# ===== 実行 =====
if __name__ == "__main__":
    run_all()
