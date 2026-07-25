from openai import OpenAI
import json, csv, os, time, re, statistics as stats

# ======== 設定 ========
BASE_DIR = "/Users/amazo/Documents/修論/実験結果/お題回答結果/被験者の発話データ"
PROMPT_PATH = os.path.join(BASE_DIR, "prompt_ja.txt")
TOPICS = ["映画", "趣味", "好きな食べ物", "行ってよかった場所"]

# ======== OpenAIクライアント初期化 ========
client = OpenAI(
    api_key="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"
)

# ======== プロンプト読み込み ========
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    prompt_template = f.read()

# ======== 各お題ごとに分析 ========
for topic in TOPICS:
    TXT_PATH = os.path.join(BASE_DIR, f"{topic}.txt")
    OUT_PATH = os.path.join(BASE_DIR, f"{topic}.csv")

    if not os.path.exists(TXT_PATH):
        print(f"⚠️ {TXT_PATH} が見つかりません。スキップします。")
        continue

    print(f"\n🎬 分析対象: {topic}.txt")

    with open(TXT_PATH, "r", encoding="utf-8") as f:
        data = json.loads(f.read())

    results = []


# ======== 導入文削除関数 ========
    def clean_intro_lines(text):
        """
        発話文の冒頭にある導入文（例：「好きな映画は」「趣味は」「行ってよかった場所は」など）を削除する関数。
        導入文の後に続く自己開示的な内容は残す。
        """
        patterns = [
            r"^.*好きな映画は[、。．!.！\s]*",
            r"^.*好きな食べ物は[、。．!.！\s]*",
            r"^.*趣味は[、。．!.！\s]*",
            r"^.*行ってよかった場所は[、。．!.！\s]*",
            r"^.*(について話します|の話をします)[、。．!.！\s]*",
            r"^.*(特にないです|わかりません|難しいです)[、。．!.！\s]*",
        ]
        for p in patterns:
            text = re.sub(p, "", text, flags=re.MULTILINE)
        return text.strip()

    for i, entry in enumerate(data, 1):
        try:
            if isinstance(entry, list) and len(entry) == 2:
                name, utterance = entry
            elif isinstance(entry, dict):
                name = entry.get("name", "不明")
                utterance = entry.get("utterance", "")
            else:
                try:
                    name, utterance = entry.split(" ", 1)
                except:
                    name, utterance = "不明", str(entry)

            print(f"🧩 {i}/{len(data)}件目: {name} 分析中...")

            # ======== プロンプト作成 ========
            prompt = prompt_template.replace("＜分析対象の内容＞", utterance.strip())

            # ======== API呼び出し ========
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            content = response.choices[0].message.content.strip()
            results.append({"name": name, "result": content})
            time.sleep(1)

        except Exception as e:
            print(f"⚠️ {name} の分析中にエラー発生: {e}")
            results.append({"name": name, "result": f"エラー: {e}"})
            continue

    with open(OUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "result"])
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ {topic} の分析完了！ 出力ファイル: {OUT_PATH}")


# ======== 📊 集計 ========
summary_counts = {}
topic_counts = {t: [] for t in TOPICS}

print("\n📊 各被験者ごとの情報単位数まとめ")

for topic in TOPICS:
    csv_path = os.path.join(BASE_DIR, f"{topic}.csv")
    if not os.path.exists(csv_path):
        continue

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            result = row["result"]
            match = re.search(r"情報単位数[:：]\s*(\d+)", result)
            count = int(match.group(1)) if match else 0

            if name not in summary_counts:
                summary_counts[name] = {t: 0 for t in TOPICS}
                summary_counts[name]["合計"] = 0
            summary_counts[name][topic] += count
            summary_counts[name]["合計"] += count
            topic_counts[topic].append(count)

# ======== 各被験者の出力 ========
for name, counts in summary_counts.items():
    topic_text = "，".join([f"{t}:{counts[t]}" for t in TOPICS])
    print(f"👤 {name} → {topic_text}，合計:{counts['合計']}")

# ======== 各お題の平均 ========
print("\n📈 各お題の平均値")
for topic, vals in topic_counts.items():
    if vals:
        avg = round(stats.mean(vals), 2)
        print(f"{topic}: {avg}")
