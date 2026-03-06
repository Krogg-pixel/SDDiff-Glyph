import json
from collections import Counter

# ========= 1. 配置 =========
annotation_path = "data/train/annotations_400.jsonl"   # 你的真实文件
semantic_key = "elements"                     # ← 关键修正点
top_k = 20

# ========= 2. 语义归一化规则 =========
def normalize_term(term):
    term = term.lower().strip()

    merge_map = {
        "bird": "bird(s)",
        "birds": "bird(s)",
        "flower": "flower",
        "flowers": "flower",
    }

    return merge_map.get(term, term)

# ========= 3. 逐行读取 JSONL 并统计 =========
counter = Counter()
total_count = 0

with open(annotation_path, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[Warning] Line {line_num} JSON decode failed: {e}")
            continue

        if semantic_key not in item:
            continue

        for term in item[semantic_key]:
            norm_term = normalize_term(term)
            counter[norm_term] += 1
            total_count += 1

# ========= 4. 输出 Top-K（含百分比） =========
print(f"\nTop-{top_k} most frequent semantic elements:\n")
print(f"{'Semantic Element':25s} {'Count':>8s} {'Percentage (%)':>15s}")
print("-" * 55)

for term, count in counter.most_common(top_k):
    percentage = count / total_count * 100 if total_count > 0 else 0
    print(f"{term:25s} {count:8d} {percentage:14.2f}")

print("\nTotal semantic tokens counted:", total_count)
