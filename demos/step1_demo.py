# -*- coding: utf-8 -*-
"""
第一步：最小可跑的「检索 → 评测 → 归因」

目的：只用 4 篇很短的文档、3 道题，把 RAG 评测这件事完整走一遍。
      每一步都打印出来，看不懂的地方可以直接对照输出。

跑法：  python step1_demo.py
"""

import math
import jieba

# ============================================================
# 【1】知识库：4 篇文档
#     故意写得很短 —— 每篇不到 30 个字，正好一整块。
#     真实项目里文档很长，一篇会切成几十块（chunk），
#     这里先跳过「切片」，免得干扰你理解后面两步。
# ============================================================
DOCS = {
    "A": "孕妇、哺乳期妇女及18岁以下未成年人请勿服用本产品。",
    "B": "本产品建议每日服用一次，每次一粒，随餐服用。",
    "C": "本产品需置于阴凉干燥处保存，保质期24个月。",
    "D": "本产品由上海保税仓发货，通常3个工作日内送达。",
}

# 中文没有空格，机器看不懂句子，先要把每篇拆成一个个「词」。
# 这个动作叫「分词」，用 jieba 库。
tokenized = {k: list(jieba.cut(v)) for k, v in DOCS.items()}

print("=" * 74)
print("【1】知识库（4 篇文档）")
print("=" * 74)
for k, v in DOCS.items():
    print(f"  {k}: {v}")

print()
print("  切好词之后，每篇文档变成这样（机器眼里的样子）：")
for k, tokens in tokenized.items():
    print(f"  {k}: {' | '.join(tokens)}")


# ============================================================
# 【2】检索：给一个问题，算出哪篇文档最相关
#     打分方法叫 BM25 —— 你问的词，在哪篇里出现得越多、越罕见，
#     那篇的分数就越高。然后从高到低排个序，取前几名。
#     检索的本质就是「打分 + 排序」，没有别的魔法。
# ============================================================
N = len(DOCS)
avgdl = sum(len(t) for t in tokenized.values()) / N

# df[词] = 这个词出现在几篇文档里。越罕见的词，区分度越高，权重越大。
df = {}
for tokens in tokenized.values():
    for w in set(tokens):
        df[w] = df.get(w, 0) + 1

K1, B = 1.5, 0.75  # BM25 的两个经验参数，业界默认值，不用管


def search(query, top_k=3):
    """输入一个问题，返回最相关的 top_k 篇文档，带分数。"""
    q_tokens = list(jieba.cut(query))
    scores = {}
    for doc_id, tokens in tokenized.items():
        dl = len(tokens)
        s = 0.0
        for w in q_tokens:
            if w not in df:
                continue  # 这个词谁都没有，跳过
            idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
            tf = tokens.count(w)
            s += idf * (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * dl / avgdl))
        scores[doc_id] = s
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]


print()
print("=" * 74)
print("【2】检索：问一个问题，看它挑出哪篇")
print("=" * 74)
demo_q = "小孩能吃吗？"
print(f"\n  问题：「{demo_q}」")
print(f"  问题分词： {list(jieba.cut(demo_q))}")
print()
for rank, (doc_id, score) in enumerate(search(demo_q), 1):
    print(f"    第{rank}名  {doc_id}  得分 {score:.2f}   {DOCS[doc_id]}")
print()
print("  ↑ 上面就是检索返回的排序：得分高 = 匹配上的词多；得分 0 = 一个词都没对上。")
print("    检索干的全部事情，就是打分 + 排序。")


# ============================================================
# 【3】评测：拿 3 道「有标准答案」的题来考试
#     没有标准答案就没法判分 —— 这就是为什么需要「标注集」。
# ============================================================
TEST_SET = [
    ("孕妇能吃吗？", "A"),   # 标准答案在 A
    ("怎么保存？", "C"),     # 标准答案在 C
    ("多久能收到？", "D"),   # 标准答案在 D·
    ("保质期多久？", "C"),   # 标准答案在 C

]

print()
print("=" * 74)
print("【3】评测：拿 3 道有标准答案的题考它")
print("=" * 74)
print()

right = 0
wrong = []
for q, gold in TEST_SET:
    ranked = search(q)
    got = [d for d, _ in ranked]
    ok = gold in got
    if ok:
        right += 1
        print(f"  ✅  问题：{q:<12} 标准答案 {gold}   →  找到了（它返回了 {got}）")
    else:
        wrong.append((q, gold, got))
        print(f"  ❌  问题：{q:<12} 标准答案 {gold}   →  没找到（它返回了 {got}）")

print()
print(f"  得分： {right}/{len(TEST_SET)} = {right / len(TEST_SET):.0%}")
print("      这就是「命中率 Recall@k」，所有 RAG 评测工具报的第一个数字。")


# ============================================================
# 【4】归因：那道错题，到底为什么错
#     评测给数字，归因给原因 —— 这就是我们要做的工具的核心。
# ============================================================
PUNCT = set("？，。、！：；（） ")

print()
print("=" * 74)
print("【4】归因：把那道错题拆开看")
print("=" * 74)

for q, gold, got in wrong:
    q_tokens = list(jieba.cut(q))
    gold_tokens = tokenized[gold]
    overlap = (set(q_tokens) & set(gold_tokens)) - PUNCT

    print(f"\n  问题：「{q}」   标准答案在 {gold}")
    print(f"  但检索返回的是：{got}（一个都没对）")
    print()
    print(f"  把问题拆成词：     {q_tokens}")
    print(f"  把 {gold} 拆成词：      {gold_tokens}")
    print()
    if overlap:
        print(f"  两边重合的词：     {overlap}")
    else:
        print("  两边重合的词：     一个都没有！")
        print()
        print("  为什么？看这里：")
        print(f"    问题里说的是「收到」，{gold} 里写的是「送达」")
        print(f"    问题里说的是「多久」，{gold} 里写的是「通常3个工作日」")
        print("    意思是一回事，但字完全不同 —— 关键词检索是「比字面」的，")
        print(f"    字对不上就是 0 分，所以 {gold} 直接被判为「完全不相关」。")
        print()
        print("  → 结论：这是「同义词对不上」导致的失败。")
        print("     调切片大小、调 BM25 参数都没用，")
        print("     只能换成「向量检索」—— 它比的是意思，不是字面。")
