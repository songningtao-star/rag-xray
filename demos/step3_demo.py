# -*- coding: utf-8 -*-
"""
第三步：加上「混合检索（RRF）」

前两步我们有了两种检索方式：
    关键词检索 BM25 —— 比字面，强在"原词"（保质期、儿童、编号）
    向量检索        —— 比意思，强在"同义词"（收到 / 送达）

这一版把它们合起来。合并的方法叫 RRF（Reciprocal Rank Fusion，倒数排名融合）。

跑法：  python step3_demo.py
"""

import math
import jieba
import numpy as np
from fastembed import TextEmbedding

# ============================================================
# 【1】知识库（还是那 4 篇，一个字没改）
# ============================================================
DOCS = {
    "A": "孕妇、哺乳期妇女及18岁以下未成年人请勿服用本产品。",
    "B": "本产品建议每日服用一次，每次一粒，随餐服用。",
    "C": "本产品需置于阴凉干燥处保存，保质期24个月。",
    "D": "本产品由上海保税仓发货，通常3个工作日内送达。",
}
IDS = list(DOCS)

TEST_SET = [
    ("孕妇能吃吗？", "A"),
    ("怎么保存？", "C"),
    ("多久能收到？", "D"),
    ("保质期多久？", "C"),
]


# ============================================================
# 【2】关键词检索 BM25
#     注意这里有一个设计决定：得分 0 的文档，直接不返回。
#     为什么？得分 0 = 一个词都没匹配上 = 完全不相关，
#     让它参与排名只会制造噪音（还记得第一步那个"假命中"吗）。
# ============================================================
tokenized = {k: list(jieba.cut(v)) for k, v in DOCS.items()}
N = len(DOCS)
avgdl = sum(len(t) for t in tokenized.values()) / N
df = {}
for tokens in tokenized.values():
    for w in set(tokens):
        df[w] = df.get(w, 0) + 1
K1, B = 1.5, 0.75


def bm25_ranking(query):
    q_tokens = list(jieba.cut(query))
    scores = {}
    for doc_id, tokens in tokenized.items():
        dl = len(tokens)
        s = 0.0
        for w in q_tokens:
            if w not in df:
                continue
            idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
            tf = tokens.count(w)
            s += idf * (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * dl / avgdl))
        scores[doc_id] = s
    hit = {d: s for d, s in scores.items() if s > 0}   # 丢掉 0 分的
    return sorted(hit.items(), key=lambda kv: kv[1], reverse=True)


# ============================================================
# 【3】向量检索
# ============================================================
print("正在加载向量模型...")
model = TextEmbedding("BAAI/bge-small-zh-v1.5")
doc_vecs = np.array(list(model.embed([DOCS[k] for k in IDS])))
doc_vecs /= np.linalg.norm(doc_vecs, axis=1, keepdims=True)


def vec_ranking(query):
    qv = np.array(list(model.embed([query])))[0]
    qv /= np.linalg.norm(qv)
    sims = doc_vecs @ qv
    return sorted(zip(IDS, sims.tolist()), key=lambda kv: kv[1], reverse=True)


# ============================================================
# 【4】混合检索 RRF
#     三步：
#       ① 两种方法各自给出排名
#       ② 每个排名换算成分数：  1 / (60 + 名次)
#       ③ 同一个文档的分数加起来，重新排序
#
#     60 是个经验常数，业界默认值。作用是让第1名不要太夸张，
#     避免某一种方法"一票独大"。
# ============================================================
RRF_K = 60


def rrf_ranking(query):
    combined = {}
    for ranking in (bm25_ranking(query), vec_ranking(query)):
        for rank, (doc_id, _) in enumerate(ranking, 1):
            combined[doc_id] = combined.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(combined.items(), key=lambda kv: kv[1], reverse=True)


def rank_of(ranking, gold):
    """正确答案在这个排名里排第几？没出现就返回 None（未召回）。"""
    for i, (doc_id, _) in enumerate(ranking, 1):
        if doc_id == gold:
            return i
    return None


# ============================================================
# 开始输出
# ============================================================
print()
print("=" * 78)
print("【1】知识库（4 篇文档，三种方法用的是同一份）")
print("=" * 78)
for k, v in DOCS.items():
    print(f"  {k}: {v}")

print()
print("=" * 78)
print("【2】三种检索方式的排名对比")
print("=" * 78)
print(f"\n  {'问题':<14}{'标准答案':<10}{'关键词BM25':<14}{'向量检索':<14}{'混合RRF'}")
print("  " + "-" * 70)


def fmt(r):
    if r is None:
        return "未召回 ❌"
    return f"第{r}名" + (" ✅" if r == 1 else (" ❌" if r > 3 else ""))


for q, gold in TEST_SET:
    b = rank_of(bm25_ranking(q), gold)
    v = rank_of(vec_ranking(q), gold)
    m = rank_of(rrf_ranking(q), gold)
    print(f"  {q:<14}{gold:<12}{fmt(b):<16}{fmt(v):<16}{fmt(m)}")

print()
print("  说明：BM25 那列里的「未召回」，指的是它一个词都没匹配上，直接没返回这篇。")

# ============================================================
# 手算一遍 RRF，让你看清它到底怎么合的
# ============================================================
DEMO_Q = "保质期多久？"
DEMO_GOLD = "C"

print()
print("=" * 78)
print(f"【3】RRF 到底怎么算的 —— 拿「{DEMO_Q}」手算一遍")
print("=" * 78)

b_list = bm25_ranking(DEMO_Q)
v_list = vec_ranking(DEMO_Q)

print("\n  ① 两种方法各自给出的排名：\n")
print("     关键词 BM25 : " + "   ".join(f"{d}(第{i}名)" for i, (d, _) in enumerate(b_list, 1)))
print("     向量检索    : " + "   ".join(f"{d}(第{i}名)" for i, (d, _) in enumerate(v_list, 1)))

print(f"\n  ② 每个名次换算成分数：  1 / (60 + 名次)\n")
contrib = {}
for i, (d, _) in enumerate(b_list, 1):
    s = 1.0 / (RRF_K + i)
    contrib[d] = contrib.get(d, 0.0) + s
    print(f"     关键词:  {d} 第{i}名 → 1/(60+{i}) = {s:.5f}")
for i, (d, _) in enumerate(v_list, 1):
    s = 1.0 / (RRF_K + i)
    contrib[d] = contrib.get(d, 0.0) + s
    print(f"     向量:    {d} 第{i}名 → 1/(60+{i}) = {s:.5f}")

print("\n  ③ 同一个文档的分数加起来，重新排序：\n")
for d, s in sorted(contrib.items(), key=lambda kv: kv[1], reverse=True):
    is_gold = "   ← 正确答案" if d == DEMO_GOLD else ""
    print(f"     {d}  总分 {s:.5f}   {DOCS[d]}{is_gold}")

# ============================================================
# 结论
# ============================================================
print()
print("=" * 78)
print("【4】结论")
print("=" * 78)
print()
print("  看上面那张对比表，混合检索那一列：每一道题，正确答案都排在第 1 名。")
print()
print("  但要注意一个诚实的发现 —— 在这 4 篇小文档上，「向量检索」单独用")
print("  其实也基本都对。混合并没有比它明显更好，只是「没有变得更差」。")
print()
print("  那为什么真实系统还是要用混合？两个原因：")
print("    1. 现实语料里，向量模型会遇到它没见过的词（品牌名、型号、内部编号），")
print("       这时候只有关键词检索能精确命中。")
print("    2. 你事先不知道哪种方法对哪类问题更有效 —— 混合的作用是")
print("       「自动拿到两者中较好的那个」，抬高下限，而不是追求上限。")
print()
print("  这正是你简历上写的「混合检索 + RRF 融合」在解决的问题。")
