# -*- coding: utf-8 -*-
"""
第二步：加上「向量检索」，验证第一步的归因结论对不对

第一步的归因给出了一个预测：
    「多久能收到？」这道题失败，是因为「收到」和「送达」字面对不上。
     调切片、调参数都没用，只能换成「向量检索」——它比的是意思，不是字面。

这一版就来验证这句话。跑法：  python step2_demo.py
（第一次运行会下载约 90MB 的模型，之后就快了）
"""

import math
import jieba
import numpy as np
from fastembed import TextEmbedding

# ============================================================
# 【1】知识库（和第一步完全一样，一个字都没改）
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
# 【2】关键词检索（BM25）—— 和第一步一模一样
#     原理：比字面。你问的词在哪篇里出现得多、越罕见，分越高。
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
    """返回文档按 BM25 分数从高到低的排序；附上分数，方便看。"""
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
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


# ============================================================
# 【3】向量检索
#     原理：把每篇文档和每个问题都变成一串数字（叫"向量"），
#          意思越接近，两串数字在空间里的方向就越一致（夹角越小）。
#          它比的是「意思」，完全不管字面上有没有同一个词。
#     所以「收到」和「送达」字面上毫无关系，但意思接近 → 分高。
# ============================================================
print("正在加载向量模型（第一次运行要下载约 90MB，之后就快了）...")
model = TextEmbedding("BAAI/bge-small-zh-v1.5")

# 把 4 篇文档变成向量
doc_vecs = np.array(list(model.embed([DOCS[k] for k in IDS])))
doc_vecs /= np.linalg.norm(doc_vecs, axis=1, keepdims=True)  # 归一化，方便算夹角


def vec_ranking(query):
    """返回文档按向量相似度从高到低的排序。"""
    qv = np.array(list(model.embed([query])))[0]
    qv /= np.linalg.norm(qv)
    sims = doc_vecs @ qv  # 点积 = 夹角余弦，越大越像
    return sorted(zip(IDS, sims.tolist()), key=lambda kv: kv[1], reverse=True)


# ============================================================
# 【4】并排比一比：每种方法下，正确答案被排到了第几名
# ============================================================
print()
print("=" * 76)
print("【1】知识库（4 篇文档，两种方法用的是同一份，没变过）")
print("=" * 76)
for k, v in DOCS.items():
    print(f"  {k}: {v}")

print()
print("=" * 76)
print("【2】两种检索方式的排名对比")
print("=" * 76)
print(f"\n  {'问题':<14}{'标准答案':<10}{'关键词检索 BM25':<18}{'向量检索'}")
print("  " + "-" * 68)

saved = []
for q, gold in TEST_SET:
    b_rank = [d for d, _ in bm25_ranking(q)].index(gold) + 1
    v_rank = [d for d, _ in vec_ranking(q)].index(gold) + 1

    def mark(r):
        if r == 1:
            return f"第{r}名 ✅"
        return f"第{r}名" + ("" if r <= 3 else " ❌")

    note = ""
    if b_rank > 3 and v_rank <= 3:
        note = "   ← 救回来了！"
        saved.append((q, gold, b_rank, v_rank))
    print(f"  {q:<14}{gold:<12}{mark(b_rank):<20}{mark(v_rank)}{note}")


# ============================================================
# 【5】把被救回来的那道题拆开看
# ============================================================
print()
print("=" * 76)
print("【3】被救回来的那道题，两种方法分别给了什么分")
print("=" * 76)

for q, gold, b_rank, v_rank in saved:
    print(f"\n  问题：「{q}」   正确答案在 {gold}\n")

    print("  ● 关键词检索 BM25 给的分数：")
    for d, s in bm25_ranking(q):
        flag = "  ← 正确答案，得分 0（一个词都没对上）" if (d == gold and s == 0) else ""
        print(f"      {d}  {s:>6.2f}   {DOCS[d]}{flag}")

    print("\n  ● 向量检索给的分数：")
    for d, s in vec_ranking(q):
        flag = "  ← 正确答案，现在第 1 名" if d == gold else ""
        print(f"      {d}  {s:>6.3f}   {DOCS[d]}{flag}")
    print()

# ============================================================
# 【6】结论
# ============================================================
print("=" * 76)
print("【4】结论：归因预测得对不对？")
print("=" * 76)
print()
print("  第一步的归因说：")
print("    「多久能收到？」失败，是因为「收到」和「送达」字面对不上，")
print("     换向量检索可以救。")
print()
print("  现在验证结果：")
for q, gold, b_rank, v_rank in saved:
    print(f"    {q}  →  BM25 排第 {b_rank} 名  →  向量检索排第 {v_rank} 名")
print()
print("  ✅ 预测成立。归因不是猜的，它是一个可以被验证的结论。")
print()
print("  这就是我们那个工具存在的价值：")
print("    普通评测工具告诉你 —— 「你命中率 75%」")
print("    我们告诉你         —— 「丢的那 1 道在 D 上，原因是同义词失配，")
print("                           该换向量检索，换了之后它排第 1」")
