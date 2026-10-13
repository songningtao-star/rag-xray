"""评测指标。"""


def find_answer_chunks(corpus, doc_id, answer):
    """
    找出所有「既来自正确文档、又完整包含答案原文」的片段下标。

    为什么要"完整包含"？因为给模型的是一整段文字。
    如果答案被切成两半、分别躺在两段里，那这两段谁都答不全，
    对模型来说等于没给。这正是「切片切断」这种失败的判定依据。
    """
    return [
        i
        for i, c in enumerate(corpus)
        if c["doc_id"] == doc_id and (not answer or answer in c["text"])
    ]


def rank_of_answer(corpus, ranked_indices, doc_id, answer):
    """
    正确答案（合格的片段）在这次检索里排第几名？
    没进榜（没被召回）就返回 None。
    """
    good = set(find_answer_chunks(corpus, doc_id, answer))
    if not good:
        return None
    for rank, idx in enumerate(ranked_indices, 1):
        if idx in good:
            return rank
    return None
