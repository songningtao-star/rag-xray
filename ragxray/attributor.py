"""
归因引擎 —— 本项目的核心。

输入：一道失败的题 + 检索结果 + 知识库
输出：失败类型 + 证据 + 能直接照做的建议

目前能自动诊断 4 类失败：
    1. 切片切断    答案被切在两段里，没有任何一段是完整的
    2. 同义词失配  正确答案的段落里，你问的词一个都没出现
    3. 排序过低    正确段落被召回了，但排在 top_k 之外
    4. 标注可疑    标注里写的答案原文，在文档里根本找不到

关键在于每一项都给「证据」——一个你可以自己去看、去核对的事实，
而不是一句"这里效果不好"。有证据，建议才可信。
"""

import jieba

from .metrics import find_answer_chunks

PUNCT = set("？，。、！：；（）【】 \n\t")


def _tokens(text):
    return {w for w in jieba.cut(text) if w.strip() and w not in PUNCT}


def _short(text, n=34):
    return text if len(text) <= n else text[:n] + "…"


def diagnose(corpus, ranked, query, doc_id, answer, doc_text):
    """
    对一道失败的题做归因。

    参数
        corpus   : build_corpus 的输出
        ranked   : [(片段下标, 分数), ...]，检索的【完整】排名（不截断）
        query    : 用户的问题
        doc_id   : 标准答案所在的文档
        answer   : 标准答案的原文短语（可为 None）
        doc_text : 那篇文档的全文
    返回
        {"type": ..., "evidence": ..., "suggestion": ...}
    """
    ranked_indices = [i for i, _ in ranked]
    complete = find_answer_chunks(corpus, doc_id, answer)

    # ---------- 第 1 类：切片切断 ----------
    # 一个「完整包含答案」的片段都不存在 → 答案被切散了
    if answer and not complete:
        return _diagnose_cut(corpus, doc_text, doc_id, answer)

    # ---------- 有完整的片段，但它没进 top_k ----------
    if complete:
        in_ranked = [i for i in complete if i in ranked_indices]
        if in_ranked:
            best = min(in_ranked, key=lambda i: ranked_indices.index(i))
            rank = ranked_indices.index(best) + 1
        else:
            best, rank = complete[0], None

        chunk_text = corpus[best]["text"]

        # ---------- 第 2 类：同义词失配 ----------
        shared = _tokens(query) & _tokens(chunk_text)
        if not shared:
            return {
                "type": "同义词失配",
                "evidence": (
                    f"答案其实就在这一段里：「{_short(chunk_text)}」\n"
                    f"                 但你问的词 {sorted(_tokens(query))}\n"
                    f"                 和这一段里的词没有一个重合 —— 字面对不上，"
                    f"关键词检索就只能给 0 分。"
                ),
                "suggestion": "把 retriever 换成 vector（向量检索），或用 hybrid（混合检索）",
            }

        # ---------- 第 3 类：排序过低 ----------
        rank_text = f"第 {rank} 名" if rank else "未被召回"
        return {
            "type": "排序过低",
            "evidence": (
                f"正确段落已经被召回了，但排在{rank_text}，掉在 top_k 外面。\n"
                f"                 段落内容：{_short(chunk_text)}"
            ),
            "suggestion": "把 top_k 调大，或在检索后面加一层 Cross-Encoder 精排",
        }

    # ---------- 兜底：整篇文档都没被召回 ----------
    doc_chunks = [i for i, c in enumerate(corpus) if c["doc_id"] == doc_id]
    if doc_chunks:
        ranks = [ranked_indices.index(i) + 1 for i in doc_chunks if i in ranked_indices]
        if ranks:
            evidence = (
                f"答案所在文档「{doc_id}」被切成 {len(doc_chunks)} 段，"
                f"其中最好的一段排第 {min(ranks)} 名 —— 但都不是完整包含答案的那段。"
            )
        else:
            evidence = f"答案所在文档「{doc_id}」被切成 {len(doc_chunks)} 段，一段都没被召回。"
        return {
            "type": "文档未召回",
            "evidence": evidence,
            "suggestion": "换检索方式；或检查这篇文档是不是被切得太碎",
        }

    return {"type": "未知", "evidence": "无法自动判断", "suggestion": "请人工检查这道题"}


def _diagnose_cut(corpus, doc_text, doc_id, answer):
    """切片切断的详细诊断：找出是哪条边界把答案劈开了。"""
    pos = doc_text.find(answer)

    # 标注里写的答案原文，在文档里根本找不到 → 是标注本身的问题
    if pos < 0:
        return {
            "type": "标注可疑",
            "evidence": (
                f"标注里写的答案原文「{answer}」，在文档「{doc_id}」里根本找不到。\n"
                f"                 要么是抄错了，要么这篇文档已经改过。"
            ),
            "suggestion": "检查标注集里这一条的 answer 字段",
        }

    end = pos + len(answer)

    # 看哪条切片边界落在答案内部
    for c in sorted((c for c in corpus if c["doc_id"] == doc_id), key=lambda c: c["start"]):
        if pos < c["start"] < end:
            b = c["start"]
            return {
                "type": "切片切断",
                "evidence": (
                    f"答案原文「{answer}」在第 {pos}~{end} 字，"
                    f"而切片边界正好落在第 {b} 字：\n"
                    f"                 上一段结尾： …{doc_text[max(0, b - 8):b]}"
                    f"   ← 只有前半截「{doc_text[pos:b]}」\n"
                    f"                 下一段开头： {doc_text[b:b + 8]}…"
                    f"   ← 只有后半截「{doc_text[b:end]}」\n"
                    f"                 没有任何一段是完整的，所以模型拿到的话是残的。"
                ),
                "suggestion": "把 chunk_overlap 调大（建议调到 chunk_size 的 1/4 左右）",
            }

    return {
        "type": "切片切断",
        "evidence": f"答案原文「{answer}」没有完整出现在任何一段里（可能被多处边界切碎）。",
        "suggestion": "把 chunk_overlap 调大，或加大 chunk_size",
    }
