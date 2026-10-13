"""
归因引擎的单元测试。

这些测试故意不用向量模型（不联网也能跑），只喂手工构造的切片，
把每条归因规则的触发条件钉死。跑法：

    python -m pytest -q
"""

from ragxray.attributor import diagnose
from ragxray.metrics import find_answer_chunks, rank_of_answer

# ---------------------------------------------------------------- 测试数据
# 手动切：把「每次一粒」从中间劈开，模拟 chunk_size 没调好的情况
DOC = "服用方法：每日服用一次，每次一粒，随餐服用。"
ANSWER = "每次一粒"  # 在 DOC 里的位置是第 12~16 字


def _corpus(pieces, doc_id="d"):
    """按 (起始位置, 文本) 直接拼一个切片列表。"""
    return [
        {
            "chunk_id": f"{doc_id}#{i}",
            "doc_id": doc_id,
            "text": text,
            "start": start,
            "end": start + len(text),
        }
        for i, (start, text) in enumerate(pieces)
    ]


def test_cut_corpus_splits_the_answer():
    """前提自检：这份切片里确实没有任何一段完整包含答案。"""
    corpus = _corpus([(0, DOC[:14]), (14, DOC[14:])])
    assert find_answer_chunks(corpus, "d", ANSWER) == []


def test_boundary_cut_is_diagnosed_as_chunk_boundary():
    corpus = _corpus([(0, DOC[:14]), (14, DOC[14:])])
    result = diagnose(corpus, [], "每天吃几粒？", "d", ANSWER, DOC)

    assert result["type"] == "切片切断"
    # 证据里必须给出那条边界的具体位置，否则建议不可信
    assert "第 14 字" in result["evidence"]
    assert "每次" in result["evidence"] and "一粒" in result["evidence"]


def test_missing_answer_is_diagnosed_as_suspect_annotation():
    corpus = _corpus([(0, DOC)])
    result = diagnose(corpus, [], "每天吃几粒？", "d", "文档里没有这句话", DOC)
    assert result["type"] == "标注可疑"


def test_synonym_mismatch_when_no_token_overlap():
    """答案就在这一段里，但问法和原文一个字都对不上 → 该换向量检索。"""
    text = "本店所有商品由上海保税仓发出，默认使用顺丰快递。"
    corpus = _corpus([(0, text), (len(text), "退货政策：支持 7 天无理由退货。")])

    # 正确段（下标 0）没进入召回结果
    result = diagnose(corpus, [(1, 0.4)], "怎么发货", "d", "顺丰快递", text)
    assert result["type"] == "同义词失配"
    assert "vector" in result["suggestion"]


def test_low_rank_when_tokens_overlap_but_ranked_below_top_k():
    """词对得上，只是排得靠后 → 这是 top_k / 精排的问题，不是检索方式的问题。"""
    text = "本店所有商品由上海保税仓发出，默认使用顺丰快递。"
    corpus = _corpus([(0, text), (len(text), "退货政策：支持 7 天无理由退货。")])

    result = diagnose(corpus, [(1, 0.4)], "用什么快递", "d", "顺丰快递", text)
    assert result["type"] == "排序过低"


# ---------------------------------------------------------------- metrics
def test_rank_of_answer_counts_from_one():
    cut = _corpus([(0, DOC[:14]), (14, DOC[14:])])
    corpus = [
        {"chunk_id": "d#0", "doc_id": "d", "text": DOC, "start": 0, "end": len(DOC)},
        {"chunk_id": "d#1", "doc_id": "d", "text": "无关内容", "start": 99, "end": 103},
    ]

    # 正确段（下标 0）排第 1 名 / 被挤到第 2 名
    assert rank_of_answer(corpus, [0, 1], "d", ANSWER) == 1
    assert rank_of_answer(corpus, [1, 0], "d", ANSWER) == 2
    # 完全没被召回
    assert rank_of_answer(corpus, [1], "d", ANSWER) is None
    # 切片切断了 → 没有任何一段完整包含答案
    assert rank_of_answer(cut, [0, 1], "d", ANSWER) is None


def test_find_answer_chunks_ignores_other_documents():
    corpus = _corpus([(0, DOC)])
    other = corpus + [
        {"chunk_id": "x#0", "doc_id": "x", "text": DOC, "start": 0, "end": len(DOC)}
    ]
    # 同样包含答案，但属于另一篇文档，不该被算进来
    assert find_answer_chunks(other, "d", ANSWER) == [0]
