"""切片：把长文档切成一段一段。"""


def fixed_chunk(text, size, overlap):
    """
    按固定字数切。

    size    : 每段多少字
    overlap : 相邻两段重叠多少字

    重叠的作用：万一关键内容正好卡在切口上，重叠能让它至少完整地
    出现在其中一段里。overlap=0 就是不留余地，最容易出事。
    """
    if size <= 0:
        raise ValueError("size 必须大于 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap 必须 >= 0 且小于 size")

    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append({"text": text[start:end], "start": start, "end": end})
        if end >= len(text):
            break
        start += size - overlap
    return chunks


def build_corpus(docs, size, overlap):
    """
    把所有文档切成片段，拼成一个统一的检索库。

    返回一个列表，每一项：
        chunk_id  唯一编号，形如 "产品说明书#2"（第 3 段）
        doc_id    来自哪篇文档
        text      这段的正文
        start     这段在原文中的起始字符位置（归因时要用它算切片边界）
        end       结束位置
    """
    corpus = []
    for doc_id, text in docs.items():
        for i, c in enumerate(fixed_chunk(text, size, overlap)):
            corpus.append(
                {
                    "chunk_id": f"{doc_id}#{i}",
                    "doc_id": doc_id,
                    "text": c["text"],
                    "start": c["start"],
                    "end": c["end"],
                }
            )
    return corpus
