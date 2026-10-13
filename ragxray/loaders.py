"""读数据：知识库文档 + 标注集。"""

import json
from pathlib import Path


def load_documents(docs_dir):
    """
    读一个目录下所有 .txt / .md 文件。

    返回 {文档名: 正文}，文档名 = 文件名去掉后缀。
    比如 data/docs/物流说明.txt  →  {"物流说明": "本店所有商品…"}
    """
    docs = {}
    for path in sorted(Path(docs_dir).iterdir()):
        if path.suffix.lower() in (".txt", ".md"):
            docs[path.stem] = path.read_text(encoding="utf-8").strip()
    if not docs:
        raise FileNotFoundError(f"{docs_dir} 里没有找到 .txt 或 .md 文件")
    return docs


def load_queries(path):
    """
    读标注集（每行一条 JSON）：

        {"query": "每天吃几粒？", "doc": "产品说明书", "answer": "每次一粒"}

    - query  : 用户会问的问题
    - doc    : 标准答案在哪篇文档里（必填）
    - answer : 标准答案的原文短语，用来判断"检索到的片段是不是完整的"（建议填）

    空行和以 # 开头的行会被忽略，方便写注释。
    """
    items = []
    text = Path(path).read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} 第 {lineno} 行不是合法的 JSON：{line}") from e
    if not items:
        raise ValueError(f"{path} 里没有读到任何标注")
    return items
