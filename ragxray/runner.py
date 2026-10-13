"""主流程：读数据 → 切片 → 建索引 → 逐题评测 → 对失败的题做归因 → 汇总。"""

from collections import Counter

from .attributor import diagnose
from .chunkers import build_corpus
from .loaders import load_documents, load_queries
from .metrics import rank_of_answer
from .retrievers import build_retriever


def run(config):
    """跑一次完整的评测。config 是 config.yaml 读出来的字典。"""
    docs = load_documents(config["docs_dir"])
    queries = load_queries(config["queries"])

    chunk_size = int(config.get("chunk_size", 200))
    chunk_overlap = int(config.get("chunk_overlap", 0))
    top_k = int(config.get("top_k", 3))
    retriever_name = config.get("retriever", "hybrid")

    corpus = build_corpus(docs, chunk_size, chunk_overlap)
    retriever = build_retriever(retriever_name).fit(corpus)

    results = []
    for item in queries:
        doc_id = item["doc"]
        if doc_id not in docs:
            raise KeyError(
                f"标注集里写的是文档「{doc_id}」，但知识库里没有这篇。"
                f"现有文档：{'、'.join(docs)}"
            )

        ranked = retriever.search_all(item["query"])
        ranked_indices = [i for i, _ in ranked]
        answer = item.get("answer")
        rank = rank_of_answer(corpus, ranked_indices, doc_id, answer)
        hit = rank is not None and rank <= top_k

        diagnosis = None
        if not hit:
            diagnosis = diagnose(
                corpus, ranked, item["query"], doc_id, answer, docs[doc_id]
            )

        results.append(
            {
                "query": item["query"],
                "doc": doc_id,
                "answer": answer,
                "rank": rank,
                "hit": hit,
                "diagnosis": diagnosis,
            }
        )

    return {
        "n_docs": len(docs),
        "n_chunks": len(corpus),
        "retriever": retriever_name,
        "top_k": top_k,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "results": results,
    }


def format_report(r):
    """把 run() 的结果排版成能直接看的文字报告。"""
    hits = sum(1 for x in r["results"] if x["hit"])
    total = len(r["results"])

    out = []
    out.append("=" * 78)
    out.append("  ragxray 检索评测报告")
    out.append("=" * 78)
    out.append(
        f"  知识库     {r['n_docs']} 篇文档 → 切成 {r['n_chunks']} 段"
        f"（每段 {r['chunk_size']} 字，重叠 {r['chunk_overlap']} 字）"
    )
    out.append(f"  检索方式   {r['retriever']}")
    out.append(f"  top_k      {r['top_k']}")
    out.append("-" * 78)
    out.append(f"  命中率     {hits}/{total} = {hits / total:.0%}")
    out.append("-" * 78)
    out.append("")
    out.append("  逐题结果")
    out.append("")

    for x in r["results"]:
        mark = "✅" if x["hit"] else "❌"
        rank = f"第 {x['rank']} 名" if x["rank"] else "未召回"
        out.append(f"    {mark}  {x['query']:<20} {rank}")
        d = x["diagnosis"]
        if d:
            out.append(f"         ├ 失败类型  {d['type']}")
            out.append(f"         ├ 证据      {d['evidence']}")
            out.append(f"         └ 建议      {d['suggestion']}")
            out.append("")

    # 归因汇总
    diags = [x["diagnosis"]["type"] for x in r["results"] if x["diagnosis"]]
    if diags:
        out.append("-" * 78)
        out.append("  失败原因汇总")
        out.append("-" * 78)
        for name, cnt in Counter(diags).most_common():
            out.append(f"    {name:<10} {cnt} 条")
        out.append("")
        out.append("  ↑ 这个汇总就是「下一步该改什么」的答案：")
        out.append("    哪一类占得最多，就先解决哪一类，不用一个个试。")
    else:
        out.append("-" * 78)
        out.append("  全部命中，没有需要归因的失败。")

    return "\n".join(out)
