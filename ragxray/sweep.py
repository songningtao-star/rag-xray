"""
参数扫描：同一份知识库、同一套标注，只换实验条件反复跑，看哪组条件最好。

为什么需要它：单个数字（"命中率 80%"）说明不了任何事。真正有用的是
"换了什么条件，命中率从多少变成多少" —— 这才是可以写进简历、可以
在技术评审上讲的结论。
"""

from collections import Counter

from .runner import run

DEFAULT_RETRIEVERS = ("bm25", "vector", "hybrid")
DEFAULT_OVERLAPS = (0, 5, 10, 15, 20)


def sweep(config, retrievers=DEFAULT_RETRIEVERS, overlaps=DEFAULT_OVERLAPS):
    """跑一整张网格：每行一个检索方式，每列一个重叠字数。"""
    rows = []
    for name in retrievers:
        for overlap in overlaps:
            cfg = dict(config)
            cfg["retriever"] = name
            cfg["chunk_overlap"] = overlap

            result = run(cfg)
            results = result["results"]
            hits = sum(1 for x in results if x["hit"])
            reasons = Counter(
                x["diagnosis"]["type"] for x in results if x["diagnosis"]
            )

            rows.append(
                {
                    "retriever": name,
                    "overlap": overlap,
                    "hits": hits,
                    "total": len(results),
                    "rate": hits / len(results) if results else 0.0,
                    "reasons": reasons,
                    "result": result,
                }
            )
    return rows


def format_sweep(rows):
    """把网格排版成终端表格。"""
    if not rows:
        return "（没有扫描结果）"

    retrievers = list(dict.fromkeys(r["retriever"] for r in rows))
    overlaps = list(dict.fromkeys(r["overlap"] for r in rows))
    by_key = {(r["retriever"], r["overlap"]): r for r in rows}

    out = []
    out.append("=" * 78)
    out.append("  参数扫描：检索方式 × 切片重叠")
    out.append("=" * 78)
    out.append("")
    out.append(f"  {'检索方式':<10}" + "".join(f"{'重叠 ' + str(o):>10}" for o in overlaps))
    out.append("  " + "-" * 74)

    for name in retrievers:
        line = f"  {name:<10}"
        for o in overlaps:
            r = by_key.get((name, o))
            cell = f"{r['hits']}/{r['total']}" if r else "-"
            line += f"{cell:>10}"
        out.append(line)
    out.append("")
    out.append("  每行含义：命中题数 / 总题数")
    out.append("")

    # 最优配置
    best = max(rows, key=lambda r: r["rate"])
    out.append("-" * 78)
    out.append(f"  最好的一组   {best['retriever']} + 重叠 {best['overlap']} 字"
               f" → {best['hits']}/{best['total']} = {best['rate']:.0%}")
    out.append("-" * 78)
    out.append("")
    out.append("  每个条件仍失败的原因：")
    out.append("")

    for r in rows:
        tag = f"{r['retriever']:<7} 重叠 {r['overlap']:<3}"
        if not r["reasons"]:
            out.append(f"    {tag} ✅ 全部命中")
        else:
            detail = "，".join(f"{k}×{v}" for k, v in r["reasons"].most_common())
            out.append(f"    {tag} {r['rate']:.0%}   {detail}")

    return "\n".join(out)
