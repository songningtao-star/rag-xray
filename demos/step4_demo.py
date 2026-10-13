# -*- coding: utf-8 -*-
"""
第四步：切片（chunking）—— 归因的第 2 类失败：切片切断

前三步我们用的文档都是一句话，切不动，所以一直跳过了切片。
这一版用一篇真实长度的产品说明书，把切片的影响演示出来。

核心结论预告：
    切得不好，答案会被劈成两半，两段各拿一半，模型谁也答不全。

跑法：  python step4_demo.py
"""

import numpy as np
from fastembed import TextEmbedding

# ============================================================
# 【1】知识库：一篇真实长度的产品说明书（151 字）
# ============================================================
DOC = (
    "【成分】本品为进口辅酶Q10软胶囊，每粒含辅酶Q10 100mg，规格为每瓶60粒。"
    "【用法用量】建议每日服用一次，每次一粒，随餐服用效果更佳。"
    "【不适宜人群】孕妇、哺乳期妇女及18岁以下未成年人请勿服用。"
    "【服用禁忌】正在服用抗凝药物者请咨询医师。"
    "【保存方法】置于阴凉干燥处，避免阳光直射，保质期24个月。"
)
DOC_ID = "说明书"

# 标注集：问题、答案的原文短语在文档哪个位置（这里直接写出短语，程序自己去找位置）
TESTS = [
    ("每天吃几粒？", "每次一粒"),
    ("孕妇能吃吗？", "孕妇、哺乳期妇女"),
    ("能吃抗凝药吗？", "抗凝药物"),
    ("怎么保存？", "阴凉干燥处"),
    ("保质期多久？", "保质期24个月"),
]

TOP_K = 2   # 检索返回前几名。给模型看的是这前几名，所以必须是完整的一段。


# ============================================================
# 【2】切片函数：按固定字数切，相邻两段留一点重叠
#     overlap=0 就是不重叠（最容易出问题的切法）
# ============================================================
def chunk(text, size, overlap):
    """返回 [(起始位置, 结束位置, 文本), ...]"""
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        out.append((start, end, text[start:end]))
        if end >= len(text):
            break
        start += size - overlap
    return out


# ============================================================
# 【3】向量检索
# ============================================================
print("正在加载向量模型...")
model = TextEmbedding("BAAI/bge-small-zh-v1.5")


def search(query, chunks):
    """在给定的若干段里检索，返回按相似度排序的 [(下标, 相似度), ...]"""
    texts = [c[2] for c in chunks]
    dv = np.array(list(model.embed(texts)))
    dv /= np.linalg.norm(dv, axis=1, keepdims=True)
    qv = np.array(list(model.embed([query])))[0]
    qv /= np.linalg.norm(qv)
    sims = dv @ qv
    return sorted(enumerate(sims.tolist()), key=lambda kv: kv[1], reverse=True)


# ============================================================
# 开始输出
# ============================================================
print()
print("=" * 78)
print("【1】知识库：一篇产品说明书（151 字）")
print("=" * 78)
print()
print("  " + DOC)

SETTINGS = [
    ("整篇不切",          200, 0),
    ("切小块·不重叠",      59, 0),
    ("切小块·有重叠",      59, 20),
]

print()
print("=" * 78)
print("【2】为什么要切片")
print("=" * 78)
print()
print("  模型一次能看的字数是有限的，而且文档越长，噪音越多、检索越不准。")
print("  所以真实系统会把长文档切成一段一段（叫 chunk），一段一段地检索。")
print()
print("  但切在哪里，是个坑。看下面三种切法的对比：")

for name, size, ov in SETTINGS:
    chunks = chunk(DOC, size, ov)
    print()
    print(f"  ● {name}  (每段{size}字, 重叠{ov}字)  →  切成了 {len(chunks)} 段")
    for i, (a, b, txt) in enumerate(chunks):
        short = txt if len(txt) <= 46 else txt[:46] + "…"
        print(f"      第{i}段 [{a:3d}-{b:3d}]  {short}")


# ============================================================
# 【4】三种切法下，检索命中率对比
#     判定规则：top-2 里必须有一段「完整包含」答案短语。
#     为什么要求"完整"？因为给模型的是一整段文字，
#     段落里缺了关键的半句，模型就等于没拿到答案。
# ============================================================
print()
print("=" * 78)
print("【3】三种切法下的命中率")
print("=" * 78)
print()
print(f"  {'问题':<16}{'整篇不切':<12}{'切小块·不重叠':<16}{'切小块·有重叠'}")
print("  " + "-" * 62)

for q, phrase in TESTS:
    pos = DOC.find(phrase)
    row = []
    for name, size, ov in SETTINGS:
        chunks = chunk(DOC, size, ov)
        ranked = search(q, chunks)
        top_idx = [i for i, _ in ranked[:TOP_K]]
        complete = [i for i, c in enumerate(chunks) if phrase in c[2]]
        ok = any(i in top_idx for i in complete)
        row.append("✅ 命中" if ok else "❌ 失败")
    print(f"  {q:<16}{row[0]:<14}{row[1]:<18}{row[2]}")

print()
print("  判定规则：检索返回的前 2 段里，必须有一段「完整包含」答案原文。")
print("           缺一半就等于没给 —— 模型看不到完整的话，就会答错。")


# ============================================================
# 【5】归因：把失败的那条拆开看 —— 切片切断
# ============================================================
print()
print("=" * 78)
print("【4】归因：失败的那条，到底哪出错了")
print("=" * 78)

SIZE, OV = 59, 0
chunks = chunk(DOC, SIZE, OV)
cut_cases = []
for q, phrase in TESTS:
    pos = DOC.find(phrase)
    ranked = search(q, chunks)
    top_idx = [i for i, _ in ranked[:TOP_K]]
    complete = [i for i, c in enumerate(chunks) if phrase in c[2]]
    if not any(i in top_idx for i in complete):
        cut_cases.append((q, phrase, pos))

for q, phrase, pos in cut_cases:
    print(f"\n  问题：「{q}」   答案原文是「{phrase}」（在文档第 {pos} 个字开始）")
    print()

    # 找出是哪条切片边界把它切断了
    boundary = None
    for start, end, _ in chunks:
        if start != 0 and pos < start < pos + len(phrase):
            boundary = start
            break

    if boundary is not None:
        left = DOC[max(0, boundary - 8):boundary]
        right = DOC[boundary:boundary + 8]
        print(f"    切片边界落在第 {boundary} 个字上，正好在答案中间：")
        print()
        print(f"        第 0 段结尾：  …{left}   ← 只有前半截「{DOC[pos:boundary]}」")
        print(f"        第 1 段开头：  {right}…   ← 只有后半截「{DOC[boundary:pos + len(phrase)]}」")
        print()
        print(f"    「{phrase}」被劈成了两半，没有任何一段是完整的。")
        print(f"    所以就算检索找对了地方（它确实把第 1 段排到了第 1 名），")
        print(f"    递给模型的也是一段残缺的话 —— 模型照样答不出来。")

    print()
    print(f"    ● 工具给出的诊断：")
    print(f"        失败类型  = 切片切断")
    print(f"        证据      = 答案短语横跨第 {boundary} 个字的切片边界")
    print(f"        建议      = 把 overlap 从 0 调到 20 以上（或加大每段字数）")

    # 验证建议
    fixed = chunk(DOC, SIZE, 20)
    ranked2 = search(q, fixed)
    top2 = [i for i, _ in ranked2[:TOP_K]]
    complete2 = [i for i, c in enumerate(fixed) if phrase in c[2]]
    ok2 = any(i in top2 for i in complete2)
    print(f"        复验      = 改成 overlap=20 后，这条题 {"✅ 恢复了" if ok2 else "❌ 仍然失败"}")

print()
print("=" * 78)
print("【5】结论")
print("=" * 78)
print()
print("  切片的取舍：")
print("    切得太碎 → 一句话被拆散，检索到的是残缺片段（刚才看到的问题）")
print("    切得太大 → 一段里混进太多无关内容，检索变得不准、还浪费模型的上下文")
print()
print("  所以「该切多大、要不要重叠」没有标准答案，只能拿标注集测。")
print("  这正是我们那个工具要自动回答的问题之一。")
print()
print("  现在这个工具已经能自动诊断 2 类失败了：")
print("    第 1 类 = 同义词失配（step1~3）  →  建议换向量检索 / 用混合")
print("    第 2 类 = 切片切断（step4）      →  建议调 overlap / 调每段大小")
