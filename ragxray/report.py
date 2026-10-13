"""
把评测结果渲染成一份「自包含」的 HTML 报告。

自包含 = 一个 .html 文件，双击就能看，不联网、不装东西、不带任何前端框架。
这样别人 clone 你的仓库跑一遍，就能直接看到一张能截图放进 README 的图。

设计规则（顺序很重要：先决定图形，再决定颜色）：
  1. 命中率是"一个数字"，就该是 hero figure，不是只有一根柱子的柱状图
  2. 网格比大小 → 热力图，用单一蓝色顺序色阶（浅 → 深），并且必须带色阶图例
  3. 数值不能只藏在悬浮提示里 → 每个格子、每根柱子都直接把数字写出来
  4. 色块之间留 2px 表面色缝隙，网格线是发丝级实线（不用虚线）
  5. 深色模式是"另外挑一套颜色"，不是把浅色反相；两套都要显式声明

排版上的一个关键决定：页面底色刻意比卡片底色深一档（#f1f0ed vs #fcfcfb），
卡片才浮得起来。两者太接近的话，再贵的字体也救不回"平"的观感。
"""

import html
from collections import Counter
from datetime import datetime

# 单一蓝色顺序色阶，浅 → 深（对应 命中率 0% → 100%）
SEQ_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]
# 前 8 档够浅，用深墨字；后 5 档够深，用白字
SEQ_FG = ["#0b0b0b"] * 8 + ["#ffffff"] * 5

# 深色模式用的色阶是【另外挑的一套】，不是把上面那套反相：
# 低值贴近深色底、融进背景；高值亮但不刺眼。
# 如果直接拿最浅那两档当最高值，满格的格子会白得发光，
# 反而盖过真正要看的对比。所以顶部收在 #9ec5f4，不冲到近白。
DARK_STEPS = [
    "#0d366b", "#0d366b", "#104281", "#104281", "#184f95", "#1c5cab",
    "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef",
    "#9ec5f4",
]
DARK_FG = ["#ffffff"] * 6 + ["#0b0b0b"] * 7

RETRIEVER_LABEL = {
    "bm25": "bm25 关键词",
    "vector": "vector 向量",
    "hybrid": "hybrid 混合",
}


def _esc(text):
    return html.escape(str(text if text is not None else ""))


def _seq_index(rate):
    """把 0.0~1.0 映射到 1~13 档色阶。"""
    return max(1, min(13, round(rate * 12) + 1))


def _seq_block(dark=False):
    """生成一组色阶变量（--s-1 最低 → --s-13 最高）。

    浅色模式：越深 = 越强，符合直觉。
    深色模式：另一套 —— 低值融进深色底，高值亮而不刺眼。
    """
    pairs = list(zip(DARK_STEPS, DARK_FG)) if dark else list(zip(SEQ_STEPS, SEQ_FG))
    lines = []
    for i, (bg, fg) in enumerate(pairs, 1):
        lines.append(f"    --s-{i}: {bg};")
        lines.append(f"    --s-{i}-fg: {fg};")
    return "\n".join(lines)


# 高 DPI 屏上中文要好看，靠的是字体栈里把系统里最好的中文字体排前面，
# 而不是引入一个要联网下载的 webfont。
FONT_SANS = (
    'system-ui, -apple-system, "Segoe UI", "PingFang SC", "Hiragino Sans GB", '
    '"Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans SC", "Source Han Sans SC", sans-serif'
)
FONT_MONO = (
    'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, '
    '"Liberation Mono", monospace'
)


def _css():
    return """
:root {
  color-scheme: light;
  /* 表面与平面：刻意拉开一档，卡片才浮得起来 */
  --surface-1:      #fcfcfb;
  --surface-2:      #f5f4f1;   /* 表头/次级底 */
  --page:           #f1f0ed;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --muted:          #898781;
  --grid:           #e6e5df;
  --axis:           #c3c2b7;
  --border:         rgba(11, 11, 11, 0.09);
  --series-1:       #2a78d6;
  --chip-bg:        rgba(11, 11, 11, 0.045);
  --shadow:         0 1px 2px rgba(11, 11, 11, 0.04), 0 2px 8px rgba(11, 11, 11, 0.03);
  --good:           #0ca30c;
  --critical:       #d03b3b;
__SEQ_LIGHT__
}

@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-1:      #1a1a19;
    --surface-2:      #232322;
    --page:           #0d0d0d;
    --text-primary:   #f0efec;
    --text-secondary: #c3c2b7;
    --muted:          #898781;
    --grid:           #2c2c2a;
    --axis:           #383835;
    --border:         rgba(255, 255, 255, 0.10);
    --series-1:       #3987e5;
    --chip-bg:        rgba(255, 255, 255, 0.07);
    --shadow:         none;
    --good:           #0ca30c;
    --critical:       #d03b3b;
__SEQ_DARK__
  }
}

:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-1:      #1a1a19;
  --surface-2:      #232322;
  --page:           #0d0d0d;
  --text-primary:   #f0efec;
  --text-secondary: #c3c2b7;
  --muted:          #898781;
  --grid:           #2c2c2a;
  --axis:           #383835;
  --border:         rgba(255, 255, 255, 0.10);
  --series-1:       #3987e5;
  --chip-bg:        rgba(255, 255, 255, 0.07);
  --shadow:         none;
  --good:           #0ca30c;
  --critical:       #d03b3b;
__SEQ_DARK__
}

* { box-sizing: border-box; }

body {
  margin: 0;
  padding: 56px 28px 80px;
  background: var(--page);
  color: var(--text-primary);
  font-family: __FONT_SANS__;
  font-size: 14px;
  line-height: 1.65;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

::selection { background: var(--s-2); color: var(--text-primary); }

.wrap { max-width: 1060px; margin: 0 auto; }

/* ================= 页头 ================= */
.masthead { position: relative; margin-bottom: 30px; padding-bottom: 22px; }
.masthead .wordmark {
  font-size: 12px;
  font-weight: 650;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--series-1);
  margin-bottom: 10px;
}
.masthead h1 {
  margin: 0 0 8px;
  font-size: 30px;
  font-weight: 680;
  letter-spacing: -0.022em;
  line-height: 1.25;
}
.masthead .lede {
  margin: 0;
  max-width: 62ch;
  font-size: 14.5px;
  color: var(--text-secondary);
}
.masthead::after {
  content: "";
  position: absolute;
  left: 0; right: 0; bottom: 0;
  height: 1px;
  background: linear-gradient(90deg, var(--axis), transparent 72%);
}

.chips { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 18px; }
.chip {
  font-size: 12px;
  padding: 5px 11px;
  border-radius: 999px;
  background: var(--surface-1);
  border: 1px solid var(--border);
  color: var(--muted);
  white-space: nowrap;
}
.chip b { color: var(--text-primary); font-weight: 620; margin-left: 3px; }

/* 主题切换：让深色模式真的够得着，而不是只跟着系统走 */
.theme-toggle {
  position: absolute;
  top: 0; right: 0;
  width: 34px; height: 34px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--surface-1);
  color: var(--text-secondary);
  font-size: 15px;
  line-height: 1;
  cursor: pointer;
  transition: color .15s, border-color .15s, transform .15s;
}
.theme-toggle:hover { color: var(--text-primary); border-color: var(--axis); transform: rotate(18deg); }

/* ================= 卡片 ================= */
.card {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 14px;
  box-shadow: var(--shadow);
  padding: 24px 26px 26px;
  margin-bottom: 16px;
}
.card > h2 {
  display: flex;
  align-items: baseline;
  gap: 10px;
  margin: 0 0 5px;
  font-size: 15.5px;
  font-weight: 640;
  letter-spacing: -0.005em;
}
/* 章节序号：让长报告有"目录感"，也帮人定位 */
.card > h2 .idx {
  font-size: 11.5px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  color: var(--muted);
  letter-spacing: 0.06em;
}
.card > .sub {
  margin: 0 0 20px;
  font-size: 13px;
  color: var(--muted);
  max-width: 74ch;
}
.card > .sub b { color: var(--text-secondary); font-weight: 600; }

/* ================= hero + 指标卡 ================= */
.hero-row {
  display: grid;
  grid-template-columns: minmax(230px, 0.85fr) 1.4fr;
  gap: 28px;
  align-items: center;
}
@media (max-width: 760px) { .hero-row { grid-template-columns: 1fr; gap: 22px; } }

.hero .value {
  /* hero 用比例数字（不是 tabular-nums），和正文同一套无衬线 */
  font-size: 68px;
  font-weight: 690;
  line-height: 0.95;
  letter-spacing: -0.04em;
}
.hero .value small { font-size: 27px; font-weight: 620; margin-left: 3px; letter-spacing: -0.02em; }
.hero .label { font-size: 13px; color: var(--muted); margin-top: 10px; }

/* 单一比例对上限 → 用 meter，不用"只有一根柱子的柱状图"。
   轨道用同一色阶最浅那档，和填充是同一套蓝。 */
.meter {
  margin-top: 14px;
  height: 9px;
  border-radius: 999px;
  background: var(--s-1);
  overflow: hidden;
}
.meter .fill { height: 100%; background: var(--s-8); border-radius: 999px; }

.tiles {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(104px, 1fr));
  gap: 2px;                       /* 2px 表面色缝隙，不画边框 */
  background: var(--grid);
  border-radius: 10px;
  overflow: hidden;
}
.tile { background: var(--surface-1); padding: 16px 18px; }
.tile .t-label { font-size: 12px; color: var(--muted); }
.tile .t-value { font-size: 23px; font-weight: 640; margin-top: 3px; letter-spacing: -0.01em; }

/* ================= 热力图 ================= */
.heat-scroll { overflow-x: auto; margin: 0 -2px; padding: 0 2px; }
table.heat {
  border-collapse: separate;
  border-spacing: 2px;            /* 色块之间 2px 表面色缝隙 */
  font-variant-numeric: tabular-nums;
  width: 100%;
}
table.heat th {
  font-size: 11.5px;
  font-weight: 500;
  color: var(--muted);
  padding: 5px 10px;
  text-align: center;
  white-space: nowrap;
}
table.heat th.rowhead { text-align: left; padding-left: 0; font-size: 12.5px; }
table.heat th.corner { text-align: left; padding-left: 0; color: var(--axis); font-weight: 400; }
table.heat td {
  /* line-height 收到 1：默认的 1.65 会把格子撑得又高又厚，
     整片色块看着就笨重 */
  padding: 14px 10px;
  line-height: 1;
  text-align: center;
  font-size: 15px;
  font-weight: 620;
  border-radius: 6px;
  letter-spacing: -0.01em;
  transition: filter .12s;
}
/* 数值已经直接标在格子里，悬浮只是"确认"，不是唯一读数途径 */
table.heat td:hover { filter: brightness(1.07); }

.legend {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 20px;
  font-size: 12px;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.legend .scale { display: flex; gap: 2px; }
.legend .scale i { width: 27px; height: 11px; border-radius: 3px; display: block; }

/* ================= 横条 ================= */
.bars { display: flex; flex-direction: column; gap: 12px; }
.barrow { display: grid; grid-template-columns: 104px 1fr 34px; align-items: center; gap: 14px; }
.barrow .name { font-size: 13px; color: var(--text-secondary); }
/* display:flex 是必须的：span 默认是行内元素，行内元素上的 height 不生效 */
.barrow .track { height: 15px; display: flex; border-bottom: 1px solid var(--axis); }
/* 数据端 4px 圆角，起点贴基线 */
.barrow .fill { height: 100%; background: var(--series-1); border-radius: 0 4px 4px 0; min-width: 3px; }
.barrow .val { font-size: 13.5px; font-weight: 600; text-align: right; font-variant-numeric: tabular-nums; }

/* 只有一类失败时，条形图就退化成"只有一根柱子"，改用 stat tile */
.statline { display: flex; align-items: baseline; gap: 10px; padding: 6px 0 0; }
.statline .n { font-size: 34px; font-weight: 680; line-height: 1; letter-spacing: -0.03em; }
.statline .u { font-size: 13px; color: var(--muted); }
.statline .what { font-size: 14.5px; color: var(--text-secondary); }

/* ================= 表格 ================= */
table.grid { width: 100%; border-collapse: collapse; font-size: 13.5px; }
table.grid th {
  text-align: left;
  font-size: 11.5px;
  font-weight: 550;
  color: var(--muted);
  padding: 0 14px 9px 0;
  border-bottom: 1px solid var(--axis);
  white-space: nowrap;
}
table.grid td {
  padding: 12px 14px 12px 0;
  border-bottom: 1px solid var(--grid);
  vertical-align: top;
}
table.grid tbody tr:last-child td { border-bottom: none; }
table.grid tbody tr { transition: background .1s; }
table.grid tbody tr:hover td { background: var(--chip-bg); }
table.grid .num { font-variant-numeric: tabular-nums; white-space: nowrap; color: var(--text-secondary); }
table.grid .mono { font-family: __FONT_MONO__; font-size: 12.5px; color: var(--text-secondary); }
table.grid td.q { font-weight: 550; }

/* ================= 状态徽章：图标 + 文字，不靠颜色单独表意 ================= */
.badge { display: inline-flex; align-items: center; gap: 5px; font-size: 12.5px; font-weight: 550; white-space: nowrap; }
.badge.ok  { color: var(--good); }
.badge.bad { color: var(--critical); }
.badge.cut {
  color: var(--text-secondary);
  background: var(--chip-bg);
  padding: 4px 11px;
  border-radius: 999px;
  font-size: 12px;
}

/* ================= 归因卡片 ================= */
.cases { display: grid; gap: 12px; }
.case {
  border: 1px solid var(--border);
  border-radius: 11px;
  padding: 17px 19px 18px;
  background: var(--surface-1);
  transition: border-color .15s;
}
.case:hover { border-color: var(--axis); }
.case .q { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 13px; }
.case .q .text { font-weight: 620; font-size: 14.5px; letter-spacing: -0.005em; }
.case dl { margin: 0; display: grid; grid-template-columns: 44px 1fr; gap: 8px 14px; font-size: 13.5px; }
.case dt { color: var(--muted); font-size: 12.5px; padding-top: 1px; }
/* 证据里带换行（左半截 / 右半截分行对照），保留换行才看得懂 */
.case dd { margin: 0; color: var(--text-secondary); white-space: pre-line; line-height: 1.75; }
.case dd.strong { color: var(--text-primary); }

.empty { color: var(--muted); font-size: 13.5px; padding: 4px 0; }

.foot {
  margin-top: 26px;
  padding-top: 18px;
  border-top: 1px solid var(--border);
  font-size: 12px;
  color: var(--muted);
  text-align: center;
}

/* 打印 / 导出 PDF 时不要阴影和按钮，让内容贴着纸面 */
@media print {
  body { background: #fff; padding: 0; }
  .card { box-shadow: none; border-color: #ddd; break-inside: avoid; }
  .theme-toggle { display: none; }
}
""".replace("__SEQ_LIGHT__", _seq_block(False)).replace(
        "__SEQ_DARK__", _seq_block(dark=True)
    ).replace("__FONT_SANS__", FONT_SANS).replace("__FONT_MONO__", FONT_MONO)


def _render_hero(result, hits, total, rate, n_reasons):
    filled = f"{rate * 100:.1f}%"
    return f"""
  <div class="hero-row">
    <div class="hero">
      <div class="value">{rate * 100:.0f}<small>%</small></div>
      <div class="meter" role="img"
           aria-label="命中率 {filled}"><div class="fill" style="width:{filled}"></div></div>
      <div class="label">{hits} / {total} 道题命中</div>
    </div>
    <div class="tiles">
      <div class="tile"><div class="t-label">命中</div><div class="t-value">{hits}</div></div>
      <div class="tile"><div class="t-label">未命中</div><div class="t-value">{total - hits}</div></div>
      <div class="tile"><div class="t-label">知识库片段</div><div class="t-value">{result['n_chunks']}</div></div>
      <div class="tile"><div class="t-label">失败类型</div><div class="t-value">{n_reasons}</div></div>
    </div>
  </div>"""


def _render_heatmap(rows):
    """检索方式（行）× 重叠字数（列）的命中率网格。"""
    retrievers = list(dict.fromkeys(r["retriever"] for r in rows))
    overlaps = sorted(set(r["overlap"] for r in rows))
    by_key = {(r["retriever"], r["overlap"]): r for r in rows}

    body = []
    for name in retrievers:
        cells = []
        for o in overlaps:
            r = by_key.get((name, o))
            if r is None:
                cells.append("<td></td>")
                continue
            i = _seq_index(r["rate"])
            # 数值直接写在格子里，不依赖悬浮提示
            cells.append(
                f"<td style='background:var(--s-{i});color:var(--s-{i}-fg)'>"
                f"{r['rate']:.0%}</td>"
            )
        body.append(
            f"<tr><th class='rowhead'>{_esc(RETRIEVER_LABEL.get(name, name))}</th>"
            + "".join(cells)
            + "</tr>"
        )

    swatches = "".join(f"<i style='background:var(--s-{i})'></i>" for i in range(1, 14))

    return f"""
  <div class="heat-scroll">
    <table class="heat">
      <thead><tr><th class="corner">检索方式 \\ 重叠字数</th>{''.join(f'<th>{o}</th>' for o in overlaps)}</tr></thead>
      <tbody>{''.join(body)}</tbody>
    </table>
  </div>
  <div class="legend">
    <span>命中率</span>
    <span>0%</span>
    <span class="scale">{swatches}</span>
    <span>100%</span>
  </div>
"""


def _render_reasons(counter):
    if not counter:
        return "<p class='empty'>没有失败，全部命中。</p>"

    # 只有一类失败时，"比较各类多少"这件事不存在，画条形图就是"一根柱子的柱状图"。
    # 这种情况数字本身就是全部信息 —— 用 stat tile。
    if len(counter) == 1:
        name, cnt = counter.most_common(1)[0]
        return (
            f"<div class='statline'>"
            f"<span class='n'>{cnt}</span><span class='u'>条</span>"
            f"<span class='what'>{_esc(name)}</span>"
            f"</div>"
        )

    top = max(counter.values())
    rows = []
    for name, cnt in counter.most_common():
        pct = cnt / top * 100
        rows.append(
            f"<div class='barrow'>"
            f"<span class='name'>{_esc(name)}</span>"
            f"<span class='track'><span class='fill' style='width:{pct:.1f}%'></span></span>"
            f"<span class='val'>{cnt}</span>"
            f"</div>"
        )
    return f"<div class='bars'>{''.join(rows)}</div>"


def _render_table(results, top_k):
    rows = []
    for x in results:
        if x["hit"]:
            status = "<span class='badge ok'>✓ 命中</span>"
            reason = "<span style='color:var(--axis)'>—</span>"
        else:
            status = "<span class='badge bad'>✕ 未命中</span>"
            reason = _esc(x["diagnosis"]["type"]) if x["diagnosis"] else "—"
        rank = f"第 {x['rank']} 名" if x["rank"] else "未召回"
        rows.append(
            f"<tr>"
            f"<td class='q'>{_esc(x['query'])}</td>"
            f"<td class='mono'>{_esc(x['doc'])}</td>"
            f"<td class='num'>{_esc(rank)}</td>"
            f"<td>{status}</td>"
            f"<td>{reason}</td>"
            f"</tr>"
        )
    return (
        "<table class='grid'><thead><tr>"
        "<th style='width:25%'>问题</th>"
        "<th style='width:19%'>标准答案所在文档</th>"
        f"<th style='width:13%'>排名（top_k={top_k}）</th>"
        "<th style='width:15%'>结果</th>"
        "<th>失败类型</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _render_cases(results):
    cases = [x for x in results if x["diagnosis"]]
    if not cases:
        return "<p class='empty'>没有需要归因的失败。</p>"
    out = []
    for x in cases:
        d = x["diagnosis"]
        out.append(
            f"<div class='case'>"
            f"<div class='q'><span class='text'>{_esc(x['query'])}</span>"
            f"<span class='badge cut'>{_esc(d['type'])}</span></div>"
            f"<dl>"
            f"<dt>证据</dt><dd class='strong'>{_esc(d['evidence'])}</dd>"
            f"<dt>建议</dt><dd>{_esc(d['suggestion'])}</dd>"
            f"</dl></div>"
        )
    return f"<div class='cases'>{''.join(out)}</div>"


def _card(idx, heading, sub, body):
    num = f"<span class='idx'>{idx:02d}</span>" if idx else ""
    return (
        f"<div class='card'><h2>{num}{_esc(heading)}</h2>"
        f"<p class='sub'>{sub}</p>{body}</div>"
    )


def build_html(result, sweep_rows=None, title="ragxray 检索评测报告"):
    """生成完整 HTML 文本。sweep_rows 为 None 时就不画热力图。"""
    results = result["results"]
    total = len(results)
    hits = sum(1 for x in results if x["hit"])
    rate = hits / total if total else 0.0
    reasons = Counter(x["diagnosis"]["type"] for x in results if x["diagnosis"])

    chips = "".join(
        f"<span class='chip'>{k}<b>{v}</b></span>"
        for k, v in [
            ("检索方式", _esc(result["retriever"])),
            ("切片", f"{result['chunk_size']} 字"),
            ("重叠", f"{result['chunk_overlap']} 字"),
            ("top_k", result["top_k"]),
            ("文档", f"{result['n_docs']} 篇"),
            ("片段", f"{result['n_chunks']} 段"),
            ("标注题", f"{total} 道"),
        ]
    )

    cards = []
    n = 0

    n += 1
    cards.append(
        _card(
            n,
            "本次命中率",
            "命中 = 正确答案所在的片段排进了 top_k。",
            _render_hero(result, hits, total, rate, len(reasons)),
        )
    )

    if sweep_rows:
        best = max(sweep_rows, key=lambda r: r["rate"])
        n += 1
        cards.append(
            _card(
                n,
                "参数扫描：哪组条件最好",
                f"共 {len(sweep_rows)} 组条件。最好的一组是 "
                f"<b>{_esc(best['retriever'])} + 重叠 {best['overlap']} 字</b>"
                f"（{best['hits']}/{best['total']}）。",
                _render_heatmap(sweep_rows),
            )
        )

    n += 1
    cards.append(
        _card(
            n,
            "失败原因汇总",
            "这是「下一步该改什么」的答案：哪一类最多，就先解决哪一类。",
            _render_reasons(reasons),
        )
    )

    n += 1
    cards.append(
        _card(
            n,
            "逐题结果",
            "每一道题、它的标准答案在哪篇、实际排第几。",
            _render_table(results, result["top_k"]),
        )
    )

    n += 1
    cards.append(
        _card(
            n,
            "失败归因：证据与建议",
            "不止说「没命中」，而是给出具体位置和可以动手改的地方。",
            _render_cases(results),
        )
    )

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{_css()}</style>
</head>
<body>
<div class="wrap">

  <header class="masthead">
    <button class="theme-toggle" id="theme-toggle" type="button"
            aria-label="切换深色 / 浅色模式" title="切换深色 / 浅色模式">◐</button>
    <div class="wordmark">ragxray</div>
    <h1>{_esc(title)}</h1>
    <p class="lede">同一套知识库、同一套标注，换检索方式与切片参数反复跑，
       并把每一次失败定位到具体的字符位置 —— 不只是给一个分数。</p>
    <div class="chips">{chips}</div>
  </header>

{''.join(cards)}

  <p class="foot">由 ragxray 生成 · {stamp}</p>
</div>

<script>
(function () {{
  var root = document.documentElement;
  try {{
    var saved = localStorage.getItem('ragxray-theme');
    if (saved) root.setAttribute('data-theme', saved);
  }} catch (e) {{}}

  document.getElementById('theme-toggle').addEventListener('click', function () {{
    var cur = root.getAttribute('data-theme');
    if (!cur) {{
      cur = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }}
    var next = cur === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try {{ localStorage.setItem('ragxray-theme', next); }} catch (e) {{}}
  }});
}})();
</script>
</body>
</html>
"""


def write_html(path, result, sweep_rows=None, title="ragxray 检索评测报告"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_html(result, sweep_rows, title))
    return path
