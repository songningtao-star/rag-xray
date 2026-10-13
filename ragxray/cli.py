"""
命令行入口。

装好之后可以直接当命令用：
    ragxray --sweep --html report.html

没装（只是 clone 下来）就用：
    python run.py --sweep --html report.html
"""

import argparse
import sys

import yaml

from .report import write_html
from .runner import format_report, run
from .sweep import format_sweep, sweep


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ragxray",
        description="RAG 检索评测与失败归因：不只给命中率，还给失败原因和改法",
    )
    parser.add_argument("-c", "--config", default="config.yaml", help="配置文件路径（默认 config.yaml）")
    parser.add_argument("-o", "--output", help="把文字报告写到这个文件")
    parser.add_argument("--html", help="生成 HTML 报告到这个文件")
    parser.add_argument(
        "--sweep", action="store_true", help="跑参数扫描（检索方式 × 重叠字数）"
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    result = run(config)
    print(format_report(result))

    sweep_rows = None
    if args.sweep:
        sweep_rows = sweep(config)
        print()
        print(format_sweep(sweep_rows))

    if args.html:
        write_html(args.html, result, sweep_rows)
        print(f"\n（HTML 报告已写入 {args.html}，双击即可打开）")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(format_report(result))
        print(f"\n（文字报告已写入 {args.output}）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
