# -*- coding: utf-8 -*-
"""
没安装包时的入口：python run.py --sweep --html report.html

真正的逻辑在 ragxray/cli.py，这里只是为了让你 clone 下来就能跑。
用法说明见 README.md。
"""

import sys

from ragxray.cli import main

if __name__ == "__main__":
    sys.exit(main())
