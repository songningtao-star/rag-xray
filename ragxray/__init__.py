"""
ragxray —— RAG 检索评测与失败归因工具

不只是一个评测工具：跑完命中率之后，它会逐条告诉你
「这道题为什么错、该改什么」，并且给出证据。

模块划分：
    loaders     读知识库文件和标注集
    chunkers    切片
    retrievers  三种检索方式（关键词 / 向量 / 混合）
    metrics     评测指标
    attributor  归因引擎 ← 本项目的核心
    runner      主流程
    sweep       参数扫描
    report      生成 HTML 报告
    cli         命令行入口
"""

__version__ = "0.1.0"
