"""
三种检索方式，接口完全一致：

    r = BM25Retriever().fit(corpus)
    r.search_all("每天吃几粒？")     → [(片段下标, 分数), ...]   完整排名
    r.search("每天吃几粒？", top_k=3) → 同上，只取前 3 名

这样上层代码（runner）不需要知道用的是哪种检索，换一个类就能换一种方式。
这就是"可插拔"的意思。
"""

import math

import jieba
import numpy as np

MODEL_NAME = "BAAI/bge-small-zh-v1.5"

_model = None


def get_model():
    """向量模型全局只加载一次（加载很慢，别每道题都重新加载）。"""
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(MODEL_NAME)
    return _model


def tokenize(text):
    """中文分词。"""
    return list(jieba.cut(text))


class BaseRetriever:
    name = "base"

    def fit(self, corpus):
        self.corpus = corpus
        self.texts = [c["text"] for c in corpus]
        self.n = len(corpus)
        return self

    def search_all(self, query):
        """完整排名。归因时要看"正确段落到底排第几"，所以不能截断。"""
        raise NotImplementedError

    def search(self, query, top_k):
        """实际返回给下游的，就是前 top_k 名。"""
        return self.search_all(query)[:top_k]


class BM25Retriever(BaseRetriever):
    """
    关键词检索。比字面：你问的词在哪段里出现得多、越罕见，分越高。
    强项是"原词"（保质期、抗凝药物、SKU 编号），弱项是同义词（收到 / 送达）。
    """

    name = "bm25"
    K1, B = 1.5, 0.75

    def fit(self, corpus):
        super().fit(corpus)
        self.tokens = [tokenize(t) for t in self.texts]
        self.n = len(self.tokens)
        self.avgdl = sum(len(t) for t in self.tokens) / self.n
        self.df = {}
        for tokens in self.tokens:
            for w in set(tokens):
                self.df[w] = self.df.get(w, 0) + 1
        return self

    def score_all(self, query):
        """给每一段打分。"""
        q_tokens = tokenize(query)
        scores = np.zeros(self.n)
        for i, tokens in enumerate(self.tokens):
            dl = len(tokens)
            s = 0.0
            for w in q_tokens:
                if w not in self.df:
                    continue
                idf = math.log(1 + (self.n - self.df[w] + 0.5) / (self.df[w] + 0.5))
                tf = tokens.count(w)
                s += idf * (tf * (self.K1 + 1)) / (
                    tf + self.K1 * (1 - self.B + self.B * dl / self.avgdl)
                )
            scores[i] = s
        return scores

    def search_all(self, query):
        scores = self.score_all(query)
        # 得分 0 = 一个词都没匹配上 = 完全不相关。让它参与排名只会制造
        # "碰巧排前面"的假命中（第一步踩过的坑），所以直接排除。
        ranked = sorted(
            (i for i in range(self.n) if scores[i] > 0),
            key=lambda i: -scores[i],
        )
        return [(i, float(scores[i])) for i in ranked]


class VectorRetriever(BaseRetriever):
    """
    向量检索。比意思：把每段和问题都变成一串数字（向量），
    方向越接近表示越相关。强项是同义词，弱项是没见过的专有名词。
    """

    name = "vector"

    def fit(self, corpus):
        super().fit(corpus)
        vecs = np.array(list(get_model().embed(self.texts)))
        self.vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return self

    def sim_all(self, query):
        qv = np.array(list(get_model().embed([query])))[0]
        qv = qv / np.linalg.norm(qv)
        return self.vecs @ qv

    def search_all(self, query):
        sims = self.sim_all(query)
        ranked = sorted(range(self.n), key=lambda i: -sims[i])
        return [(i, float(sims[i])) for i in ranked]


class HybridRetriever(BaseRetriever):
    """
    混合检索（RRF：倒数排名融合）。

    把关键词检索和向量检索各自的排名，换算成 1/(60+名次) 再相加，
    重新排序。作用是"自动拿到两者中较好的那个"，抬高下限。
    """

    name = "hybrid"
    RRF_K = 60

    def fit(self, corpus):
        super().fit(corpus)
        self.bm25 = BM25Retriever().fit(corpus)
        self.vector = VectorRetriever().fit(corpus)
        return self

    def search_all(self, query):
        combined = {}
        for retriever in (self.bm25, self.vector):
            for rank, (idx, _) in enumerate(retriever.search_all(query), 1):
                combined[idx] = combined.get(idx, 0.0) + 1.0 / (self.RRF_K + rank)
        ranked = sorted(combined.items(), key=lambda kv: -kv[1])
        return [(i, s) for i, s in ranked]


RETRIEVERS = {
    "bm25": BM25Retriever,
    "vector": VectorRetriever,
    "hybrid": HybridRetriever,
}


def build_retriever(name):
    if name not in RETRIEVERS:
        raise ValueError(f"不认识的检索方式 '{name}'，可选：{'、'.join(RETRIEVERS)}")
    return RETRIEVERS[name]()
