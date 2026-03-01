# cross encoder reranker built using HuggingFace models
from sentence_transformers import CrossEncoder

class CEReranker():
    """
    Minimal Cross Encoder-based Reranker.
    - Used to rank retrieval results w.r.t. query.
    """

    def __init__(self):
        # NOTE: main model defintion, can be swapped out
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    
    def rerank(self, pairs: list[tuple[str, str]]) -> list[tuple[float, str]]:
        """
        Helper to rerank retrieval results given a list of (query, doc) tuple pairs.
        - Returns a list of reranked (score, doc) tuples, first element being most relevant.
        """
        docs = [doc for _, doc in pairs] # parse out docs
        scores = self.model.predict(pairs)

        # format reranked results
        # NOTE: first doc in reranked list has the highest relevancy (score)
        reranked = sorted(zip(scores, docs), reverse=True)

        return reranked
