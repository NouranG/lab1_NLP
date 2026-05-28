from rank_bm25 import BM25Okapi

class BM25Vectorizer:
    def __init__(self, text_processor):
        self._text_processor = text_processor
        self._bm25 = None
        self._corpus = None

    def fit(self, texts):
        tokenized_corpus = [self._text_processor(text) for text in texts]
        self._corpus = tokenized_corpus
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(self,query,top_k=5):
        tokenized_query = self._text_processor(query)
        scores = self._bm25.get_scores(tokenized_query)
        top_k_indices = scores.argsort()[-top_k:][::-1]
        return top_k_indices, scores[top_k_indices]
