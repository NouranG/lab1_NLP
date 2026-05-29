from rank_bm25 import BM25Okapi
import numpy as np
from .base import BaseVectorizer

class BM25Vectorizer(BaseVectorizer):

    def __init__(self):

        self.model = None
        self.tokenized_corpus = None

    def fit(self, X):

        self.tokenized_corpus = [
            doc.split() for doc in X
        ]

        self.model = BM25Okapi(self.tokenized_corpus)

        return self

    def transform(self, X):

        tokenized_queries = [
            doc.split() for doc in X
        ]

        return np.array([
            self.model.get_scores(query)
            for query in tokenized_queries
        ])
    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)

    def search(self, query, top_k=5):

        tokenized_query = query.split()

        scores = self.model.get_scores(tokenized_query)

        top_k_indices = scores.argsort()[-top_k:][::-1]

        return top_k_indices, scores[top_k_indices]