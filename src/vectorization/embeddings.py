from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
from .base import BaseVectorizer


class Embeddings(BaseVectorizer):
    def __init__(self,embedding_type: str,embedding_path: str,vector_size=100,window=5,min_count=2,
                 workers=4):
        self.embedding_type = embedding_type
        self.embedding_path = embedding_path
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def fit(self, X):

        tokenized = [doc.split() for doc in X]

        if self.embedding_type == "word2vec":

            self.model = Word2Vec(
                tokenized,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                workers=self.workers
            )

        elif self.embedding_type == "glove":

            word2vec_file = self.embedding_path + ".word2vec"

            glove2word2vec(
                self.embedding_path,
                word2vec_file
            )

            self.model = KeyedVectors.load_word2vec_format(
                word2vec_file
            )

        else:

            self.model = KeyedVectors.load_word2vec_format(
                self.embedding_path,
                binary=True
            )

        return self
        

    def get_embedding(self, word):

        if hasattr(self.model, "wv"):
            return self.model.wv[word] if word in self.model.wv else None

        return self.model[word] if word in self.model else None

    def most_similar(self, word: str, topn=5):

        if self.model is None:
            raise ValueError(
                "Load embeddings first."
            )

        return self.model.most_similar(
            word,
            topn=topn
        )
    #document pooling
    
    def _doc_vector(self, text):

        vectors = []

        for word in text.split():

            vec = self.get_embedding(word)

            if vec is not None:
                vectors.append(vec)

        if len(vectors) == 0:
            return np.zeros(self.vector_size)

        return np.mean(vectors, axis=0)
    

    def transform(self, texts):
        return np.array([
        self._doc_vector(text)
        for text in texts
    ])

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)