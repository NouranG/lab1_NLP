from sklearn.feature_extraction.text import CountVectorizer
from .base import BaseVectorizer

class BoWVectorizer(BaseVectorizer):
    """BoW vectorizer, using sklearn's CountVectorizer."""
    def __init__(self, text_processor,max_features=5000,ngram_range=(1, 1)):
        self._text_processor = text_processor
        self._vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)

    def fit(self, texts):
        self._vectorizer.fit(texts)
        return self

    def transform(self, texts):
        return self._vectorizer.transform(texts).toarray()
    
    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names(self):
        return self._vectorizer.get_feature_names_out()