from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFVectorizer:
    """TF-IDF vectorizer, using sklearn's TfidfVectorizer."""
    def __init__(self, text_processor,max_features=5000,ngram_range=(1, 1)):
        self._text_processor = text_processor
        self._vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    def fit_transform(self,texts):
        return self._vectorizer.fit_transform(texts)
    
    def transform(self,texts):
        return self._vectorizer.transform(texts)

    def get_feature_names(self):
        return self._vectorizer.get_feature_names_out()