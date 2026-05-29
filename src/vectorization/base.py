from abc import ABC, abstractmethod
import numpy as np

class BaseVectorizer(ABC):
    @abstractmethod
    def fit(self, texts):
        pass

    @abstractmethod
    def transform(self, texts):
        pass

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)