from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


class Embeddings:
    def __init__(self,embedding_type: str,embedding_path: str):
        self.embedding_type = embedding_type
        self.embedding_path = embedding_path
        self.model = None
    def load_embeddings(self):
        if self.embedding_type == 'word2vec':
            self.model = KeyedVectors.load_word2vec_format(self.embedding_path, binary=True)
        elif self.embedding_type == 'glove':
            word2vec_output_file = self.embedding_path + '.word2vec'
            glove2word2vec(self.embedding_path, word2vec_output_file)
            self.model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
        else:
            raise ValueError(f"Unsupported embedding type: {self.embedding_type}")
        
    def get_embedding(self, word):
        if self.model is None:
            raise ValueError("Embeddings not loaded. Call load_embeddings() first.")
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