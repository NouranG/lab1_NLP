from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


class Embeddings:
    def __init__(self,embedding_type: str,embedding_path: str,vector_size=100,window=5,min_count=2,
                 workers=4):
        self.embedding_type = embedding_type
        self.embedding_path = embedding_path
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
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
        
    def train_embeddings(self, texts):
        tokenized_texts = [
            text.split()
            for text in texts
        ]
        self.model = Word2Vec(tokenized_texts, vector_size=self.vector_size,
                               window=self.window, min_count=self.min_count, 
                               workers=self.workers)

    def get_embedding(self, word):
        if self.model is None:
            raise ValueError("Embeddings not loaded or trained.")
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