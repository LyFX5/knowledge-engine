from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from models import Node


class BaseEmbedder:
    def compute_embeddings(self, nodes: List[Node]) -> List[Node]:
        raise NotImplementedError


class LocalTfidfEmbedder(BaseEmbedder):
    """
    Первый шаг автоматизации: строит векторы на основе частотности
    токенов (слов), очищенных от шума.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )  # Можно расширить под ру-язык

    def compute_embeddings(self, nodes: List[Node]) -> List[Node]:
        if not nodes:
            return nodes

        corpus = [node.content for node in nodes]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)

        # Переводим разреженную матрицу в массивы float
        dense_matrix = tfidf_matrix.toarray()
        for idx, node in enumerate(nodes):
            node.vector = dense_matrix[idx].tolist()
        return nodes
