import numpy as np
from pathlib import Path
from typing import List
from models import KnowledgeGraph, Edge
from parser import MarkdownParser
from embedder import LocalTfidfEmbedder


class KnowledgeEngine:
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.parser = MarkdownParser()
        self.embedder = LocalTfidfEmbedder()
        self.graph = KnowledgeGraph()

    def scan_vault(self):
        raw_nodes = []
        # Шаг 1: Сканируем локальную директорию с Markdown
        for file_path in self.vault_path.glob("**/*.md"):
            node = self.parser.parse_file(file_path)
            raw_nodes.append(node)

        # Шаг 2: Считаем семантические векторы (токенизация/эмбеддинги)
        processed_nodes = self.embedder.compute_embeddings(raw_nodes)
        for node in processed_nodes:
            self.graph.nodes[node.id] = node

        # Шаг 3: Извлекаем явные связи
        for node in self.graph.nodes.values():
            explicit_edges = self.parser.extract_explicit_edges(node)
            self.graph.edges.extend(explicit_edges)

    def generate_auto_links(self, threshold: float = 0.3):
        """
        Вычисляет косинусное сходство между векторами заметок.
        Если они похожи, но явной ссылки нет — создает авто-линковку.
        """
        nodes_list = list(self.graph.nodes.values())
        num_nodes = len(nodes_list)

        if num_nodes < 2 or nodes_list[0].vector is None:
            return

        # Собираем матрицу эмбеддингов
        vectors = np.array([node.vector for node in nodes_list])

        # Считаем косинусное сходство (нормализуем и умножаем матрицы)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Защита от деления на ноль
        norms[norms == 0] = 1.0
        norm_vectors = vectors / norms
        similarity_matrix = np.dot(norm_vectors, norm_vectors.T)

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                sim = similarity_matrix[i, j]
                if sim >= threshold:
                    src = nodes_list[i].id
                    tgt = nodes_list[j].id

                    # Проверяем, нет ли уже явной связи между ними
                    exists = any(
                        (e.source == src and e.target == tgt)
                        or (e.source == tgt and e.target == src)
                        for e in self.graph.edges
                    )

                    if not exists:
                        self.graph.edges.append(
                            Edge(
                                source=src,
                                target=tgt,
                                edge_type="semantic",
                                weight=float(sim),
                            )
                        )
