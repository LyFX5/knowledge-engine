from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Node:
    id: str  # Уникальный идентификатор (например, имя файла или UUID блока)
    title: str  # Название заметки
    content: str  # Сырой текст
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[List[float]] = None  # Сюда запишем эмбеддинг


@dataclass
class Edge:
    source: str  # ID исходного узла
    target: str  # ID целевого узла
    edge_type: str  # "explicit" (ссылка), "semantic" (авто-связь), "tag"
    weight: float = 1.0  # Сила связи (особенно важно для семантики)


@dataclass
class KnowledgeGraph:
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
