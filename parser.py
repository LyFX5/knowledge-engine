import re
from pathlib import Path
from models import Node, Edge


class MarkdownParser:
    def __init__(self):
        # Регулярка для поиска [[Имя Заметки]] или [[Имя Заметки|Альтернативное имя]]
        self.wikilink_re = re.compile(r"\[\[(.*?)\]\]")
        self.tag_re = re.compile(r"#(\w+)")

    def parse_file(self, file_path: Path) -> Node:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        title = file_path.stem
        # Базовая метаинформация
        metadata = {
            "path": str(file_path),
            "tags": self.tag_re.findall(content),
        }
        return Node(id=title, title=title, content=content, metadata=metadata)

    def extract_explicit_edges(self, node: Node) -> List[Edge]:
        edges = []
        links = self.wikilink_re.findall(node.content)
        for link in links:
            # Отрезаем alias, если ссылка вида [[TargetNode|Display Name]]
            target = link.split("|")[0].strip()
            edges.append(
                Edge(source=node.id, target=target, edge_type="explicit")
            )
        return edges
