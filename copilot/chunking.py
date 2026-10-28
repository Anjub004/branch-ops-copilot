"""Turn markdown SOPs into small, citable chunks (one paragraph each)."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    source: str   # file name, used for citations
    title: str    # document title (the '# ' heading)
    text: str     # one paragraph

    @property
    def indexed_text(self) -> str:
        """What we search over: the title gives each paragraph its context."""
        return f"{self.title}. {self.text}"


def load_chunks(folder="data/sops"):
    chunks = []
    for path in sorted(Path(folder).glob("*.md")):
        title, paragraphs = path.stem, []
        for block in path.read_text(encoding="utf-8").split("\n\n"):
            block = block.strip()
            if not block:
                continue
            if block.startswith("# "):
                title = block[2:].strip()
            else:
                paragraphs.append(block)
        chunks += [Chunk(path.name, title, p) for p in paragraphs]
    return chunks
