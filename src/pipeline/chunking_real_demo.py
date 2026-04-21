"""
Public fragment based on a real chunking stage from a book-processing pipeline.

Смысл этого слоя:
не просто разрезать текст на куски,
а превратить большой сырой текст книги в смысловые chunks,
которые реально пригодны для retrieval и RAG.
"""

from __future__ import annotations

from typing import List


MIN_CHARS = 300
MAX_CHARS = 1500


def normalize_text(text: str) -> str:
    """
    Убираем пустые строки и лишние пробелы.
    Это важно как первый шаг перед chunking:
    retrieval quality ломается уже на грязном тексте.
    """
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def split_into_paragraphs(text: str) -> List[str]:
    """
    Разбиение, устойчивое к PDF и плохо распарсенным источникам.

    Логика:
    1. Сначала пробуем нормальные абзацы через двойной перевод строки.
    2. Если PDF «слипся» в единый блок — режем по строкам
       и собираем абзацы по логическим паузам.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if len(paragraphs) <= 1:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        paragraphs = []
        current = []

        for line in lines:
            current.append(line)
            if line.endswith(".") or line.endswith(":"):
                paragraphs.append(" ".join(current))
                current = []

        if current:
            paragraphs.append(" ".join(current))

    return paragraphs


def build_chunks(paragraphs: List[str]) -> List[str]:
    """
    Собираем chunks из абзацев.

    Принцип:
    - не терять смысловые границы
    - не делать слишком короткие шумовые chunks
    - не делать слишком длинные блоки, которые ухудшают retrieval
    """
    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > MAX_CHARS:
            start = 0
            while start < len(paragraph):
                part = paragraph[start:start + MAX_CHARS]
                if len(part) >= MIN_CHARS:
                    chunks.append(part)
                start += MAX_CHARS
            continue

        if len(current) + len(paragraph) <= MAX_CHARS:
            current = current + "\n\n" + paragraph if current else paragraph
        else:
            if len(current) >= MIN_CHARS:
                chunks.append(current)
            current = paragraph

    if current:
        if len(current) >= MIN_CHARS:
            chunks.append(current)
        else:
            chunks.append(current)

    return chunks


def chunk_document(text: str) -> List[str]:
    """
    Полный chunking pipeline:
    raw text -> normalization -> paragraph split -> semantic chunks
    """
    normalized = normalize_text(text)
    paragraphs = split_into_paragraphs(normalized)
    return build_chunks(paragraphs)


if __name__ == "__main__":
    demo_text = """
    Ребенок может испытывать устойчивую тревогу в ситуации,
    когда учебная нагрузка переживается как источник давления.

    Важно смотреть не только на сам симптом,
    но и на повторяемость реакции, контекст школы,
    ожидание оценки и поведение ребенка вечером перед учебными днями.

    Если текст из PDF приходит «слитно»,
    chunking не должен ломаться и превращать весь документ в один блок.
    """

    chunks = chunk_document(demo_text)

    print("Chunks created:", len(chunks))
    for idx, chunk in enumerate(chunks):
        print(f"\n--- chunk {idx} ---\n{chunk}")