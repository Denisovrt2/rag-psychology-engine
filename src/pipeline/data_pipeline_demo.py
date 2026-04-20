from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DocumentRecord:
    source_id: str
    source_type: str
    language: str
    raw_text: str


@dataclass
class ProcessedDocument:
    source_id: str
    cleaned_text: str
    translated_text: str
    chunks: List[str]
    embedding_ready: bool


def extract_text_from_source(record: DocumentRecord) -> str:
    """
    Упрощённая публичная demo-функция.

    В рабочем проекте на этом этапе могут использоваться:
    - парсинг исходного документа
    - OCR для сложных PDF / сканов
    - fallback-логика для шумных источников
    """
    return record.raw_text.strip()


def clean_text(text: str) -> str:
    """
    Базовая нормализация текста перед дальнейшим pipeline:
    - удаление лишних пробелов
    - выравнивание переносов
    - подготовка к переводу / чанкингу
    """
    return " ".join(text.replace("\n", " ").split())


def translate_text_if_needed(text: str, language: str) -> str:
    """
    В публичной demo-версии перевод упрощён.

    В рабочем пайплайне этот слой отвечает за:
    - API-перевод части корпуса
    - унификацию языкового слоя
    - подготовку единой retrieval-среды
    """
    if language.lower() in {"ru", "russian"}:
        return text
    return f"[translated] {text}"


def chunk_text(text: str, chunk_size: int = 120) -> List[str]:
    """
    Простая demo-разбивка на chunks.

    В production-логике чанкинг — один из критичных слоёв качества:
    именно он влияет на retrieval, шум и качество evidence.
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    current: List[str] = []

    for word in words:
        current.append(word)
        if len(current) >= chunk_size:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks


def prepare_document_for_vectorization(record: DocumentRecord) -> ProcessedDocument:
    """
    Полный demo-pipeline подготовки документа:

    source -> extraction -> cleaning -> translation -> chunking -> embedding-ready output
    """
    extracted = extract_text_from_source(record)
    cleaned = clean_text(extracted)
    translated = translate_text_if_needed(cleaned, record.language)
    chunks = chunk_text(translated)

    return ProcessedDocument(
        source_id=record.source_id,
        cleaned_text=cleaned,
        translated_text=translated,
        chunks=chunks,
        embedding_ready=len(chunks) > 0,
    )


def build_pipeline_summary(processed: ProcessedDocument) -> Dict[str, object]:
    """
    Публичное summary для GitHub demo:
    показывает, что система мыслит pipeline-стадиями, а не 'магией AI'.
    """
    return {
        "source_id": processed.source_id,
        "stages": [
            "text_extraction_or_ocr",
            "cleaning",
            "translation_if_needed",
            "chunking",
            "vectorization_ready",
        ],
        "chunk_count": len(processed.chunks),
        "embedding_ready": processed.embedding_ready,
        "product_meaning": (
            "Качество downstream RAG-системы определяется не только моделью, "
            "но и качеством ingestion pipeline: OCR, перевод, нормализация, чанкинг."
        ),
    }


if __name__ == "__main__":
    demo_record = DocumentRecord(
        source_id="book_001",
        source_type="pdf",
        language="ru",
        raw_text="""
        Ребенок может испытывать устойчивую тревогу в ситуации,
        когда ожидаемая учебная нагрузка воспринимается как источник
        давления, оценки или эмоциональной нестабильности.
        """,
    )

    processed = prepare_document_for_vectorization(demo_record)
    summary = build_pipeline_summary(processed)

    print(summary)