from __future__ import annotations

from pathlib import Path
from typing import Sequence

import lancedb

from .markdown import ParsedMarkdown, TextChunk


TABLE_NAME = "knowledge_chunks"


class LanceStoreError(RuntimeError):
    """LanceDB 检索副本写入或查询失败。"""


def _database(index_path: Path):
    index_path.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(index_path))


def _has_table(database) -> bool:
    try:
        response = database.list_tables()
        return TABLE_NAME in response.tables
    except Exception as exc:
        raise LanceStoreError(f"读取 LanceDB 表目录失败：{exc}") from exc


def has_document(index_path: Path, document_id: int) -> bool:
    database = _database(index_path)
    if not _has_table(database):
        return False
    table = database.open_table(TABLE_NAME)
    try:
        rows = table.search().where(
            f"document_id = {int(document_id)}",
            prefilter=True,
        ).limit(1).to_list()
    except Exception as exc:
        raise LanceStoreError(f"检查 LanceDB 资料失败：{exc}") from exc
    return bool(rows)


def remove_document(index_path: Path, document_id: int) -> None:
    database = _database(index_path)
    if not _has_table(database):
        return
    table = database.open_table(TABLE_NAME)
    try:
        table.delete(f"document_id = {int(document_id)}")
    except Exception as exc:
        raise LanceStoreError(f"清理 LanceDB 资料失败：{exc}") from exc


def sync_chunks(
    index_path: Path,
    document_id: int,
    stable_key: str,
    parsed: ParsedMarkdown,
    chunks: Sequence[TextChunk],
    embeddings: Sequence[Sequence[float]],
    embedding_model: str,
) -> int:
    if len(chunks) != len(embeddings):
        raise LanceStoreError("切片数量和嵌入数量不一致")
    if not chunks:
        raise LanceStoreError("没有可写入 LanceDB 的切片")

    rows = [
        {
            "chunk_key": f"{stable_key}#{index}",
            "document_id": int(document_id),
            "chunk_index": index,
            "text": chunk.content,
            "source_locator": chunk.locator,
            "stable_key": stable_key,
            "title": parsed.title,
            "module": parsed.module,
            "content_status": parsed.content_status,
            "validity_status": parsed.validity_status,
            "source_label": parsed.source_label,
            "source_url": parsed.source_url or "",
            "embedding_model": embedding_model,
            "vector": list(vector),
        }
        for index, (chunk, vector) in enumerate(zip(chunks, embeddings))
    ]

    database = _database(index_path)
    try:
        if _has_table(database):
            table = database.open_table(TABLE_NAME)
            table.delete(f"document_id = {int(document_id)}")
            table.add(rows)
        else:
            database.create_table(TABLE_NAME, data=rows)
    except Exception as exc:
        raise LanceStoreError(f"写入 LanceDB 失败：{exc}") from exc
    return len(rows)


def vector_search(index_path: Path, query_vector: Sequence[float], limit: int = 5) -> list[dict]:
    database = _database(index_path)
    if not _has_table(database):
        return []
    table = database.open_table(TABLE_NAME)
    try:
        return table.search(list(query_vector), vector_column_name="vector").limit(limit).to_list()
    except Exception as exc:
        raise LanceStoreError(f"LanceDB 向量检索失败：{exc}") from exc
