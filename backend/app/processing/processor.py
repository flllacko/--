from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from app.config import settings
from app.db.init_db import initialize_database, utc_now

from .lance_store import LanceStoreError, has_document, remove_document, sync_chunks
from .file_extractors import ExtractedDocument, FileExtractionError, extract_file
from .markdown import (
    MarkdownProcessingError,
    ParsedMarkdown,
    TextChunk,
    chunk_markdown,
    metadata_value,
    parse_markdown,
)
from .ollama_embeddings import EmbeddingError, embed_texts


ProcessingStatus = Literal["处理成功", "处理失败"]


@dataclass(frozen=True)
class ProcessingResult:
    stable_key: str
    document_id: int | None
    status: ProcessingStatus
    chunk_count: int
    error_message: str | None = None
    duplicate: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _connection(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _input_identity(file_path: Path, source_root: Path) -> tuple[Path, str, str, str]:
    root = source_root.resolve()
    path = file_path if file_path.is_absolute() else root / file_path
    path = path.resolve()
    relative_path = path.relative_to(root)
    stable_key = f"public:markdown:{relative_path.as_posix()}"
    title = path.stem
    module = relative_path.parts[0] if len(relative_path.parts) > 1 else "未分类"
    return path, stable_key, title, module


def _failure(
    connection: sqlite3.Connection,
    stable_key: str,
    title: str,
    module: str,
    error_message: str,
    content_hash: str | None,
    processing_task_id: int | None = None,
    *,
    owner_user_id: int | None = None,
    is_private: bool = False,
) -> ProcessingResult:
    now = utc_now()
    with connection:
        existing = connection.execute(
            "SELECT id FROM knowledge_documents WHERE stable_key = ?",
            (stable_key,),
        ).fetchone()
        if existing:
            document_id = int(existing[0])
            connection.execute(
                """
                UPDATE knowledge_documents
                SET processing_status = '处理失败',
                    content_hash = COALESCE(?, content_hash),
                    owner_user_id = ?, is_private = ?,
                    content_status = CASE WHEN ? = 1 THEN '私人资料' ELSE content_status END,
                    updated_at = ?
                WHERE id = ?
                """,
                (content_hash, owner_user_id, int(is_private), int(is_private), now, document_id),
            )
        else:
            cursor = connection.execute(
                """
                INSERT INTO knowledge_documents
                    (stable_key, title, module, source_text, source_url,
                     owner_user_id, is_private, content_status, validity_status,
                     processing_status, content_hash, version, created_at, updated_at)
                VALUES (?, ?, ?, '', NULL, ?, ?, ?, '有效',
                        '处理失败', ?, NULL, ?, ?)
                """,
                (
                    stable_key,
                    title,
                    module,
                    owner_user_id,
                    int(is_private),
                    "私人资料" if is_private else "禁止作为依据",
                    content_hash,
                    now,
                    now,
                ),
            )
            document_id = int(cursor.lastrowid)

        if processing_task_id is None:
            connection.execute(
                """
                INSERT INTO processing_tasks
                    (document_id, task_type, status, error_message, created_at, finished_at)
                VALUES (?, ?, '处理失败', ?, ?, ?)
                """,
                (
                    document_id,
                    "private_document" if is_private else "markdown_public_document",
                    error_message,
                    now,
                    now,
                ),
            )
        else:
            updated = connection.execute(
                """
                UPDATE processing_tasks
                SET status = '处理失败', error_message = ?, finished_at = ?
                WHERE id = ? AND document_id = ?
                """,
                (error_message, now, processing_task_id, document_id),
            ).rowcount
            if updated != 1:
                raise sqlite3.Error("队列任务不存在或不属于当前资料")

    return ProcessingResult(
        stable_key=stable_key,
        document_id=document_id,
        status="处理失败",
        chunk_count=0,
        error_message=error_message,
    )


def _success(
    connection: sqlite3.Connection,
    stable_key: str,
    parsed: ParsedMarkdown,
    content_hash: str,
    chunks: list[TextChunk],
    processing_task_id: int | None = None,
    version_group_key: str | None = None,
    supersedes_document_id: int | None = None,
    *,
    owner_user_id: int | None = None,
    is_private: bool = False,
) -> ProcessingResult:
    now = utc_now()
    published_date = metadata_value(parsed.metadata, "发布日期", "发布时间", "发文日期")
    version = published_date or now
    publisher_department = metadata_value(
        parsed.metadata,
        "发布部门",
        "发布单位",
        "发布机构",
        "来源部门",
    )
    with connection:
        existing = connection.execute(
            "SELECT id FROM knowledge_documents WHERE stable_key = ?",
            (stable_key,),
        ).fetchone()
        if existing:
            document_id = int(existing[0])
            connection.execute(
                """
                UPDATE knowledge_documents
                SET title = ?, module = ?, source_text = ?, source_label = ?, source_url = ?,
                    published_date = ?, publisher_department = ?,
                    owner_user_id = ?, is_private = ?,
                    content_status = ?, validity_status = ?,
                    processing_status = '处理中', content_hash = ?,
                    version = ?, version_group_key = ?, supersedes_document_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    parsed.title,
                    parsed.module,
                    parsed.source_text,
                    parsed.source_label,
                    parsed.source_url,
                    published_date,
                    publisher_department,
                    owner_user_id,
                    int(is_private),
                    parsed.content_status,
                    parsed.validity_status,
                    content_hash,
                    version,
                    version_group_key,
                    supersedes_document_id,
                    now,
                    document_id,
                ),
            )
            connection.execute("DELETE FROM knowledge_chunks WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM source_records WHERE document_id = ?", (document_id,))
        else:
            cursor = connection.execute(
                """
                INSERT INTO knowledge_documents
                    (stable_key, title, module, source_text, source_label, source_url,
                     published_date, publisher_department,
                     owner_user_id, is_private, content_status, validity_status,
                     processing_status, content_hash, version, version_group_key,
                     supersedes_document_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '处理中', ?, ?, ?, ?, ?, ?)
                """,
                (
                    stable_key,
                    parsed.title,
                    parsed.module,
                    parsed.source_text,
                    parsed.source_label,
                    parsed.source_url,
                    published_date,
                    publisher_department,
                    owner_user_id,
                    int(is_private),
                    parsed.content_status,
                    parsed.validity_status,
                    content_hash,
                    version,
                    version_group_key,
                    supersedes_document_id,
                    now,
                    now,
                ),
            )
            document_id = int(cursor.lastrowid)

        if processing_task_id is None:
            task_cursor = connection.execute(
                """
                INSERT INTO processing_tasks
                    (document_id, task_type, status, error_message, created_at, finished_at)
                VALUES (?, ?, '处理中', NULL, ?, NULL)
                """,
                (
                    document_id,
                    "private_document" if is_private else "markdown_public_document",
                    now,
                ),
            )
            task_id = int(task_cursor.lastrowid)
        else:
            task_row = connection.execute(
                "SELECT id FROM processing_tasks WHERE id = ? AND document_id = ?",
                (processing_task_id, document_id),
            ).fetchone()
            if not task_row:
                raise sqlite3.Error("队列任务不存在或不属于当前资料")
            task_id = int(task_row[0])
            connection.execute(
                """
                UPDATE processing_tasks
                SET status = '处理中', error_message = NULL, finished_at = NULL
                WHERE id = ?
                """,
                (task_id,),
            )

        connection.executemany(
            """
            INSERT INTO knowledge_chunks
                (document_id, chunk_index, content, source_locator, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (document_id, index, chunk.content, chunk.locator, now)
                for index, chunk in enumerate(chunks)
            ],
        )
        connection.execute(
            """
            INSERT INTO source_records
                (document_id, label, url, locator, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (document_id, parsed.source_label, parsed.source_url, "原文链接", now),
        )
        connection.execute(
            """
            UPDATE processing_tasks
            SET status = '处理成功', finished_at = ?
            WHERE id = ?
            """,
            (now, task_id),
        )
        connection.execute(
            """
            UPDATE knowledge_documents
            SET processing_status = '处理成功', updated_at = ?
            WHERE id = ?
            """,
            (now, document_id),
        )

    return ProcessingResult(
        stable_key=stable_key,
        document_id=document_id,
        status="处理成功",
        chunk_count=len(chunks),
    )


def _mark_vector_failure(
    connection: sqlite3.Connection,
    document_id: int,
    error_message: str,
) -> None:
    now = utc_now()
    with connection:
        task = connection.execute(
            """
            SELECT id FROM processing_tasks
            WHERE document_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (document_id,),
        ).fetchone()
        connection.execute("DELETE FROM knowledge_chunks WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM source_records WHERE document_id = ?", (document_id,))
        connection.execute(
            """
            UPDATE knowledge_documents
            SET processing_status = '处理失败', updated_at = ?
            WHERE id = ?
            """,
            (now, document_id),
        )
        if task:
            connection.execute(
                """
                UPDATE processing_tasks
                SET status = '处理失败', error_message = ?, finished_at = ?
                WHERE id = ?
                """,
                (error_message, now, task[0]),
            )


def _claim_processing_task(
    connection: sqlite3.Connection,
    processing_task_id: int,
    stable_key: str,
) -> None:
    now = utc_now()
    with connection:
        row = connection.execute(
            """
            SELECT t.document_id, d.stable_key
            FROM processing_tasks AS t
            JOIN knowledge_documents AS d ON d.id = t.document_id
            WHERE t.id = ?
            """,
            (processing_task_id,),
        ).fetchone()
        if not row or row[1] != stable_key:
            raise sqlite3.Error("队列任务不存在或资料标识不一致")
        connection.execute(
            """
            UPDATE processing_tasks
            SET status = '处理中', error_message = NULL, finished_at = NULL
            WHERE id = ?
            """,
            (processing_task_id,),
        )
        connection.execute(
            """
            UPDATE knowledge_documents
            SET processing_status = '处理中', updated_at = ?
            WHERE id = ?
            """,
            (now, row[0]),
        )


def process_markdown_file(
    file_path: Path,
    source_root: Path | None = None,
    db_path: Path | None = None,
    index_path: Path | None = None,
    ollama_base_url: str | None = None,
    embedding_model: str | None = None,
    processing_task_id: int | None = None,
    stable_key_override: str | None = None,
    version_group_key: str | None = None,
    supersedes_document_id: int | None = None,
    owner_user_id: int | None = None,
    is_private: bool = False,
) -> ProcessingResult:
    """处理一份 Markdown 公共资料，成功或失败都会留下可追溯状态。"""

    root = (source_root or settings.public_source_root).resolve()
    vector_index = index_path or settings.lancedb_path
    embedding_base_url = ollama_base_url or settings.ollama_base_url
    embedding_model_name = embedding_model or settings.ollama_embedding_model
    try:
        path, derived_stable_key, fallback_title, fallback_module = _input_identity(file_path, root)
        stable_key = stable_key_override or derived_stable_key
    except ValueError as exc:
        stable_key = stable_key_override or (
            f"private:{owner_user_id}:file:invalid:{Path(file_path).as_posix()}"
            if is_private
            else f"public:markdown:invalid:{Path(file_path).as_posix()}"
        )
        database = db_path or settings.sqlite_path
        initialize_database(database)
        connection = _connection(database)
        try:
            return _failure(
                connection,
                stable_key,
                Path(file_path).stem,
                "私人知识库" if is_private else "未分类",
                str(exc),
                None,
                owner_user_id=owner_user_id,
                is_private=is_private,
            )
        finally:
            connection.close()

    content_hash: str | None = None
    try:
        raw_bytes = path.read_bytes()
        content_hash = hashlib.sha256(raw_bytes).hexdigest()
    except OSError as exc:
        database = db_path or settings.sqlite_path
        initialize_database(database)
        connection = _connection(database)
        try:
            return _failure(
                connection,
                stable_key,
                fallback_title,
                fallback_module,
                f"读取资料失败：{exc}",
                None,
                processing_task_id,
                owner_user_id=owner_user_id,
                is_private=is_private,
            )
        finally:
            connection.close()

    database = db_path or settings.sqlite_path
    initialize_database(database)
    connection = _connection(database)
    try:
        if processing_task_id is not None:
            _claim_processing_task(connection, processing_task_id, stable_key)
        existing = connection.execute(
            """
            SELECT id, content_hash, processing_status
            FROM knowledge_documents
            WHERE stable_key = ?
            """,
            (stable_key,),
        ).fetchone()
        vector_ready = bool(existing and has_document(vector_index, int(existing[0])))
        if (
            existing
            and content_hash
            and existing[1] == content_hash
            and existing[2] == "处理成功"
            and vector_ready
        ):
            chunk_count = connection.execute(
                "SELECT COUNT(*) FROM knowledge_chunks WHERE document_id = ?",
                (existing[0],),
            ).fetchone()[0]
            return ProcessingResult(
                stable_key=stable_key,
                document_id=int(existing[0]),
                status="处理成功",
                chunk_count=int(chunk_count),
                duplicate=True,
            )

        if is_private:
            duplicate_row = connection.execute(
                """
                SELECT id, stable_key
                FROM knowledge_documents
                WHERE content_hash = ?
                  AND is_private = 1
                  AND owner_user_id = ?
                  AND processing_status = '处理成功'
                  AND id != COALESCE(?, -1)
                ORDER BY id
                LIMIT 1
                """,
                (content_hash, owner_user_id, existing[0] if existing else None),
            ).fetchone()
        else:
            duplicate_row = connection.execute(
                """
                SELECT id, stable_key
                FROM knowledge_documents
                WHERE content_hash = ?
                  AND is_private = 0
                  AND owner_user_id IS NULL
                  AND processing_status = '处理成功'
                  AND id != COALESCE(?, -1)
                ORDER BY id
                LIMIT 1
                """,
                (content_hash, existing[0] if existing else None),
            ).fetchone()
        if duplicate_row and not is_private:
            duplicate_ready = has_document(vector_index, int(duplicate_row[0]))
            if duplicate_ready:
                chunk_count = connection.execute(
                    "SELECT COUNT(*) FROM knowledge_chunks WHERE document_id = ?",
                    (duplicate_row[0],),
                ).fetchone()[0]
                return ProcessingResult(
                    stable_key=stable_key,
                    document_id=int(duplicate_row[0]),
                    status="处理成功",
                    chunk_count=int(chunk_count),
                    duplicate=True,
                )

        try:
            parsed = parse_markdown(path, root)
            chunks = chunk_markdown(parsed.source_text)
            if not chunks:
                raise MarkdownProcessingError("正文切片结果为空")
        except (MarkdownProcessingError, OSError, ValueError) as exc:
            return _failure(
                connection,
                stable_key,
                fallback_title,
                fallback_module,
                str(exc),
                content_hash,
                processing_task_id,
                owner_user_id=owner_user_id,
                is_private=is_private,
            )

        try:
            embeddings = embed_texts(
                [chunk.content for chunk in chunks],
                base_url=embedding_base_url,
                model=embedding_model_name,
            )
        except EmbeddingError as exc:
            return _failure(
                connection,
                stable_key,
                parsed.title,
                parsed.module,
                str(exc),
                content_hash,
                processing_task_id,
                owner_user_id=owner_user_id,
                is_private=is_private,
            )

        try:
            result = _success(
                connection,
                stable_key,
                parsed,
                content_hash,
                chunks,
                processing_task_id,
                version_group_key,
                supersedes_document_id,
                owner_user_id=owner_user_id,
                is_private=is_private,
            )
            sync_chunks(
                vector_index,
                int(result.document_id),
                stable_key,
                parsed,
                chunks,
                embeddings,
                embedding_model_name,
            )
            return result
        except LanceStoreError as exc:
            document_id = result.document_id if "result" in locals() and result.document_id else None
            if document_id is not None:
                try:
                    remove_document(vector_index, int(document_id))
                except LanceStoreError:
                    pass
                _mark_vector_failure(connection, int(document_id), str(exc))
                return ProcessingResult(
                    stable_key=stable_key,
                    document_id=int(document_id),
                    status="处理失败",
                    chunk_count=0,
                    error_message=str(exc),
                )
            return _failure(
                connection,
                stable_key,
                parsed.title,
                parsed.module,
                str(exc),
                content_hash,
                processing_task_id,
                owner_user_id=owner_user_id,
                is_private=is_private,
            )
        except sqlite3.Error as exc:
            connection.rollback()
            return _failure(
                connection,
                stable_key,
                parsed.title,
                parsed.module,
                f"写入数据库失败：{exc}",
                content_hash,
                processing_task_id,
                owner_user_id=owner_user_id,
                is_private=is_private,
            )
    finally:
        connection.close()


def process_extracted_file(
    file_path: Path,
    *,
    source_root: Path,
    db_path: Path | None = None,
    index_path: Path | None = None,
    ollama_base_url: str | None = None,
    embedding_model: str | None = None,
    module: str | None = None,
    source_label: str | None = None,
    source_url: str | None = None,
    content_status: str = "禁止作为依据",
    validity_status: str = "有效",
    processing_task_id: int | None = None,
    stable_key_override: str | None = None,
    owner_user_id: int | None = None,
    is_private: bool = False,
) -> ProcessingResult:
    """处理 PDF/DOCX/TXT/Excel/HTML 等已抽取文件，复用同一向量入库边界。"""

    root = source_root.resolve()
    path = file_path.resolve() if file_path.is_absolute() else (root / file_path).resolve()
    try:
        relative = path.relative_to(root)
        extracted = extract_file(path)
    except (ValueError, FileExtractionError) as exc:
        database = db_path or settings.sqlite_path
        initialize_database(database)
        connection = _connection(database)
        try:
            relative = path.relative_to(root)
            stable_key = stable_key_override or (
                f"private:{owner_user_id}:file:{relative.as_posix()}"
                if is_private
                else f"public:file:{relative.as_posix()}"
            )
        except ValueError:
            stable_key = stable_key_override or (
                f"private:{owner_user_id}:file:invalid:{path.as_posix()}"
                if is_private
                else f"public:file:invalid:{path.as_posix()}"
            )
        try:
            return _failure(
                connection,
                stable_key,
                path.stem,
                "未分类",
                str(exc),
                None,
                processing_task_id,
                owner_user_id=owner_user_id,
                is_private=is_private,
            )
        finally:
            connection.close()

    stable_key = stable_key_override or (
        f"private:{owner_user_id}:file:{relative.as_posix()}"
        if is_private
        else f"public:file:{relative.as_posix()}"
    )
    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    vector_index = index_path or settings.lancedb_path
    database = db_path or settings.sqlite_path
    parsed = ParsedMarkdown(
        title=extracted.title or path.stem,
        module=module or ("私人知识库" if is_private else (relative.parts[0] if len(relative.parts) > 1 else "未分类")),
        source_text=extracted.text,
        source_url=source_url,
        source_label=source_label or path.name,
        content_status="私人资料" if is_private else content_status,
        validity_status=validity_status,
        metadata={"资料类型": extracted.media_type},
    )
    initialize_database(database)
    connection = _connection(database)
    try:
        if processing_task_id is not None:
            _claim_processing_task(connection, processing_task_id, stable_key)
        existing = connection.execute(
            """
            SELECT id, content_hash, processing_status
            FROM knowledge_documents WHERE stable_key = ?
            """,
            (stable_key,),
        ).fetchone()
        vector_ready = bool(existing and has_document(vector_index, int(existing[0])))
        if (
            existing
            and existing[1] == content_hash
            and existing[2] == "处理成功"
            and vector_ready
        ):
            chunk_count = connection.execute(
                "SELECT COUNT(*) FROM knowledge_chunks WHERE document_id = ?",
                (existing[0],),
            ).fetchone()[0]
            return ProcessingResult(
                stable_key=stable_key,
                document_id=int(existing[0]),
                status="处理成功",
                chunk_count=int(chunk_count),
                duplicate=True,
            )

        if is_private:
            duplicate_row = connection.execute(
                """
                SELECT id FROM knowledge_documents
                WHERE content_hash = ?
                  AND is_private = 1
                  AND owner_user_id = ?
                  AND processing_status = '处理成功'
                  AND id != COALESCE(?, -1)
                ORDER BY id LIMIT 1
                """,
                (content_hash, owner_user_id, existing[0] if existing else None),
            ).fetchone()
        else:
            duplicate_row = connection.execute(
                """
                SELECT id FROM knowledge_documents
                WHERE content_hash = ?
                  AND is_private = 0
                  AND owner_user_id IS NULL
                  AND processing_status = '处理成功'
                  AND id != COALESCE(?, -1)
                ORDER BY id LIMIT 1
                """,
                (content_hash, existing[0] if existing else None),
            ).fetchone()
        if duplicate_row and not is_private and has_document(vector_index, int(duplicate_row[0])):
            chunk_count = connection.execute(
                "SELECT COUNT(*) FROM knowledge_chunks WHERE document_id = ?",
                (duplicate_row[0],),
            ).fetchone()[0]
            return ProcessingResult(
                stable_key=stable_key,
                document_id=int(duplicate_row[0]),
                status="处理成功",
                chunk_count=int(chunk_count),
                duplicate=True,
            )

        chunks = chunk_markdown(extracted.text)
        if not chunks:
            return _failure(
                connection,
                stable_key,
                parsed.title,
                parsed.module,
                "抽取正文切片结果为空",
                content_hash,
                processing_task_id,
                owner_user_id=owner_user_id,
                is_private=is_private,
            )
        try:
            embeddings = embed_texts(
                [chunk.content for chunk in chunks],
                base_url=ollama_base_url or settings.ollama_base_url,
                model=embedding_model or settings.ollama_embedding_model,
            )
        except EmbeddingError as exc:
            return _failure(
                connection,
                stable_key,
                parsed.title,
                parsed.module,
                str(exc),
                content_hash,
                processing_task_id,
                owner_user_id=owner_user_id,
                is_private=is_private,
            )

        try:
            result = _success(
                connection,
                stable_key,
                parsed,
                content_hash,
                chunks,
                processing_task_id,
                owner_user_id=owner_user_id,
                is_private=is_private,
            )
            sync_chunks(
                vector_index,
                int(result.document_id),
                stable_key,
                parsed,
                chunks,
                embeddings,
                embedding_model or settings.ollama_embedding_model,
            )
            return result
        except LanceStoreError as exc:
            document_id = result.document_id if "result" in locals() and result.document_id else None
            if document_id is not None:
                try:
                    remove_document(vector_index, int(document_id))
                except LanceStoreError:
                    pass
                _mark_vector_failure(connection, int(document_id), str(exc))
                return ProcessingResult(
                    stable_key=stable_key,
                    document_id=int(document_id),
                    status="处理失败",
                    chunk_count=0,
                    error_message=str(exc),
                )
            return _failure(
                connection,
                stable_key,
                parsed.title,
                parsed.module,
                str(exc),
                content_hash,
                processing_task_id,
                owner_user_id=owner_user_id,
                is_private=is_private,
            )
        except sqlite3.Error as exc:
            connection.rollback()
            return _failure(
                connection,
                stable_key,
                parsed.title,
                parsed.module,
                f"写入数据库失败：{exc}",
                content_hash,
                processing_task_id,
                owner_user_id=owner_user_id,
                is_private=is_private,
            )
    finally:
        connection.close()
