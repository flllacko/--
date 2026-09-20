from __future__ import annotations

import hashlib
import sqlite3
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Thread
from typing import Literal

from app.admin import record_operation
from app.config import settings
from app.db.init_db import initialize_database, utc_now

from .lance_store import LanceStoreError, has_document
from .processor import (
    ProcessingResult,
    process_extracted_file,
    process_markdown_file,
)


QueueStatus = Literal["排队中", "处理中", "处理成功", "处理失败", "已取消"]


class QueueError(RuntimeError):
    """资料处理队列无法接受或恢复任务。"""


@dataclass(frozen=True)
class QueueTaskResult:
    task_id: int
    document_id: int
    status: QueueStatus
    accepted: bool
    duplicate: bool = False
    message: str | None = None


@dataclass(frozen=True)
class _QueueItem:
    task_id: int
    document_id: int
    stable_key: str
    file_path: Path
    source_root: Path
    db_path: Path
    index_path: Path
    ollama_base_url: str
    embedding_model: str
    owner_user_id: int | None
    is_private: bool


class ProcessingQueue:
    """单 worker 的持久化状态队列。

    队列记录写入 SQLite，内存队列只负责调度；进程重启后会恢复仍处于“排队中”的
    Markdown 和已支持文件格式的公共资料。已开始的任务不会被强制中断，符合第一版需求。
    """

    def __init__(
        self,
        source_root: Path | None = None,
        db_path: Path | None = None,
        index_path: Path | None = None,
        ollama_base_url: str | None = None,
        embedding_model: str | None = None,
        *,
        is_private: bool = False,
        default_owner_user_id: int | None = None,
        autostart: bool = True,
    ) -> None:
        self.is_private = is_private
        self.default_owner_user_id = default_owner_user_id
        self.source_root = (
            source_root
            or (settings.private_source_root if is_private else settings.public_source_root)
        ).resolve()
        self.db_path = (db_path or settings.sqlite_path).resolve()
        self.index_path = (index_path or settings.lancedb_path).resolve()
        self.ollama_base_url = ollama_base_url or settings.ollama_base_url
        self.embedding_model = embedding_model or settings.ollama_embedding_model
        self._condition = Condition()
        self._pending: deque[_QueueItem] = deque()
        self._active: _QueueItem | None = None
        self._stopping = False
        self._thread: Thread | None = None

        initialize_database(self.db_path)
        self._recover_pending()
        if autostart:
            self.start()

    def start(self) -> None:
        with self._condition:
            if self._stopping:
                raise QueueError("队列已经关闭，不能重新启动")
            if self._thread and self._thread.is_alive():
                return
            worker_name = "private-knowledge-worker" if self.is_private else "public-knowledge-worker"
            self._thread = Thread(target=self._worker_loop, name=worker_name, daemon=True)
            self._thread.start()

    def close(self, wait: bool = True) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
            thread = self._thread
        if wait and thread:
            thread.join(timeout=30)

    def enqueue(
        self,
        file_path: Path,
        source_root: Path | None = None,
        db_path: Path | None = None,
        index_path: Path | None = None,
        *,
        owner_user_id: int | None = None,
    ) -> QueueTaskResult:
        with self._condition:
            if self._stopping:
                raise QueueError("队列已经关闭，不能加入新任务")
        root = (source_root or self.source_root).resolve()
        database = (db_path or self.db_path).resolve()
        vector_index = (index_path or self.index_path).resolve()
        effective_owner_user_id = owner_user_id if owner_user_id is not None else self.default_owner_user_id
        if self.is_private and effective_owner_user_id is None:
            raise QueueError("私人资料处理必须绑定所属用户。")
        try:
            path, stable_key, fallback_title, fallback_module = _queue_identity(
                file_path,
                root,
                is_private=self.is_private,
                owner_user_id=effective_owner_user_id,
            )
        except ValueError as exc:
            raise QueueError(f"资料路径不在允许的资料根目录内：{exc}") from exc

        content_hash: str | None = None
        try:
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            # 文件不存在时也保留任务，让正式处理器记录可重试的失败原因。
            pass

        initialize_database(database)
        connection = _connection(database)
        try:
            existing = connection.execute(
                """
                SELECT id, content_hash, processing_status
                FROM knowledge_documents
                WHERE stable_key = ?
                """,
                (stable_key,),
            ).fetchone()

            if existing:
                document_id = int(existing[0])
                latest = connection.execute(
                    """
                    SELECT id, status, error_message
                    FROM processing_tasks
                    WHERE document_id = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (document_id,),
                ).fetchone()
                if latest and latest[1] in ("排队中", "处理中"):
                    if latest[1] == "排队中":
                        self._restore_item(
                            task_id=int(latest[0]),
                            document_id=document_id,
                            stable_key=stable_key,
                            file_path=path,
                            source_root=root,
                            db_path=database,
                            index_path=vector_index,
                            owner_user_id=effective_owner_user_id,
                            is_private=self.is_private,
                        )
                    return QueueTaskResult(
                        task_id=int(latest[0]),
                        document_id=document_id,
                        status=latest[1],
                        accepted=False,
                        duplicate=True,
                        message="同一资料已经在队列中或正在处理",
                    )

                vector_ready = False
                if (
                    content_hash
                    and existing[1] == content_hash
                    and existing[2] == "处理成功"
                ):
                    try:
                        vector_ready = has_document(vector_index, document_id)
                    except LanceStoreError as exc:
                        raise QueueError(str(exc)) from exc
                if vector_ready and latest and latest[1] == "处理成功":
                    return QueueTaskResult(
                        task_id=int(latest[0]),
                        document_id=document_id,
                        status="处理成功",
                        accepted=False,
                        duplicate=True,
                        message="内容和向量索引均未变化，跳过重复处理",
                    )

                with connection:
                    connection.execute(
                        """
                        UPDATE knowledge_documents
                        SET processing_status = '排队中', content_hash = COALESCE(?, content_hash), updated_at = ?
                        WHERE id = ?
                        """,
                        (content_hash, utc_now(), document_id),
                    )
            else:
                now = utc_now()
                with connection:
                    cursor = connection.execute(
                        """
                        INSERT INTO knowledge_documents
                            (stable_key, title, module, source_text, source_url,
                             owner_user_id, is_private, content_status, validity_status,
                             processing_status, content_hash, version, created_at, updated_at)
                        VALUES (?, ?, ?, '', NULL, ?, ?, ?, '有效',
                                '排队中', ?, NULL, ?, ?)
                        """,
                        (
                            stable_key,
                            fallback_title,
                            fallback_module,
                            effective_owner_user_id,
                            int(self.is_private),
                            "私人资料" if self.is_private else "禁止作为依据",
                            content_hash,
                            now,
                            now,
                        ),
                    )
                    document_id = int(cursor.lastrowid)

            now = utc_now()
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO processing_tasks
                        (document_id, task_type, status, error_message, created_at, finished_at)
                    VALUES (?, ?, '排队中', NULL, ?, NULL)
                    """,
                    (
                        document_id,
                        (
                            "private_document"
                            if self.is_private
                            else ("markdown_public_document" if path.suffix.lower() == ".md" else "extracted_public_document")
                        ),
                        now,
                    ),
                )
                task_id = int(cursor.lastrowid)
        finally:
            connection.close()

        self._restore_item(
            task_id=task_id,
            document_id=document_id,
            stable_key=stable_key,
            file_path=path,
            source_root=root,
            db_path=database,
            index_path=vector_index,
            owner_user_id=effective_owner_user_id,
            is_private=self.is_private,
        )
        return QueueTaskResult(
            task_id=task_id,
            document_id=document_id,
            status="排队中",
            accepted=True,
        )

    def cancel(self, task_id: int) -> QueueTaskResult:
        pending_item: _QueueItem | None = None
        with self._condition:
            for item in self._pending:
                if item.task_id == task_id:
                    pending_item = item
                    break
            if pending_item is not None:
                self._pending.remove(pending_item)

        connection = _connection(self.db_path)
        try:
            row = connection.execute(
                """
                SELECT t.document_id, t.status
                FROM processing_tasks AS t
                WHERE t.id = ?
                """,
                (task_id,),
            ).fetchone()
            if not row:
                raise QueueError(f"找不到处理任务：{task_id}")
            document_id = int(row[0])
            status = row[1]
            if status == "已取消":
                return QueueTaskResult(task_id, document_id, "已取消", True, message="任务已经取消，可重新处理")
            if status == "处理中":
                if pending_item is not None:
                    with self._condition:
                        self._pending.appendleft(pending_item)
                return QueueTaskResult(
                    task_id,
                    document_id,
                    "处理中",
                    False,
                    message="任务已经开始，第一版不强制中断",
                )
            if status != "排队中":
                return QueueTaskResult(
                    task_id,
                    document_id,
                    status,
                    False,
                    message="任务已经结束，不能取消",
                )

            now = utc_now()
            with connection:
                connection.execute(
                    """
                    UPDATE processing_tasks
                    SET status = '已取消', error_message = '已取消，可重新处理', finished_at = ?
                    WHERE id = ? AND status = '排队中'
                    """,
                    (now, task_id),
                )
                connection.execute(
                    """
                    UPDATE knowledge_documents
                    SET processing_status = '处理失败', updated_at = ?
                    WHERE id = ?
                    """,
                    (now, document_id),
                )
            return QueueTaskResult(
                task_id,
                document_id,
                "已取消",
                True,
                message="已取消，可重新处理",
            )
        finally:
            connection.close()

    def wait_for_idle(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._pending or self._active is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    @property
    def pending_count(self) -> int:
        with self._condition:
            return len(self._pending)

    @property
    def active_task_id(self) -> int | None:
        with self._condition:
            return self._active.task_id if self._active else None

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._stopping and not self._pending:
                    self._condition.wait()
                if self._stopping:
                    return
                item = self._pending.popleft()
                self._active = item

            try:
                self._mark_processing(item)
                if item.file_path.suffix.lower() == ".md":
                    result = process_markdown_file(
                        item.file_path,
                        source_root=item.source_root,
                        db_path=item.db_path,
                        index_path=item.index_path,
                        ollama_base_url=item.ollama_base_url,
                        embedding_model=item.embedding_model,
                        processing_task_id=item.task_id,
                        stable_key_override=item.stable_key,
                        owner_user_id=item.owner_user_id,
                        is_private=item.is_private,
                    )
                else:
                    result = process_extracted_file(
                        item.file_path,
                        source_root=item.source_root,
                        db_path=item.db_path,
                        index_path=item.index_path,
                        ollama_base_url=item.ollama_base_url,
                        embedding_model=item.embedding_model,
                        processing_task_id=item.task_id,
                        stable_key_override=item.stable_key,
                        owner_user_id=item.owner_user_id,
                        is_private=item.is_private,
                    )
                self._persist_result(item, result)
            except Exception as exc:  # noqa: BLE001 - worker 必须把异常变成可重试状态
                self._mark_unexpected_failure(item, f"队列执行失败：{exc}")
            finally:
                with self._condition:
                    self._active = None
                    self._condition.notify_all()

    def _mark_processing(self, item: _QueueItem) -> None:
        connection = _connection(item.db_path)
        try:
            with connection:
                connection.execute(
                    """
                    UPDATE processing_tasks
                    SET status = '处理中', error_message = NULL, finished_at = NULL
                    WHERE id = ? AND status = '排队中'
                    """,
                    (item.task_id,),
                )
                connection.execute(
                    """
                    UPDATE knowledge_documents
                    SET processing_status = '处理中', updated_at = ?
                    WHERE id = ?
                    """,
                    (utc_now(), item.document_id),
                )
        finally:
            connection.close()

    def _persist_result(self, item: _QueueItem, result: ProcessingResult) -> None:
        connection = _connection(item.db_path)
        try:
            now = utc_now()
            with connection:
                connection.execute(
                    """
                    UPDATE processing_tasks
                    SET status = ?, error_message = ?, finished_at = ?
                    WHERE id = ?
                    """,
                    (result.status, result.error_message, now, item.task_id),
                )
                connection.execute(
                    """
                    UPDATE knowledge_documents
                    SET processing_status = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (result.status, now, item.document_id),
                )
                record_operation(
                    connection,
                    actor_user_id=None,
                    action="资料处理成功" if result.status == "处理成功" else "资料处理失败",
                    target_type="knowledge_document",
                    target_id=str(item.document_id),
                    detail={"status": result.status, "error_message": result.error_message},
                )
        finally:
            connection.close()

    def _mark_unexpected_failure(self, item: _QueueItem, message: str) -> None:
        connection = _connection(item.db_path)
        try:
            now = utc_now()
            with connection:
                connection.execute(
                    """
                    UPDATE processing_tasks
                    SET status = '处理失败', error_message = ?, finished_at = ?
                    WHERE id = ?
                    """,
                    (message, now, item.task_id),
                )
                connection.execute(
                    """
                    UPDATE knowledge_documents
                    SET processing_status = '处理失败', updated_at = ?
                    WHERE id = ?
                    """,
                    (now, item.document_id),
                )
                record_operation(
                    connection,
                    actor_user_id=None,
                    action="资料处理失败",
                    target_type="knowledge_document",
                    target_id=str(item.document_id),
                    detail={"status": "处理失败", "error_message": message},
                )
        finally:
            connection.close()

    def _recover_pending(self) -> None:
        connection = _connection(self.db_path)
        try:
            rows = connection.execute(
                """
                SELECT t.id, t.document_id, d.stable_key
                FROM processing_tasks AS t
                JOIN knowledge_documents AS d ON d.id = t.document_id
                WHERE t.status = '排队中'
                ORDER BY t.id
                """
            ).fetchall()
        finally:
            connection.close()

        for task_id, document_id, stable_key in rows:
            if self.is_private:
                parts = str(stable_key).split(":", 3)
                if len(parts) != 4 or parts[0] != "private" or parts[2] != "file":
                    continue
                try:
                    owner_user_id = int(parts[1])
                except ValueError:
                    continue
                user_root = (self.source_root / str(owner_user_id)).resolve()
                relative = Path(parts[3])
                path = (user_root / relative).resolve()
                try:
                    path.relative_to(user_root)
                except ValueError:
                    continue
                item_root = user_root
            else:
                if stable_key.startswith("public:markdown:"):
                    prefix = "public:markdown:"
                elif stable_key.startswith("public:file:"):
                    prefix = "public:file:"
                else:
                    continue
                relative = Path(stable_key.removeprefix(prefix))
                path = (self.source_root / relative).resolve()
                try:
                    path.relative_to(self.source_root)
                except ValueError:
                    continue
                owner_user_id = None
                item_root = self.source_root
            self._restore_item(
                task_id=int(task_id),
                document_id=int(document_id),
                stable_key=str(stable_key),
                file_path=path,
                source_root=item_root,
                db_path=self.db_path,
                index_path=self.index_path,
                owner_user_id=owner_user_id,
                is_private=self.is_private,
            )

    def _restore_item(
        self,
        task_id: int,
        document_id: int,
        stable_key: str,
        file_path: Path,
        source_root: Path,
        db_path: Path,
        index_path: Path,
        owner_user_id: int | None,
        is_private: bool,
    ) -> None:
        item = _QueueItem(
            task_id=task_id,
            document_id=document_id,
            stable_key=stable_key,
            file_path=file_path,
            source_root=source_root,
            db_path=db_path,
            index_path=index_path,
            ollama_base_url=self.ollama_base_url,
            embedding_model=self.embedding_model,
            owner_user_id=owner_user_id,
            is_private=is_private,
        )
        with self._condition:
            if any(existing.task_id == task_id for existing in self._pending):
                return
            if self._active and self._active.task_id == task_id:
                return
            self._pending.append(item)
            self._condition.notify_all()


def _connection(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _queue_identity(
    file_path: Path,
    source_root: Path,
    *,
    is_private: bool = False,
    owner_user_id: int | None = None,
) -> tuple[Path, str, str, str]:
    root = source_root.resolve()
    path = file_path if file_path.is_absolute() else root / file_path
    path = path.resolve()
    relative = path.relative_to(root)
    if is_private:
        if owner_user_id is None:
            raise ValueError("私人资料缺少所属用户")
        stable_key = f"private:{owner_user_id}:file:{relative.as_posix()}"
        module = "私人知识库"
    else:
        prefix = "public:markdown:" if path.suffix.lower() == ".md" else "public:file:"
        stable_key = f"{prefix}{relative.as_posix()}"
        module = relative.parts[0] if len(relative.parts) > 1 else "未分类"
    return path, stable_key, path.stem, module
