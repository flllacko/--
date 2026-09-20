from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch

from app.processing.processor import ProcessingResult
from app.processing.lance_store import has_document
from app.processing.queue import ProcessingQueue


def _write_source(root: Path, name: str) -> Path:
    path = root / "学生资助" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {name}\n\n测试正文。", encoding="utf-8")
    return path


class ProcessingQueueTest(unittest.TestCase):
    def test_queue_dispatches_non_markdown_file_to_extractor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "资料原件"
            source = root / "学生资助" / "queued.txt"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("国家助学金按照公开政策分为三档。", encoding="utf-8")
            database = base / "app.db"
            index = base / "index"

            with patch(
                "app.processing.processor.embed_texts",
                side_effect=lambda texts, **_: [[0.1, 0.2, 0.3] for _ in texts],
            ):
                queue = ProcessingQueue(root, database, index)
                try:
                    task = queue.enqueue(source)
                    self.assertTrue(queue.wait_for_idle(timeout=10))
                finally:
                    queue.close()

            connection = sqlite3.connect(database)
            try:
                task_status = connection.execute(
                    "SELECT status FROM processing_tasks WHERE id = ?", (task.task_id,)
                ).fetchone()[0]
                chunk_count = connection.execute(
                    "SELECT COUNT(*) FROM knowledge_chunks WHERE document_id = ?",
                    (task.document_id,),
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(task_status, "处理成功")
            self.assertEqual(chunk_count, 1)

    def test_unsupported_file_becomes_retryable_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "资料原件"
            source = root / "学生资助" / "unsupported.doc"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"legacy binary")
            database = base / "app.db"
            index = base / "index"

            queue = ProcessingQueue(root, database, index)
            try:
                task = queue.enqueue(source)
                self.assertTrue(queue.wait_for_idle(timeout=10))
            finally:
                queue.close()

            connection = sqlite3.connect(database)
            try:
                task_row = connection.execute(
                    "SELECT status, error_message FROM processing_tasks WHERE id = ?",
                    (task.task_id,),
                ).fetchone()
                document_count = connection.execute(
                    "SELECT COUNT(*) FROM knowledge_documents"
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(task_row[0], "处理失败")
            self.assertIn("旧式 .doc", task_row[1])
            self.assertEqual(document_count, 1)

    def test_queue_runs_real_markdown_processor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "资料正文"
            source = root / "学生资助" / "queued.md"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(
                """# 队列测试政策

## 资料信息

- 资料编号：TEST-QUEUE-001
- 所属模块：学生资助
- 内容状态：可直接回答
- 时效状态：有效
- 来源文字：测试官方来源
- 原文链接：[官方页面](https://example.com/queued)
- 发布日期：2026-09-17

## 官方正文

国家助学金按照公开政策分为三档。
""",
                encoding="utf-8",
            )
            database = base / "app.db"
            index = base / "index"

            with patch("app.processing.processor.embed_texts", return_value=[[0.1, 0.2, 0.3]]):
                queue = ProcessingQueue(root, database, index)
                try:
                    task = queue.enqueue(source)
                    self.assertTrue(queue.wait_for_idle(timeout=10))
                finally:
                    queue.close()

            connection = sqlite3.connect(database)
            try:
                status = connection.execute(
                    "SELECT status FROM processing_tasks WHERE id = ?", (task.task_id,)
                ).fetchone()[0]
                chunk_count = connection.execute(
                    "SELECT COUNT(*) FROM knowledge_chunks WHERE document_id = ?",
                    (task.document_id,),
                ).fetchone()[0]
            finally:
                connection.close()

            self.assertEqual(status, "处理成功")
            self.assertEqual(chunk_count, 1)
            self.assertTrue(has_document(index, task.document_id))

    def test_fifo_and_pending_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "资料正文"
            first_path = _write_source(root, "first.md")
            second_path = _write_source(root, "second.md")
            database = base / "app.db"
            index = base / "index"
            calls: list[str] = []

            def fake_process(path: Path, **_: object) -> ProcessingResult:
                calls.append(path.name)
                return ProcessingResult(
                    stable_key=f"public:markdown:学生资助/{path.name}",
                    document_id=None,
                    status="处理成功",
                    chunk_count=1,
                )

            with patch("app.processing.queue.process_markdown_file", side_effect=fake_process):
                queue = ProcessingQueue(root, database, index, autostart=False)
                try:
                    first = queue.enqueue(first_path)
                    second = queue.enqueue(second_path)
                    cancelled = queue.cancel(second.task_id)
                    self.assertTrue(cancelled.accepted)
                    self.assertEqual(cancelled.status, "已取消")
                    self.assertEqual(cancelled.message, "已取消，可重新处理")

                    queue.start()
                    self.assertTrue(queue.wait_for_idle(timeout=5))
                finally:
                    queue.close()

            self.assertEqual(calls, [first_path.name])
            connection = sqlite3.connect(database)
            try:
                first_status = connection.execute(
                    "SELECT status FROM processing_tasks WHERE id = ?", (first.task_id,)
                ).fetchone()[0]
                second_status = connection.execute(
                    "SELECT status FROM processing_tasks WHERE id = ?", (second.task_id,)
                ).fetchone()[0]
                second_document_status = connection.execute(
                    "SELECT processing_status FROM knowledge_documents WHERE id = ?",
                    (second.document_id,),
                ).fetchone()[0]
            finally:
                connection.close()

            self.assertEqual(first_status, "处理成功")
            self.assertEqual(second_status, "已取消")
            self.assertEqual(second_document_status, "处理失败")

    def test_started_task_is_not_force_cancelled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "资料正文"
            source = _write_source(root, "running.md")
            database = base / "app.db"
            index = base / "index"
            started = Event()
            release = Event()

            def fake_process(path: Path, **_: object) -> ProcessingResult:
                started.set()
                self.assertTrue(release.wait(timeout=5))
                return ProcessingResult(
                    stable_key=f"public:markdown:学生资助/{path.name}",
                    document_id=None,
                    status="处理成功",
                    chunk_count=1,
                )

            with patch("app.processing.queue.process_markdown_file", side_effect=fake_process):
                queue = ProcessingQueue(root, database, index)
                try:
                    task = queue.enqueue(source)
                    self.assertTrue(started.wait(timeout=5))
                    cancelled = queue.cancel(task.task_id)
                    self.assertFalse(cancelled.accepted)
                    self.assertEqual(cancelled.status, "处理中")
                    release.set()
                    self.assertTrue(queue.wait_for_idle(timeout=5))
                finally:
                    release.set()
                    queue.close()

            self.assertEqual(queue.active_task_id, None)


if __name__ == "__main__":
    unittest.main()
