from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.auth import register_user
from app.db.init_db import initialize_database
from app.private_knowledge import PrivateKnowledgeService
from app.processing.knowledge_scope import private_document_ids_for_user
from app.processing.lance_store import has_document
from app.processing.queue import ProcessingQueue
from app.qa.retrieval import search_knowledge


class PrivateKnowledgeTests(unittest.TestCase):
    def test_owner_isolation_and_private_duplicate_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            database = base / "app.db"
            index = base / "index"
            private_root = base / "private"
            initialize_database(database)
            first_user = register_user(database, "私人甲", "abcdef", "abcdef")
            second_user = register_user(database, "私人乙", "abcdef", "abcdef")
            first_path = private_root / str(first_user["id"]) / "个人说明.txt"
            second_path = private_root / str(second_user["id"]) / "个人说明.txt"
            first_path.parent.mkdir(parents=True)
            second_path.parent.mkdir(parents=True)
            first_path.write_text("这是甲用户的私人资料，学号是 A-001。", encoding="utf-8")
            second_path.write_text("这是甲用户的私人资料，学号是 A-001。", encoding="utf-8")

            with patch(
                "app.processing.processor.embed_texts",
                side_effect=lambda texts, **_: [[0.1, 0.2, 0.3] for _ in texts],
            ):
                queue = ProcessingQueue(private_root, database, index, is_private=True)
                try:
                    first = queue.enqueue(
                        first_path,
                        source_root=first_path.parent,
                        owner_user_id=int(first_user["id"]),
                    )
                    second = queue.enqueue(
                        second_path,
                        source_root=second_path.parent,
                        owner_user_id=int(second_user["id"]),
                    )
                    self.assertTrue(queue.wait_for_idle(timeout=10))
                finally:
                    queue.close()

            self.assertNotEqual(first.document_id, second.document_id)
            service = PrivateKnowledgeService(database, private_root, index)
            self.assertEqual(len(service.list_documents(int(first_user["id"]))), 1)
            self.assertEqual(len(service.list_documents(int(second_user["id"]))), 1)
            self.assertEqual(private_document_ids_for_user(database, int(first_user["id"])), {first.document_id})
            self.assertEqual(private_document_ids_for_user(database, int(second_user["id"])), {second.document_id})

            connection = sqlite3.connect(database)
            try:
                rows = connection.execute(
                    "SELECT owner_user_id, is_private, content_status FROM knowledge_documents ORDER BY id"
                ).fetchall()
            finally:
                connection.close()
            self.assertEqual(rows, [(int(first_user["id"]), 1, "私人资料"), (int(second_user["id"]), 1, "私人资料")])

    def test_failed_private_file_is_not_in_normal_list_and_delete_cleans_all(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            database = base / "app.db"
            index = base / "index"
            private_root = base / "private"
            initialize_database(database)
            user = register_user(database, "失败测试", "abcdef", "abcdef")
            user_root = private_root / str(user["id"])
            user_root.mkdir(parents=True)
            failed_path = user_root / "旧版.doc"
            failed_path.write_bytes(b"not supported")

            queue = ProcessingQueue(private_root, database, index, is_private=True)
            try:
                failed = queue.enqueue(
                    failed_path,
                    source_root=user_root,
                    owner_user_id=int(user["id"]),
                )
                self.assertTrue(queue.wait_for_idle(timeout=10))
            finally:
                queue.close()

            service = PrivateKnowledgeService(database, private_root, index)
            self.assertEqual(service.list_documents(int(user["id"])), [])
            result = service.processing_result(int(user["id"]), failed.document_id)
            self.assertEqual(result["status"], "处理失败")

            valid_path = user_root / "可用.txt"
            valid_path.write_text("这是可以检索的私人资料。", encoding="utf-8")
            with patch("app.processing.processor.embed_texts", return_value=[[0.1, 0.2, 0.3]]):
                queue = ProcessingQueue(private_root, database, index, is_private=True)
                try:
                    valid = queue.enqueue(
                        valid_path,
                        source_root=user_root,
                        owner_user_id=int(user["id"]),
                    )
                    self.assertTrue(queue.wait_for_idle(timeout=10))
                finally:
                    queue.close()
            self.assertTrue(has_document(index, valid.document_id))

            plan = service.create_delete_plan(int(user["id"]), valid.document_id)
            service.confirm_and_delete(int(user["id"]), plan.confirmation_token)
            self.assertFalse(valid_path.exists())
            self.assertFalse(has_document(index, valid.document_id))
            self.assertEqual(service.list_documents(int(user["id"])), [])

    def test_retrieval_accepts_only_current_user_private_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            database = base / "app.db"
            index = base / "index"
            private_root = base / "private"
            initialize_database(database)
            first_user = register_user(database, "检索甲", "abcdef", "abcdef")
            second_user = register_user(database, "检索乙", "abcdef", "abcdef")
            with patch("app.processing.processor.embed_texts", return_value=[[0.1, 0.2, 0.3]]):
                queue = ProcessingQueue(private_root, database, index, is_private=True)
                try:
                    paths = []
                    for user in (first_user, second_user):
                        path = private_root / str(user["id"]) / "资料.txt"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(f"用户 {user['id']} 的私人问答资料。", encoding="utf-8")
                        paths.append((user, path))
                    tasks = [
                        queue.enqueue(path, source_root=path.parent, owner_user_id=int(user["id"]))
                        for user, path in paths
                    ]
                    self.assertTrue(queue.wait_for_idle(timeout=10))
                finally:
                    queue.close()

            rows = [
                {
                    "chunk_key": f"private:{int(user['id'])}:file:资料.txt#0",
                    "document_id": task.document_id,
                    "chunk_index": 0,
                    "text": f"用户 {user['id']} 的私人问答资料。",
                    "source_locator": "",
                    "_distance": 0.1,
                }
                for user, task in zip((first_user, second_user), tasks)
            ]
            result = search_knowledge(
                "私人问答资料",
                db_path=database,
                index_path=index,
                ollama_base_url="http://unused",
                embedding_model="unused",
                owner_user_id=int(first_user["id"]),
                include_private=True,
                embedder=lambda *args, **kwargs: [[0.1, 0.2, 0.3]],
                vector_searcher=lambda *args, **kwargs: rows,
            )
            hit_ids = {int(hit["document_id"]) for hit in result["hits"]}
            self.assertEqual(hit_ids, {tasks[0].document_id})


if __name__ == "__main__":
    unittest.main()

