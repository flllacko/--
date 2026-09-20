from __future__ import annotations

import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from fastapi import HTTPException, Response
from starlette.requests import Request

import app.main as main_module
from app.admin import AdminError, ensure_public_document, list_operation_logs, list_public_documents, list_users, reset_user_password
from app.auth import authenticate_user, create_session, register_user
from app.config import settings as original_settings
from app.db.init_db import initialize_database, utc_now


class AdminTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.database = root / "app.db"
        initialize_database(self.database)
        self.user = register_user(self.database, "普通账号", "abcdef", "abcdef")
        connection = sqlite3.connect(self.database)
        try:
            now = utc_now()
            connection.execute(
                """
                INSERT INTO knowledge_documents
                    (stable_key, title, module, source_text, source_label, source_url,
                     owner_user_id, is_private, content_status, validity_status,
                     processing_status, created_at, updated_at)
                VALUES ('public:test', '公共测试资料', '学生事务', '公共正文', '官方来源',
                        'https://example.test', NULL, 0, '可直接回答', '有效', '处理成功', ?, ?)
                """,
                (now, now),
            )
            connection.execute(
                """
                INSERT INTO knowledge_documents
                    (stable_key, title, module, source_text, source_label,
                     owner_user_id, is_private, content_status, validity_status,
                     processing_status, created_at, updated_at)
                VALUES ('private:test', '私人测试资料', '学生事务', '私人正文', '私人来源',
                        ?, 1, '可直接回答', '有效', '处理成功', ?, ?)
                """,
                (self.user["id"], now, now),
            )
            connection.commit()
        finally:
            connection.close()
        self.previous_settings = main_module.settings
        main_module.settings = replace(
            original_settings,
            sqlite_path=self.database,
            data_root=root,
            public_source_root=root / "public",
            lancedb_path=root / "index",
            log_dir=root / "logs",
        )

    def tearDown(self) -> None:
        main_module.settings = self.previous_settings
        self.temp_dir.cleanup()

    def _request(self, token: str) -> Request:
        return Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/",
                "headers": [(b"cookie", f"cdutetc_session={token}".encode())],
                "query_string": b"",
            }
        )

    def test_admin_can_list_and_reset_only_ordinary_users(self) -> None:
        admin_token = create_session(self.database, 1)
        users = list_users(self.database)
        self.assertEqual([item["username"] for item in users], ["普通账号"])

        reset = reset_user_password(
            self.database,
            actor_user_id=1,
            target_user_id=int(self.user["id"]),
        )
        self.assertTrue(reset["must_change_password"])
        logged = authenticate_user(self.database, "普通账号", "123456")
        self.assertTrue(logged["must_change_password"])
        logs = list_operation_logs(self.database)
        self.assertEqual(logs[0]["action"], "重置用户密码")
        self.assertNotIn("问答", str(logs[0]))

        with self.assertRaises(HTTPException) as denied:
            main_module.admin_users(self._request(create_session(self.database, int(self.user["id"]))))
        self.assertEqual(denied.exception.status_code, 403)
        self.assertEqual(main_module.admin_users(self._request(admin_token))["items"][0]["username"], "普通账号")

    def test_public_document_boundary_and_validity_log(self) -> None:
        items = list_public_documents(self.database)
        self.assertEqual(items["total"], 1)
        self.assertEqual(items["items"][0]["title"], "公共测试资料")
        self.assertNotIn("source_text", items["items"][0])
        public_id = int(items["items"][0]["id"])
        admin_request = self._request(create_session(self.database, 1))
        preview = main_module.admin_document_preview(public_id, admin_request)
        self.assertEqual(preview["source_text"], "公共正文")
        disabled = main_module.admin_disable_document(public_id, admin_request)
        self.assertEqual(disabled["validity_status"], "无效")
        self.assertEqual(list_operation_logs(self.database)[0]["action"], "停用资料")
        with self.assertRaises(HTTPException) as private_error:
            main_module.admin_document_preview(public_id + 1, admin_request)
        self.assertEqual(private_error.exception.status_code, 404)

    def test_batch_validity_prevalidates_public_boundary(self) -> None:
        public_id = int(list_public_documents(self.database)["items"][0]["id"])
        private_id = public_id + 1
        admin_request = self._request(create_session(self.database, 1))

        with self.assertRaises(HTTPException) as denied:
            main_module.admin_batch_validity(
                main_module.DocumentBatchValidityRequest(
                    document_ids=[public_id, private_id],
                    validity_status="无效",
                ),
                admin_request,
            )
        self.assertEqual(denied.exception.status_code, 400)
        self.assertEqual(list_public_documents(self.database)["items"][0]["validity_status"], "有效")

        result = main_module.admin_batch_validity(
            main_module.DocumentBatchValidityRequest(
                document_ids=[public_id, public_id],
                validity_status="无效",
            ),
            admin_request,
        )
        self.assertEqual(len(result["updated"]), 1)
        ensure_public_document(self.database, public_id)
        with self.assertRaises(AdminError):
            ensure_public_document(self.database, private_id)

    def test_raw_download_only_resolves_public_root(self) -> None:
        source = Path(self.temp_dir.name) / "public" / "学生事务" / "raw-test.txt"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("原始测试资料", encoding="utf-8")
        connection = sqlite3.connect(self.database)
        try:
            now = utc_now()
            connection.execute(
                "UPDATE knowledge_documents SET stable_key = ? WHERE title = '公共测试资料'",
                ("public:file:学生事务/raw-test.txt",),
            )
            connection.commit()
        finally:
            connection.close()
        admin_request = self._request(create_session(self.database, 1))
        public_id = int(list_public_documents(self.database)["items"][0]["id"])
        response = main_module.admin_document_raw(public_id, admin_request)
        self.assertEqual(Path(response.path).resolve(), source.resolve())


if __name__ == "__main__":
    unittest.main()
