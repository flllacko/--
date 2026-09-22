from __future__ import annotations

import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from starlette.requests import Request

import app.main as main_module
from app.auth import create_session, register_user
from app.config import settings as original_settings
from app.conversations import (
    ConversationPermissionError,
    delete_conversations,
    get_conversation,
    import_guest_entries,
    list_conversations,
    save_completed_answer,
    update_conversation,
)
from app.db.init_db import initialize_database


class ConversationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Path(self.temp_dir.name) / "app.db"
        initialize_database(self.database)
        self.user = register_user(self.database, "历史用户", "abcdef", "abcdef")
        self.other_user = register_user(self.database, "其他用户", "abcdef", "abcdef")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _answer(self) -> dict[str, object]:
        return {
            "conclusion": "这是完整回答。",
            "conditions": ["适用于当前公开规则"],
            "materials": [],
            "steps": ["查看官方通知"],
            "time": "以最新通知为准",
            "notes": [],
            "follow_up_questions": [],
        }

    def test_save_sources_list_and_permission_boundary(self) -> None:
        source = {"title": "官方通知", "text": "证据正文", "source_url": "https://example.test"}
        conversation_id = save_completed_answer(
            self.database,
            user_id=int(self.user["id"]),
            conversation_id=None,
            question="我该看什么？",
            answer=self._answer(),
            answer_status="completed",
            sources=[source],
            identity="student",
        )
        items = list_conversations(self.database, user_id=int(self.user["id"]))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["message_count"], 2)
        detail = get_conversation(
            self.database, user_id=int(self.user["id"]), conversation_id=conversation_id
        )
        assistant = [item for item in detail["messages"] if item["role"] == "assistant"][0]
        self.assertEqual(assistant["sources"], [source])
        with self.assertRaises(ConversationPermissionError):
            get_conversation(
                self.database,
                user_id=int(self.other_user["id"]),
                conversation_id=conversation_id,
            )

        renamed = update_conversation(
            self.database,
            user_id=int(self.user["id"]),
            conversation_id=conversation_id,
            title="重要会话",
            is_pinned=True,
        )
        self.assertEqual(renamed["title"], "重要会话")
        self.assertTrue(renamed["is_pinned"])

    def test_guest_import_and_delete_cascade(self) -> None:
        conversation_id = import_guest_entries(
            self.database,
            user_id=int(self.user["id"]),
            identity="student",
            entries=[
                {
                    "question": "临时问题",
                    "answer": self._answer(),
                    "answer_status": "completed",
                    "sources": [],
                }
            ],
        )
        deleted = delete_conversations(
            self.database,
            user_id=int(self.user["id"]),
            conversation_ids=[conversation_id],
        )
        self.assertEqual(deleted, 1)
        connection = sqlite3.connect(self.database)
        try:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 0
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM conversations").fetchone()[0], 0
            )
        finally:
            connection.close()


class AnswerPersistenceTests(unittest.TestCase):
    def test_guest_is_not_saved_and_logged_user_is_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "app.db"
            initialize_database(database)
            user = register_user(database, "回答用户", "abcdef", "abcdef")
            token = create_session(database, int(user["id"]))
            previous_settings = main_module.settings
            main_module.settings = replace(
                original_settings,
                sqlite_path=database,
                data_root=root,
                lancedb_path=root / "index",
                log_dir=root / "logs",
            )
            retrieval = {
                "hits": [
                    {
                        "rank": 1,
                        "document_id": 7,
                        "chunk_index": 0,
                        "title": "官方资料",
                        "text": "这是证据正文。",
                    }
                ],
                "retrieval": {"hit_count": 1},
            }
            generated = {
                "answer_status": "completed",
                "answer": {
                    "conclusion": "完整回答",
                    "conditions": [],
                    "materials": [],
                    "steps": [],
                    "time": "",
                    "notes": [],
                    "follow_up_questions": [],
                },
                "used_source_ranks": [1],
            }

            def request_with_token(value: str | None) -> Request:
                headers = [] if value is None else [(b"cookie", f"cdutetc_session={value}".encode())]
                return Request(
                    {
                        "type": "http",
                        "method": "POST",
                        "path": "/api/answer",
                        "headers": headers,
                        "query_string": b"",
                    }
                )

            planner = patch.object(main_module, "_dialogue_plan", side_effect=lambda request, user: {
                "mode": "general" if request.question == "你好" else "school",
                "query": request.question, "reply": "你好，有什么可以帮你？",
            })
            planner.start()
            try:
                with patch.object(main_module, "search_knowledge", return_value=retrieval), patch.object(
                    main_module, "generate_answer", return_value=generated
                ):
                    logged_result = main_module.answer(
                        main_module.SearchRequest(question="登录用户问题"),
                        request_with_token(token),
                    )
                    guest_result = main_module.answer(
                        main_module.SearchRequest(question="访客问题"),
                        request_with_token(None),
                    )
                self.assertTrue(logged_result["saved"])
                self.assertFalse(guest_result["saved"])
                greeting = main_module.answer(
                    main_module.SearchRequest(question="你好", conversation_id=logged_result["conversation_id"]),
                    request_with_token(token),
                )
                self.assertTrue(greeting["saved"])
                self.assertEqual(greeting["conversation_id"], logged_result["conversation_id"])
                self.assertEqual(greeting["sources"], [])
                guest_greeting = main_module.answer(
                    main_module.SearchRequest(question="你好"), request_with_token(None),
                )
                self.assertFalse(guest_greeting["saved"])
                connection = sqlite3.connect(database)
                try:
                    self.assertEqual(
                        connection.execute("SELECT COUNT(*) FROM conversations").fetchone()[0],
                        1,
                    )
                    self.assertEqual(
                        connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 4
                    )
                finally:
                    connection.close()
            finally:
                planner.stop()
                main_module.settings = previous_settings


if __name__ == "__main__":
    unittest.main()
