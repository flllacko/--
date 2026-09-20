from __future__ import annotations

import sqlite3
import tempfile
import unittest
from dataclasses import replace
from http.cookies import SimpleCookie
from pathlib import Path

from fastapi import HTTPException, Response
from starlette.requests import Request

import app.main as main_module
from app.auth import (
    authenticate_user,
    get_user_for_session,
    hash_password,
    register_user,
    verify_password,
)
from app.config import settings as original_settings
from app.db.init_db import initialize_database


class AuthServiceTests(unittest.TestCase):
    def test_password_hash_and_username_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "app.db"
            initialize_database(database)
            user = register_user(database, "测试User", "abcdef", "abcdef")
            self.assertNotIn("abcdef", str(user))
            connection = sqlite3.connect(database)
            try:
                stored = connection.execute(
                    "SELECT password_hash FROM users WHERE id = ?", (user["id"],)
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertNotIn("abcdef", stored)
            self.assertTrue(verify_password("abcdef", stored))
            self.assertFalse(verify_password("abcdeg", stored))

    def test_session_token_is_not_stored_raw(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "app.db"
            initialize_database(database)
            user = register_user(database, "会话用户", "abcdef", "abcdef")
            from app.auth import create_session

            token = create_session(database, int(user["id"]))
            self.assertTrue(get_user_for_session(database, token))
            connection = sqlite3.connect(database)
            try:
                stored = connection.execute("SELECT token_hash FROM sessions").fetchone()[0]
            finally:
                connection.close()
            self.assertNotEqual(token, stored)
            self.assertEqual(len(stored), 64)


class AuthApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.database = root / "app.db"
        initialize_database(self.database)
        self.previous_settings = main_module.settings
        main_module.settings = replace(
            original_settings,
            sqlite_path=self.database,
            data_root=root,
            lancedb_path=root / "index",
            log_dir=root / "logs",
        )

    def _request(self, token: str | None = None) -> Request:
        headers = []
        if token:
            headers.append(
                (
                    b"cookie",
                    f"{main_module.settings.session_cookie_name}={token}".encode(),
                )
            )
        return Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/",
                "headers": headers,
                "query_string": b"",
            }
        )

    def _token_from_response(self, response: Response) -> str:
        cookies = SimpleCookie()
        cookies.load(response.headers["set-cookie"])
        return cookies[main_module.settings.session_cookie_name].value

    def tearDown(self) -> None:
        main_module.settings = self.previous_settings
        self.temp_dir.cleanup()

    def test_register_login_logout_and_admin(self) -> None:
        registration_response = Response()
        registered = main_module.register(
            main_module.RegisterRequest(
                username="普通用户", password="abcdef", password_confirmation="abcdef"
            ),
            registration_response,
        )
        self.assertEqual(registered["user"]["role"], "user")
        token = self._token_from_response(registration_response)
        self.assertTrue(main_module.auth_me(self._request(token))["authenticated"])

        with self.assertRaises(HTTPException) as duplicate_error:
            main_module.register(
                main_module.RegisterRequest(
                    username="普通用户", password="abcdef", password_confirmation="abcdef"
                ),
                Response(),
            )
        self.assertEqual(duplicate_error.exception.status_code, 409)

        logout_response = Response()
        self.assertEqual(
            main_module.logout(self._request(token), logout_response)["logged_out"], True
        )
        self.assertFalse(main_module.auth_me(self._request(token))["authenticated"])

        with self.assertRaises(HTTPException) as bad_login_error:
            main_module.login(
                main_module.LoginRequest(username="普通用户", password="wrong1"), Response()
            )
        self.assertEqual(bad_login_error.exception.status_code, 401)
        login_response = Response()
        good_login = main_module.login(
            main_module.LoginRequest(username="普通用户", password="abcdef"), login_response
        )
        self.assertEqual(good_login["user"]["username"], "普通用户")

        admin_response = Response()
        admin_login = main_module.login(
            main_module.LoginRequest(username="admin", password="123456"), admin_response
        )
        self.assertEqual(admin_login["user"]["role"], "admin")

    def test_password_change_and_account_delete(self) -> None:
        registration_response = Response()
        registered = main_module.register(
            main_module.RegisterRequest(
                username="删除用户", password="abcdef", password_confirmation="abcdef"
            ),
            registration_response,
        )
        token = self._token_from_response(registration_response)
        changed = main_module.password_change(
            main_module.ChangePasswordRequest(
                current_password="abcdef",
                new_password="ghijkl",
                new_password_confirmation="ghijkl",
            ),
            self._request(token),
        )
        self.assertTrue(changed["changed"])
        main_module.logout(self._request(token), Response())
        login_response = Response()
        main_module.login(
            main_module.LoginRequest(username="删除用户", password="ghijkl"), login_response
        )
        token = self._token_from_response(login_response)
        deleted = main_module.account_delete(self._request(token), Response())
        self.assertTrue(deleted["deleted"])
        self.assertFalse(main_module.auth_me(self._request(token))["authenticated"])
        connection = sqlite3.connect(self.database)
        try:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM users WHERE username = '删除用户'"
                ).fetchone()[0],
                0,
            )
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
