from __future__ import annotations

"""P7 账号、密码和服务端会话。

这里故意只使用 Python 标准库：密码不会以明文进入数据库，浏览器只拿到
随机会话令牌，数据库保存令牌的 SHA-256 摘要。这样即使数据库被复制，
也不能直接把密码或会话令牌还原出来。
"""

import base64
import hashlib
import hmac
import re
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "123456"
SESSION_TTL_SECONDS = 7 * 24 * 60 * 60
_USERNAME_PATTERN = re.compile(r"^[\u4e00-\u9fffA-Za-z0-9]{2,10}$")
_PASSWORD_SCRYPT_N = 2**14
_PASSWORD_SCRYPT_R = 8
_PASSWORD_SCRYPT_P = 1


class AuthError(ValueError):
    """可安全展示给用户的账号错误。"""


class DuplicateUsernameError(AuthError):
    pass


class InvalidCredentialsError(AuthError):
    pass


class SessionError(AuthError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect_database(db_path: Path | str) -> sqlite3.Connection:
    connection = sqlite3.connect(Path(db_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def validate_username(username: str) -> str:
    if not isinstance(username, str) or username != username.strip():
        raise AuthError("用户名不能包含首尾空格。")
    if not _USERNAME_PATTERN.fullmatch(username):
        raise AuthError("用户名须为 2～10 个中文、英文或数字，不能包含空格或特殊符号。")
    return username


def validate_password(password: str) -> str:
    if not isinstance(password, str) or not 6 <= len(password) <= 12:
        raise AuthError("密码须为 6～12 位。")
    return password


def validate_registration(
    username: str,
    password: str,
    password_confirmation: str,
) -> tuple[str, str]:
    valid_username = validate_username(username)
    valid_password = validate_password(password)
    if password != password_confirmation:
        raise AuthError("两次输入的密码不一致。")
    return valid_username, valid_password


def hash_password(password: str) -> str:
    """使用带随机盐的 scrypt，返回可验证但不可直接还原的字符串。"""

    validate_password(password)
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_PASSWORD_SCRYPT_N,
        r=_PASSWORD_SCRYPT_R,
        p=_PASSWORD_SCRYPT_P,
    )
    salt_text = base64.urlsafe_b64encode(salt).decode("ascii")
    digest_text = base64.urlsafe_b64encode(digest).decode("ascii")
    return (
        f"scrypt$n={_PASSWORD_SCRYPT_N},r={_PASSWORD_SCRYPT_R},p={_PASSWORD_SCRYPT_P}"
        f"${salt_text}${digest_text}"
    )


def verify_password(password: str, encoded_hash: str) -> bool:
    try:
        algorithm, parameters, salt_text, digest_text = encoded_hash.split("$", 3)
        if algorithm != "scrypt":
            return False
        values = {
            item.split("=", 1)[0]: int(item.split("=", 1)[1])
            for item in parameters.split(",")
        }
        salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
        expected = base64.urlsafe_b64decode(digest_text.encode("ascii"))
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=values["n"],
            r=values["r"],
            p=values["p"],
        )
        return hmac.compare_digest(actual, expected)
    except (KeyError, ValueError, TypeError, UnicodeError):
        return False


def _public_user(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "username": str(row["username"]),
        "role": str(row["role"]),
        "must_change_password": bool(row["must_change_password"]),
        "created_at": row["created_at"],
        "last_login_at": row["last_login_at"],
    }


def register_user(
    db_path: Path | str,
    username: str,
    password: str,
    password_confirmation: str,
) -> dict[str, Any]:
    valid_username, valid_password = validate_registration(
        username, password, password_confirmation
    )
    now = utc_now()
    connection = connect_database(db_path)
    try:
        try:
            cursor = connection.execute(
                """
                INSERT INTO users
                    (username, password_hash, role, must_change_password, status, created_at)
                VALUES (?, ?, 'user', 0, 'active', ?)
                """,
                (valid_username, hash_password(valid_password), now),
            )
        except sqlite3.IntegrityError as exc:
            raise DuplicateUsernameError("该用户名已存在，请换一个用户名。") from exc
        connection.commit()
        row = connection.execute(
            "SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        if row is None:
            raise AuthError("注册失败，请稍后重试。")
        return _public_user(row)
    finally:
        connection.close()


def authenticate_user(
    db_path: Path | str,
    username: str,
    password: str,
) -> dict[str, Any]:
    if not isinstance(username, str) or not username:
        raise InvalidCredentialsError("用户名或密码错误。")
    if not isinstance(password, str) or not password:
        raise InvalidCredentialsError("用户名或密码错误。")
    connection = connect_database(db_path)
    try:
        row = connection.execute(
            "SELECT * FROM users WHERE username = ? AND status = 'active'",
            (username,),
        ).fetchone()
        if row is None or not verify_password(password, str(row["password_hash"])):
            raise InvalidCredentialsError("用户名或密码错误。")
        now = utc_now()
        connection.execute(
            "UPDATE users SET last_login_at = ? WHERE id = ?", (now, row["id"])
        )
        connection.commit()
        updated = connection.execute(
            "SELECT * FROM users WHERE id = ?", (row["id"],)
        ).fetchone()
        return _public_user(updated or row)
    finally:
        connection.close()


def create_session(
    db_path: Path | str,
    user_id: int,
    ttl_seconds: int = SESSION_TTL_SECONDS,
) -> str:
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc)
    expires = now + timedelta(seconds=ttl_seconds)
    connection = connect_database(db_path)
    try:
        connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now.isoformat(),))
        connection.execute(
            """
            INSERT INTO sessions
                (user_id, token_hash, created_at, last_seen_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, token_hash, now.isoformat(), now.isoformat(), expires.isoformat()),
        )
        connection.commit()
        return token
    finally:
        connection.close()


def get_user_for_session(
    db_path: Path | str,
    token: str | None,
) -> dict[str, Any] | None:
    if not token:
        return None
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    now = utc_now()
    connection = connect_database(db_path)
    try:
        row = connection.execute(
            """
            SELECT u.*
            FROM sessions AS s
            JOIN users AS u ON u.id = s.user_id
            WHERE s.token_hash = ?
              AND s.expires_at > ?
              AND u.status = 'active'
            """,
            (token_hash, now),
        ).fetchone()
        if row is None:
            connection.execute(
                "DELETE FROM sessions WHERE token_hash = ? OR expires_at <= ?",
                (token_hash, now),
            )
            connection.commit()
            return None
        connection.execute(
            "UPDATE sessions SET last_seen_at = ? WHERE token_hash = ?",
            (now, token_hash),
        )
        connection.commit()
        return _public_user(row)
    finally:
        connection.close()


def delete_session(db_path: Path | str, token: str | None) -> None:
    if not token:
        return
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    connection = connect_database(db_path)
    try:
        connection.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
        connection.commit()
    finally:
        connection.close()


def change_password(
    db_path: Path | str,
    user_id: int,
    current_password: str,
    new_password: str,
    new_password_confirmation: str,
) -> None:
    validate_password(new_password)
    if new_password != new_password_confirmation:
        raise AuthError("两次输入的新密码不一致。")
    connection = connect_database(db_path)
    try:
        row = connection.execute(
            "SELECT password_hash FROM users WHERE id = ? AND status = 'active'",
            (user_id,),
        ).fetchone()
        if row is None or not verify_password(current_password, str(row["password_hash"])):
            raise InvalidCredentialsError("当前密码不正确。")
        connection.execute(
            "UPDATE users SET password_hash = ?, must_change_password = 0 WHERE id = ?",
            (hash_password(new_password), user_id),
        )
        connection.commit()
    finally:
        connection.close()


def delete_account(db_path: Path | str, user_id: int) -> None:
    """永久删除普通用户的账号、会话、历史会话和私人资料。"""

    connection = connect_database(db_path)
    try:
        connection.execute("BEGIN")
        conversation_rows = connection.execute(
            "SELECT id FROM conversations WHERE user_id = ?", (user_id,)
        ).fetchall()
        conversation_ids = [int(row["id"]) for row in conversation_rows]
        if conversation_ids:
            placeholders = ",".join("?" for _ in conversation_ids)
            connection.execute(
                f"DELETE FROM messages WHERE conversation_id IN ({placeholders})",
                conversation_ids,
            )
            connection.execute(
                f"DELETE FROM conversations WHERE id IN ({placeholders})",
                conversation_ids,
            )
        connection.execute(
            "DELETE FROM knowledge_documents WHERE owner_user_id = ? AND is_private = 1",
            (user_id,),
        )
        connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        connection.execute("DELETE FROM operation_logs WHERE actor_user_id = ?", (user_id,))
        connection.execute("DELETE FROM users WHERE id = ?", (user_id,))
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def get_user_by_id(db_path: Path | str, user_id: int) -> dict[str, Any] | None:
    connection = connect_database(db_path)
    try:
        row = connection.execute(
            "SELECT * FROM users WHERE id = ? AND status = 'active'", (user_id,)
        ).fetchone()
        return _public_user(row) if row else None
    finally:
        connection.close()
