from __future__ import annotations

import unittest

from app.admin_upload import UploadParseError, parse_multipart, safe_upload_filename


class AdminUploadTests(unittest.TestCase):
    def test_parse_multipart_and_safe_filename(self) -> None:
        boundary = "----codex-test"
        body = (
            f"--{boundary}\r\n"
            "Content-Disposition: form-data; name=\"module\"\r\n\r\n"
            "学生事务\r\n"
            f"--{boundary}\r\n"
            "Content-Disposition: form-data; name=\"file\"; filename=\"通知.md\"\r\n"
            "Content-Type: text/markdown\r\n\r\n"
            "# 官方通知\n正文\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")
        parts = parse_multipart(f"multipart/form-data; boundary={boundary}", body)
        self.assertEqual(parts[0].field_name, "module")
        self.assertEqual(parts[0].content.decode(), "学生事务")
        self.assertEqual(parts[1].filename, "通知.md")
        self.assertEqual(safe_upload_filename("..\\通知.md"), "通知.md")

    def test_reject_unsupported_extension(self) -> None:
        with self.assertRaises(UploadParseError):
            safe_upload_filename("password.exe")


if __name__ == "__main__":
    unittest.main()
