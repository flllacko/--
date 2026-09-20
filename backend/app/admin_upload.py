from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_UPLOAD_EXTENSIONS = {
    ".md",
    ".txt",
    ".html",
    ".htm",
    ".pdf",
    ".docx",
    ".xlsx",
    ".xls",
}


class UploadParseError(ValueError):
    pass


@dataclass(frozen=True)
class UploadedPart:
    field_name: str
    filename: str | None
    content: bytes


def _header_text(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def parse_multipart(content_type: str, body: bytes) -> list[UploadedPart]:
    match = re.search(r"boundary=(?:\"([^\"]+)\"|([^;]+))", content_type, re.IGNORECASE)
    if not match:
        raise UploadParseError("上传请求缺少 multipart 边界。")
    boundary = (match.group(1) or match.group(2) or "").strip().encode("utf-8")
    if not boundary:
        raise UploadParseError("上传边界为空。")

    marker = b"--" + boundary
    parts: list[UploadedPart] = []
    for raw_part in body.split(marker)[1:]:
        if raw_part.startswith(b"--"):
            break
        raw_part = raw_part.lstrip(b"\r\n")
        if not raw_part:
            continue
        separator = b"\r\n\r\n"
        if separator not in raw_part:
            raise UploadParseError("上传字段结构不完整。")
        raw_headers, content = raw_part.split(separator, 1)
        if content.endswith(b"\r\n"):
            content = content[:-2]
        headers = _header_text(raw_headers)
        disposition = re.search(
            r"Content-Disposition:\s*form-data;([^\r\n]+)", headers, re.IGNORECASE
        )
        if not disposition:
            continue
        parameters = disposition.group(1)
        field_match = re.search(
            r'name=(?:"([^"]+)"|([^;\s]+))',
            parameters,
            re.IGNORECASE,
        )
        if not field_match:
            continue
        filename_match = re.search(
            r'filename=(?:"([^"]*)"|([^;\s]+))',
            parameters,
            re.IGNORECASE,
        )
        filename = (
            filename_match.group(1) or filename_match.group(2)
            if filename_match
            else None
        )
        field_name = field_match.group(1) or field_match.group(2)
        parts.append(UploadedPart(field_name, filename, content))
    if not parts:
        raise UploadParseError("上传请求中没有字段。")
    return parts


def safe_upload_filename(filename: str) -> str:
    candidate = Path(filename).name.strip()
    candidate = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", candidate)
    if not candidate or candidate in {".", ".."}:
        raise UploadParseError("上传文件名无效。")
    suffix = Path(candidate).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise UploadParseError(f"暂不支持上传格式：{suffix or '无扩展名'}。")
    return candidate
