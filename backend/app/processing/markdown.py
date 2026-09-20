from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


class MarkdownProcessingError(ValueError):
    """Markdown 正文无法进入公共知识库时抛出的可读错误。"""


@dataclass(frozen=True)
class ParsedMarkdown:
    title: str
    module: str
    source_text: str
    source_url: str | None
    source_label: str
    content_status: str
    validity_status: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class TextChunk:
    content: str
    locator: str


FIELD_PATTERN = re.compile(
    r"^- (?P<key>[^：:\n]+)[：:]\s*(?P<value>.+?)\s*$",
    re.MULTILINE,
)
TITLE_PATTERN = re.compile(r"^#\s+(?P<title>.+?)\s*$", re.MULTILINE)
SOURCE_URL_PATTERN = re.compile(
    r"^- 原文链接：.*?\]\((?P<url>https?://[^)\s]+)\)",
    re.MULTILINE,
)
UNKNOWN_METADATA_VALUES = {"", "未识别", "未知", "无", "暂无", "—", "-"}
PUBLIC_CONTENT_STATUS_VALUES = (
    "可直接回答",
    "历史可回答",
    "只能提供入口",
    "禁止作为依据",
)


def metadata_value(metadata: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = metadata.get(key, "").strip()
        if value not in UNKNOWN_METADATA_VALUES:
            return value
    return None


def normalize_content_status(value: str | None) -> str:
    candidate = (value or "").strip()
    for status in PUBLIC_CONTENT_STATUS_VALUES:
        if candidate.startswith(status):
            return status
    return candidate or "禁止作为依据"


def clean_markdown(text: str) -> str:
    """保留 Markdown 结构，只清除不可见字符和明显的空白噪声。"""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _metadata(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for match in FIELD_PATTERN.finditer(text):
        key = match.group("key").strip()
        value = match.group("value").strip().strip(chr(96))
        values[key] = value
    return values


def parse_markdown(file_path: Path, source_root: Path) -> ParsedMarkdown:
    path = file_path.resolve()
    root = source_root.resolve()
    if not path.is_relative_to(root):
        raise MarkdownProcessingError("资料文件必须位于公共资料正文目录内")
    if path.suffix.lower() != ".md":
        raise MarkdownProcessingError("P5-S01 当前只支持 Markdown 正文副本")

    try:
        raw_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise MarkdownProcessingError("文件不是有效的 UTF-8 文本") from exc
    except OSError as exc:
        raise MarkdownProcessingError(f"读取资料失败：{exc}") from exc

    source_text = clean_markdown(raw_text)
    if not source_text:
        raise MarkdownProcessingError("正文为空")

    metadata = _metadata(source_text)
    title_match = TITLE_PATTERN.search(source_text)
    title = title_match.group("title").strip() if title_match else path.stem
    relative_path = path.relative_to(root)
    module = metadata.get("所属模块", "").strip() or (
        relative_path.parts[0] if len(relative_path.parts) > 1 else "未分类"
    )

    if not metadata.get("资料编号"):
        raise MarkdownProcessingError("缺少资料编号，不能确认资料身份")
    if not metadata.get("所属模块") and module == "未分类":
        raise MarkdownProcessingError("缺少所属模块，不能确认资料归属")

    source_url_match = SOURCE_URL_PATTERN.search(source_text)
    source_url = source_url_match.group("url").strip() if source_url_match else None
    source_label = metadata_value(metadata, "来源文字") or title
    content_status = normalize_content_status(metadata.get("内容状态"))
    validity_raw = metadata.get("时效状态", "").strip()
    validity_status = "无效" if validity_raw.startswith("无效") else "有效"

    return ParsedMarkdown(
        title=title,
        module=module,
        source_text=source_text,
        source_url=source_url,
        source_label=source_label,
        content_status=content_status,
        validity_status=validity_status,
        metadata=metadata,
    )


def _split_long_block(block: str, max_chars: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    start = 0
    while start < len(block):
        end = min(start + max_chars, len(block))
        piece = block[start:end].strip()
        if piece:
            pieces.append(piece)
        if end == len(block):
            break
        start = end - overlap
    return pieces


def chunk_markdown(text: str, max_chars: int = 900, overlap: int = 120) -> list[TextChunk]:
    """按 Markdown 段落切片，长段落再做带少量重叠的硬切分。"""

    if max_chars <= 0:
        raise ValueError("max_chars 必须大于 0")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap 必须大于等于 0 且小于 max_chars")

    blocks = [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]
    chunks: list[TextChunk] = []
    current_parts: list[str] = []
    current_length = 0
    current_locator = "正文"

    def flush() -> None:
        nonlocal current_parts, current_length
        if current_parts:
            content = "\n\n".join(current_parts).strip()
            if content:
                chunks.append(TextChunk(content=content, locator=current_locator))
        current_parts = []
        current_length = 0

    for block in blocks:
        heading_match = re.match(r"^#{1,6}\s+(.+?)\s*$", block)
        if heading_match:
            current_locator = heading_match.group(1).strip()
        units = _split_long_block(block, max_chars, overlap)
        for unit in units:
            added_length = len(unit) + (2 if current_parts else 0)
            if current_parts and current_length + added_length > max_chars:
                flush()
            current_parts.append(unit)
            current_length += len(unit) + (2 if len(current_parts) > 1 else 0)
            if len(unit) >= max_chars:
                flush()
    flush()
    return chunks
