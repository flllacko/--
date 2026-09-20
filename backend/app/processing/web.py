from __future__ import annotations

import re
from dataclasses import dataclass, replace
from html.parser import HTMLParser
from pathlib import PurePosixPath
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen


class WebExtractionError(ValueError):
    """网页没有形成可作为资料处理的正文。"""


ATTACHMENT_EXTENSIONS = {
    ".pdf": "PDF",
    ".doc": "Word",
    ".docx": "Word",
    ".xls": "Excel",
    ".xlsx": "Excel",
}
SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "nav", "header", "footer", "aside", "form"}
ROOT_TAGS = {"main", "article"}
BLOCK_TAGS = {
    "address",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "tr",
    "ul",
}


@dataclass(frozen=True)
class AttachmentCandidate:
    url: str
    label: str
    file_type: str
    selected: bool = False


@dataclass(frozen=True)
class WebPageExtraction:
    url: str
    title: str
    body_text: str
    attachments: tuple[AttachmentCandidate, ...]


@dataclass
class _Anchor:
    url: str
    label_parts: list[str]
    in_content: bool


class _PageParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title_parts: list[str] = []
        self.segments: list[tuple[bool, str]] = []
        self.attachments: list[tuple[bool, AttachmentCandidate]] = []
        self._skip_depth = 0
        self._root_depth = 0
        self._root_seen = False
        self._title_depth = 0
        self._anchors: list[_Anchor] = []

    @property
    def content_root_seen(self) -> bool:
        return self._root_seen

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = dict(attrs)
        if tag == "title":
            self._title_depth += 1
        if tag in SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag in ROOT_TAGS:
            self._root_seen = True
            self._root_depth += 1
        if tag in BLOCK_TAGS:
            self.segments.append((self._root_depth > 0, "\n"))
        if tag == "a" and self._skip_depth == 0:
            href = (attributes.get("href") or "").strip()
            absolute = urljoin(self.base_url, href)
            parsed = urlparse(absolute)
            if parsed.scheme in {"http", "https"}:
                self._anchors.append(
                    _Anchor(
                        url=absolute,
                        label_parts=[],
                        in_content=self._root_depth > 0,
                    )
                )

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title" and self._title_depth:
            self._title_depth -= 1
        if tag == "a" and self._anchors:
            anchor = self._anchors.pop()
            if self._is_attachment(anchor.url):
                parsed = urlparse(anchor.url)
                path_name = PurePosixPath(parsed.path).name
                label = " ".join(" ".join(anchor.label_parts).split()) or unquote(path_name) or anchor.url
                extension = PurePosixPath(parsed.path).suffix.lower()
                self.attachments.append(
                    (
                        anchor.in_content,
                        AttachmentCandidate(
                            url=anchor.url,
                            label=label,
                            file_type=ATTACHMENT_EXTENSIONS[extension],
                        ),
                    )
                )
        if tag in ROOT_TAGS and self._root_depth:
            self._root_depth -= 1
        if tag in SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._title_depth:
            self.title_parts.append(data)
            return
        if self._skip_depth:
            return
        if self._anchors:
            self._anchors[-1].label_parts.append(data)
        if data.strip():
            self.segments.append((self._root_depth > 0, data))

    @staticmethod
    def _is_attachment(url: str) -> bool:
        extension = PurePosixPath(urlparse(url).path).suffix.lower()
        return extension in ATTACHMENT_EXTENSIONS


def _normalize_text(parts: Iterable[str]) -> str:
    lines: list[str] = []
    for raw_line in "".join(parts).splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line:
            lines.append(line)
        elif lines and lines[-1] != "":
            lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def extract_web_page(html: str, url: str, *, min_body_chars: int = 30) -> WebPageExtraction:
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"}:
        raise WebExtractionError("网页地址必须使用 http 或 https")

    parser = _PageParser(url)
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # noqa: BLE001 - HTML 容错解析统一转为可读失败
        raise WebExtractionError(f"网页 HTML 解析失败：{exc}") from exc

    if parser.content_root_seen:
        body_parts = [text for is_content, text in parser.segments if is_content]
        attachment_parts = [candidate for candidate in parser.attachments if candidate[0]]
    else:
        body_parts = parser.segments
        attachment_parts = parser.attachments
    body_text = _normalize_text(body_parts)
    if len(body_text) < min_body_chars:
        raise WebExtractionError("网页没有足够的正文内容，可能只是入口页、导航页或空页面")

    unique_attachments: list[AttachmentCandidate] = []
    seen_urls: set[str] = set()
    for _, candidate in attachment_parts:
        if candidate.url not in seen_urls:
            seen_urls.add(candidate.url)
            unique_attachments.append(candidate)

    title = _normalize_text(parser.title_parts) or (PurePosixPath(parsed_url.path).name or parsed_url.netloc)
    return WebPageExtraction(
        url=url,
        title=title,
        body_text=body_text,
        attachments=tuple(unique_attachments),
    )


def fetch_web_page(url: str, *, timeout: float = 30.0, user_agent: str = "StudentServiceRAG/0.1") -> WebPageExtraction:
    request = Request(url, headers={"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml"})
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
            content_type = response.headers.get_content_type()
            charset = response.headers.get_content_charset() or "utf-8"
            final_url = response.geturl()
    except HTTPError as exc:
        raise WebExtractionError(f"网页返回 HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise WebExtractionError(f"网页读取失败：{exc}") from exc

    if content_type not in {"text/html", "application/xhtml+xml"}:
        raise WebExtractionError(f"来源不是网页正文，而是 {content_type}")
    try:
        html = body.decode(charset, errors="replace")
    except LookupError:
        html = body.decode("utf-8", errors="replace")
    return extract_web_page(html, final_url)


def select_attachments(
    candidates: Iterable[AttachmentCandidate],
    selected_urls: Iterable[str],
) -> tuple[AttachmentCandidate, ...]:
    candidate_list = tuple(candidates)
    allowed_urls = {candidate.url for candidate in candidate_list}
    selected_set = set(selected_urls)
    unknown = selected_set - allowed_urls
    if unknown:
        raise WebExtractionError(f"选择了未列出的附件：{', '.join(sorted(unknown))}")
    return tuple(
        replace(candidate, selected=candidate.url in selected_set)
        for candidate in candidate_list
    )


def selected_attachments(candidates: Iterable[AttachmentCandidate]) -> tuple[AttachmentCandidate, ...]:
    return tuple(candidate for candidate in candidates if candidate.selected)
