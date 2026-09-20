from __future__ import annotations

import json
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class EmbeddingError(RuntimeError):
    """Ollama 嵌入调用失败。"""


def embed_texts(
    texts: Sequence[str],
    base_url: str,
    model: str,
    timeout: float = 180.0,
) -> list[list[float]]:
    if not texts:
        return []

    payload = json.dumps(
        {
            "model": model,
            "input": list(texts),
            "truncate": False,
            "keep_alive": "5m",
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise EmbeddingError(f"Ollama 嵌入接口返回 HTTP {exc.code}：{detail[:300]}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise EmbeddingError(f"无法连接 Ollama 嵌入接口：{exc}") from exc

    try:
        result = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EmbeddingError("Ollama 嵌入接口返回了无法解析的 JSON") from exc

    embeddings = result.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise EmbeddingError("Ollama 返回的嵌入数量与切片数量不一致")
    if not embeddings or not all(isinstance(vector, list) and vector for vector in embeddings):
        raise EmbeddingError("Ollama 返回了空嵌入向量")

    dimension = len(embeddings[0])
    if not all(len(vector) == dimension for vector in embeddings):
        raise EmbeddingError("同一批嵌入向量维度不一致")
    return [[float(value) for value in vector] for vector in embeddings]
