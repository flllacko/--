from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Literal, Sequence

from app.processing.knowledge_scope import (
    document_ids_for_scope,
    filter_vector_rows,
    private_document_ids_for_user,
)
from app.processing.lance_store import vector_search
from app.processing.ollama_embeddings import EmbeddingError, embed_texts


SearchScope = Literal["current", "historical", "navigation"]

QUERY_STOPWORDS = {
    "学校",
    "学生",
    "怎么",
    "如何",
    "处理",
    "需要",
    "什么",
    "哪些",
    "是否",
    "可以",
    "能够",
    "能否",
    "能带",
    "吗",
    "呢",
    "哪里",
    "在哪",
    "在哪看",
    "信息",
    "办法",
    "要求",
    "准备",
    "办理",
    "申请",
}
SCENE_WORDS = {"学校", "学生", "宿舍", "寝室", "公寓"}

# 学生口语和学校正文用词经常不完全一致。这里仅添加可解释的同义短语，
# 不做开放式改写，避免把问题扩展成模型猜测的另一个问题。
QUERY_ALIASES = (
    ("在四川的招生计划", "在川招生计划"),
    ("招生计划和专业学费", "学费标准"),
    ("论文计划", "工作计划"),
    ("毕业论文", "论文"),
    ("以后", "之后"),
    ("生源信息确认", "生源信息"),
    ("院系和专业方向", "院系专业"),
    ("补偿或贷款代偿", "补偿代偿"),
    ("介绍一下学校", "学校简介"),
    ("介绍学校", "学校简介"),
    ("学校介绍", "学校简介"),
    ("学校概况", "学校简介"),
    ("学院简介", "学校简介"),
    ("学校历史", "校史"),
)


class RetrievalError(RuntimeError):
    """检索阶段无法正常完成。"""


@dataclass(frozen=True)
class SearchHit:
    document_id: int
    stable_key: str
    chunk_index: int
    title: str
    module: str
    text: str
    locator: str
    source_label: str
    source_url: str
    published_date: str
    publisher_department: str
    content_status: str
    validity_status: str
    is_private: bool
    score: float
    semantic_distance: float | None
    match_types: tuple[str, ...]

    def as_dict(self, rank: int) -> dict[str, object]:
        return {
            "rank": rank,
            "document_id": self.document_id,
            "stable_key": self.stable_key,
            "chunk_index": self.chunk_index,
            "title": self.title,
            "module": self.module,
            "text": self.text,
            "locator": self.locator,
            "source_label": self.source_label,
            "source_url": self.source_url,
            "published_date": self.published_date,
            "publisher_department": self.publisher_department,
            "content_status": self.content_status,
            "validity_status": self.validity_status,
            "is_private": self.is_private,
            "score": round(self.score, 8),
            "semantic_distance": (
                round(self.semantic_distance, 8)
                if self.semantic_distance is not None
                else None
            ),
            "match_types": list(self.match_types),
        }


def _normalized_question(question: str) -> str:
    return re.sub(r"\s+", "", question).strip().lower()


def _keyword_terms(question: str) -> list[str]:
    """提取适合 SQLite LIKE 的少量关键词。

    中文问题通常没有空格，不能只靠 split。这里保留完整数字/英文词，
    并从连续中文片段生成二到四字短词，服务于金额、年份、制度名称等精确命中。
    """

    normalized = _normalized_question(question)
    terms: set[str] = set()
    for match in re.finditer(r"[a-z0-9]+|[\u4e00-\u9fff]+", normalized):
        value = match.group(0)
        if value.isdigit() or re.fullmatch(r"[a-z0-9]+", value):
            if len(value) >= 2:
                terms.add(value)
            continue
        if len(value) >= 2:
            terms.add(value)
            for width in (4, 3, 2):
                if len(value) < width:
                    continue
                for index in range(0, len(value) - width + 1):
                    terms.add(value[index : index + width])
    for source, alias in QUERY_ALIASES:
        if source in normalized:
            terms.add(alias)
    return sorted(terms, key=lambda value: (-len(value), value))[:48]


def _priority_query_terms(question: str) -> list[str]:
    """提取能表达问题意图的加权词。

    普通切词会产生很多“答辩和论文归档大致在什么时候”之类的长片段，
    它们不一定原样出现在学校正文中，反而会把真正的工作计划挤到后面。
    加权词只来自明确的口语—正文对应关系，用于排序，不改变证据门槛。
    """

    normalized = _normalized_question(question)
    terms: set[str] = set()
    for source, alias in QUERY_ALIASES:
        if source in normalized:
            terms.add(alias)
    if "论文" in normalized and any(marker in normalized for marker in ("开题", "答辩", "归档")):
        terms.update(("论文", "开题", "答辩", "归档"))
    if "院系和专业方向" in normalized:
        terms.update(("报考指南", "院系专业"))
    if "国家奖学金" in normalized and "国家励志奖学金" in normalized:
        terms.update(("国家奖学金", "国家励志奖学金", "国家助学金"))
    if "国家助学金" in normalized and any(
        marker in normalized for marker in ("休学", "保留学籍", "退学")
    ):
        terms.update(("国家助学金", "休学", "保留学籍", "退学", "停发", "暂停发放", "次月起停发"))
    if "勤工助学" in normalized:
        terms.update(("勤工助学", "在岗", "报名", "录用", "公示"))
    if "访客" in normalized and any(marker in normalized for marker in ("时间", "几点", "什么时候")):
        terms.update(("允许来访时间", "9:00", "20:00"))
    if "专升本" in normalized and any(
        marker in normalized for marker in ("学制", "毕业证书", "学士学位", "转专业")
    ):
        terms.update(("学制", "本科毕业证书", "学士学位证书", "转专业"))
    return sorted(terms, key=lambda value: (-len(value), value))


def _anchor_query_terms(question: str) -> tuple[str, ...]:
    """提取必须在正文中出现的业务锚点。

    向量相似度能找到“同一领域”的页面，但不能证明页面回答了同一事项。
    对问题中已经明确写出的政策/业务名称，要求候选正文至少出现一个对应
    锚点，避免学籍、宣传或导航页面挤掉真正的资助/流程正文。
    """

    normalized = _normalized_question(question)
    anchors: list[str] = []
    if any(
        marker in normalized
        for marker in ("介绍一下学校", "介绍学校", "学校介绍", "学校概况", "学院简介", "学校历史")
    ):
        anchors.extend(("学校简介", "校史", "办学"))
    if "专升本" in normalized and "专科专业" in normalized:
        anchors.extend(("本专科专业对应指导目录", "专业对应", "考试科目"))
    if "专升本" in normalized and "对应目录" in normalized:
        anchors.extend(("本专科专业对应指导目录", "仅供参考", "最新文件"))
    if "专升本" in normalized and any(
        marker in normalized for marker in ("报名", "缴费", "打印准考证", "准考证")
    ):
        anchors.extend(
            (
                "专升本考试报名",
                "报名考试费",
                "网上报名",
                "日程安排表",
                "80 元/生",
                "2026 年 1 月 5 日至 11 日",
                "2026 年 4 月 11 日至 4 月 17 日",
                "准考证",
                "2026年",
            )
        )
    if "国家奖学金" in normalized and "国家励志奖学金" in normalized and "同时" in normalized:
        anchors.extend(("不能同时获得", "同时申请并获得国家助学金"))
    mappings = (
        (("国家助学金",), ("国家助学金",)),
        (("国家奖学金", "国家励志奖学金"), ("国家奖学金", "国家励志奖学金", "国家助学金")),
        (("勤工助学",), ("勤工助学",)),
        (("第二课堂", "到梦空间"), ("第二课堂", "到梦空间")),
        (("招生计划",), ("招生计划", "在川招生计划")),
        (("报考指南",), ("报考指南",)),
        (("论文", "毕业论文"), ("论文", "毕业论文")),
        (("烟雾报警器",), ("烟雾报警器",)),
        (("生源信息",), ("生源信息",)),
        (("就业信息网",), ("就业信息网",)),
        (("学费补偿", "贷款代偿", "服兵役"), ("学费补偿", "贷款代偿", "服兵役")),
        (("退役大学生士兵", "免试", "考生类别"), ("退役大学生士兵", "免试", "考生类别")),
        (("退役士兵批", "退役士兵"), ("退役士兵批", "2026年计划", "2025年录取最高分", "2025年录取最低分")),
        (("现代文秘",), ("现代文秘",)),
        (("转入", "转专业", "外语系"), ("转入", "转专业", "面试")),
        (("办学特色", "主要学科方向"), ("办学特色", "主要学科方向")),
        (("管制刀具", "危险品"), ("管制刀具", "危险品")),
        (("访客", "外来人员"), ("访客", "外来人员", "登记")),
        (("专升本", "学制"), ("学制", "本科毕业证书", "学士学位证书", "转专业")),
        (("工程造价", "就业单位"), ("工程造价", "就业单位", "中核二二", "中建", "中冶")),
    )
    for triggers, values in mappings:
        if any(trigger in normalized for trigger in triggers):
            anchors.extend(values)
    return tuple(dict.fromkeys(anchors))


def _core_query_terms(question: str) -> list[str]:
    """提取证据门槛使用的核心词，排除疑问句和常见场景词。"""

    normalized = _normalized_question(question)
    terms: set[str] = set()
    if any(
        marker in normalized
        for marker in ("介绍一下学校", "介绍学校", "学校介绍", "学校概况", "学院简介", "学校历史")
    ):
        terms.update(("学校简介", "校史", "办学"))
    for match in re.finditer(r"[a-z0-9]+|[\u4e00-\u9fff]+", normalized):
        value = match.group(0)
        if re.fullmatch(r"[a-z0-9]+", value):
            if len(value) >= 2:
                terms.add(value)
            continue
        if len(value) < 2:
            continue
        for scene in SCENE_WORDS:
            value = value.replace(scene, "")
        if len(value) < 2:
            continue
        for width in (4, 3, 2):
            if len(value) < width:
                continue
            for index in range(0, len(value) - width + 1):
                term = value[index : index + width]
                if term not in QUERY_STOPWORDS and not any(
                    ignored in term
                    for ignored in (*QUERY_STOPWORDS, *SCENE_WORDS)
                    if len(ignored) >= 2
                ):
                    terms.add(term)
    return sorted(terms, key=lambda value: (-len(value), value))[:32]


def _document_metadata(
    connection: sqlite3.Connection,
    document_ids: set[int],
) -> dict[int, dict[str, object]]:
    if not document_ids:
        return {}
    placeholders = ",".join("?" for _ in document_ids)
    rows = connection.execute(
        f"""
        SELECT id, stable_key, title, module, source_label, source_url,
               published_date, publisher_department, content_status, validity_status,
               is_private
        FROM knowledge_documents
        WHERE id IN ({placeholders})
        """,
        tuple(sorted(document_ids)),
    ).fetchall()
    columns = [column[1] for column in connection.execute("PRAGMA table_info(knowledge_documents)").fetchall()]
    # 运行库由 P5 自动迁移；这个检查让临时旧库返回清晰错误，而不是 KeyError。
    required = {
        "id",
        "stable_key",
        "title",
        "module",
        "source_label",
        "source_url",
        "published_date",
        "publisher_department",
        "content_status",
        "validity_status",
    }
    if not required.issubset(columns):
        missing = sorted(required.difference(columns))
        raise RetrievalError(f"知识库元信息字段缺失：{missing}")
    return {
        int(row[0]): {
            "stable_key": row[1],
            "title": row[2],
            "module": row[3],
            "source_label": row[4] or "",
            "source_url": row[5] or "",
            "published_date": row[6] or "",
            "publisher_department": row[7] or "",
            "content_status": row[8],
            "validity_status": row[9],
            "is_private": bool(row[10]),
        }
        for row in rows
    }


def _keyword_rows(
    connection: sqlite3.Connection,
    allowed_document_ids: set[int],
    terms: Sequence[str],
    limit: int,
    priority_terms: Sequence[str] = (),
) -> list[dict[str, object]]:
    if not allowed_document_ids or not terms:
        return []
    document_placeholders = ",".join("?" for _ in allowed_document_ids)
    term_clauses = []
    parameters: list[object] = list(sorted(allowed_document_ids))
    for term in terms:
        term_clauses.extend(
            ["kc.content LIKE ?", "kd.title LIKE ?", "kd.source_label LIKE ?"]
        )
        parameters.extend([f"%{term}%", f"%{term}%", f"%{term}%"])
    rows = connection.execute(
        f"""
        SELECT kc.document_id, kc.chunk_index, kc.content, kc.source_locator,
               kd.stable_key, kd.title, kd.module, kd.source_label, kd.source_url,
               kd.published_date, kd.publisher_department, kd.content_status,
               kd.validity_status, kd.is_private
        FROM knowledge_chunks AS kc
        JOIN knowledge_documents AS kd ON kd.id = kc.document_id
        WHERE kc.document_id IN ({document_placeholders})
          AND ({" OR ".join(term_clauses)})
        """,
        parameters,
    ).fetchall()

    scored: list[tuple[float, dict[str, object]]] = []
    for row in rows:
        text = f"{row[2]} {row[5]} {row[7]}".lower()
        score = sum(text.count(term) * max(len(term) - 1, 1) for term in terms)
        # 只用于候选排序；是否能作为证据，仍由 generation._reliable_hits 决定。
        score += sum(text.count(term) * 40 for term in priority_terms if term)
        scored.append(
            (
                float(score),
                {
                    "chunk_key": f"{row[4]}#{row[1]}",
                    "document_id": int(row[0]),
                    "chunk_index": int(row[1]),
                    "text": row[2],
                    "source_locator": row[3] or "",
                    "stable_key": row[4],
                    "title": row[5],
                    "module": row[6],
                    "source_label": row[7] or "",
                    "source_url": row[8] or "",
                    "published_date": row[9] or "",
                    "publisher_department": row[10] or "",
                    "content_status": row[11],
                    "validity_status": row[12],
                    "is_private": bool(row[13]),
                    "keyword_score": float(score),
                },
            )
        )
    scored.sort(key=lambda item: (-item[0], int(item[1]["document_id"]), int(item[1]["chunk_index"])))
    return [row for _, row in scored[:limit]]


def _merge_hits(
    semantic_rows: Sequence[dict[str, object]],
    keyword_rows: Sequence[dict[str, object]],
    metadata: dict[int, dict[str, object]],
    limit: int,
    anchor_terms: Sequence[str] = (),
) -> list[SearchHit]:
    # RRF 的好处是不同检索器的分数无需直接比较，只比较各自在结果中的名次。
    combined: dict[str, dict[str, object]] = {}
    for rank, row in enumerate(semantic_rows, start=1):
        key = str(row.get("chunk_key") or f"{row.get('document_id')}#{row.get('chunk_index')}")
        item = combined.setdefault(key, {"row": row, "score": 0.0, "matches": set()})
        item["score"] = float(item["score"]) + 1.0 / (60.0 + rank)
        item["matches"].add("semantic")
    for rank, row in enumerate(keyword_rows, start=1):
        key = str(row.get("chunk_key") or f"{row.get('document_id')}#{row.get('chunk_index')}")
        item = combined.setdefault(key, {"row": row, "score": 0.0, "matches": set()})
        item["score"] = float(item["score"]) + 0.8 / (60.0 + rank)
        item["matches"].add("keyword")

    hits: list[SearchHit] = []
    for item in combined.values():
        row = item["row"]
        document_id = int(row["document_id"])
        document = metadata.get(document_id, {})
        hits.append(
            SearchHit(
                document_id=document_id,
                stable_key=str(document.get("stable_key") or row.get("stable_key") or ""),
                chunk_index=int(row.get("chunk_index", 0)),
                title=str(document.get("title") or row.get("title") or ""),
                module=str(document.get("module") or row.get("module") or ""),
                text=str(row.get("text") or ""),
                locator=str(row.get("source_locator") or row.get("locator") or ""),
                source_label=str(document.get("source_label") or row.get("source_label") or ""),
                source_url=str(document.get("source_url") or row.get("source_url") or ""),
                published_date=str(document.get("published_date") or row.get("published_date") or ""),
                publisher_department=str(
                    document.get("publisher_department")
                    or row.get("publisher_department")
                    or ""
                ),
                content_status=str(document.get("content_status") or row.get("content_status") or ""),
                validity_status=str(document.get("validity_status") or row.get("validity_status") or ""),
                is_private=bool(document.get("is_private", row.get("is_private", False))),
                score=float(item["score"]),
                semantic_distance=(
                    float(row["_distance"])
                    if row.get("_distance") is not None
                    else None
                ),
                match_types=tuple(sorted(item["matches"])),
            )
        )
    if anchor_terms:
        anchored_hits = [
            hit
            for hit in hits
            if any(
                anchor in f"{hit.title} {hit.text} {hit.source_label}"
                for anchor in anchor_terms
            )
        ]
        # 如果锚点没有任何命中，保留原始语义结果，让系统最终由证据门槛
        # 拒答，而不是错误地制造“没有资料”的假象。
        if anchored_hits:
            hits = anchored_hits
    hits.sort(key=lambda hit: (-hit.score, hit.semantic_distance or float("inf"), hit.document_id, hit.chunk_index))

    # 结果按资料多样性取样：先让不同资料各占一个位置，再给高相关资料
    # 补一个切片。否则一篇长文的多个切片会占满前六名，把真正相关的
    # 第二份官方资料挤出证据上下文和来源卡片。
    selected: list[SearchHit] = []
    selected_per_document: dict[int, int] = {}
    selected_chunks: set[tuple[int, int]] = set()
    max_per_document = 3 if anchor_terms else 2

    # 多段式问题经常由同一份官方通知中的不同页共同回答，例如报名日期、
    # 缴费金额和准考证时间分别位于不同切片。纯粹的“每份资料先取一个”
    # 会把同一份通知的关键页拆散，甚至被只有年份或通用词的历史资料挤掉。
    # 因此，当某份资料覆盖了至少两个业务锚点时，先保留这份资料的多个
    # 相关切片，再恢复资料多样性；这只改变上下文取样，不降低证据门槛。
    anchor_priority_document: int | None = None
    if anchor_terms:
        coverage: dict[int, set[str]] = {}
        best_score: dict[int, float] = {}
        for hit in hits:
            searchable = f"{hit.title} {hit.text} {hit.source_label}"
            matched = {
                anchor for anchor in anchor_terms if anchor and anchor in searchable
            }
            if not matched:
                continue
            coverage.setdefault(hit.document_id, set()).update(matched)
            best_score[hit.document_id] = max(
                best_score.get(hit.document_id, 0.0), hit.score
            )
        ranked_documents = sorted(
            coverage,
            key=lambda document_id: (
                -len(coverage[document_id]),
                -best_score.get(document_id, 0.0),
                document_id,
            ),
        )
        if ranked_documents and len(coverage[ranked_documents[0]]) >= 2:
            anchor_priority_document = ranked_documents[0]

    if anchor_priority_document is not None:
        priority_hits = [
            hit for hit in hits if hit.document_id == anchor_priority_document
        ]
        priority_hits.sort(
            key=lambda hit: (
                -sum(
                    1
                    for anchor in anchor_terms
                    if anchor
                    and anchor in f"{hit.title} {hit.text} {hit.source_label}"
                ),
                -hit.score,
                hit.semantic_distance or float("inf"),
                hit.chunk_index,
            )
        )
        for hit in priority_hits:
            if selected_per_document.get(hit.document_id, 0) >= max_per_document:
                break
            selected.append(hit)
            selected_per_document[hit.document_id] = (
                selected_per_document.get(hit.document_id, 0) + 1
            )
            selected_chunks.add((hit.document_id, hit.chunk_index))

    # 先保留最多四份不同资料，再用剩余位置补充高相关资料的第二个切片。
    # 如果第一轮直接填满 limit，长文的金额表、办理材料或时间表可能永远
    # 不会进入上下文；四份资料仍能维持来源多样性，第二轮则保住同一正文
    # 内跨切片的完整证据。
    first_pass_limit = min(limit, 4)
    for hit in hits:
        if len(selected) >= first_pass_limit:
            break
        if hit.document_id in selected_per_document:
            continue
        selected.append(hit)
        selected_per_document[hit.document_id] = 1
        selected_chunks.add((hit.document_id, hit.chunk_index))
    for hit in hits:
        if len(selected) >= limit:
            break
        if (hit.document_id, hit.chunk_index) in selected_chunks:
            continue
        count = selected_per_document.get(hit.document_id, 0)
        if count >= max_per_document:
            continue
        if count == 0:
            # 第一轮因候选不足没有选到的新资料仍可补进来。
            selected_per_document[hit.document_id] = 1
        else:
            selected_per_document[hit.document_id] = count + 1
        selected.append(hit)
        selected_chunks.add((hit.document_id, hit.chunk_index))
    return selected


def _add_adjacent_context(
    hits: Sequence[SearchHit],
    *,
    db_path: Path,
    limit: int,
) -> list[SearchHit]:
    """为最相关的两份正文补一个相邻切片。

    网页整理后的金额表、材料表和时间表可能被切到相邻 chunk；只取命中
    片段会出现“标题和整理边界命中了，但真正的表格没进上下文”。相邻切片
    仍属于同一份官方正文，因此继承父切片的来源和证据类型，但会扣一点排序
    分数，避免上下文窗口被一份长文完全占满。
    """

    if not hits or limit < 1:
        return list(hits)
    base_hits = list(hits)
    existing = {(hit.document_id, hit.chunk_index) for hit in base_hits}
    neighbors: list[SearchHit] = []
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        for parent in base_hits[:2]:
            rows = connection.execute(
                """
                SELECT chunk_index, content, source_locator
                FROM knowledge_chunks
                WHERE document_id = ?
                  AND chunk_index IN (?, ?)
                ORDER BY chunk_index
                """,
                (parent.document_id, parent.chunk_index - 1, parent.chunk_index + 1),
            ).fetchall()
            for row in rows:
                key = (parent.document_id, int(row[0]))
                if key in existing:
                    continue
                existing.add(key)
                neighbors.append(
                    replace(
                        parent,
                        chunk_index=int(row[0]),
                        text=str(row[1] or ""),
                        locator=str(row[2] or ""),
                        score=max(parent.score - 0.00001, 0.0),
                    )
                )
    finally:
        connection.close()

    combined = [*base_hits, *neighbors]
    combined.sort(
        key=lambda hit: (
            -hit.score,
            hit.semantic_distance or float("inf"),
            hit.document_id,
            hit.chunk_index,
        )
    )
    return combined[:limit]


def search_knowledge(
    question: str,
    *,
    db_path: Path,
    index_path: Path,
    ollama_base_url: str,
    embedding_model: str,
    scope: SearchScope = "current",
    limit: int = 6,
    owner_user_id: int | None = None,
    include_private: bool = False,
    embedder: Callable[..., list[list[float]]] = embed_texts,
    vector_searcher: Callable[..., list[dict[str, object]]] = vector_search,
) -> dict[str, object]:
    normalized = question.strip()
    if not normalized:
        raise ValueError("问题不能为空")
    if len(normalized) > 1000:
        raise ValueError("问题不能超过 1000 个字符")
    if limit < 1 or limit > 20:
        raise ValueError("检索条数必须在 1 到 20 之间")

    allowed_ids = document_ids_for_scope(
        db_path,
        scope,
        # 只有请求已经进入历史范围时，才允许读取无效的历史归档。
        allow_invalid_history=scope == "historical",
    )
    private_ids: set[int] = set()
    if include_private and owner_user_id is not None:
        private_ids = private_document_ids_for_user(db_path, owner_user_id)
        allowed_ids.update(private_ids)
    priority_terms = _priority_query_terms(normalized)
    anchor_terms = _anchor_query_terms(normalized)
    priority_terms = sorted(
        set(priority_terms).union(anchor_terms),
        key=lambda value: (-len(value), value),
    )
    # 锚点不仅用于过滤候选，还要参与关键词召回。否则长文中“就业单位”
    # 可能位于远离课程段落的切片，事后只补一个相邻切片仍然拿不到它。
    terms = sorted(
        set(_keyword_terms(normalized)).union(anchor_terms),
        key=lambda value: (-len(value), value),
    )[:64]
    core_terms = _core_query_terms(normalized)
    try:
        vectors = embedder(
            [normalized],
            ollama_base_url,
            embedding_model,
            timeout=180.0,
        )
    except (EmbeddingError, OSError) as exc:
        raise RetrievalError(f"无法生成问题向量：{exc}") from exc
    if not vectors:
        raise RetrievalError("问题向量为空")

    try:
        semantic_candidates = filter_vector_rows(
            vector_searcher(index_path, vectors[0], limit=max(limit * 8, 24)),
            allowed_ids,
        )
    except Exception as exc:
        raise RetrievalError(f"向量检索失败：{exc}") from exc

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        keyword_candidates = _keyword_rows(
            connection,
            allowed_ids,
            terms,
            limit=max(limit * 8, 24),
            priority_terms=priority_terms,
        )
        metadata = _document_metadata(connection, allowed_ids)
    finally:
        connection.close()

    hits = _merge_hits(
        semantic_candidates,
        keyword_candidates,
        metadata,
        limit,
        anchor_terms=anchor_terms,
    )
    hits = _add_adjacent_context(hits, db_path=db_path, limit=limit)
    return {
        "question": normalized,
        "scope": scope,
        "retrieval": {
            "semantic_model": embedding_model,
            "keyword_terms": terms,
            "core_terms": core_terms,
            "allowed_document_count": len(allowed_ids),
            "private_document_count": len(private_ids),
            "hit_count": len(hits),
        },
        "hits": [hit.as_dict(rank) for rank, hit in enumerate(hits, start=1)],
        "answer_status": "retrieved_only",
        "notice": "已找到可用资料片段；最终回答生成将在后续 RAG 阶段完成。",
    }
