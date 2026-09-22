from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.admin import (
    AdminError,
    ensure_public_document,
    list_operation_logs,
    list_public_documents,
    list_processing_tasks,
    list_users,
    public_source_path,
    record_operation,
    reset_user_password,
)
from app.admin_upload import MAX_UPLOAD_BYTES, UploadParseError, parse_multipart, safe_upload_filename
from app.auth import (
    AuthError,
    DuplicateUsernameError,
    InvalidCredentialsError,
    authenticate_user,
    change_password,
    connect_database,
    create_session,
    delete_account,
    delete_session,
    get_user_for_session,
    register_user,
)
from app.config import settings
from app.conversations import (
    ConversationError,
    ConversationPermissionError,
    ConversationNotFoundError,
    delete_conversations,
    ensure_conversation_owner,
    get_conversation,
    import_guest_entries,
    list_conversations,
    save_completed_answer,
    update_conversation,
)
from app.db.init_db import initialize_database
from app.qa.generation import (
    GenerationError,
    _build_prompt,
    finalize_stream_answer,
    generate_answer,
    has_reliable_evidence,
    reliable_evidence_hits,
    stream_ollama_chat,
    structure_answer,
)
from app.qa.query_routing import is_greeting_query, resolve_scope
from app.qa.dialogue import plan_dialogue, general_messages
from app.qa.retrieval import RetrievalError, search_knowledge
from app.processing.metadata import MetadataError, update_document_metadata
from app.processing.deletion import DeletionError, DeletionService
from app.processing.lance_store import remove_document
from app.processing.queue import ProcessingQueue, QueueError
from app.private_knowledge import (
    PrivateKnowledgeError,
    PrivateKnowledgeService,
)


public_processing_queue: ProcessingQueue | None = None
private_processing_queue: ProcessingQueue | None = None
public_deletion_service: DeletionService | None = None
private_knowledge_service: PrivateKnowledgeService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global public_processing_queue, private_processing_queue, public_deletion_service, private_knowledge_service
    settings.ensure_runtime_dirs()
    initialize_database()
    public_processing_queue = ProcessingQueue(autostart=True)
    private_processing_queue = ProcessingQueue(
        source_root=settings.private_source_root,
        is_private=True,
        autostart=True,
    )
    public_deletion_service = DeletionService()
    private_knowledge_service = PrivateKnowledgeService(
        settings.sqlite_path,
        settings.private_source_root,
        settings.lancedb_path,
    )
    yield
    if public_processing_queue is not None:
        public_processing_queue.close()
        public_processing_queue = None
    if private_processing_queue is not None:
        private_processing_queue.close()
        private_processing_queue = None
    public_deletion_service = None
    private_knowledge_service = None


app = FastAPI(
    title="成都理工大学工程技术学院学生服务问答助手",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


class DialogueMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    scope: Literal["current", "historical", "navigation"] = "current"
    limit: int = Field(default=6, ge=1, le=20)
    identity: Literal["student", "parent", "teacher", "skip"] = "student"
    conversation_id: int | None = Field(default=None, ge=1)
    use_private_knowledge: bool = False
    history: list[DialogueMessage] = Field(default_factory=list, max_length=8)


class RegisterRequest(BaseModel):
    username: str
    password: str
    password_confirmation: str


class LoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str
    new_password_confirmation: str


class GuestEntry(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    answer: dict[str, object]
    answer_status: Literal["completed", "no_reliable_evidence"] = "completed"
    sources: list[dict[str, object]] = Field(default_factory=list)


class GuestImportRequest(BaseModel):
    entries: list[GuestEntry] = Field(min_length=1, max_length=50)
    identity: Literal["student", "parent", "teacher", "skip"] = "student"


class ConversationUpdateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=80)
    is_pinned: bool | None = None


class ConversationBatchDeleteRequest(BaseModel):
    conversation_ids: list[int] = Field(min_length=1, max_length=100)


class DocumentDeleteConfirmRequest(BaseModel):
    confirmation_token: str = Field(min_length=10, max_length=200)


class DocumentBatchValidityRequest(BaseModel):
    document_ids: list[int] = Field(min_length=1, max_length=100)
    validity_status: Literal["有效", "无效"]


class DocumentBatchDeletePlanRequest(BaseModel):
    document_ids: list[int] = Field(min_length=1, max_length=100)


class DocumentBatchDeleteConfirmRequest(BaseModel):
    confirmation_tokens: list[str] = Field(min_length=1, max_length=100)


class AdminDocumentQuery(BaseModel):
    title: str | None = None
    module: str | None = None
    source_label: str | None = None
    source_url: str | None = None
    published_date: str | None = None
    publisher_department: str | None = None
    content_status: str | None = None
    validity_status: str | None = None


class PrivateTitleUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class PrivateDeleteConfirmRequest(BaseModel):
    confirmation_token: str = Field(min_length=10, max_length=200)


def _require_admin(request: Request) -> dict[str, object]:
    user = _require_user(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="只有管理员可以访问管理后台。")
    return user


def _public_user_from_request(request: Request) -> dict[str, object] | None:
    return get_user_for_session(
        settings.sqlite_path,
        request.cookies.get(settings.session_cookie_name),
    )


def _require_user(
    request: Request,
    *,
    allow_must_change_password: bool = False,
) -> dict[str, object]:
    user = _public_user_from_request(request)
    if user is None:
        raise HTTPException(status_code=401, detail="请先登录后再使用此功能。")
    if user["must_change_password"] and not allow_must_change_password:
        raise HTTPException(status_code=403, detail="请先在个人中心修改密码。")
    return user


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.session_cookie_name,
        value=token,
        max_age=settings.session_ttl_seconds,
        httponly=True,
        samesite="lax",
        secure=settings.app_env == "production",
        path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(settings.session_cookie_name, path="/")


def _save_answer_if_logged_in(
    request: Request,
    user: dict[str, object] | None,
    *,
    question: str,
    answer: dict[str, object],
    answer_status: str,
    sources: list[dict[str, object]],
    identity: str,
    conversation_id: int | None,
) -> tuple[bool, int | None]:
    if user is None:
        return False, None
    try:
        saved_id = save_completed_answer(
            settings.sqlite_path,
            user_id=int(user["id"]),
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            answer_status=answer_status,
            sources=sources,
            identity=identity,
        )
        return True, saved_id
    except ConversationPermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ConversationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _sources_for_ranks(retrieval: dict[str, object], used_ranks: object) -> list[dict[str, object]]:
    ranks = set(used_ranks) if isinstance(used_ranks, list) else set()
    sources_by_document: dict[object, dict[str, object]] = {}
    for hit in retrieval["hits"]:
        if not isinstance(hit, dict) or hit.get("rank") not in ranks:
            continue
        document_id = hit.get("document_id")
        existing = sources_by_document.get(document_id)
        if existing is None:
            source = dict(hit)
            source["fragment_count"] = 1
            sources_by_document[document_id] = source
            continue
        existing_text = str(existing.get("text") or "")
        next_text = str(hit.get("text") or "")
        if next_text and next_text not in existing_text:
            existing["text"] = f"{existing_text}\n\n{next_text}"[:5000]
        existing["fragment_count"] = int(existing.get("fragment_count", 1)) + 1
    return list(sources_by_document.values())


def _greeting_answer() -> dict[str, object]:
    return {
        "conclusion": "你好！我是成都理工大学工程技术学院学生服务问答助手。你可以直接问我学生资助、教务、学生事务、招生、就业、公寓与校园生活等学校服务问题。",
        "conditions": [],
        "materials": [],
        "steps": [],
        "time": "",
        "notes": ["涉及金额、截止日期和办理要求的问题，我会优先依据学校公开资料回答。"],
        "follow_up_questions": [],
    }


def _greeting_payload(request: SearchRequest, http_request: Request, user, reply=None) -> dict[str, object]:
    answer_body = _greeting_answer()
    if reply is not None:
        answer_body.update(conclusion=reply, notes=[])
    saved, conversation_id = _save_answer_if_logged_in(
        http_request, user, question=request.question, answer=answer_body,
        answer_status="completed", sources=[], identity=request.identity,
        conversation_id=request.conversation_id,
    )
    return {
        "question": request.question.strip(),
        "answer_status": "completed",
        "answer": answer_body,
        "sources": [],
        "retrieval": {"hit_count": 0},
        "saved": saved,
        "conversation_id": conversation_id,
    }


def _sse(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _dialogue_plan(request, user, draft=True):
    history = [message.model_dump() for message in request.history]
    if user and request.conversation_id:
        detail = get_conversation(settings.sqlite_path, user_id=int(user["id"]), conversation_id=request.conversation_id)
        history = []
        for message in detail["messages"][-8:]:
            content = message.get("content") or ""
            if message["role"] == "assistant":
                content = json.dumps(message.get("answer") or {}, ensure_ascii=False)
            history.append({"role": message["role"], "content": content[:2000]})
    plan = plan_dialogue(request.question, history, base_url=settings.ollama_base_url,
                        model=settings.ollama_generation_model, private=request.use_private_knowledge, draft=draft)
    plan["history"] = history
    return plan


@app.get("/api/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "phase": "release",
        "version": "1.0",
        "app_env": settings.app_env,
        "generation_model": settings.ollama_generation_model,
        "embedding_model": settings.ollama_embedding_model,
    }


@app.post("/api/auth/register")
def register(request: RegisterRequest, response: Response) -> dict[str, object]:
    try:
        user = register_user(
            settings.sqlite_path,
            request.username,
            request.password,
            request.password_confirmation,
        )
        token = create_session(
            settings.sqlite_path,
            int(user["id"]),
            settings.session_ttl_seconds,
        )
        _set_session_cookie(response, token)
        return {"user": user}
    except DuplicateUsernameError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/auth/login")
def login(request: LoginRequest, response: Response) -> dict[str, object]:
    try:
        user = authenticate_user(settings.sqlite_path, request.username, request.password)
        token = create_session(
            settings.sqlite_path,
            int(user["id"]),
            settings.session_ttl_seconds,
        )
        _set_session_cookie(response, token)
        return {"user": user}
    except InvalidCredentialsError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@app.get("/api/auth/me")
def auth_me(request: Request) -> dict[str, object]:
    user = _public_user_from_request(request)
    return {"authenticated": user is not None, "user": user}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response) -> dict[str, bool]:
    delete_session(settings.sqlite_path, request.cookies.get(settings.session_cookie_name))
    _clear_session_cookie(response)
    return {"logged_out": True}


@app.post("/api/auth/password")
def password_change(
    request: ChangePasswordRequest,
    http_request: Request,
) -> dict[str, bool]:
    user = _require_user(http_request, allow_must_change_password=True)
    if user["role"] == "admin":
        raise HTTPException(status_code=403, detail="第一版不提供管理员修改密码功能。")
    try:
        change_password(
            settings.sqlite_path,
            int(user["id"]),
            request.current_password,
            request.new_password,
            request.new_password_confirmation,
        )
        return {"changed": True}
    except AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/auth/account")
def account_delete(request: Request, response: Response) -> dict[str, bool]:
    user = _require_user(request, allow_must_change_password=True)
    user_id = int(user["id"])
    service = private_knowledge_service or PrivateKnowledgeService(
        settings.sqlite_path,
        settings.private_source_root,
        settings.lancedb_path,
    )
    private_paths = service.source_paths_for_user(user_id)
    for document_id in service.document_ids_for_user(user_id):
        remove_document(settings.lancedb_path, document_id)
    delete_account(settings.sqlite_path, user_id)
    for path in private_paths:
        path.unlink(missing_ok=True)
    _clear_session_cookie(response)
    return {"deleted": True}


def _private_user(request: Request) -> dict[str, object]:
    user = _require_user(request)
    if user["role"] != "user":
        raise HTTPException(status_code=403, detail="管理员没有私人知识库。")
    return user


def _private_service() -> PrivateKnowledgeService:
    if private_knowledge_service is not None:
        return private_knowledge_service
    return PrivateKnowledgeService(
        settings.sqlite_path,
        settings.private_source_root,
        settings.lancedb_path,
    )


@app.get("/api/private/documents")
def private_documents(request: Request, q: str = "", limit: int = 100) -> dict[str, object]:
    user = _private_user(request)
    service = _private_service()
    items = service.list_documents(int(user["id"]), query=q[:100], limit=limit)
    return {"items": items, "has_successful": bool(items) or service.has_successful_documents(int(user["id"]))}


@app.post("/api/private/documents/upload")
async def private_upload_documents(request: Request) -> dict[str, object]:
    user = _private_user(request)
    queue = private_processing_queue
    if queue is None:
        raise HTTPException(status_code=503, detail="私人资料处理队列尚未启动。")
    try:
        content_length = int(request.headers.get("content-length", "0"))
    except ValueError:
        content_length = 0
    if content_length > 200 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="本次上传总大小不能超过 200 MB。")

    try:
        parts = parse_multipart(request.headers.get("content-type", ""), await request.body())
        files = [part for part in parts if part.filename]
        if not files:
            raise UploadParseError("没有找到上传文件。")
        if len(files) > 20:
            raise UploadParseError("一次最多上传 20 个文件。")
        user_id = int(user["id"])
        user_root = (settings.private_source_root / str(user_id)).resolve()
        user_root.mkdir(parents=True, exist_ok=True)
        queued_items: list[dict[str, object]] = []
        for file_part in files:
            filename = safe_upload_filename(file_part.filename or "")
            if len(file_part.content) > MAX_UPLOAD_BYTES:
                queued_items.append(
                    {"filename": filename, "status": "处理失败", "accepted": False, "error_message": "单个文件不能超过 50 MB。"}
                )
                continue
            target = user_root / filename
            if target.exists():
                target = user_root / f"{target.stem}-上传-{uuid4().hex[:8]}{target.suffix}"
            target.write_bytes(file_part.content)
            try:
                queued = queue.enqueue(
                    target,
                    source_root=user_root,
                    owner_user_id=user_id,
                )
                queued_items.append(
                    {
                        "filename": target.name,
                        "document_id": queued.document_id,
                        "task_id": queued.task_id,
                        "accepted": True,
                    }
                )
            except QueueError as exc:
                target.unlink(missing_ok=True)
                queued_items.append(
                    {"filename": filename, "status": "处理失败", "accepted": False, "error_message": str(exc)}
                )

        # 私人上传接口在返回前等待本次文件完成，用户可以直接看到“成功/失败”；
        # 超时只会显示失败和重试入口，不把内部“排队中/处理中”暴露为私人状态。
        queue.wait_for_idle(timeout=600.0)
        service = _private_service()
        results: list[dict[str, object]] = []
        for item in queued_items:
            if not item.get("accepted"):
                results.append(item)
                continue
            result = service.processing_result(user_id, int(item["document_id"]))
            results.append(
                {
                    **item,
                    "status": result["status"],
                    "accepted": result["status"] == "处理成功",
                    "error_message": result.get("error_message"),
                }
            )
        return {"items": results}
    except UploadParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/private/documents/{document_id}/retry")
def private_retry_document(document_id: int, request: Request) -> dict[str, object]:
    user = _private_user(request)
    queue = private_processing_queue
    if queue is None:
        raise HTTPException(status_code=503, detail="私人资料处理队列尚未启动。")
    service = _private_service()
    user_id = int(user["id"])
    try:
        source_path = service.source_path(user_id, document_id)
        queued = queue.enqueue(
            source_path,
            source_root=source_path.parent,
            owner_user_id=user_id,
        )
        queue.wait_for_idle(timeout=600.0)
        result = service.processing_result(user_id, document_id)
        return {
            "document_id": document_id,
            "task_id": queued.task_id,
            "status": result["status"],
            "accepted": result["status"] == "处理成功",
            "error_message": result.get("error_message"),
        }
    except (PrivateKnowledgeError, QueueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.patch("/api/private/documents/{document_id}")
def private_update_document(
    document_id: int,
    payload: PrivateTitleUpdateRequest,
    request: Request,
) -> dict[str, object]:
    user = _private_user(request)
    try:
        return _private_service().update_title(int(user["id"]), document_id, payload.title)
    except PrivateKnowledgeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/private/documents/{document_id}/download")
def private_download_document(document_id: int, request: Request) -> FileResponse:
    user = _private_user(request)
    try:
        source_path = _private_service().source_path(int(user["id"]), document_id)
    except PrivateKnowledgeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not source_path.is_file():
        raise HTTPException(status_code=404, detail="私人资料原始文件不存在。")
    return FileResponse(path=source_path, filename=source_path.name)


@app.post("/api/private/documents/{document_id}/delete-plan")
def private_delete_plan(document_id: int, request: Request) -> dict[str, object]:
    user = _private_user(request)
    try:
        plan = _private_service().create_delete_plan(int(user["id"]), document_id)
        return {"document_id": plan.document_id, "title": plan.title, "confirmation_token": plan.confirmation_token}
    except PrivateKnowledgeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/private/documents/delete-confirm")
def private_delete_confirm(
    payload: PrivateDeleteConfirmRequest,
    request: Request,
) -> dict[str, object]:
    user = _private_user(request)
    try:
        result = _private_service().confirm_and_delete(int(user["id"]), payload.confirmation_token)
        return {
            "document_id": result.document_id,
            "deleted_source_file": result.deleted_source_file,
            "deleted_vector_rows": result.deleted_vector_rows,
        }
    except PrivateKnowledgeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/admin/users")
def admin_users(request: Request) -> dict[str, object]:
    _require_admin(request)
    return {"items": list_users(settings.sqlite_path)}


@app.post("/api/admin/users/{target_user_id}/reset-password")
def admin_reset_password(target_user_id: int, request: Request) -> dict[str, object]:
    admin = _require_admin(request)
    try:
        return reset_user_password(
            settings.sqlite_path,
            actor_user_id=int(admin["id"]),
            target_user_id=target_user_id,
        )
    except AdminError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/admin/documents")
def admin_documents(
    request: Request,
    q: str = "",
    module: str = "",
    content_status: str = "",
    validity_status: str = "",
    processing_status: str = "",
    limit: int = 100,
    offset: int = 0,
) -> dict[str, object]:
    _require_admin(request)
    return list_public_documents(
        settings.sqlite_path,
        query=q[:100],
        module=module[:40],
        content_status=content_status[:20],
        validity_status=validity_status[:10],
        processing_status=processing_status[:10],
        limit=max(1, min(limit, 200)),
        offset=max(0, offset),
    )


@app.post("/api/admin/documents/upload")
async def admin_upload_documents(request: Request) -> dict[str, object]:
    admin = _require_admin(request)
    queue = public_processing_queue
    if queue is None:
        raise HTTPException(status_code=503, detail="公共资料处理队列尚未启动。")
    try:
        content_length = int(request.headers.get("content-length", "0"))
    except ValueError:
        content_length = 0
    if content_length > 200 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="本次上传总大小不能超过 200 MB。")
    try:
        parts = parse_multipart(request.headers.get("content-type", ""), await request.body())
        module_parts = [part for part in parts if part.field_name == "module" and part.filename is None]
        module = module_parts[0].content.decode("utf-8", errors="replace").strip() if module_parts else ""
        allowed_modules = {"学生资助", "教务", "学生事务", "招生", "就业", "公寓与校园生活", "学校概况"}
        if module not in allowed_modules:
            raise UploadParseError("必须选择一个允许的公共资料模块。")
        files = [part for part in parts if part.filename]
        if not files:
            raise UploadParseError("没有找到上传文件。")
        if len(files) > 20:
            raise UploadParseError("一次最多上传 20 个文件。")
        results: list[dict[str, object]] = []
        upload_root = settings.public_source_root.resolve() / module
        upload_root.mkdir(parents=True, exist_ok=True)
        for file_part in files:
            filename = safe_upload_filename(file_part.filename or "")
            if len(file_part.content) > MAX_UPLOAD_BYTES:
                results.append({"filename": filename, "accepted": False, "message": "单个文件不能超过 50 MB。"})
                continue
            target = upload_root / filename
            if target.exists():
                target = upload_root / f"{target.stem}-上传-{uuid4().hex[:8]}{target.suffix}"
            target.write_bytes(file_part.content)
            try:
                queued = queue.enqueue(target)
            except QueueError as exc:
                target.unlink(missing_ok=True)
                results.append({"filename": filename, "accepted": False, "message": str(exc)})
                continue
            connection = connect_database(settings.sqlite_path)
            try:
                with connection:
                    record_operation(
                        connection,
                        actor_user_id=int(admin["id"]),
                        action="上传资料",
                        target_type="knowledge_document",
                        target_id=str(queued.document_id),
                        detail={"filename": target.name, "module": module, "status": queued.status},
                    )
            finally:
                connection.close()
            results.append({"filename": target.name, "accepted": queued.accepted, "status": queued.status, "document_id": queued.document_id, "task_id": queued.task_id})
        return {"items": results}
    except UploadParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.patch("/api/admin/documents/{document_id}")
def admin_update_document(
    document_id: int,
    payload: AdminDocumentQuery,
    request: Request,
) -> dict[str, object]:
    _require_admin(request)
    try:
        ensure_public_document(settings.sqlite_path, document_id)
    except AdminError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    fields = {key: value for key, value in payload.model_dump().items() if value is not None}
    if not fields:
        raise HTTPException(status_code=400, detail="没有需要更新的资料字段。")
    try:
        metadata = update_document_metadata(
            settings.sqlite_path,
            document_id,
            **fields,
        )
        return {
            "document_id": metadata.document_id,
            "title": metadata.title,
            "module": metadata.module,
            "source_label": metadata.source_label,
            "source_url": metadata.source_url,
            "published_date": metadata.published_date,
            "publisher_department": metadata.publisher_department,
            "content_status": metadata.content_status,
            "validity_status": metadata.validity_status,
        }
    except MetadataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _set_public_document_validity(
    document_id: int,
    request: Request,
    validity_status: str,
    action: str,
) -> dict[str, object]:
    admin = _require_admin(request)
    try:
        ensure_public_document(settings.sqlite_path, document_id)
        metadata = update_document_metadata(
            settings.sqlite_path,
            document_id,
            validity_status=validity_status,
        )
        connection = connect_database(settings.sqlite_path)
        try:
            with connection:
                record_operation(
                    connection,
                    actor_user_id=int(admin["id"]),
                    action=action,
                    target_type="knowledge_document",
                    target_id=str(document_id),
                    detail={"title": metadata.title, "validity_status": validity_status},
                )
        finally:
            connection.close()
        return {"document_id": document_id, "validity_status": validity_status}
    except MetadataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AdminError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/admin/documents/{document_id}/disable")
def admin_disable_document(document_id: int, request: Request) -> dict[str, object]:
    return _set_public_document_validity(document_id, request, "无效", "停用资料")


@app.post("/api/admin/documents/{document_id}/restore")
def admin_restore_document(document_id: int, request: Request) -> dict[str, object]:
    return _set_public_document_validity(document_id, request, "有效", "恢复资料")


@app.get("/api/admin/documents/{document_id}/preview")
def admin_document_preview(document_id: int, request: Request) -> dict[str, object]:
    _require_admin(request)
    connection = connect_database(settings.sqlite_path)
    try:
        row = connection.execute(
            """
            SELECT id, title, module, source_text, source_label, source_url,
                   published_date, publisher_department, content_status, validity_status,
                   processing_status
            FROM knowledge_documents
            WHERE id = ? AND is_private = 0
            """,
            (document_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="找不到公共资料。")
        return {key: row[key] for key in row.keys()}
    finally:
        connection.close()


@app.get("/api/admin/documents/{document_id}/raw")
def admin_document_raw(document_id: int, request: Request) -> FileResponse:
    """下载公共资料原文件；路径必须由受控 stable_key 映射到公共根目录。"""

    _require_admin(request)
    try:
        source_path = public_source_path(
            settings.sqlite_path,
            settings.public_source_root,
            document_id,
        )
    except AdminError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not source_path.is_file():
        raise HTTPException(status_code=404, detail="原始资料文件不存在。")
    return FileResponse(path=source_path, filename=source_path.name)


@app.post("/api/admin/documents/batch-validity")
def admin_batch_validity(
    payload: DocumentBatchValidityRequest,
    request: Request,
) -> dict[str, object]:
    admin = _require_admin(request)
    document_ids = list(dict.fromkeys(payload.document_ids))
    try:
        for document_id in document_ids:
            ensure_public_document(settings.sqlite_path, document_id)
        updated: list[dict[str, object]] = []
        for document_id in document_ids:
            metadata = update_document_metadata(
                settings.sqlite_path,
                document_id,
                validity_status=payload.validity_status,
            )
            updated.append({"document_id": document_id, "title": metadata.title})
        connection = connect_database(settings.sqlite_path)
        try:
            with connection:
                record_operation(
                    connection,
                    actor_user_id=int(admin["id"]),
                    action="批量更新资料时效",
                    target_type="knowledge_documents",
                    detail={"count": len(updated), "validity_status": payload.validity_status},
                )
        finally:
            connection.close()
        return {"updated": updated, "validity_status": payload.validity_status}
    except (AdminError, MetadataError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/admin/logs")
def admin_logs(request: Request, limit: int = 100) -> dict[str, object]:
    _require_admin(request)
    return {"items": list_operation_logs(settings.sqlite_path, limit=max(1, min(limit, 200)))}


@app.get("/api/admin/processing")
def admin_processing(request: Request, limit: int = 100) -> dict[str, object]:
    _require_admin(request)
    queue = public_processing_queue
    return {
        "pending_count": queue.pending_count if queue else 0,
        "active_task_id": queue.active_task_id if queue else None,
        "items": list_processing_tasks(settings.sqlite_path, limit=max(1, min(limit, 200))),
    }


@app.post("/api/admin/documents/{document_id}/retry")
def admin_retry_document(document_id: int, request: Request) -> dict[str, object]:
    admin = _require_admin(request)
    queue = public_processing_queue
    if queue is None:
        raise HTTPException(status_code=503, detail="公共资料处理队列尚未启动。")
    try:
        source_path = public_source_path(
            settings.sqlite_path,
            settings.public_source_root,
            document_id,
        )
        result = queue.enqueue(source_path)
        connection = connect_database(settings.sqlite_path)
        try:
            with connection:
                record_operation(
                    connection,
                    actor_user_id=int(admin["id"]),
                    action="重试资料处理",
                    target_type="knowledge_document",
                    target_id=str(document_id),
                    detail={"status": result.status, "source_path": source_path.name},
                )
        finally:
            connection.close()
        return result.__dict__
    except (AdminError, QueueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/admin/processing/{task_id}/cancel")
def admin_cancel_processing(task_id: int, request: Request) -> dict[str, object]:
    admin = _require_admin(request)
    queue = public_processing_queue
    if queue is None:
        raise HTTPException(status_code=503, detail="公共资料处理队列尚未启动。")
    try:
        result = queue.cancel(task_id)
        if result.accepted:
            connection = connect_database(settings.sqlite_path)
            try:
                with connection:
                    record_operation(
                        connection,
                        actor_user_id=int(admin["id"]),
                        action="取消资料处理",
                        target_type="processing_task",
                        target_id=str(task_id),
                        detail={"status": result.status, "message": result.message},
                    )
            finally:
                connection.close()
        return result.__dict__
    except QueueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/admin/documents/{document_id}/delete-plan")
def admin_delete_document_plan(document_id: int, request: Request) -> dict[str, object]:
    _require_admin(request)
    service = public_deletion_service
    if service is None:
        raise HTTPException(status_code=503, detail="公共资料删除服务尚未启动。")
    try:
        source_path = public_source_path(
            settings.sqlite_path,
            settings.public_source_root,
            document_id,
        )
        plan = service.create_plan(document_id, source_path=source_path)
        return {
            "document_id": plan.document_id,
            "title": plan.title,
            "source_path": str(plan.source_path) if plan.source_path else None,
            "chunk_count": plan.chunk_count,
            "source_record_count": plan.source_record_count,
            "task_count": plan.task_count,
            "confirmation_token": plan.confirmation_token,
        }
    except (AdminError, DeletionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/admin/documents/batch-delete-plan")
def admin_batch_delete_plan(
    payload: DocumentBatchDeletePlanRequest,
    request: Request,
) -> dict[str, object]:
    _require_admin(request)
    service = public_deletion_service
    if service is None:
        raise HTTPException(status_code=503, detail="公共资料删除服务尚未启动。")
    document_ids = list(dict.fromkeys(payload.document_ids))
    try:
        source_paths: dict[int, Path] = {}
        for document_id in document_ids:
            source_paths[document_id] = public_source_path(
                settings.sqlite_path,
                settings.public_source_root,
                document_id,
            )
        plans = [service.create_plan(document_id, source_path=source_paths[document_id]) for document_id in document_ids]
        return {
            "plans": [
                {
                    "document_id": plan.document_id,
                    "title": plan.title,
                    "chunk_count": plan.chunk_count,
                    "source_record_count": plan.source_record_count,
                    "task_count": plan.task_count,
                    "confirmation_token": plan.confirmation_token,
                }
                for plan in plans
            ]
        }
    except (AdminError, DeletionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/admin/documents/delete-confirm")
def admin_delete_document_confirm(
    payload: DocumentDeleteConfirmRequest,
    request: Request,
) -> dict[str, object]:
    admin = _require_admin(request)
    service = public_deletion_service
    if service is None:
        raise HTTPException(status_code=503, detail="公共资料删除服务尚未启动。")
    try:
        result = service.confirm_and_delete(payload.confirmation_token)
        connection = connect_database(settings.sqlite_path)
        try:
            with connection:
                record_operation(
                    connection,
                    actor_user_id=int(admin["id"]),
                    action="删除资料",
                    target_type="knowledge_document",
                    target_id=str(result.document_id),
                    detail={"deleted_source_file": result.deleted_source_file, "deleted_vector_rows": result.deleted_vector_rows},
                )
        finally:
            connection.close()
        return {
            "document_id": result.document_id,
            "deleted_source_file": result.deleted_source_file,
            "deleted_vector_rows": result.deleted_vector_rows,
        }
    except DeletionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/admin/documents/batch-delete-confirm")
def admin_batch_delete_confirm(
    payload: DocumentBatchDeleteConfirmRequest,
    request: Request,
) -> dict[str, object]:
    admin = _require_admin(request)
    service = public_deletion_service
    if service is None:
        raise HTTPException(status_code=503, detail="公共资料删除服务尚未启动。")
    deleted: list[dict[str, object]] = []
    failed: list[dict[str, str]] = []
    for token in payload.confirmation_tokens:
        try:
            result = service.confirm_and_delete(token)
            connection = connect_database(settings.sqlite_path)
            try:
                with connection:
                    record_operation(
                        connection,
                        actor_user_id=int(admin["id"]),
                        action="批量删除资料",
                        target_type="knowledge_document",
                        target_id=str(result.document_id),
                        detail={"deleted_source_file": result.deleted_source_file, "deleted_vector_rows": result.deleted_vector_rows},
                    )
            finally:
                connection.close()
            deleted.append({
                "document_id": result.document_id,
                "deleted_source_file": result.deleted_source_file,
                "deleted_vector_rows": result.deleted_vector_rows,
            })
        except DeletionError as exc:
            failed.append({"confirmation_token": token, "message": str(exc)})
    return {"deleted": deleted, "failed": failed}


@app.get("/api/conversations")
def conversations(request: Request, q: str = "") -> dict[str, object]:
    user = _require_user(request)
    return {
        "items": list_conversations(
            settings.sqlite_path,
            user_id=int(user["id"]),
            query=q[:80],
        )
    }


@app.post("/api/conversations/guest-import")
def guest_import(payload: GuestImportRequest, request: Request) -> dict[str, int]:
    user = _require_user(request)
    try:
        conversation_id = import_guest_entries(
            settings.sqlite_path,
            user_id=int(user["id"]),
            entries=[entry.model_dump() for entry in payload.entries],
            identity=payload.identity,
        )
        return {"conversation_id": conversation_id}
    except ConversationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/conversations/{conversation_id}")
def conversation_detail(conversation_id: int, request: Request) -> dict[str, object]:
    user = _require_user(request)
    try:
        return get_conversation(
            settings.sqlite_path,
            user_id=int(user["id"]),
            conversation_id=conversation_id,
        )
    except ConversationPermissionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/api/conversations/{conversation_id}")
def conversation_update(
    conversation_id: int,
    payload: ConversationUpdateRequest,
    request: Request,
) -> dict[str, object]:
    user = _require_user(request)
    try:
        return update_conversation(
            settings.sqlite_path,
            user_id=int(user["id"]),
            conversation_id=conversation_id,
            title=payload.title,
            is_pinned=payload.is_pinned,
        )
    except ConversationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/conversations/{conversation_id}")
def conversation_delete(conversation_id: int, request: Request) -> dict[str, int]:
    user = _require_user(request)
    try:
        deleted = delete_conversations(
            settings.sqlite_path,
            user_id=int(user["id"]),
            conversation_ids=[conversation_id],
        )
        return {"deleted": deleted}
    except ConversationError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/conversations/batch-delete")
def conversation_batch_delete(
    payload: ConversationBatchDeleteRequest,
    request: Request,
) -> dict[str, int]:
    user = _require_user(request)
    try:
        deleted = delete_conversations(
            settings.sqlite_path,
            user_id=int(user["id"]),
            conversation_ids=payload.conversation_ids,
        )
        return {"deleted": deleted}
    except ConversationError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/search")
def search(request: SearchRequest, http_request: Request) -> dict[str, object]:
    try:
        user = _public_user_from_request(http_request)
        if request.use_private_knowledge and user is None:
            raise HTTPException(status_code=401, detail="使用私人知识库前请先登录。")
        if request.use_private_knowledge and user is not None and user["role"] != "user":
            raise HTTPException(status_code=403, detail="管理员没有私人知识库。")
        scope = resolve_scope(request.question, request.scope)
        return search_knowledge(
            request.question,
            db_path=settings.sqlite_path,
            index_path=settings.lancedb_path,
            ollama_base_url=settings.ollama_base_url,
            embedding_model=settings.ollama_embedding_model,
            scope=scope,
            limit=request.limit,
            owner_user_id=int(user["id"]) if request.use_private_knowledge and user and user["role"] == "user" else None,
            include_private=bool(request.use_private_knowledge and user and user["role"] == "user"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RetrievalError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/answer")
def answer(request: SearchRequest, http_request: Request) -> dict[str, object]:
    user = _public_user_from_request(http_request)
    if user is not None and user["must_change_password"]:
        raise HTTPException(status_code=403, detail="请先在个人中心修改密码。")
    if request.use_private_knowledge and user is None:
        raise HTTPException(status_code=401, detail="使用私人知识库前请先登录。")
    if request.use_private_knowledge and user is not None and user["role"] != "user":
        raise HTTPException(status_code=403, detail="管理员没有私人知识库。")
    if request.conversation_id is not None:
        if user is None:
            raise HTTPException(status_code=401, detail="请先登录后再继续该历史会话。")
        try:
            ensure_conversation_owner(
                settings.sqlite_path,
                user_id=int(user["id"]),
                conversation_id=request.conversation_id,
            )
        except ConversationPermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
    try:
        plan = _dialogue_plan(request, user)
        if plan["mode"] == "general":
            return _greeting_payload(request, http_request, user, plan["reply"])
        scope = resolve_scope(plan["query"], request.scope)
        retrieval = search_knowledge(
            plan["query"],
            db_path=settings.sqlite_path,
            index_path=settings.lancedb_path,
            ollama_base_url=settings.ollama_base_url,
            embedding_model=settings.ollama_embedding_model,
            scope=scope,
            limit=request.limit,
            owner_user_id=int(user["id"]) if request.use_private_knowledge and user and user["role"] == "user" else None,
            include_private=bool(request.use_private_knowledge and user and user["role"] == "user"),
        )
        generated = generate_answer(
            plan["query"],
            retrieval,
            base_url=settings.ollama_base_url,
            model=settings.ollama_generation_model,
        )
        sources = _sources_for_ranks(retrieval, generated["used_source_ranks"])
        saved, saved_conversation_id = _save_answer_if_logged_in(
            http_request,
            user,
            question=request.question,
            answer=generated["answer"],
            answer_status=str(generated["answer_status"]),
            sources=sources,
            identity=request.identity,
            conversation_id=request.conversation_id,
        )
        return {
            "question": request.question.strip(),
            "answer_status": generated["answer_status"],
            "answer": generated["answer"],
            "sources": sources,
            "retrieval": retrieval["retrieval"],
            "saved": saved,
            "conversation_id": saved_conversation_id,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (RetrievalError, GenerationError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/answer/stream")
def answer_stream(request: SearchRequest, http_request: Request) -> StreamingResponse:
    user = _public_user_from_request(http_request)
    if user is not None and user["must_change_password"]:
        raise HTTPException(status_code=403, detail="请先在个人中心修改密码。")
    if request.use_private_knowledge and user is None:
        raise HTTPException(status_code=401, detail="使用私人知识库前请先登录。")
    if request.use_private_knowledge and user is not None and user["role"] != "user":
        raise HTTPException(status_code=403, detail="管理员没有私人知识库。")
    if request.conversation_id is not None and user is None:
        raise HTTPException(status_code=401, detail="请先登录后再继续该历史会话。")
    if request.conversation_id is not None and user is not None:
        try:
            ensure_conversation_owner(
                settings.sqlite_path,
                user_id=int(user["id"]),
                conversation_id=request.conversation_id,
            )
        except ConversationPermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
    try:
        plan = _dialogue_plan(request, user, draft=False)
    except GenerationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if plan["mode"] == "general":
        def greeting_events():
            yield _sse("progress", {"stage": "understanding", "label": "已理解问题，正在整理回答"})
            parts = []
            try:
                for text in stream_ollama_chat(
                    general_messages(request.question, plan["history"]), settings.ollama_base_url,
                    settings.ollama_generation_model, json_mode=False,
                ):
                    parts.append(text)
                    yield _sse("answer_delta", {"text": text})
            except GenerationError as exc:
                yield _sse("error", {"message": str(exc)})
                return
            payload = _greeting_payload(request, http_request, user, "".join(parts))
            yield _sse("final", payload)
            yield _sse("done", {"saved": payload["saved"], "conversation_id": payload["conversation_id"]})

        return StreamingResponse(
            greeting_events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    try:
        scope = resolve_scope(plan["query"], request.scope)
        retrieval = search_knowledge(
            plan["query"],
            db_path=settings.sqlite_path,
            index_path=settings.lancedb_path,
            ollama_base_url=settings.ollama_base_url,
            embedding_model=settings.ollama_embedding_model,
            scope=scope,
            limit=request.limit,
            owner_user_id=int(user["id"]) if request.use_private_knowledge and user and user["role"] == "user" else None,
            include_private=bool(request.use_private_knowledge and user and user["role"] == "user"),
        )
    except (ValueError, RetrievalError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    def events():
        yield _sse("progress", {"stage": "retrieving", "label": "正在检索学校官方资料"})
        if not has_reliable_evidence(retrieval):
            yield _sse("progress", {"stage": "checking", "label": "正在核对资料适用范围"})
            result = structure_answer("", retrieval)
            sources: list[dict[str, object]] = []
            saved, saved_conversation_id = _save_answer_if_logged_in(
                http_request,
                user,
                question=request.question,
                answer=result["answer"],
                answer_status=str(result["answer_status"]),
                sources=sources,
                identity=request.identity,
                conversation_id=request.conversation_id,
            )
            payload = {
                "question": request.question.strip(),
                "answer_status": result["answer_status"],
                "answer": result["answer"],
                "sources": sources,
                "retrieval": retrieval["retrieval"],
                "saved": saved,
                "conversation_id": saved_conversation_id,
            }
            yield _sse("final", payload)
            yield _sse("done", {"saved": saved, "conversation_id": saved_conversation_id})
            return

        reliable_hits = reliable_evidence_hits(retrieval)
        yield _sse("progress", {"stage": "checking", "label": "已找到资料，正在核对证据"})
        raw_parts: list[str] = []
        try:
            yield _sse("progress", {"stage": "answering", "label": "正在组织回答"})
            try:
                for delta in stream_ollama_chat(
                    _build_prompt(plan["query"], reliable_hits),
                    settings.ollama_base_url,
                    settings.ollama_generation_model,
                    timeout=240.0,
                ):
                    raw_parts.append(delta)
                    yield _sse("delta", {"text": delta})
            except GenerationError as stream_error:
                # 流式接口可能在 JSON 闭合前达到 num_predict 上限，或中途断开。
                # 此时复用非流式接口的严格/精简重试，不把内部截断原因直接展示给用户。
                try:
                    result = generate_answer(
                        plan["query"],
                        retrieval,
                        base_url=settings.ollama_base_url,
                        model=settings.ollama_generation_model,
                    )
                except GenerationError as fallback_error:
                    raise stream_error from fallback_error
            else:
                result = finalize_stream_answer(
                    plan["query"],
                    "".join(raw_parts),
                    retrieval,
                    base_url=settings.ollama_base_url,
                    model=settings.ollama_generation_model,
                )
            sources = _sources_for_ranks(retrieval, result["used_source_ranks"])
            saved, saved_conversation_id = _save_answer_if_logged_in(
                http_request,
                user,
                question=request.question,
                answer=result["answer"],
                answer_status=str(result["answer_status"]),
                sources=sources,
                identity=request.identity,
                conversation_id=request.conversation_id,
            )
            payload = {
                "question": request.question.strip(),
                "answer_status": result["answer_status"],
                "answer": result["answer"],
                "sources": sources,
                "retrieval": retrieval["retrieval"],
                "saved": saved,
                "conversation_id": saved_conversation_id,
            }
            yield _sse("final", payload)
            yield _sse("done", {"saved": saved, "conversation_id": saved_conversation_id})
        except GeneratorExit:
            # 客户端停止或断开时，不产生 final 事件，也不写入会话。
            return
        except (GenerationError, ValueError) as exc:
            yield _sse("error", {"message": str(exc), "saved": False})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
