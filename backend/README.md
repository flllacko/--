# FastAPI 后端

当前后端已完成 P4 基础工程、P5 公共资料处理、P6 RAG 问答、P7 账号与历史会话、P8 公共资料管理后台和 P11 私人知识库。P9 联网增强暂缓，当前问答不把学生问题发送到第三方搜索服务。

## 主要能力

- 公共资料：Markdown、网页正文、PDF、DOCX、Excel 的提取、清洗、切片、Ollama 向量化和 LanceDB 检索。
- 问答：通用问答与学校资料路由、连续追问补全、关键词/向量检索、结构化回答、来源绑定、证据不足说明、SSE 流式输出和停止边界。
- 账号：注册、登录、退出、密码修改、账号注销、访客临时会话和登录用户历史会话。
- 管理员：公共资料上传、单 worker 队列、失败重试、排队取消、批量时效、下载、二次确认删除、用户管理和操作记录。
- 私人知识库：普通用户多文件上传、逐份处理、失败重试、标题修改、下载、删除和当前对话添加自己的资料。

私人资料使用 `owner_user_id` 和 `is_private=1` 隔离。公共检索始终排除私人资料；私人检索只加入当前登录用户自己的“处理成功”资料；管理员接口返回 403，不提供私人正文预览。账号注销会清理私人原文件、切片、向量和数据库记录。

## 重要接口

- `GET /api/health`：健康检查并返回模型配置。
- `POST /api/search`：检索公共资料；登录用户可显式开启 `use_private_knowledge`。
- `POST /api/answer`、`POST /api/answer/stream`：非流式和 SSE 流式问答。
- `/api/auth/*`：注册、登录、当前用户、退出、密码修改和账号注销。
- `/api/conversations/*`：历史会话读取、导入、重命名和删除。
- `/api/private/documents/*`：私人资料列表、上传、重试、标题修改、下载、二次确认删除。
- `/api/admin/*`：管理员用户、公共资料、处理队列和操作记录接口。

## 关键目录

- `app/config.py`：读取环境变量和建立运行目录。
- `app/db/schema.sql`、`app/db/init_db.py`：SQLite 表结构和初始化。
- `app/processing/`：资料提取、清洗、切片、队列和向量化。
- `app/qa/`：检索、回答生成、证据边界和来源绑定。
- `app/private_knowledge.py`：私人资料的用户隔离、列表、删除和文件边界。
- `runtime-data/sqlite/app.db`：业务状态真源，不修改原始资料目录。
- `runtime-data/index/`：LanceDB 检索副本，可以重新构建。
- `runtime-data/private/`：按用户编号隔离的私人原文件。

## 本地启动和配置

```powershell
Set-Location .\backend
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

配置示例见 `.env.example`。当前默认模型为：

```text
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_GENERATION_MODEL=qwen3.5:9b
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b
```

资料回答调用显式设置 `num_ctx=8192`、`num_predict=768` 和 JSON 格式；通用回答使用普通文本流。问题类型和回答解析异常只允许受控重试，不无限调用模型。

## 当前验收结果

- P5 真实批处理：752 份正文、3,642 个切片和 3,642 个向量行。
- P6 自然问法答案级回归：138 条，其中 131 条完成回答、7 条按证据门槛拒答、0 条请求错误。
- 当前后端标准库测试：`75/75` 通过，包含智能路由、宽泛办事意图引导、证据优先级、会话、私人资料隔离、失败处理、删除清理和私人检索范围测试。
- `runtime-data/` 与项目原始 `D:\sturjh\codexschool\data` 分离；原始资料不由后端覆盖。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

私人资料不支持 OCR、扫描 PDF、图片识别和复杂多模态；文件由用户自行负责内容正确性，不进入公共知识库。
