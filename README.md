# 学生服务问答助手

这是成都理工大学工程技术学院学生服务问答助手的应用代码。

- backend/：FastAPI 后端，负责资料检索、问答、账号、管理员功能和私人知识库。
- frontend/：React + TypeScript 前端，负责聊天、登录、历史会话和资料管理页面。
- 模型：Ollama 本地运行 qwen3.5:9b 和 qwen3-embedding:0.6b。
- 资料正文和项目文档由上一级 Obsidian 项目库管理。

## 架构

~~~text
资料正文
  ↓ 提取、清洗、切片、向量化
SQLite + LanceDB
  ↑ 检索
FastAPI 后端
  ├─ 账号、会话、资料和任务管理
  ├─ 公共资料与私人资料的权限隔离
  └─ Ollama：生成回答和生成向量
  ↑ HTTP / SSE
React + TypeScript 前端
~~~

- SQLite 保存账号、会话、资料元信息和处理任务，是业务状态记录。
- LanceDB 保存向量检索数据，用于查找相关资料片段。
- Ollama 在本机运行回答模型和嵌入模型，不调用第三方联网搜索。
- 前端只通过 FastAPI 访问数据；私人资料由后端按用户身份隔离。

## 第一次运行

后端：

~~~powershell
cd "D:\sturjh\codexschool\成都理工大学工程技术学院问答助手\应用\backend"
uv venv
uv pip install -r requirements.txt
~~~

前端：

~~~powershell
cd "D:\sturjh\codexschool\成都理工大学工程技术学院问答助手\应用\frontend"
npm install
~~~

## 以后每次运行

打开两个 PowerShell 窗口。

窗口一，启动后端：

~~~powershell
cd "D:\sturjh\codexschool\成都理工大学工程技术学院问答助手\应用\backend"
uv run uvicorn app.main:app
~~~

窗口二，启动前端：

~~~powershell
cd "D:\sturjh\codexschool\成都理工大学工程技术学院问答助手\应用\frontend"
npm run dev
~~~

浏览器访问：

~~~text
http://127.0.0.1:5173
~~~

停止服务：在对应窗口按 Ctrl + C。

## 检查服务

后端健康检查：

~~~text
http://127.0.0.1:8000/api/health
~~~

能返回 ok 就说明后端正常。

## 常用检查命令

后端测试：

~~~powershell
cd backend
uv run python -m unittest discover -s tests -q
~~~

前端检查：

~~~powershell
cd frontend
npm run typecheck
npm run build
~~~

## 注意事项

- Ollama 必须正在运行，并且已经有两个项目模型。
- uv venv、uv pip install 和 npm install 只需要第一次运行时执行。
- runtime-data/ 保存本地数据库、向量索引、日志和私人资料，不要随意删除。
- 私人资料只属于上传用户，不会进入公共知识库。
- 当前不支持 OCR、扫描 PDF、图片识别和第三方联网搜索。
