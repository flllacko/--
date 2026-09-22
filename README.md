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

## 第一次安装

以下命令都假设 PowerShell 当前位于项目的“应用”目录，不依赖项目存放在哪个盘符。

先确认已安装并启动 Ollama，然后准备两个本地模型：

~~~powershell
ollama pull qwen3.5:9b
ollama pull qwen3-embedding:0.6b
~~~

安装后端依赖：

~~~powershell
cd .\backend
uv venv
uv pip install -r requirements.txt
cd ..
~~~

安装前端依赖：

~~~powershell
cd .\frontend
npm install
cd ..
~~~

上面这些安装命令只需执行一次。

## 日常启动：使用终端

需要两个终端窗口，并让两个命令保持运行。

终端一进入 `backend`，启动 FastAPI 后端。后端负责模型调用、知识检索、账号和数据接口，监听 `http://127.0.0.1:8000`：

~~~powershell
cd .\backend
uv run uvicorn app.main:app
~~~

如果已经激活 `.venv`，也可以直接执行 `uvicorn app.main:app`。

终端二进入 `frontend`，启动 React 前端开发服务器。它提供网页界面、自动刷新，并把 `/api` 请求转发给后端，监听 `http://127.0.0.1:5173`：

~~~powershell
cd .\frontend
npm run dev
~~~

浏览器打开：

~~~text
http://127.0.0.1:5173
~~~

停止服务时，在两个终端中分别按 `Ctrl + C`。

## 交付给别人：可选一键启动

完成“第一次安装”后，可以直接双击：

~~~text
启动应用.cmd
~~~

`启动应用.cmd` 会检查 Ollama、两个模型、端口和健康状态，启动前后端后打开浏览器。结束时双击 `停止应用.cmd`；它只停止由脚本启动并记录的本项目前后端，Ollama 保持运行。脚本不会自动安装依赖或下载模型。

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

这条命令运行后端自动测试，不会启动网站。

前端检查：

~~~powershell
cd frontend
npm run typecheck
npm run build
~~~

- `npm run typecheck`：检查 TypeScript 类型，不启动网站。
- `npm run build`：生成可部署的前端文件，不启动开发服务器。

## 注意事项

- Ollama 必须正在运行，并且已经有两个项目模型。
- `ollama pull`、`uv venv`、`uv pip install` 和 `npm install` 通常只需第一次执行；依赖变更后再执行对应安装命令。
- runtime-data/ 保存本地数据库、向量索引、日志和私人资料，不要随意删除。
- 私人资料只属于上传用户，不会进入公共知识库。
- 当前不支持 OCR、扫描 PDF、图片识别和第三方联网搜索。
- 如果出现 `WinError 10048`，表示 8000 端口已有后端运行，不要重复启动；先访问健康检查确认现有服务。
- 通用知识由本地模型回答，可能出错；学校具体规定继续要求知识库证据，资料不足时会明确说明。
