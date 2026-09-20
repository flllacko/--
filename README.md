# 学生服务问答助手应用工程

> 阶段：P4 项目基础工程（P4-S01～P4-S04）  
> 对应需求：`REQ-SCOPE-006`、`REQ-AUTH-009`、`REQ-CHAT-004`、`REQ-PKB-008`、`REQ-PKB-009`

## 当前范围

这里是应用代码目录，不是资料正文目录。

- `frontend/`：React + Vite + TypeScript 页面骨架，当前先完成聊天主界面、导航和已确认页面的占位交互。
- `backend/`：FastAPI 最小服务、配置读取和 SQLite 结构初始化。
- `runtime-data/`：应用运行数据目录，和 `D:\sturjh\codexschool\data` 分开，避免覆盖两份原始基线文件。

当前版本还没有接入 RAG、模型问答、真实登录、公共资料处理或私人资料上传。页面上的占位提示会明确说明这一点，不伪造成功结果。

## 本地启动

### 前端

```powershell
Set-Location .\frontend
npm install
npm run dev
```

浏览器打开 `http://localhost:5173`。

### 后端

```powershell
Set-Location ..\backend
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

健康检查地址：`http://127.0.0.1:8000/api/health`。

## 配置原则

模型名称、Ollama 地址、资料正文目录和运行数据目录通过环境变量配置，示例见前后端各自的 `.env.example`。首次接入 RAG 时再安装 LanceDB 和嵌入模型，不在 P4 阶段提前引入大型依赖。
