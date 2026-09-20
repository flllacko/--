from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


APP_ROOT = Path(__file__).resolve().parents[2]
VAULT_ROOT = APP_ROOT.parent
BACKEND_ROOT = APP_ROOT / "backend"
load_dotenv(BACKEND_ROOT / ".env")


def resolve_path(value: str, default: Path) -> Path:
    candidate = Path(value) if value else default
    return candidate if candidate.is_absolute() else (APP_ROOT / candidate).resolve()


@dataclass(frozen=True)
class Settings:
    app_env: str
    frontend_origin: str
    data_root: Path
    public_source_root: Path
    private_source_root: Path
    sqlite_path: Path
    lancedb_path: Path
    log_dir: Path
    ollama_base_url: str
    ollama_generation_model: str
    ollama_embedding_model: str
    session_cookie_name: str
    session_ttl_seconds: int

    @classmethod
    def from_environment(cls) -> "Settings":
        runtime_root = APP_ROOT / "runtime-data"
        return cls(
            app_env=os.getenv("APP_ENV", "development"),
            frontend_origin=os.getenv("FRONTEND_ORIGIN", "http://127.0.0.1:5173"),
            data_root=resolve_path(os.getenv("DATA_ROOT", "runtime-data"), runtime_root),
            public_source_root=resolve_path(os.getenv("PUBLIC_SOURCE_ROOT", "../资料正文"), VAULT_ROOT / "资料正文"),
            private_source_root=resolve_path(
                os.getenv("PRIVATE_SOURCE_ROOT", "runtime-data/private"),
                runtime_root / "private",
            ),
            sqlite_path=resolve_path(os.getenv("SQLITE_PATH", "runtime-data/sqlite/app.db"), runtime_root / "sqlite/app.db"),
            lancedb_path=resolve_path(os.getenv("LANCEDB_PATH", "runtime-data/index"), runtime_root / "index"),
            log_dir=resolve_path(os.getenv("LOG_DIR", "runtime-data/logs"), runtime_root / "logs"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            ollama_generation_model=os.getenv("OLLAMA_GENERATION_MODEL", "qwen3.5:9b"),
            ollama_embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
            session_cookie_name=os.getenv("SESSION_COOKIE_NAME", "cdutetc_session"),
            session_ttl_seconds=int(os.getenv("SESSION_TTL_SECONDS", str(7 * 24 * 60 * 60))),
        )

    def ensure_runtime_dirs(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.lancedb_path.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.private_source_root.mkdir(parents=True, exist_ok=True)


settings = Settings.from_environment()
