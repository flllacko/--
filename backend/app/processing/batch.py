from __future__ import annotations

import argparse
import sqlite3
import time
from collections import Counter
from pathlib import Path

from app.config import settings

from .markdown import parse_markdown
from .queue import ProcessingQueue


def _candidate_files(source_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in sorted(source_root.rglob("*.md")):
        try:
            parse_markdown(path, source_root)
        except Exception:
            # README 和索引等目录文件不是知识资料，不进入公共向量库。
            continue
        candidates.append(path)
    return candidates


def _status_counts(database: Path) -> Counter[str]:
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            "SELECT status, COUNT(*) FROM processing_tasks GROUP BY status"
        ).fetchall()
        return Counter({str(status): int(count) for status, count in rows})
    finally:
        connection.close()


def run_batch(source_root: Path, database: Path, index_path: Path) -> int:
    files = _candidate_files(source_root)
    queue = ProcessingQueue(source_root, database, index_path)
    try:
        for index, path in enumerate(files, start=1):
            queue.enqueue(path)
            if index % 50 == 0 or index == len(files):
                print(f"enqueued={index}/{len(files)}", flush=True)

        while not queue.wait_for_idle(timeout=5):
            print(f"progress={dict(_status_counts(database))}", flush=True)
        print(f"completed={dict(_status_counts(database))}", flush=True)
    finally:
        queue.close()
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description="批量处理项目库中的 Markdown 公共资料")
    parser.add_argument("--source-root", type=Path, default=settings.public_source_root)
    parser.add_argument("--db", type=Path, default=settings.sqlite_path)
    parser.add_argument("--index", type=Path, default=settings.lancedb_path)
    args = parser.parse_args()
    run_batch(args.source_root.resolve(), args.db.resolve(), args.index.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
