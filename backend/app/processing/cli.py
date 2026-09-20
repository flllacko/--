from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import settings

from .processor import process_markdown_file


def main() -> int:
    parser = argparse.ArgumentParser(description="处理一份 Markdown 公共资料")
    parser.add_argument("--file", required=True, help="资料正文路径，可填写相对资料正文目录的路径")
    parser.add_argument("--source-root", type=Path, default=settings.public_source_root)
    parser.add_argument("--db", type=Path, default=settings.sqlite_path)
    args = parser.parse_args()

    result = process_markdown_file(
        file_path=Path(args.file),
        source_root=args.source_root,
        db_path=args.db,
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return 0 if result.status == "处理成功" else 1


if __name__ == "__main__":
    raise SystemExit(main())
