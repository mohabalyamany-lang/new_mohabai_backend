from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.config import get_settings

settings = get_settings()


class StorageService:
    def __init__(self, base_dir: str = "storage") -> None:
        self.base_path = Path(base_dir)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_bytes(self, filename: str, content: bytes, subdir: str = "uploads") -> str:
        safe_name = filename.replace("/", "_").replace("\\", "_").strip() or "file.bin"
        final_name = f"{uuid4()}_{safe_name}"
        folder = self.base_path / subdir
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / final_name
        path.write_bytes(content)
        return str(path)
