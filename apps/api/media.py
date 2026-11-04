from pathlib import Path

from .store import DATA


def video_file(ident: str, record: dict | None) -> Path | None:
    """按当前数据目录优先定位视频，避免工作区搬家后仍指向旧副本的绝对路径。"""
    current = DATA / 'videos' / (ident + '.normalized.mp4')
    if current.exists():
        return current
    stored = Path((record or {}).get('path') or '')
    if stored.is_file():
        return stored
    fallback = DATA / 'videos' / stored.name if stored.name else None
    if fallback and fallback.is_file():
        return fallback
    return None
