from pathlib import Path

from apps.api import media


def test_video_file_prefers_current_data_dir(monkeypatch, tmp_path):
    videos = tmp_path / 'videos'
    videos.mkdir()
    (videos / 'v1.normalized.mp4').write_bytes(b'x')
    monkeypatch.setattr(media, 'DATA', tmp_path)
    resolved = media.video_file('v1', {'path': 'D:/旧工作区/data/videos/v1.normalized.mp4'})
    assert resolved == videos / 'v1.normalized.mp4'


def test_video_file_falls_back_to_stored_path(monkeypatch, tmp_path):
    empty = tmp_path / 'current'
    empty.mkdir()
    legacy = tmp_path / 'legacy'
    legacy.mkdir()
    (legacy / 'v2.normalized.mp4').write_bytes(b'x')
    monkeypatch.setattr(media, 'DATA', empty)
    assert media.video_file('v2', {'path': str(legacy / 'v2.normalized.mp4')}) == legacy / 'v2.normalized.mp4'
    assert media.video_file('v3', {'path': str(legacy / 'missing.mp4')}) is None
    assert media.video_file('v4', {}) is None
