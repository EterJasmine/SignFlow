import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import upload_batch  # noqa: E402


def test_scan_skips_normalized_and_non_video(tmp_path):
    (tmp_path / 'a.mp4').write_bytes(b'x')
    (tmp_path / 'b.normalized.mp4').write_bytes(b'x')
    (tmp_path / 'notes.txt').write_bytes(b'x')
    sub = tmp_path / 'sub'
    sub.mkdir()
    (sub / 'c.MOV').write_bytes(b'x')

    found, notes = upload_batch.scan([str(tmp_path)])
    names = sorted(item.name for item in found)
    assert names == ['a.mp4', 'c.MOV']
    assert notes == []


def test_scan_reports_missing_path():
    found, notes = upload_batch.scan(['不存在的目录'])
    assert found == [] and notes and '路径不存在' in notes[0]


def test_upload_one_polls_until_completed(tmp_path, monkeypatch):
    clip = tmp_path / 'clip.mp4'
    clip.write_bytes(b'x' * 16)
    polls = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == 'POST' and path.endswith('/videos'):
            assert request.headers['content-type'].startswith('multipart/form-data')
            return httpx.Response(201, json={'id': 'video_1', 'duration_ms': 2000})
        if request.method == 'POST' and path.endswith('/analysis-jobs'):
            return httpx.Response(202, json={'id': 'job_1', 'status': 'queued'})
        if path == '/api/v1/jobs/job_1':
            polls.append(1)
            status = 'completed' if len(polls) >= 2 else 'processing'
            return httpx.Response(200, json={'id': 'job_1', 'status': status})
        raise AssertionError(f'未预期的请求 {request.method} {path}')

    client = httpx.Client(base_url='http://test', transport=httpx.MockTransport(handler))
    row = upload_batch.upload_one(client, 'project_1', clip, analyze=True, timeout=30, poll=0)
    assert row['status'] == 'analyzed' and row['video_id'] == 'video_1'
    assert len(polls) == 2


def test_upload_one_surfaces_server_reason(tmp_path):
    clip = tmp_path / 'broken.mp4'
    clip.write_bytes(b'x')

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={'detail': '视频转码失败'})

    client = httpx.Client(base_url='http://test', transport=httpx.MockTransport(handler))
    row = upload_batch.upload_one(client, 'project_1', clip, analyze=True, timeout=5, poll=0)
    assert row['status'] == 'failed' and '视频转码失败' in row['detail']


def test_upload_one_skips_oversized_before_sending(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_batch, 'MAX_BYTES', 8)
    clip = tmp_path / 'big.mp4'
    clip.write_bytes(b'x' * 32)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError('超限文件不应发出请求')

    client = httpx.Client(base_url='http://test', transport=httpx.MockTransport(handler))
    row = upload_batch.upload_one(client, 'project_1', clip, analyze=True, timeout=5, poll=0)
    assert row['status'] == 'skipped' and '80MB' in row['detail']


def test_ensure_project_reuses_existing_name():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == 'GET':
            return httpx.Response(200, json=[{'id': 'project_old', 'name': '批量导入'}])
        raise AssertionError('已存在同名项目时不应新建')

    client = httpx.Client(base_url='http://test', transport=httpx.MockTransport(handler))
    assert upload_batch.ensure_project(client, '批量导入') == 'project_old'
