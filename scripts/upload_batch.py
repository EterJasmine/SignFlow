"""批量上传手势舞素材并跑完分析。

服务必须先启动（双击工作区根目录 ``启动手势舞.cmd``，或自己起 uvicorn）。

用法::

    python scripts/upload_batch.py "D:/我的手势舞素材"
    python scripts/upload_batch.py a.mp4 b.mp4 --no-analyze
    python scripts/upload_batch.py 素材目录 --skip-existing

文件夹会被递归扫描，只取 mp4/mov/webm/avi/mkv；``*.normalized.mp4`` 是服务自己生成的
中间文件，会被跳过。默认串行上传并等待每一条分析完成，因为本地 worker 一次只处理一个任务。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

VIDEO_SUFFIXES = {'.mp4', '.mov', '.webm', '.avi', '.mkv'}
MAX_BYTES = 80 * 1024 * 1024
DEFAULT_SERVER = 'http://127.0.0.1:8000'


def scan(paths: list[str]) -> tuple[list[Path], list[str]]:
    """展开文件与目录，返回 (待上传文件, 跳过说明)。"""
    found: list[Path] = []
    notes: list[str] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            candidates = sorted(p for p in path.rglob('*') if p.is_file())
        elif path.is_file():
            candidates = [path]
        else:
            notes.append(f'{raw}: 路径不存在')
            continue
        for item in candidates:
            name = item.name
            if name.lower().endswith('.normalized.mp4'):
                continue
            if item.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            if name.startswith('.'):
                continue
            if item in found:
                continue
            found.append(item)
    return found, notes


def ensure_project(client: httpx.Client, name: str) -> str:
    projects = client.get('/api/v1/projects').json()
    for project in projects:
        if project.get('name') == name:
            return project['id']
    response = client.post('/api/v1/projects', json={'name': name})
    response.raise_for_status()
    return response.json()['id']


def upload_one(client: httpx.Client, project_id: str, path: Path, analyze: bool,
               timeout: float, poll: float = 3.0) -> dict:
    """上传单个文件，可选等待分析完成。返回一条结果记录。"""
    size = path.stat().st_size
    if size > MAX_BYTES:
        return {'file': path.name, 'status': 'skipped', 'detail': f'超过 80MB（{size/1048576:.1f}MB）'}
    with path.open('rb') as handle:
        response = client.post('/api/v1/projects/' + project_id + '/videos',
                               files={'file': (path.name, handle, 'application/octet-stream')},
                               timeout=httpx.Timeout(300.0))
    if response.status_code >= 400:
        reason = ''
        try:
            reason = response.json().get('detail') or ''
        except ValueError:
            reason = response.text[:120]
        return {'file': path.name, 'status': 'failed', 'detail': f'上传失败 {response.status_code} {reason}'}
    video = response.json()
    record = {'file': path.name, 'video_id': video['id'], 'duration_ms': video.get('duration_ms'),
              'status': 'uploaded', 'detail': ''}
    if not analyze:
        return record
    response = client.post('/api/v1/videos/' + video['id'] + '/analysis-jobs')
    if response.status_code >= 400:
        record.update(status='failed', detail=f'启动分析失败 {response.status_code}')
        return record
    job_id = response.json()['id']
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get('/api/v1/jobs/' + job_id).json()
        if job['status'] == 'completed':
            record['status'] = 'analyzed'
            return record
        if job['status'] in ('failed', 'cancelled'):
            record.update(status=job['status'], detail=str(job.get('error') or '')[:120])
            return record
        time.sleep(poll)
    record.update(status='timeout', detail=f'超过 {timeout:.0f} 秒未完成，任务仍在后台跑')
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='批量上传手势舞素材并分析')
    parser.add_argument('paths', nargs='+', help='视频文件或文件夹')
    parser.add_argument('--server', default=DEFAULT_SERVER)
    parser.add_argument('--project', default='批量导入')
    parser.add_argument('--no-analyze', action='store_true', help='只上传，不启动分析')
    parser.add_argument('--timeout', type=float, default=900, help='单条分析等待上限（秒）')
    parser.add_argument('--dry-run', action='store_true', help='只列出将要上传的文件')
    args = parser.parse_args(argv)

    files, notes = scan(args.paths)
    for note in notes:
        print('•', note)
    if not files:
        print('没有找到可上传的视频（支持 mp4/mov/webm/avi/mkv）。')
        return 2
    print(f'找到 {len(files)} 个视频：')
    for item in files:
        print(f'  - {item}  ({item.stat().st_size/1048576:.1f} MB)')
    if args.dry_run:
        return 0

    try:
        # 本机地址不走系统代理：机器上装了 http_proxy 时，httpx 会把 127.0.0.1 也代理走，
        # 表现是 health 直接 502。
        local = any(host in args.server for host in ('127.0.0.1', 'localhost', '::1'))
        with httpx.Client(base_url=args.server, timeout=httpx.Timeout(60.0),
                          trust_env=not local) as client:
            health = client.get('/api/v1/health')
            health.raise_for_status()
            project_id = ensure_project(client, args.project)
            print(f'\n项目 {args.project} → {project_id}\n')
            rows = []
            for index, path in enumerate(files, 1):
                print(f'[{index}/{len(files)}] {path.name} …', end=' ', flush=True)
                try:
                    row = upload_one(client, project_id, path, not args.no_analyze, args.timeout)
                except httpx.HTTPError as exc:
                    row = {'file': path.name, 'status': 'failed', 'detail': f'网络错误 {exc}'[:120]}
                rows.append(row)
                print(row['status'], row.get('detail') or '')
    except httpx.HTTPError as exc:
        print(f'连接服务失败：{exc}\n先启动服务：双击工作区根目录的 启动手势舞.cmd')
        return 3

    print('\n结果：')
    for row in rows:
        detail = f"  {row['detail']}" if row.get('detail') else ''
        duration = f"  {row['duration_ms']}ms" if row.get('duration_ms') else ''
        print(f"  {row['file'][:40]:42s} {row['status']:9s}{duration}{detail}")
    done = sum(1 for row in rows if row['status'] == 'analyzed')
    print(f'\n完成 {done} / {len(rows)}。')
    if done and not args.no_analyze:
        print('\n下一步（新增素材并入待标清单，已标的不受影响）：')
        print('  python scripts/eval_truth.py build --merge')
        print('  python scripts/eval_truth.py label')
    return 0 if done == len(rows) else 1


if __name__ == '__main__':
    sys.exit(main())
