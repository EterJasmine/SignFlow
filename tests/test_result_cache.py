"""结果缓存的行为：同一份文件只解析一次，文件或版本变了就重读。"""
import os
import time

import pytest

from apps.api import store, versions


@pytest.fixture(autouse=True)
def clear_cache():
    versions._RESULT_CACHE.clear()
    yield
    versions._RESULT_CACHE.clear()


def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(store, 'DATA', tmp_path)
    monkeypatch.setattr(versions, 'DATA', tmp_path)
    folder = tmp_path / 'assets' / 'v'
    folder.mkdir(parents=True)
    return folder / 'result.json'


def test_repeated_reads_reuse_the_parsed_result(tmp_path, monkeypatch):
    path = fresh(tmp_path, monkeypatch)
    path.write_text('{"a": 1}', encoding='utf8')
    first = versions.load_result('v')
    assert first is versions.load_result('v')


def test_rewriting_the_file_drops_the_cached_copy(tmp_path, monkeypatch):
    path = fresh(tmp_path, monkeypatch)
    path.write_text('{"a": 1}', encoding='utf8')
    versions.load_result('v')
    path.write_text('{"a": 2}', encoding='utf8')
    os.utime(path, (time.time() + 5, time.time() + 5))
    assert versions.load_result('v')['a'] == 2


def test_committing_a_new_revision_drops_the_cached_copy(tmp_path, monkeypatch):
    path = fresh(tmp_path, monkeypatch)
    path.write_text('{"a": 1, "result_revision": 0}', encoding='utf8')
    versions.load_result('v')
    versions.commit_result('v', {'a': 2}, 'test')
    assert versions.load_result('v')['a'] == 2
