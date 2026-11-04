"""实时识别的边界：没有可用示范、帧数不足、窗口过短，都要给出能照着做的一句话。"""
import pytest

from apps.api import live_routes as live


class FakeEngine:
    def __init__(self, frames=6):
        self.count = frames

    def window(self, start, end):
        return [{'timestamp_ms': start}] * self.count

    def features(self, start, end):
        return {'version': 'v', 'names': [], 'rows': []}


def test_empty_library_explains_the_next_step(monkeypatch):
    monkeypatch.setattr(live, '_usable_templates', lambda: [])
    result = live._recognise(FakeEngine(), 0, 600)
    assert result['matched'] is False
    assert '示范库' in result['reason']


def test_too_few_frames_asks_the_user_to_slow_down(monkeypatch):
    monkeypatch.setattr(live, '_usable_templates', lambda: [{'id': 't', 'meaning': '谢谢',
                                                            'sequence': {'rows': [[1.0]]}}])
    result = live._recognise(FakeEngine(frames=1), 0, 600)
    assert result['matched'] is False
    assert '帧' in result['reason']


def test_window_shorter_than_a_word_is_ignored(monkeypatch):
    monkeypatch.setattr(live, '_usable_templates', lambda: [{'id': 't', 'meaning': '谢谢',
                                                            'sequence': {'rows': [[1.0]]}}])
    assert live._recognise(FakeEngine(), 0, 100) is None


def test_thresholds_stay_ordered():
    assert live.MIN_WORD_MS < live.MAX_WORD_MS
    assert live.QUIET_SPEED < live.ACTIVE_SPEED
    assert live.QUIET_HOLD_MS < live.MAX_WORD_MS
