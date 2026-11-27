"""Check that a queued MODE:FILE exits live mode without keyboard input."""
from collections import deque
from queue import Queue
import worker


class FakeCapture:
    def isOpened(self): return True
    def read(self): raise AssertionError('MODE:FILE should stop camera before frame capture')
    def release(self): pass


def test_camera_switch_back_to_file(monkeypatch):
    monkeypatch.setattr(worker.cv2, 'VideoCapture', lambda index: FakeCapture())
    monkeypatch.setattr(worker.cv2, 'destroyAllWindows', lambda: None)
    commands = Queue()
    commands.put('MODE:FILE')
    keep_running = worker.run_camera_stream(None, None, {}, commands, deque(), 'cpu', None)
    assert keep_running is True
