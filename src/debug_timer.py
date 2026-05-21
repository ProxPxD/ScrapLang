import time
from typing import Any


class Timer:
    def __init__(self, *, default_new_point: bool = False) -> None:
        self._default_new_point = default_new_point
        self._points = {}
        self._times = {}

    def time(self, label: Any = None, *, new_point: bool = None) -> None:
        new_point = self._default_new_point if new_point is None else new_point
        point_label = None if None in self._points else label
        if label is None or point_label not in self._points:
            self._points[label] = time.time()
        else:
            self._times[label] = time.time() - self._points[point_label]
            self._points.pop(point_label)
        if new_point:
            self.time()

    def print_all(self, *, del_printed: bool = False) -> None:
        l_longest_label = max(len(label) for label in self._times if label) if self._times else 0
        for label, t in self._times.items():
            print(f'{label}: {" " * (l_longest_label - len(label)) + str(t)}')
        if del_printed:
            self.clear()

    def clear(self) -> None:
        self._points.clear()
        self._times.clear()