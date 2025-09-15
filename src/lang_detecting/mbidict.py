from __future__ import annotations

from typing import Hashable, Iterable, Callable

from numpy.random.mtrand import Sequence


def mbidict(left_name: str, right_name: str) -> MBidict:
    return MBidict(left_name, right_name)


class KeySet(set):
    def __init__(self, parent: DictWrapper, key: Hashable):
        self._parent: DictWrapper = parent
        self._key = key
        super().__init__()

    def add(self, elem: Hashable) -> None:
        if elem not in self:
            super().add(elem)
            self._parent.op_side[elem].add(self._key)

    def extend(self, elems: Sequence[Hashable]) -> None:
        for elem in elems:
            self.add(elem)

    def remove(self, elem: Hashable) -> None:
        if elem in self:
            super().remove(elem)
            self._parent.op_side[elem].remove(self._key)

    def __del__(self):
        for elem in list(self):
            self.remove(elem)

    def __repr__(self):
        quoted = (f"'{e}'" for e in self)
        return f"{{{', '.join(quoted)}}}"


class DictWrapper(dict):
    def __init__(self, parent: MBidict, op_name: str):
        self._parent = parent
        self._op_name = op_name
        super().__init__()

    @property
    def op_side(self) -> DictWrapper:
        return self._parent[self._op_name]

    def __getitem__(self, key: Hashable):
        return self.setdefault(key, KeySet(self, key))

    def __setitem__(self, left: Hashable, right: Sequence | Hashable):
        if not isinstance(right, Hashable):
            right = [right]
        for r in right:
            self[left].add(r)

    def __repr__(self):
        quoted = (f"'{k}': {v}" for k, v in self.items())
        return f"{{{', '.join(quoted)}}}"

class MBidict:  # TODO: Think of changing the upper class to Box
    def __init__(self, left_name: str = 'left', right_name: str = 'right'):
        super().__init__()
        self._left_name = left_name
        self._right_name = right_name
        self._left_dict: DictWrapper[Hashable, KeySet] = DictWrapper(self, self._right_name)
        self._right_dict: DictWrapper[Hashable, KeySet] = DictWrapper(self, self._left_name)

    @property
    def left(self):
        return self._left_dict

    @property
    def right(self):
        return self._right_dict

    def __getattr__(self, item):
        if item == self._left_name:
            return self.left
        elif item == self._right_name:
            return self.right
        else:
            return self.__getattribute__(item)

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __setitem__(self, left: Hashable, right: Hashable | Iterable) -> None:
        if isinstance(right, Iterable) and not isinstance(right, Hashable):
            for r in right:
                self.__setitem__(left, r)
            return
        self.left[left].add(right)

    def add(self, left: Hashable, right: Hashable) -> None:
        self.__setitem__(left, right)

    def __repr__(self):
        return f'MBidict({self._left_name}={self._left_dict}, {self._right_name}={self._right_dict})'

    @property
    def pretty_string(self) -> str:
        string = ''
        for name in (self._left_name, self._right_name):
            string += f'*** {name.capitalize()} *****\n'
            for key, vals in self[name].items():
                string += f'{key}: {vals}\n'
        return string

