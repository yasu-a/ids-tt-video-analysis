from typing import NamedTuple

import numpy as np


def mark_entry_iterable(fn):
    setattr(fn, '_fp_entry_iterable', None)


def cascade_entry_iterable(fn_src, fn_dst):
    if check_entry_iterable(fn_src):
        mark_entry_iterable(fn_dst)


def check_entry_iterable(fn):
    return hasattr(fn, '_fp_entry_iterable')


class FrameEntry(NamedTuple):
    image: np.ndarray
    index: int
    position: float
    stack: dict

    @classmethod
    def create_instance(cls, *, image, index, position):
        return cls(
            image=image,
            index=index,
            position=position,
            stack={}
        )

    def as_result(self):
        return Product(self)

    def __repr__(self):
        s = [
            f'image={self.image.shape}',
            f'index={self.index}',
            f'position={self.position:.3f}',
            f'stack={list(self.stack.keys())}'
        ]
        return 'FrameEntry(' + ', '.join(s) + ')'


class Product:
    def __init__(self, entry: FrameEntry):
        self.__entry = entry

    @property
    def image(self):
        return self.__entry.image

    @property
    def position(self):
        return self.__entry.position

    @property
    def index(self):
        return self.__entry.index

    def __getitem__(self, stack_key):
        return self.__entry.stack[stack_key]
