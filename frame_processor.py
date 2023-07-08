import glob
import itertools
import os
import time
import numpy as np

from typing import Iterable, Union, Optional, TypeVar, Callable, Generator

T = TypeVar('T')


def pair_generator(it: Iterable[T]) -> Generator[tuple[T, T], None, None]:
    """
    イテレータ`it`のある要素`cur_item`とその前の要素`prev_item`について`(cur_item, prev_item)`のタプルを生成するジェネレータ

    :param it: イテレータ
    :return:
    """
    prev_item = None
    for cur_item in it:
        if prev_item is not None:
            yield prev_item, cur_item
        prev_item = cur_item


UnaryMapperType = Callable[[T], T]
BinaryMapperType = Callable[[T, T], T]


def unary_mapper(fn: UnaryMapperType):
    def mapper(it: Iterable[T]) -> Generator[T, None, None]:
        for item in it:
            yield fn(item)

    return mapper


def binary_mapper(fn: BinaryMapperType):
    def mapper(it: Iterable[tuple[T, T]]) -> Generator[T, None, None]:
        for prev_item, cur_item in it:
            yield fn(prev_item, cur_item)

    return mapper


FrameItemType = tuple[float, np.ndarray]


def unary_mapping_stage(fn):
    def wrapper(item):
        frame_time, frame_array = item

        new_time = frame_time
        new_frame = fn(frame_array)

        return new_time, new_frame

    return wrapper


def binary_mapping_stage(fn):
    def wrapper(item_left, item_right):
        time_left, frame_left = item_left
        time_right, frame_right = item_right

        new_time = (time_right + time_left) / 2  # 時刻は平均をとる
        new_frame = fn(frame_left, frame_right)

        return new_time, new_frame

    return wrapper


@unary_mapping_stage
def _frame_to_double(frame_array):
    return frame_array.astype(float) / 256.0


def to_float(it: Iterable[FrameItemType]):
    return unary_mapper(_frame_to_double)(it)


@unary_mapping_stage
def _frame_to_uint(frame_array):
    return (frame_array * 256.0).astype(np.uint8)


def to_uint(it: Iterable[FrameItemType]):
    return unary_mapper(_frame_to_uint)(it)


class FramePipeline:
    class StageDirective:
        def __init__(self, name, **kwargs):
            self.name = name
            self.params = kwargs

    @classmethod
    def produce(cls, product_name):
        return cls.StageDirective('produce', product_name=product_name)

    def __init__(self, stages: list[Callable]):
        self.__stages = stages

    def __call__(self, it: Iterable[FrameItemType]) -> Iterable[dict[str, FrameItemType]]:
        it_dct = {}

        for stage in self.__stages:
            if isinstance(stage, self.StageDirective):
                if stage.name == 'produce':
                    it, produced_it = itertools.tee(it)
                    it_dct[stage.params['product_name']] = produced_it
                else:
                    raise TypeError('invalid directive', stage)
            else:
                it = stage(it)

        it_dct_key_lst = list(it_dct.keys())
        for items in zip(*(it_dct[k] for k in it_dct_key_lst)):
            product = {k: items[i] for i, k in enumerate(it_dct_key_lst)}
            yield product
