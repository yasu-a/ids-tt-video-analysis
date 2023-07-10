import collections
from typing import Iterable, TypeVar, Callable, Generator

import numpy as np

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

EntryType = dict


def mapper(window_size=1, center=None):
    center = center or window_size // 2

    def decorator(fn):
        def wrapper(entry_it: Iterable[EntryType]):
            buffer = []

            def buffer_full_filled():
                return len(buffer) >= window_size

            def append_buffer(item):
                if buffer_full_filled():
                    buffer.pop(0)
                buffer.append(item)

            def split_buffer():
                nonlocal center
                left, middle, right = buffer[:center], buffer[center], buffer[center:]
                return left, middle, right

            for entry in entry_it:
                append_buffer(entry)
                if buffer_full_filled():
                    left, middle, right = split_buffer()
                    processed_middle = fn(left, middle, right)
                    yield processed_middle

        return wrapper

    return decorator


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
        def it_ctx_wrapper(it):
            for item in it:
                yield dict(item=item, ctx={})

        def stage_wrapper(fn):
            def wrapper(it):
                # TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                itd = fn(d['item'] for d in it)
                for dct, processed_item in zip(it, itd):
                    dct['item'] = processed_item

            return wrapper

        def context_recorder(name):
            def recorder(it):
                for dct in it:
                    dct['ctx'][name] = dct['item']
                    yield dct

            return recorder

        it = it_ctx_wrapper(it)
        for stage in self.__stages:
            if isinstance(stage, self.StageDirective):
                if stage.name == 'produce':
                    recorder = context_recorder(stage.params['product_name'])
                    it = recorder(it)
                else:
                    raise TypeError('invalid directive', stage)
            else:
                stage = stage_wrapper(stage)
                it = stage(it)

        for dct in it:
            item, ctx = dct['item'], dct['ctx']
            product = ctx
            yield product
