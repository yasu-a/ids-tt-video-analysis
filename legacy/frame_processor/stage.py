import collections
from typing import Iterable

from .common import FrameEntry, mark_entry_iterable


def neighbour(window_size=1, *, center=None):
    center = window_size // 2 if center is None else center

    def decorator(mapper_fn):
        def wrapper(entry_it: Iterable[FrameEntry]):
            buffer = collections.deque([None] * window_size, maxlen=window_size)
            buffer_unfilled = window_size

            for entry in entry_it:
                # append buffer
                buffer_unfilled = buffer_unfilled and buffer_unfilled - 1
                buffer.append(entry)

                if not buffer_unfilled:
                    # split buffer
                    lst = tuple(buffer)
                    left, middle, right = lst[:center], lst[center], lst[center + 1:]

                    processed_middle = mapper_fn(left, middle, right)
                    yield processed_middle

        mark_entry_iterable(wrapper)

        return wrapper

    return decorator


each = neighbour(1, center=0)
contiguous = neighbour(2, center=1)


def store(name):
    def mapper_fn(entry_it: Iterable[FrameEntry]):
        for item in entry_it:
            item.stack[name] = item.image
            yield item

    mark_entry_iterable(mapper_fn)

    return mapper_fn
