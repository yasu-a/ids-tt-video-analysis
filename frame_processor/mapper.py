import enum
from collections.abc import Iterable

from .common import *


class Fields(str, enum.Enum):
    IMAGE = 'image'  # FrameEntry.image.__name__
    POSITION = 'position'  # FrameEntry.position.__name__


def process(field_key: Fields):
    def take_field(obj):
        return getattr(obj, field_key)

    def processor(mapper_fn):
        def wrapper(left, middle, right):
            assert isinstance(middle, FrameEntry)
            new_value = mapper_fn(
                list(map(take_field, left)),
                take_field(middle),
                list(map(take_field, right))
            )
            new_entry = middle._replace(**{field_key: new_value})
            return new_entry

        return wrapper

    return processor


def unary(mapper_fn):
    def wrapper(left, middle, right):
        return mapper_fn(middle)

    return wrapper


def past(mapper_fn):
    def wrapper(left, middle, right):
        return mapper_fn(left[-1], middle)

    return wrapper


def future(mapper_fn):
    def wrapper(left, middle, right):
        return mapper_fn(middle, right[0])

    return wrapper


@process(Fields.IMAGE)
@unary
def to_double(image):
    return image.astype(float) / 256.0


@process(Fields.IMAGE)
@unary
def to_uint(image):
    return (image * 256.0).astype(np.uint8)
