import functools
import json
from typing import NamedTuple

import numpy as np
import pandas as pd


def load(path):
    df = pd.read_csv(path)

    def time_mapper(s):
        minute, second = map(int, s.split(':'))
        return minute * 60 + second

    df = df.applymap(time_mapper)
    df = df.astype(float)

    return df


def load_rally_mask(path, timestamps):
    train_input_df = load(path)

    s, e = train_input_df.start.to_numpy(), train_input_df.end.to_numpy()
    r = np.logical_and(s <= timestamps[:, None], timestamps[:, None] <= e).sum(axis=1)
    r = r > 0

    rally_mask = r.astype(np.uint8)

    return train_input_df, rally_mask


# @functools.cache
# def _load_rect_float(video_name):
#     with open('label_data/rect.json', 'r') as f:
#         json_root = json.load(f)
#     rect_lst = json_root.get(video_name)
#     return rect_lst

class _Point2F(NamedTuple):
    x: float
    y: float


class Point2F(_Point2F):
    def __new__(cls, x: float, y: float):
        assert isinstance(x, float), x
        assert isinstance(y, float), y
        assert 0 <= x < 1, x
        assert 0 <= y < 1, y
        # noinspection PyArgumentList
        return super().__new__(cls, x=x, y=y)


class _Point2I(NamedTuple):
    x: int
    y: int


class Point2I(_Point2I):
    def __new__(cls, x: int, y: int):
        assert isinstance(x, int), x
        assert isinstance(y, int), y
        assert 0 <= x, x
        assert 0 <= y, y
        # noinspection PyArgumentList
        return super().__new__(cls, x=x, y=y)


class RectNormalized(NamedTuple):
    p_min: Point2F
    p_max: Point2F

    @classmethod
    def from_corner_xy(cls, x1, x2, y1, y2):
        values = x1, x2, y1, y2
        assert all(0 <= v < 1 for v in values), values
        assert x1 < x2 and y1 < y2, values

        p_min = Point2F(x1, y1)
        p_max = Point2F(x2, y2)

        return cls(p_min=p_min, p_max=p_max)

    def to_actual_scaled(self, width, height) -> 'RectActualScaled':
        return RectActualScaled(
            rect_normalized=self,
            scale=Point2I(width, height)
        )


class RectActualScaled(NamedTuple):
    rect_normalized: RectNormalized
    scale: Point2I

    @property
    def p_min(self) -> Point2I:
        x = self.rect_normalized.p_min.x * self.scale.x
        y = self.rect_normalized.p_min.y * self.scale.y
        return Point2I(int(x), int(y))

    @property
    def p_max(self) -> Point2I:
        x = self.rect_normalized.p_max.x * self.scale.x
        y = self.rect_normalized.p_max.y * self.scale.y
        return Point2I(int(x), int(y))

    @property
    def size(self) -> Point2I:
        dx = self.p_max.x - self.p_min.x
        dy = self.p_max.y - self.p_min.y
        return Point2I(x=dx, y=dy)

    @property
    def index2d(self) -> tuple[slice, slice]:
        """
        `(y-axis along height, x-axis along width)`の2軸のインデックスを生成する
        :return: tuple[slice, slice]
        """
        return slice(self.p_min.y, self.p_max.y), slice(self.p_min.x, self.p_max.x)

    @property
    def index3d(self) -> tuple[slice, slice, slice]:
        """
        `(y-axis along height, x-axis along width, rgb-channel)`の3軸のインデックスを生成する
        :return: tuple[slice, slice, slice]
        """
        return self.index2d[0], self.index2d[1], slice(None, None)

    def normalize_points_inside_based_on_origin(self, points: np.ndarray) \
            -> np.ndarray:
        # check the shape of input array
        assert points.ndim == 2 and points.shape[1] == 2, points

        # extract the values of x-axis and y-axis
        points_x = points[:, 0]
        points_y = points[:, 1]

        # check if the values of x-axis and y-axis are inside the rect
        assert np.all((self.p_min.x <= points_x) & (points_x < self.p_max.x)), \
            (self.p_min.x, list(points_x), list(points_y), self.p_max.x)
        assert np.all((self.p_min.y <= points_y) & (points_y < self.p_max.y)), \
            (self.p_min.y, list(points_x), list(points_y), self.p_max.y)

        # normalize values
        normalized_x = (points_x - self.p_min.x) / self.scale.x
        normalized_y = (points_y - self.p_min.y) / self.scale.y

        # generate normalized array
        result = np.concatenate([normalized_x[:, None], normalized_y[:, None]], axis=1)
        assert points.shape == result.shape, (points.shape, result.shape)

        return result

    def normalize_points_inside_based_on_corner(self, points_based_on_top_left_corner: np.ndarray) \
            -> np.ndarray:
        return self.normalize_points_inside_based_on_origin(
            points_based_on_top_left_corner + self.scale
        )


class RectFactory:
    def __init__(self, path):
        self.__path = path

        with open(self.__path, 'r') as f:
            self.__json_root = json.load(f)

    @functools.cache
    def normalized(self, video_name) -> RectNormalized:
        y1, y2, x1, x2 = self.__json_root[video_name]

        width = x2 - x1
        half_width = width / 2
        x1, x2 = x1 - half_width, x2 + half_width

        return RectNormalized.from_corner_xy(x1=x1, x2=x2, y1=y1, y2=y2)

    @functools.cache
    def actual_scaled(self, video_name, width, height) -> RectActualScaled:
        return self.normalized(video_name).to_actual_scaled(width=width, height=height)


frame_rects = RectFactory('./label_data/rect.json')


# def load_rect(video_name, height, width) \
#         -> tuple[slice, slice]:  # height-slice, width-slice, not normalized
#     y1, y2, x1, x2 = _load_rect_float(video_name)
#
#     rect_width = rect_lst[3] - rect_lst[2]
#     rect_x_expand = rect_width / 2
#
#     rect_lst = (
#         slice(
#             int(rect_lst[0] * height),  # y1
#             int(rect_lst[1] * height)  # y2
#         ),
#         slice(
#             int((rect_lst[2] - rect_x_expand) * width),  # x1
#             int((rect_lst[3] + rect_x_expand) * width)  # x2
#         )
#     )
#     return rect_lst


def update_rect(video_name, rect):
    # rect_lst = rect[0].start, rect[0].stop, rect[1].start, rect[1].stop
    rect_lst = [rect[0][0], rect[0][1], rect[1][0], rect[1][1]]
    with open('label_data/rect.json', 'r') as f:
        json_root = json.load(f)
    json_root[video_name] = rect_lst
    with open('label_data/rect.json', 'w') as f:
        json.dump(json_root, f, indent=2, sort_keys=True)
