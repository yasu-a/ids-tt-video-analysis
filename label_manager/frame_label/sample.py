import functools
import json

import numpy as np

from .json_struct import LabelJson


class VideoFrameLabelSampleMixin:
    content: LabelJson = ...

    # TODO: implement me

    @functools.cached_property
    def full_frame_index_array(self):
        a = np.array(sorted(self.content.labels.markers.keys()))
        a.setflags(write=False)
        return a

    @functools.cache
    def frame_index_array(self, label_name=None):
        if label_name is None:
            return self.full_frame_index_array

        a = np.array([
            fi
            for fi, item_label_name in zip(
                self.full_frame_index_array,
                self.frame_label_name_list
            )
            if item_label_name == label_name
        ])
        a.setflags(write=False)
        return a

    @functools.cached_property
    def frame_label_name_list(self) -> tuple[str, ...]:
        return tuple(self.content.labels.markers.values())

    @functools.cached_property
    def frame_label_name_set(self) -> tuple[str, ...]:
        return tuple(sorted(frozenset(self.frame_label_name_list)))


class VideoFrameLabelSample(VideoFrameLabelSampleMixin):
    def __init__(self, json_path):
        self.__json_path = json_path

        with open(json_path, 'r') as f:
            self.__content = LabelJson.from_json(json.load(f))

    @property
    def content(self) -> LabelJson:
        return self.__content

    def __repr__(self):
        return f'VFLSample(json_path={self.__json_path!r})'
