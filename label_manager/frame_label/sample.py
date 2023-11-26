import functools
import json

import numpy as np

from .json_struct import LabelJson


class VideoFrameLabelSampleMixin:
    label_json: LabelJson = ...

    @functools.cache
    def fi_array_filtered_by_name(self, target_label_name=None):
        if target_label_name is None:
            return self.fi_array

        a = np.array([
            entry.fi
            for entry in self.label_json.frames
            if entry.label == target_label_name
        ])
        a.setflags(write=False)
        return a

    @functools.cached_property
    def fi_array(self):
        return self.fi_array_filtered_by_name()

    @functools.cached_property
    def label_array(self) -> tuple[str, ...]:
        return tuple(entry.label for entry in self.label_json.frames)

    @functools.cached_property
    def frame_label_name_set(self) -> tuple[str, ...]:
        return tuple(sorted(frozenset(self.label_array)))


class VideoFrameLabelSample(VideoFrameLabelSampleMixin):
    def __init__(self, json_path):
        self.__json_path = json_path

        with open(json_path, 'r') as f:
            self.__label_json = LabelJson.from_json(json.load(f))

    @property
    def json_path(self):
        return self.__json_path

    @property
    def label_json(self) -> LabelJson:
        return self.__label_json

    def __repr__(self):
        return f'VFLSample(json_path={self.__json_path!r})'
