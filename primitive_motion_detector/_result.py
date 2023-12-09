import functools

import numpy as np

from ._util import OneWriteManyReadDescriptor


class PMDetectorResult:
    original_images_clipped: np.ndarray
    diff_images_clipped: np.ndarray
    keypoints: list[np.ndarray]
    keyframes: list[np.ndarray]
    match_index_pair: np.ndarray
    distance_matrix: np.ndarray
    local_centroid: np.ndarray  # centroids as rect coordinate
    global_centroid: np.ndarray  # centroids as global coordinate
    local_centroid_normalized: np.ndarray  # array of float32
    centroid_delta: np.ndarray
    velocity_normalized: np.ndarray
    valid: bool

    @classmethod
    def __field_names(cls):
        return tuple(cls.__annotations__.keys())

    def __init__(self):
        self.__descriptors = []
        for field_name in self.__field_names():
            field_accessor = OneWriteManyReadDescriptor(default=None)
            self.__descriptors.append(field_accessor)
            setattr(self, field_name, field_accessor)

    @classmethod
    def frozen(cls, **kwargs):
        assert set(cls.__field_names()) == set(kwargs.keys()), set(kwargs.keys())
        ins = cls()
        for field_name in cls.__field_names():
            setattr(ins, field_name, kwargs[field_name])
        return ins

    def _freeze_fields(self):
        for accessor in self.__descriptors:
            accessor.freeze()

    @functools.cached_property
    def contains_keypoints(self):
        assert self.keypoints is not None
        return all(keypoints.size > 0 for keypoints in self.keypoints)

    @functools.cached_property
    def contains_matches(self):
        assert self.keypoints is not None
        return self.match_index_pair.size > 0

    @functools.cached_property
    def n_matches(self):
        assert self.match_index_pair is not None
        return len(self.match_index_pair.T)