import collections
import functools
import os
from typing import NamedTuple, Iterable

import numpy as np

from . import util
from .sample import VideoFrameLabelSample


class FrameAggregationEntry(NamedTuple):
    frame_index_center: int
    n_total_sources: int
    n_sources: int

    @property
    def reliability(self) -> float:
        return self.n_sources / self.n_total_sources


class FrameAggregationResult(NamedTuple):
    frame_index_center: np.ndarray
    n_total_sources: np.ndarray
    n_sources: np.ndarray
    reliability: np.ndarray

    __dtypes = dict(
        frame_index_center=int,
        n_total_sources=int,
        n_sources=int,
        reliability=float
    )

    @classmethod
    def from_entries(cls, entries: Iterable[FrameAggregationEntry]):
        names = cls._fields
        var_dct = {}
        for name in names:
            value_lst = [getattr(e, name) for e in entries]
            a = np.array(value_lst).astype(cls.__dtypes[name])
            a.setflags(write=False)
            var_dct[name] = a
        return cls(**var_dct)


class VideoFrameLabelSampleSetMixin:
    def __len__(self):
        ...

    def __getitem__(self, i: int) -> VideoFrameLabelSample:
        ...

    @functools.cached_property
    def frame_label_name_set(self) -> tuple[str]:
        s = sorted({
            label_name
            for sample_labels in self
            for label_name in sample_labels.frame_label_name_set
        })
        return tuple(s)

    # TODO: implement me

    def __aggregate(self, label_name: str = None):
        if label_name is not None and label_name not in self.frame_label_name_set:
            raise ValueError(f'Invalid label name {label_name!r}')

        # compute clusters
        arrays = [
            sample_labels.frame_index_array(label_name)
            for sample_labels in self
        ]
        cluster_labels, info = util.cluster(arrays)

        # aggregate labels and associated frame indexes
        cluster_label_and_fi_pair = (
            (label, fi)
            for a, a_labels in zip(arrays, cluster_labels)
            for fi, label in zip(a, a_labels)
        )
        cluster_label_to_fi_set = collections.defaultdict(list)
        for label, fi in cluster_label_and_fi_pair:
            cluster_label_to_fi_set[label].append(fi)

        # remove noise entry with label of -1
        cluster_label_to_fi_set.pop(-1)

        # aggregate to entries
        fi_center_to_entry = {}
        for _, fi_set in cluster_label_to_fi_set.items():
            fi_center = int(np.mean(fi_set))
            entry = FrameAggregationEntry(
                frame_index_center=fi_center,
                n_total_sources=len(arrays),
                n_sources=len(fi_set),
            )
            fi_center_to_entry[fi_center] = entry

        # create results and return them
        fi_center_array = np.array(sorted(fi_center_to_entry.keys()))
        entries = [fi_center_to_entry[fi_center] for fi_center in fi_center_array]
        used_margin = info['margin']
        return (
            fi_center_array,
            FrameAggregationResult.from_entries(entries),
            used_margin
        )

    @functools.cache
    def __aggregate_cached(self, label_name: str = None):
        fi_center_array, result, used_margin = self.__aggregate(
            label_name=label_name
        )

        # set array read-only
        fi_center_array.setflags(write=False)

        return fi_center_array, result, used_margin

    def agg_frame_indexes(self, **kwargs) -> np.ndarray:
        return self.__aggregate_cached(**kwargs)[0]  # fi_center_array

    def agg_results(self, **kwargs) -> FrameAggregationResult:
        return self.__aggregate_cached(**kwargs)[1]  # entries

    def agg_difference_margin(self, **kwargs) -> float:
        return self.__aggregate_cached(**kwargs)[2]  # used_margin


class VideoFrameLabelSampleSet(VideoFrameLabelSampleSetMixin):
    def __init__(self, path: str):
        self.__path = path

    @functools.cache
    def list_json_names(self):
        return tuple(sorted(os.listdir(self.__path)))

    def __len__(self):
        return len(self.list_json_names())

    def __getitem__(self, i: int) -> VideoFrameLabelSample:
        return VideoFrameLabelSample(os.path.join(self.__path, self.list_json_names()[i]))

    def __repr__(self):
        return f'VFLSampleSet(path={self.__path!r})'
