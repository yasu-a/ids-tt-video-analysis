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
    label_indexes: np.array  # int, [N_SOURCES]

    @property
    def reliability(self) -> float:
        return self.n_sources / self.n_total_sources


class FrameAggregationResult(NamedTuple):
    frame_index_center: np.ndarray  # int, [N_LABELS]
    n_total_sources: np.ndarray  # int, [N_LABELS]
    n_sources: np.ndarray  # int, [N_LABELS]
    reliability: np.ndarray  # float, [N_LABELS]
    label_indexes: np.ndarray  # int, [N_LABELS, N_SOURCES]
    margin: int  # float, []

    __dtypes = dict(
        frame_index_center=int,
        n_total_sources=int,
        n_sources=int,
        reliability=float,
        label_indexes=int,
    )

    @classmethod
    def from_entries(cls, entries: Iterable[FrameAggregationEntry], **extract_arguments):
        var_dct = {} | extract_arguments
        for name in cls.__dtypes.keys():
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
    def frame_label_name_set(self) -> tuple[str, ...]:
        s = sorted({
            label_name
            for sample_labels in self
            for label_name in sample_labels.frame_label_name_set
        })
        return tuple(s)

    # TODO: implement me

    @functools.cache
    def aggregate(self, label_name: str = None, nan_value=-1) -> FrameAggregationResult:
        if label_name is not None and label_name not in self.frame_label_name_set:
            raise ValueError(f'Invalid label name {label_name!r}')

        # compute clusters
        arrays = [
            sample_labels.frame_index_array(label_name)
            for sample_labels in self
        ]

        # cl_result: int, [array_items, array_indexes, group_indexes, labels]
        cl_result, info = util.cluster(arrays)

        # aggregate labels and associated array items
        labels = np.sort(np.unique(cl_result[3]))
        labels = labels[labels >= 0]  # remove noise items

        label_agg = [
            [
                np.squeeze(cl_result[:, (cl_result[3, :] == lbl) & (cl_result[2, :] == gi)])
                for gi in range(len(self))
            ]
            for lbl in labels
        ]
        label_agg = [
            [
                a if a.size else [nan_value] * cl_result.shape[0]
                for a in a_label_agg
            ]
            for a_label_agg in label_agg
        ]
        # label_agg: int, [N_CLUSTERS, N_TOTAL_SOURCES, 4]
        label_agg = np.stack(label_agg)

        # aggregate to entries
        entries = [
            FrameAggregationEntry(
                frame_index_center=int(
                    np.mean(label_agg[lbl, label_agg[lbl, :, 0] != nan_value, 0]).round(0)),
                n_total_sources=len(self),
                n_sources=np.count_nonzero(label_agg[lbl, label_agg[lbl, :, 0] != nan_value, 0]),
                label_indexes=label_agg[lbl, :, 1],
            )
            for lbl in labels
        ]
        entries.sort(key=lambda e: e.frame_index_center)

        # create results and return them
        return FrameAggregationResult.from_entries(
            entries,
            margin=info['margin']
        )


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
