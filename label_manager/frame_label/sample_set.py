import functools
import os
from dataclasses import dataclass
from typing import NamedTuple, Iterable, Callable

import numpy as np

from . import util
from .sample import VideoFrameLabelSample


class FrameAggregationEntry(NamedTuple):
    frame_index_center: int
    n_total_sources: int
    n_sources: int
    cluster_indexes: np.array  # int, [N_SOURCES]
    label_index: int

    @property
    def reliability(self) -> float:
        return self.n_sources / self.n_total_sources


class FrameAggregationResult(NamedTuple):
    frame_index_center: np.ndarray  # int, [N_LABELS]
    n_total_sources: np.ndarray  # int, [N_LABELS]
    n_sources: np.ndarray  # int, [N_LABELS]
    reliability: np.ndarray  # float, [N_LABELS]
    cluster_indexes: np.ndarray  # int, [N_LABELS, N_SOURCES]
    label_index: np.ndarray  # int, [N_LABELS]
    margin: int  # int, []

    @property
    def n_frames(self):
        return len(self.frame_index_center)

    def __len__(self):
        return self.n_frames

    def __getitem__(self, i):
        return FrameAggregationEntry(**{
            name: None if getattr(self, name) is None else getattr(self, name)[i]
            for name in FrameAggregationEntry._fields
        })

    def get(self, i, default=None):
        if i < 0 or len(self) <= i:
            return default
        return self[i]

    def filter(self, mask):
        mask = np.array(mask)
        if mask.dtype != np.bool_ or mask.ndim != 1 or len(mask) != self.n_frames:
            raise ValueError('invalid mask')

        dct = {
            k: getattr(self, k)[mask, ...]
            for k in self.__dtypes.keys()
            if getattr(self, k) is not None
        }

        for k in self._fields:
            if k not in dct:
                dct[k] = getattr(self, k)

        return type(self)(**dct)

    __dtypes = dict(
        frame_index_center=int,
        n_total_sources=int,
        n_sources=int,
        reliability=float,
        cluster_indexes=int,
        label_index=int
    )

    __to_numpy_exclude = {'cluster_indexes'}

    @classmethod
    def from_entries(cls, entries: Iterable[FrameAggregationEntry], **extract_arguments):
        var_dct = {} | extract_arguments
        for name in cls.__dtypes.keys():
            value_lst = [getattr(e, name) for e in entries]
            a = np.array(value_lst).astype(cls.__dtypes[name])
            a.setflags(write=False)
            var_dct[name] = a
        return cls(**var_dct)

    def to_numpy(self) -> np.ndarray:
        return np.stack([
            (getattr(self, k) * (1 if tp == int else 100)).astype(int)
            for k, tp in self.__dtypes.items()
            if k not in self.__to_numpy_exclude
        ])

    @classmethod
    def from_numpy(cls, mat):
        dct = {
            k: mat[i, :] if tp == int else (mat[i, :] / 100).astype(float)
            for i, (k, tp) in enumerate(
                (k, tp)
                for k, tp in cls.__dtypes.items()
                if k not in cls.__to_numpy_exclude
            )
        }

        for k in cls._fields:
            dct[k] = dct.get(k)

        return cls(**dct)

    @dataclass()
    class GroupingResult:
        prev_frame: FrameAggregationEntry
        frames: tuple[FrameAggregationEntry, ...]
        next_frame: FrameAggregationEntry

    def extract_ordered_label_groups(
            self,
            label_order: list[int],
            predicate: Callable[[FrameAggregationEntry], bool] = None
    ):
        predicate = predicate or (lambda *_: True)

        i = 0
        j = None
        while True:
            current = self.get(i)
            if current is None:
                break

            if j is None:
                j = i

            if label_order[i - j] == current.label_index and predicate(current):
                if len(label_order) - 1 == i - j:
                    yield self.GroupingResult(
                        prev_frame=self.get(j - 1),
                        frames=tuple(self[k] for k in range(j, i + 1)),
                        next_frame=self.get(i + 1)
                    )
                    j = None
            else:
                j = None

            i += 1


class VideoFrameLabelSampleSetMixin:
    def __len__(self):
        ...

    def __getitem__(self, i: int) -> VideoFrameLabelSample:
        ...

    @functools.cached_property
    def frame_label_name_list(self) -> tuple[str, ...]:
        s = sorted({
            label_name
            for sample_labels in self
            for label_name in sample_labels.frame_label_name_set
        })
        return tuple(s)

    # TODO: implement me

    @functools.cache
    def aggregate(self, label_name: str, nan_value=-1) -> FrameAggregationResult:
        if label_name not in self.frame_label_name_list:
            raise ValueError(f'Invalid label name {label_name!r}')

        label_index = self.frame_label_name_list.index(label_name)

        # compute clusters
        arrays = [
            sample_labels.frame_index_array(label_name)
            for sample_labels in self
        ]

        # cl_result: int, [array_items, array_indexes, group_indexes, cluster_indexes]
        cl_result, info = util.cluster(arrays)

        # aggregate labels and associated array items
        cluster_indexes = np.sort(np.unique(cl_result[3]))
        cluster_indexes = cluster_indexes[cluster_indexes >= 0]  # remove noise items

        label_agg = [
            [
                np.squeeze(cl_result[:, (cl_result[3, :] == lbl) & (cl_result[2, :] == gi)])
                for gi in range(len(self))
            ]
            for lbl in cluster_indexes
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
                    np.mean(label_agg[ci, label_agg[ci, :, 0] != nan_value, 0]).round(0)),
                n_total_sources=len(self),
                n_sources=np.count_nonzero(label_agg[ci, label_agg[ci, :, 0] != nan_value, 0]),
                cluster_indexes=label_agg[ci, :, 1],
                label_index=label_index
            )
            for ci in cluster_indexes
        ]
        entries.sort(key=lambda e: e.frame_index_center)

        # create results and return them
        return FrameAggregationResult.from_entries(
            entries,
            margin=info['margin']
        )

    @functools.cache
    def aggregate_full(self, nan_value=-1) -> FrameAggregationResult:
        mat = np.concatenate([
            self.aggregate(label_name=label_name, nan_value=nan_value).to_numpy()
            for label_name in self.frame_label_name_list
        ], axis=1)
        mat = mat[:, mat[0, :].argsort()]

        return FrameAggregationResult.from_numpy(mat)


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
