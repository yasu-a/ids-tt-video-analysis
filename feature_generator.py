import contextlib
import functools
import typing
from functools import cached_property

import numpy as np
from tqdm import tqdm

import dataset
import train_input


def process_rect(rect):
    w = rect[1].stop - rect[1].start
    aw = int(w * 1.0)
    return slice(rect[0].start, rect[0].stop), slice(rect[1].start - aw, rect[1].stop + aw)


class FeatureGenerator:
    def __init__(self, video_name):
        self.__video_name = video_name
        self.__cache = dataset.PickleCache(video_name)

    @property
    def video_name(self):
        return self.__video_name

    @contextlib.contextmanager
    def __frame_store(self) -> dataset.VideoFrameStorage:
        with dataset.VideoFrameStorage(
                dataset.get_video_frame_dump_dir_path(video_name=self.__video_name),
                mode='r',
        ) as vf_store:
            yield vf_store

    @contextlib.contextmanager
    def __motion_store(self) -> dataset.MotionStorage:
        with dataset.MotionStorage(
                dataset.get_motion_dump_dir_path(video_name=self.__video_name),
                mode='r',
        ) as m_store:
            yield m_store

    @cached_property
    def timestamp(self):
        with self.__motion_store() as s:
            return s.get_all_of('timestamp')

    def __iter_diff_frames(self):
        rect = process_rect(train_input.load_rect(self.__video_name))

        with self.__frame_store() as s:
            timestamp = s.get_all_of('timestamp')[:-1]
            assert timestamp.shape == self.timestamp.shape
            assert np.all(timestamp == self.timestamp)
            for i in range(s.count() - 1):
                yield s.get(i)['motion'][rect]

    def __iter_motion_vector(self):
        with self.__motion_store() as s:
            start = s.get_all_of('start')
            end = s.get_all_of('end')
            for s, e in zip(start, end):
                yield s, e

    @classmethod
    @functools.cache
    def __create_split_slices_2d(cls, shape, n_axis_split):
        def slice_iterator():
            w, h, *_ = shape
            nw = w // n_axis_split
            nh = h // n_axis_split
            for i in range(n_axis_split):
                si = slice(nw * i, None if i == n_axis_split - 1 else nw * (i + 1))
                for j in range(n_axis_split):
                    sj = slice(nh * j, None if j == n_axis_split - 1 else nh * (j + 1))
                    yield si, sj

        return tuple(slice_iterator())

    @classmethod
    def __array_split_mean_2d(cls, a, n_axis_split):
        return np.array([a[s].mean() for s in cls.__create_split_slices_2d(a.shape, n_axis_split)])

    N_DIFF_FRAME_FEATURE = 3

    def iter_diff_frame_feature(self):
        n_axis_split = 5

        for diff in self.__iter_diff_frames():
            split_mean = self.__array_split_mean_2d(diff, n_axis_split)
            split_mean_sorted = np.sort(split_mean)
            split_mean_arg_sorted = np.argsort(split_mean)
            diff_frame_feature = np.concatenate([
                split_mean_sorted[-self.N_DIFF_FRAME_FEATURE:],
                (split_mean_arg_sorted[-self.N_DIFF_FRAME_FEATURE:] % self.N_DIFF_FRAME_SPLIT),
                (split_mean_arg_sorted[-self.N_DIFF_FRAME_FEATURE:] // self.N_DIFF_FRAME_SPLIT)
            ])
            yield diff_frame_feature

    def iter_diff_frame_feature_full(self):
        n_axis_split = 5

        for diff in self.__iter_diff_frames():
            split_mean = self.__array_split_mean_2d(diff, n_axis_split)
            diff_frame_feature = split_mean.flatten()
            yield diff_frame_feature

    N_MOTION_SPLIT = 5
    N_MOTION_FEATURE_BLOCK_SIZE = 3
    assert N_MOTION_FEATURE_BLOCK_SIZE % 2 == 1 and N_MOTION_FEATURE_BLOCK_SIZE >= 3
    N_MOTION_FEATURE_BLOCKS = 11
    MOTION_FEATURE_WIDTH = N_MOTION_FEATURE_BLOCK_SIZE * N_MOTION_FEATURE_BLOCKS
    assert MOTION_FEATURE_WIDTH % 2 == 1

    @classmethod
    def motion_feature_names(cls):
        diff_names = [
            f'diff {i}-{j}'
            for i in range(cls.N_MOTION_FEATURE_BLOCKS)
            for j in range(cls.N_MOTION_FEATURE_BLOCKS)
            if i > j
        ]
        return diff_names

    def iter_motion_vector_diff_feature(self):
        rect = process_rect(train_input.load_rect(self.__video_name))

        def split_vertically(n_split, offset, height, points):
            axis = 0
            n = height // n_split
            criteria = points[:, axis] - offset
            nth_split = criteria.astype(int) // n
            return tuple(
                np.where(np.minimum(nth_split, n_split - 1) == i)[0]
                for i in range(n_split)
            )

        vals = []
        for start, end in self.__iter_motion_vector():
            start_split_arg = split_vertically(
                n_split=self.N_MOTION_SPLIT,
                offset=rect[0].start,
                height=rect[0].stop - rect[0].start,
                points=start
            )

            val_split = []
            for j, arg in enumerate(start_split_arg):
                s, e = start[arg], end[arg]
                vec = e - s
                val_split.append(np.nanmean(vec[:, 0]))
            vals.append(val_split)

        vals = np.array(vals).T
        vals[np.isnan(vals)] = 0

        vals_slide = np.lib.stride_tricks.sliding_window_view(
            np.pad(
                vals,
                pad_width=(
                    (0, 0),
                    (self.MOTION_FEATURE_WIDTH // 2, self.MOTION_FEATURE_WIDTH // 2)
                ),
                mode='reflect'
            ),
            window_shape=(vals.shape[0], self.MOTION_FEATURE_WIDTH)
        )[0]
        vals_slide = np.swapaxes(vals_slide, 0, 1)
        print(f'{vals_slide.shape=}')
        block_split = np.stack(np.split(vals_slide, self.N_MOTION_FEATURE_BLOCKS, axis=2), axis=3)
        print(f'{block_split.shape=}')
        mean = block_split.mean(axis=2)

        def block_diff_feature(blocks):
            assert blocks.ndim == 1, blocks.shape
            n = blocks.size
            lst = [blocks[i] - blocks[j] for i in range(n) for j in range(n) if i > j]

            return np.array(lst)

        diff_features = np.apply_along_axis(block_diff_feature, 2, mean)
        features = np.concatenate([diff_features], axis=2)
        features = np.concatenate(features, axis=1)
        print(f'{features.shape=}')

        for feature in features:
            yield feature

    def iter_motion_vector_feature(self):
        rect = process_rect(train_input.load_rect(self.__video_name))

        def split_vertically(n_split, offset, height, points):
            axis = 0
            n = height // n_split
            criteria = points[:, axis] - offset
            nth_split = criteria.astype(int) // n
            return tuple(
                np.where(np.minimum(nth_split, n_split - 1) == i)[0]
                for i in range(n_split)
            )

        vals = []
        for start, end in self.__iter_motion_vector():
            start_split_arg = split_vertically(
                n_split=self.N_MOTION_SPLIT,
                offset=rect[0].start,
                height=rect[0].stop - rect[0].start,
                points=start
            )

            val_split = []
            for j, arg in enumerate(start_split_arg):
                s, e = start[arg], end[arg]
                vec = e - s
                val_split.append(np.nanmean(vec[:, 0]))
            vals.append(val_split)

        vals = np.array(vals).T
        vals[np.isnan(vals)] = 0

        vals_slide = np.lib.stride_tricks.sliding_window_view(
            np.pad(
                vals,
                pad_width=(
                    (0, 0),
                    (self.MOTION_FEATURE_WIDTH // 2, self.MOTION_FEATURE_WIDTH // 2)
                ),
                mode='reflect'
            ),
            window_shape=(vals.shape[0], self.MOTION_FEATURE_WIDTH)
        )[0]
        vals_slide = np.swapaxes(vals_slide, 0, 1)
        print(f'{vals_slide.shape=}')
        block_split = np.stack(np.split(vals_slide, self.N_MOTION_FEATURE_BLOCKS, axis=2), axis=3)
        print(f'{block_split.shape=}')
        mean = block_split.mean(axis=2)

        def block_diff_feature(blocks):
            assert blocks.ndim == 1, blocks.shape
            n = blocks.size
            lst = [blocks[i] - blocks[j] for i in range(n) for j in range(n) if i > j]

            return np.array(lst)

        diff_features = np.apply_along_axis(block_diff_feature, 2, mean)
        features = np.concatenate([mean, diff_features], axis=2)
        features = np.concatenate(features, axis=1)
        print(f'{features.shape=}')

        for feature in features:
            yield feature

    class Feature(typing.NamedTuple):
        timestamp: np.ndarray
        feature: np.ndarray
        label: typing.Optional[np.ndarray]

    def __create_feature(self, name, feature_iters, with_label):
        def create_feature_vector():
            def iter_feature_vector():
                for fs in tqdm(zip(*feature_iters)):
                    feature = np.concatenate(fs)
                    yield feature

            return np.stack(list(iter_feature_vector()))

        def load_or_create_feature():
            # convert feature cache file name
            feature_cache_name = f'feature_{name}'

            # retrieve data from cache
            cached_feature_dct = self.__cache.load(feature_cache_name)

            if cached_feature_dct is None:
                # if cache data not found then generate data and add it to cache
                result = self.Feature(
                    timestamp=self.timestamp,
                    feature=create_feature_vector(),
                    label=None
                )
                cached_feature_dct = result._asdict()
                self.__cache.dump(feature_cache_name, cached_feature_dct)
            else:
                # if cache data found then convert it to Feature instance
                result = self.Feature(**cached_feature_dct)

            # add label if with_label is True
            if with_label:
                label = train_input.load_rally_mask(
                    f'./train/iDSTTVideoAnalysis_{self.__video_name}.csv',
                    timestamps=self.timestamp
                )[1].astype(bool)
                result = result._replace(label=label)

            return result

        return load_or_create_feature()

    def create(self, with_label=False):
        return self.__create_feature(
            name='diff_frame+motion_vector',
            feature_iters=[
                self.iter_diff_frame_feature(),
                self.iter_motion_vector_feature()
            ],
            with_label=with_label
        )

    def create2(self, with_label=False):
        return self.__create_feature(
            name='diff_frame_full+motion_vector',
            feature_iters=[
                self.iter_diff_frame_feature_full(),
                self.iter_motion_vector_feature()
            ],
            with_label=with_label
        )
