import functools
import os
from typing import Iterable, Union

import decord
import numpy as np

import frame_processor as fp

from tqdm import tqdm


class VideoFrameReader:
    def __normalize_slice(self, s: slice) -> slice:
        start = s.start
        if start is None:
            start = 0

        stop = s.stop
        if stop is None:
            stop = self.__frame_count

        step = s.step
        if step is None:
            step = 1

        return slice(start, stop, step)

    BATCH_MAX_MEMORY_USAGE = 2 ** 27  # ~ 256MB

    @functools.cached_property
    def __proper_batch_size(self) -> int:
        nbytes = self.__frame_nbytes or 32
        max_frames = self.BATCH_MAX_MEMORY_USAGE // nbytes
        return max(1, max_frames)

    def __iter_batch_indexes(self, frame_indexes):
        batch_size = self.__proper_batch_size

        i = 0
        bar = tqdm(frame_indexes)
        while True:
            batch_indexes = frame_indexes[i:i + batch_size]
            if len(batch_indexes) == 0:
                break
            yield batch_indexes
            i += batch_size
            bar.update(batch_size)

    def __init__(self, video_path):
        video_path = os.path.normpath(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError('video not found', video_path)
        self.__vr = decord.VideoReader(video_path, ctx=decord.cpu(0))

        self.__frame_count = len(self.__vr)
        if self.__frame_count:
            self.__frame_nbytes = np.product(self.__vr[0].asnumpy().nbytes)
        else:
            self.__frame_nbytes = None

    @property
    def _decord_video_reader(self):
        return self.__vr

    @property
    def frame_count(self):
        return len(self.__vr)

    @property
    def fps(self):
        return self.__vr.get_avg_fps()

    def __len__(self):
        return self.frame_count

    def time_to_frame_indexes(self,
                              time: Union[float, Iterable[float]]) -> np.ndarray:
        times = self.__vr.get_frame_timestamp(range(self.frame_count))[:, 0]
        indexes = np.searchsorted(times, time)
        # Use `np.bitwise_or` so it works both with scalars and numpy arrays.
        return np.where(
            (indexes == 0) | (times[indexes] - time <= time - times[indexes - 1]),
            indexes,
            indexes - 1
        )

    def __iter_frames_by_indexes(self, frame_indexes, with_index=False) \
            -> Iterable[tuple[float, np.ndarray]]:
        for batch_frame_indexes in self.__iter_batch_indexes(frame_indexes):
            batch_frame_times = self.__vr.get_frame_timestamp(batch_frame_indexes)[:, 0]
            batch_frames = self.__vr.get_batch(batch_frame_indexes).asnumpy()
            for frame_index, frame_time, frame in zip(batch_frame_indexes, batch_frame_times,
                                                      batch_frames):
                if with_index:
                    yield frame_index, frame_time, frame
                else:
                    yield frame_time, frame

    def __iter_frames_by_slice(self, s: slice, with_index=False):
        s = self.__normalize_slice(s)
        frame_indexes = np.arange(s.start, s.stop, s.step)
        yield from self.__iter_frames_by_indexes(frame_indexes, with_index=with_index)

    def __get_frame_by_index(self, frame_index):
        return self.__vr[frame_index]

    def __getitem__(self, key: Union[Iterable[int], slice, int]):
        if isinstance(key, float):
            raise ValueError('key must be an integer')
        if isinstance(key, int):
            return self.__get_frame_by_index(key)
        if isinstance(key, slice):
            return self.__iter_frames_by_slice(key)
        key_arr = np.array(key)
        if np.issubdtype(key_arr.dtype, np.integer):
            return self.__iter_batch_indexes(key_arr)
        raise TypeError('invalid type of key')

    def iter_frame_entries(self, s: slice) -> fp.FrameEntry:
        for index, position, image in self.__iter_frames_by_slice(s, with_index=True):
            yield fp.FrameEntry.create_instance(
                image=image,
                position=position,
                index=index,
            )
