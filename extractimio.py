import datetime
import collections
from typing import NamedTuple
import glob
import os
import pickle
import time
import zipfile

import cv2
import imageio.v2 as iio
import numpy as np

import gzip

import multiprocessing as mp


def _dump_array(f, a):
    dtype = a.dtype
    shape = a.shape
    raw = a.flatten().tobytes()
    obj = dict(
        dtype=dtype,
        shape=shape,
        raw=gzip.compress(raw)
    )
    pickle.dump(obj, f)


def _load_array(f):
    obj = pickle.load(f)
    dtype = obj['dtype']
    shape = obj['shape']
    raw = gzip.decompress(obj['raw'])
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


class VideoFrameCache:
    def __init__(self, cache_path):
        self.__cache_path = cache_path
        self.__zf: zipfile.ZipFile | None = None

    def open(self):
        if self.__zf is not None:
            raise ValueError()
        self.__zf = zipfile.ZipFile(
            self.__cache_path,
            mode='a',
            compression=zipfile.ZIP_STORED
        )

    def close(self):
        if self.__zf is None:
            raise ValueError()
        self.__zf.close()
        self.__zf = None

    def __encode_name(self, i):
        return str(i)

    def __decode_name(self, name):
        i = int(name)
        return i

    # TODO: cache, careful to modification
    def __get_name_by_index(self, i):
        target_name = self.__encode_name(i)
        for info in self.__zf.infolist():
            if info.filename == target_name:
                return info.filename
        return None

    def get_frame(self, i):
        name = self.__get_name_by_index(i)
        if name is None:
            return None
        with self.__zf.open(name, 'r') as f:
            try:
                return _load_array(f)
            except EOFError:
                return None

    def put_frame(self, i, frame):
        if frame is None:
            raise ValueError()
        name = self.__encode_name(i)
        with self.__zf.open(name, 'w') as f:
            _dump_array(f, frame)


class Locator:
    def __init__(self, lst):
        self.__lst = lst

    def __getitem__(self, locator: slice):
        i = locator.start or 0
        while locator.stop is None or i < locator.stop:
            try:
                item = self.__lst[i]
            except IndexError:
                break
            yield i, item
            i += locator.step or 1


class VideoFrameIO:
    def __init__(self, video_path, cache_base_path=None):
        cache_base_path = cache_base_path or './vio_cache'

        os.makedirs(cache_base_path, exist_ok=True)

        self.__video_path = video_path

        _, video_name = os.path.split(video_path)
        cache_path = os.path.join(cache_base_path, video_name + '.zip')
        self.__cache = VideoFrameCache(cache_path)

        self.__reader = None
        self.__props = {}

        self.loc = Locator(self)

    def __len__(self):
        return self.num_frames

    def __read_frame(self, frame_index):
        frame = self.__reader.get_data(frame_index)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def __getitem__(self, frame_index):
        if frame_index < 0 or self.num_frames <= frame_index:
            raise IndexError()

        frame = self.__cache.get_frame(frame_index)
        if frame is None:
            frame = self.__read_frame(frame_index)
            self.__cache.put_frame(frame_index, frame)
        return frame

    def pos(self, frame_index):
        return frame_index / self.fps

    @property
    def fps(self):
        return self.__props['fps']

    @property
    def num_frames(self):
        return self.__props['num_frames']

    def _start(self):
        self.__reader = iio.get_reader(self.__video_path)

        self.__props['fps'] = self.__reader.get_meta_data()['fps']

        video = cv2.VideoCapture(self.__video_path)
        self.__props['num_frames'] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        del video

        self.__cache.open()

    def __enter__(self):
        self._start()
        return self

    def _close(self):
        self.__cache.close()
        self.__reader.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close()
        return False


WORKER_DEBUG = False
NUM_FUTURE_FRAMES = 16
WORKER_CACHE_MAX_SIZE = 256
WORKER_CACHE_REDUCED_SIZE = 128


def _io_worker(kwargs, command_q: mp.Queue, return_q: mp.Queue):
    if WORKER_DEBUG:
        print(datetime.datetime.now(), 'PROCESS BEGIN')

    class CacheEntry(NamedTuple):
        frame: np.ndarray
        accessed: datetime.datetime

    class FrameCache:
        def __init__(self):
            self.frames: dict[int, CacheEntry] = {}

        def organize_entry(self):
            if len(self.frames) < WORKER_CACHE_MAX_SIZE:
                return

            lst = list(self.frames.items())
            lst.sort(key=lambda pair: pair[1].accessed)
            old = lst[:WORKER_CACHE_REDUCED_SIZE]
            for frame_index, _ in old:
                self.frames.pop(frame_index)

            if WORKER_DEBUG:
                print(datetime.datetime.now(), 'reduce', len(self.frames))

        def get(self, frame_index):
            entry = self.frames.get(frame_index)
            if entry is None:
                return None
            return entry.frame

        def put(self, frame_index, frame):
            if frame_index in self.frames:
                self.frames[frame_index].accessed = datetime.datetime.now()
            else:
                self.frames[frame_index] = CacheEntry(
                    frame=frame,
                    accessed=datetime.datetime.now()
                )
            self.organize_entry()

    frame_cache = FrameCache()
    expected_frame_indexes = collections.deque()

    def reset_prediction(start_frame_index):
        if not expected_frame_indexes:
            return
        expected_frame_indexes.clear()
        for frame_index in range(start_frame_index, start_frame_index + NUM_FUTURE_FRAMES):
            if frame_index < vio.num_frames:
                expected_frame_indexes.append(frame_index)

    def pop_predicted_frame_index():
        if expected_frame_indexes:
            return expected_frame_indexes.popleft()
        return None

    with VideoFrameIO(**kwargs) as vio:
        while True:
            just_cache = False
            if command_q.empty():
                frame_index = pop_predicted_frame_index()
                if frame_index is not None:
                    command, *args = ['request', frame_index]
                    just_cache = True
                else:
                    time.sleep(0.001)
                    continue

            def pop():
                return command_q.get(block=False)

            if not just_cache:
                command, *args = pop()

            if WORKER_DEBUG:
                print(datetime.datetime.now(), 'cache' if just_cache else 'return', command, *args)

            if command == 'exit':
                break
            elif command == 'request':
                def request():
                    frame_index, = args
                    frame = frame_cache.get(frame_index)
                    if frame is None:
                        frame = vio[frame_index]
                        frame_cache.put(frame_index, frame)
                    if not just_cache:
                        reset_prediction(frame_index)
                    return frame

                ret = request()
                if not just_cache:
                    return_q.put(ret)
            elif command == 'meta':
                name, = args
                ret = getattr(vio, name)
                return_q.put(ret)
            else:
                import warnings
                warnings.warn('invalid command', command, *args)


class MultiprocessingVideoFrameIO:
    def __init__(self, video_path, cache_base_path=None):
        self.__command_q = mp.Queue()
        self.__return_q = mp.Queue()
        kwargs = dict(video_path=video_path, cache_base_path=cache_base_path)
        self.__process = mp.Process(
            target=_io_worker,
            args=(kwargs, self.__command_q, self.__return_q)
        )

        self.__props = {}

        self.loc = Locator(self)

    def __ensure_return_queue_empty(self):
        assert self.__return_q.empty(), self.__return_q.get(block=False)

    def __send_command(self, command, *args):
        item = command, *args
        self.__command_q.put(item)

    def __wait_return(self):
        self.__ensure_return_queue_empty()
        while self.__return_q.empty():
            time.sleep(0.01)
        return self.__return_q.get(block=False)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, frame_index):
        self.__send_command('request', frame_index)
        return self.__wait_return()

    def pos(self, frame_index):
        return frame_index / self.fps

    @property
    def fps(self):
        return self.__props['fps']

    @property
    def num_frames(self):
        return self.__props['num_frames']

    def _start(self):
        self.__process.start()

        for key in ['fps', 'num_frames']:
            self.__send_command('meta', key)
            self.__props[key] = self.__wait_return()

    def __enter__(self):
        self._start()
        return self

    def _close(self):
        self.__send_command('exit')
        self.__process.join()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close()
        return False


def main():
    path = os.path.expanduser(r'~\Desktop/idsttvideos/singles')
    glob_pattern = os.path.join(path, r'**/*.mp4')
    video_path = glob.glob(glob_pattern, recursive=True)[0]
    print(video_path)

    with VideoFrameIO(video_path, './vio_cache') as vio:
        def f():
            for i in range(100):
                vio[i]
            # cv2.imwrite(r'out.png', img)

        import timeit

        print(timeit.timeit(f, number=1) / 1)


if __name__ == '__main__':
    main()
