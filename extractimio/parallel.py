import collections
import datetime
import glob
import multiprocessing as mp
import os
import time
from typing import NamedTuple

import numpy as np

from .common import Locator

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
